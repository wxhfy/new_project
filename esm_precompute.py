#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESM-2 蛋白质嵌入分布式预计算系统 - (Transformers 版本)
优化版：支持分块存储与数据集映射增强

- 分块存储：将嵌入和注意力结果分块存储在同一文件夹，便于大数据集处理
- 数据集映射：维护序列ID与原始数据集的对应关系
- 多GPU并行：在多GPU上高效分布式计算
- 容错机制：支持断点续传、负载均衡和错误恢复

作者: wxhfy (基于 Transformers ESM-2 适配)
日期: 2025-04-15
版本: 2.0
"""

import os
import sys
import time
import json
import torch
import pickle
import hashlib
import argparse
import numpy as np
import logging
import traceback
import multiprocessing as mp
from pathlib import Path

from packaging import metadata
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from functools import partial
from datetime import datetime
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import autocast  # 用于混合精度推理
from collections import defaultdict

# 导入 Transformers 库
from transformers import AutoModel, AutoTokenizer

import h5py  # 保留 HDF5 导入以备将来使用，但当前默认输出 PT

# --- 日志配置 ---
log_file = "esm2_precompute.log"  # 日志文件名
logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # 保存到文件，使用 UTF-8 编码
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger("esm2_precompute")  # 日志记录器名称


# --- 日志配置结束 ---

# --- 辅助函数 ---
def _extract_sequences_with_metadata(data_path):
    """从数据文件中提取序列信息及元数据，增强版本"""
    logger.info(f"从 {data_path} 加载数据")
    try:
        # 加载 .pt 文件
        data = torch.load(data_path, map_location='cpu')

        # 确保数据是列表格式
        if not isinstance(data, list):
            if isinstance(data, dict):
                data = list(data.values())
            else:
                data = [data]

        sequences = []
        sequence_ids = []
        metadata_list = []
        unique_sequences = set()

        dataset_name = Path(data_path).stem

        logger.info(f"开始从 {len(data)} 个图对象中提取序列...")
        for idx, item in enumerate(tqdm(data, desc=f"提取序列 {dataset_name}")):
            # 初始化元数据字典
            metadata = {
                "source_file": data_path,
                "source_index": idx,
                "dataset_name": dataset_name,
                "original_index": idx,  # 明确存储原始索引
                "original_attributes": {}
            }

            # 提取序列和ID
            seq = None
            seq_id = None

            # 尝试从常见属性中获取序列
            if hasattr(item, 'sequence') and isinstance(item.sequence, str) and item.sequence:
                seq = item.sequence
                metadata["original_attributes"]["has_sequence"] = True
            elif isinstance(item, dict) and 'sequence' in item and item['sequence']:
                seq = item['sequence']
                metadata["original_attributes"]["has_sequence"] = True

            # 从不同属性中尝试提取ID
            potential_id_fields = ['protein_id', 'id', 'name', 'protein_name', 'uniprot_id']

            # 尝试从对象属性中获取ID
            for field in potential_id_fields:
                if hasattr(item, field):
                    value = getattr(item, field)
                    if value:
                        seq_id = str(value)
                        metadata["original_attributes"][field] = value
                        break
                # 或者从字典中获取
                elif isinstance(item, dict) and field in item and item[field]:
                    seq_id = str(item[field])
                    metadata["original_attributes"][field] = item[field]
                    break

            # 如果没有找到有效ID，使用序列哈希和索引创建一个
            if not seq_id and seq:
                seq_hash = hashlib.md5(seq.encode('utf-8')).hexdigest()[:8]
                seq_id = f"{dataset_name}_{idx}_{seq_hash}"
                metadata["is_generated_id"] = True
            elif not seq_id:
                seq_id = f"{dataset_name}_{idx}"
                metadata["is_generated_id"] = True

            # 添加其他可能有用的对象属性到元数据
            if hasattr(item, '__dict__'):
                for key, value in item.__dict__.items():
                    if key not in ['sequence'] and not key.startswith('_') and isinstance(value, (str, int, float, bool)):
                        metadata["original_attributes"][key] = value

            # 如果成功获取序列和ID
            if seq and seq_id:
                # 使用序列和ID作为组合键，避免重复
                seq_key = (seq, seq_id)
                if seq_key not in unique_sequences:
                    sequences.append(seq)
                    sequence_ids.append(seq_id)
                    metadata["sequence_hash"] = _create_seq_hash(seq)
                    metadata_list.append(metadata)
                    unique_sequences.add(seq_key)

        logger.info(f"成功提取 {len(sequences)} 个唯一序列")
        return sequences, sequence_ids, metadata_list
    except Exception as e:
        logger.error(f"数据加载或序列提取失败: {e}")
        logger.error(traceback.format_exc())
        return [], [], []

def _create_seq_hash(sequence):
    """为序列创建哈希值"""
    if not sequence:  # 处理空序列
        return hashlib.md5("empty".encode('utf-8')).hexdigest()
    # 使用 UTF-8 编码
    return hashlib.md5(sequence.encode('utf-8')).hexdigest()


# --- 辅助函数结束 ---


class DistributedESM2Computer:
    """分布式 ESM-2 嵌入计算器 (优化版)"""

    def __init__(
            self,
            model_path_or_name,  # 模型路径或名称
            output_dir,  # 输出目录
            gpu_ids,  # 使用的 GPU ID 列表
            batch_size=16,  # 每个 GPU 的批处理大小
            max_length=1022,  # ESM-2 的最大序列长度 (不含特殊 token)
            force_recompute=False,  # 是否强制重新计算
            master_port=29501,  # 分布式通信端口
            output_attentions=True,  # 是否计算并保存注意力图
            use_chunked_storage=True,  # 使用分块存储而不是合并文件
            chunk_size=1000  # 每个分块文件中的最大序列数
    ):
        self.model_path_or_name = model_path_or_name
        self.output_dir = os.path.join(output_dir, "embeddings")
        self.gpu_ids = gpu_ids  # 使用的 GPU ID 列表
        self.num_gpus = len(gpu_ids) if gpu_ids else 0  # 使用的 GPU 数量
        self.batch_size = batch_size
        self.max_length = max_length
        self.force_recompute = force_recompute
        self.master_port = master_port
        self.output_attentions = output_attentions
        self.use_chunked_storage = use_chunked_storage
        self.chunk_size = chunk_size
        self.format = "pt"  # 固定输出格式为 .pt

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 临时目录 - 用于存储每个 GPU 的部分结果 (每个输入文件有自己的临时子目录)
        self.temp_base_dir = os.path.join(self.output_dir, "temp")
        os.makedirs(self.temp_base_dir, exist_ok=True)

        # 为分块存储创建最终目录结构
        if self.use_chunked_storage:
            self.chunks_dir = os.path.join(self.output_dir,)
            os.makedirs(self.chunks_dir, exist_ok=True)

        # 统计信息跟踪
        self.stats = {}  # 每个文件分开统计

        # 映射关系数据
        self.dataset_mapping = {}  # 数据集映射信息，记录原始数据与嵌入之间的对应关系

    # --- 索引和文件路径管理函数 ---
    def _get_file_specific_paths(self, input_filename_stem):
        """获取特定输入文件的索引和检查点路径"""
        # 确保每个数据集有自己的子目录
        dataset_dir = os.path.join(self.output_dir, input_filename_stem)
        os.makedirs(dataset_dir, exist_ok=True)

        index_file = os.path.join(dataset_dir, f"{input_filename_stem}_index.pkl")
        mapping_file = os.path.join(dataset_dir, f"{input_filename_stem}_mapping.json")
        checkpoint_file = os.path.join(dataset_dir, f"{input_filename_stem}_checkpoint.json")
        temp_dir = os.path.join(self.temp_base_dir, input_filename_stem)

        # 为分块存储创建专用目录
        if self.use_chunked_storage:
            chunks_subdir = os.path.join(dataset_dir, "chunks")
            os.makedirs(chunks_subdir, exist_ok=True)
        else:
            chunks_subdir = dataset_dir  # 非分块模式下用数据集根目录

        os.makedirs(temp_dir, exist_ok=True)  # 确保临时目录存在

        return index_file, mapping_file, checkpoint_file, temp_dir, chunks_subdir

    def _load_or_create_index(self, index_file):
        """加载或创建特定文件的索引"""
        if os.path.exists(index_file) and not self.force_recompute:
            try:
                with open(index_file, "rb") as f:
                    embedding_index = pickle.load(f)
                logger.info(f"已加载文件索引 {Path(index_file).name}，包含 {len(embedding_index)} 条记录")
                return embedding_index
            except Exception as e:
                logger.error(f"加载索引文件 {Path(index_file).name} 失败: {e}")
        logger.info(f"为文件 {Path(index_file).stem.replace('_index', '')} 创建新的嵌入索引")
        return {}

    def _save_index(self, embedding_index, index_file):
        """保存特定文件的索引"""
        try:
            with open(index_file, "wb") as f:
                pickle.dump(embedding_index, f)
            logger.info(f"已保存文件索引 {Path(index_file).name}，包含 {len(embedding_index)} 条记录")
        except Exception as e:
            logger.error(f"保存索引文件 {Path(index_file).name} 失败: {e}")

    def _save_mapping(self, mapping_data, mapping_file):
        """保存数据集映射信息，优化索引结构"""
        try:
            # 增强映射数据结构
            enhanced_mapping = {
                "dataset_name": mapping_data["dataset_name"],
                "source_file": mapping_data["source_file"],
                "total_sequences": mapping_data["total_sequences"],
                "id_mapping": {},
                "index_mapping": {},  # 新增：通过原始索引查找蛋白质ID的映射
                "storage_format": "chunked" if self.use_chunked_storage else "merged"
            }

            # 处理ID映射
            for seq_id, info in mapping_data["id_mapping"].items():
                enhanced_mapping["id_mapping"][seq_id] = info

                # 添加反向索引映射
                if "original_index" in info:
                    idx = info["original_index"]
                    enhanced_mapping["index_mapping"][str(idx)] = seq_id

            with open(mapping_file, "w") as f:
                json.dump(enhanced_mapping, f, indent=2)
            logger.info(f"已保存增强数据集映射文件 {Path(mapping_file).name}")
        except Exception as e:
            logger.error(f"保存映射文件 {Path(mapping_file).name} 失败: {e}")

    def _load_checkpoint(self, checkpoint_file):
        """加载特定文件的检查点信息"""
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r") as f:
                    checkpoint = json.load(f)
                processed_count = len(checkpoint.get("processed_sequences", []))
                logger.info(f"已加载文件检查点 {Path(checkpoint_file).name}: 已处理 {processed_count} 个序列")
                return checkpoint
            except Exception as e:
                logger.error(f"加载检查点文件 {Path(checkpoint_file).name} 失败: {e}")
        return {"processed_sequences": [], "chunk_counter": 0}

    def _save_checkpoint(self, processed_sequences, chunk_counter, checkpoint_file):
        """保存特定文件的检查点信息"""
        try:
            checkpoint = {
                "processed_sequences": processed_sequences,
                "chunk_counter": chunk_counter
            }
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
            logger.debug(f"已保存文件检查点 {Path(checkpoint_file).name}")
        except Exception as e:
            logger.error(f"保存检查点文件 {Path(checkpoint_file).name} 失败: {e}")

    # --- 索引和文件路径管理函数结束 ---

    def _partition_sequences(self, sequences, sequence_ids, metadata_list, embedding_index, checkpoint_file):
        """根据缓存状态和任务平衡为单个文件分配序列任务"""
        checkpoint = self._load_checkpoint(checkpoint_file)
        processed_ids = set(checkpoint.get("processed_sequences", []))
        chunk_counter = checkpoint.get("chunk_counter", 0)  # 当前分块计数

        need_compute = []
        already_cached_count = 0

        for i, (seq_id, seq, metadata) in enumerate(zip(sequence_ids, sequences, metadata_list)):
            seq_hash = metadata.get("sequence_hash", _create_seq_hash(seq))
            # 检查索引（现在是文件特定的）
            cache_exists = seq_id in embedding_index  # or seq_hash in embedding_index

            if (seq_id in processed_ids or cache_exists) and not self.force_recompute:
                already_cached_count += 1
            else:
                need_compute.append((i, seq_id, seq, metadata))  # (原始索引, ID, 序列, 元数据)

        # 文件级别的统计
        file_stats = {
            "total": len(sequences),
            "cached": already_cached_count,
            "to_compute": len(need_compute)
        }
        logger.info(
            f"序列分区统计 - 总数: {file_stats['total']}, 已缓存/处理: {file_stats['cached']}, 待计算: {file_stats['to_compute']}")

        # 按序列长度排序以提高效率（短序列优先可能减少内存碎片）
        need_compute.sort(key=lambda x: len(x[2]))

        # 为 GPU 或 CPU 分配任务
        if self.num_gpus > 0:
            partitions = [[] for _ in range(self.num_gpus)]
            seq_lens = [0] * self.num_gpus  # 按总长度进行负载均衡
            for item in need_compute:
                # 分配给当前总长度最短的 GPU
                min_gpu_idx = seq_lens.index(min(seq_lens))
                partitions[min_gpu_idx].append(item)
                seq_lens[min_gpu_idx] += len(item[2])  # 更新该 GPU 的总长度负载
            for gpu_idx, partition in enumerate(partitions):
                logger.info(f"  GPU {self.gpu_ids[gpu_idx]} 分配到 {len(partition)} 个序列，总长度 {seq_lens[gpu_idx]}")
        else:  # 仅 CPU
            partitions = [need_compute]  # 所有任务都在一个分区
            logger.info(f"  CPU 分配到 {len(need_compute)} 个序列")

        return partitions, list(processed_ids), chunk_counter, file_stats  # 返回分区、已处理ID列表、分块计数器、文件统计

    def _init_process(self, rank_in_comm, partitions, world_size, init_method, temp_dir, chunks_subdir,
                      input_filename_stem, chunk_counter):
        """初始化分布式计算进程"""
        try:
            gpu_id_to_use = -1  # CPU 默认为 -1
            # 确定设备
            if self.num_gpus > 0:
                gpu_id_to_use = self.gpu_ids[rank_in_comm]  # 从提供的列表中获取 GPU ID
                device = torch.device(f"cuda:{gpu_id_to_use}")
                torch.cuda.set_device(device)
                # 初始化分布式进程组
                dist.init_process_group(
                    backend='nccl',  # NCCL 后端适用于 GPU
                    init_method=init_method,
                    world_size=world_size,
                    rank=rank_in_comm  # 使用通信器内的排名
                )
                logger.info(f"进程 {rank_in_comm}/{world_size} (全局排名) 初始化在 GPU {gpu_id_to_use} (设备ID)")
            else:
                device = torch.device("cpu")
                logger.info(f"进程 {rank_in_comm}/{world_size} 初始化在 CPU")

            # 获取当前进程的任务分区
            partition = partitions[rank_in_comm]

            # 设置此进程的随机种子
            seed = 42 + rank_in_comm
            torch.manual_seed(seed)
            if torch.cuda.is_available() and self.num_gpus > 0:
                torch.cuda.manual_seed(seed)

            # 为此进程加载模型和 Tokenizer
            logger.info(f"进程 {rank_in_comm}: 从 {self.model_path_or_name} 加载 Tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path_or_name)
            logger.info(f"进程 {rank_in_comm}: 从 {self.model_path_or_name} 加载模型到 {device}")

            # 使用 AutoModel 来获取嵌入
            model_class = AutoModel  # 默认使用 AutoModel

            # 使用半精度（float16）可能在兼容的 GPU 上加速推理并减少显存占用
            model_dtype = torch.float16 if torch.cuda.is_available() and self.num_gpus > 0 else torch.float32
            model = model_class.from_pretrained(
                self.model_path_or_name,
                torch_dtype=model_dtype  # 加载时指定数据类型
            ).to(device)
            model.eval()  # 设置为评估模式，禁用 dropout 等

            logger.info(f"进程 {rank_in_comm}: 模型和 Tokenizer 加载成功")

            # 处理分配到的序列分区
            # 传递临时目录、分块目录和文件名用于保存结果
            self._process_partition(
                model, tokenizer, partition, rank_in_comm, device,
                temp_dir, chunks_subdir, input_filename_stem, chunk_counter
            )

            # 清理模型和 Tokenizer 占用的内存
            del model, tokenizer
            if torch.cuda.is_available() and self.num_gpus > 0:
                torch.cuda.empty_cache()  # 清空 GPU 缓存

            # 销毁分布式进程组（如果已初始化）
            if self.num_gpus > 0 and dist.is_initialized():
                dist.destroy_process_group()
            logger.info(f"进程 {rank_in_comm} 完成并清理资源")

        except Exception as e:
            logger.error(f"进程 {rank_in_comm} 遇到严重错误: {e}")
            logger.error(traceback.format_exc())
            # 确保即使出错也销毁进程组
            if self.num_gpus > 0 and dist.is_initialized():
                dist.destroy_process_group()

    def _process_partition(self, model, tokenizer, partition, rank, device, temp_dir, chunks_subdir,
                           input_filename_stem, initial_chunk_counter):
        """处理分配给当前进程的序列分区"""
        if not partition:
            logger.info(f"进程 {rank}: 无序列分配，退出。")
            if self.use_chunked_storage:
                # 即使没有序列，也创建一个空的结果文件
                self._save_empty_chunk(rank, chunks_subdir, input_filename_stem, initial_chunk_counter)
            else:
                self._save_empty_result(rank, temp_dir, input_filename_stem)
            return

        logger.info(f"进程 {rank}: 开始处理 {len(partition)} 个序列。")

        # 是否使用混合精度计算
        use_amp = torch.cuda.is_available() and self.num_gpus > 0

        # 设置进度条（仅主进程或CPU模式下显示）
        is_main_process = rank == 0 or self.num_gpus == 0
        pbar = tqdm(total=len(partition), desc=f"GPU {self.gpu_ids[rank] if self.num_gpus > 0 else 'CPU'}",
                    position=rank, disable=not is_main_process)

        # 分块存储模式
        if self.use_chunked_storage:
            # 初始化分块计数器
            chunk_counter = initial_chunk_counter + rank  # 每个进程使用自己的计数器基数
            current_chunk = {}  # 当前正在处理的分块
            items_in_current_chunk = 0  # 当前分块中的序列数量
            chunk_file_paths = []  # 保存分块文件路径列表
            processed_id_to_chunk = {}  # 跟踪哪个序列ID保存在哪个分块文件中

            # 按批次处理序列
            for i in range(0, len(partition), self.batch_size):
                # 获取当前批次的序列和信息
                batch_items = partition[i: i + self.batch_size]
                batch_indices = [item[0] for item in batch_items]
                batch_seq_ids = [item[1] for item in batch_items]
                batch_sequences = [item[2] for item in batch_items]
                batch_metadata = [item[3] for item in batch_items]

                # 计算嵌入和处理结果
                batch_results = self._compute_embeddings_for_batch(
                    model, tokenizer, batch_sequences, device, use_amp
                )

                # 处理批次结果，并添加到当前分块
                for j, (seq_id, original_seq, metadata) in enumerate(
                        zip(batch_seq_ids, batch_sequences, batch_metadata)):
                    # 获取当前序列的计算结果
                    embedding = batch_results["embeddings"][j]
                    attention = batch_results.get("attentions", {}).get(j, None)

                    # 创建结果条目
                    result_data = {
                        "embedding": embedding,
                        "sequence": original_seq,
                        "metadata": metadata,  # 包含与原始数据集关联的元数据
                    }
                    if attention is not None:
                        result_data["attention"] = attention

                    # 添加到当前分块
                    current_chunk[seq_id] = result_data
                    items_in_current_chunk += 1

                    # 检查是否需要保存当前分块
                    if items_in_current_chunk >= self.chunk_size:
                        # 保存当前分块
                        chunk_filename = f"{input_filename_stem}_chunk_{chunk_counter:04d}.pt"
                        chunk_path = os.path.join(chunks_subdir, chunk_filename)
                        torch.save(current_chunk, chunk_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

                        # 更新跟踪信息
                        chunk_file_paths.append(chunk_path)
                        for processed_id in current_chunk.keys():
                            processed_id_to_chunk[processed_id] = chunk_filename

                        # 重置分块
                        logger.info(f"进程 {rank}: 保存分块 {chunk_filename} (含 {items_in_current_chunk} 条记录)")
                        chunk_counter += self.num_gpus  # 增加分块计数器，跳过其他进程的计数
                        current_chunk = {}
                        items_in_current_chunk = 0

                # 更新进度条
                if is_main_process:
                    pbar.update(len(batch_items))

            # 处理最后一个未满的分块
            if items_in_current_chunk > 0:
                chunk_filename = f"{input_filename_stem}_chunk_{chunk_counter:04d}.pt"
                chunk_path = os.path.join(chunks_subdir, chunk_filename)
                torch.save(current_chunk, chunk_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

                # 更新跟踪信息
                chunk_file_paths.append(chunk_path)
                for processed_id in current_chunk.keys():
                    processed_id_to_chunk[processed_id] = chunk_filename

                logger.info(f"进程 {rank}: 保存最终分块 {chunk_filename} (含 {items_in_current_chunk} 条记录)")

            # 保存分块映射信息
            mapping_path = os.path.join(temp_dir, f"{input_filename_stem}_chunks_mapping_{rank}.json")
            with open(mapping_path, "w") as f:
                json.dump({
                    "processed_ids": list(processed_id_to_chunk.keys()),
                    "id_to_chunk": processed_id_to_chunk,
                    "chunk_files": chunk_file_paths,
                    "last_chunk_counter": chunk_counter
                }, f, indent=2)

            logger.info(
                f"进程 {rank}: 已处理 {len(processed_id_to_chunk)} 个序列，保存到 {len(chunk_file_paths)} 个分块文件")
        else:
            # 传统模式：所有结果合并到一个临时文件中
            embeddings_dict = {}
            processed_count_in_partition = 0

            # 按批次处理序列
            for i in range(0, len(partition), self.batch_size):
                # 获取当前批次的序列和信息
                batch_items = partition[i: i + self.batch_size]
                batch_indices = [item[0] for item in batch_items]
                batch_seq_ids = [item[1] for item in batch_items]
                batch_sequences = [item[2] for item in batch_items]
                batch_metadata = [item[3] for item in batch_items]

                # 计算嵌入和处理结果
                batch_results = self._compute_embeddings_for_batch(
                    model, tokenizer, batch_sequences, device, use_amp
                )

                # 处理批次结果
                for j, (seq_id, original_seq, metadata) in enumerate(
                        zip(batch_seq_ids, batch_sequences, batch_metadata)):
                    # 获取当前序列的计算结果
                    embedding = batch_results["embeddings"][j]
                    attention = batch_results.get("attentions", {}).get(j, None)

                    # 创建结果条目
                    result_data = {
                        "embedding": embedding,
                        "sequence": original_seq,
                        "metadata": metadata,  # 包含与原始数据集关联的元数据
                    }
                    if attention is not None:
                        result_data["attention"] = attention

                    # 添加到结果字典
                    embeddings_dict[seq_id] = result_data
                    processed_count_in_partition += 1

                # 更新进度条
                if is_main_process:
                    pbar.update(len(batch_items))

            # 保存所有结果到临时文件
            temp_file = os.path.join(temp_dir, f"{input_filename_stem}_part_{rank}.pt")
            try:
                torch.save(embeddings_dict, temp_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"进程 {rank}: 为文件 {input_filename_stem} 保存了 {len(embeddings_dict)} 条临时结果到 {Path(temp_file).name}")
            except Exception as e:
                logger.error(f"进程 {rank}: 保存临时结果文件 {Path(temp_file).name} 失败: {e}")
                logger.error(traceback.format_exc())

        # 关闭进度条
        if is_main_process:
            pbar.close()

    def _compute_embeddings_for_batch(self, model, tokenizer, sequences, device, use_amp):
        """
        为一批序列计算嵌入

        参数:
            model: ESM模型
            tokenizer: ESM分词器
            sequences: 序列列表
            device: 计算设备
            use_amp: 是否使用混合精度

        返回:
            dict: 包含嵌入和注意力的批次结果
        """
        try:
            # 使用 Tokenizer 处理批次
            inputs = tokenizer(
                sequences,
                return_tensors="pt",  # 返回 PyTorch 张量
                padding=True,  # 填充到批次中最长序列
                truncation=True,  # 截断超过最大长度的序列
                max_length=self.max_length,  # 使用实例的最大长度设置
            ).to(device)  # 将输入数据移动到目标设备

            # 计算嵌入和注意力
            with torch.no_grad(), autocast(enabled=use_amp):  # 禁用梯度计算，并启用混合精度（如果可用）
                outputs = model(
                    **inputs,
                    output_hidden_states=True,  # 确保返回隐藏状态
                    output_attentions=self.output_attentions  # 根据设置决定是否计算注意力
                )

                # 提取最后一层隐藏状态作为嵌入
                # 形状: [batch_size, seq_len, embedding_dim]
                last_hidden_states = outputs.last_hidden_state

                # 提取注意力图（如果计算了）
                batch_attentions = None
                if self.output_attentions and outputs.attentions:
                    # 获取最后一层的注意力图
                    # 形状: [batch_size, num_heads, seq_len, seq_len]
                    last_layer_attentions = outputs.attentions[-1]
                    # 在头部维度上取平均，得到更简洁的注意力表示
                    # 形状: [batch_size, seq_len, seq_len]
                    batch_attentions = last_layer_attentions.mean(dim=1)

            # 处理结果
            batch_results = {
                "embeddings": {},
                "attentions": {}
            }

            # 遍历批次中的每个结果
            for j in range(last_hidden_states.size(0)):
                # 获取真实的序列长度（去除填充部分）
                attention_mask = inputs['attention_mask'][j]
                true_len = attention_mask.sum().item()

                # 提取每个残基的嵌入（去除 BOS 和 EOS token）
                # 索引 0 是 BOS, true_len-1 是 EOS
                embedding = last_hidden_states[j, 1:true_len - 1, :]  # 形状: [真实序列长度, embedding_dim]

                # 提取对应的注意力图（如果计算了）
                attention_map = None
                if batch_attentions is not None:
                    # 提取注意力图中对应真实序列的部分（去除 BOS/EOS 行和列）
                    attention_map = batch_attentions[j, 1:true_len - 1, 1:true_len - 1]  # 形状: [真实序列长度, 真实序列长度]

                # 将结果移回 CPU 并转换为半精度（float16）以节省存储空间
                embedding_cpu = embedding.detach().cpu().half()
                batch_results["embeddings"][j] = embedding_cpu

                if attention_map is not None:
                    attention_cpu = attention_map.detach().cpu().half()
                    batch_results["attentions"][j] = attention_cpu

            return batch_results

        except Exception as e:
            logger.error(f"批次处理失败: {e}")
            logger.error(traceback.format_exc())
            # 返回空结果
            return {"embeddings": {}, "attentions": {}}

    def _save_empty_chunk(self, rank, chunks_subdir, input_filename_stem, chunk_counter):
        """保存一个空的分块文件"""
        chunk_filename = f"{input_filename_stem}_chunk_{chunk_counter + rank:04d}.pt"
        chunk_path = os.path.join(chunks_subdir, chunk_filename)
        try:
            torch.save({}, chunk_path)
            logger.debug(f"进程 {rank}: 为文件 {input_filename_stem} 保存了空的分块文件。")
            return chunk_path
        except Exception as e:
            logger.error(f"进程 {rank}: 保存空的分块文件失败: {e}")
            return None

    def _save_empty_result(self, rank, temp_dir, input_filename_stem):
        """为当前文件保存一个空的临时结果文件"""
        temp_file = os.path.join(temp_dir, f"{input_filename_stem}_part_{rank}.pt")
        try:
            torch.save({}, temp_file)  # 保存空字典
            logger.debug(f"进程 {rank}: 为文件 {input_filename_stem} 保存了空结果文件。")
        except Exception as e:
            logger.error(f"进程 {rank}: 保存空结果文件 {Path(temp_file).name} 失败: {e}")

    def _merge_chunks_and_update_index(self, input_filename_stem, temp_dir, chunks_subdir, embedding_index, index_file):
        """合并分块处理结果并更新索引文件"""
        logger.info(f"更新文件 {input_filename_stem} 的索引信息...")

        # 在分块存储模式下，查找保存的分块映射信息
        mapping_files = sorted([
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
            if f.startswith(f"{input_filename_stem}_chunks_mapping_") and f.endswith(".json")
        ])

        if not mapping_files:
            logger.warning(f"未找到文件 {input_filename_stem} 的分块映射文件，无法更新索引。")
            return False

        # 合并所有进程的分块映射信息
        all_processed_ids = set()
        id_to_chunk = {}
        all_chunk_files = set()
        max_chunk_counter = 0

        for mapping_file in mapping_files:
            try:
                with open(mapping_file, 'r') as f:
                    process_mapping = json.load(f)

                # 更新处理过的ID集合和映射信息
                all_processed_ids.update(process_mapping.get("processed_ids", []))
                id_to_chunk.update(process_mapping.get("id_to_chunk", {}))
                all_chunk_files.update(process_mapping.get("chunk_files", []))

                # 跟踪最大的分块计数器
                last_counter = process_mapping.get("last_chunk_counter", 0)
                max_chunk_counter = max(max_chunk_counter, last_counter)

            except Exception as e:
                logger.error(f"读取分块映射文件 {mapping_file} 失败: {e}")

            # 清理映射文件
            try:
                os.remove(mapping_file)
            except:
                pass

        # 更新索引文件
        updated_index_count = 0
        for seq_id, chunk_filename in id_to_chunk.items():
            # 检查该序列是否已经存在于索引中
            if seq_id in embedding_index and not self.force_recompute:
                continue

            # 在索引更新部分添加
            embedding_index[seq_id] = {
                "file": chunk_filename,  # 指向包含此序列的分块文件名
                "original_index": metadata.get("original_index", -1),  # 保存原始索引
                "protein_id": seq_id if seq_id.startswith("AF-") else None,  # 如果是标准蛋白质ID则保存
                "hash": metadata.get("sequence_hash", ""),
                "timestamp": time.time()
            }
            updated_index_count += 1

        # 保存更新后的索引
        self._save_index(embedding_index, index_file)
        logger.info(
            f"文件 {input_filename_stem} 的索引已更新，新增 {updated_index_count} 条记录，总计 {len(embedding_index)} 条。")

        # 返回更新了索引的数量和最新分块计数器
        return {
            "updated_count": updated_index_count,
            "processed_ids": list(all_processed_ids),
            "chunk_files": list(all_chunk_files),
            "max_chunk_counter": max_chunk_counter
        }

    def _merge_embeddings_for_file(self, input_filename_stem, temp_dir, final_output_path, embedding_index, index_file):
        """合并特定输入文件的所有临时嵌入文件到最终输出文件"""
        logger.info(f"开始合并文件 {input_filename_stem} 的嵌入结果...")
        # 查找属于此文件的所有临时部分文件
        temp_files = sorted([
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
            if f.startswith(f"{input_filename_stem}_part_") and f.endswith(".pt")
        ])

        if not temp_files:
            logger.warning(f"未找到文件 {input_filename_stem} 的临时嵌入文件，无法合并。")
            return False

        merged_embeddings = {}  # 用于存储合并后的嵌入
        total_loaded_count = 0
        failed_files = []  # 记录合并失败的文件

        # 遍历并加载每个临时文件
        for temp_file in tqdm(temp_files, desc=f"合并 {input_filename_stem}"):
            try:
                # 从 CPU 加载，避免 GPU 内存问题
                part_embeddings = torch.load(temp_file, map_location='cpu')
                if isinstance(part_embeddings, dict):
                    merged_embeddings.update(part_embeddings)  # 合并字典
                    total_loaded_count += len(part_embeddings)
                    # 合并成功后删除临时文件
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.warning(f"无法删除临时文件 {temp_file}: {e}")
                else:
                    logger.warning(f"跳过无效的临时文件（非字典格式）: {temp_file}")
                    failed_files.append(temp_file)
            except Exception as e:
                logger.error(f"加载或合并文件 {temp_file} 失败: {e}. 跳过。")
                failed_files.append(temp_file)

        # 如果有合并失败的文件，发出警告
        if failed_files:
            logger.warning(f"合并文件 {input_filename_stem} 时，{len(failed_files)} 个临时文件处理失败。")
            logger.warning(f"失败的文件列表: {failed_files}")

        # 如果合并后的结果非空，则保存
        if merged_embeddings:
            try:
                # 保存最终的 .pt 文件
                torch.save(merged_embeddings, final_output_path, pickle_protocol=pickle.HIGHEST_PROTOCOL,
                           _use_new_zipfile_serialization=True)
                logger.info(f"成功合并 {len(merged_embeddings)} 条嵌入记录到: {final_output_path}")

                # 更新文件索引（指向最终合并的文件）
                output_filename = Path(final_output_path).name
                for seq_id, data in merged_embeddings.items():
                    # 提取序列和元数据（如果有）
                    sequence = data.get("sequence", "")
                    metadata = data.get("metadata", {})

                    # 创建或更新索引条目
                    embedding_index[seq_id] = {
                        "hash": _create_seq_hash(sequence),
                        "sequence": sequence,
                        "file": output_filename,  # 指向合并后的文件名
                        "timestamp": time.time(),
                        "metadata": metadata  # 保存元数据以便于查找
                    }

                self._save_index(embedding_index, index_file)  # 保存更新后的索引

                return True
            except Exception as e:
                logger.error(f"保存最终合并文件 {final_output_path} 失败: {e}")
                logger.error(traceback.format_exc())
                return False
        else:
            logger.error(f"文件 {input_filename_stem} 没有有效的嵌入记录可合并。")
            return False

    def compute_embeddings_for_file(self, input_file_path):
        """为单个输入文件计算嵌入"""
        input_filename_stem = Path(input_file_path).stem  # 获取文件名（不含扩展名）
        logger.info(f"===== 开始处理文件: {input_filename_stem} =====")

        # 初始化此文件的统计信息
        self.stats[input_filename_stem] = {
            "total": 0, "computed": 0, "cached": 0, "failed": 0,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 获取文件特定的路径
        index_file, mapping_file, checkpoint_file, temp_dir, chunks_subdir = self._get_file_specific_paths(
            input_filename_stem)

        # 加载此文件的索引
        embedding_index = self._load_or_create_index(index_file)

        # 提取序列和元数据
        sequences, sequence_ids, metadata_list = _extract_sequences_with_metadata(input_file_path)
        if not sequences:
            logger.error(f"文件 {input_filename_stem} 未提取到有效序列，跳过。")
            self.stats[input_filename_stem]['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return

        # 创建数据集映射结构
        dataset_mapping = {
            "dataset_name": input_filename_stem,
            "source_file": input_file_path,
            "total_sequences": len(sequences),
            "id_mapping": {seq_id: {"original_index": idx} for idx, seq_id in enumerate(sequence_ids)}
        }

        # 增强映射，添加哈希和部分元数据
        for idx, (seq_id, metadata) in enumerate(zip(sequence_ids, metadata_list)):
            dataset_mapping["id_mapping"][seq_id].update({
                "sequence_hash": metadata.get("sequence_hash", ""),
                "source_index": metadata.get("source_index", idx),
                "attributes": {k: v for k, v in metadata.get("original_attributes", {}).items() if
                               isinstance(v, (str, int, float, bool))}
            })

        # 保存映射文件
        self._save_mapping(dataset_mapping, mapping_file)

        # 分配任务
        partitions, processed_ids, chunk_counter, file_stats = self._partition_sequences(
            sequences, sequence_ids, metadata_list, embedding_index, checkpoint_file
        )
        self.stats[input_filename_stem].update(file_stats)  # 更新统计

        total_to_compute = file_stats['to_compute']
        if total_to_compute == 0 and not self.force_recompute:
            logger.info(f"文件 {input_filename_stem} 的所有序列已处理或缓存，跳过计算。")
            self.stats[input_filename_stem]['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return

        logger.info(f"文件 {input_filename_stem}: 需要计算 {total_to_compute} 个序列。")

        # 设置分布式环境或在 CPU 上运行
        if self.num_gpus > 0:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(self.master_port)
            world_size = self.num_gpus
            init_method = f"tcp://localhost:{self.master_port}"
            processes = []
            try:
                # 尝试设置 'spawn' 启动方法，这在某些系统上更稳定
                mp.set_start_method('spawn', force=True)
                logger.info("设置多进程启动方法为 'spawn'")
            except RuntimeError:
                logger.warning("无法设置多进程启动方法为 'spawn'，继续使用默认方法。")
                pass  # 如果设置失败，继续使用默认方法

            try:
                logger.info(f"为文件 {input_filename_stem} 启动 {world_size} 个 GPU 进程...")
                # 启动进程，传递特定于文件的临时目录和文件名
                for rank_in_comm in range(world_size):
                    p = Process(target=self._init_process, args=(
                        rank_in_comm, partitions, world_size, init_method,
                        temp_dir, chunks_subdir, input_filename_stem, chunk_counter
                    ))
                    p.start()
                    processes.append(p)

                # 等待所有进程完成
                for p in processes:
                    p.join()
                logger.info(f"文件 {input_filename_stem} 的所有 GPU 进程已完成。")

            except Exception as e:
                logger.error(f"文件 {input_filename_stem} 的多进程执行失败: {e}")
                logger.error(traceback.format_exc())
                # 终止仍在运行的进程
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                        p.join()
        else:  # CPU 计算
            logger.info(f"在 CPU 上计算文件 {input_filename_stem} 的嵌入...")
            # 在主进程中直接运行
            self._init_process(
                0, partitions, 1, None, temp_dir,
                chunks_subdir, input_filename_stem, chunk_counter
            )
            logger.info(f"文件 {input_filename_stem} 的 CPU 计算完成。")

        # --- 处理和索引更新 ---
        if self.use_chunked_storage:
            # 分块存储模式：更新索引以指向分块文件
            update_results = self._merge_chunks_and_update_index(
                input_filename_stem, temp_dir, chunks_subdir, embedding_index, index_file
            )

            if update_results and update_results["updated_count"] > 0:
                # 更新统计信息
                self.stats[input_filename_stem]['computed'] = update_results["updated_count"]
                self.stats[input_filename_stem]['failed'] = file_stats['to_compute'] - update_results["updated_count"]

                # 保存更新后的检查点
                all_processed_ids = list(set(processed_ids).union(set(update_results["processed_ids"])))
                self._save_checkpoint(
                    all_processed_ids,
                    update_results["max_chunk_counter"],
                    checkpoint_file
                )
            else:
                self.stats[input_filename_stem]['failed'] = file_stats['to_compute']
                self.stats[input_filename_stem]['computed'] = 0
        else:
            # 传统模式：合并所有临时文件到单个输出文件
            final_output_path = os.path.join(self.output_dir, f"{input_filename_stem}_embedding.pt")
            merge_success = self._merge_embeddings_for_file(
                input_filename_stem, temp_dir, final_output_path, embedding_index, index_file
            )

            # 更新统计信息
            if merge_success:
                # 从合并后的文件读取实际计算的数量
                try:
                    merged_data = torch.load(final_output_path, map_location='cpu')
                    actual_computed = len(merged_data)
                    self.stats[input_filename_stem]['computed'] = actual_computed - file_stats['cached']
                    self.stats[input_filename_stem]['failed'] = file_stats['total'] - actual_computed
                except Exception as e:
                    logger.warning(f"读取合并文件 {final_output_path} 以更新统计信息失败: {e}")
                    self.stats[input_filename_stem]['failed'] = file_stats['to_compute'] - \
                                                                self.stats[input_filename_stem]['computed']
            else:
                self.stats[input_filename_stem]['failed'] = file_stats['to_compute']
                self.stats[input_filename_stem]['computed'] = 0

            # 保存检查点（更加简单，因为所有内容都在一个文件中）
            newly_processed_ids = list(set(seq_id for _, seq_id, _, _ in sum(partitions, [])))
            all_processed_ids = list(set(processed_ids).union(set(newly_processed_ids)))
            self._save_checkpoint(all_processed_ids, 0, checkpoint_file)

        # 记录结束时间
        self.stats[input_filename_stem]['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"===== 完成文件: {input_filename_stem} =====")
        logger.info(f"  统计: 总序列数: {self.stats[input_filename_stem]['total']}, "
                    f"计算新嵌入: {self.stats[input_filename_stem]['computed']}, "
                    f"使用缓存: {self.stats[input_filename_stem]['cached']}, "
                    f"失败: {self.stats[input_filename_stem]['failed']}")
        logger.info(f"  存储模式: {'分块存储' if self.use_chunked_storage else '单一文件'}")
        logger.info(f"  输出: {chunks_subdir if self.use_chunked_storage else final_output_path}")
        logger.info(f"==========================================")

    def save_global_stats(self):
        """保存全局统计信息"""
        global_stats_file = os.path.join(self.output_dir, "precompute_summary_stats.json")
        # 添加全局汇总信息
        total_processed = sum(s['total'] for s in self.stats.values())
        total_computed = sum(s['computed'] for s in self.stats.values())
        total_cached = sum(s['cached'] for s in self.stats.values())
        total_failed = sum(s['failed'] for s in self.stats.values())

        summary_data = {
            "global_summary": {
                "storage_mode": "chunked" if self.use_chunked_storage else "merged",
                "chunk_size": self.chunk_size if self.use_chunked_storage else None,
                "total_sequences_processed": total_processed,
                "total_embeddings_computed": total_computed,
                "total_from_cache": total_cached,
                "total_failed": total_failed,
                "embedding_dim": self.stats.get("embedding_dim", 0),
                "attention_layers": self.stats.get("attention_layers", 0),
                "attention_heads": self.stats.get("attention_heads", 0),
                "model_name": self.model_path_or_name,
                "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "file_details": self.stats  # 包含每个文件的详细统计
        }

        try:
            with open(global_stats_file, "w") as f:
                json.dump(summary_data, f, indent=2)
            logger.info(f"全局统计信息已保存到: {global_stats_file}")
        except Exception as e:
            logger.error(f"保存全局统计信息失败: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ESM-2 嵌入分布式预计算工具 (优化版)")
    parser.add_argument("--data-dir", type=str, default="data", help="包含数据文件的目录")
    parser.add_argument("--output-dir", type=str, default="data", help="保存嵌入文件的输出目录")
    parser.add_argument("--model-path", type=str, default="esm2_t33_650M_UR50D",
                        help="ESM-2模型路径或HuggingFace模型名称")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3",
                        help="要使用的GPU设备ID，用逗号分隔。留空则使用所有可用GPU。")
    parser.add_argument("--batch-size", type=int, default=12, help="每个GPU的批处理大小")
    parser.add_argument("--max-length", type=int, default=1022, help="最大序列长度")
    parser.add_argument("--force", action="store_true", help="强制重新计算所有嵌入，忽略缓存")
    parser.add_argument("--port", type=int, default=29501, help="分布式通信端口")
    parser.add_argument("--no-attentions", action="store_true", help="禁用注意力图计算和保存")
    parser.add_argument("--use-merged", action="store_true", help="使用合并存储而非分块存储")
    parser.add_argument("--chunk-size", type=int, default=60000, help="分块存储时每个文件的最大序列数")
    parser.add_argument("--file-names", type=str, default="train_data.pt,val_data.pt,test_data.pt",
                        help="要处理的文件名，用逗号分隔")
    return parser.parse_args()


def main():
    """主执行函数 - 处理参数并协调整个预计算流程"""
    args = parse_arguments()

    # --- GPU ID 处理 ---
    gpu_ids_to_use = []
    if args.gpu_ids:
        try:
            # 解析用户指定的GPU ID列表
            gpu_ids_to_use = [int(gid.strip()) for gid in args.gpu_ids.split(',')]
            available_gpus = list(range(torch.cuda.device_count()))
            # 过滤掉无效或不存在的 GPU ID
            valid_gpu_ids = [gid for gid in gpu_ids_to_use if gid in available_gpus]
            if len(valid_gpu_ids) != len(gpu_ids_to_use):
                logger.warning(f"提供的 GPU IDs: {gpu_ids_to_use} 包含无效ID。仅使用有效的: {valid_gpu_ids}")
            gpu_ids_to_use = valid_gpu_ids
            if not gpu_ids_to_use:
                logger.warning("未指定有效的 GPU ID，将在 CPU 上运行。")
        except ValueError:
            logger.error(f"无效的 GPU ID 格式: '{args.gpu_ids}'。请使用逗号分隔的整数。将在 CPU 上运行。")
            gpu_ids_to_use = []
    elif torch.cuda.is_available():
        # 如果未指定，则默认使用所有可用 GPU
        gpu_ids_to_use = list(range(torch.cuda.device_count()))
        if not gpu_ids_to_use:
            logger.warning("未检测到 CUDA 设备，将在 CPU 上运行。")
    else:
        logger.warning("CUDA 不可用，将在 CPU 上运行。")

    num_gpus = len(gpu_ids_to_use)
    # --- GPU ID 处理结束 ---

    # --- 输出系统配置信息 ---
    logger.info(f"=== ESM-2 蛋白质嵌入预计算系统 v2.0 ===")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"使用的 GPU IDs: {gpu_ids_to_use if num_gpus > 0 else 'CPU'}")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"最大序列长度: {args.max_length}")
    logger.info(f"输出注意力图: {not args.no_attentions}")
    logger.info(f"存储模式: {'合并存储' if args.use_merged else '分块存储'}")
    if not args.use_merged:
        logger.info(f"分块大小: {args.chunk_size} 序列/块")
    logger.info(f"强制重新计算: {args.force}")
    logger.info(f"通信端口: {args.port}")
    logger.info(f"处理文件: {args.file_names}")
    logger.info(f"=====================================")

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化分布式计算器
    computer = DistributedESM2Computer(
        model_path_or_name=args.model_path,
        output_dir=args.output_dir,
        gpu_ids=gpu_ids_to_use,
        batch_size=args.batch_size,
        max_length=args.max_length,
        force_recompute=args.force,
        master_port=args.port,
        output_attentions=not args.no_attentions,
        use_chunked_storage=not args.use_merged,
        chunk_size=args.chunk_size
    )

    # 处理指定的文件
    file_names = [name.strip() for name in args.file_names.split(',')]
    input_files = []

    # 查找要处理的文件
    for filename in file_names:
        filepath = os.path.join(args.data_dir, filename)
        if os.path.exists(filepath):
            input_files.append(filepath)
        else:
            logger.warning(f"输入文件未找到，将跳过: {filepath}")

    if not input_files:
        logger.error("未找到任何有效的输入文件。请检查数据目录和文件名。")
        return

    # 依次处理每个文件
    global_start_time = time.time()

    for input_file in input_files:
        try:
            file_start_time = time.time()
            logger.info(f"开始处理文件: {Path(input_file).name}")
            computer.compute_embeddings_for_file(input_file)
            file_time_taken = time.time() - file_start_time
            logger.info(f"文件 {Path(input_file).name} 处理完成，耗时 {file_time_taken:.2f} 秒")
        except Exception as e:
            logger.error(f"处理文件 {Path(input_file).name} 时发生错误: {e}")
            logger.error(traceback.format_exc())
            logger.info("继续处理下一个文件...")

    # 保存全局统计信息
    computer.save_global_stats()

    # 输出总体完成信息
    global_end_time = time.time()
    total_time = global_end_time - global_start_time

    # 计算总处理数量
    total_processed = sum(s['total'] for s in computer.stats.values())
    total_computed = sum(s['computed'] for s in computer.stats.values())
    total_cached = sum(s['cached'] for s in computer.stats.values())
    total_failed = sum(s['failed'] for s in computer.stats.values())

    logger.info(f"==================== 总结 ====================")
    logger.info(f"所有文件处理完成!")
    logger.info(f"总序列数: {total_processed}")
    logger.info(f"成功计算: {total_computed}")
    logger.info(f"使用缓存: {total_cached}")
    logger.info(f"处理失败: {total_failed}")

    if total_processed > 0:
        success_rate = (total_computed + total_cached) / total_processed * 100
        logger.info(f"成功率: {success_rate:.2f}%")

    # 计算吞吐量
    if total_time > 0 and total_computed > 0:
        throughput = total_computed / total_time
        logger.info(f"计算吞吐量: {throughput:.2f} 序列/秒")

    logger.info(f"总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
    logger.info(f"==========================================")

    # 输出结果存储位置信息
    if computer.use_chunked_storage:
        logger.info(f"嵌入结果以分块方式存储在: {computer.chunks_dir}")
        logger.info(f"可通过索引文件访问: {computer.output_dir}/*_index.pkl")
    else:
        logger.info(f"嵌入结果合并存储在: {computer.output_dir}/*_embedding.pt")

    logger.info(f"数据集映射关系保存在: {computer.output_dir}/*_mapping.json")
    logger.info(f"统计信息保存在: {os.path.join(computer.output_dir, 'precompute_summary_stats.json')}")

    logger.info("ESM2 预计算脚本执行完毕。")


if __name__ == "__main__":
    # 增加对 Windows 平台的兼容性检查
    if os.name == 'nt':
        # 在 Windows 上，需要将多进程代码放在 if __name__ == "__main__": 块内
        # 并且可能需要调整启动方法或使用其他策略
        logger.warning("检测到 Windows 平台，多进程行为可能与 Linux 不同。建议在 Linux 环境运行以获得最佳性能和稳定性。")
        # Windows 可能不支持 'spawn' 之外的启动方法
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            logger.warning("设置 'spawn' 启动方法失败。")

    main()