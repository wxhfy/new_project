#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESM-2 蛋白质嵌入分布式预计算系统 - (Transformers 版本)

使用 Hugging Face transformers 库处理 ESM-2 模型。
在多个 GPU 上并行计算蛋白质序列的嵌入 (以及可选的注意力图)，
并为每个输入数据集文件 (train/val/test) 生成独立的输出文件。
支持断点续传、负载均衡和错误恢复机制。

作者: wxhfy (基于 Transformers ESM-2 适配)
日期: 2025-04-13
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
from tqdm import tqdm
from functools import partial
from datetime import datetime
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import autocast # 用于混合精度推理

# 导入 Transformers 库
from transformers import AutoModel, AutoTokenizer

import h5py # 保留 HDF5 导入以备将来使用，但当前默认输出 PT

# --- 日志配置 ---
log_file = "esm2_precompute.log" # 日志文件名
logging.basicConfig(
    level=logging.INFO, # 日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # 日志格式
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'), # 保存到文件，使用 UTF-8 编码
        logging.StreamHandler() # 输出到控制台
    ]
)
logger = logging.getLogger("esm2_precompute") # 日志记录器名称
# --- 日志配置结束 ---

# --- 辅助函数 ---
def _extract_sequences(data_path):
    """从数据文件中提取序列信息"""
    logger.info(f"从 {data_path} 加载数据")
    try:
        # 加载 .pt 文件，假设包含一个列表的图数据对象或字典
        data = torch.load(data_path, map_location='cpu') # 加载到 CPU 避免占用 GPU 内存

        # 确保数据是列表格式
        if not isinstance(data, list):
            # 如果是字典，尝试提取值作为列表
            if isinstance(data, dict):
                data = list(data.values())
            else:
                data = [data] # 单个对象转为列表

        sequences = []
        sequence_ids = []
        unique_sequences = set() # 跟踪唯一序列以避免初始重复

        logger.info(f"开始从 {len(data)} 个图对象中提取序列...")
        for idx, item in enumerate(tqdm(data, desc=f"提取序列 {Path(data_path).stem}")):
            seq = None
            seq_id = None
            # 尝试从常见属性中获取序列和ID
            if hasattr(item, 'sequence') and isinstance(item.sequence, str) and item.sequence:
                seq = item.sequence
                # 尝试获取ID
                if hasattr(item, 'protein_id'):
                    seq_id = item.protein_id
                elif hasattr(item, 'id'):
                     seq_id = item.id
                elif hasattr(item, 'name'): # 备选ID字段
                     seq_id = item.name
                else:
                    # 如果没有显式ID，使用索引作为基础构建一个
                    seq_id = f"{Path(data_path).stem}_seq_{idx}"

            # 进一步检查字典格式（如果item是字典）
            elif isinstance(item, dict) and 'sequence' in item and item['sequence']:
                 seq = item['sequence']
                 seq_id = item.get('protein_id', item.get('id', item.get('name', f"{Path(data_path).stem}_seq_{idx}")))

            # 如果成功获取序列和ID
            if seq and seq_id:
                # 仅添加未见过的序列
                if seq not in unique_sequences:
                    sequences.append(seq)
                    sequence_ids.append(str(seq_id)) # 确保ID是字符串
                    unique_sequences.add(seq)
            # else:
                 # logger.debug(f"跳过索引 {idx}: 未找到有效的序列或ID。")

        logger.info(f"成功提取 {len(sequences)} 个唯一序列")
        return sequences, sequence_ids
    except FileNotFoundError:
        logger.error(f"数据文件未找到: {data_path}")
        return [], []
    except Exception as e:
        logger.error(f"数据加载或序列提取失败: {e}")
        logger.error(traceback.format_exc())
        return [], []

def _create_seq_hash(sequence):
    """为序列创建哈希值"""
    if not sequence: # 处理空序列
        return hashlib.md5("empty".encode('utf-8')).hexdigest()
    # 使用 UTF-8 编码
    return hashlib.md5(sequence.encode('utf-8')).hexdigest()
# --- 辅助函数结束 ---

class DistributedESM2Computer:
    """分布式 ESM-2 嵌入计算器 (Transformers 版本)"""

    def __init__(
        self,
        model_path_or_name, # 模型路径或名称
        output_dir, # 输出目录
        gpu_ids, # 使用的 GPU ID 列表
        batch_size=16, # 每个 GPU 的批处理大小
        max_length=1022, # ESM-2 的最大序列长度 (不含特殊 token)
        force_recompute=False, # 是否强制重新计算
        master_port=29501, # 分布式通信端口
        output_attentions=True # 是否计算并保存注意力图
    ):
        self.model_path_or_name = model_path_or_name
        self.output_dir = os.path.join(output_dir, "embeddings")
        self.gpu_ids = gpu_ids # 使用的 GPU ID 列表
        self.num_gpus = len(gpu_ids) if gpu_ids else 0 # 使用的 GPU 数量
        self.batch_size = batch_size
        self.max_length = max_length
        self.force_recompute = force_recompute
        self.master_port = master_port
        self.output_attentions = output_attentions
        self.format = "pt" # 固定输出格式为 .pt
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 临时目录 - 用于存储每个 GPU 的部分结果 (每个输入文件有自己的临时子目录)
        self.temp_base_dir = os.path.join(self.output_dir, "temp_embeddings_esm2")
        os.makedirs(self.temp_base_dir, exist_ok=True)

        # 统计信息跟踪
        self.stats = {} # 每个文件分开统计

    # --- Checkpoint 和 Index 相关函数 (调整为文件级别) ---
    def _get_file_specific_paths(self, input_filename_stem):
        """获取特定输入文件的索引和检查点路径"""
        index_file = os.path.join(self.output_dir, f"{input_filename_stem}_index.pkl")
        checkpoint_file = os.path.join(self.output_dir, f"{input_filename_stem}_checkpoint.json")
        temp_dir = os.path.join(self.temp_base_dir, input_filename_stem)
        os.makedirs(temp_dir, exist_ok=True) # 确保临时目录存在
        return index_file, checkpoint_file, temp_dir

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
        logger.info(f"为文件 {Path(index_file).stem.replace('_index','')} 创建新的嵌入索引")
        return {}

    def _save_index(self, embedding_index, index_file):
        """保存特定文件的索引"""
        try:
            with open(index_file, "wb") as f:
                pickle.dump(embedding_index, f)
            logger.debug(f"已保存文件索引 {Path(index_file).name}，包含 {len(embedding_index)} 条记录")
        except Exception as e:
            logger.error(f"保存索引文件 {Path(index_file).name} 失败: {e}")

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
        return {"processed_sequences": []}

    def _save_checkpoint(self, processed_sequences, checkpoint_file):
        """保存特定文件的检查点信息"""
        try:
            checkpoint = {"processed_sequences": processed_sequences}
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
            logger.debug(f"已保存文件检查点 {Path(checkpoint_file).name}")
        except Exception as e:
            logger.error(f"保存检查点文件 {Path(checkpoint_file).name} 失败: {e}")
    # --- Checkpoint 和 Index 相关函数结束 ---

    def _partition_sequences(self, sequences, sequence_ids, embedding_index, checkpoint_file):
        """根据缓存状态和任务平衡为单个文件分配序列任务"""
        checkpoint = self._load_checkpoint(checkpoint_file)
        processed_ids = set(checkpoint.get("processed_sequences", []))

        need_compute = []
        already_cached_count = 0

        for i, (seq_id, seq) in enumerate(zip(sequence_ids, sequences)):
            seq_hash = _create_seq_hash(seq)
            # 检查索引（现在是文件特定的）
            cache_exists = seq_id in embedding_index # or seq_hash in embedding_index (哈希查找可能不太必要，如果ID唯一)

            if (seq_id in processed_ids or cache_exists) and not self.force_recompute:
                already_cached_count += 1
            else:
                need_compute.append((i, seq_id, seq)) # (原始索引, ID, 序列)

        # 文件级别的统计
        file_stats = {
            "total": len(sequences),
            "cached": already_cached_count,
            "to_compute": len(need_compute)
        }
        logger.info(f"序列分区统计 - 总数: {file_stats['total']}, 已缓存/处理: {file_stats['cached']}, 待计算: {file_stats['to_compute']}")

        # 按序列长度排序以提高效率（短序列优先可能减少内存碎片）
        need_compute.sort(key=lambda x: len(x[2]))

        # 为 GPU 或 CPU 分配任务
        if self.num_gpus > 0:
            partitions = [[] for _ in range(self.num_gpus)]
            seq_lens = [0] * self.num_gpus # 按总长度进行负载均衡
            for item in need_compute:
                # 分配给当前总长度最短的 GPU
                min_gpu_idx = seq_lens.index(min(seq_lens))
                partitions[min_gpu_idx].append(item)
                seq_lens[min_gpu_idx] += len(item[2]) # 更新该 GPU 的总长度负载
            for gpu_idx, partition in enumerate(partitions):
                 logger.info(f"  GPU {self.gpu_ids[gpu_idx]} 分配到 {len(partition)} 个序列，总长度 {seq_lens[gpu_idx]}")
        else: # 仅 CPU
            partitions = [need_compute] # 所有任务都在一个分区
            logger.info(f"  CPU 分配到 {len(need_compute)} 个序列")

        return partitions, list(processed_ids), file_stats # 返回分区、已处理ID列表、文件统计

    def _init_process(self, rank_in_comm, partitions, world_size, init_method, temp_dir, input_filename_stem):
        """初始化分布式计算进程"""
        try:
            gpu_id_to_use = -1 # CPU 默认为 -1
            # 确定设备
            if self.num_gpus > 0:
                gpu_id_to_use = self.gpu_ids[rank_in_comm] # 从提供的列表中获取 GPU ID
                device = torch.device(f"cuda:{gpu_id_to_use}")
                torch.cuda.set_device(device)
                # 初始化分布式进程组
                dist.init_process_group(
                    backend='nccl', # NCCL 后端适用于 GPU
                    init_method=init_method,
                    world_size=world_size,
                    rank=rank_in_comm # 使用通信器内的排名
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

            # 决定加载 AutoModel 还是 AutoModelForMaskedLM
            # 如果只需要嵌入，AutoModel 更轻量。如果需要 logits（虽然这里不需要），用 MaskedLM。
            # 通常 AutoModel 就足够了。
            model_class = AutoModel # 默认使用 AutoModel
            # model_class = AutoModelForMaskedLM # 如果需要 MLM 头相关的输出

            # 使用半精度（float16）可能在兼容的 GPU 上加速推理并减少显存占用
            model_dtype = torch.float16 if torch.cuda.is_available() and self.num_gpus > 0 else torch.float32
            model = model_class.from_pretrained(
                self.model_path_or_name,
                torch_dtype=model_dtype # 加载时指定数据类型
            ).to(device)
            model.eval() # 设置为评估模式，禁用 dropout 等

            logger.info(f"进程 {rank_in_comm}: 模型和 Tokenizer 加载成功")

            # 处理分配到的序列分区
            # 传递临时目录和文件名用于保存部分结果
            self._process_partition(model, tokenizer, partition, rank_in_comm, device, temp_dir, input_filename_stem)

            # 清理模型和 Tokenizer 占用的内存
            del model, tokenizer
            if torch.cuda.is_available() and self.num_gpus > 0:
                torch.cuda.empty_cache() # 清空 GPU 缓存

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

    def _process_partition(self, model, tokenizer, partition, rank, device, temp_dir, input_filename_stem):
        """处理分配给当前进程的序列分区"""
        if not partition:
            logger.info(f"进程 {rank}: 无序列分配，退出。")
            self._save_empty_result(rank, temp_dir, input_filename_stem)
            return

        logger.info(f"进程 {rank}: 开始处理 {len(partition)} 个序列。")
        # 用于存储此分区结果的字典
        embeddings_dict = {}
        processed_count_in_partition = 0

        # 是否使用混合精度计算
        use_amp = torch.cuda.is_available() and self.num_gpus > 0

        # 设置进度条（仅主进程或CPU模式下显示）
        is_main_process = rank == 0 or self.num_gpus == 0
        pbar = tqdm(total=len(partition), desc=f"GPU {self.gpu_ids[rank]}" if self.num_gpus > 0 else "CPU", position=rank, disable=not is_main_process)

        # 按批次处理序列
        for i in range(0, len(partition), self.batch_size):
            # 获取当前批次的序列和 ID
            batch_items = partition[i : i + self.batch_size]
            # batch_indices = [item[0] for item in batch_items] # 原始索引
            batch_seq_ids = [item[1] for item in batch_items] # 序列 ID
            batch_sequences = [item[2] for item in batch_items] # 序列本身

            try:
                # 使用 Tokenizer 处理批次
                inputs = tokenizer(
                    batch_sequences,
                    return_tensors="pt", # 返回 PyTorch 张量
                    padding=True,       # 填充到批次中最长序列
                    truncation=True,    # 截断超过最大长度的序列
                    max_length=self.max_length, # 使用实例的最大长度设置
                ).to(device) # 将输入数据移动到目标设备

                # 计算嵌入和注意力
                with torch.no_grad(), autocast(enabled=use_amp): # 禁用梯度计算，并启用混合精度（如果可用）
                    outputs = model(
                        **inputs,
                        output_hidden_states=True,  # 确保返回隐藏状态
                        output_attentions=self.output_attentions # 根据设置决定是否计算注意力
                    )

                    # 提取最后一层隐藏状态作为嵌入
                    # 形状: [batch_size, seq_len, embedding_dim]
                    last_hidden_states = outputs.last_hidden_state

                    # 提取注意力图（如果计算了）
                    attentions = None
                    last_layer_attentions = None # 用于保存最后一层注意力
                    if self.output_attentions and outputs.attentions:
                        # 获取最后一层的注意力图
                        # 形状: [batch_size, num_heads, seq_len, seq_len]
                        last_layer_attentions = outputs.attentions[-1]
                        # 在头部维度上取平均，得到更简洁的注意力表示
                        # 形状: [batch_size, seq_len, seq_len]
                        attentions = last_layer_attentions.mean(dim=1)

                # --- 结果后处理与保存 ---
                # 遍历批次中的每个结果
                for j in range(last_hidden_states.size(0)):
                    seq_id = batch_seq_ids[j]           # 当前序列的 ID
                    original_seq = batch_sequences[j]   # 原始序列

                    # 获取真实的序列长度（去除填充部分）
                    attention_mask = inputs['attention_mask'][j]
                    true_len = attention_mask.sum().item()

                    # 提取每个残基的嵌入（去除 BOS 和 EOS token）
                    # 索引 0 是 BOS, true_len-1 是 EOS
                    embedding = last_hidden_states[j, 1:true_len-1, :] # 形状: [真实序列长度, embedding_dim]

                    # 提取对应的注意力图（如果计算了）
                    attention_map = None
                    if attentions is not None:
                        # 提取注意力图中对应真实序列的部分（去除 BOS/EOS 行和列）
                        attention_map = attentions[j, 1:true_len-1, 1:true_len-1] # 形状: [真实序列长度, 真实序列长度]

                    # 将结果移回 CPU 并转换为半精度（float16）以节省存储空间
                    embedding_cpu = embedding.detach().cpu().half()
                    attention_cpu = attention_map.detach().cpu().half() if attention_map is not None else None

                    # 准备要保存的数据
                    result_data = {
                        "embedding": embedding_cpu,
                        "sequence": original_seq, # 保存原始序列以供参考
                        # 如果计算了注意力，则包含它
                        "attention": attention_cpu,
                    }
                    # 如果 attention 是 None，从字典中移除该键
                    if result_data["attention"] is None:
                        del result_data["attention"]

                    # 将结果存入字典
                    embeddings_dict[seq_id] = result_data
                    processed_count_in_partition += 1

                    # 更新全局统计信息（仅需一次）
                    if self.stats.get("embedding_dim", 0) == 0 and embedding_cpu is not None:
                         self.stats["embedding_dim"] = embedding_cpu.shape[-1]
                    if self.stats.get("attention_layers", 0) == 0 and self.output_attentions and last_layer_attentions is not None:
                         self.stats["attention_layers"] = len(outputs.attentions) # 总层数
                         self.stats["attention_heads"] = last_layer_attentions.shape[1] # 头数

                # --- 后处理结束 ---

                # 更新进度条
                if is_main_process:
                    pbar.update(len(batch_items))

            except Exception as e:
                logger.error(f"进程 {rank}: 处理批次（起始索引 {i}）时出错: {e}")
                logger.error(traceback.format_exc())
                # 记录失败的序列 ID
                for item_id in batch_seq_ids:
                     # 在文件统计中标记失败
                     self.stats[input_filename_stem]['failed'] += 1
                     logger.warning(f"进程 {rank}: 失败的序列 ID: {item_id}")

        # 关闭进度条
        if is_main_process:
            pbar.close()

        # 保存此分区的最终结果
        self._save_temp_results(embeddings_dict, rank, temp_dir, input_filename_stem)
        # 更新文件统计中的计算数量
        self.stats[input_filename_stem]['computed'] += processed_count_in_partition
        logger.info(f"进程 {rank}: 处理完成。计算了 {processed_count_in_partition} 个嵌入。")


    def _save_empty_result(self, rank, temp_dir, input_filename_stem):
        """为当前文件保存一个空的临时结果文件"""
        temp_file = os.path.join(temp_dir, f"{input_filename_stem}_part_{rank}.pt")
        try:
            torch.save({}, temp_file) # 保存空字典
            logger.debug(f"进程 {rank}: 为文件 {input_filename_stem} 保存了空结果文件。")
        except Exception as e:
            logger.error(f"进程 {rank}: 保存空结果文件 {Path(temp_file).name} 失败: {e}")

    def _save_temp_results(self, embeddings_dict, rank, temp_dir, input_filename_stem):
        """将临时结果保存到特定于文件和进程的文件中"""
        if not embeddings_dict:
            self._save_empty_result(rank, temp_dir, input_filename_stem)
            return

        # 文件名包含输入文件名和进程排名
        temp_file = os.path.join(temp_dir, f"{input_filename_stem}_part_{rank}.pt")
        try:
            # 使用 pickle 最高协议可能提高效率
            torch.save(embeddings_dict, temp_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"进程 {rank}: 为文件 {input_filename_stem} 保存了 {len(embeddings_dict)} 条临时结果到 {Path(temp_file).name}")
        except Exception as e:
            logger.error(f"进程 {rank}: 保存临时结果文件 {Path(temp_file).name} 失败: {e}")
            logger.error(traceback.format_exc())

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

        merged_embeddings = {} # 用于存储合并后的嵌入
        total_loaded_count = 0
        failed_files = [] # 记录合并失败的文件

        # 遍历并加载每个临时文件
        for temp_file in tqdm(temp_files, desc=f"合并 {input_filename_stem}"):
            try:
                # 从 CPU 加载，避免 GPU 内存问题
                part_embeddings = torch.load(temp_file, map_location='cpu')
                if isinstance(part_embeddings, dict):
                    merged_embeddings.update(part_embeddings) # 合并字典
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
                torch.save(merged_embeddings, final_output_path, pickle_protocol=pickle.HIGHEST_PROTOCOL, _use_new_zipfile_serialization=True)
                logger.info(f"成功合并 {len(merged_embeddings)} 条嵌入记录到: {final_output_path}")

                # 更新文件索引（指向最终合并的文件）
                output_filename = Path(final_output_path).name
                for seq_id, data in merged_embeddings.items():
                     embedding_index[seq_id] = {
                         "hash": _create_seq_hash(data.get("sequence", "")),
                         "sequence": data.get("sequence", ""),
                         "file": output_filename, # 指向合并后的文件名
                         "timestamp": time.time()
                     }
                self._save_index(embedding_index, index_file) # 保存更新后的索引

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
        input_filename_stem = Path(input_file_path).stem # 获取文件名（不含扩展名）
        logger.info(f"===== 开始处理文件: {input_filename_stem} =====")

        # 初始化此文件的统计信息
        self.stats[input_filename_stem] = {
            "total": 0, "computed": 0, "cached": 0, "failed": 0,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 获取文件特定的路径
        index_file, checkpoint_file, temp_dir = self._get_file_specific_paths(input_filename_stem)

        # 加载此文件的索引
        embedding_index = self._load_or_create_index(index_file)

        # 提取序列
        sequences, sequence_ids = _extract_sequences(input_file_path)
        if not sequences:
            logger.error(f"文件 {input_filename_stem} 未提取到有效序列，跳过。")
            self.stats[input_filename_stem]['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return

        # 分配任务
        partitions, processed_ids, file_stats = self._partition_sequences(sequences, sequence_ids, embedding_index, checkpoint_file)
        self.stats[input_filename_stem].update(file_stats) # 更新统计

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
                pass # 如果设置失败，继续使用默认方法

            try:
                logger.info(f"为文件 {input_filename_stem} 启动 {world_size} 个 GPU 进程...")
                # 启动进程，传递特定于文件的临时目录和文件名
                for rank_in_comm in range(world_size):
                    p = Process(target=self._init_process, args=(rank_in_comm, partitions, world_size, init_method, temp_dir, input_filename_stem))
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
        else: # CPU 计算
            logger.info(f"在 CPU 上计算文件 {input_filename_stem} 的嵌入...")
            # 在主进程中直接运行
            self._init_process(0, partitions, 1, None, temp_dir, input_filename_stem) # rank=0, world_size=1
            logger.info(f"文件 {input_filename_stem} 的 CPU 计算完成。")

        # --- 合并结果 ---
        # 定义最终输出文件路径
        final_output_path = os.path.join(self.output_dir, f"{input_filename_stem}_embedding.pt")
        # 合并此文件的临时结果
        merge_success = self._merge_embeddings_for_file(input_filename_stem, temp_dir, final_output_path, embedding_index, index_file)

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
                  self.stats[input_filename_stem]['failed'] = file_stats['to_compute'] - self.stats[input_filename_stem]['computed'] # 估算失败数

        else:
            self.stats[input_filename_stem]['failed'] = file_stats['to_compute'] # 如果合并失败，认为所有待计算的都失败了
            self.stats[input_filename_stem]['computed'] = 0

        # 保存此文件的检查点（标记所有已处理的ID）
        # 首先，获取本次计算中新处理的所有 ID
        newly_processed_ids_set = set(id for _, id, _ in sum(partitions, []))
        # 然后，将从检查点加载的 processed_ids (确保是集合类型) 与新处理的 ID 集合合并
        current_processed_ids_set = set(processed_ids).union(newly_processed_ids_set) # 确保 processed_ids 是集合
        # 最后，转换回列表以保存到 JSON
        self._save_checkpoint(list(current_processed_ids_set), checkpoint_file)

        # 记录结束时间
        self.stats[input_filename_stem]['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"===== 完成文件: {input_filename_stem} =====")
        logger.info(f"  统计: {self.stats[input_filename_stem]}")
        logger.info(f"  输出文件: {final_output_path if merge_success else '失败'}")
        logger.info(f"==========================================")

    def save_global_stats(self):
        """保存所有文件的全局统计信息"""
        global_stats_file = os.path.join(self.output_dir, "precompute_summary_stats.json")
        # 添加全局汇总信息
        total_processed = sum(s['total'] for s in self.stats.values())
        total_computed = sum(s['computed'] for s in self.stats.values())
        total_cached = sum(s['cached'] for s in self.stats.values())
        total_failed = sum(s['failed'] for s in self.stats.values())

        summary_data = {
             "global_summary": {
                 "total_sequences_processed": total_processed,
                 "total_embeddings_computed": total_computed,
                 "total_from_cache": total_cached,
                 "total_failed": total_failed,
                 "embedding_dim": self.stats.get("embedding_dim", 0), # 从第一个文件的统计中获取
                 "attention_layers": self.stats.get("attention_layers", 0),
                 "attention_heads": self.stats.get("attention_heads", 0),
                 "model_name": self.model_path_or_name,
                 "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
             },
             "file_details": self.stats # 包含每个文件的详细统计
        }

        try:
            with open(global_stats_file, "w") as f:
                json.dump(summary_data, f, indent=2)
            logger.info(f"全局统计信息已保存到: {global_stats_file}")
        except Exception as e:
            logger.error(f"保存全局统计信息失败: {e}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ESM-2 嵌入分布式预计算工具 (Transformers 版本, 中文注释)")
    parser.add_argument("--data-dir", type=str, default="data", help="包含 train.pt, val.pt, test.pt 的数据目录")
    parser.add_argument("--output-dir", type=str, default="data", help="保存 *_embedding.pt 文件的输出目录 (默认与 --data-dir 相同)")
    parser.add_argument("--model-path", type=str, default="/home/fyh0106/project/encoder/data/esm2/", help="本地 ESM-2 模型目录路径")
    parser.add_argument("--gpu-ids", type=str, default="1,2,3", help="要使用的 GPU 设备 ID 列表，用逗号分隔 (例如 '0,1,2')。留空则使用所有可用 GPU。")
    parser.add_argument("--batch-size", type=int, default=256, help="每个 GPU (或 CPU 进程) 的批处理大小")
    parser.add_argument("--max-length", type=int, default=1022, help="Tokenizer 处理的最大序列长度 (不含特殊 token)")
    parser.add_argument("--force", action="store_true", help="强制重新计算所有嵌入，忽略缓存和检查点")
    parser.add_argument("--port", type=int, default=29501, help="分布式通信的主端口")
    parser.add_argument("--no-attentions", action="store_true", help="禁用计算和保存注意力图 (默认为保存)")
    return parser.parse_args()

def main():
    """主执行函数"""
    args = parse_arguments()

    # --- GPU ID 处理 ---
    gpu_ids_to_use = []
    if args.gpu_ids:
        try:
            gpu_ids_to_use = [int(gid.strip()) for gid in args.gpu_ids.split(',')]
            available_gpus = list(range(torch.cuda.device_count()))
            # 过滤掉无效或不存在的 GPU ID
            valid_gpu_ids = [gid for gid in gpu_ids_to_use if gid in available_gpus]
            if len(valid_gpu_ids) != len(gpu_ids_to_use):
                logger.warning(f"提供的 GPU IDs: {gpu_ids_to_use} 包含无效 ID。仅使用有效的: {valid_gpu_ids}")
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

    logger.info(f"--- ESM-2 嵌入预计算 (中文注释版) ---")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"使用的 GPU IDs: {gpu_ids_to_use if num_gpus > 0 else 'CPU'}")
    logger.info(f"每个设备的批大小: {args.batch_size}")
    logger.info(f"最大序列长度: {args.max_length}")
    logger.info(f"输出注意力图: {not args.no_attentions}")
    logger.info(f"强制重新计算: {args.force}")
    logger.info(f"主端口: {args.port}")
    logger.info(f"------------------------------------")

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化分布式计算器
    computer = DistributedESM2Computer(
        model_path_or_name=args.model_path,
        output_dir=args.output_dir, # 输出目录
        gpu_ids=gpu_ids_to_use,     # 使用解析后的 GPU ID 列表
        batch_size=args.batch_size,
        max_length=args.max_length,
        force_recompute=args.force,
        master_port=args.port,
        output_attentions=not args.no_attentions
    )

    # 查找要处理的文件
    input_files = []
    for filename in ["train_data.pt", "val_data.pt", "test_data.pt"]:
        filepath = os.path.join(args.data_dir, filename)
        if os.path.exists(filepath):
            input_files.append(filepath)
        else:
            logger.warning(f"输入文件未找到，将跳过: {filepath}")

    if not input_files:
        logger.error("在数据目录中未找到任何 train_data.pt, val_data.pt, test_data.pt 文件。退出。")
        return

    # 依次处理每个文件
    global_start_time = time.time()
    for input_file in input_files:
        computer.compute_embeddings_for_file(input_file)

    # 保存全局统计信息
    computer.save_global_stats()
    global_end_time = time.time()
    logger.info(f"所有文件处理完成，总耗时: {global_end_time - global_start_time:.2f} 秒")
    logger.info("预计算脚本执行完毕。")

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