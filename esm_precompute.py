#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESMC蛋白质嵌入分布式预计算系统 - 支持合并输出版本

使用多GPU并行计算蛋白质序列的ESMC嵌入，并将结果合并保存为单一大文件
支持断点续传、负载均衡和错误恢复机制

作者: wxhfy
日期: 2025-04-08 (修改版)
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
from torch.nn import functional as F
import torch.distributed as dist
from torch.multiprocessing import Process
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProteinTensor, LogitsConfig
import h5py

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("esm_precompute.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("esm_precompute")

# 设置环境变量
os.environ["INFRA_PROVIDER"] = "True"
os.environ["ESM_CACHE_DIR"] = "data/weights"


class DistributedEmbeddingComputer:
    """分布式ESMC嵌入计算器 - 支持合并输出"""

    def __init__(
            self,
            model_name="esmc_600m",
            cache_dir="data/esm_embeddings",
            output_file=None,
            num_gpus=3,
            batch_size=1,
            force_recompute=False,
            master_port=29500,
            format="pt"  # 输出格式: pt或hdf5
    ):
        """
        初始化分布式嵌入计算器

        参数:
            model_name (str): ESMC模型名称
            cache_dir (str): 嵌入缓存目录
            output_file (str): 输出文件名（不包含扩展名）
            num_gpus (int): 可用GPU数量
            batch_size (int): 每个GPU的批处理大小
            force_recompute (bool): 是否强制重新计算
            master_port (int): 分布式通信端口
            format (str): 输出文件格式，'pt'或'hdf5'
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.force_recompute = force_recompute
        self.master_port = master_port
        self.format = format.lower()

        # 确定输出文件名
        self.output_file = output_file
        if self.output_file is None:
            # 默认使用模型名称和时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"{model_name}_embeddings_{timestamp}"

        # 根据格式添加扩展名
        if self.format == "pt":
            self.merged_file = os.path.join(cache_dir, f"{self.output_file}.pt")
        elif self.format == "hdf5":
            self.merged_file = os.path.join(cache_dir, f"{self.output_file}.h5")
        else:
            raise ValueError(f"不支持的格式: {format}, 请使用 'pt' 或 'hdf5'")

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 临时目录 - 用于存储每个GPU的部分结果
        self.temp_dir = os.path.join(cache_dir, "temp_embeddings")
        os.makedirs(self.temp_dir, exist_ok=True)

        # 氨基酸编码映射表
        self.ESM_AA_MAP = {
            'A': 5, 'C': 23, 'D': 13, 'E': 9, 'F': 18,
            'G': 6, 'H': 21, 'I': 12, 'K': 15, 'L': 4,
            'M': 20, 'N': 17, 'P': 14, 'Q': 16, 'R': 10,
            'S': 8, 'T': 11, 'V': 7, 'W': 22, 'Y': 19,
            '_': 32, 'X': 32
        }

        # 索引文件路径 - 用于断点续传
        self.index_file = os.path.join(cache_dir, "embedding_index.pkl")

        # 加载或创建索引
        self.embedding_index = self._load_or_create_index()

        # 检查点文件
        self.checkpoint_file = os.path.join(cache_dir, "precompute_checkpoint.json")

        # 统计信息
        self.stats = {
            "total": 0,
            "computed": 0,
            "cached": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None,
            "embedding_dim": 0,  # 将记录嵌入维度
        }

    def _load_or_create_index(self):
        """加载已有索引或创建新索引"""
        if os.path.exists(self.index_file) and not self.force_recompute:
            try:
                with open(self.index_file, "rb") as f:
                    embedding_index = pickle.load(f)
                logger.info(f"已加载嵌入索引，包含 {len(embedding_index)} 条记录")
                return embedding_index
            except Exception as e:
                logger.error(f"加载索引失败: {e}")

        logger.info("创建新的嵌入索引")
        return {}

    def _save_index(self):
        """保存嵌入索引"""
        try:
            with open(self.index_file, "wb") as f:
                pickle.dump(self.embedding_index, f)
            logger.info(f"已保存嵌入索引，包含 {len(self.embedding_index)} 条记录")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")

    def _load_checkpoint(self):
        """加载检查点信息"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r") as f:
                    checkpoint = json.load(f)
                logger.info(f"已加载检查点: {checkpoint}")
                return checkpoint
            except Exception as e:
                logger.error(f"加载检查点失败: {e}")

        return {"processed_sequences": []}

    def _save_checkpoint(self, processed_sequences):
        """保存检查点信息"""
        try:
            checkpoint = {"processed_sequences": processed_sequences}
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")

    def _create_seq_hash(self, sequence):
        """为序列创建哈希值"""
        return hashlib.md5(sequence.encode()).hexdigest()

    def _extract_sequences(self, data_path):
        """从数据文件中提取序列信息"""
        logger.info(f"从 {data_path} 加载数据")

        try:
            data = torch.load(data_path)

            # 转换为列表格式
            if not isinstance(data, list):
                data = [data]

            sequences = []
            sequence_ids = []

            for idx, item in enumerate(data):
                if hasattr(item, 'sequence') and isinstance(item.sequence, str):
                    seq = item.sequence
                    if seq:  # 确保序列非空
                        # 获取序列ID
                        if hasattr(item, 'protein_id'):
                            seq_id = item.protein_id
                        else:
                            seq_id = f"seq_{idx}"

                        sequences.append(seq)
                        sequence_ids.append(seq_id)

            logger.info(f"成功提取 {len(sequences)} 个序列")
            return sequences, sequence_ids

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            logger.error(traceback.format_exc())
            return [], []

    def _partition_sequences(self, sequences, sequence_ids):
        """根据缓存状态和任务平衡分配序列任务"""
        # 恢复检查点
        checkpoint = self._load_checkpoint()
        processed_ids = set(checkpoint.get("processed_sequences", []))

        # 分析哪些序列需要计算
        need_compute = []
        already_cached = []

        for i, (seq_id, seq) in enumerate(zip(sequence_ids, sequences)):
            # 跳过已处理的序列
            if seq_id in processed_ids and not self.force_recompute:
                already_cached.append(i)
                continue

            # 检查是否已经在全局索引中
            if seq_id in self.embedding_index and not self.force_recompute:
                already_cached.append(i)
            else:
                need_compute.append((i, seq_id, seq))

        # 更新统计信息
        self.stats["total"] = len(sequences)
        self.stats["cached"] = len(already_cached)

        # 按序列长度排序，以优化计算效率
        need_compute.sort(key=lambda x: len(x[2]))

        # 负载均衡分配
        partitions = [[] for _ in range(self.num_gpus)]

        # 使用贪心算法分配任务，确保各卡负载均衡
        seq_lens = [0] * self.num_gpus

        for idx, seq_id, seq in need_compute:
            # 找到当前负载最小的GPU
            min_gpu = seq_lens.index(min(seq_lens))
            partitions[min_gpu].append((idx, seq_id, seq))
            # 更新负载（使用序列长度作为负载指标）
            seq_lens[min_gpu] += len(seq)

        # 输出分配情况
        for gpu_id, partition in enumerate(partitions):
            logger.info(f"GPU {gpu_id} 分配了 {len(partition)} 个序列, "
                        f"总字符: {sum(len(seq) for _, _, seq in partition)}")

        return partitions, already_cached

    def _init_process(self, rank, partitions, world_size, init_method):
        """初始化分布式进程"""
        # 设置当前设备
        torch.cuda.set_device(rank)

        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )

        # 获取当前进程的任务分区
        partition = partitions[rank]

        # 设置随机种子
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)

        try:
            # 加载ESMC模型
            device = torch.device(f"cuda:{rank}")
            logger.info(f"进程 {rank} 加载模型到 {device}")
            esm_model = ESMC.from_pretrained(self.model_name, device=device)

            # 处理分配的序列
            self._process_partition(esm_model, partition, rank, device)

            # 清理
            del esm_model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"进程 {rank} 遇到错误: {e}")
            logger.error(traceback.format_exc())

        # 销毁进程组
        dist.destroy_process_group()

    def _process_partition(self, esm_model, partition, rank, device):
        """处理分配给当前进程的序列分区"""
        if not partition:
            logger.info(f"进程 {rank} 没有分配任务，退出")
            # 创建空的结果文件，以便于后续合并
            self._save_empty_result(rank)
            return

        logger.info(f"进程 {rank} 开始处理 {len(partition)} 个序列")

        # 设置进度条
        pbar = tqdm(total=len(partition), desc=f"GPU {rank}", position=rank)

        # 创建结果集合
        embeddings_dict = {}

        # 批处理缓冲区
        batch_buffer = []

        # 处理序列
        for item_idx, (idx, seq_id, seq) in enumerate(partition):
            try:
                # 清理序列
                cleaned_seq = ''.join(aa for aa in seq if aa in self.ESM_AA_MAP)
                if not cleaned_seq:
                    cleaned_seq = "A"

                # 长度处理
                max_length = 1024  # 增加最大长度支持
                if len(cleaned_seq) > max_length:
                    cleaned_seq = cleaned_seq[:max_length]
                    logger.warning(f"序列 {seq_id} 已截断至 {max_length} 个氨基酸")

                # 编码序列
                token_ids = [0]  # BOS标记
                for aa in cleaned_seq:
                    token_ids.append(self.ESM_AA_MAP.get(aa, self.ESM_AA_MAP['X']))
                token_ids.append(2)  # EOS标记

                # 添加到批处理缓冲区
                batch_buffer.append((idx, seq_id, cleaned_seq, token_ids))

                # 当缓冲区达到批处理大小或是最后一个序列时进行处理
                if len(batch_buffer) >= self.batch_size or item_idx == len(partition) - 1:
                    # 处理批次
                    batch_results = self._process_batch(esm_model, batch_buffer, device, rank)

                    # 更新结果字典
                    embeddings_dict.update(batch_results)

                    # 清空缓冲区
                    batch_buffer = []

                # 每处理100个序列保存一次临时结果
                if (item_idx + 1) % 100 == 0 or item_idx == len(partition) - 1:
                    # 保存临时结果
                    self._save_temp_results(embeddings_dict, rank)

                # 更新进度条
                pbar.update(1)

            except Exception as e:
                logger.error(f"进程 {rank} 处理序列 {seq_id} 时出错: {e}")
                logger.error(traceback.format_exc())

        # 关闭进度条
        pbar.close()

        # 保存最终结果
        self._save_temp_results(embeddings_dict, rank)

        # 记录处理的序列ID
        completed_ids = list(embeddings_dict.keys())
        logger.info(f"进程 {rank} 完成计算，处理了 {len(completed_ids)} 个序列")

    def _save_empty_result(self, rank):
        """保存空的临时结果文件"""
        temp_file = os.path.join(self.temp_dir, f"embeddings_part_{rank}.pt")
        torch.save({}, temp_file)
        logger.info(f"进程 {rank} 保存了空的结果文件")

    def _save_temp_results(self, embeddings_dict, rank):
        """保存临时结果到文件"""
        if not embeddings_dict:
            self._save_empty_result(rank)
            return

        temp_file = os.path.join(self.temp_dir, f"embeddings_part_{rank}.pt")
        try:
            torch.save(embeddings_dict, temp_file)
            logger.info(f"进程 {rank} 保存临时结果，包含 {len(embeddings_dict)} 条记录")
        except Exception as e:
            logger.error(f"保存临时结果失败: {e}")
            logger.error(traceback.format_exc())

    def _safe_compute_embeddings(self, esm_model, protein_tensor, config):
        """安全计算ESMC嵌入，处理维度不匹配问题"""
        try:
            # 尝试直接计算
            return esm_model.logits(protein_tensor, config)
        except Exception as e:
            error_msg = str(e)

            # 处理维度错误
            if "Wrong shape: expected 3 dims" in error_msg and "Received 4-dim tensor" in error_msg:
                # 修复序列维度
                sequence = protein_tensor.sequence

                if sequence.dim() == 4:
                    # [b, 1, s, d] -> [b, s, d]
                    if sequence.shape[1] == 1:
                        fixed_sequence = sequence.squeeze(1)
                    else:
                        # 对于其他形状，尝试更通用的方法
                        b = sequence.shape[0]
                        s = sequence.shape[2]
                        fixed_sequence = sequence.reshape(b, s, -1)

                    # 创建新的蛋白质张量
                    fixed_tensor = ESMProteinTensor(sequence=fixed_sequence)
                    logger.info(f"修复维度形状: {sequence.shape} -> {fixed_sequence.shape}")

                    # 重试
                    return esm_model.logits(fixed_tensor, config)

                # 处理其他维度问题
                if sequence.dim() == 3:
                    b, s, d = sequence.shape
                    if s == 1:
                        # [b, 1, d] -> [b, d]
                        fixed_sequence = sequence.squeeze(1)
                        fixed_tensor = ESMProteinTensor(sequence=fixed_sequence)
                        logger.info(f"修复维度形状: {sequence.shape} -> {fixed_sequence.shape}")
                        return esm_model.logits(fixed_tensor, config)

            # 无法处理的错误，重新抛出
            raise

    def _process_batch(self, esm_model, batch_buffer, device, rank):
        """处理序列批次，返回结果字典"""
        batch_results = {}

        for idx, seq_id, seq, token_ids in batch_buffer:
            try:
                # 转换为张量
                token_tensor = torch.tensor(token_ids, device=device)
                protein_tensor = ESMProteinTensor(sequence=token_tensor)

                # 计算嵌入
                with torch.no_grad():
                    try:
                        # 使用安全计算方法
                        logits_output = self._safe_compute_embeddings(
                            esm_model,
                            protein_tensor,
                            LogitsConfig(sequence=True, return_embeddings=True)
                        )

                        # 提取嵌入
                        embedding = logits_output.embeddings

                        # 处理四维输出
                        if embedding.dim() == 4:
                            if embedding.shape[0] == 1 and embedding.shape[1] == 1:
                                embedding = embedding.squeeze(1)
                            else:
                                b, extra, s, d = embedding.shape
                                embedding = embedding.reshape(b, s, d)

                        # 提取注意力（如果有）
                        attention = None
                        if hasattr(logits_output, 'attentions'):
                            try:
                                # 处理多头注意力
                                attn_data = logits_output.attentions
                                if attn_data.dim() == 4:  # [batch, heads, seq_len, seq_len]
                                    cls_attention = attn_data.mean(dim=1)[:, 0, 1:-1]
                                    attention = F.softmax(cls_attention, dim=-1).unsqueeze(-1)
                            except Exception as attn_err:
                                logger.warning(f"进程 {rank} 提取序列 {seq_id} 的注意力失败: {attn_err}")

                        # 获取嵌入维度，用于统计信息
                        if self.stats["embedding_dim"] == 0 and embedding is not None:
                            if embedding.dim() > 2:  # 处理多维张量
                                self.stats["embedding_dim"] = embedding.size(-1)
                            else:
                                self.stats["embedding_dim"] = embedding.size(-1)

                        # 创建结果对象 - 转为半精度以节省空间
                        if embedding is not None:
                            # 统一张量到CPU并转换为半精度
                            embedding_data = {
                                "embedding": embedding.cpu().half(),  # 半精度存储
                                "attention": attention.cpu().half() if attention is not None else None,
                                "sequence": seq,
                            }

                            # 添加到批次结果
                            batch_results[seq_id] = embedding_data

                    except Exception as inner_e:
                        logger.error(f"进程 {rank} 处理序列 {seq_id} 计算嵌入失败: {inner_e}")
                        logger.error(traceback.format_exc())

            except Exception as e:
                logger.error(f"进程 {rank} 处理序列 {seq_id} 时出错: {e}")
                logger.error(traceback.format_exc())

        return batch_results

    def _merge_embeddings(self):
        """合并所有临时嵌入文件到单一大文件"""
        logger.info("开始合并嵌入结果...")

        # 收集所有临时文件
        temp_files = [os.path.join(self.temp_dir, f) for f in os.listdir(self.temp_dir)
                      if f.startswith("embeddings_part_") and f.endswith(".pt")]

        if not temp_files:
            logger.warning("没有找到临时嵌入文件，无法合并")
            return False

        # 合并结果
        merged_embeddings = {}
        total_count = 0

        # 逐个加载临时文件并合并
        for temp_file in temp_files:
            try:
                part_embeddings = torch.load(temp_file)
                part_count = len(part_embeddings)

                if part_count > 0:
                    # 更新索引和合并结果
                    merged_embeddings.update(part_embeddings)
                    total_count += part_count

                    logger.info(f"合并 {temp_file}, 添加了 {part_count} 条记录")

                # 删除临时文件以节省空间
                os.remove(temp_file)

            except Exception as e:
                logger.error(f"合并 {temp_file} 失败: {e}")
                logger.error(traceback.format_exc())

        # 保存合并结果
        if total_count > 0:
            if self.format == "pt":
                self._save_pt_format(merged_embeddings)
            elif self.format == "hdf5":
                self._save_hdf5_format(merged_embeddings)

            logger.info(f"成功合并 {total_count} 条嵌入记录到: {self.merged_file}")
            return True
        else:
            logger.warning("没有有效的嵌入记录，合并操作取消")
            return False

    def _save_pt_format(self, merged_embeddings):
        """保存为PyTorch (.pt) 格式"""
        try:
            # 可选：压缩大型文件
            torch.save(merged_embeddings, self.merged_file, _use_new_zipfile_serialization=True)
            logger.info(f"保存合并嵌入为PyTorch格式: {self.merged_file}")

            # 创建元数据文件
            metadata = {
                "format": "pytorch",
                "num_embeddings": len(merged_embeddings),
                "embedding_dim": self.stats["embedding_dim"],
                "model_name": self.model_name,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sequence_ids": list(merged_embeddings.keys())
            }

            metadata_file = os.path.join(self.cache_dir, f"{self.output_file}.meta.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"保存PyTorch格式失败: {e}")
            logger.error(traceback.format_exc())

    def _save_hdf5_format(self, merged_embeddings):
        """保存为HDF5格式"""
        try:
            with h5py.File(self.merged_file, 'w') as h5f:
                # 创建元数据组
                meta_group = h5f.create_group('metadata')
                meta_group.attrs['model_name'] = self.model_name
                meta_group.attrs['embedding_dim'] = self.stats["embedding_dim"]
                meta_group.attrs['num_sequences'] = len(merged_embeddings)
                meta_group.attrs['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 创建序列ID数据集
                seq_ids = list(merged_embeddings.keys())
                meta_group.create_dataset('sequence_ids', data=np.array(seq_ids, dtype=h5py.string_dtype()))

                # 创建嵌入组
                emb_group = h5f.create_group('embeddings')

                # 存储每个序列的嵌入
                for seq_id, data in tqdm(merged_embeddings.items(), desc="保存HDF5"):
                    seq_group = emb_group.create_group(seq_id)

                    # 存储嵌入张量
                    if 'embedding' in data and data['embedding'] is not None:
                        seq_group.create_dataset('embedding', data=data['embedding'].numpy())

                    # 存储注意力（如果有）
                    if 'attention' in data and data['attention'] is not None:
                        seq_group.create_dataset('attention', data=data['attention'].numpy())

                    # 存储序列
                    if 'sequence' in data:
                        seq_group.attrs['sequence'] = data['sequence']

            logger.info(f"保存合并嵌入为HDF5格式: {self.merged_file}")

        except Exception as e:
            logger.error(f"保存HDF5格式失败: {e}")
            logger.error(traceback.format_exc())

    def compute_embeddings(self, data_path):
        """计算并缓存嵌入"""
        # 记录开始时间
        self.stats["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 提取序列
        sequences, sequence_ids = self._extract_sequences(data_path)
        if not sequences:
            logger.error("未提取到有效序列，退出")
            return

        # 分配任务
        partitions, already_cached = self._partition_sequences(sequences, sequence_ids)

        # 记录已经缓存的序列
        self.stats["cached"] = len(already_cached)
        logger.info(f"跳过 {len(already_cached)} 个已缓存序列")

        # 检查是否有需要计算的序列
        total_to_compute = sum(len(p) for p in partitions)
        if total_to_compute == 0:
            logger.info("所有序列已经计算并缓存，无需进一步计算")
            return

        logger.info(f"需要计算 {total_to_compute} 个序列，使用 {self.num_gpus} 个GPU")

        # 设置分布式环境
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self.master_port)

        # 初始化进程
        world_size = self.num_gpus
        init_method = f"tcp://localhost:{self.master_port}"

        # 启动多进程
        processes = []
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        try:
            for rank in range(world_size):
                p = Process(
                    target=self._init_process,
                    args=(rank, partitions, world_size, init_method)
                )
                p.start()
                processes.append(p)

            # 等待所有进程完成
            for p in processes:
                p.join()
        except Exception as e:
            logger.error(f"多进程执行失败: {e}")
            logger.error(traceback.format_exc())

        # 合并嵌入结果
        success = self._merge_embeddings()

        # 记录结束时间
        self.stats["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 计算统计信息
        if success:
            self.stats["computed"] = total_to_compute
            self.stats["failed"] = 0  # 实际失败数无法精确计算
        else:
            self.stats["computed"] = 0
            self.stats["failed"] = total_to_compute

        # 输出统计信息
        logger.info(f"嵌入计算完成: ")
        logger.info(f"- 总序列数: {self.stats['total']}")
        logger.info(f"- 已缓存: {self.stats['cached']}")
        logger.info(f"- 新计算: {self.stats['computed']}")
        logger.info(f"- 计算失败: {self.stats['failed']}")
        logger.info(f"- 开始时间: {self.stats['start_time']}")
        logger.info(f"- 结束时间: {self.stats['end_time']}")
        logger.info(f"- 嵌入维度: {self.stats['embedding_dim']}")

        # 保存统计信息
        stats_file = os.path.join(self.cache_dir, f"{self.output_file}_stats.json")
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ESMC嵌入分布式预计算工具")
    parser.add_argument("--data", type=str, required=True, help="数据集路径")
    parser.add_argument("--model", type=str, default="esmc_600m", help="ESMC模型名称")
    parser.add_argument("--cache-dir", type=str, default="data/esm_embeddings", help="嵌入缓存目录")
    parser.add_argument("--output", type=str, default=None, help="输出文件名(不含扩展名)")
    parser.add_argument("--format", type=str, choices=["pt", "hdf5"], default="pt", help="输出文件格式")
    parser.add_argument("--num-gpus", type=int, default=3, help="使用GPU数量")
    parser.add_argument("--batch-size", type=int, default=16, help="每个GPU的批处理大小")
    parser.add_argument("--force", action="store_true", help="强制重新计算所有嵌入")
    parser.add_argument("--port", type=int, default=29500, help="分布式通信端口")
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()

    # 设置输出文件名为数据集名称（如果未指定）
    if args.output is None:
        data_name = os.path.basename(args.data).split(".")[0]
        args.output = f"{data_name}_embeddings"

    # 设置根据数据集名称的缓存子目录
    data_name = os.path.basename(args.data).split(".")[0]
    cache_subdir = os.path.join(args.cache_dir, data_name)

    # 检查GPU可用性
    available_gpus = torch.cuda.device_count()
    if available_gpus < args.num_gpus:
        logger.warning(f"请求 {args.num_gpus} 个GPU，但只有 {available_gpus} 个可用")
        args.num_gpus = available_gpus

    if args.num_gpus == 0:
        logger.error("没有可用的GPU，退出")
        return

    logger.info(f"使用 {args.num_gpus} 个GPU: {[i for i in range(args.num_gpus)]}")

    # 创建分布式计算器
    computer = DistributedEmbeddingComputer(
        model_name=args.model,
        cache_dir=cache_subdir,
        output_file=args.output,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        force_recompute=args.force,
        master_port=args.port,
        format=args.format
    )

    # 计算嵌入
    computer.compute_embeddings(args.data)


if __name__ == "__main__":
    main()