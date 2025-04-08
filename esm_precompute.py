#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESMC蛋白质嵌入分布式预计算系统

使用多GPU并行计算蛋白质序列的ESMC嵌入，并进行缓存管理
支持断点续传、负载均衡和错误恢复机制

作者: wxhfy
日期: 2025-04-05
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
    """分布式ESMC嵌入计算器"""

    def __init__(
            self,
            model_name="esmc_600m",
            cache_dir="data/esm_embeddings",
            num_gpus=3,
            batch_size=1,
            force_recompute=False,
            master_port=29500
    ):
        """
        初始化分布式嵌入计算器

        参数:
            model_name (str): ESMC模型名称
            cache_dir (str): 嵌入缓存目录
            num_gpus (int): 可用GPU数量
            batch_size (int): 每个GPU的批处理大小
            force_recompute (bool): 是否强制重新计算
            master_port (int): 分布式通信端口
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.force_recompute = force_recompute
        self.master_port = master_port

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 氨基酸编码映射表
        self.ESM_AA_MAP = {
            'A': 5, 'C': 23, 'D': 13, 'E': 9, 'F': 18,
            'G': 6, 'H': 21, 'I': 12, 'K': 15, 'L': 4,
            'M': 20, 'N': 17, 'P': 14, 'Q': 16, 'R': 10,
            'S': 8, 'T': 11, 'V': 7, 'W': 22, 'Y': 19,
            '_': 32, 'X': 32
        }

        # 索引文件路径
        self.index_file = os.path.join(cache_dir, "embedding_index.pkl")

        # 加载或创建索引
        self.embedding_index = self._load_or_create_index()

        # 进程间通信队列
        self.task_queue = None
        self.result_queue = None

        # 检查点文件
        self.checkpoint_file = os.path.join(cache_dir, "precompute_checkpoint.json")

        # 统计信息
        self.stats = {
            "total": 0,
            "computed": 0,
            "cached": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None
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

            # 检查缓存
            seq_hash = self._create_seq_hash(seq)
            embedding_file = os.path.join(self.cache_dir, f"{seq_hash}.pt")

            if os.path.exists(embedding_file) and not self.force_recompute:
                # 更新索引
                self.embedding_index[seq_id] = {
                    "hash": seq_hash,
                    "sequence": seq,
                    "file": embedding_file
                }
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
            # 加载ESMC模型 - 修复：转换为torch.device对象
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
            return

        logger.info(f"进程 {rank} 开始处理 {len(partition)} 个序列")

        # 设置进度条
        pbar = tqdm(total=len(partition), desc=f"GPU {rank}", position=rank)

        # 创建本地索引
        local_index = {}

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
                max_length = 512
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
                    self._process_batch(esm_model, batch_buffer, device, local_index, rank)
                    # 清空缓冲区
                    batch_buffer = []

                # 每处理100个序列保存一次本地索引
                if (item_idx + 1) % 100 == 0 or item_idx == len(partition) - 1:
                    # 保存本地索引
                    local_index_file = os.path.join(self.cache_dir, f"embedding_index_gpu{rank}.pkl")
                    with open(local_index_file, "wb") as f:
                        pickle.dump(local_index, f)
                    logger.info(f"进程 {rank} 已保存本地索引，包含 {len(local_index)} 条记录")

                # 更新进度条
                pbar.update(1)

            except Exception as e:
                logger.error(f"进程 {rank} 处理序列 {seq_id} 时出错: {e}")
                logger.error(traceback.format_exc())

        # 关闭进度条
        pbar.close()

        # 广播已完成的任务
        completed_ids = list(local_index.keys())
        logger.info(f"进程 {rank} 完成计算，处理了 {len(completed_ids)} 个序列")

        # 保存本地索引
        local_index_file = os.path.join(self.cache_dir, f"embedding_index_gpu{rank}.pkl")
        with open(local_index_file, "wb") as f:
            pickle.dump(local_index, f)

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

    def _process_batch(self, esm_model, batch_buffer, device, local_index, rank):
        """处理序列批次"""
        for idx, seq_id, seq, token_ids in batch_buffer:
            try:
                # 转换为张量 - 修复：不添加额外维度
                token_tensor = torch.tensor(token_ids, device=device)
                protein_tensor = ESMProteinTensor(sequence=token_tensor)

                # 创建序列哈希
                seq_hash = self._create_seq_hash(seq)
                embedding_file = os.path.join(self.cache_dir, f"{seq_hash}.pt")

                # 计算嵌入 - 使用安全计算方法
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

                        # 创建存储对象 - 转为半精度以节省空间
                        embedding_data = {
                            "embedding": embedding.cpu().half(),  # 半精度存储
                            "attention": attention.cpu().half() if attention is not None else None,
                            "sequence": seq,
                            "id": seq_id
                        }

                        # 保存嵌入
                        torch.save(embedding_data, embedding_file, _use_new_zipfile_serialization=True)

                        # 更新本地索引
                        local_index[seq_id] = {
                            "hash": seq_hash,
                            "sequence": seq,
                            "file": embedding_file
                        }

                    except Exception as inner_e:
                        logger.error(f"进程 {rank} 处理序列 {seq_id} 计算嵌入失败: {inner_e}")
                        logger.error(traceback.format_exc())

            except Exception as e:
                logger.error(f"进程 {rank} 处理序列 {seq_id} 时出错: {e}")
                logger.error(traceback.format_exc())

    def _merge_local_indices(self):
        """合并所有本地索引"""
        for rank in range(self.num_gpus):
            local_index_file = os.path.join(self.cache_dir, f"embedding_index_gpu{rank}.pkl")
            if os.path.exists(local_index_file):
                try:
                    with open(local_index_file, "rb") as f:
                        local_index = pickle.load(f)

                    # 更新全局索引
                    self.embedding_index.update(local_index)
                    logger.info(f"合并了GPU {rank} 的本地索引，添加了 {len(local_index)} 条记录")

                    # 可选：删除本地索引文件
                    os.remove(local_index_file)
                except Exception as e:
                    logger.error(f"合并GPU {rank} 的本地索引失败: {e}")

        # 保存全局索引
        self._save_index()

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

        # 合并本地索引
        self._merge_local_indices()

        # 记录结束时间
        self.stats["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 计算统计信息
        self.stats["computed"] = len(self.embedding_index) - self.stats["cached"]
        self.stats["failed"] = total_to_compute - self.stats["computed"]

        # 输出统计信息
        logger.info(f"嵌入计算完成: ")
        logger.info(f"- 总序列数: {self.stats['total']}")
        logger.info(f"- 已缓存: {self.stats['cached']}")
        logger.info(f"- 新计算: {self.stats['computed']}")
        logger.info(f"- 计算失败: {self.stats['failed']}")
        logger.info(f"- 开始时间: {self.stats['start_time']}")
        logger.info(f"- 结束时间: {self.stats['end_time']}")

        # 保存统计信息
        stats_file = os.path.join(self.cache_dir, "compute_stats.json")
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ESMC嵌入分布式预计算工具")
    parser.add_argument("--data", type=str, required=True, help="数据集路径")
    parser.add_argument("--model", type=str, default="esmc_600m", help="ESMC模型名称")
    parser.add_argument("--cache-dir", type=str, default="data/esm_embeddings", help="嵌入缓存目录")
    parser.add_argument("--num-gpus", type=int, default=3, help="使用GPU数量")
    parser.add_argument("--batch-size", type=int, default=16, help="每个GPU的批处理大小")
    parser.add_argument("--force", action="store_true", help="强制重新计算所有嵌入")
    parser.add_argument("--port", type=int, default=29500, help="分布式通信端口")
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    data_name = args.data[args.data.rfind("/") + 1:args.data.rfind(".")]
    args.cache_dir = os.path.join(args.cache_dir, data_name)
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
        cache_dir=args.cache_dir,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        force_recompute=args.force,
        master_port=args.port
    )

    # 计算嵌入
    computer.compute_embeddings(args.data)


if __name__ == "__main__":
    main()