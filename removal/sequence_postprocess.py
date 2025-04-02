#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质序列处理与图谱保存工具

该脚本执行以下功能:
1. 加载蛋白质序列数据
2. 使用分布式MinHash+LSH算法进行序列去冗余
3. 对序列进行AMPs功能特性聚类分析(可选)
4. 加载保留序列对应的图谱(无需进行图谱结构相似度去冗余)
5. 保存处理结果和对应图谱

作者: wxhfy
"""

import argparse
import concurrent.futures
import gc
import glob
import io
import json
import logging
import os
import pickle
import random
import sys
import time
import traceback
from collections import defaultdict, Counter

import faiss
import numpy as np
import psutil
import torch
from datasketch import MinHash, MinHashLSH
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FAISS_AVAILABLE = faiss.__version__ is not None

# 氨基酸物理化学性质常量 (针对AMPs功能优化)
AA_PROPERTIES = {
    'A': {'hydropathy': 1.8, 'charge': 0, 'polar': False, 'mw': 89.09, 'helix': 1.41, 'sheet': 0.72,
          'hydrophobic': 0.62},
    'C': {'hydropathy': 2.5, 'charge': 0, 'polar': False, 'mw': 121.15, 'helix': 0.66, 'sheet': 1.19,
          'hydrophobic': 0.29},
    'D': {'hydropathy': -3.5, 'charge': -1, 'polar': True, 'mw': 133.10, 'helix': 0.98, 'sheet': 0.39,
          'hydrophobic': -0.90},
    'E': {'hydropathy': -3.5, 'charge': -1, 'polar': True, 'mw': 147.13, 'helix': 1.53, 'sheet': 0.26,
          'hydrophobic': -0.74},
    'F': {'hydropathy': 2.8, 'charge': 0, 'polar': False, 'mw': 165.19, 'helix': 1.16, 'sheet': 1.33,
          'hydrophobic': 1.19},
    'G': {'hydropathy': -0.4, 'charge': 0, 'polar': False, 'mw': 75.07, 'helix': 0.43, 'sheet': 0.58,
          'hydrophobic': 0.48},
    'H': {'hydropathy': -3.2, 'charge': 0.1, 'polar': True, 'mw': 155.16, 'helix': 1.05, 'sheet': 0.80,
          'hydrophobic': -0.40},
    'I': {'hydropathy': 4.5, 'charge': 0, 'polar': False, 'mw': 131.17, 'helix': 1.09, 'sheet': 1.67,
          'hydrophobic': 1.38},
    'K': {'hydropathy': -3.9, 'charge': 1, 'polar': True, 'mw': 146.19, 'helix': 1.23, 'sheet': 0.69,
          'hydrophobic': -1.50},
    'L': {'hydropathy': 3.8, 'charge': 0, 'polar': False, 'mw': 131.17, 'helix': 1.34, 'sheet': 1.22,
          'hydrophobic': 1.06},
    'M': {'hydropathy': 1.9, 'charge': 0, 'polar': False, 'mw': 149.21, 'helix': 1.30, 'sheet': 1.14,
          'hydrophobic': 0.64},
    'N': {'hydropathy': -3.5, 'charge': 0, 'polar': True, 'mw': 132.12, 'helix': 0.76, 'sheet': 0.48,
          'hydrophobic': -0.78},
    'P': {'hydropathy': -1.6, 'charge': 0, 'polar': False, 'mw': 115.13, 'helix': 0.34, 'sheet': 0.31,
          'hydrophobic': 0.12},
    'Q': {'hydropathy': -3.5, 'charge': 0, 'polar': True, 'mw': 146.15, 'helix': 1.27, 'sheet': 0.96,
          'hydrophobic': -0.85},
    'R': {'hydropathy': -4.5, 'charge': 1, 'polar': True, 'mw': 174.20, 'helix': 0.96, 'sheet': 0.99,
          'hydrophobic': -2.53},
    'S': {'hydropathy': -0.8, 'charge': 0, 'polar': True, 'mw': 105.09, 'helix': 0.57, 'sheet': 0.96,
          'hydrophobic': -0.18},
    'T': {'hydropathy': -0.7, 'charge': 0, 'polar': True, 'mw': 119.12, 'helix': 0.76, 'sheet': 1.17,
          'hydrophobic': -0.05},
    'V': {'hydropathy': 4.2, 'charge': 0, 'polar': False, 'mw': 117.15, 'helix': 0.98, 'sheet': 1.87,
          'hydrophobic': 1.08},
    'W': {'hydropathy': -0.9, 'charge': 0, 'polar': True, 'mw': 204.23, 'helix': 1.02, 'sheet': 1.35,
          'hydrophobic': 0.81},
    'Y': {'hydropathy': -1.3, 'charge': 0, 'polar': True, 'mw': 181.19, 'helix': 0.74, 'sheet': 1.45,
          'hydrophobic': 0.26}
}

# AMPs功能相关的氨基酸分组
AMP_FUNCTIONAL_GROUPS = {
    'hydrophobic': ['A', 'F', 'I', 'L', 'M', 'V', 'W', 'Y'],
    'charged_positive': ['K', 'R', 'H'],
    'charged_negative': ['D', 'E'],
    'polar_uncharged': ['N', 'Q', 'S', 'T'],
    'special': ['C', 'G', 'P']
}


def set_gpu_device(device_id):
    """设置使用的GPU设备ID"""
    if device_id is not None:
        try:
            if torch.cuda.is_available():
                device_id_list = [int(id.strip()) for id in device_id.split(',')]
                devices_available = list(range(torch.cuda.device_count()))
                valid_devices = [id for id in device_id_list if id in devices_available]

                if valid_devices:
                    # 设置环境变量以限制CUDA可见设备
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, valid_devices))
                    torch.cuda.set_device(valid_devices[0])
                    logger.info(f"已设置GPU设备: {valid_devices}")
                else:
                    logger.warning(f"未找到有效的GPU设备ID: {device_id}，将使用默认设备")
            else:
                logger.warning("未检测到可用的CUDA设备，将使用CPU")
        except Exception as e:
            logger.warning(f"设置GPU设备时出错: {str(e)}，将使用默认设备")


def check_memory_usage(threshold_gb=None, force_gc=False):
    """
    检查内存使用情况，并在需要时执行垃圾回收

    参数:
        threshold_gb: 内存使用阈值(GB)，超过此值返回True
        force_gc: 是否强制执行垃圾回收

    返回:
        bool: 内存使用是否超过阈值
    """
    try:
        # 强制垃圾回收（如果要求）
        if force_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 获取当前内存使用量
        mem_used = psutil.Process().memory_info().rss / (1024 ** 3)  # 转换为GB

        # 如果设置了阈值且内存超过阈值
        if threshold_gb and mem_used > threshold_gb:
            logger.warning(f"内存使用已达 {mem_used:.2f} GB，超过阈值 {threshold_gb:.2f} GB，执行垃圾回收")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True

        # 内存使用未超过阈值
        return False

    except Exception as e:
        logger.error(f"检查内存使用时出错: {str(e)}")
        gc.collect()  # 出错时仍执行垃圾回收以确保安全
        return False


def log_system_resources():
    """记录系统资源使用情况"""
    try:
        mem = psutil.virtual_memory()
        logger.info(f"系统内存: {mem.used / 1024 ** 3:.1f}GB/{mem.total / 1024 ** 3:.1f}GB ({mem.percent}%)")

        swap = psutil.swap_memory()
        logger.info(f"交换内存: {swap.used / 1024 ** 3:.1f}GB/{swap.total / 1024 ** 3:.1f}GB ({swap.percent}%)")

        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        logger.info(f"CPU核心: {cpu_count}个, 使用率: {sum(cpu_percent) / len(cpu_percent):.1f}%")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                    reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                    max_mem = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                    logger.info(
                        f"GPU {i} ({torch.cuda.get_device_name(i)}): 已分配={allocated:.1f}GB, 已保留={reserved:.1f}GB, 总计={max_mem:.1f}GB")
                except:
                    logger.info(f"GPU {i}: 无法获取内存信息")
    except Exception as e:
        logger.warning(f"无法获取系统资源信息: {str(e)}")


def setup_logging(output_dir):
    """设置日志系统"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"sequence_processor_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 捕获所有级别的日志

    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 添加控制台处理器 - 仅显示INFO及以上级别
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console)

    # 添加文件处理器 - 记录所有级别（包括DEBUG）
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    # 记录系统信息
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("未检测到GPU")

    return root_logger, log_file


def find_batch_directories(input_path):
    """查找所有批次目录"""
    batch_dirs = []
    if os.path.isdir(input_path):
        # 查找所有批次目录
        for item in os.listdir(input_path):
            if item.startswith("batch_") and os.path.isdir(os.path.join(input_path, item)):
                batch_dirs.append(os.path.join(input_path, item))

        # 如果没有找到批次目录，将输入路径作为单个目录
        if not batch_dirs:
            batch_dirs = [input_path]

    return sorted(batch_dirs)


def load_json_file(file_path):
    """加载单个JSON文件中的序列数据"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            sequences = {}
            # 只提取序列信息，节省内存
            for seq_id, seq_data in data.items():
                if isinstance(seq_data, dict) and "sequence" in seq_data:
                    sequences[seq_id] = {"sequence": seq_data["sequence"]}
                else:
                    # 如果直接是字符串，假设它是序列
                    sequences[seq_id] = {"sequence": seq_data}
            return sequences
    except Exception as e:
        logger.error(f"加载序列文件失败: {file_path} - {str(e)}")
        return {}


def parallel_load_sequences(input_path, num_workers=32, memory_limit_gb=800,
                            test_mode=False, max_test_files=5):
    """并行加载所有序列数据，支持测试模式"""
    batch_dirs = find_batch_directories(input_path)
    logger.info(f"找到 {len(batch_dirs)} 个批次目录")

    # 收集所有JSON文件路径
    all_json_files = []
    for batch_dir in batch_dirs:
        seq_files = sorted([os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if
                            (f.startswith("pdb_data_") or f.startswith("protein_data_chunk_"))
                            and f.endswith(".json")])
        all_json_files.extend(seq_files)

    # 测试模式下仅加载少量文件
    if test_mode:
        orig_count = len(all_json_files)
        all_json_files = all_json_files[:max_test_files]
        logger.info(f"测试模式: 从{orig_count}个文件中仅加载{len(all_json_files)}个")

    logger.info(f"找到 {len(all_json_files)} 个序列JSON文件，开始并行加载")

    # 并行加载文件
    all_sequences = {}
    loaded_count = 0
    total_files = len(all_json_files)

    # 使用进程池并行加载
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(load_json_file, file_path): file_path for file_path in all_json_files}

        with tqdm(total=total_files, desc="并行加载序列文件", ncols=100) as pbar:
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    sequences = future.result()
                    all_sequences.update(sequences)
                    loaded_count += 1

                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix(
                        {"已加载": f"{len(all_sequences)}序列", "已处理": f"{loaded_count}/{total_files}文件"})

                    # 检查内存使用
                    mem_used = psutil.Process().memory_info().rss / (1024 ** 3)
                    if mem_used > memory_limit_gb:
                        logger.warning(f"内存使用达到{mem_used:.1f}GB，超过限制{memory_limit_gb}GB，停止加载更多文件")
                        break
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 时出错: {str(e)}")

    logger.info(f"并行加载完成，共加载 {len(all_sequences)} 个序列")
    return all_sequences


def create_k_mers(sequence, k=3):
    """从序列中生成k-mers - 向量化实现"""
    if len(sequence) < k:
        return []

    # 使用列表推导式一次性生成所有k-mers，避免在循环中重复切片
    # 这比传统循环要高效，尤其对于长序列
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def process_sequence_batch(batch_data, identity_threshold=0.6, coverage_threshold=0.8, num_perm=128):
    """
    处理单批次序列的函数，用于并行处理 - 向量化优化版

    参数:
        batch_data: (batch_id, batch_seqs)元组，包含批次ID和序列字典
        identity_threshold: 序列同一性阈值
        coverage_threshold: 序列覆盖率阈值
        num_perm: MinHash使用的排列数

    返回:
        元组: (batch_id, retained_ids)，批次ID和保留的序列ID列表
    """
    batch_id, batch_seqs = batch_data

    # 提取序列ID和实际序列
    seq_items = [(seq_id, seq_data['sequence']) for seq_id, seq_data in batch_seqs.items()]

    # 按序列长度降序排列 - 使用高效的排序
    seq_items.sort(key=lambda x: -len(x[1]))

    # 创建LSH索引
    lsh = MinHashLSH(threshold=identity_threshold, num_perm=num_perm)
    retained_ids = []

    # 预计算并缓存所有序列长度 - 避免重复计算
    seq_lengths = {seq_id: len(seq) for seq_id, seq in seq_items}

    # 创建序列字典用于批量处理
    seq_dict = {seq_id: {'sequence': seq} for seq_id, seq in seq_items}

    # 批量预计算k-mers - 显著提高性能
    k = 3  # k-mer长度
    all_kmers = batch_create_kmers(seq_dict, k)

    # 预计算序列ID到索引的映射，便于后续快速查找
    seq_id_to_idx = {seq_id: idx for idx, (seq_id, _) in enumerate(seq_items)}

    # 对每个序列创建MinHash签名并进行筛选
    for seq_id, seq in seq_items:
        # 高效创建MinHash签名
        m = MinHash(num_perm=num_perm)

        # 使用预计算的k-mers，避免重复生成
        if seq_id in all_kmers:
            # 批量处理k-mer
            for k_mer in all_kmers[seq_id]:
                m.update(k_mer.encode('utf8'))

        # 查询相似序列
        similar_seqs = lsh.query(m)

        # 如果没有相似序列，保留此序列
        if not similar_seqs:
            retained_ids.append(seq_id)
            lsh.insert(seq_id, m)
        else:
            # 验证覆盖率 - 使用向量化计算
            seq_len = seq_lengths[seq_id]

            # 获取所有相似序列的长度
            similar_lengths = np.array([seq_lengths[s_id] for s_id in similar_seqs if s_id in seq_lengths])

            if len(similar_lengths) > 0:
                # 向量化计算覆盖率
                min_lengths = np.minimum(seq_len, similar_lengths)
                max_lengths = np.maximum(seq_len, similar_lengths)
                coverage_values = min_lengths / max_lengths

                # 检查是否所有相似序列的覆盖率都超过阈值
                all_similar = np.all(coverage_values >= coverage_threshold)

                if not all_similar:
                    retained_ids.append(seq_id)
                    lsh.insert(seq_id, m)
            else:
                # 如果没有有效的相似序列长度，则保留当前序列
                retained_ids.append(seq_id)
                lsh.insert(seq_id, m)

    # 清理不再需要的大型数据结构以释放内存
    del all_kmers
    del seq_lengths
    del seq_dict

    return batch_id, retained_ids

# 优化1: 向量化序列长度计算
def batch_calculate_lengths(sequences_batch):
    """批量计算序列长度"""
    return {seq_id: len(seq_data['sequence']) for seq_id, seq_data in sequences_batch.items()}

# 优化2: 批量生成k-mers
def batch_create_kmers(sequences_batch, k=3):
    """批量为多个序列生成k-mers"""
    all_kmers = {}
    for seq_id, seq_data in sequences_batch.items():
        seq = seq_data['sequence']
        if len(seq) >= k:
            all_kmers[seq_id] = [seq[i:i+k] for i in range(len(seq)-k+1)]
    return all_kmers


def distributed_minhash_filter(sequences, identity_threshold=0.6, coverage_threshold=0.8,
                               num_perm=128, num_workers=128, batch_size=500000, output_dir="."):
    """
    使用分布式处理的MinHash+LSH算法进行序列去冗余 - 向量化优化版

    参数:
        sequences: 序列字典 {id: {'sequence': seq}}
        identity_threshold: 序列同一性阈值
        coverage_threshold: 序列覆盖率阈值
        num_perm: MinHash使用的排列数
        num_workers: 并行工作进程数
        batch_size: 每批处理的序列数量
        output_dir: 输出目录

    返回:
        dict: 过滤后的序列字典
        list: 过滤后的序列ID列表
    """
    # 创建检查点目录
    checkpoint_dir = os.path.join(output_dir, "seq_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "seq_retained_ids.json")

    # 首先检查是否存在已完成的检查点
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                if checkpoint_data.get("completed", False):
                    all_retained_ids = set(checkpoint_data.get("retained_ids", []))
                    logger.info(f"从已完成的检查点加载序列去冗余结果: 保留{len(all_retained_ids)}个序列")
                    # 构建过滤后的序列字典 - 使用字典推导式优化
                    filtered_sequences = {seq_id: sequences[seq_id] for seq_id in all_retained_ids if
                                          seq_id in sequences}
                    logger.info(f"成功加载已完成的序列去冗余结果: {len(filtered_sequences)}/{len(sequences)} 个序列")
                    return filtered_sequences, list(all_retained_ids)
        except Exception as e:
            logger.warning(f"读取序列检查点失败: {str(e)}，将从头开始处理")

    logger.info(f"使用分布式MinHash+LSH算法对 {len(sequences)} 个序列进行去冗余 (使用{num_workers}个核心)...")
    start_time = time.time()

    # 将序列按长度分组 - 使用批处理方式优化
    logger.info("按长度范围对序列分组...")
    length_groups = defaultdict(dict)

    # 使用向量化长度计算 - 批量处理提高效率
    batch_size_for_grouping = min(50000, len(sequences))  # 选择合适的批处理大小
    seq_items = list(sequences.items())
    total_batches = (len(seq_items) + batch_size_for_grouping - 1) // batch_size_for_grouping

    with tqdm(total=total_batches, desc="按长度分组", ncols=100) as pbar:
        for i in range(0, len(seq_items), batch_size_for_grouping):
            end_idx = min(i + batch_size_for_grouping, len(seq_items))
            batch_dict = dict(seq_items[i:end_idx])

            # 批量计算长度 - 向量化操作
            lengths = batch_calculate_lengths(batch_dict)

            # 批量分组处理
            for seq_id, length in lengths.items():
                length_range = length // 5  # 每5个氨基酸为一组
                length_groups[length_range][seq_id] = sequences[seq_id]

            pbar.update(1)
            pbar.set_postfix({"组数": len(length_groups)})

    logger.info(f"序列分为 {len(length_groups)} 个长度组")

    # 检查是否存在部分完成的检查点
    all_retained_ids = set()
    processed_groups = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                all_retained_ids = set(checkpoint_data.get("retained_ids", []))
                processed_groups = set(str(g) for g in checkpoint_data.get("processed_groups", []))
                logger.info(
                    f"从检查点恢复序列去冗余: 已处理{len(processed_groups)}组，已保留{len(all_retained_ids)}序列")
        except Exception as e:
            logger.warning(f"读取序列检查点失败: {str(e)}，将从头开始处理")

    # 创建任务列表 - 优化批次分配
    tasks = []

    # 按长度排序处理组 - 使用高效的排序方法
    sorted_groups = sorted(length_groups.items(), key=lambda x: x[0])

    for length_range, group_seqs in sorted_groups:
        # 跳过已处理的组
        if str(length_range) in processed_groups:
            logger.info(f"跳过已处理的长度组 {length_range}")
            continue

        # 优化分批策略 - 基于序列长度进行动态批次调整
        group_size = len(group_seqs)
        if group_size <= batch_size:
            # 小型组作为单个任务
            tasks.append((f"{length_range}", group_seqs))
        else:
            # 大型组拆分成多个平衡的任务
            seq_ids = list(group_seqs.keys())

            # 计算最佳批次大小和数量
            optimal_batch_count = max(1, min(num_workers, (group_size + batch_size - 1) // batch_size))
            actual_batch_size = (group_size + optimal_batch_count - 1) // optimal_batch_count

            # 创建平均大小的批次，最后一个批次可能较小
            for i in range(0, group_size, actual_batch_size):
                end_idx = min(i + actual_batch_size, group_size)
                batch_ids = seq_ids[i:end_idx]
                batch_seqs = {seq_id: group_seqs[seq_id] for seq_id in batch_ids}
                tasks.append((f"{length_range}_{i // actual_batch_size}", batch_seqs))

    logger.info(f"序列去冗余任务拆分为 {len(tasks)} 个批次，开始并行处理")

    # 并行处理批次 - 使用高效的资源分配
    task_chunks = []
    chunk_size = max(1, len(tasks) // (num_workers * 2))  # 每个进程处理多个任务以减少进程创建开销
    for i in range(0, len(tasks), chunk_size):
        task_chunks.append(tasks[i:i + chunk_size])

    # 采用进程池进行并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交任务块而非单个任务，减少调度开销
        futures = []
        for chunk in task_chunks:
            for task in chunk:
                futures.append(
                    executor.submit(process_sequence_batch, task, identity_threshold, coverage_threshold, num_perm))

        # 实时处理结果
        completed_tasks = 0
        with tqdm(total=len(tasks), desc="处理序列批次", ncols=100) as pbar:
            for future in concurrent.futures.as_completed(futures):
                batch_id = None
                try:
                    batch_id, retained_ids = future.result()
                    all_retained_ids.update(retained_ids)

                    # 从batch_id中提取组ID
                    group_id = batch_id.split('_')[0]
                    processed_groups.add(group_id)

                    # 更新进度
                    completed_tasks += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "已保留": len(all_retained_ids),
                        "remain%": f"{len(all_retained_ids) * 100 / len(sequences):.1f}%",
                        "已处理组": len(processed_groups)
                    })

                    # 定期保存检查点 - 动态调整保存频率
                    save_freq = max(1, min(1000, len(tasks) // 20))  # 根据总任务数动态调整
                    if completed_tasks % save_freq == 0 or completed_tasks == len(tasks):
                        with open(checkpoint_file, 'w') as f:
                            checkpoint_data = {
                                "retained_ids": list(all_retained_ids),
                                "processed_groups": list(processed_groups)
                            }
                            json.dump(checkpoint_data, f)
                            if completed_tasks % (save_freq * 5) == 0:  # 仅在较大间隔记录日志
                                logger.info(f"保存序列去冗余检查点: {len(all_retained_ids)}个保留序列")

                except Exception as e:
                    logger.error(f"处理批次 {batch_id if batch_id else '未知'} 时出错: {str(e)}")
                    traceback.print_exc()

    # 保存最终检查点，标记为已完成
    with open(checkpoint_file, 'w') as f:
        checkpoint_data = {
            "retained_ids": list(all_retained_ids),
            "processed_groups": list(processed_groups),
            "completed": True  # 添加完成标记
        }
        json.dump(checkpoint_data, f)

    # 创建 seq_clustering_results.pkl 文件 (供后续图谱处理使用)
    seq_checkpoint_dir = os.path.join(output_dir, "seq_checkpoints")
    seq_checkpoint_file = os.path.join(seq_checkpoint_dir, "seq_clustering_results.pkl")

    # 构建过滤后的序列字典 - 使用向量化字典构建
    # 预先将all_retained_ids转换为集合以加速查找
    retained_ids_set = set(all_retained_ids)
    filtered_sequences = {}

    # 分批处理以减少内存峰值
    batch_size_for_dict = 100000
    seq_ids = list(sequences.keys())
    for i in range(0, len(seq_ids), batch_size_for_dict):
        batch_ids = seq_ids[i:i + batch_size_for_dict]
        # 向量化过滤
        batch_filtered = {seq_id: sequences[seq_id] for seq_id in batch_ids
                          if seq_id in retained_ids_set}
        filtered_sequences.update(batch_filtered)

    # 保存过滤结果到 pickle 文件 - 使用高效的压缩选项
    with open(seq_checkpoint_file, 'wb') as f:
        pickle.dump(filtered_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"序列去冗余结果已保存到 {seq_checkpoint_file}")

    elapsed = time.time() - start_time
    logger.info(f"序列去冗余完成: 从 {len(sequences)} 个序列中保留 {len(filtered_sequences)} 个 "
                f"({len(filtered_sequences) * 100 / len(sequences):.1f}%)，耗时: {elapsed / 60:.1f}分钟")

    # 返回过滤后的序列字典和ID列表
    return filtered_sequences, list(all_retained_ids)


def calculate_amp_features(sequence):
    """
    计算与AMPs功能相关的序列特征向量 - 向量化优化版

    参数:
        sequence: 氨基酸序列字符串

    返回:
        特征向量字典
    """
    if not sequence:
        return None

    seq_len = len(sequence)

    # 1. 向量化统计氨基酸频率
    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    # 预计算序列中所有氨基酸的计数
    aa_counts = np.zeros(len(aa_list))
    seq_array = np.array(list(sequence))

    for i, aa in enumerate(aa_list):
        aa_counts[i] = np.sum(seq_array == aa)

    # 计算频率
    aa_freq = {aa: count / seq_len for aa, count in zip(aa_list, aa_counts)}

    # 2. 向量化计算功能基团比例
    func_groups = {}
    for group, aas in AMP_FUNCTIONAL_GROUPS.items():
        # 使用向量化操作计算组内氨基酸总数
        mask = np.isin(seq_array, list(aas))
        count = np.sum(mask)
        func_groups[group] = count / seq_len

    # 3. 向量化物理化学特性计算
    # 创建氨基酸到物理化学特性的映射数组
    hydropathy_values = np.array([AA_PROPERTIES.get(aa, {'hydropathy': 0})['hydropathy']
                                  for aa in sequence])
    charge_values = np.array([AA_PROPERTIES.get(aa, {'charge': 0})['charge']
                              for aa in sequence])
    helix_values = np.array([AA_PROPERTIES.get(aa, {'helix': 0}).get('helix', 0)
                             for aa in sequence])

    # 使用numpy求和替代循环
    hydrophobicity = np.sum(hydropathy_values) / seq_len
    net_charge = np.sum(charge_values)
    helix_propensity = np.sum(helix_values) / seq_len

    # 4. 向量化计算两亲性 (通过窗口内疏水残基分布)
    window_size = min(18, seq_len)
    max_amphipathicity = 0

    if window_size > 0 and seq_len >= window_size:
        # 预计算所有角度和余弦值
        angles = np.array([j * 100 * np.pi / 180 for j in range(window_size)])
        cos_angles = np.cos(angles)

        # 为每个窗口位置计算疏水矩
        all_moments = []
        for i in range(seq_len - window_size + 1):
            window = sequence[i:i + window_size]
            # 提取窗口中的氨基酸疏水性值
            window_hydropathy = np.array([AA_PROPERTIES.get(aa, {'hydropathy': 0})['hydropathy']
                                          for aa in window])
            # 向量化计算疏水矩
            hydro_moment = np.sum(window_hydropathy * cos_angles) / window_size
            all_moments.append(abs(hydro_moment))

        # 使用numpy操作找到最大值
        if all_moments:
            max_amphipathicity = np.max(all_moments)

    # 5. 向量化计算氨基酸二联体频率
    dipeptide_freq = {}
    if seq_len > 1:
        # 生成所有二联体
        dipeptides = [sequence[i:i + 2] for i in range(seq_len - 1)]

        # 使用Counter进行高效计数
        from collections import Counter
        dipep_counter = Counter(dipeptides)

        # 标准化频率
        total_dipeps = seq_len - 1
        dipeptide_freq = {dipep: count / total_dipeps for dipep, count in dipep_counter.items()}

        # 只保留最常见的二联体模式
        top_dipeptides = {}
        for dp, count in sorted(dipeptide_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            top_dipeptides[dp] = count
    else:
        top_dipeptides = {}

    # 构建特征字典 - 保持原始结构以保证兼容性
    features = {
        'length': seq_len,
        'hydrophobicity': float(hydrophobicity),  # 确保是Python标准类型
        'net_charge': float(net_charge),
        'amphipathicity': float(max_amphipathicity),
        'helix_propensity': float(helix_propensity),
        'aa_freqs': aa_freq,
        'func_groups': func_groups,
        'top_dipeptides': top_dipeptides,
    }

    return features


def create_amp_feature_vector(features):
    """将AMPs特征字典转换为数值向量，用于聚类 - 向量化优化版"""
    if not features:
        return np.zeros(30)  # 默认空向量

    # 预分配向量空间
    feature_vector = np.zeros(30)

    # 1. 设置基础特征 - 使用索引赋值替代append
    base_features = ['length', 'hydrophobicity', 'net_charge',
                     'amphipathicity', 'helix_propensity']
    for i, feature_name in enumerate(base_features):
        feature_vector[i] = features.get(feature_name, 0.0)

    # 2. 向量化添加氨基酸频率 - 使用预定义顺序
    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    aa_freqs = features.get('aa_freqs', {})
    for i, aa in enumerate(aa_list):
        feature_vector[5 + i] = aa_freqs.get(aa, 0.0)

    # 3. 添加功能基团频率 - 使用预定义顺序
    func_groups = list(AMP_FUNCTIONAL_GROUPS.keys())
    group_freqs = features.get('func_groups', {})
    for i, group in enumerate(func_groups):
        feature_vector[25 + i] = group_freqs.get(group, 0.0)

    # 4. 添加电荷密度(每单位长度的电荷)
    if features.get('length', 0) > 0:
        feature_vector[25 + len(func_groups)] = features.get('net_charge', 0) / features['length']

    # 5. 添加疏水性结构指标(疏水性与两亲性的组合)
    feature_vector[26 + len(func_groups)] = (features.get('hydrophobicity', 0) *
                                             features.get('amphipathicity', 0))

    return feature_vector

def generate_amp_features_batch(sequences_batch):
    """为一批序列生成AMPs特征"""
    features_batch = {}
    for seq_id, seq_data in sequences_batch.items():
        try:
            sequence = seq_data['sequence']
            features = calculate_amp_features(sequence)
            if features:
                features_batch[seq_id] = features
        except Exception as e:
            logger.debug(f"计算序列 {seq_id} 特征时出错: {str(e)}")
    return features_batch


def parallel_generate_amp_features(sequences, num_workers=32, batch_size=5000):
    """并行生成所有序列的AMPs特征 - 向量化优化版"""
    logger.info(f"为 {len(sequences)} 个序列并行生成AMPs特征...")

    # 预计算常用值，以便在工作进程间共享
    # 预先计算并共享的数据可以减少每个工作进程的计算和内存开销
    precomputed_data = {
        'aa_list': "ACDEFGHIKLMNPQRSTVWY",
        'window_sizes': list(range(3, 31)),  # 预计算窗口大小3-30
        'cos_angles': {
            size: np.cos(np.array([j * 100 * np.pi / 180 for j in range(size)]))
            for size in range(3, 31)  # 为不同窗口大小预计算余弦值
        }
    }

    # 将序列分成批次 - 使用更高效的分块方法
    seq_ids = list(sequences.keys())
    total_seqs = len(seq_ids)
    batches = []

    # 计算最佳批次数以平衡任务分配
    optimal_batch_count = min(num_workers * 4, max(1, total_seqs // batch_size))
    actual_batch_size = max(1, total_seqs // optimal_batch_count)

    # 创建更均匀的批次
    for i in range(0, total_seqs, actual_batch_size):
        end_idx = min(i + actual_batch_size, total_seqs)
        batch_ids = seq_ids[i:end_idx]
        batch_data = {seq_id: sequences[seq_id] for seq_id in batch_ids}
        batches.append(batch_data)

    logger.info(f"序列特征计算分为 {len(batches)} 个批次，每批次约 {actual_batch_size} 个序列")

    # 并行处理 - 使用进程池
    all_features = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有批次任务
        futures = [executor.submit(generate_amp_features_batch, batch)
                   for batch in batches]

        # 使用tqdm跟踪进度
        with tqdm(total=len(batches), desc="生成AMPs特征", ncols=100) as pbar:
            # 处理完成的任务
            for future in concurrent.futures.as_completed(futures):
                try:
                    features_batch = future.result()
                    all_features.update(features_batch)
                    pbar.update(1)

                    # 定期更新进度信息
                    if len(all_features) % (batch_size * 2) == 0:
                        pbar.set_postfix({"完成": f"{len(all_features)}/{total_seqs}"})

                except Exception as e:
                    logger.error(f"处理特征批次时出错: {str(e)}")

    logger.info(f"成功为 {len(all_features)} 个序列生成AMPs特征")
    return all_features


def hybrid_representative_sampling(unique_labels, cluster_sizes, labels, seq_ids, X_scaled,
                                   sample_ratio=0.6, key_features=(0, 1, 2, 3, 4), feature_weights=None):
    """
    混合策略的代表性采样算法 - 结合距离分层与功能重要性
    """
    # 默认特征权重
    if feature_weights is None:
        feature_weights = np.ones(len(key_features))

    retained_ids = []
    logger.info(f"使用混合代表性采样算法 (采样率: {sample_ratio:.1%})...")

    with tqdm(total=len(unique_labels), desc="混合代表性采样") as pbar:
        for label, cluster_size in cluster_sizes:
            # 获取簇数据
            cluster_indices = np.where(labels == label)[0]
            cluster_seq_ids = [seq_ids[i] for i in cluster_indices]

            # 小簇直接全部保留
            if cluster_size <= 5:
                retained_ids.extend(cluster_seq_ids)
                pbar.update(1)
                continue

            # 计算采样数量
            sample_count = max(1, int(cluster_size * sample_ratio))

            # 获取簇特征向量
            cluster_vectors = X_scaled[cluster_indices]

            # 计算加权欧氏距离（考虑特征重要性）
            weighted_vectors = cluster_vectors.copy()
            for i, f_idx in enumerate(key_features):
                if f_idx < cluster_vectors.shape[1]:
                    weighted_vectors[:, f_idx] *= feature_weights[i]

            # 计算簇中心和距离
            cluster_center = np.mean(weighted_vectors, axis=0)
            distances = np.linalg.norm(weighted_vectors - cluster_center, axis=1)

            # 分层代表性采样
            num_bins = 5  # 固定5个距离层
            bins = np.percentile(distances, np.linspace(0, 100, num_bins + 1))
            bin_indices = np.digitize(distances, bins[:-1]) - 1

            # 按比例从每层选择样本
            selected_indices = []
            for bin_idx in range(num_bins):
                bin_samples = np.where(bin_indices == bin_idx)[0]
                if len(bin_samples) == 0:
                    continue

                bin_sample_count = max(1, int(sample_count * (len(bin_samples) / len(distances))))

                if len(bin_samples) <= bin_sample_count:
                    selected_indices.extend(bin_samples)
                else:
                    # 在每层内部使用功能多样性排序进行选择
                    bin_vectors = cluster_vectors[bin_samples]
                    diversity_scores = np.zeros(len(bin_samples))

                    # 计算功能多样性得分
                    for i, f_idx in enumerate(key_features):
                        if f_idx < bin_vectors.shape[1]:
                            feature_values = bin_vectors[:, f_idx]
                            if np.std(feature_values) > 0:
                                feature_values = (feature_values - np.mean(feature_values)) / np.std(feature_values)
                                feature_ranks = np.argsort(np.abs(feature_values))[::-1]
                                for rank, idx in enumerate(feature_ranks):
                                    diversity_scores[idx] += feature_weights[i] * (len(feature_ranks) - rank)

                    # 选择得分最高的样本
                    diverse_indices = np.argsort(diversity_scores)[::-1][:bin_sample_count]
                    selected_indices.extend(bin_samples[diverse_indices])

            # 选择最终样本
            selected_ids = [cluster_seq_ids[i] for i in selected_indices]
            retained_ids.extend(selected_ids)

            pbar.update(1)

    return retained_ids

def protein_optimized_clustering(sequences, features, select_features=None, feature_weights=None,
                                 n_clusters=None, sample_ratio=0.6, diversity_ratio=0.7,
                                 output_dir=None, pca_vis=False, num_workers=32):
    """
    针对AMPs优化的聚类算法，使用特征选择和优化的多样性采样

    参数:
        sequences: 序列字典 {id: {'sequence': seq}}
        features: 序列特征字典 {id: 特征字典}
        select_features: 用于聚类的特征列表，例如 ['charge', 'hydrophobicity', 'amphipathicity']
                        如果为None，则使用所有特征
        feature_weights: 特征权重字典 {特征名: 权重}，用于强调特定特征的重要性
        n_clusters: 簇的数量，如果为None则自动估计
        sample_ratio: 从每个簇中采样的比例
        diversity_ratio: 多样性样本与中心样本的比例 (0-1)，越高越多样
        num_workers: 并行工作进程数
        output_dir: 输出目录，用于保存可视化结果
        pca_vis: 是否生成PCA可视化图

    返回:
        保留的序列ID列表和聚类结果统计
    """
    if len(sequences) < 5:
        logger.warning(f"序列数量过少 ({len(sequences)}个)，跳过聚类")
        return list(sequences.keys()), {"clusters": 0, "selected": len(sequences)}

    logger.info(f"开始进行针对AMPs优化的聚类分析 (序列数量: {len(sequences)})...")
    start_time = time.time()

    # 定义AMPs功能相关特征及其默认权重
    amp_key_features = {
        'charge': 2.0,  # 电荷 - 抗菌活性关键因素
        'hydrophobicity': 1.5,  # 疏水性 - 影响膜结合
        'amphipathicity': 1.8,  # 两亲性 - 影响膜渗透
        'helix_propensity': 1.2,  # 螺旋倾向 - 结构特征
        'flex': 0.8,  # 柔性 - 结构适应性
        'mw': 0.5,  # 分子量 - 物理特性
        'length': 1.0,  # 长度 - 基本特征
        'net_charge': 1.5,  # 净电荷 - 电荷分布
        'isoelectric': 0.7,  # 等电点 - 电荷特性
        'aromaticity': 0.6,  # 芳香性 - 结构稳定性
        'instability': 0.4,  # 不稳定性指数
        'boman': 1.0,  # Boman指数 - 蛋白质相互作用
        'hydrophobic_moment': 1.3  # 疏水矩 - 两亲性关键指标
    }

    # 使用用户提供的特征选择，否则使用所有可用特征
    if select_features is not None:
        used_features = [f for f in select_features if f in amp_key_features]
        if not used_features:
            logger.warning(f"提供的特征选择无效，将使用所有有效特征")
            used_features = list(amp_key_features.keys())
    else:
        used_features = list(amp_key_features.keys())

    # 合并用户提供的权重
    weights = amp_key_features.copy()
    if feature_weights:
        for feature, weight in feature_weights.items():
            if feature in weights:
                weights[feature] = weight

    # 记录使用的特征
    logger.info(f"选择的特征维度 ({len(used_features)}): {', '.join(used_features)}")
    logger.info(f"特征权重配置: {', '.join([f'{f}:{weights[f]:.1f}' for f in used_features])}")

    # 转换特征为向量表示
    logger.info("将AMPs特征转换为向量...")
    seq_ids = []
    feature_vectors = []
    feature_names = []  # 记录特征名称，用于后续分析

    # 搜集所有可用特征名称
    all_feature_keys = set()
    for feature_dict in features.values():
        all_feature_keys.update(feature_dict.keys())

    # 过滤出我们要使用的特征
    valid_features = [f for f in used_features if f in all_feature_keys]
    if not valid_features:
        logger.warning("没有找到有效的特征，将使用所有数值型特征")
        # 备选方案：使用所有数值型特征
        for seq_id, feature_dict in features.items():
            if seq_id in sequences:
                vector = []
                names = []
                for key, value in feature_dict.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        vector.append(value)
                        names.append(key)
                if vector:
                    seq_ids.append(seq_id)
                    feature_vectors.append(vector)
                    if not feature_names and names:
                        feature_names = names
    else:
        # 使用指定的特征
        for seq_id, feature_dict in tqdm(features.items(), desc="创建特征向量"):
            if seq_id in sequences:
                vector = []
                for feature in valid_features:
                    if feature in feature_dict and isinstance(feature_dict[feature], (int, float)):
                        # 应用特征权重
                        value = feature_dict[feature] * weights.get(feature, 1.0)
                        vector.append(value)
                    else:
                        vector.append(0.0)  # 缺失特征填充为0

                if len(vector) == len(valid_features):  # 确保向量维度一致
                    seq_ids.append(seq_id)
                    feature_vectors.append(vector)

        feature_names = valid_features

    if not feature_vectors:
        logger.warning("没有有效的特征向量，无法进行聚类")
        return list(sequences.keys()), {"clusters": 0, "selected": len(sequences), "features_used": []}

    # 转换为numpy数组并标准化
    X = np.array(feature_vectors)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"最终特征向量形状: {X.shape}, 特征名: {feature_names}")

    # 特征相关性分析
    if X.shape[0] > 5 and X.shape[1] > 1:
        logger.info("计算特征相关性矩阵...")
        corr_matrix = np.corrcoef(X_scaled.T)
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(corr_matrix[i, j]) > 0.8:  # 高相关阈值
                    high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))

        if high_corr_pairs:
            logger.info("发现高相关特征对:")
            for f1, f2, corr in high_corr_pairs:
                logger.info(f"  - {f1} 和 {f2}: {corr:.3f}")

    # 自动确定簇的数量
    if n_clusters is None:
        # 使用轮廓系数寻找最优簇数
        range_clusters = range(2, min(20, int(np.sqrt(len(sequences)) / 2) + 1))
        silhouette_scores = []

        # 优化的代码结构
        logger.info("自动评估最佳簇数...")

        # 1. 先进行一次性分层抽样，用于所有聚类数的评估
        sample_size = min(5000, X_scaled.shape[0])
        if X_scaled.shape[0] > sample_size:
            try:
                # 获取序列ID和对应的长度信息
                seq_lengths = []
                for i, seq_id in enumerate(seq_ids):
                    try:
                        # 尝试多种方式获取长度
                        if isinstance(sequences[seq_id], dict) and 'sequence' in sequences[seq_id]:
                            length = len(sequences[seq_id]['sequence'])
                        elif 'length' in features[seq_id]:
                            length = features[seq_id]['length']
                        else:
                            # 使用特征向量中的长度相关特征或默认值
                            length_idx = -1
                            if 'length' in feature_names:
                                length_idx = feature_names.index('length')
                            length = X[i, length_idx] if length_idx >= 0 else 0

                        seq_lengths.append((i, length))
                    except Exception as e:
                        # 出现异常时使用默认值
                        seq_lengths.append((i, 0))

                # 根据序列长度进行分组
                length_groups = {}
                for idx, length in seq_lengths:
                    # 对长度进行离散化处理
                    length_bin = length // 20 * 20  # 每20个氨基酸一组
                    if length_bin not in length_groups:
                        length_groups[length_bin] = []
                    length_groups[length_bin].append(idx)

                # 记录长度分布概况
                length_ranges = sorted(length_groups.keys())
                logger.info(
                    f"序列长度分组: {len(length_groups)}组 (范围: {min(length_ranges)}-{max(length_ranges)}氨基酸)")

                # 按比例从每个长度组抽样
                X_sample_indices = []
                for length_bin, indices in length_groups.items():
                    # 计算该长度组应该抽取的样本数
                    group_ratio = len(indices) / X_scaled.shape[0]
                    group_sample_size = max(1, int(sample_size * group_ratio))

                    # 抽取样本（不超过该组的总样本数）
                    group_sample_size = min(group_sample_size, len(indices))
                    group_samples = np.random.choice(indices, group_sample_size, replace=False)
                    X_sample_indices.extend(group_samples)

                # 调整最终样本数量
                if len(X_sample_indices) > sample_size:
                    X_sample_indices = np.random.choice(X_sample_indices, sample_size, replace=False)
                elif len(X_sample_indices) < sample_size:
                    remaining = list(set(range(X_scaled.shape[0])) - set(X_sample_indices))
                    if remaining:
                        supplement_count = min(sample_size - len(X_sample_indices), len(remaining))
                        supplement_indices = np.random.choice(remaining, supplement_count, replace=False)
                        X_sample_indices.extend(supplement_indices)

                # 提取抽样后的特征矩阵
                X_sample = X_scaled[X_sample_indices]
                logger.info(f"长度分层抽样完成: 从{X_scaled.shape[0]}个样本中抽取{len(X_sample_indices)}个用于评估")

            except Exception as e:
                # 采样出错时回退到随机采样
                logger.warning(f"分层抽样时出错: {str(e)}，将使用随机抽样")
                indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
                X_sample = X_scaled[indices]
        else:
            X_sample = X_scaled

        # 2. 使用固定的样本集评估不同聚类数
        silhouette_scores = []
        for n in range_clusters:
            kmeans = MiniBatchKMeans(n_clusters=n, random_state=42, batch_size=min(1024, X_sample.shape[0]))
            cluster_labels = kmeans.fit_predict(X_sample)

            if len(np.unique(cluster_labels)) <= 1:
                continue

            score = silhouette_score(X_sample, cluster_labels)
            silhouette_scores.append((n, score))

            # 可以添加简洁的进度日志
            if len(range_clusters) > 10 and n % (len(range_clusters) // 5) == 0:
                logger.info(f"  评估进度: {n}/{max(range_clusters)}，当前轮廓系数: {score:.4f}")

            kmeans = MiniBatchKMeans(n_clusters=n, random_state=42, batch_size=min(1024, X_sample.shape[0]))
            cluster_labels = kmeans.fit_predict(X_sample)

            if len(np.unique(cluster_labels)) <= 1:
                continue

            score = silhouette_score(X_sample, cluster_labels)
            silhouette_scores.append((n, score))

        if silhouette_scores:
            # 选择轮廓系数最高的簇数
            n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
            logger.info(f"基于轮廓系数选择的最佳簇数: {n_clusters}")
        else:
            n_clusters = max(3, int(np.sqrt(len(sequences)) / 2))
            logger.info(f"未能确定最佳簇数，使用默认公式: {n_clusters}")

    # 使用MiniBatchKMeans进行聚类
    logger.info(f"使用MiniBatchKMeans进行聚类，簇数 = {n_clusters}...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=min(1024, X_scaled.shape[0]))
    labels = kmeans.fit_predict(X_scaled)

    # 统计每个簇的样本数
    unique_labels = np.unique(labels)
    cluster_counts = Counter(labels)
    logger.info(f"聚类完成，共形成 {len(unique_labels)} 个簇")

    # 计算每个簇的大小并排序
    cluster_sizes = [(label, count) for label, count in cluster_counts.items()]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    # 显示簇的分布情况
    logger.info("簇分布情况:")
    total_seqs = sum(count for _, count in cluster_sizes)
    for rank, (label, count) in enumerate(cluster_sizes[:10]):  # 只显示前10个最大的簇
        percentage = count * 100 / total_seqs
        logger.info(f"  簇 {label} (排名 {rank + 1}): {count} 个序列 ({percentage:.1f}%)")

    # 计算特征重要性 - 使用簇中心间的方差
    feature_importance = np.std(kmeans.cluster_centers_, axis=0)
    feature_ranks = np.argsort(feature_importance)[::-1]  # 降序排序

    logger.info("特征重要性排名:")
    for i, idx in enumerate(feature_ranks[:min(5, len(feature_names))]):
        if idx < len(feature_names):
            importance = feature_importance[idx]
            logger.info(f"  {i + 1}. {feature_names[idx]}: {importance:.4f}")

    # 从每个簇中选择代表性样本
    retained_info = {}  # 记录选择信息

    logger.info("从每个簇中选择代表性样本...")
    retained_ids = hybrid_representative_sampling(
        unique_labels=unique_labels,
        cluster_sizes=cluster_sizes,
        labels=labels,
        seq_ids=seq_ids,
        X_scaled=X_scaled,
        sample_ratio=sample_ratio,
        key_features=feature_ranks[:5],  # 使用前5个重要特征
        feature_weights=feature_importance[feature_ranks[:5]]  # 对应特征的权重
    )

    # 生成聚类可视化 (如果需要)
    if pca_vis and output_dir:
        try:
            logger.info("生成PCA可视化...")
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            # 使用PCA降维到2D
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # 创建散点图
            plt.figure(figsize=(12, 10))
            colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

            # 绘制所有点
            for label, color in zip(unique_labels, colors):
                idx = np.where(labels == label)[0]
                plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[color], label=f'Cluster {label}',
                            alpha=0.5, s=20, edgecolors='none')

            # 标记所选样本
            retained_indices = [i for i, seq_id in enumerate(seq_ids) if seq_id in retained_ids]
            plt.scatter(X_pca[retained_indices, 0], X_pca[retained_indices, 1],
                        marker='*', s=100, c='red', label='Selected')

            # 添加图例与说明
            plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title(f'AMPs Clustering Result (n_clusters={n_clusters}, kept={len(retained_ids)})')
            plt.legend(loc='upper right')

            # 保存图片
            vis_path = os.path.join(output_dir, 'amp_clustering_pca.png')
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            logger.info(f"PCA可视化保存至: {vis_path}")

            # 生成每个维度重要性的条形图
            plt.figure(figsize=(12, 6))
            sorted_idx = np.argsort(feature_importance)
            plt.barh(range(len(feature_names)), feature_importance[sorted_idx])
            plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance for AMPs Clustering')

            # 保存特征重要性图
            feat_path = os.path.join(output_dir, 'amp_feature_importance.png')
            plt.savefig(feat_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图保存至: {feat_path}")

        except Exception as e:
            logger.error(f"生成可视化时出错: {str(e)}")

    # 确保返回的ID是可哈希类型
    safe_retained_ids = []
    for id_item in retained_ids:
        if isinstance(id_item, list):
            safe_retained_ids.append(tuple(id_item))
        else:
            safe_retained_ids.append(id_item)

    # 去重，确保没有重复ID
    retained_ids = list(set(retained_ids))

    elapsed = time.time() - start_time
    reduction_ratio = (len(sequences) - len(retained_ids)) * 100 / len(sequences)
    logger.info(f"AMPs优化聚类完成: 从 {len(sequences)} 个序列中选择了 {len(retained_ids)} 个代表性序列 "
                f"(占 {len(retained_ids) * 100 / len(sequences):.1f}%, 减少率 {reduction_ratio:.1f}%)，"
                f"耗时: {elapsed / 60:.1f}分钟")
    # 返回结果及统计信息
    return retained_ids, {
        "total": len(sequences),
        "selected": len(retained_ids),
        "clusters": len(unique_labels),
        "reduction_ratio": reduction_ratio,
        "features_used": feature_names,
        "cluster_info": retained_info,
        "elapsed_minutes": elapsed / 60
    }


def check_graph_files_exist(batch_dirs):
    """检查是否存在知识图谱文件"""
    for batch_dir in batch_dirs:
        kg_pyg_dir = os.path.join(batch_dir, "knowledge_graphs_pyg")
        if os.path.exists(kg_pyg_dir):
            pt_files = glob.glob(os.path.join(kg_pyg_dir, "protein_kg_chunk_*.pt"))
            if pt_files:
                return True
    return False


def safe_load_graph(file_path, map_location=None):
    """
    安全加载图谱文件，避免内存映射问题并处理所有异常

    参数:
        file_path: 图谱文件路径
        map_location: PyTorch加载时的设备位置

    返回:
        成功时返回加载的图谱，失败时返回空字典
    """
    try:
        # 检查文件是否存在
        if not os.path.isfile(file_path):
            logging.error(f"文件不存在: {file_path}")
            return {}

        # 读取文件内容到内存
        with open(file_path, 'rb') as f:
            file_content = f.read()
            if not file_content:
                logging.error(f"文件为空: {file_path}")
                return {}

            # 创建内存缓冲区
            buffer = io.BytesIO(file_content)

            # 直接从内存缓冲区加载，不使用mmap
            result = torch.load(buffer, map_location=map_location)
            buffer.close()

            # 确保返回值是可迭代的
            if result is None:
                logging.error(f"加载结果为None: {file_path}")
                return {}

            return result

    except Exception as e:
        logging.error(f"加载文件失败: {file_path}, 错误: {str(e)}")
        return {}  # 返回空字典而不是None，避免迭代错误


def process_file_mapping_chunk(chunk, remaining_ids):
    """处理文件映射块 - 向量化优化版"""
    # 预先将remaining_ids转换为集合以加速查找
    remaining_ids_set = set(remaining_ids)

    # 使用字典推导式批量计算交集
    # 这比循环逐一处理每个文件要高效得多
    intersections = {file_path: ids.intersection(remaining_ids_set)
                     for file_path, ids in chunk.items()}

    # 筛选出有交集的文件
    local_matched_files = {file_path for file_path, common_ids in intersections.items()
                           if common_ids}

    # 构建ID到文件的映射 - 使用defaultdict避免重复检查键是否存在
    local_id_to_files = defaultdict(list)

    # 只处理有交集的文件，减少计算量
    for file_path, common_ids in intersections.items():
        if common_ids:  # 如果有交集，则此文件需要加载
            for graph_id in common_ids:
                local_id_to_files[graph_id].append(file_path)

    return dict(local_id_to_files), local_matched_files


def check_pt_file_content(file_path):
    """检查PT文件包含的图谱ID - 向量化优化版"""
    try:
        # 使用安全加载函数加载文件
        data = safe_load_graph(file_path, map_location='cpu')

        # 向量化处理字典键
        if isinstance(data, dict):
            # 使用set()直接从字典键创建集合，比循环添加更高效
            file_ids = set(data.keys())

            # 清理内存
            del data

            return file_path, file_ids

    except Exception as e:
        logger.debug(f"检查文件 {os.path.basename(file_path)} 失败: {str(e)}")

    return file_path, set()


def process_file_batch(batch_files, target_ids, id_mapping):
    """处理文件批次，加载包含目标ID的图谱 - 向量化优化版"""
    batch_start_time = time.time()
    batch_graphs = {}
    processed_count = 0

    # 预处理 - 将target_ids转换为集合以加速查找
    target_ids_set = set(target_ids)

    # 预筛选包含目标ID的文件，减少不必要的加载
    files_to_process = []
    file_to_target_ids = {}

    for file_path in batch_files:
        # 获取文件中包含的ID与目标ID的交集
        file_ids = id_mapping.get(file_path, set())
        common_ids = target_ids_set.intersection(file_ids)

        # 如果文件包含目标ID，则加入待处理列表
        if common_ids:
            files_to_process.append(file_path)
            file_to_target_ids[file_path] = common_ids

    # 如果没有文件包含目标ID，则直接返回
    if not files_to_process:
        return {}, 0, 0

    # 处理每个包含目标ID的文件
    for file_path in files_to_process:
        try:
            # 获取当前文件需要提取的ID
            common_ids = file_to_target_ids[file_path]

            # 只加载包含目标ID的图谱
            graphs_data = safe_load_graph(file_path, map_location='cpu')

            if not graphs_data or not isinstance(graphs_data, dict):
                logger.warning(f"文件格式不正确或为空: {os.path.basename(file_path)}")
                continue

            # 向量化提取所需的图谱 - 使用集合操作优化
            # 一次性获取所有公共ID，避免循环中重复查找
            for graph_id in common_ids:
                if graph_id in graphs_data:
                    batch_graphs[graph_id] = graphs_data[graph_id]
                    processed_count += 1

            # 释放内存
            del graphs_data

            # 定期检查内存使用情况
            if processed_count % 10000 == 0:
                check_memory_usage(threshold_gb=900, force_gc=True)

        except Exception as e:
            logger.debug(f"处理文件失败: {os.path.basename(file_path)}, 错误: {str(e)}")

    batch_time = time.time() - batch_start_time
    return batch_graphs, processed_count, batch_time

def load_cached_graphs(cache_dir, cache_id, selected_ids_set):
    """超高速缓存加载器 - 向量化优化无mmap版本"""
    cached_graphs = {}
    cache_meta_file = os.path.join(cache_dir, f"{cache_id}_meta.json")

    if not os.path.exists(cache_meta_file):
        return cached_graphs

    try:
        # 查找所有缓存文件并按修改时间排序
        cache_files = glob.glob(os.path.join(cache_dir, f"{cache_id}_part_*.pt"))
        if not cache_files:
            return cached_graphs

        # 使用numpy进行高效排序，比Python的sorted更快
        cache_files_array = np.array(cache_files)
        mtimes = np.array([os.path.getmtime(f) for f in cache_files])
        sorted_indices = np.argsort(-mtimes)  # 降序排序
        cache_files = cache_files_array[sorted_indices].tolist()

        logger.info(f"发现 {len(cache_files)} 个图谱缓存文件")

        # 预计算已加载ID集合，避免重复加载
        loaded_ids = set()

        # 将文件分成批次处理以平衡内存使用
        batch_size = max(1, len(cache_files) // 4)
        batches = [cache_files[i:i + batch_size] for i in range(0, len(cache_files), batch_size)]

        total_loaded = 0
        with tqdm(total=len(cache_files), desc="加载缓存文件", ncols=100) as pbar:
            # 处理每个批次
            for batch_idx, batch_files in enumerate(batches):
                # 统计当前批次加载信息
                batch_loaded = 0
                batch_start = time.time()

                # 顺序处理当前批次的文件，避免使用线程池
                for file_path in batch_files:
                    try:
                        # 使用向量化IO操作加载文件内容到内存
                        with open(file_path, 'rb') as f:
                            file_content = f.read()
                            if not file_content:
                                pbar.update(1)
                                continue

                            # 使用BytesIO避免内存映射问题
                            buffer = io.BytesIO(file_content)
                            file_data = torch.load(buffer, map_location='cpu')
                            buffer.close()

                        # 向量化图谱筛选
                        if file_data and isinstance(file_data, dict):
                            # 找出未加载且在选定ID中的图谱
                            ids_to_add = set(file_data.keys()) & selected_ids_set - loaded_ids

                            # 批量添加图谱
                            for graph_id in ids_to_add:
                                graph = file_data[graph_id]
                                # 确保对象没有锁
                                if isinstance(graph, Data):
                                    # 将tensor移到CPU
                                    for key, value in graph:
                                        if hasattr(value, 'is_cuda') and value.is_cuda:
                                            graph[key] = value.cpu()

                                cached_graphs[graph_id] = graph
                                batch_loaded += 1

                            # 更新已加载ID集合
                            loaded_ids.update(ids_to_add)

                        # 更新进度条
                        pbar.update(1)

                    except Exception as e:
                        logger.debug(f"处理缓存文件失败: {os.path.basename(file_path)}, 错误: {str(e)}")
                        pbar.update(1)

                # 更新总计数
                total_loaded += batch_loaded
                batch_time = time.time() - batch_start
                logger.info(f"批次 {batch_idx + 1}/{len(batches)} 完成，此批次加载 {batch_loaded} 个图谱，"
                            f"耗时: {batch_time:.2f}秒 ({batch_loaded / batch_time if batch_time > 0 else 0:.1f}图谱/秒)")

                # 每批次后强制垃圾回收
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 计算覆盖率
        coverage = len(cached_graphs) * 100 / len(selected_ids_set) if selected_ids_set else 0
        logger.info(f"缓存加载完成: 命中率 {coverage:.1f}% ({len(cached_graphs)}/{len(selected_ids_set)})")

    except Exception as e:
        logger.error(f"加载缓存失败: {str(e)}")
        traceback.print_exc()

    return cached_graphs

# 定义文件处理函数
def process_file(file_path):
    try:
        # 快速扫描文件以提取ID，而不是完全加载
        file_ids = scan_file_for_ids(file_path)
        return file_path, file_ids
    except Exception as e:
        logger.debug(f"处理文件 {file_path} 时出错: {str(e)}")
        return file_path, []
def index_batch_directory(batch_dir, num_workers=64):
    """
    为单个批次目录构建文件到ID的映射索引

    参数:
        batch_dir (str): 批次目录路径
        num_workers (int): 并行处理的工作线程数

    返回:
        dict: 文件路径到包含ID列表的映射 {file_path: [id1, id2, ...]}
    """
    import os
    import glob
    import time
    import pickle
    import concurrent.futures
    import logging

    # 创建logger对象（如果在主函数中已定义）
    logger = logging.getLogger(__name__)

    start_time = time.time()
    batch_file_to_ids = {}

    # 查找批次目录中的所有图谱文件
    file_pattern = os.path.join(batch_dir, "*.pkl")
    graph_files = glob.glob(file_pattern)

    if not graph_files:
        logger.warning(f"批次目录 {batch_dir} 中未找到图谱文件")
        return {}

    logger.debug(f"在 {batch_dir} 中找到 {len(graph_files)} 个文件")

    # 并行处理文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, file_path): file_path for file_path in graph_files}

        for future in concurrent.futures.as_completed(futures):
            try:
                file_path, file_ids = future.result()
                if file_ids:  # 只有在找到ID时才添加到映射
                    batch_file_to_ids[file_path] = file_ids
            except Exception as e:
                file_path = futures[future]
                logger.debug(f"处理文件 {file_path} 结果时出错: {str(e)}")

    elapsed = time.time() - start_time
    logger.debug(f"索引批次目录 {batch_dir} 完成: {len(batch_file_to_ids)} 文件, 耗时 {elapsed:.2f}秒")

    return batch_file_to_ids


def scan_file_for_ids(file_path):
    """
    扫描图谱文件以提取ID列表，无需完全加载文件内容

    参数:
        file_path (str): 图谱文件路径

    返回:
        list: 文件中包含的图谱ID列表
    """
    import os
    import pickle
    import mmap

    # 文件太小，无法使用mmap时的阈值
    MIN_SIZE_FOR_MMAP = 1024 * 10  # 10KB

    try:
        file_size = os.path.getsize(file_path)

        # 对于非常小的文件，直接加载
        if file_size < MIN_SIZE_FOR_MMAP:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

                # 根据数据结构提取ID
                if isinstance(data, dict):
                    # 如果是字典格式，键可能是ID
                    return list(data.keys())
                elif isinstance(data, list):
                    # 如果是列表，每个元素可能包含ID字段
                    ids = []
                    for item in data:
                        if isinstance(item, dict) and 'id' in item:
                            ids.append(item['id'])
                    return ids
                else:
                    # 假设数据是单个图谱对象
                    return [getattr(data, 'id', None)] if hasattr(data, 'id') else []

        # 对于大文件，使用内存映射读取头部来提取信息
        with open(file_path, 'rb') as f:
            # 只映射文件的前部分来读取头信息
            header_size = min(file_size, 4096)  # 读取前4KB
            mm = mmap.mmap(f.fileno(), header_size, access=mmap.ACCESS_READ)

            # 尝试从文件头部提取结构信息
            try:
                # 读取pickle头部信息
                if mm[0:1] == b'\x80':  # pickle协议头的常见字节
                    # 如果是标准pickle格式，分析文件名来提取ID
                    file_name = os.path.basename(file_path)
                    file_name_without_ext = os.path.splitext(file_name)[0]

                    # 假设文件名包含ID信息（通常情况）
                    if '_' in file_name_without_ext:
                        parts = file_name_without_ext.split('_')
                        potential_ids = [part for part in parts if len(part) > 4]  # ID通常较长
                        if potential_ids:
                            return potential_ids

                    # 如果无法从文件名提取，返回文件名作为ID
                    return [file_name_without_ext]
            except:
                pass

            # 释放内存映射
            mm.close()

        # 如果无法从头部提取，加载完整文件（可能较慢）
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

            # 重复上面的ID提取逻辑
            if isinstance(data, dict):
                return list(data.keys())
            elif isinstance(data, list):
                ids = []
                for item in data:
                    if isinstance(item, dict) and 'id' in item:
                        ids.append(item['id'])
                return ids
            else:
                return [getattr(data, 'id', None)] if hasattr(data, 'id') else []

    except Exception as e:
        # 出错时返回空列表
        return []


def optimized_process_file_batch(file_batch, target_ids, use_mmap=True):
    """
    优化的文件批处理函数，支持内存映射和增量处理

    参数:
        file_batch (list): 要处理的文件路径列表
        target_ids (list): 目标ID列表
        use_mmap (bool): 是否使用内存映射技术

    返回:
        tuple: (加载的图谱字典, 成功处理的ID列表, 批处理总耗时)
    """
    import os
    import time
    import pickle
    import mmap
    import logging

    # 创建logger对象（如果在主函数中已定义）
    logger = logging.getLogger(__name__)

    start_time = time.time()
    batch_graphs = {}
    processed_ids = set()

    # 转换为集合加速查找
    target_ids_set = set(target_ids)

    for file_path in file_batch:
        try:
            file_size = os.path.getsize(file_path)

            # 小文件直接加载
            if file_size < 1024 * 1024 * 10 or not use_mmap:  # <10MB
                with open(file_path, 'rb') as f:
                    file_data = pickle.load(f)
            else:
                # 大文件使用内存映射加载
                with open(file_path, 'rb') as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    file_data = pickle.load(mm)
                    mm.close()

            # 处理不同的数据格式
            if isinstance(file_data, dict):
                # 字典格式：{id: graph_data, ...}
                relevant_ids = target_ids_set.intersection(file_data.keys())
                for id_ in relevant_ids:
                    batch_graphs[id_] = file_data[id_]
                    processed_ids.add(id_)

            elif isinstance(file_data, list):
                # 列表格式：[{id: id1, data: ...}, ...]
                for item in file_data:
                    if isinstance(item, dict) and 'id' in item:
                        id_ = item['id']
                        if id_ in target_ids_set:
                            batch_graphs[id_] = item
                            processed_ids.add(id_)

            else:
                # 单个图谱对象
                if hasattr(file_data, 'id'):
                    id_ = getattr(file_data, 'id')
                    if id_ in target_ids_set:
                        batch_graphs[id_] = file_data
                        processed_ids.add(id_)

        except Exception as e:
            logger.debug(f"处理文件 {file_path} 时出错: {str(e)}")
            continue

    elapsed = time.time() - start_time

    return batch_graphs, list(processed_ids), elapsed


# 并行构建映射
def build_id_mapping_chunk(file_chunk):
    chunk_id_to_files = {}
    for file_path, ids in file_chunk.items():
        for id_ in ids:
            if id_ not in chunk_id_to_files:
                chunk_id_to_files[id_] = []
            chunk_id_to_files[id_].append(file_path)
    return chunk_id_to_files

def accelerated_graph_loader(input_path, filtered_ids, num_workers=128, memory_limit_gb=900,
                             use_cache=True, cache_dir=None, cache_id="graph_data_cache",
                             load_from_cache_only=False):
    """
    加速版图谱加载器 - 优先从缓存加载，然后按需从源文件加载缺失部分

    参数:
        input_path: 输入目录路径
        filtered_ids: 已过滤的ID列表
        num_workers: 工作进程数
        memory_limit_gb: 内存使用限制(GB)
        use_cache: 是否使用缓存
        cache_dir: 缓存目录路径
        cache_id: 缓存ID前缀
        use_mmap: 是否使用内存映射加载
        load_from_cache_only: 是否只从缓存加载，不尝试从源文件加载

    返回:
        dict: 加载的图谱字典
    """
    # 构建ID集合以加速查找
    filtered_ids_set = set(filtered_ids)
    logger.info(f"图谱加载目标: {len(filtered_ids_set)}个ID")

    # 初始化结果容器
    loaded_graphs = {}
    remaining_ids = filtered_ids_set.copy()  # 跟踪尚未加载的ID

    # 检查缓存目录
    cache_exists = False
    if use_cache and cache_dir:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"创建缓存目录: {cache_dir}")
        else:
            # 检查是否有缓存文件
            cache_meta_file = os.path.join(cache_dir, f"{cache_id}_meta.json")
            cache_files = glob.glob(os.path.join(cache_dir, f"{cache_id}_part_*.pt"))

            if os.path.exists(cache_meta_file) and cache_files:
                cache_exists = True
                logger.info(f"发现缓存文件: {len(cache_files)}个文件和元数据")
            else:
                logger.info(f"缓存目录存在，但未找到有效缓存文件")

    # 第一步: 尝试从缓存加载
    if use_cache and cache_dir and os.path.exists(cache_dir):
        cache_meta_file = os.path.join(cache_dir, f"{cache_id}_meta.json")
        logger.info(f"从缓存目录加载图谱: {cache_dir}")
        cache_start = time.time()

        # 检查缓存元数据是否存在
        use_existing_cache = False
        if os.path.exists(cache_meta_file):
            try:
                with open(cache_meta_file, 'r') as f:
                    cache_meta_data = json.load(f)

                # 检查元数据中的ID数量与当前需要加载的ID数量是否匹配
                if "id_count" in cache_meta_data and cache_meta_data["id_count"] == len(filtered_ids_set):
                    logger.info(f"缓存元数据ID数量匹配当前请求：{len(filtered_ids_set)}个ID")
                    use_existing_cache = True
                else:
                    logger.info(
                        f"缓存元数据ID数量({cache_meta_data.get('id_count', 'unknown')})与当前请求({len(filtered_ids_set)})不匹配")
            except Exception as e:
                logger.warning(f"读取缓存元数据失败: {str(e)}")

        # 如果元数据匹配，则从缓存加载
        if use_existing_cache:
            # 从缓存加载图谱
            cached_graphs = load_cached_graphs(cache_dir, cache_id, filtered_ids_set)

            if cached_graphs:
                # 更新已加载和待加载的ID集合
                loaded_graphs.update(cached_graphs)
                cached_ids = set(cached_graphs.keys())
                remaining_ids = filtered_ids_set - cached_ids

                logger.info(f"从缓存中加载了 {len(cached_graphs)} 个图谱，剩余 {len(remaining_ids)} 个需要从源文件加载")
                logger.info(f"缓存加载耗时: {time.time() - cache_start:.1f}秒")

                # 释放内存
                del cached_graphs
                check_memory_usage(force_gc=True)
            else:
                logger.info("缓存中未找到有效图谱数据")
                # 重置remaining_ids，因为缓存加载失败
                remaining_ids = filtered_ids_set.copy()
        else:
            logger.info("缓存元数据不匹配或不存在，将重新生成缓存")
            # 确保使用完整的filtered_ids_set进行源文件加载
            remaining_ids = filtered_ids_set.copy()
    else:
        # 如果不使用缓存，确保处理所有ID
        remaining_ids = filtered_ids_set.copy()

    # 如果所有ID已从缓存加载或只允许从缓存加载，直接返回
    if not remaining_ids or load_from_cache_only:
        if load_from_cache_only:
            logger.info("按要求仅从缓存加载图谱，不尝试从源文件加载")
        else:
            logger.info("所有请求的图谱已从缓存加载完成")
        return loaded_graphs

    # 第二步: 从源文件加载剩余ID
    logger.info(f"将从源文件加载 {len(remaining_ids)} 个图谱")

    # 查找所有批次目录
    batch_dirs = find_batch_directories(input_path)
    logger.info(f"找到 {len(batch_dirs)} 个批次目录")

    # 收集所有图谱文件
    pt_files = []
    for batch_dir in batch_dirs:
        kg_pyg_dir = os.path.join(batch_dir, "knowledge_graphs_pyg")
        if os.path.exists(kg_pyg_dir):
            files = glob.glob(os.path.join(kg_pyg_dir, "protein_kg_chunk_*.pt"))
            pt_files.extend(files)

    if not pt_files:
        logger.warning("未找到任何图谱文件")
        return loaded_graphs

    logger.info(f"找到 {len(pt_files)} 个图谱文件")

    # 为提高效率，首先扫描所有文件以确定ID到文件的映射
    logger.info("扫描文件内容以确定ID到文件的映射...")
    file_to_ids = {}  # 文件路径 -> ID集合

    # 并行扫描所有文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(check_pt_file_content, file_path) for file_path in pt_files]

        with tqdm(total=len(pt_files), desc="扫描文件", ncols=100) as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    file_path, file_ids = future.result()
                    if file_ids:
                        file_to_ids[file_path] = file_ids
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"扫描文件时出错: {str(e)}")

    # 构建ID到文件的映射和需要加载的文件集合
    id_to_files = {}  # ID -> 文件路径列表
    matched_files = set()  # 包含目标ID的文件路径集合

    # 将映射任务分成多个块并行处理
    chunk_size = max(1, len(file_to_ids) // num_workers)
    file_to_ids_items = list(file_to_ids.items())
    chunks = [dict(file_to_ids_items[i:i + chunk_size]) for i in range(0, len(file_to_ids_items), chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_workers, len(chunks))) as executor:
        futures = [executor.submit(process_file_mapping_chunk, chunk, remaining_ids) for chunk in chunks]

        for future in concurrent.futures.as_completed(futures):
            try:
                local_id_to_files, local_matched_files = future.result()
                id_to_files.update(local_id_to_files)
                matched_files.update(local_matched_files)
            except Exception as e:
                logger.error(f"处理文件映射时出错: {str(e)}")

    logger.info(f"找到 {len(matched_files)} 个包含目标ID的文件")

    # 将文件划分成批次并行加载
    batch_size = max(1, len(matched_files) // num_workers)
    matched_files_list = list(matched_files)
    batches = [matched_files_list[i:i + batch_size] for i in range(0, len(matched_files_list), batch_size)]

    logger.info(f"文件分为 {len(batches)} 个批次进行加载")

    # 并行加载每个批次
    total_processed = 0
    total_start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_workers, len(batches))) as executor:
        futures = [executor.submit(process_file_batch, batch, list(remaining_ids), file_to_ids) for batch in batches]

        with tqdm(total=len(batches), desc="加载文件批次", ncols=100) as pbar:
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    batch_graphs, processed_count, batch_time = future.result()

                    # 更新结果
                    loaded_graphs.update(batch_graphs)
                    total_processed += processed_count

                    # 更新剩余ID
                    remaining_ids -= set(batch_graphs.keys())

                    # 更新进度
                    pbar.update(1)
                    pbar.set_postfix({
                        'loaded': len(loaded_graphs),
                        'remaining': len(remaining_ids),
                        'elapsed': f"{(time.time() - total_start_time):.1f}s"
                    })

                    # 检查内存使用
                    check_memory_usage(threshold_gb=memory_limit_gb)

                except Exception as e:
                    logger.error(f"处理批次时出错: {str(e)}")

    # 计算统计信息
    total_time = time.time() - total_start_time
    coverage = len(loaded_graphs) * 100 / len(filtered_ids_set) if filtered_ids_set else 0

    logger.info(f"图谱加载完成:")
    logger.info(f"- 总共加载: {len(loaded_graphs)}/{len(filtered_ids_set)} 个图谱 (覆盖率: {coverage:.1f}%)")
    logger.info(
        f"- 总耗时: {total_time:.1f}秒 (平均 {total_time / len(loaded_graphs):.3f}秒/图谱)" if loaded_graphs else "- 总耗时: 0秒")
    logger.info(f"- 未找到的图谱数: {len(remaining_ids)}")

    return loaded_graphs


def save_results_chunked(data, output_dir, base_name="filtered_data", chunk_size=5000):
    """分块保存数据到文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 将数据分块
    data_ids = list(data.keys())
    chunks = [data_ids[i:i + chunk_size] for i in range(0, len(data_ids), chunk_size)]

    output_files = []
    for i, chunk_ids in enumerate(tqdm(chunks, desc=f"保存{base_name}")):
        chunk_data = {id: data[id] for id in chunk_ids}
        output_file = os.path.join(output_dir, f"{base_name}_chunk_{i + 1}.pt")

        try:
            torch.save(chunk_data, output_file)
            output_files.append(output_file)
        except Exception as e:
            logger.error(f"保存数据块 {i + 1} 时出错: {str(e)}")

    # 保存元数据
    metadata = {
        "base_name": base_name,
        "total_items": len(data),
        "chunk_size": chunk_size,
        "chunks": len(chunks),
        "files": output_files,
        "timestamp": time.time()
    }

    metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"保存了 {len(data)} 个{base_name}到 {len(chunks)} 个数据块")
    return output_files, metadata


def save_filtered_data(filtered_sequences, filtered_graphs, output_dir, chunk_size=60000, input_path=None):
    """
    保存去冗余后的数据 - 优化版本，使用统一的缓存格式

    参数:
        filtered_sequences: 过滤后的序列字典
        filtered_graphs: 过滤后的图谱字典
        output_dir: 输出目录
        chunk_size: 分块大小
        input_path: 输入路径，用于元数据记录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 创建时间戳，用于文件命名和元数据
    timestamp = int(time.time())

    # 保存序列ID列表 - 轻量级方式
    logger.info("保存过滤后的序列ID列表...")
    sequence_ids = list(filtered_sequences.keys())
    seq_ids_file = os.path.join(output_dir, "filtered_sequence_ids.json")
    with open(seq_ids_file, 'w') as f:
        json.dump(sequence_ids, f)
    logger.info(f"序列ID列表已保存: {len(sequence_ids)} 个ID -> {seq_ids_file}")

    # 同时保存一份兼容旧代码的序列ID文件
    seq_retained_file = os.path.join(output_dir, "seq_retained_ids.json")
    with open(seq_retained_file, 'w') as f:
        json.dump({"retained_ids": sequence_ids, "completed": True}, f)

    # 保存图谱引用信息 - 不保存实际数据，只保存ID
    if filtered_graphs:
        logger.info("保存过滤后的图谱ID列表...")
        graph_ids = list(filtered_graphs.keys())
        graph_ids_file = os.path.join(output_dir, "filtered_graph_ids.json")
        with open(graph_ids_file, 'w') as f:
            json.dump(graph_ids, f)
        logger.info(f"图谱ID列表已保存: {len(graph_ids)} 个ID -> {graph_ids_file}")

        # 创建图谱缓存目录
        cache_dir = os.path.join(output_dir, "graph_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # 保存图谱缓存 - 分批处理以减少内存压力
        logger.info(f"将图谱数据保存到缓存: {cache_dir}")
        cache_id = "filtered_graphs"

        # 将数据分块保存，使用时间戳命名
        graph_ids_chunks = [graph_ids[i:i + chunk_size] for i in range(0, len(graph_ids), chunk_size)]
        cache_files = []

        for i, chunk_ids in enumerate(tqdm(graph_ids_chunks, desc="保存图谱缓存")):
            chunk_data = {gid: filtered_graphs[gid] for gid in chunk_ids}
            cache_file = os.path.join(cache_dir, f"{cache_id}_part_{timestamp}_{i}.pt")

            try:
                torch.save(chunk_data, cache_file)
                cache_files.append(cache_file)
            except Exception as e:
                logger.error(f"保存图谱缓存块 {i + 1} 时出错: {str(e)}")

        # 保存缓存元数据 - 使用统一的格式
        meta_file = os.path.join(cache_dir, f"{cache_id}_meta.json")
        cache_meta = {
            "input_path": input_path if input_path else "unknown",
            "id_count": len(sequence_ids),  # 序列ID数量
            "cached_count": len(graph_ids),  # 成功缓存的图谱数量
            "expected_graph_count": len(graph_ids),  # 预期图谱总数
            "is_complete_set": True,  # 明确标记这是完整集
            "timestamp": timestamp,
            "version": "3.4",
            "total_graphs": len(graph_ids),
            "chunks": len(graph_ids_chunks),
            "chunk_size": chunk_size,
            "files": cache_files
        }

        with open(meta_file, 'w') as f:
            json.dump(cache_meta, f, indent=2)

        logger.info(f"图谱缓存已保存: {len(graph_ids)} 个图谱分为 {len(graph_ids_chunks)} 个缓存块")

    # 保存统计信息
    save_statistics(output_dir, len(filtered_sequences), len(filtered_graphs))

    # 返回缓存信息，方便后续处理
    return {
        "seq_count": len(filtered_sequences),
        "graph_count": len(filtered_graphs) if filtered_graphs else 0,
        "timestamp": timestamp,
        "output_dir": output_dir
    }


def save_statistics(output_dir, filtered_sequences_count, filtered_graphs_count):
    """保存过滤统计信息 - 简化版"""
    summary_file = os.path.join(output_dir, "filtering_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"过滤统计信息 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
        f.write(f"过滤后序列数量: {filtered_sequences_count}\n")
        f.write(f"过滤后图谱数量: {filtered_graphs_count}\n")
        f.write(f"数据保存格式: 仅ID列表+图谱缓存\n")  # 添加说明
        f.write("-" * 50 + "\n")

    logger.info(f"统计信息已保存至: {summary_file}")


def process_sequences_and_graphs(input_path, output_dir,
                                 identity_threshold=0.6, coverage_threshold=0.8,
                                 n_clusters=None, sample_ratio=0.3,
                                 use_clustering=True, chunk_size=5000,
                                 num_workers=32, memory_limit_gb=800,
                                 batch_size=500000, num_perm=128,
                                 test_mode=False, max_test_files=5,
                                 use_cache=True, cache_dir=None):
    """
    处理序列和图谱的主函数

    参数:
        input_path: 输入目录路径
        output_dir: 输出目录路径
        identity_threshold: 序列同一性阈值
        coverage_threshold: 序列覆盖率阈值
        n_clusters: 聚类数量，None表示自动确定
        sample_ratio: 聚类采样比例
        use_clustering: 是否使用聚类分析
        chunk_size: 保存结果时的分块大小
        num_workers: 并行处理的工作进程数
        memory_limit_gb: 内存使用上限(GB)
        batch_size: 每批处理的数据量
        num_perm: MinHash使用的排列数
        test_mode: 是否为测试模式
        max_test_files: 测试模式下最多处理的文件数
        use_cache: 是否使用缓存加载图谱
        cache_dir: 图谱缓存目录

    返回:
        tuple: (过滤后的序列, 过滤后的图谱)
    """
    # 记录开始时间
    start_time = time.time()

    # 第一步：加载序列数据
    logger.info("步骤1: 加载序列数据...")
    sequences = parallel_load_sequences(
        input_path,
        num_workers=num_workers,
        memory_limit_gb=memory_limit_gb,
        test_mode=test_mode,
        max_test_files=max_test_files
    )
    logger.info(f"加载了 {len(sequences)} 个序列")

    # 第二步：序列去冗余
    logger.info(f"步骤2: 序列去冗余 (同一性阈值: {identity_threshold}, 覆盖率阈值: {coverage_threshold})...")
    filtered_sequences, filtered_ids = distributed_minhash_filter(
        sequences,
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
        num_perm=num_perm,
        num_workers=num_workers,
        batch_size=batch_size,
        output_dir=output_dir
    )
    logger.info(f"序列去冗余完成，保留 {len(filtered_sequences)}/{len(sequences)} 个序列")

    # 清理内存
    del sequences
    check_memory_usage(force_gc=True)

    # 第三步(可选)：序列聚类
    if use_clustering:
        logger.info("步骤3: 序列聚类分析...")
        cluster_path = os.path.join(output_dir, "clustering_results")
        os.makedirs(cluster_path, exist_ok=True)
        # 定义聚类结果文件路径
        cluster_cache_path = os.path.join(cluster_path, "cluster_results.pkl")
        feature_cache_path = os.path.join(cluster_path, "protein_features.pkl")

        # 检查是否已有聚类结果缓存
        if os.path.exists(cluster_cache_path) and os.path.exists(feature_cache_path):
            try:
                logger.info(f"检测到已有聚类结果缓存，正在加载: {cluster_cache_path}")

                # 加载聚类结果
                import pickle
                with open(cluster_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    cluster_filtered_ids = cache_data['cluster_filtered_ids']
                    cluster_info = cache_data['cluster_info']


                logger.info(f"成功加载缓存的聚类结果: {len(cluster_filtered_ids)}个代表性序列")
                logger.info(
                    f"聚类统计: {cluster_info.get('clusters', '未知')}个聚类，去冗余率: {cluster_info.get('reduction_ratio', 0):.1f}%")

            except Exception as e:
                logger.warning(f"加载聚类缓存失败: {str(e)}，将重新执行聚类分析")
                use_cache = False
            else:
                use_cache = True

        elif os.path.exists(feature_cache_path):
            import pickle
            use_cache = False
            # 加载特征
            with open(feature_cache_path, 'rb') as f:
                protein_features = pickle.load(f)
        else:
            use_cache = False

        # 如果没有可用缓存，执行聚类
        if not use_cache:
            logger.info("未找到有效聚类缓存，开始执行特征生成和聚类分析...")
            if protein_features is None:
                logger.info("未找到特征缓存，开始执行特征生成...")
                # 生成序列特征
                protein_features = parallel_generate_amp_features(
                    filtered_sequences,
                    num_workers=num_workers,
                    batch_size=min(5000, len(filtered_sequences))
                )

                # 保存特征缓存
                try:
                    import pickle
                    os.makedirs(os.path.dirname(feature_cache_path), exist_ok=True)
                    with open(feature_cache_path, 'wb') as f:
                        pickle.dump(protein_features, f)
                    logger.info(f"序列特征已保存至: {feature_cache_path}")
                except Exception as e:
                    logger.warning(f"保存序列特征失败: {str(e)}")
            else:
                logger.info(f"加载了 {len(protein_features)} 个特征")

            # 执行聚类
            cluster_filtered_ids, cluster_info = protein_optimized_clustering(
                filtered_sequences,
                protein_features,
                n_clusters=n_clusters,
                sample_ratio=sample_ratio,
                output_dir=output_dir,
                num_workers=num_workers,
            )

            # 确保ID列表中的元素是可哈希类型
            safe_cluster_ids = []
            for seq_id in cluster_filtered_ids:
                if isinstance(seq_id, list):
                    safe_cluster_ids.append(tuple(seq_id))  # 转换列表为元组
                else:
                    safe_cluster_ids.append(seq_id)

            cluster_filtered_ids = safe_cluster_ids

            # 保存聚类结果
            try:
                import pickle
                os.makedirs(os.path.dirname(cluster_cache_path), exist_ok=True)
                cache_data = {
                    'cluster_filtered_ids': cluster_filtered_ids,
                    'cluster_info': cluster_info,
                    'timestamp': time.time(),
                    'version': '1.0'
                }
                with open(cluster_cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info(f"聚类结果已保存至: {cluster_cache_path}")
            except Exception as e:
                logger.warning(f"保存聚类结果失败: {str(e)}")

        # 更新过滤后的序列
        old_count = len(filtered_sequences)
        filtered_sequences = {seq_id: filtered_sequences[seq_id] for seq_id in cluster_filtered_ids if
                              seq_id in filtered_sequences}
        filtered_ids = list(filtered_sequences.keys())
        logger.info(f"聚类分析完成，保留 {len(filtered_sequences)}/{old_count} 个序列")

        # 清理内存
        del protein_features
        check_memory_usage(force_gc=True)

    # 第四步：检查是否存在图谱文件
    batch_dirs = find_batch_directories(input_path)
    has_graphs = check_graph_files_exist(batch_dirs)

    filtered_graphs = {}
    if has_graphs:
        logger.info("步骤4: 加载序列对应的图谱...")

        # 加载对应的图谱
        filtered_graphs = accelerated_graph_loader(
            input_path=input_path,
            filtered_ids=filtered_ids,
            num_workers=num_workers,
            memory_limit_gb=memory_limit_gb,
            use_cache=use_cache,
            cache_dir=cache_dir
        )

        logger.info(f"图谱加载完成，成功加载 {len(filtered_graphs)}/{len(filtered_ids)} 个图谱")
    else:
        logger.info("未检测到图谱文件，跳过图谱加载步骤")

    # 第五步：保存处理结果
    logger.info("步骤5: 保存处理结果...")
    save_filtered_data(
        filtered_sequences,
        filtered_graphs,
        output_dir,
        chunk_size=chunk_size
    )

    # 计算总耗时
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    logger.info(f"处理完成！总耗时: {hours}时{minutes}分{seconds}秒")
    logger.info(f"保留 {len(filtered_sequences)} 个序列和 {len(filtered_graphs)} 个图谱")

    return filtered_sequences, filtered_graphs


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="蛋白质序列处理与图谱保存工具")

    # 输入输出参数
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入目录路径")
    parser.add_argument("--output_dir", "-o", type=str, default="./filtered_data",
                        help="输出目录路径 (默认: ./filtered_data)")

    # 序列处理参数
    parser.add_argument("--identity_threshold", "-id", type=float, default=0.6,
                        help="序列同一性阈值 (默认: 0.6)")
    parser.add_argument("--coverage_threshold", "-cov", type=float, default=0.8,
                        help="序列覆盖率阈值 (默认: 0.8)")
    parser.add_argument("--num_perm", "-np", type=int, default=128,
                        help="MinHash使用的排列数 (默认: 128)")

    # 聚类参数
    parser.add_argument("--use_clustering", "-uc", action="store_true", default=True,
                        help="使用聚类分析 (默认: True)")
    parser.add_argument("--n_clusters", "-nc", type=int, default=None,
                        help="聚类数量，None表示自动确定 (默认: None)")
    parser.add_argument("--sample_ratio", "-sr", type=float, default=0.6,
                        help="从每个簇中选取的样本比例 (默认: 0.6)")

    # 性能参数
    parser.add_argument("--num_workers", "-nw", type=int, default=128,
                        help="并行工作进程数 (默认: 128)")
    parser.add_argument("--memory_limit_gb", "-ml", type=int, default=800,
                        help="内存使用上限(GB) (默认: 800)")
    parser.add_argument("--batch_size", "-bs", type=int, default=500000,
                        help="批处理大小 (默认: 500000)")
    parser.add_argument("--chunk_size", "-cs", type=int, default=5000,
                        help="保存结果时的分块大小 (默认: 5000)")

    # 测试参数
    parser.add_argument("--test_mode", "-tm", action="store_true", default=False,
                        help="测试模式，仅处理少量文件 (默认: False)")
    parser.add_argument("--max_test_files", "-mtf", type=int, default=1,
                        help="测试模式下最多处理的文件数 (默认: 5)")

    # 缓存参数
    parser.add_argument("--use_cache", action="store_true", default=False,
                        help="使用缓存加载图谱 (默认: True)")
    parser.add_argument("--cache_dir", "-cd", type=str, default=None,
                        help="图谱缓存目录，默认为输入目录下的graph_cache")

    # GPU参数
    parser.add_argument("--gpu_device", "-g", type=str, default=1, help="指定GPU设备ID，例如'0,1'，默认使用所有可用GPU")

    args = parser.parse_args()

    # 设置GPU设备
    if args.gpu_device is not None:
        set_gpu_device(args.gpu_device)

    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    global logger
    logger, log_file_path = setup_logging(args.output_dir)
    logger.info(f"日志将写入文件: {log_file_path}")

    # 记录系统资源状态
    log_system_resources()

    # 打印运行配置
    logger.info("运行配置:")
    logger.info(f"- 输入目录: {args.input}")
    logger.info(f"- 输出目录: {args.output_dir}")
    logger.info(f"- 序列同一性阈值: {args.identity_threshold}")
    logger.info(f"- 序列覆盖率阈值: {args.coverage_threshold}")
    logger.info(f"- MinHash排列数: {args.num_perm}")
    logger.info(f"- 使用聚类分析: {'是' if args.use_clustering else '否'}")
    logger.info(f"- 聚类数量: {args.n_clusters if args.n_clusters else '自动确定'}")
    logger.info(f"- 聚类采样比例: {args.sample_ratio}")
    logger.info(f"- 并行工作进程数: {args.num_workers}")
    logger.info(f"- 内存使用上限: {args.memory_limit_gb}GB")
    logger.info(f"- 批处理大小: {args.batch_size}")
    logger.info(f"- 分块大小: {args.chunk_size}")
    logger.info(f"- 测试模式: {'是' if args.test_mode else '否'}")
    if args.test_mode:
        logger.info(f"- 最多处理文件数: {args.max_test_files}")
    logger.info(f"- 使用缓存: {'是' if args.use_cache else '否'}")
    logger.info(f"- 缓存目录: {args.cache_dir if args.cache_dir else '输入目录下的graph_cache'}")
    logger.info(f"- GPU设备: {args.gpu_device if args.gpu_device else '全部'}")

    try:
        # 如果缓存目录未指定，设置默认路径
        if args.cache_dir is None and args.use_cache:
            args.cache_dir = os.path.join(os.path.dirname(args.input), "graph_cache")
            logger.info(f"graph_cache_dir is :{args.cache_dir}")
        # 调用主处理函数
        filtered_sequences, filtered_graphs = process_sequences_and_graphs(
            input_path=args.input,
            output_dir=args.output_dir,
            identity_threshold=args.identity_threshold,
            coverage_threshold=args.coverage_threshold,
            n_clusters=args.n_clusters,
            sample_ratio=args.sample_ratio,
            use_clustering=args.use_clustering,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            memory_limit_gb=args.memory_limit_gb,
            batch_size=args.batch_size,
            num_perm=args.num_perm,
            test_mode=args.test_mode,
            max_test_files=args.max_test_files,
            use_cache=args.use_cache,
            cache_dir=args.cache_dir
        )

        logger.info("处理完成!")

    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")
        logger.error(traceback.format_exc())

    # 再次记录系统资源状态
    log_system_resources()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if 'logger' in globals():
            logger.exception(f"程序执行时出现错误: {str(e)}")
        print(f"错误: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)