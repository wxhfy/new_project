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
    """从序列中生成k-mers"""
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def process_sequence_batch(batch_data, identity_threshold=0.6, coverage_threshold=0.8, num_perm=128):
    """处理单批次序列的函数，用于并行处理"""
    batch_id, batch_seqs = batch_data

    # 提取序列ID和实际序列
    seq_items = [(seq_id, seq_data['sequence']) for seq_id, seq_data in batch_seqs.items()]

    # 按序列长度降序排列
    seq_items.sort(key=lambda x: -len(x[1]))

    # 创建LSH索引
    lsh = MinHashLSH(threshold=identity_threshold, num_perm=num_perm)
    retained_ids = []

    # 对每个序列创建MinHash签名并进行筛选
    for seq_id, seq in seq_items:
        # 创建MinHash签名
        m = MinHash(num_perm=num_perm)

        # 生成k-mers
        k_mers = create_k_mers(seq, k=3)
        for k_mer in k_mers:
            m.update(k_mer.encode('utf8'))

        # 查询相似序列
        similar_seqs = lsh.query(m)

        # 如果没有相似序列，保留此序列
        if not similar_seqs:
            retained_ids.append(seq_id)
            lsh.insert(seq_id, m)
        else:
            # 验证覆盖率
            seq_len = len(seq)
            all_similar = True

            for similar_id in similar_seqs:
                # 找到相似序列
                for s_id, s_seq in seq_items:
                    if s_id == similar_id:
                        similar_len = len(s_seq)
                        coverage = min(seq_len, similar_len) / max(seq_len, similar_len)

                        # 如果覆盖率低于阈值，不认为是冗余
                        if coverage < coverage_threshold:
                            all_similar = False
                            break

                if not all_similar:
                    break

            if not all_similar:
                retained_ids.append(seq_id)
                lsh.insert(seq_id, m)

    return batch_id, retained_ids


def distributed_minhash_filter(sequences, identity_threshold=0.6, coverage_threshold=0.8,
                               num_perm=128, num_workers=128, batch_size=500000, output_dir="."):
    """
    使用分布式处理的MinHash+LSH算法

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
                    # 构建过滤后的序列字典
                    filtered_sequences = {seq_id: sequences[seq_id] for seq_id in all_retained_ids if
                                          seq_id in sequences}
                    logger.info(f"成功加载已完成的序列去冗余结果: {len(filtered_sequences)}/{len(sequences)} 个序列")
                    return filtered_sequences, list(all_retained_ids)
        except Exception as e:
            logger.warning(f"读取序列检查点失败: {str(e)}，将从头开始处理")

    logger.info(f"使用分布式MinHash+LSH算法对 {len(sequences)} 个序列进行去冗余 (使用{num_workers}个核心)...")
    start_time = time.time()

    # 将序列按长度分组
    logger.info("按长度范围对序列分组...")
    length_groups = defaultdict(dict)

    for seq_id, seq_data in tqdm(sequences.items(), desc="按长度分组"):
        seq = seq_data['sequence']
        length = len(seq)
        length_range = length // 5  # 每5个氨基酸为一组
        length_groups[length_range][seq_id] = seq_data

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

    # 创建任务列表 - 每个长度组可能被拆分为多个批次
    tasks = []
    for length_range, group_seqs in sorted(length_groups.items()):
        # 跳过已处理的组
        if str(length_range) in processed_groups:
            logger.info(f"跳过已处理的长度组 {length_range}")
            continue

        if len(group_seqs) <= batch_size:
            # 小型组作为单个任务
            tasks.append((f"{length_range}", group_seqs))
        else:
            # 大型组拆分成多个任务
            seq_ids = list(group_seqs.keys())
            for i in range(0, len(seq_ids), batch_size):
                batch_ids = seq_ids[i:i + batch_size]
                batch_seqs = {seq_id: group_seqs[seq_id] for seq_id in batch_ids}
                tasks.append((f"{length_range}_{i // batch_size}", batch_seqs))

    logger.info(f"序列去冗余任务拆分为 {len(tasks)} 个批次，开始并行处理")

    # 并行处理批次
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_sequence_batch, task, identity_threshold, coverage_threshold, num_perm): task[0]
            for task in tasks}

        # 实时处理结果
        with tqdm(total=len(tasks), desc="处理序列批次", ncols=100) as pbar:
            for future in concurrent.futures.as_completed(futures):
                batch_id = futures[future]
                try:
                    _, retained_ids = future.result()
                    all_retained_ids.update(retained_ids)

                    # 标记组为已处理
                    group_id = batch_id.split('_')[0]
                    processed_groups.add(group_id)

                    # 更新进度
                    pbar.update(1)
                    pbar.set_postfix({
                        "已保留序列": len(all_retained_ids),
                        "remain%": f"{len(all_retained_ids) * 100 / len(sequences):.1f}%"
                    })

                    # 定期保存检查点
                    if len(all_retained_ids) % 100000 == 0 or len(all_retained_ids) == 0:
                        with open(checkpoint_file, 'w') as f:
                            checkpoint_data = {
                                "retained_ids": list(all_retained_ids),
                                "processed_groups": list(processed_groups)
                            }
                            json.dump(checkpoint_data, f)
                            logger.info(f"保存序列去冗余检查点: {len(all_retained_ids)}个保留序列")

                except Exception as e:
                    logger.error(f"处理批次 {batch_id} 时出错: {str(e)}")

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

    # 构建过滤后的序列字典
    filtered_sequences = {seq_id: sequences[seq_id] for seq_id in all_retained_ids if seq_id in sequences}

    # 保存过滤结果到 pickle 文件
    with open(seq_checkpoint_file, 'wb') as f:
        pickle.dump(filtered_sequences, f)

    logger.info(f"序列去冗余结果已保存到 {seq_checkpoint_file}")

    elapsed = time.time() - start_time
    logger.info(f"序列去冗余完成: 从 {len(sequences)} 个序列中保留 {len(filtered_sequences)} 个 "
                f"({len(filtered_sequences) * 100 / len(sequences):.1f}%)，耗时: {elapsed / 60:.1f}分钟")

    return filtered_sequences, list(all_retained_ids)


def calculate_amp_features(sequence):
    """
    计算与AMPs功能相关的序列特征向量

    参数:
        sequence: 氨基酸序列字符串

    返回:
        特征向量字典
    """
    if not sequence:
        return None

    # 统计氨基酸频率和功能组成
    aa_freq = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        aa_freq[aa] = sequence.count(aa) / len(sequence)

    # 计算功能基团比例
    func_groups = {}
    for group, aas in AMP_FUNCTIONAL_GROUPS.items():
        count = sum(sequence.count(aa) for aa in aas)
        func_groups[group] = count / len(sequence)

    # 计算物理化学特性
    hydrophobicity = 0
    net_charge = 0
    amphipathicity = 0  # 两亲性估计
    helix_propensity = 0

    for aa in sequence:
        if aa in AA_PROPERTIES:
            props = AA_PROPERTIES[aa]
            hydrophobicity += props['hydropathy']
            net_charge += props['charge']
            helix_propensity += props.get('helix', 0)

    # 归一化
    hydrophobicity /= len(sequence)
    helix_propensity /= len(sequence)

    # 计算两亲性 (通过窗口内疏水残基分布)
    window_size = min(18, len(sequence))
    max_amphipathicity = 0

    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        hydro_moment = 0
        for j, aa in enumerate(window):
            if aa in AA_PROPERTIES:
                # 计算螺旋周期性疏水力矩
                angle = j * 100 * np.pi / 180  # 每个残基旋转100度
                hydro_moment += AA_PROPERTIES[aa]['hydropathy'] * np.cos(angle)
        amphipathicity = max(amphipathicity, abs(hydro_moment) / window_size)

    # 计算氨基酸二联体频率 (捕捉短模式)
    dipeptide_freq = {}
    for i in range(len(sequence) - 1):
        dipep = sequence[i:i + 2]
        if dipep not in dipeptide_freq:
            dipeptide_freq[dipep] = 0
        dipeptide_freq[dipep] += 1

    # 只保留最常见的二联体模式
    top_dipeptides = {}
    for dp, count in sorted(dipeptide_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        top_dipeptides[dp] = count / (len(sequence) - 1)

    # 构建特征字典
    features = {
        'length': len(sequence),
        'hydrophobicity': hydrophobicity,
        'net_charge': net_charge,
        'amphipathicity': amphipathicity,
        'helix_propensity': helix_propensity,
        'aa_freqs': aa_freq,
        'func_groups': func_groups,
        'top_dipeptides': top_dipeptides,
    }

    return features


def create_amp_feature_vector(features):
    """将AMPs特征字典转换为数值向量，用于聚类"""
    if not features:
        return np.zeros(30)  # 默认空向量

    # 创建基础特征向量
    feature_vector = [
        features['length'],
        features['hydrophobicity'],
        features['net_charge'],
        features['amphipathicity'],
        features['helix_propensity'],
    ]

    # 添加氨基酸频率
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        feature_vector.append(features['aa_freqs'].get(aa, 0))

    # 添加功能基团频率
    for group in AMP_FUNCTIONAL_GROUPS.keys():
        feature_vector.append(features['func_groups'].get(group, 0))

    # 添加电荷密度(每单位长度的电荷)
    feature_vector.append(features['net_charge'] / features['length'] if features['length'] > 0 else 0)

    # 添加疏水性结构指标(疏水性与两亲性的组合)
    feature_vector.append(features['hydrophobicity'] * features['amphipathicity'])

    return np.array(feature_vector)


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
    """并行生成所有序列的AMPs特征"""
    logger.info(f"为 {len(sequences)} 个序列并行生成AMPs特征...")

    # 将序列分成批次
    seq_ids = list(sequences.keys())
    batches = []
    for i in range(0, len(seq_ids), batch_size):
        batch_ids = seq_ids[i:i + batch_size]
        batch_data = {seq_id: sequences[seq_id] for seq_id in batch_ids}
        batches.append(batch_data)

    logger.info(f"序列特征计算分为 {len(batches)} 个批次")

    # 并行处理
    all_features = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_amp_features_batch, batch) for batch in batches]

        with tqdm(total=len(batches), desc="生成AMPs特征", ncols=100) as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    features_batch = future.result()
                    all_features.update(features_batch)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"处理特征批次时出错: {str(e)}")

    logger.info(f"成功为 {len(all_features)} 个序列生成AMPs特征")
    return all_features


def amp_optimized_clustering(sequences, features, select_features=None, feature_weights=None,
                             n_clusters=None, sample_ratio=0.6, diversity_ratio=0.7,
                             num_workers=32, output_dir=None, pca_vis=False):
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

        logger.info("自动评估最佳簇数...")
        for n in range_clusters:
            # 为了效率，在小样本上评估
            sample_size = min(5000, X_scaled.shape[0])
            if X_scaled.shape[0] > sample_size:
                indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
                X_sample = X_scaled[indices]
            else:
                X_sample = X_scaled

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
    retained_ids = []
    retained_info = {}  # 记录选择信息

    logger.info("从每个簇中选择代表性样本...")
    with tqdm(total=len(unique_labels), desc="处理簇") as pbar:
        for cluster_idx, (label, cluster_size) in enumerate(cluster_sizes):
            # 获取当前簇的索引
            cluster_indices = np.where(labels == label)[0]
            cluster_seq_ids = [seq_ids[i] for i in cluster_indices]

            # 计算采样数量
            sample_count = max(1, int(cluster_size * sample_ratio))

            # 对于小簇，保留所有样本
            if cluster_size <= 5:
                retained_ids.extend(cluster_seq_ids)
                retained_info[f"cluster_{label}"] = {
                    "size": cluster_size,
                    "selected": cluster_size,
                    "method": "all_kept"
                }
                pbar.update(1)
                continue

            # 获取簇的特征向量
            cluster_vectors = X_scaled[cluster_indices]

            # 计算簇的中心
            cluster_center = np.mean(cluster_vectors, axis=0)

            # 计算到中心的距离
            distances_to_center = np.linalg.norm(cluster_vectors - cluster_center, axis=1)

            # 分配中心样本和多样性样本的数量
            center_samples = max(1, int(sample_count * (1 - diversity_ratio)))
            diverse_samples = sample_count - center_samples

            # 选择最接近中心的样本
            center_indices = np.argsort(distances_to_center)[:center_samples]
            center_ids = [cluster_seq_ids[i] for i in center_indices]
            retained_ids.extend(center_ids)

            # 如果还需要选择多样性样本
            if diverse_samples > 0:
                # 排除已选的中心样本
                remaining_indices = np.setdiff1d(np.arange(len(cluster_indices)), center_indices)

                if len(remaining_indices) > 0:
                    remaining_local_indices = [cluster_indices[i] for i in remaining_indices]
                    remaining_vectors = X_scaled[remaining_local_indices]
                    remaining_seq_ids = [cluster_seq_ids[i] for i in remaining_indices]

                    # 使用最大化距离(MaxMin)策略选择多样性样本
                    selected_diverse = []
                    selected_indices = []

                    # 从已选的中心样本开始
                    selected_vectors = X_scaled[[cluster_indices[i] for i in center_indices]]

                    # 迭代选择最远的样本
                    for _ in range(diverse_samples):
                        if not remaining_vectors.size or not remaining_seq_ids:
                            break

                        # 计算每个剩余点到已选点集的最小距离
                        min_distances = np.full(len(remaining_vectors), np.inf)

                        for i, vec in enumerate(remaining_vectors):
                            # 计算到所有已选点的距离
                            dists = np.linalg.norm(selected_vectors - vec, axis=1)
                            # 取最小距离
                            if dists.size > 0:
                                min_distances[i] = np.min(dists)

                        # 选择具有最大最小距离的点
                        if min_distances.size > 0 and not np.all(np.isinf(min_distances)):
                            best_idx = np.argmax(min_distances)
                            selected_diverse.append(remaining_seq_ids[best_idx])
                            selected_indices.append(best_idx)
                            selected_vectors = np.vstack([selected_vectors, remaining_vectors[best_idx]])

                            # 移除已选的点
                            remaining_vectors = np.delete(remaining_vectors, best_idx, axis=0)
                            remaining_seq_ids.pop(best_idx)
                        else:
                            break

                    retained_ids.extend(selected_diverse)

            # 记录选择结果
            retained_info[f"cluster_{label}"] = {
                "size": cluster_size,
                "selected": center_samples + diverse_samples,
                "center_samples": center_samples,
                "diverse_samples": diverse_samples,
                "method": "hybrid"
            }

            pbar.update(1)

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
    local_id_to_files = {}
    local_matched_files = set()

    for file_path, file_ids in chunk.items():
        # 计算与目标ID集合的交集
        common_ids = remaining_ids.intersection(file_ids)
        if common_ids:  # 如果有交集，则此文件需要加载
            local_matched_files.add(file_path)
            for graph_id in common_ids:
                if graph_id not in local_id_to_files:
                    local_id_to_files[graph_id] = []
                local_id_to_files[graph_id].append(file_path)

    return local_id_to_files, local_matched_files


def check_pt_file_content(file_path):
    """检查PT文件包含的图谱ID（禁用mmap）"""
    try:
        # 显式禁用内存映射加载
        data = safe_load_graph(file_path, map_location='cpu')
        if isinstance(data, dict):
            file_ids = set(data.keys())
            return file_path, file_ids
    except Exception as e:
        logger.debug(f"检查文件 {os.path.basename(file_path)} 失败: {str(e)}")
    return file_path, set()


def process_file_batch(batch_files, target_ids, id_mapping):
    """处理文件批次，加载包含目标ID的图谱（禁用mmap）"""
    batch_start_time = time.time()
    batch_graphs = {}
    processed_count = 0

    # 处理每个文件
    for file_path in batch_files:
        try:
            # 检查此文件是否包含目标ID
            file_ids = id_mapping.get(file_path, set())
            common_ids = set(target_ids) & file_ids

            if not common_ids:
                continue

            # 只加载包含目标ID的图谱
            graphs_data = safe_load_graph(file_path, map_location='cpu')

            if not graphs_data or not isinstance(graphs_data, dict):
                logger.warning(f"文件格式不正确或为空: {os.path.basename(file_path)}")
                continue

            # 只提取所需的ID
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

    # 直接返回三个值
    return batch_graphs, processed_count, batch_time

def load_cached_graphs(cache_dir, cache_id, selected_ids_set):
    """
    超高速缓存加载器 - 无mmap版本，避免RLock错误
    """
    cached_graphs = {}
    cache_meta_file = os.path.join(cache_dir, f"{cache_id}_meta.json")

    if not os.path.exists(cache_meta_file):
        return cached_graphs

    try:
        # 查找所有缓存文件并按修改时间排序
        cache_files = glob.glob(os.path.join(cache_dir, f"{cache_id}_part_*.pt"))
        if not cache_files:
            return cached_graphs

        cache_files.sort(key=os.path.getmtime, reverse=True)
        logger.info(f"发现 {len(cache_files)} 个图谱缓存文件")

        # 将文件分成批次处理以平衡内存使用
        batch_size = max(1, len(cache_files) // 4)
        batches = [cache_files[i:i + batch_size] for i in range(0, len(cache_files), batch_size)]

        total_loaded = 0
        with tqdm(total=len(cache_files), desc="加载缓存文件", ncols=100) as pbar:
            # 处理每个批次
            for batch_idx, batch_files in enumerate(batches):
                # 统计当前批次加载信息
                batch_loaded = 0

                # 顺序处理当前批次的文件，避免使用线程池
                for file_path in batch_files:
                    try:
                        # 加载文件内容到内存，避免内存映射
                        with open(file_path, 'rb') as f:
                            file_content = f.read()
                            if not file_content:
                                pbar.update(1)
                                continue

                            # 使用BytesIO避免内存映射问题
                            buffer = io.BytesIO(file_content)
                            file_data = torch.load(buffer, map_location='cpu')
                            buffer.close()

                        # 保存当前文件的数据
                        if file_data and isinstance(file_data, dict):
                            # 仅保留目标ID的图谱
                            for graph_id, graph in file_data.items():
                                if graph_id in selected_ids_set and graph_id not in cached_graphs:
                                    # 确保对象没有锁
                                    if isinstance(graph, Data):
                                        # 将tensor移到CPU
                                        for key, value in graph:
                                            if hasattr(value, 'is_cuda') and value.is_cuda:
                                                graph[key] = value.cpu()

                                    cached_graphs[graph_id] = graph
                                    batch_loaded += 1

                        # 更新进度条
                        pbar.update(1)

                    except Exception as e:
                        logger.debug(f"处理缓存文件失败: {os.path.basename(file_path)}, 错误: {str(e)}")
                        pbar.update(1)

                # 更新总计数
                total_loaded += batch_loaded
                logger.info(f"批次 {batch_idx + 1}/{len(batches)} 完成，此批次加载 {batch_loaded} 个图谱")

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

def accelerated_graph_loader(input_path, filtered_ids, num_workers=128, memory_limit_gb=900,
                             use_cache=True, cache_dir=None, cache_id="graph_data_cache", use_mmap=False,
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

        # 生成序列特征
        amp_features = parallel_generate_amp_features(
            filtered_sequences,
            num_workers=num_workers,
            batch_size=min(5000, len(filtered_sequences))
        )

        # 执行聚类
        cluster_filtered_ids = amp_optimized_clustering(
            filtered_sequences,
            amp_features,
            n_clusters=n_clusters,
            sample_ratio=sample_ratio,
            num_workers=num_workers,
            output_dir=output_dir
        )

        # 更新过滤后的序列
        old_count = len(filtered_sequences)
        filtered_sequences = {seq_id: filtered_sequences[seq_id] for seq_id in cluster_filtered_ids if
                              seq_id in filtered_sequences}
        filtered_ids = list(filtered_sequences.keys())
        logger.info(f"聚类分析完成，保留 {len(filtered_sequences)}/{old_count} 个序列")

        # 清理内存
        del amp_features
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
    parser.add_argument("--max_test_files", "-mtf", type=int, default=5,
                        help="测试模式下最多处理的文件数 (默认: 5)")

    # 缓存参数
    parser.add_argument("--use_cache", action="store_true", default=False,
                        help="使用缓存加载图谱 (默认: True)")
    parser.add_argument("--cache_dir", "-cd", type=str, default=None,
                        help="图谱缓存目录，默认为输入目录下的graph_cache")

    # GPU参数
    parser.add_argument("--gpu_device", "-g", type=str, default=3, help="指定GPU设备ID，例如'0,1'，默认使用所有可用GPU")

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