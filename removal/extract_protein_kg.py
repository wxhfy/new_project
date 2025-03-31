#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质结构解析与知识图谱构建工具 (GPU优化版)

针对大规模蛋白质结构文件处理和图谱构建的高性能实现。
支持CPU/GPU混合计算，优化了内存使用和并行效率。

作者: wxhfy (优化版)
"""
import gc
import argparse
import concurrent.futures
import gzip
import json
import multiprocessing
import os
import random
import shutil
import logging
import time
import sys
import platform
import traceback
import uuid
import psutil
from collections import defaultdict
from io import StringIO
from queue import Queue
import pickle

# 核心依赖
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
# 使用更快的cKDTree替代KDTree
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# 生物信息学库
import mdtraj as md
from Bio import PDB, SeqIO, AlignIO, Align
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# 图数据处理
from torch_geometric.data import Data

# 配置是否使用GPU
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    # 设置GPU内存增长策略
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

import warnings

warnings.filterwarnings("ignore", message="Unlikely unit cell vectors")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================= 常量与配置 =======================

# 设置基本日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 氨基酸物理化学性质常量
# 转换为NumPy构造化数组提高访问效率
AA_PROPERTIES = {
    'A': {'hydropathy': 1.8, 'charge': 0, 'polar': False, 'mw': 89.09, 'volume': 88.6, 'flexibility': 0.36,
          'aromatic': False},
    'C': {'hydropathy': 2.5, 'charge': 0, 'polar': False, 'mw': 121.15, 'volume': 108.5, 'flexibility': 0.35,
          'aromatic': False},
    'D': {'hydropathy': -3.5, 'charge': -1, 'polar': True, 'mw': 133.10, 'volume': 111.1, 'flexibility': 0.51,
          'aromatic': False},
    'E': {'hydropathy': -3.5, 'charge': -1, 'polar': True, 'mw': 147.13, 'volume': 138.4, 'flexibility': 0.50,
          'aromatic': False},
    'F': {'hydropathy': 2.8, 'charge': 0, 'polar': False, 'mw': 165.19, 'volume': 189.9, 'flexibility': 0.31,
          'aromatic': True},
    'G': {'hydropathy': -0.4, 'charge': 0, 'polar': False, 'mw': 75.07, 'volume': 60.1, 'flexibility': 0.54,
          'aromatic': False},
    'H': {'hydropathy': -3.2, 'charge': 0.1, 'polar': True, 'mw': 155.16, 'volume': 153.2, 'flexibility': 0.32,
          'aromatic': True},
    'I': {'hydropathy': 4.5, 'charge': 0, 'polar': False, 'mw': 131.17, 'volume': 166.7, 'flexibility': 0.30,
          'aromatic': False},
    'K': {'hydropathy': -3.9, 'charge': 1, 'polar': True, 'mw': 146.19, 'volume': 168.6, 'flexibility': 0.47,
          'aromatic': False},
    'L': {'hydropathy': 3.8, 'charge': 0, 'polar': False, 'mw': 131.17, 'volume': 166.7, 'flexibility': 0.40,
          'aromatic': False},
    'M': {'hydropathy': 1.9, 'charge': 0, 'polar': False, 'mw': 149.21, 'volume': 162.9, 'flexibility': 0.41,
          'aromatic': False},
    'N': {'hydropathy': -3.5, 'charge': 0, 'polar': True, 'mw': 132.12, 'volume': 114.1, 'flexibility': 0.46,
          'aromatic': False},
    'P': {'hydropathy': -1.6, 'charge': 0, 'polar': False, 'mw': 115.13, 'volume': 112.7, 'flexibility': 0.51,
          'aromatic': False},
    'Q': {'hydropathy': -3.5, 'charge': 0, 'polar': True, 'mw': 146.15, 'volume': 143.8, 'flexibility': 0.49,
          'aromatic': False},
    'R': {'hydropathy': -4.5, 'charge': 1, 'polar': True, 'mw': 174.20, 'volume': 173.4, 'flexibility': 0.47,
          'aromatic': False},
    'S': {'hydropathy': -0.8, 'charge': 0, 'polar': True, 'mw': 105.09, 'volume': 89.0, 'flexibility': 0.51,
          'aromatic': False},
    'T': {'hydropathy': -0.7, 'charge': 0, 'polar': True, 'mw': 119.12, 'volume': 116.1, 'flexibility': 0.44,
          'aromatic': False},
    'V': {'hydropathy': 4.2, 'charge': 0, 'polar': False, 'mw': 117.15, 'volume': 140.0, 'flexibility': 0.34,
          'aromatic': False},
    'W': {'hydropathy': -0.9, 'charge': 0, 'polar': True, 'mw': 204.23, 'volume': 227.8, 'flexibility': 0.31,
          'aromatic': True},
    'Y': {'hydropathy': -1.3, 'charge': 0, 'polar': True, 'mw': 181.19, 'volume': 193.6, 'flexibility': 0.42,
          'aromatic': True},
    'X': {'hydropathy': 0.0, 'charge': 0, 'polar': False, 'mw': 110.0, 'volume': 135.0, 'flexibility': 0.45,
          'aromatic': False}
}

# 预计算氨基酸属性向量，加速访问
AA_PROPERTY_VECTORS = {}
for aa, props in AA_PROPERTIES.items():
    AA_PROPERTY_VECTORS[aa] = np.array([
        props['hydropathy'],
        props['charge'],
        props['mw'] / 200.0,  # 归一化
        props['volume'] / 200.0,  # 归一化
        props['flexibility'],
        1.0 if props['aromatic'] else 0.0
    ], dtype=np.float32)  # 使用float32减少内存占用

# 预计算氨基酸集合，加速检查
HBOND_DONORS = set('NQRKWST')
HBOND_ACCEPTORS = set('DEQNSTYHW')
HYDROPHOBIC_AA = set('AVILMFYW')
POSITIVE_AA = set('KRH')
NEGATIVE_AA = set('DE')


# BLOSUM62矩阵 - 优化为NumPy数组
def create_blosum62_matrix():
    # 标准氨基酸顺序
    standard_aa = "ARNDCQEGHILKMFPSTWYV"
    aa_to_idx = {aa: i for i, aa in enumerate(standard_aa)}

    # BLOSUM62矩阵数据
    blosum_data = np.array([
        [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
        [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
        [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
        [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
        [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
        [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
        [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
        [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
        [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
        [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
        [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
        [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
        [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
        [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
        [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
        [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
        [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
        [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]  # V
    ], dtype=np.int8)  # 使用int8存储，减少内存占用

    return standard_aa, aa_to_idx, blosum_data


# 预计算BLOSUM62矩阵
STANDARD_AA, AA_TO_IDX, BLOSUM62_MATRIX = create_blosum62_matrix()

# 二级结构映射（MDTraj到标准8类）
SS_MAPPING = {
    'H': 'H',  # α-helix
    'G': 'G',  # 3-10-helix
    'I': 'I',  # π-helix
    'E': 'E',  # β-strand
    'B': 'B',  # β-bridge
    'T': 'T',  # Turn
    'S': 'S',  # Bend
    'C': 'C',  # Coil
    '-': 'C',  # Undefined (映射到Coil)
    ' ': 'C'  # 空格 (映射到Coil)
}

# 二级结构类型映射到索引 (加速one-hot编码)
SS_TYPE_TO_IDX = {
    'H': 0,  # alpha-helix
    'G': 0,  # 3-10-helix (映射到alpha类别)
    'I': 0,  # pi-helix (映射到alpha类别)
    'E': 1,  # beta-strand
    'B': 1,  # beta-bridge (映射到beta类别)
    'T': 2,  # turn (映射到coil类别)
    'S': 2,  # bend (映射到coil类别)
    'C': 2,  # coil
}


# ======================= 工具函数 =======================

def setup_logging(output_dir):
    """设置日志系统：控制台显示高级摘要，文件保存完整处理细节"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"extraction_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 捕获所有级别的日志

    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 添加控制台处理器 - 仅显示INFO及以上级别
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 过滤器 - 排除处理文件的详细信息
    class ProcessingFilter(logging.Filter):
        def filter(self, record):
            # 排除特定处理消息
            return not any(msg in record.getMessage() for msg in
                           ["解析文件", "处理文件", "添加节点", "处理残基", "跳过"])

    console.addFilter(ProcessingFilter())
    root_logger.addHandler(console)

    # 添加文件处理器 - 记录所有级别（包括DEBUG）
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    # 记录系统信息
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"运行平台: {platform.platform()}")

    if USE_GPU:
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU #{i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"使用GPU加速计算")
    else:
        logger.info("未检测到GPU或CUDA不可用，使用CPU计算")

    return root_logger, log_file


def check_memory_usage(threshold_gb=None, force_gc=False):
    """高效检查并管理内存使用情况"""
    try:
        # 获取当前进程
        process = psutil.Process()

        # 获取当前内存使用量
        mem_used_bytes = process.memory_info().rss
        mem_used_gb = mem_used_bytes / (1024 ** 3)  # 转换为GB

        # 确定阈值，如果未指定则使用系统内存的80%
        if threshold_gb is None:
            system_mem = psutil.virtual_memory()
            total_mem_gb = system_mem.total / (1024 ** 3)
            threshold_gb = total_mem_gb * 0.8

        # 检查是否超过阈值或强制执行
        if mem_used_gb > threshold_gb or force_gc:
            logger.info(f"内存使用达到 {mem_used_gb:.2f} GB，执行垃圾回收")

            # 先释放GPU内存
            if USE_GPU:
                torch.cuda.empty_cache()

            # 执行主动垃圾回收
            before_gc = mem_used_gb
            gc.collect()

            # 检查回收效果
            mem_after_gc = process.memory_info().rss / (1024 ** 3)
            logger.info(f"垃圾回收后内存使用: {mem_after_gc:.2f} GB (释放了 {max(0, before_gc - mem_after_gc):.2f} GB)")

            return True
        return False
    except Exception as e:
        logger.warning(f"检查内存使用时出错: {str(e)[:100]}")
        # 预防性执行垃圾回收
        gc.collect()
        if USE_GPU:
            torch.cuda.empty_cache()
        return False


def create_dynamic_worker_pool(n_workers, file_list, chunk_min_size=100):
    """
    实现动态任务分配工作池，基于文件大小均衡负载

    优化点:
    - 考虑文件大小进行负载均衡分配
    - 大文件单独处理，小文件批量分配
    - 自适应分块大小
    """
    task_queue = Queue()

    # 根据文件大小排序
    try:
        # 对文件按大小分组：测量文件大小，跳过不存在的文件
        file_sizes = []
        for f in file_list:
            try:
                if os.path.exists(f):
                    file_sizes.append((f, os.path.getsize(f)))
                else:
                    logger.debug(f"文件不存在，跳过: {f}")
            except Exception as e:
                logger.debug(f"获取文件大小失败，使用默认大小: {f}, 错误: {str(e)[:50]}")
                file_sizes.append((f, 1000000))  # 默认1MB

        # 对文件按大小降序排序
        file_sizes.sort(key=lambda x: x[1], reverse=True)

        # 贪心算法分配文件到任务组
        groups = [[] for _ in range(n_workers)]
        group_sizes = [0] * n_workers

        # 分配超大文件 (每个一个任务)
        very_large_files = [(f, size) for f, size in file_sizes if size > 50 * 1024 * 1024]  # >50MB

        # 提前处理超大文件
        for f, size in very_large_files:
            min_idx = group_sizes.index(min(group_sizes))
            groups[min_idx].append(f)
            group_sizes[min_idx] += size

        # 分配剩余文件
        remaining_files = [(f, size) for f, size in file_sizes if size <= 50 * 1024 * 1024]

        # 继续贪心分配
        for f, size in remaining_files:
            min_idx = group_sizes.index(min(group_sizes))
            groups[min_idx].append(f)
            group_sizes[min_idx] += size

        # 将分组添加到队列
        for i, group in enumerate(groups):
            if group:  # 确保组不为空
                task_queue.put(group)
                logger.debug(f"创建任务组 {i + 1}: {len(group)} 文件, 总大小: {group_sizes[i] / 1024 / 1024:.1f}MB")

    except Exception as e:
        logger.warning(f"创建动态工作池失败: {str(e)[:100]}，回退到简单分块")
        # 回退到简单分块
        chunk_size = max(1, len(file_list) // n_workers)
        for i in range(0, len(file_list), chunk_size):
            chunk = file_list[i:i + chunk_size]
            if chunk:  # 确保块不为空
                task_queue.put(chunk)

    return task_queue, n_workers


def three_to_one(residue_name):
    """优化的氨基酸三字母到一字母代码转换"""
    # 使用查找表加速常见氨基酸转换
    three_to_one_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        # 添加常见的修饰氨基酸
        'MSE': 'M', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
        'SEP': 'S', 'TPO': 'T', 'PTR': 'Y', 'MLY': 'K',
        'CSO': 'C', 'CAS': 'C'
    }

    try:
        # 尝试直接映射（最常见情况，速度最快）
        upper_name = residue_name.upper()
        if upper_name in three_to_one_map:
            return three_to_one_map[upper_name]

        # 调用Bio.SeqUtils.seq1作为后备
        return seq1(residue_name)
    except Exception:
        return 'X'  # 未知氨基酸


def get_blosum62_encoding(residue):
    """
    优化的BLOSUM62编码获取函数
    直接使用预计算的NumPy数组，避免字典查找
    """
    try:
        idx = AA_TO_IDX.get(residue, -1)
        if idx >= 0:
            return BLOSUM62_MATRIX[idx].copy()
        else:
            return np.zeros(20, dtype=np.int8)  # 非标准氨基酸返回零向量
    except:
        return np.zeros(20, dtype=np.int8)  # 出错时返回零向量


def normalize_coordinates(coords_array):
    """
    高效标准化坐标：中心化到质心并归一化

    使用NumPy广播加速计算，优化内存使用
    """
    if len(coords_array) == 0:
        return coords_array

    # 转换为float32减少内存占用
    coords = np.asarray(coords_array, dtype=np.float32)

    # 计算质心
    centroid = np.mean(coords, axis=0)

    # 中心化坐标（使用广播）
    centered_coords = coords - centroid

    # 计算到质心的最大距离
    max_dist = np.max(np.sqrt(np.sum(centered_coords ** 2, axis=1)))

    # 避免除零错误
    if max_dist > 1e-6:
        # 归一化（使用广播）
        normalized_coords = centered_coords / max_dist
    else:
        normalized_coords = centered_coords

    return normalized_coords


def gpu_compute_residue_distances(ca_coords):
    """
    使用GPU加速计算残基间距离矩阵

    参数:
        ca_coords: Alpha碳原子坐标数组 (N x 3)
    返回:
        distance_matrix: 残基间距离矩阵 (N x N)
    """
    if USE_GPU and len(ca_coords) > 10:  # 对小规模计算使用CPU更有效
        try:
            # 转换为PyTorch张量
            coords_tensor = torch.tensor(ca_coords, dtype=torch.float32, device='cuda')

            # 计算距离矩阵 - 使用广播计算欧氏距离
            # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x·y
            dot_prod = torch.matmul(coords_tensor, coords_tensor.transpose(0, 1))
            sq_norm = torch.sum(coords_tensor ** 2, dim=1, keepdim=True)
            dist_matrix = torch.sqrt(sq_norm + sq_norm.transpose(0, 1) - 2 * dot_prod + 1e-8)

            # 返回NumPy数组
            return dist_matrix.cpu().numpy()
        except Exception as e:
            logger.debug(f"GPU距离计算失败，切换到CPU: {str(e)[:100]}")
            # 回退到CPU计算
            return compute_residue_distances_vectorized(ca_coords)
    else:
        # 使用CPU向量化计算
        return compute_residue_distances_vectorized(ca_coords)


def compute_residue_distances_vectorized(ca_coords):
    """
    优化的CPU版本残基距离矩阵计算
    使用NumPy广播实现向量化操作
    """
    # 使用float32减少内存占用
    coords = np.asarray(ca_coords, dtype=np.float32)

    # 使用广播计算欧氏距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # shape: (n_res, n_res, 3)
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))  # shape: (n_res, n_res)

    return dist_matrix


def classify_interaction_vectorized(res_codes, distance_matrix, threshold=8.0):
    """
    向量化判断残基间相互作用类型

    参数:
        res_codes: 残基类型列表
        distance_matrix: 残基间距离矩阵
        threshold: 相互作用距离阈值

    返回:
        interaction_types: 相互作用类型矩阵 (值: 0-4)
    """
    n_res = len(res_codes)

    # 创建残基类型的特征矩阵
    is_donor = np.array([r in HBOND_DONORS for r in res_codes])
    is_acceptor = np.array([r in HBOND_ACCEPTORS for r in res_codes])
    is_hydrophobic = np.array([r in HYDROPHOBIC_AA for r in res_codes])
    is_positive = np.array([r in POSITIVE_AA for r in res_codes])
    is_negative = np.array([r in NEGATIVE_AA for r in res_codes])

    # 初始化交互矩阵
    interaction_types = np.zeros((n_res, n_res), dtype=np.int8)

    # 1. 空间距离过滤
    spatial_contacts = distance_matrix <= threshold
    interaction_types[spatial_contacts] = 1  # 标记为空间近邻

    # 对角线和相邻位置不考虑空间近邻
    np.fill_diagonal(interaction_types, 0)
    for i in range(n_res - 1):
        interaction_types[i, i + 1] = 0
        interaction_types[i + 1, i] = 0

    # 2. 氢键相互作用 (供体-受体对)
    donors_matrix = is_donor[:, np.newaxis] & is_acceptor[np.newaxis, :]
    acceptors_matrix = is_acceptor[:, np.newaxis] & is_donor[np.newaxis, :]
    potential_hbonds = donors_matrix | acceptors_matrix

    # 应用距离条件 (<5.0Å)
    hbond_contacts = potential_hbonds & (distance_matrix < 5.0)
    interaction_types[hbond_contacts] = 2  # 标记为氢键

    # 3. 疏水相互作用
    hydrophobic_matrix = is_hydrophobic[:, np.newaxis] & is_hydrophobic[np.newaxis, :]
    hydrophobic_contacts = hydrophobic_matrix & (distance_matrix < 6.0)
    interaction_types[hydrophobic_contacts] = 4  # 标记为疏水相互作用

    # 4. 离子相互作用
    pos_neg_matrix = is_positive[:, np.newaxis] & is_negative[np.newaxis, :]
    neg_pos_matrix = is_negative[:, np.newaxis] & is_positive[np.newaxis, :]
    potential_ionic = pos_neg_matrix | neg_pos_matrix

    # 应用距离条件 (<6.0Å)
    ionic_contacts = potential_ionic & (distance_matrix < 6.0)
    interaction_types[ionic_contacts] = 3  # 标记为离子相互作用

    return interaction_types


def build_optimized_spatial_index(coordinates):
    """
    构建高效的空间索引结构
    使用cKDTree替代KDTree，计算速度提升2-10倍
    """
    # 确保坐标为float32减少内存占用
    coords_f32 = np.asarray(coordinates, dtype=np.float32)

    # 优化leafsize参数
    # leafsize越大，构建树越快，查询可能变慢
    # 对蛋白质结构，通常leafsize=16是个好的平衡点
    tree = KDTree(coords_f32, leafsize=16)

    return tree


def batch_nearest_neighbor_query(tree, points, k, max_batch=10000):
    """
    批量查询最近邻，避免一次性大查询导致内存溢出
    """
    n_points = len(points)
    results_dist = np.zeros((n_points, k))
    results_idx = np.zeros((n_points, k), dtype=np.int32)

    # 分批查询
    for i in range(0, n_points, max_batch):
        end = min(i + max_batch, n_points)
        dist, idx = tree.query(points[i:end], k=k)
        results_dist[i:end] = dist
        results_idx[i:end] = idx

    return results_dist, results_idx


def extract_plddt_from_bfactor(structure):
    """
    从B因子提取pLDDT值的优化版本
    使用向量化操作代替循环，显著提高计算速度
    """
    try:
        # 检查结构是否直接有B因子属性
        has_b_factors = hasattr(structure, 'b_factors')

        if has_b_factors:
            # 使用numpy高效聚合B因子
            residue_indices = np.array([atom.residue.index for atom in structure.topology.atoms])
            max_res_idx = np.max(residue_indices)

            # 创建残基B因子累加数组和计数数组
            res_b_factors = np.zeros(max_res_idx + 1, dtype=np.float32)
            res_atom_counts = np.zeros(max_res_idx + 1, dtype=np.int32)

            # 使用numpy的高效聚合操作
            for i, res_idx in enumerate(residue_indices):
                res_b_factors[res_idx] += structure.b_factors[i]
                res_atom_counts[res_idx] += 1

            # 计算每个残基的平均B因子
            # 避免除零错误
            res_atom_counts = np.maximum(res_atom_counts, 1)
            avg_b_factors = res_b_factors / res_atom_counts

            return avg_b_factors
        else:
            # 如果没有B因子，使用默认值
            return np.full(structure.n_residues, 70.0, dtype=np.float32)
    except Exception as e:
        logger.error(f"从B因子提取pLDDT失败: {str(e)[:100]}")
        return np.full(structure.n_residues, 70.0, dtype=np.float32)  # 默认中等置信度


def load_structure_mdtraj(file_path):
    """
    优化的MDTraj结构加载函数
    增加了错误处理和临时文件清理
    """
    try:
        # 检查文件格式
        is_gzipped = file_path.endswith('.gz')
        temp_file = None

        # 处理gzip压缩文件
        if is_gzipped:
            temp_dir = '/dev/shm' if os.path.exists('/dev/shm') else '/tmp'
            temp_file = os.path.join(temp_dir, f"protein_{uuid.uuid4().hex}{os.path.basename(file_path)[:-3]}")

            with gzip.open(file_path, 'rb') as f_in:
                with open(temp_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            file_to_use = temp_file
        else:
            file_to_use = file_path

        # 加载结构
        if file_path.endswith(('.pdb', '.pdb.gz')):
            structure = md.load_pdb(file_to_use)
        elif file_path.endswith(('.cif', '.cif.gz')):
            structure = md.load(file_to_use)
        else:
            structure = md.load(file_to_use)

        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

        # 检查结构有效性
        if structure.n_residues <= 0 or structure.n_atoms <= 0:
            logger.warning(f"加载结构无效 (残基数: {structure.n_residues}, 原子数: {structure.n_atoms}): {file_path}")
            return None

        return structure

    except Exception as e:
        logger.error(f"MDTraj加载结构失败: {str(e)[:100]}")
        if 'temp_file' in locals() and temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return None


def compute_secondary_structure(structure):
    """
    使用MDTraj高效计算二级结构
    包含错误处理和矢量化操作
    """
    try:
        # 使用MDTraj的compute_dssp函数计算二级结构
        ss = md.compute_dssp(structure, simplified=False)

        # 使用numpy的矢量化操作映射为标准的8类
        ss_map_func = np.vectorize(lambda x: SS_MAPPING.get(x, 'C'))
        mapped_ss = ss_map_func(ss)

        return mapped_ss
    except Exception as e:
        logger.error(f"计算二级结构失败: {str(e)[:100]}")
        # 返回全C（coil）作为默认值
        return np.full((structure.n_frames, structure.n_residues), 'C')


def build_enhanced_residue_graph(structure, ss_array, fragment_range,
                                 k_neighbors=8, distance_threshold=8.0, plddt_threshold=70):
    """
    高性能残基知识图谱构建函数

    优化点:
    - GPU加速复杂数值计算
    - 向量化操作代替循环
    - 批处理模式显著减少内存占用
    - 优化数据结构减少冗余计算
    - 增强容错和并行性能

    参数:
        structure: MDTraj结构对象
        ss_array: 二级结构数组
        fragment_range: (start_idx, end_idx, fragment_id)元组
        k_neighbors: K近邻数量
        distance_threshold: 空间接触距离阈值(Å)
        plddt_threshold: pLDDT置信度阈值

    返回:
        NetworkX图对象
    """
    start_idx, end_idx, fragment_id = fragment_range

    # 快速检查片段有效性
    if start_idx < 0 or end_idx > structure.n_residues or start_idx >= end_idx:
        logger.warning(f"无效片段范围: {start_idx}-{end_idx}, 蛋白质残基数: {structure.n_residues}")
        return None

    # 初始化图和统计信息
    graph = nx.Graph()
    debug_stats = {
        'total_residues': end_idx - start_idx,
        'filtered_by_plddt': 0,
        'filtered_by_nonstandard': 0,
        'valid_nodes': 0,
        'edges_created': 0
    }

    try:
        # 1. 预处理 - 一次性获取所有必要数据
        # 获取pLDDT值
        plddt_values = extract_plddt_from_bfactor(structure)

        # 获取CA原子坐标
        ca_indices = []
        ca_coords = []
        residue_codes = []
        valid_residues = []
        valid_indices = []

        # 提取CA坐标和有效残基
        for res_idx in range(start_idx, end_idx):
            try:
                # 跳过低质量残基
                if res_idx < len(plddt_values) and plddt_values[res_idx] < plddt_threshold:
                    debug_stats['filtered_by_plddt'] += 1
                    continue

                # 获取残基信息
                res = structure.topology.residue(res_idx)
                one_letter = three_to_one(res.name)

                if one_letter == 'X':
                    debug_stats['filtered_by_nonstandard'] += 1
                    continue

                # 查找CA原子索引
                ca_idx = None
                for atom in res.atoms:
                    if atom.name == 'CA':
                        ca_idx = atom.index
                        break

                if ca_idx is not None and ca_idx < len(structure.xyz[0]):
                    ca_indices.append(ca_idx)
                    ca_coords.append(structure.xyz[0, ca_idx])
                    residue_codes.append(one_letter)
                    valid_residues.append(res)
                    valid_indices.append(res_idx)
            except Exception as e:
                logger.debug(f"残基处理错误 {res_idx}: {str(e)[:50]}")
                continue

        # 检查有效节点数量
        if len(valid_indices) < 2:
            logger.warning(f"有效节点数量不足 ({len(valid_indices)}): {fragment_id}")
            return None

        # 2. 处理坐标数据
        ca_coords = np.array(ca_coords, dtype=np.float32)

        # 标准化坐标
        normalized_coords = normalize_coordinates(ca_coords)

        # 3. 批量构建节点 - 使用向量化操作
        node_ids = [f"res_{idx}" for idx in valid_indices]

        # 4. 计算节点特征 - 高效向量化
        for i, res_idx in enumerate(valid_indices):
            try:
                node_id = node_ids[i]
                one_letter = residue_codes[i]
                position = normalized_coords[i].tolist()

                # 获取二级结构
                try:
                    ss_code = ss_array[0, res_idx] if ss_array.ndim > 1 else ss_array[res_idx]
                except (IndexError, TypeError):
                    return None

                # 高效计算二级结构one-hot
                ss_idx = SS_TYPE_TO_IDX.get(ss_code, 2)  # 默认为coil(2)
                ss_onehot = [0, 0, 0]
                ss_onehot[ss_idx] = 1

                # 获取BLOSUM编码和AA属性
                blosum = get_blosum62_encoding(one_letter)
                prop_vector = AA_PROPERTY_VECTORS.get(one_letter, AA_PROPERTY_VECTORS['X'])

                # 添加节点
                node_attrs = {
                    'residue_name': valid_residues[i].name,
                    'residue_code': one_letter,
                    'residue_idx': res_idx,
                    'position': position,
                    'plddt': float(plddt_values[res_idx]) if res_idx < len(plddt_values) else 70.0,

                    # 氨基酸理化特性
                    'hydropathy': prop_vector[0],
                    'charge': prop_vector[1],
                    'molecular_weight': prop_vector[2] * 200.0,  # 反归一化
                    'volume': prop_vector[3] * 200.0,  # 反归一化
                    'flexibility': prop_vector[4],
                    'is_aromatic': bool(prop_vector[5]),

                    # 二级结构信息
                    'secondary_structure': ss_code,
                    'ss_alpha': ss_onehot[0],
                    'ss_beta': ss_onehot[1],
                    'ss_coil': ss_onehot[2],

                    # 序列信息
                    'blosum62': blosum.tolist(),
                    'fragment_id': fragment_id
                }

                graph.add_node(node_id, **node_attrs)
                debug_stats['valid_nodes'] += 1

            except Exception as e:
                logger.debug(f"节点创建失败 {res_idx}: {str(e)[:50]}")
                continue

        # 5. 添加序列连接边 - 批量方式
        seq_edges = []
        for i in range(len(valid_indices) - 1):
            idx1 = valid_indices[i]
            idx2 = valid_indices[i + 1]

            if idx2 == idx1 + 1:  # 序列相邻
                # 计算CA-CA距离
                dist = np.linalg.norm(normalized_coords[i + 1] - normalized_coords[i])

                # 序列相邻边 - 类型1
                seq_edges.append((
                    node_ids[i],
                    node_ids[i + 1],
                    {
                        'edge_type': 1,
                        'type_name': 'peptide',
                        'distance': float(dist),
                        'interaction_strength': 1.0,
                        'direction': [1.0, 0.0]  # N->C方向
                    }
                ))

        # 批量添加序列边
        if seq_edges:
            graph.add_edges_from(seq_edges)
            debug_stats['edges_created'] += len(seq_edges)

        # 6. 添加空间相互作用边 - 使用KD树高效查询
        try:
            # 构建高效KD树
            kd_tree = build_optimized_spatial_index(normalized_coords)

            # 高效批量查询K近邻
            k = min(k_neighbors + 1, len(normalized_coords))
            distances, indices = kd_tree.query(normalized_coords, k=k)

            # 预处理相互作用类型 - 向量化操作
            interaction_matrix = classify_interaction_vectorized(
                residue_codes,
                gpu_compute_residue_distances(normalized_coords),
                threshold=distance_threshold
            )

            # 批量创建边数据
            space_edges = []

            for i, neighbors in enumerate(indices):
                node_i = node_ids[i]
                res_i = valid_indices[i]

                # 跳过第一个(自身)
                for j_idx, dist in zip(neighbors[1:], distances[i, 1:]):
                    if j_idx >= len(node_ids) or dist > distance_threshold:
                        continue

                    node_j = node_ids[j_idx]
                    res_j = valid_indices[j_idx]

                    # 跳过序列相邻残基(已添加序列边)
                    if abs(res_j - res_i) <= 1:
                        continue

                    # 获取相互作用类型
                    edge_type = interaction_matrix[i, j_idx]

                    # 只添加有意义的交互
                    if edge_type > 0:
                        # 相互作用类型映射
                        if edge_type == 1:
                            type_name = 'spatial'
                            interaction_strength = 0.3
                        elif edge_type == 2:
                            type_name = 'hbond'
                            interaction_strength = 0.8
                        elif edge_type == 3:
                            type_name = 'ionic'
                            interaction_strength = 0.7
                        elif edge_type == 4:
                            type_name = 'hydrophobic'
                            interaction_strength = 0.5
                        else:
                            type_name = 'unknown'
                            interaction_strength = 0.2

                        # 计算方向向量
                        direction = normalized_coords[j_idx] - normalized_coords[i]
                        norm = np.linalg.norm(direction)

                        if norm > 0:
                            dir_vec_2d = direction[:2] / norm
                        else:
                            dir_vec_2d = np.array([0.0, 0.0])

                        # 添加边数据
                        space_edges.append((
                            node_i,
                            node_j,
                            {
                                'edge_type': edge_type,
                                'type_name': type_name,
                                'distance': float(dist),
                                'interaction_strength': interaction_strength,
                                'direction': dir_vec_2d.tolist()
                            }
                        ))

            # 批量添加空间边
            if space_edges:
                graph.add_edges_from(space_edges)
                debug_stats['edges_created'] += len(space_edges)

        except Exception:
            return None

        # 7. 检查结果图
        # 如果只有一个节点，添加自环以确保图的有效性
        if graph.number_of_nodes() == 1:
            return None

        # 更新最终统计
        debug_stats['edges_created'] = graph.number_of_edges()
        debug_stats['valid_nodes'] = graph.number_of_nodes()

        logger.debug(
            f"图谱构建完成 - 片段: {fragment_id}, 节点: {debug_stats['valid_nodes']}, 边: {debug_stats['edges_created']}")

        return graph

    except Exception as e:
        logger.error(f"构建图谱失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def process_structure_file(file_path, min_length=5, max_length=50, k_neighbors=8,
                           distance_threshold=8.0, plddt_threshold=70.0, respect_ss=True, respect_domains=True):
    """
    高性能优化版蛋白质结构处理函数 - 增强容错机制
    提取片段并构建知识图谱

    优化点:
    - 向量化操作代替循环
    - 严格容错：计算特征失败时直接跳过片段
    - 高效内存管理
    """
    logger.debug(f"处理文件: {file_path}")

    protein_id = os.path.basename(file_path).split('.')[0]
    extracted_fragments = {}
    knowledge_graphs = {}
    fragment_stats = {
        "file_name": os.path.basename(file_path),
        "protein_id": protein_id,
        "fragments": 0,
        "valid_residues": 0,
        "edges": 0,
        "failed_fragments": 0,
        "skipped_fragments": 0,  # 新增：记录因计算失败而跳过的片段数
        "processing_time": 0
    }

    start_time = time.time()

    try:
        # 1. 高效加载结构
        structure = load_structure_mdtraj(file_path)
        if structure is None:
            logger.error(f"无法加载结构: {file_path}")
            return {}, {}, fragment_stats

        # 2. 计算二级结构 - 失败时设置标志位而非使用默认值
        ss_array = None
        ss_computation_failed = False
        try:
            ss_array = compute_secondary_structure(structure)
            if ss_array is None or len(ss_array) == 0:
                logger.warning(f"{protein_id} 二级结构计算结果为空")
                ss_computation_failed = True
        except Exception as e:
            logger.warning(f"{protein_id} 二级结构计算失败: {str(e)[:100]}")
            ss_computation_failed = True

        # 如果二级结构计算失败但不要求respect_ss，我们可以继续处理
        if ss_computation_failed and respect_ss:
            logger.warning(f"由于二级结构计算失败且需要遵循二级结构边界(respect_ss=True)，跳过处理 {protein_id}")
            fragment_stats["processing_time"] = time.time() - start_time
            return {}, {}, fragment_stats

        # 3. 计算接触图 - 失败设置标志位而非使用空列表
        contact_map = None
        residue_pairs = None
        contact_computation_failed = False
        ca_coords = None  # 保存CA坐标供后续使用

        try:
            # 提取CA原子坐标
            ca_indices = [atom.index for atom in structure.topology.atoms if atom.name == 'CA']
            if ca_indices:
                ca_coords = structure.xyz[0, ca_indices]

                # 计算残基对距离矩阵
                dist_matrix = gpu_compute_residue_distances(ca_coords)

                # 获取接触残基对
                n_res = len(ca_indices)
                tmp_contact_map = []
                tmp_residue_pairs = []

                for i in range(n_res):
                    for j in range(i + 2, n_res):  # 跳过相邻残基
                        if dist_matrix[i, j] <= 0.8:  # 0.8 nm 截断值
                            tmp_contact_map.append(dist_matrix[i, j])
                            tmp_residue_pairs.append([i, j])

                contact_map = np.array(tmp_contact_map) if tmp_contact_map else None
                residue_pairs = np.array(tmp_residue_pairs) if tmp_residue_pairs else None

                if contact_map is None or residue_pairs is None:
                    logger.warning(f"{protein_id} 接触图计算结果为空")
                    contact_computation_failed = True
            else:
                logger.warning(f"{protein_id} 未找到CA原子")
                contact_computation_failed = True

        except Exception as e:
            logger.warning(f"{protein_id} 接触图计算失败: {str(e)[:100]}")
            contact_computation_failed = True

        # 如果接触图计算失败且需要尊重结构域，则跳过整个蛋白质
        if contact_computation_failed and respect_domains:
            logger.warning(f"由于接触图计算失败且需要遵循结构域边界(respect_domains=True)，跳过处理 {protein_id}")
            fragment_stats["processing_time"] = time.time() - start_time
            return {}, {}, fragment_stats

        # 4. 提取残基编码和计算有效残基数量
        valid_residues = 0
        res_codes = []
        residue_extraction_failed = False

        try:
            # 批量获取残基名称
            for res in structure.topology.residues:
                one_letter = three_to_one(res.name)
                res_codes.append(one_letter)
                if one_letter != 'X':
                    valid_residues += 1

            if len(res_codes) == 0:
                logger.warning(f"{protein_id} 未提取到有效残基编码")
                residue_extraction_failed = True

        except Exception as e:
            logger.warning(f"{protein_id} 残基编码提取失败: {str(e)[:100]}")
            residue_extraction_failed = True

        if residue_extraction_failed:
            logger.warning(f"由于残基编码提取失败，跳过处理 {protein_id}")
            fragment_stats["processing_time"] = time.time() - start_time
            return {}, {}, fragment_stats

        fragment_stats["valid_residues"] = valid_residues

        # 5. 如果残基数量小于最小长度，则跳过
        if valid_residues < min_length:
            logger.info(f"结构 {protein_id} 残基数量 ({valid_residues}) 小于最小长度 {min_length}，跳过")
            fragment_stats["processing_time"] = time.time() - start_time
            return {}, {}, fragment_stats

        # 6. 创建智能片段
        fragments = []
        n_residues = structure.n_residues
        fragment_creation_failed = False

        try:
            # 根据二级结构边界
            ss_boundaries = []
            if respect_ss and ss_array is not None:
                # 获取第一帧的二级结构
                ss = ss_array[0] if ss_array.ndim > 1 else ss_array

                # 检测二级结构变化
                current_ss = ss[0] if len(ss) > 0 else 'C'
                for i in range(1, len(ss)):
                    if ss[i] != current_ss:
                        ss_boundaries.append(i)
                        current_ss = ss[i]

            # 线性分割作为基础
            linear_boundaries = list(range(0, n_residues, max_length))
            if linear_boundaries[-1] != n_residues:
                linear_boundaries.append(n_residues)

            # 合并所有边界
            all_boundaries = sorted(set(linear_boundaries + ss_boundaries))

            # 创建不超过最大长度的片段
            for i in range(len(all_boundaries) - 1):
                start = all_boundaries[i]
                end = all_boundaries[i + 1]

                # 如果片段过长，进一步分割
                if end - start > max_length:
                    # 使用滑动窗口
                    for j in range(start, end, max(1, max_length // 2)):
                        sub_end = min(j + max_length, end)
                        if sub_end - j >= min_length:
                            frag_id = f"{j + 1}-{sub_end}"
                            fragments.append((j, sub_end, frag_id))
                        if sub_end >= end:
                            break
                elif end - start >= min_length:
                    # 片段长度合适
                    frag_id = f"{start + 1}-{end}"
                    fragments.append((start, end, frag_id))

            if len(fragments) == 0:
                logger.warning(f"{protein_id} 未生成有效片段")
                fragment_creation_failed = True

        except Exception as e:
            logger.warning(f"{protein_id} 片段创建失败: {str(e)[:100]}")
            fragment_creation_failed = True

        # 如果智能片段创建失败，尝试使用简单分段策略
        if fragment_creation_failed:
            logger.warning(f"{protein_id} 使用简单分段策略作为备选")
            try:
                fragments = []
                for i in range(0, n_residues, max(min_length, max_length // 2)):
                    end = min(i + max_length, n_residues)
                    if end - i >= min_length:
                        frag_id = f"{i + 1}-{end}"
                        fragments.append((i, end, frag_id))

                if len(fragments) == 0:
                    logger.warning(f"{protein_id} 简单分段也未能生成有效片段，跳过处理")
                    fragment_stats["processing_time"] = time.time() - start_time
                    return {}, {}, fragment_stats
            except Exception as e:
                logger.error(f"{protein_id} 简单分段也失败: {str(e)[:100]}，跳过处理")
                fragment_stats["processing_time"] = time.time() - start_time
                return {}, {}, fragment_stats

        fragment_stats["fragments"] = len(fragments)
        successful_fragments = 0

        # 7. 处理每个片段 - 这里是逐片段独立处理，如果某片段失败不影响其他片段
        for start_idx, end_idx, frag_id in fragments:
            # 为每个片段创建一个标志，表明是否在处理过程中遇到错误
            fragment_failed = False
            fragment_id = f"{protein_id}_{frag_id}"

            # 提取序列
            sequence = ""
            try:
                for res_idx in range(start_idx, end_idx):
                    if res_idx < len(res_codes):
                        aa = res_codes[res_idx]
                        if aa != "X":
                            sequence += aa

                # 如果序列太短，跳过此片段
                if len(sequence) < min_length:
                    logger.debug(f"片段 {fragment_id} 序列长度 {len(sequence)} 小于最小长度 {min_length}，跳过")
                    fragment_stats["skipped_fragments"] += 1
                    continue
            except Exception as e:
                logger.debug(f"片段 {fragment_id} 序列提取失败: {str(e)[:50]}，跳过")
                fragment_stats["skipped_fragments"] += 1
                continue

            # 保存片段信息
            fragment_data = {
                "protein_id": protein_id,
                "fragment_id": frag_id,
                "sequence": sequence,
                "length": len(sequence),
                "start_idx": start_idx,
                "end_idx": end_idx
            }

            # 构建知识图谱 - 如果失败，跳过该片段
            try:
                # 严格检查所需的所有数据是否都准备好了
                if (respect_ss and ss_array is None) or (
                        respect_domains and (contact_map is None or residue_pairs is None)):
                    logger.debug(f"片段 {fragment_id} 所需数据不完整，跳过")
                    fragment_stats["skipped_fragments"] += 1
                    continue

                kg = build_enhanced_residue_graph(
                    structure, ss_array if ss_array is not None else np.full((1, structure.n_residues), 'C'),
                    (start_idx, end_idx, fragment_id),
                    k_neighbors, distance_threshold, plddt_threshold
                )

                # 只保存有效图谱
                if kg is not None and kg.number_of_nodes() >= 2:
                    extracted_fragments[fragment_id] = fragment_data
                    knowledge_graphs[fragment_id] = nx.node_link_data(kg)
                    fragment_stats["edges"] += kg.number_of_edges()
                    successful_fragments += 1
                else:
                    logger.debug(f"片段 {fragment_id} 生成的图谱无效")
                    fragment_stats["failed_fragments"] += 1
            except Exception as e:
                logger.debug(f"片段 {fragment_id} 图谱构建失败: {str(e)[:100]}")
                fragment_stats["failed_fragments"] += 1

        # 更新统计
        fragment_stats["successful_fragments"] = successful_fragments
        fragment_stats["processing_time"] = time.time() - start_time

        # 如果没有成功处理任何片段，记录警告
        if successful_fragments == 0:
            logger.warning(f"{protein_id} 所有片段处理失败")

        # 释放内存
        del structure
        if 'ss_array' in locals(): del ss_array
        if 'contact_map' in locals(): del contact_map
        if 'residue_pairs' in locals(): del residue_pairs
        if 'ca_coords' in locals(): del ca_coords

        return extracted_fragments, knowledge_graphs, fragment_stats

    except Exception as e:
        logger.error(f"处理文件 {file_path} 失败: {str(e)}")
        logger.error(traceback.format_exc())
        fragment_stats["processing_time"] = time.time() - start_time
        return {}, {}, fragment_stats


def process_file_chunk(file_list, min_length=5, max_length=50, k_neighbors=8,
                       distance_threshold=8.0, plddt_threshold=70.0,
                       respect_ss=True, respect_domains=True):
    """
    优化的文件块处理函数 - 一次处理多个文件，减少进程创建开销

    每个进程处理一批文件，提高整体效率
    """
    results = []
    total_files = len(file_list)

    # 创建一个文件内进度条 (非重要，不影响主进度)
    for i, file_path in enumerate(file_list):
        try:
            start_time = time.time()

            # 处理单个文件
            proteins, kg, fragment_stats = process_structure_file(
                file_path, min_length, max_length, k_neighbors,
                distance_threshold, plddt_threshold, respect_ss, respect_domains
            )

            elapsed = time.time() - start_time
            result_info = {
                "file_path": file_path,
                "elapsed": elapsed,
                "fragment_count": len(proteins),
                "kg_count": len(kg),
                "success": True
            }
            results.append((result_info, proteins, kg, fragment_stats))

            # 定期清理内存
            if (i + 1) % 50 == 0:
                gc.collect()
                if USE_GPU:
                    torch.cuda.empty_cache()

        except Exception as e:
            error_info = {"error": str(e)[:200], "file_path": file_path}
            results.append((error_info, {}, {}, None))

    return results


def process_file_parallel(file_list, output_dir, min_length=5, max_length=50,
                          n_workers=None, batch_size=50000, memory_limit_gb=800,
                          k_neighbors=8, distance_threshold=8.0, plddt_threshold=70.0,
                          respect_ss=True, respect_domains=True, format_type="pyg",
                          use_gpu=None):
    """
    高性能蛋白质结构批处理系统 - 适用于TB级内存和百核处理器

    优化点:
    - 动态负载均衡代替静态分块
    - 渐进式内存管理避免OOM
    - 多层并行提高CPU利用率
    - 高效IO减少磁盘瓶颈

    参数:
        file_list: 待处理文件列表
        output_dir: 输出目录
        batch_size: 每批处理的文件数量
        memory_limit_gb: 内存使用上限(GB)
        n_workers: 并行工作进程数
        use_gpu: 是否使用GPU (None: 自动检测)
    """
    # 设置GPU使用
    global USE_GPU
    if use_gpu is not None:
        USE_GPU = use_gpu

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    # 创建目录结构
    base_data_dir = os.path.join(output_dir, "ALL")
    temp_dir = os.path.join(output_dir, "TEMP")
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # 预计算处理批次
    total_files = len(file_list)
    total_batches = (total_files + batch_size - 1) // batch_size
    logger.info(f"总文件数: {total_files}, 划分为 {total_batches} 批次处理，每批次最多 {batch_size} 文件")
    logger.info(f"使用 {n_workers} 个CPU核心并行处理")
    logger.info(f"内存限制设置为 {memory_limit_gb}GB")

    # 初始化日志文件
    sequences_log_path = os.path.join(output_dir, "sequences.log")
    fragments_log_path = os.path.join(output_dir, "fragments_stats.log")
    processing_log_path = os.path.join(output_dir, "processing.log")

    with open(sequences_log_path, 'w', buffering=1) as s_log:
        s_log.write("fragment_id,protein_id,length,sequence\n")
    with open(fragments_log_path, 'w') as f_log:
        f_log.write("file_name,protein_id,valid_residues,fragments,successful_fragments,edges,processing_time\n")
    with open(processing_log_path, 'w') as p_log:
        p_log.write("timestamp,file_path,status,elapsed,fragments,knowledge_graphs,error\n")

    # 划分批次
    batches = [file_list[i:i + batch_size] for i in range(0, total_files, batch_size)]

    # 全局统计
    global_stats = {
        "processed_files": 0,
        "extracted_fragments": 0,
        "knowledge_graphs": 0,
        "failed_files": 0,
        "total_edges": 0,
        "start_time": time.time()
    }

    # 处理每个批次
    for batch_id, batch_files in enumerate(batches):
        batch_start_time = time.time()
        logger.info(f"开始处理批次 {batch_id + 1}/{len(batches)} ({len(batch_files)} 文件)")

        batch_output_dir = os.path.join(base_data_dir, f"batch_{batch_id + 1}")
        os.makedirs(batch_output_dir, exist_ok=True)

        # 批次级缓存
        batch_proteins = {}
        batch_graphs = {}

        # 每批处理前清理内存
        check_memory_usage(threshold_gb=memory_limit_gb, force_gc=True)

        # 创建动态工作池
        task_queue, num_tasks = create_dynamic_worker_pool(n_workers, batch_files)

        # 使用进度条
        with tqdm(total=len(batch_files), desc=f"批次 {batch_id + 1} 处理进度",
                  file=sys.stdout, ncols=100, dynamic_ncols=True) as pbar:

            # 创建进程池
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=multiprocessing.get_context('spawn')
            ) as executor:
                futures = []

                # 提交任务
                while not task_queue.empty():
                    chunk_files = task_queue.get()
                    future = executor.submit(
                        process_file_chunk, chunk_files, min_length, max_length,
                        k_neighbors, distance_threshold, plddt_threshold,
                        respect_ss, respect_domains
                    )
                    futures.append(future)

                # 处理结果
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results = future.result()
                        for result, proteins, kg, fragment_stats in results:
                            pbar.update(1)
                            pbar.refresh()  # 强制刷新

                            if "error" in result:
                                global_stats["failed_files"] += 1
                                with open(processing_log_path, 'a') as p_log:
                                    p_log.write(
                                        f"{time.strftime('%Y-%m-%d %H:%M:%S')},{result['file_path']},FAILED,0,0,0,{result['error']}\n"
                                    )
                                continue

                            # 更新统计信息
                            global_stats["processed_files"] += 1
                            global_stats["extracted_fragments"] += len(proteins)
                            global_stats["knowledge_graphs"] += len(kg)

                            if fragment_stats:
                                global_stats["total_edges"] += fragment_stats.get('edges', 0)

                                # 记录片段统计
                                with open(fragments_log_path, 'a') as f_log:
                                    f_log.write(
                                        f"{fragment_stats['file_name']},{fragment_stats['protein_id']},"
                                        f"{fragment_stats.get('valid_residues', 0)},"
                                        f"{fragment_stats.get('fragments', 0)},"
                                        f"{fragment_stats.get('successful_fragments', 0)},"
                                        f"{fragment_stats.get('edges', 0)},"
                                        f"{fragment_stats.get('processing_time', 0):.2f}\n"
                                    )

                            # 更新序列日志
                            with open(sequences_log_path, 'a', buffering=1) as s_log:
                                for fragment_id, data in proteins.items():
                                    s_log.write(
                                        f"{fragment_id},{data['protein_id']},"
                                        f"{len(data['sequence'])},{data['sequence']}\n"
                                    )

                            # 累积数据
                            batch_proteins.update(proteins)
                            batch_graphs.update(kg)

                            # 记录处理日志
                            with open(processing_log_path, 'a') as p_log:
                                elapsed = result.get('elapsed', 0)
                                p_log.write(
                                    f"{time.strftime('%Y-%m-%d %H:%M:%S')},{result['file_path']},"
                                    f"SUCCESS,{elapsed:.2f},"
                                    f"{len(proteins)},{len(kg)},\n"
                                )
                    except Exception as e:
                        logger.error(f"处理任务时出错: {str(e)}")
                        logger.error(traceback.format_exc())

        # 保存批次结果
        batch_elapsed = time.time() - batch_start_time
        logger.info(
            f"批次 {batch_id + 1} 处理完成，耗时 {batch_elapsed:.1f} 秒，"
            f"保存 {len(batch_proteins)} 个蛋白质片段和 {len(batch_graphs)} 个图谱"
        )

        # 异步保存数据
        save_start_time = time.time()
        try:
            # 并行保存数据
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as save_executor:
                futures = []

                # 保存蛋白质数据
                if batch_proteins:
                    futures.append(
                        save_executor.submit(
                            save_results_chunked, batch_proteins, batch_output_dir,
                            base_name="protein_data", chunk_size=100000
                        )
                    )

                # 保存图谱数据
                if batch_graphs:
                    futures.append(
                        save_executor.submit(
                            save_knowledge_graphs, batch_graphs, batch_output_dir,
                            base_name="protein_kg", chunk_size=100000, format_type=format_type
                        )
                    )

                # 等待所有保存任务完成
                concurrent.futures.wait(futures)

            logger.info(f"批次 {batch_id + 1} 数据保存完成，耗时 {time.time() - save_start_time:.1f} 秒")
        except Exception as e:
            logger.error(f"批次 {batch_id + 1} 数据保存失败: {str(e)}")

        # 清理内存
        batch_proteins.clear()
        batch_graphs.clear()
        check_memory_usage(force_gc=True)

        # 当前进度报告
        elapsed_hours = (time.time() - global_stats["start_time"]) / 3600
        files_per_hour = global_stats["processed_files"] / max(0.1, elapsed_hours)

        logger.info(
            f"当前进度: {global_stats['processed_files']}/{total_files} 文件 "
            f"({global_stats['processed_files'] / total_files * 100:.1f}%) | "
            f"速率: {files_per_hour:.1f} 文件/小时 | "
            f"片段: {global_stats['extracted_fragments']} | "
            f"图谱: {global_stats['knowledge_graphs']} | "
            f"边数: {global_stats['total_edges']}"
        )

        # 预计完成时间
        if global_stats["processed_files"] > 0:
            remaining_files = total_files - global_stats["processed_files"]
            remaining_hours = remaining_files / files_per_hour
            est_completion = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_hours * 3600))
            logger.info(f"预计完成时间: {est_completion} (还需 {remaining_hours:.1f} 小时)")

    # 处理完成，返回统计信息
    total_time = time.time() - global_stats["start_time"]
    global_stats["total_time"] = total_time
    global_stats["files_per_second"] = global_stats["processed_files"] / max(1.0, total_time)

    return global_stats, base_data_dir


def save_results_chunked(all_proteins, output_dir, base_name="protein_data", chunk_size=100000):
    """
    分块高效保存蛋白质数据

    优化点:
    - 使用pickle/msgpack加速序列化
    - 批量处理减少IO开销
    - 自动压缩大型数据集
    """
    os.makedirs(output_dir, exist_ok=True)

    # 首先检查是否有数据要保存
    if not all_proteins:
        logger.warning(f"没有蛋白质数据要保存到 {output_dir}")
        return None, None

    try:
        # 使用pickle格式保存数据 - 比JSON更高效
        use_pickle = True
        use_compression = len(all_proteins) > 10000  # 对大型数据集使用压缩

        protein_ids = list(all_proteins.keys())
        chunks = [protein_ids[i:i + chunk_size] for i in range(0, len(protein_ids), chunk_size)]

        output_files = []

        for i, chunk_ids in enumerate(chunks):
            # 创建子字典
            chunk_data = {pid: all_proteins[pid] for pid in chunk_ids}

            # 确定输出文件名
            if use_pickle:
                ext = ".pkl.gz" if use_compression else ".pkl"
                output_file = os.path.join(output_dir, f"{base_name}_chunk_{i + 1}{ext}")

                # 保存pickle格式
                if use_compression:
                    with gzip.open(output_file, 'wb', compresslevel=3) as f:
                        pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(output_file, 'wb') as f:
                        pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # 备选方案: JSON格式
                ext = ".json.gz" if use_compression else ".json"
                output_file = os.path.join(output_dir, f"{base_name}_chunk_{i + 1}{ext}")

                # 处理NumPy类型
                clean_data = {}
                for pid, protein_data in chunk_data.items():
                    clean_data[pid] = {}
                    for key, value in protein_data.items():
                        if isinstance(value, np.ndarray):
                            clean_data[pid][key] = value.tolist()
                        elif isinstance(value, (np.integer, np.int64, np.int32, np.int8)):
                            clean_data[pid][key] = int(value)
                        elif isinstance(value, (np.floating, np.float64, np.float32)):
                            clean_data[pid][key] = float(value)
                        else:
                            clean_data[pid][key] = value

                # 保存JSON格式
                if use_compression:
                    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                        json.dump(clean_data, f)
                else:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(clean_data, f)

            output_files.append(output_file)

        # 保存元数据
        metadata = {
            "total_proteins": len(all_proteins),
            "chunk_count": len(chunks),
            "chunk_files": [os.path.basename(f) for f in output_files],
            "format": "pickle" if use_pickle else "json",
            "compressed": use_compression,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"已保存 {len(all_proteins)} 个蛋白质数据到 {len(output_files)} 个文件")
        return output_files, metadata

    except Exception as e:
        logger.error(f"保存蛋白质数据失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None


def save_knowledge_graphs(kg_data, output_dir, base_name="protein_kg", chunk_size=100000, format_type="pyg"):
    """
    高效保存知识图谱数据

    优化点:
    - PyG格式使用高效二进制存储
    - 批量处理和压缩大型图谱集
    - 自动转换数据类型减少内存使用

    参数:
        kg_data: 知识图谱字典，格式为 {fragment_id: graph_data}
        output_dir: 输出目录
        base_name: 输出文件基本名称
        chunk_size: 每个文件中的最大图谱数
        format_type: 输出格式("pyg"或"json")
    """
    if not kg_data:
        logger.warning(f"没有知识图谱数据可保存")
        return None

    # 创建输出目录
    kg_dir = os.path.join(output_dir, f"knowledge_graphs_{format_type}")
    os.makedirs(kg_dir, exist_ok=True)

    # 统计信息
    all_protein_ids = list(kg_data.keys())
    num_chunks = (len(all_protein_ids) + chunk_size - 1) // chunk_size

    # 文件大小决定是否使用压缩
    use_compression = len(all_protein_ids) > 5000

    logger.info(f"将 {len(all_protein_ids)} 个知识图谱拆分为 {num_chunks} 个块，每块最多 {chunk_size} 个")

    # 创建索引字典
    index = {
        "total_proteins": len(all_protein_ids),
        "chunks_count": num_chunks,
        "chunk_size": chunk_size,
        "total_nodes": 0,
        "total_edges": 0,
        "error_count": 0,
        "compressed": use_compression,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": format_type
    }

    # 处理每个块
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_protein_ids))
        chunk_ids = all_protein_ids[start_idx:end_idx]

        logger.info(f"处理知识图谱块 {chunk_id + 1}/{num_chunks}，共 {len(chunk_ids)} 个图谱")

        # 确定输出文件名
        if format_type == "pyg":
            ext = ".pt.gz" if use_compression else ".pt"
            output_file = os.path.join(kg_dir, f"{base_name}_chunk_{chunk_id + 1}{ext}")

            # PyG格式保存
            try:
                graphs_data = {}
                node_count = 0
                edge_count = 0

                for pid in chunk_ids:
                    try:
                        # 获取NetworkX图数据
                        nx_data = kg_data[pid]

                        # 如果是字典格式，转换为NetworkX图
                        if isinstance(nx_data, dict):
                            nx_graph = nx.node_link_graph(nx_data)
                        else:
                            nx_graph = nx_data

                        # 转换为PyG格式
                        node_features = []  # 节点特征列表
                        edge_index = [[], []]  # 边索引
                        edge_attr = []  # 边特征

                        # 创建节点映射
                        node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}

                        # BLOSUM62编码维度
                        blosum_dim = 20

                        # 提取节点特征
                        for node in nx_graph.nodes():
                            node_attrs = nx_graph.nodes[node]

                            # 1. BLOSUM62编码 (20维)
                            blosum = node_attrs.get('blosum62', [0] * blosum_dim)
                            if not blosum or len(blosum) != blosum_dim:
                                blosum = [0] * blosum_dim

                            # 2. 相对空间坐标 (3维)
                            position = node_attrs.get('position', [0, 0, 0])[:3]

                            # 3. 氨基酸理化特性
                            residue_code = node_attrs.get('residue_code', 'X')
                            props = AA_PROPERTIES.get(residue_code, AA_PROPERTIES['X'])

                            hydropathy = props['hydropathy']
                            charge = props['charge']
                            molecular_weight = props['mw'] / 200.0  # 归一化
                            volume = props['volume'] / 200.0  # 归一化
                            flexibility = props['flexibility']
                            is_aromatic = 1.0 if props['aromatic'] else 0.0

                            # 4. 二级结构编码 (3维)
                            ss_alpha = float(node_attrs.get('ss_alpha', 0))
                            ss_beta = float(node_attrs.get('ss_beta', 0))
                            ss_coil = float(node_attrs.get('ss_coil', 0))

                            # 5. 表面暴露程度 (1维)
                            sasa = float(node_attrs.get('sasa', 0.5))

                            # 6. pLDDT值 (1维)
                            plddt = float(node_attrs.get('plddt', 70.0)) / 100.0  # 归一化到0-1

                            # 合并所有特征
                            features = blosum + position + [hydropathy, charge, molecular_weight,
                                                            volume, flexibility, is_aromatic,
                                                            ss_alpha, ss_beta, ss_coil, sasa, plddt]

                            node_features.append(features)

                        # 提取边特征
                        for src, tgt, edge_data in nx_graph.edges(data=True):
                            edge_index[0].append(node_mapping[src])
                            edge_index[1].append(node_mapping[tgt])

                            # 1. 边类型
                            edge_type = edge_data.get('edge_type', 0)

                            # 2. 空间距离 (1维)
                            distance = float(edge_data.get('distance', 0))

                            # 3. 相互作用强度 (1维)
                            interaction_strength = float(edge_data.get('interaction_strength', 0.5))

                            # 4. 方向性 (2维)
                            direction = edge_data.get('direction', [0, 0])
                            if len(direction) != 2:
                                direction = [0, 0]

                            # 合并边特征
                            edge_features = [edge_type, distance, interaction_strength] + direction
                            edge_attr.append(edge_features)

                        # 创建PyG数据对象
                        if node_features and edge_index[0]:  # 确保有节点和边
                            data = Data(
                                x=torch.tensor(node_features, dtype=torch.float),
                                edge_index=torch.tensor(edge_index, dtype=torch.long),
                                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                                protein_id=pid
                            )

                            # 添加额外元数据
                            if 'fragment_id' in next(iter(nx_graph.nodes(data=True)))[1]:
                                fragment_id = next(iter(nx_graph.nodes(data=True)))[1]['fragment_id']
                                data.fragment_id = fragment_id

                            # 保存序列信息（如果可用）
                            sequence = ""
                            for node, attrs in sorted(nx_graph.nodes(data=True),
                                                      key=lambda x: x[1].get('residue_idx', 0)):
                                sequence += attrs.get('residue_code', 'X')
                            data.sequence = sequence

                            graphs_data[pid] = data

                            # 更新统计信息
                            node_count += len(node_features)
                            edge_count += len(edge_attr)
                    except Exception as e:
                        logger.debug(f"转换图谱 {pid} 失败: {str(e)[:100]}")
                        index["error_count"] += 1

                # 保存为PyTorch文件
                if graphs_data:
                    # 使用压缩或非压缩保存
                    if use_compression:
                        buffer = io.BytesIO()
                        torch.save(graphs_data, buffer)
                        buffer.seek(0)
                        with gzip.open(output_file, 'wb', compresslevel=4) as f:
                            f.write(buffer.getvalue())
                    else:
                        torch.save(graphs_data, output_file)

                    # 更新总统计
                    index["total_nodes"] += node_count
                    index["total_edges"] += edge_count

                    logger.info(f"已保存 {len(graphs_data)} 个PyG格式图谱到 {output_file}")
            except Exception as e:
                logger.error(f"保存PyG格式失败 (块 {chunk_id + 1}): {str(e)}")
        else:
            # JSON格式保存
            ext = ".json.gz" if use_compression else ".json"
            output_file = os.path.join(kg_dir, f"{base_name}_chunk_{chunk_id + 1}{ext}")

            try:
                chunk_data = {}
                node_count = 0
                edge_count = 0

                # 预处理当前块中的所有图谱
                for pid in chunk_ids:
                    nx_data = kg_data[pid]

                    if isinstance(nx_data, nx.Graph):
                        # 转换为字典表示
                        nx_data = nx.node_link_data(nx_data)

                    # 清理NumPy类型
                    clean_graph = {"nodes": [], "links": []}

                    # 处理节点
                    for node in nx_data.get("nodes", []):
                        clean_node = {}
                        for k, v in node.items():
                            if isinstance(v, np.ndarray):
                                clean_node[k] = v.tolist()
                            elif isinstance(v, (np.integer, np.floating)):
                                clean_node[k] = float(v) if isinstance(v, np.floating) else int(v)
                            else:
                                clean_node[k] = v
                        clean_graph["nodes"].append(clean_node)

                    # 处理边
                    for link in nx_data.get("links", []):
                        clean_link = {}
                        for k, v in link.items():
                            if isinstance(v, np.ndarray):
                                clean_link[k] = v.tolist()
                            elif isinstance(v, (np.integer, np.floating)):
                                clean_link[k] = float(v) if isinstance(v, np.floating) else int(v)
                            else:
                                clean_link[k] = v
                        clean_graph["links"].append(clean_link)

                    # 保存清理后的图谱
                    chunk_data[pid] = clean_graph

                    # 更新统计
                    node_count += len(clean_graph["nodes"])
                    edge_count += len(clean_graph["links"])

                # 保存JSON文件
                if use_compression:
                    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                        json.dump(chunk_data, f)
                else:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(chunk_data, f)

                # 更新总统计
                index["total_nodes"] += node_count
                index["total_edges"] += edge_count

                logger.info(f"已保存 {len(chunk_data)} 个JSON格式图谱到 {output_file}")
            except Exception as e:
                logger.error(f"保存JSON格式失败 (块 {chunk_id + 1}): {str(e)}")

    # 保存索引文件
    index_file = os.path.join(kg_dir, f"{base_name}_index.json")
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)

    return index


def find_pdb_files(root_dir):
    """
    高性能递归搜索所有PDB和CIF文件，并去重

    优化点:
    - 使用多线程并行搜索大目录
    - 哈希表快速去重
    - 内存高效文件列表构建
    """
    logger.info(f"开始扫描目录: {root_dir}")
    start_time = time.time()

    # 使用集合快速去重
    processed_ids = set()
    pdb_files = []

    # 对大型目录进行并行搜索
    if os.path.isdir(root_dir) and sum(1 for _ in os.scandir(root_dir)) > 100:
        # 并行搜索大目录
        subdirs = [os.path.join(root_dir, d.name) for d in os.scandir(root_dir) if d.is_dir()]

        if len(subdirs) > 5:
            # 创建线程池
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(subdirs))) as executor:
                futures = []

                # 并行搜索子目录
                def search_subdir(subdir):
                    subdir_files = []
                    subdir_ids = set()

                    for root, _, files in os.walk(subdir):
                        for file in files:
                            if file.endswith(('.pdb', '.pdb.gz', '.cif', '.cif.gz')):
                                # 提取蛋白质ID (去除扩展名)
                                protein_id = file.split('.')[0]

                                # 跳过已处理的ID
                                if protein_id not in subdir_ids:
                                    subdir_ids.add(protein_id)
                                    subdir_files.append(os.path.join(root, file))

                    return subdir_files, subdir_ids

                # 提交任务
                for subdir in subdirs:
                    futures.append(executor.submit(search_subdir, subdir))

                # 收集结果
                for future in concurrent.futures.as_completed(futures):
                    subdir_files, subdir_ids = future.result()

                    # 过滤掉已处理的ID
                    new_files = []
                    for i, file in enumerate(subdir_files):
                        protein_id = os.path.basename(file).split('.')[0]
                        if protein_id not in processed_ids:
                            processed_ids.add(protein_id)
                            new_files.append(file)

                    pdb_files.extend(new_files)

    # 顺序处理根目录(如果没有使用并行或对小目录)
    if not pdb_files:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.pdb', '.pdb.gz', '.cif', '.cif.gz')):
                    # 提取蛋白质ID (去除扩展名)
                    protein_id = file.split('.')[0]

                    # 跳过已处理的ID
                    if protein_id not in processed_ids:
                        processed_ids.add(protein_id)
                        pdb_files.append(os.path.join(root, file))

    elapsed = time.time() - start_time
    logger.info(f"扫描完成，耗时: {elapsed:.1f}秒，找到 {len(pdb_files)} 个不重复的PDB/CIF文件")
    return pdb_files


def main():
    """
    主程序入口 - 支持大规模处理和GPU加速
    """
    parser = argparse.ArgumentParser(description="蛋白质结构知识图谱构建系统 - 高性能优化版")
    parser.add_argument("input", help="输入PDB/CIF文件或包含这些文件的目录")
    parser.add_argument("--output_dir", "-o", default="./kg", help="输出目录 (默认: ./kg)")
    parser.add_argument("--min_length", "-m", type=int, default=5, help="最小序列长度 (默认: 5)")
    parser.add_argument("--max_length", "-M", type=int, default=50, help="最大序列长度 (默认: 50)")
    parser.add_argument("--batch_size", "-b", type=int, default=100000, help="大规模批处理大小 (默认: 100000)")
    parser.add_argument("--memory_limit", type=int, default=800, help="内存使用上限GB (默认: 800)")
    parser.add_argument("--workers", "-w", type=int, default=None, help="并行工作进程数 (默认: CPU核心数-1)")
    parser.add_argument("--k_neighbors", type=int, default=8, help="空间邻接的K近邻数 (默认: 8)")
    parser.add_argument("--distance_threshold", type=float, default=8.0, help="空间邻接距离阈值 (默认: 8.0埃)")
    parser.add_argument("--plddt_threshold", type=float, default=70.0, help="AlphaFold pLDDT质量得分阈值 (默认: 70.0)")
    parser.add_argument("--respect_ss", action="store_true", default=True, help="是否尊重二级结构边界进行片段划分")
    parser.add_argument("--respect_domains", action="store_true", default=True, help="是否尊重结构域边界进行片段划分")
    parser.add_argument("--format", choices=["pyg", "json"], default="pyg", help="知识图谱保存格式 (默认: pyg)")
    parser.add_argument("--limit", type=int, help="限制处理的文件数量 (用于测试)")
    parser.add_argument("--no_gpu", action="store_true", help="禁用GPU加速 (默认: 自动检测)")

    args = parser.parse_args()

    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    global logger
    logger, log_file_path = setup_logging(args.output_dir)
    logger.info(f"日志将写入文件: {log_file_path}")

    # 设置GPU使用
    use_gpu = not args.no_gpu and torch.cuda.is_available()

    if use_gpu:
        logger.info(f"GPU加速已启用 ({torch.cuda.get_device_name(0)})")
    else:
        logger.info("GPU加速已禁用，使用CPU运算")

    logger.info(f"大规模处理模式: 使用最大 {args.memory_limit}GB 内存和 {args.workers or '自动检测'} 个CPU核心")

    # 查找输入文件
    if os.path.isdir(args.input):
        pdb_files = find_pdb_files(args.input)
    else:
        pdb_files = [args.input] if os.path.isfile(args.input) else []

    if args.limit:
        pdb_files = pdb_files[:args.limit]
        logger.info(f"限制处理文件数量: {args.limit}")

    if not pdb_files:
        logger.error(f"未找到PDB/CIF文件，请检查输入路径: {args.input}")
        return

    logger.info(f"开始处理 {len(pdb_files)} 个PDB/CIF文件...")
    start_proc_time = time.time()

    # 使用优化后的大规模处理函数
    stats, all_data_dir = process_file_parallel(
        pdb_files,
        args.output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        n_workers=args.workers,
        batch_size=args.batch_size,
        memory_limit_gb=args.memory_limit,
        k_neighbors=args.k_neighbors,
        distance_threshold=args.distance_threshold,
        plddt_threshold=args.plddt_threshold,
        respect_ss=args.respect_ss,
        respect_domains=args.respect_domains,
        format_type=args.format,
        use_gpu=use_gpu
    )

    total_time = time.time() - start_proc_time
    files_per_second = stats['processed_files'] / max(1.0, total_time)

    # 处理结果统计
    logger.info("\n处理完成:")
    logger.info(f"- 总耗时: {total_time / 3600:.2f}小时 ({total_time:.1f}秒)")
    logger.info(f"- 平均每文件: {total_time / max(1, stats['processed_files']):.3f}秒")
    logger.info(f"- 处理速度: {files_per_second:.2f}文件/秒")
    logger.info(f"- 处理的文件总数: {stats['processed_files']}")
    logger.info(f"- 提取的蛋白质片段总数: {stats['extracted_fragments']}")
    logger.info(f"- 生成的知识图谱总数: {stats['knowledge_graphs']}")
    logger.info(f"- 知识图谱边总数: {stats.get('total_edges', 0)}")
    logger.info(f"- 失败的文件数: {stats.get('failed_files', 0)}")
    logger.info(f"- 结果保存在: {all_data_dir}")

    # 添加总结信息
    summary_file = os.path.join(args.output_dir, "extraction_summary.json")
    with open(summary_file, 'w') as f:
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time_seconds": total_time,
            "execution_time_hours": total_time / 3600,
            "files_per_second": files_per_second,
            "processed_files": stats['processed_files'],
            "failed_files": stats.get('failed_files', 0),
            "extracted_fragments": stats['extracted_fragments'],
            "knowledge_graphs": stats['knowledge_graphs'],
            "total_edges": stats.get('total_edges', 0),
            "gpu_accelerated": use_gpu,
            "parameters": {
                "min_length": args.min_length,
                "max_length": args.max_length,
                "k_neighbors": args.k_neighbors,
                "distance_threshold": args.distance_threshold,
                "plddt_threshold": args.plddt_threshold,
                "respect_ss": args.respect_ss,
                "respect_domains": args.respect_domains,
                "format": args.format,
                "batch_size": args.batch_size,
                "workers": args.workers,
                "memory_limit_gb": args.memory_limit
            },
            "output_dir": os.path.abspath(args.output_dir),
            "all_data_dir": os.path.abspath(all_data_dir),
            "log_file": os.path.abspath(log_file_path)
        }
        json.dump(summary, f, indent=2)

    logger.info(f"摘要信息已保存到: {summary_file}")
    logger.info("蛋白质结构知识图谱构建系统处理完成！")


if __name__ == "__main__":
    import io  # 导入io用于内存缓冲区操作

    main()