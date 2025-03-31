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

import shutil
import logging
import time
import sys
import platform
import traceback

# 核心依赖
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm

from scipy.spatial import KDTree


# 生物信息学库
import mdtraj as md

from Bio.SeqUtils import seq1


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
    console = logging.StreamHandler()
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

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("未检测到GPU")

    return root_logger, log_file


def check_memory_usage(threshold_gb=None, force_gc=False):
    """检查内存使用情况，如果超过阈值则进行垃圾回收"""
    try:
        import psutil
        # 获取当前进程
        process = psutil.Process()

        # 获取当前内存使用量
        mem_used_bytes = process.memory_info().rss
        mem_used_gb = mem_used_bytes / (1024 ** 3)  # 转换为GB

        # 获取系统总内存
        if threshold_gb is None:
            system_mem = psutil.virtual_memory()
            total_mem_gb = system_mem.total / (1024 ** 3)
            threshold_gb = total_mem_gb * 0.8  # 默认使用系统内存的80%作为阈值

        # 检查是否超过阈值或强制执行
        if mem_used_gb > threshold_gb or force_gc:
            logger.warning(f"内存使用达到 {mem_used_gb:.2f} GB，执行垃圾回收")

            # 执行垃圾回收
            collected = gc.collect()

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 检查回收效果
            mem_after_gc = process.memory_info().rss / (1024 ** 3)
            logger.info(f"垃圾回收后内存使用: {mem_after_gc:.2f} GB (释放了 {mem_used_gb - mem_after_gc:.2f} GB)")

            return True
        return False
    except Exception as e:
        logger.warning(f"检查内存使用时出错: {str(e)}")
        # 预防性执行垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def create_dynamic_worker_pool(file_list, n_workers, min_chunk_size=50, max_chunk_size=1000):
    """
    实现动态任务分配工作池，根据文件大小平衡负载

    参数:
        file_list: 待处理文件列表
        n_workers: 工作进程数量
        min_chunk_size: 最小分块大小
        max_chunk_size: 最大分块大小

    返回:
        chunks_list: 平衡后的任务列表，每个元素为一个文件列表
    """
    import math
    import numpy as np
    from collections import defaultdict

    # 获取文件大小
    file_sizes = []
    for f in file_list:
        try:
            size = os.path.getsize(f) if os.path.exists(f) else 1024 * 1024  # 默认1MB
            file_sizes.append((f, size))
        except (OSError, IOError):
            file_sizes.append((f, 1024 * 1024))  # 无法获取大小时使用默认值

    # 按大小从大到小排序
    file_sizes.sort(key=lambda x: x[1], reverse=True)

    # 创建n_workers个桶
    buckets = [[] for _ in range(n_workers)]
    bucket_sizes = [0] * n_workers

    # 使用贪心算法分配任务 - 总是分配给当前总大小最小的桶
    for file_path, file_size in file_sizes:
        min_idx = np.argmin(bucket_sizes)
        buckets[min_idx].append(file_path)
        bucket_sizes[min_idx] += file_size

    # 细分过大的桶，确保每个任务块大小适中
    final_chunks = []
    for bucket in buckets:
        if len(bucket) <= max_chunk_size:
            if len(bucket) >= min_chunk_size:
                final_chunks.append(bucket)
            else:
                # 如果桶太小，与其他小桶合并
                if final_chunks and len(final_chunks[-1]) < max_chunk_size:
                    final_chunks[-1].extend(bucket)
                else:
                    final_chunks.append(bucket)
        else:
            # 将大桶分成多个小块
            n_sub_chunks = math.ceil(len(bucket) / max_chunk_size)
            sub_size = math.ceil(len(bucket) / n_sub_chunks)

            for i in range(0, len(bucket), sub_size):
                sub_chunk = bucket[i:i + sub_size]
                if len(sub_chunk) >= min_chunk_size:
                    final_chunks.append(sub_chunk)
                elif final_chunks:
                    final_chunks[-1].extend(sub_chunk)
                else:
                    final_chunks.append(sub_chunk)

    return final_chunks


def optimized_memory_tracking(threshold_gb=None, force_gc=False, ratio_threshold=0.75):
    """
    优化的内存跟踪和释放函数

    参数:
        threshold_gb: 内存阈值(GB)
        force_gc: 是否强制执行垃圾回收
        ratio_threshold: 内存使用比例阈值

    返回:
        cleaned: 是否执行了内存清理
    """
    import psutil
    import gc
    import torch

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    used_gb = mem_info.rss / (1024 ** 3)

    # 确定阈值
    if threshold_gb is None:
        total_mem = psutil.virtual_memory().total / (1024 ** 3)
        threshold_gb = total_mem * ratio_threshold

    # 检查是否需要清理
    need_cleanup = used_gb > threshold_gb

    if need_cleanup or force_gc:
        # 记录清理前状态
        before_gc = used_gb

        # 1. 禁用垃圾收集器自动运行
        gc.disable()

        # 2. 收集未引用对象
        unreachable = gc.collect(2)  # 执行完整收集

        # 3. 如果使用了PyTorch，清除GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. 启用自动垃圾收集
        gc.enable()

        # 获取清理后状态
        mem_info_post = process.memory_info()
        after_gc = mem_info_post.rss / (1024 ** 3)

        # 记录清理效果
        freed_gb = before_gc - after_gc
        logger.info(
            f"内存清理: 从 {before_gc:.2f}GB 减少到 {after_gc:.2f}GB (释放 {freed_gb:.2f}GB, {unreachable} 个对象)")

        return freed_gb > 0

    return False

def three_to_one(residue_name):
    """将三字母氨基酸代码转换为单字母代码"""
    try:
        return seq1(residue_name)
    except Exception:
        d = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return d.get(residue_name.upper(), 'X')


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


# ======================= MDTraj二级结构与特征提取 =======================

def load_structure_mdtraj(file_path):
    """使用MDTraj加载蛋白质结构"""
    try:
        # 检查文件格式
        is_gzipped = file_path.endswith('.gz')

        # 处理gzip压缩文件
        if is_gzipped:
            temp_file = f"/tmp/{os.path.basename(file_path)[:-3]}"
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
        if is_gzipped and os.path.exists(temp_file):
            os.remove(temp_file)

        return structure
    except Exception as e:
        logger.error(f"MDTraj加载结构失败: {str(e)}")
        if is_gzipped and 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return None


def compute_secondary_structure(structure):
    """使用MDTraj计算二级结构"""
    try:
        # 使用MDTraj的compute_dssp函数计算二级结构
        ss = md.compute_dssp(structure, simplified=False)

        # 将MDTraj的二级结构编码映射为标准的8类
        mapped_ss = np.vectorize(lambda x: SS_MAPPING.get(x, 'C'))(ss)

        return mapped_ss
    except Exception as e:
        logger.error(f"计算二级结构失败: {str(e)}")
        # 返回全C（coil）作为默认值
        return np.full((structure.n_frames, structure.n_residues), 'C')


def compute_solvent_accessibility(structure):
    """计算溶剂可及性"""
    try:
        # 使用MDTraj计算溶剂可及表面积
        sasa = md.shrake_rupley(structure, probe_radius=0.14, n_sphere_points=960)
        # 归一化（除以最大观测值或理论最大值）
        max_sasa = np.max(sasa)
        if max_sasa > 0:
            normalized_sasa = sasa / max_sasa
        else:
            normalized_sasa = sasa
        return normalized_sasa
    except Exception as e:
        logger.error(f"计算溶剂可及性失败: {str(e)}")
        # 返回默认值0.5
        return np.full((structure.n_frames, structure.n_residues), 0.5)


def compute_contacts(structure, cutoff=0.8):
    """计算残基接触图（兼容新版MDTraj）

    参数:
        structure: MDTraj结构对象
        cutoff: 接触距离阈值，单位为nm

    返回:
        contacts: 接触矩阵
        residue_pairs: 接触残基对
    """
    try:
        # 使用MDTraj计算残基接触
        # 而是需要在获取contacts后进行过滤
        residue_pairs = []
        n_res = structure.n_residues

        # 创建所有可能的残基对（不包括相邻的）
        for i in range(n_res):
            for j in range(i + 2, n_res):  # 跳过相邻残基
                residue_pairs.append([i, j])

        if not residue_pairs:
            return [], []

        # 转换为numpy数组
        residue_pairs = np.array(residue_pairs)

        # 计算这些残基对之间的最小距离
        # 使用'ca'方案只考虑alpha碳原子之间的距离
        distances, _ = md.compute_contacts(structure, contacts=residue_pairs, scheme='ca')

        # 过滤出小于截断值的接触
        # 注意：distances的shape是(n_frames, n_pairs)
        contacts = distances[0] <= cutoff  # 使用第一帧

        # 获取满足接触条件的残基对
        contact_pairs = residue_pairs[contacts]
        contact_distances = distances[0][contacts]

        return contact_distances, contact_pairs
    except Exception as e:
        logger.error(f"计算接触图失败: {str(e)}")
        logger.error(traceback.format_exc())
        return [], []


def compute_hydrogen_bonds(structure):
    """识别氢键"""
    try:
        hbonds = md.baker_hubbard(structure, freq=0.1, periodic=False)
        return hbonds
    except Exception as e:
        logger.error(f"计算氢键失败: {str(e)}")
        return []


def compute_hydrophobic_contacts(structure, cutoff=0.5):
    """识别疏水相互作用（兼容新版MDTraj）"""
    try:
        # 定义疏水氨基酸索引
        hydrophobic_indices = []

        for i, res in enumerate(structure.topology.residues):
            res_name = res.name
            one_letter = three_to_one(res_name)
            if one_letter in 'AVILMFYW':  # 疏水氨基酸
                hydrophobic_indices.append(i)

        # 计算疏水残基间的接触
        if len(hydrophobic_indices) >= 2:
            pairs = []
            for i in range(len(hydrophobic_indices)):
                for j in range(i + 1, len(hydrophobic_indices)):
                    pairs.append([hydrophobic_indices[i], hydrophobic_indices[j]])

            pairs = np.array(pairs)
            # 计算这些残基对之间的距离
            distances = md.compute_contacts(structure, contacts=pairs, scheme='ca')[0][0]

            # 找出小于阈值的接触对
            contacts = pairs[distances < cutoff]
            return contacts
        else:
            return []
    except Exception as e:
        logger.error(f"计算疏水接触失败: {str(e)}")
        logger.error(traceback.format_exc())
        return []


def compute_ionic_interactions(structure, cutoff=0.4):
    """识别离子相互作用（盐桥）（兼容新版MDTraj）"""
    try:
        # 定义带电氨基酸索引
        pos_indices = []  # 正电荷 (K, R, H)
        neg_indices = []  # 负电荷 (D, E)

        for i, res in enumerate(structure.topology.residues):
            res_name = res.name
            one_letter = three_to_one(res_name)
            if one_letter in 'KRH':
                pos_indices.append(i)
            elif one_letter in 'DE':
                neg_indices.append(i)

        # 计算正负电荷残基间的接触对
        pairs = []
        for pos in pos_indices:
            for neg in neg_indices:
                pairs.append([pos, neg])

        if pairs:
            pairs = np.array(pairs)
            # 计算这些残基对之间的距离
            distances = md.compute_contacts(structure, contacts=pairs, scheme='ca')[0][0]

            # 找出小于阈值的接触对
            contacts = pairs[distances < cutoff]
            return contacts
        else:
            return []
    except Exception as e:
        logger.error(f"计算离子相互作用失败: {str(e)}")
        logger.error(traceback.format_exc())
        return []


def batch_nearest_neighbor_query(tree, points, k=10, max_distance=None, max_batch=10000):
    """
    分批执行K近邻查询以避免内存溢出

    参数:
        tree: cKDTree对象
        points: 查询点坐标数组
        k: 近邻数量
        max_distance: 最大距离限制
        max_batch: 单批次最大处理点数

    返回:
        distances: 距离数组，形状为(n_points, k)
        indices: 索引数组，形状为(n_points, k)
    """
    n_points = len(points)
    points = np.asarray(points, dtype=np.float32)

    # 预分配结果数组
    distances = np.zeros((n_points, k), dtype=np.float32)
    indices = np.zeros((n_points, k), dtype=np.int32)

    # 分批处理
    for start in range(0, n_points, max_batch):
        end = min(start + max_batch, n_points)
        batch_points = points[start:end]

        # 使用额外参数提高性能
        batch_dist, batch_idx = tree.query(
            batch_points,
            k=k,
            distance_upper_bound=max_distance if max_distance else np.inf,
            workers=-1,  # 使用多线程
            eps=0.01  # 允许近似查询以提高速度
        )

        distances[start:end] = batch_dist
        indices[start:end] = batch_idx

    return distances, indices


def compute_residue_distances_vectorized(ca_coords):
    """
    使用向量化操作高效计算所有残基对之间的距离

    参数:
        ca_coords: Alpha碳原子坐标数组，形状为(n_residues, 3)

    返回:
        distance_matrix: 距离矩阵，形状为(n_residues, n_residues)
    """
    # 优化数据类型减少内存使用
    ca_coords = np.asarray(ca_coords, dtype=np.float32)

    # 使用广播计算欧氏距离矩阵
    # 公式: √[(x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²]
    # 通过广播避免显式循环，显著提高速度
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1))

    return dist_matrix


def identify_interactions_vectorized(distance_matrix, residue_types, threshold=8.0):
    """
    向量化判断残基间相互作用类型

    参数:
        distance_matrix: 残基间距离矩阵
        residue_types: 残基类型列表
        threshold: 相互作用距离阈值

    返回:
        interaction_types: 相互作用类型矩阵 (0=无, 1=空间近邻, 2=氢键, 3=离子, 4=疏水)
    """
    n_res = len(residue_types)

    # 初始化交互矩阵
    interaction_types = np.zeros((n_res, n_res), dtype=np.int8)

    # 1. 空间距离过滤（矩阵操作）
    spatial_contacts = distance_matrix <= threshold
    interaction_types[spatial_contacts] = 1  # 标记为空间近邻

    # 对角线和相邻位置不考虑空间近邻
    np.fill_diagonal(interaction_types, 0)  # 清除对角线
    for i in range(n_res - 1):
        interaction_types[i, i + 1] = 0
        interaction_types[i + 1, i] = 0

    # 2. 氢键相互作用（向量化计算）
    hbond_donors = np.array([aa in 'NQRKWST' for aa in residue_types])
    hbond_acceptors = np.array([aa in 'DEQNSTYHW' for aa in residue_types])

    # 创建供体与受体的交叉矩阵
    donors_matrix = hbond_donors[:, np.newaxis] & hbond_acceptors[np.newaxis, :]
    acceptors_matrix = hbond_acceptors[:, np.newaxis] & hbond_donors[np.newaxis, :]
    potential_hbonds = donors_matrix | acceptors_matrix

    # 应用距离条件（氢键通常<3.5Å，这里用5.0作为上限）
    hbond_contacts = potential_hbonds & (distance_matrix < 5.0)
    interaction_types[hbond_contacts] = 2  # 标记为氢键

    # 3. 疏水相互作用（向量化计算）
    hydrophobic_residues = np.array([aa in 'AVILMFYW' for aa in residue_types])
    hydrophobic_matrix = hydrophobic_residues[:, np.newaxis] & hydrophobic_residues[np.newaxis, :]
    hydrophobic_contacts = hydrophobic_matrix & (distance_matrix < 6.0)
    interaction_types[hydrophobic_contacts] = 4  # 标记为疏水相互作用

    # 4. 离子相互作用（向量化计算）
    positive_residues = np.array([aa in 'KRH' for aa in residue_types])
    negative_residues = np.array([aa in 'DE' for aa in residue_types])

    pos_neg_matrix = positive_residues[:, np.newaxis] & negative_residues[np.newaxis, :]
    neg_pos_matrix = negative_residues[:, np.newaxis] & positive_residues[np.newaxis, :]
    potential_ionic = pos_neg_matrix | neg_pos_matrix

    ionic_contacts = potential_ionic & (distance_matrix < 6.0)
    interaction_types[ionic_contacts] = 3  # 标记为离子相互作用

    return interaction_types


def extract_plddt_from_bfactor(structure):
    try:
        # 正确获取B因子
        b_factors = []

        # 尝试从结构直接获取B因子
        has_b_factors = hasattr(structure, 'b_factors')

        for atom in structure.topology.atoms:
            res_idx = atom.residue.index

            # 正确获取B因子
            if has_b_factors:
                b_factor = structure.b_factors[atom.index]
            else:
                # 如果没有B因子，可以使用默认值
                b_factor = 70.0

            if len(b_factors) <= res_idx:
                b_factors.extend([0] * (res_idx - len(b_factors) + 1))
            b_factors[res_idx] += b_factor / atom.residue.n_atoms

        return np.array(b_factors)
    except Exception as e:
        logger.error(f"从B因子提取pLDDT失败: {str(e)}")
        return np.full(structure.n_residues, 70.0)  # 默认中等置信度

# ======================= 蛋白质切割策略 =======================
def create_intelligent_fragments(structure, ss_array, contact_map, residue_pairs,
                                       min_length=5, max_length=50, respect_ss=True, respect_domains=True):
    """
    创建智能蛋白质结构片段的全面NumPy向量化实现

    根据二级结构、结构域边界和空间接触特征智能划分片段，充分利用NumPy向量操作

    优化点:
    1. 全面NumPy向量化操作
    2. 批量边界检测和过滤
    3. 高效内存管理与数据结构
    4. 矩阵化计算代替循环
    5. 高性能数值分析技术

    参数:
        structure: MDTraj结构对象
        ss_array: 二级结构数组
        contact_map: 接触图距离列表
        residue_pairs: 接触残基对索引
        min_length: 最小片段长度
        max_length: 最大片段长度
        respect_ss: 是否遵循二级结构边界
        respect_domains: 是否遵循结构域边界

    返回:
        fragments: 片段列表，每个元素为(start_idx, end_idx, fragment_id)元组
    """
    n_residues = structure.n_residues

    try:
        # 1. 初始化边界数组 - 使用NumPy数组进行高效操作
        boundaries = np.array([0, n_residues], dtype=np.int32)

        # 2. 识别结构域边界 (如果需要) - 完全向量化
        if respect_domains and contact_map is not None and len(contact_map) > 0 and residue_pairs is not None:
            try:
                # 2.1 创建接触密度矩阵 - 使用高效的稀疏表示
                density_matrix = np.zeros((n_residues, n_residues), dtype=np.float32)

                # 向量化填充接触矩阵
                if len(residue_pairs) > 0:
                    # 批量过滤有效接触对
                    valid_pairs_mask = (residue_pairs[:, 0] < n_residues) & (residue_pairs[:, 1] < n_residues)
                    valid_pairs = residue_pairs[valid_pairs_mask]

                    if len(valid_pairs) > 0:
                        # 批量填充矩阵
                        rows = valid_pairs[:, 0]
                        cols = valid_pairs[:, 1]
                        density_matrix[rows, cols] = 1.0
                        density_matrix[cols, rows] = 1.0  # 对称填充

                # 2.2 计算接触密度向量 - 向量化求和
                contact_density = np.sum(density_matrix, axis=1)

                # 2.3 应用平滑与梯度计算 - 信号处理向量化技术
                window_size = min(5, max(3, n_residues // 50))  # 动态窗口大小

                # 高效卷积平滑
                kernel = np.ones(window_size) / window_size
                try:
                    from scipy.signal import convolve
                    smoothed_density = convolve(contact_density, kernel, mode='same')
                except ImportError:
                    # 纯NumPy实现的平滑
                    pad_width = window_size // 2
                    padded = np.pad(contact_density, pad_width, mode='edge')
                    smoothed_density = np.zeros_like(contact_density)

                    # 高效实现滑动窗口
                    cumsum = np.cumsum(padded)
                    smoothed_density = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

                # 计算密度梯度
                density_gradient = np.abs(np.gradient(smoothed_density))

                # 2.4 归一化梯度并检测高变化点
                if np.max(density_gradient) > 0:
                    norm_gradient = density_gradient / np.max(density_gradient)

                    # 自适应阈值 - 数据驱动的识别策略
                    threshold = max(0.2, np.mean(norm_gradient) + np.std(norm_gradient))
                    domain_boundary_indices = np.where(norm_gradient > threshold)[0]

                    # 过滤太近的边界 - 向量化方法
                    if len(domain_boundary_indices) > 0:
                        # 初始化过滤边界列表
                        filtered_bounds = [domain_boundary_indices[0]]

                        # 向量化计算间距
                        for idx in domain_boundary_indices[1:]:
                            if idx - filtered_bounds[-1] >= min_length:
                                filtered_bounds.append(idx)

                        # 更新边界数组
                        domain_boundaries = np.array(filtered_bounds, dtype=np.int32)
                        boundaries = np.union1d(boundaries, domain_boundaries)

            except Exception as e:
                logger.debug(f"结构域边界识别失败 (向量化): {str(e)[:100]}")

        # 3. 识别二级结构边界 (如果需要) - 全向量化实现
        if respect_ss and ss_array is not None:
            try:
                # 3.1 获取一致的二级结构表示
                ss = ss_array[0] if ss_array.ndim > 1 else ss_array

                # 确保为NumPy数组
                if not isinstance(ss, np.ndarray):
                    ss = np.array(list(ss))

                # 3.2 将字符二级结构编码为数值 - 高效映射
                # 创建映射字典
                ss_type_map = {
                    'H': 1, 'G': 1, 'I': 1,  # 所有螺旋类型
                    'E': 2, 'B': 2,  # 所有折叠类型
                    'T': 3, 'S': 3, 'C': 3  # 所有卷曲/环类型
                }

                # 批量转换为数值表示
                ss_numeric = np.zeros(len(ss), dtype=np.int8)

                # 向量化处理每种类型
                for ss_type, type_code in ss_type_map.items():
                    ss_numeric[ss == ss_type] = type_code

                # 3.3 检测二级结构变化点 - 差分向量化
                ss_diff = np.diff(ss_numeric)
                ss_changes = np.where(ss_diff != 0)[0] + 1

                # 3.4 合并到边界列表
                if len(ss_changes) > 0:
                    boundaries = np.union1d(boundaries, ss_changes)

            except Exception as e:
                logger.debug(f"二级结构边界识别失败 (向量化): {str(e)[:100]}")

        # 4. 识别柔性区域边界 - 向量化实现
        try:
            # 4.1 批量提取CA原子索引
            ca_indices = np.array([atom.index for atom in structure.topology.atoms
                                   if atom.name == 'CA' and
                                   atom.index < len(structure.xyz[0])])

            if len(ca_indices) > 0:
                # 4.2 获取B因子或使用坐标作为替代
                b_factors = None

                # 尝试使用原始B因子
                if hasattr(structure, 'b_factors') and len(structure.b_factors) > 0:
                    # 向量化提取有效B因子
                    valid_b_indices = ca_indices[ca_indices < len(structure.b_factors)]
                    if len(valid_b_indices) > 0:
                        b_factors = structure.b_factors[valid_b_indices]

                # 如果没有B因子，使用坐标平方和作为替代
                if b_factors is None or len(b_factors) < 0.5 * n_residues:
                    # 向量化计算坐标平方和
                    valid_ca = ca_indices[ca_indices < len(structure.xyz[0])]
                    if len(valid_ca) > 0:
                        b_factors = np.sum(structure.xyz[0, valid_ca] ** 2, axis=1)

                # 4.3 确保B因子长度与残基数匹配
                if b_factors is not None and len(b_factors) > 0:
                    # 规范化长度处理
                    if len(b_factors) < n_residues:
                        # 向量化填充数组
                        mean_b = np.mean(b_factors)
                        b_factors = np.pad(b_factors,
                                           (0, n_residues - len(b_factors)),
                                           mode='constant',
                                           constant_values=mean_b)
                    elif len(b_factors) > n_residues:
                        b_factors = b_factors[:n_residues]

                    # 4.4 向量化平滑计算
                    window_size = min(5, max(3, n_residues // 50))
                    kernel = np.ones(window_size) / window_size

                    # 使用卷积进行平滑
                    smoothed_b = np.convolve(b_factors, kernel, mode='same')

                    # 4.5 梯度分析识别变化点
                    b_gradient = np.abs(np.gradient(smoothed_b))

                    # 归一化和阈值选择
                    if np.max(b_gradient) > 0:
                        norm_b_gradient = b_gradient / np.max(b_gradient)

                        # 自适应阈值
                        mean_grad = np.mean(norm_b_gradient)
                        std_grad = np.std(norm_b_gradient)
                        threshold = max(0.3, mean_grad + 0.5 * std_grad)

                        # 向量化检测高梯度点
                        flex_boundaries = np.where(norm_b_gradient > threshold)[0]

                        # 更新边界列表
                        if len(flex_boundaries) > 0:
                            boundaries = np.union1d(boundaries, flex_boundaries)
        except Exception as e:
            logger.debug(f"柔性区域边界识别失败 (向量化): {str(e)[:100]}")

        # 5. 添加线性分割边界 - 确保没有过大片段
        linear_bounds = np.arange(0, n_residues, max_length)
        if linear_bounds[-1] != n_residues:
            linear_bounds = np.append(linear_bounds, n_residues)

        # 合并所有边界
        boundaries = np.union1d(boundaries, linear_bounds)

        # 6. 规范化边界: 限制范围、排序、去重
        boundaries = np.clip(boundaries, 0, n_residues)
        boundaries = np.unique(boundaries).astype(np.int32)

        # 7. 向量化生成最终片段列表
        fragments = []

        # 7.1 处理所有相邻边界对
        for i in range(len(boundaries) - 1):
            start = int(boundaries[i])
            end = int(boundaries[i + 1])
            length = end - start

            # 跳过过小片段
            if length < min_length:
                continue

            # 7.2 根据长度决定处理策略
            if length <= max_length:
                # 片段长度适中，直接添加
                frag_id = f"{start + 1}-{end}"
                fragments.append((start, end, frag_id))
            else:
                # 片段过长，使用滑动窗口切割
                # 向量化计算滑动窗口起始位置
                step = max(min_length, max_length // 2)
                window_starts = np.arange(start, end, step)

                for win_start in window_starts:
                    win_end = min(win_start + max_length, end)
                    win_length = win_end - win_start

                    if win_length >= min_length:
                        frag_id = f"{win_start + 1}-{win_end}"
                        fragments.append((win_start, win_end, frag_id))

                    # 如果已覆盖到末端，退出循环
                    if win_end >= end:
                        break

        # 8. 确保至少有一个片段 - 当没有生成有效片段时
        if not fragments and n_residues >= min_length:
            # 向量化生成均匀分布的片段
            step = max(min_length, max_length // 2)
            starts = np.arange(0, n_residues, step)

            for start in starts:
                end = min(start + max_length, n_residues)
                length = end - start

                if length >= min_length:
                    frag_id = f"{start + 1}-{end}"
                    fragments.append((start, end, frag_id))

                if end >= n_residues:
                    break

        return fragments

    except Exception as e:
        logger.error(f"智能片段创建失败 (向量化): {str(e)}")
        logger.debug(traceback.format_exc())

        # 故障恢复：使用简单的线性分割 - 向量化实现
        fallback_fragments = []
        step = max(min_length, max_length // 2)

        # 向量化生成线性片段
        starts = np.arange(0, n_residues, step)
        for start in starts:
            end = min(start + max_length, n_residues)
            if end - start >= min_length:
                frag_id = f"{start + 1}-{end}"
                fallback_fragments.append((start, end, frag_id))
            if end >= n_residues:
                break

        return fallback_fragments

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


def gpu_compute_residue_distances(ca_coords, device_id=None, memory_threshold=0.7):
    """
    使用向量化方法计算残基间距离矩阵，支持指定GPU设备或智能选择最优GPU

    参数:
        ca_coords: Alpha碳原子坐标数组，形状为(n_residues, 3)
        device_id: 指定使用的GPU设备ID，None表示自动选择，-1表示强制使用CPU
        memory_threshold: GPU内存占用阈值，超过此阈值的设备将不会被自动选择

    返回:
        distance_matrix: 距离矩阵，形状为(n_residues, n_residues)
    """
    # 检查输入数据有效性
    if ca_coords is None or len(ca_coords) < 2:
        return np.zeros((0, 0), dtype=np.float32)

    # 数据规模检查 - 对小规模数据直接使用CPU计算
    n_residues = len(ca_coords)
    matrix_size_mb = (n_residues * n_residues * 4) / (1024 * 1024)  # 估计矩阵大小(MB)

    try:
        # 检查是否有可用的GPU
        if torch.cuda.is_available():
            # 获取可用的GPU数量
            n_gpus = torch.cuda.device_count()

            if n_gpus == 0:
                return _cpu_compute_distances(ca_coords)

            # 选择GPU设备
            selected_device = 3

            # 如果指定了设备ID
            if device_id is not None:
                if 0 <= device_id < n_gpus:
                    selected_device = device_id
                    logger.debug(f"使用指定的GPU设备: {device_id}")
                else:
                    logger.warning(f"指定的GPU设备ID {device_id} 超出范围(0-{n_gpus - 1})，将自动选择GPU")

            # 如果未指定设备ID或指定的设备无效，自动选择
            if selected_device is None:
                # 找到内存占用最少的GPU
                max_free_memory = -1
                for i in range(n_gpus):
                    try:
                        # 获取当前设备内存统计
                        torch.cuda.set_device(i)
                        free_memory = torch.cuda.get_device_properties(i).total_memory
                        free_memory -= torch.cuda.memory_allocated(i)
                        free_memory_mb = free_memory / (1024 * 1024)

                        # 计算内存占用率
                        total_memory_mb = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                        memory_usage = 1.0 - (free_memory_mb / total_memory_mb)

                        logger.debug(f"GPU:{i} 内存占用率: {memory_usage:.2f}, 空闲内存: {free_memory_mb:.2f}MB")

                        # 检查是否有足够内存计算距离矩阵
                        if memory_usage < memory_threshold and free_memory_mb > matrix_size_mb * 3 and free_memory_mb > max_free_memory:
                            selected_device = i
                            max_free_memory = free_memory_mb
                    except Exception as e:
                        logger.debug(f"检查GPU {i}时出错: {str(e)[:100]}")

                if selected_device is not None:
                    logger.debug(f"自动选择GPU {selected_device}，空闲内存: {max_free_memory:.2f}MB")
                else:
                    logger.debug(f"没有找到符合条件的GPU（内存阈值:{memory_threshold}），所有设备内存占用过高或空间不足")
                    return _cpu_compute_distances(ca_coords)

            # 设置到选择的设备
            device = torch.device(f"cuda:{selected_device}")
            torch.cuda.set_device(device)

            # 转换为PyTorch张量并移至所选GPU
            coords_tensor = torch.tensor(ca_coords, dtype=torch.float32).to(device)

            # 高效计算距离矩阵 - 使用广播避免循环
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2a·b
            a_square = torch.sum(coords_tensor ** 2, dim=1, keepdim=True)
            b_square = torch.sum(coords_tensor ** 2, dim=1).unsqueeze(0)
            ab = torch.matmul(coords_tensor, coords_tensor.transpose(0, 1))

            dist_square = a_square + b_square - 2 * ab
            # 处理数值误差导致的负值
            dist_square = torch.clamp(dist_square, min=0.0)

            # 计算距离
            distances = torch.sqrt(dist_square)

            # 将结果转回CPU并转为NumPy数组
            return distances.cpu().numpy()

    except Exception as e:
        logger.warning(f"GPU距离计算失败: {str(e)[:100]}，回退到CPU计算")

    # 如果GPU计算失败，回退到CPU计算
    return _cpu_compute_distances(ca_coords)


def _cpu_compute_distances(ca_coords):
    """
    使用CPU进行残基间距离计算的优化实现
    """
    # 使用NumPy的向量化广播计算距离
    try:
        # 检查是否可以使用更快的SciPy实现
        try:
            from scipy.spatial.distance import pdist, squareform
            # pdist计算的是压缩的距离矩阵，squareform转换为方阵
            condensed_distances = pdist(ca_coords, metric='euclidean')
            return squareform(condensed_distances)
        except ImportError:
            # 回退到纯NumPy实现
            diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
            return np.sqrt(np.sum(diff * diff, axis=2))
    except Exception as e:
        logger.warning(f"CPU距离计算出错: {str(e)}")
        # 最终回退方案：循环计算
        n = len(ca_coords)
        distances = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((ca_coords[i] - ca_coords[j]) ** 2))
                distances[i, j] = distances[j, i] = d
        return distances

# ======================= 知识图谱构建函数 =======================

def build_enhanced_residue_graph(structure, ss_array, fragment_range,
                                 k_neighbors=8, distance_threshold=8.0, plddt_threshold=70):
    """
    高性能残基知识图谱构建函数

    优化点:
    - 全面NumPy向量化操作
    - 批量特征计算
    - 矩阵化节点属性生成
    - 并行边构建策略
    - 内存优化数据结构

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
        # 1. 高效预处理 - 向量化方式获取所有数据
        # 获取pLDDT值 (向量化)
        plddt_values = extract_plddt_from_bfactor(structure)

        # 创建片段的索引范围数组
        fragment_indices = np.arange(start_idx, end_idx)

        # 1.1 批量过滤有效残基 (向量化)
        # 创建pLDDT过滤掩码
        plddt_mask = np.ones(len(fragment_indices), dtype=bool)
        valid_indices_in_plddt = np.where(fragment_indices < len(plddt_values))[0]

        if len(valid_indices_in_plddt) > 0:
            valid_plddt = plddt_values[fragment_indices[valid_indices_in_plddt]]
            plddt_mask[valid_indices_in_plddt] = valid_plddt >= plddt_threshold
            debug_stats['filtered_by_plddt'] = np.sum(~plddt_mask)

        # 初始过滤后的索引
        filtered_indices = fragment_indices[plddt_mask]

        # 1.2 提取残基名称和CA坐标 (批量处理)
        residue_names = []
        ca_indices = []

        # 一次性获取所有残基对象 (这部分难以向量化，因为依赖MDTraj API)
        for res_idx in filtered_indices:
            try:
                res = structure.topology.residue(res_idx)
                residue_names.append(res.name)

                # 查找CA原子索引
                for atom in res.atoms:
                    if atom.name == 'CA':
                        ca_indices.append(atom.index)
                        break
                else:
                    ca_indices.append(-1)  # 没找到CA原子
            except:
                residue_names.append("UNK")
                ca_indices.append(-1)

        # 转换为NumPy数组
        residue_names = np.array(residue_names)
        ca_indices = np.array(ca_indices)

        # 1.3 转换残基名称为单字母编码 (向量化)
        residue_codes = np.array([three_to_one(name) for name in residue_names])

        # 1.4 过滤非标准氨基酸 (向量化)
        standard_aa_mask = residue_codes != 'X'
        ca_found_mask = ca_indices >= 0
        valid_mask = standard_aa_mask & ca_found_mask

        debug_stats['filtered_by_nonstandard'] = np.sum(~standard_aa_mask)

        # 1.5 最终有效残基过滤
        valid_indices = filtered_indices[valid_mask]
        valid_ca_indices = ca_indices[valid_mask]
        valid_residue_codes = residue_codes[valid_mask]

        # 检查有效节点数量
        if len(valid_indices) < 2:
            logger.warning(f"有效节点数量不足 ({len(valid_indices)}): {fragment_id}")
            return None

        # 2. 提取坐标并标准化 (批量操作)
        ca_coords = structure.xyz[0, valid_ca_indices]
        normalized_coords = normalize_coordinates(ca_coords)

        # 3. 创建节点ID列表 (向量化)
        node_ids = np.array([f"res_{idx}" for idx in valid_indices])

        # 4. 批量处理二级结构信息
        # 创建二级结构编码映射
        SS_TYPE_TO_IDX = {'H': 0, 'G': 0, 'I': 0,  # 螺旋
                          'E': 1, 'B': 1,  # 折叠
                          'T': 2, 'S': 2, 'C': 2}  # 卷曲

        # 4.1 批量提取二级结构代码
        ss_codes = np.full(len(valid_indices), 'C', dtype='U1')  # 默认为卷曲

        for i, res_idx in enumerate(valid_indices):
            if ss_array.ndim > 1 and res_idx < ss_array.shape[1]:
                ss_codes[i] = ss_array[0, res_idx]
            elif ss_array.ndim == 1 and res_idx < len(ss_array):
                ss_codes[i] = ss_array[res_idx]

        # 4.2 批量计算二级结构One-hot编码
        ss_indices = np.array([SS_TYPE_TO_IDX.get(ss, 2) for ss in ss_codes])
        ss_onehot = np.zeros((len(valid_indices), 3), dtype=np.float32)
        for i in range(len(valid_indices)):
            ss_onehot[i, ss_indices[i]] = 1.0

        # 5. 批量构建节点属性
        # 5.1 获取理化特性向量
        # 假设这是一个映射氨基酸到其特性向量的字典
        AA_PROPERTY_VECTORS = {aa: np.array([
            AA_PROPERTIES[aa]['hydropathy'],
            AA_PROPERTIES[aa]['charge'],
            AA_PROPERTIES[aa]['mw'] / 200.0,  # 归一化
            AA_PROPERTIES[aa]['volume'] / 200.0,  # 归一化
            AA_PROPERTIES[aa]['flexibility'],
            1.0 if AA_PROPERTIES[aa]['aromatic'] else 0.0
        ], dtype=np.float32) for aa in AA_PROPERTIES}

        # 5.2 批量获取特性向量
        property_vectors = np.array([AA_PROPERTY_VECTORS.get(aa, AA_PROPERTY_VECTORS['X'])
                                     for aa in valid_residue_codes])

        # 5.3 批量获取BLOSUM编码
        blosum_vectors = np.array([get_blosum62_encoding(aa) for aa in valid_residue_codes])

        # 5.4 批量构建所有节点
        for i in range(len(valid_indices)):
            node_id = node_ids[i]
            res_idx = valid_indices[i]

            # 构建节点属性
            node_attrs = {
                'residue_name': residue_names[valid_mask][i],
                'residue_code': valid_residue_codes[i],
                'residue_idx': res_idx,
                'position': normalized_coords[i].tolist(),
                'plddt': float(plddt_values[res_idx]) if res_idx < len(plddt_values) else 70.0,

                # 氨基酸理化特性 (使用预计算的向量)
                'hydropathy': float(property_vectors[i, 0]),
                'charge': float(property_vectors[i, 1]),
                'molecular_weight': float(property_vectors[i, 2] * 200.0),  # 反归一化
                'volume': float(property_vectors[i, 3] * 200.0),  # 反归一化
                'flexibility': float(property_vectors[i, 4]),
                'is_aromatic': bool(property_vectors[i, 5]),

                # 二级结构信息 (使用预计算的One-hot)
                'secondary_structure': ss_codes[i],
                'ss_alpha': float(ss_onehot[i, 0]),
                'ss_beta': float(ss_onehot[i, 1]),
                'ss_coil': float(ss_onehot[i, 2]),

                # 序列信息
                'blosum62': blosum_vectors[i].tolist(),
                'fragment_id': fragment_id
            }

            graph.add_node(node_id, **node_attrs)

        debug_stats['valid_nodes'] = len(valid_indices)

        # 6. 批量添加序列连接边 (向量化)
        # 6.1 找到序列相连的残基对 (向量化)
        idx_pairs = np.column_stack((valid_indices[:-1], valid_indices[1:]))
        seq_adjacent = idx_pairs[:, 1] - idx_pairs[:, 0] == 1

        # 6.2 计算相邻残基间距离 (向量化)
        if np.any(seq_adjacent):
            seq_indices = np.where(seq_adjacent)[0]
            seq_distances = np.linalg.norm(
                normalized_coords[seq_indices + 1] - normalized_coords[seq_indices],
                axis=1
            )

            # 6.3 批量创建边数据
            seq_edges = []
            for i, dist_idx in enumerate(seq_indices):
                seq_edges.append((
                    node_ids[dist_idx],
                    node_ids[dist_idx + 1],
                    {
                        'edge_type': 1,
                        'type_name': 'peptide',
                        'distance': float(seq_distances[i]),
                        'interaction_strength': 1.0,
                        'direction': [1.0, 0.0]  # N->C方向
                    }
                ))

            # 批量添加序列边
            if seq_edges:
                graph.add_edges_from(seq_edges)
                debug_stats['edges_created'] += len(seq_edges)

        # 7. 批量添加空间相互作用边
        # 7.1 构建KD树并批量查询K近邻
        try:
            kd_tree = KDTree(normalized_coords)

            # 批量查询所有点的K近邻 (向量化)
            k = min(k_neighbors + 1, len(normalized_coords))
            distances, indices = kd_tree.query(normalized_coords, k=k)

            # 7.2 计算所有残基间的距离矩阵 (使用GPU加速)
            distance_matrix = gpu_compute_residue_distances(normalized_coords)

            # 7.3 批量分类相互作用类型 (向量化)
            interaction_matrix = classify_interaction_vectorized(
                valid_residue_codes, distance_matrix, threshold=distance_threshold)

            # 7.4 批量处理相互作用边
            space_edges = []

            # 创建边类型到属性的映射 (避免重复计算)
            edge_type_props = {
                0: ('none', 0.0),
                1: ('spatial', 0.3),
                2: ('hbond', 0.8),
                3: ('ionic', 0.7),
                4: ('hydrophobic', 0.5),
                5: ('unknown', 0.2)
            }

            # 高效的边批处理
            for i in range(len(normalized_coords)):
                neighbors = indices[i, 1:]  # 跳过自身
                neighbor_dists = distances[i, 1:]

                # 过滤掉超出距离阈值的邻居 (向量化)
                valid_neighbors_mask = neighbor_dists <= distance_threshold
                valid_neighbors = neighbors[valid_neighbors_mask]
                valid_dists = neighbor_dists[valid_neighbors_mask]

                if len(valid_neighbors) == 0:
                    continue

                # 过滤序列相邻残基 (向量化)
                res_i = valid_indices[i]
                res_j = valid_indices[valid_neighbors]
                non_adjacent_mask = np.abs(res_j - res_i) > 1

                valid_neighbors = valid_neighbors[non_adjacent_mask]
                valid_dists = valid_dists[non_adjacent_mask]

                if len(valid_neighbors) == 0:
                    continue

                # 过滤无相互作用的残基对 (向量化)
                edge_types = interaction_matrix[i, valid_neighbors]
                interacting_mask = edge_types > 0

                valid_neighbors = valid_neighbors[interacting_mask]
                valid_dists = valid_dists[interacting_mask]
                valid_edge_types = edge_types[interacting_mask]

                if len(valid_neighbors) == 0:
                    continue

                # 批量计算方向向量 (向量化)
                directions = normalized_coords[valid_neighbors] - normalized_coords[i]
                norms = np.linalg.norm(directions, axis=1)
                norms_mask = norms > 0

                # 创建方向向量 (处理零向量情况)
                dir_vecs = np.zeros((len(directions), 2), dtype=np.float32)
                if np.any(norms_mask):
                    dir_vecs[norms_mask] = directions[norms_mask, :2] / norms[norms_mask].reshape(-1, 1)

                # 批量创建边
                for j in range(len(valid_neighbors)):
                    j_idx = valid_neighbors[j]
                    edge_type = valid_edge_types[j]

                    # 获取边属性
                    type_name, interaction_strength = edge_type_props.get(edge_type, edge_type_props[5])

                    # 添加边
                    space_edges.append((
                        node_ids[i],
                        node_ids[j_idx],
                        {
                            'edge_type': int(edge_type),
                            'type_name': type_name,
                            'distance': float(valid_dists[j]),
                            'interaction_strength': interaction_strength,
                            'direction': dir_vecs[j].tolist()
                        }
                    ))

            # 批量添加空间边
            if space_edges:
                graph.add_edges_from(space_edges)
                debug_stats['edges_created'] += len(space_edges)

        except Exception as e:
            logger.error(f"空间边构建失败: {str(e)[:100]}")
            # 继续处理，即使空间边失败

        # 8. 检查结果图
        if graph.number_of_nodes() < 2:
            logger.warning(f"节点数量不足 ({graph.number_of_nodes()}): {fragment_id}")
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


# ======================= 文件处理函数 =======================
def process_structure_file(file_path, min_length=5, max_length=50, k_neighbors=8,
                           distance_threshold=8.0, plddt_threshold=70.0, respect_ss=True, respect_domains=True):
    """
    高性能向量化优化版蛋白质结构处理函数 - 增强容错机制与计算效率
    提取片段并构建知识图谱

    优化点:
    - 全面NumPy向量化操作替代循环
    - 批量特征计算与数据处理
    - 优化内存访问模式
    - 严格容错与性能监控
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
        "skipped_fragments": 0,  # 记录因计算失败而跳过的片段数
        "processing_time": 0
    }

    start_time = time.time()

    try:
        # 1. 高效加载结构
        structure = load_structure_mdtraj(file_path)
        if structure is None:
            logger.error(f"无法加载结构: {file_path}")
            return {}, {}, fragment_stats

        # 2. 计算二级结构 - 向量化处理
        ss_array = None
        ss_computation_failed = False
        try:
            ss_array = compute_secondary_structure(structure)
            if ss_array is None or len(ss_array) == 0:
                logger.warning(f"{protein_id} 二级结构计算结果为空")
                ss_computation_failed = True
            else:
                # 向量化转换为NumPy数组便于处理
                ss_array = np.array(ss_array) if not isinstance(ss_array, np.ndarray) else ss_array
        except Exception as e:
            logger.warning(f"{protein_id} 二级结构计算失败: {str(e)[:100]}")
            ss_computation_failed = True

        # 如果二级结构计算失败但不要求respect_ss，我们可以继续处理
        if ss_computation_failed and respect_ss:
            logger.warning(f"由于二级结构计算失败且需要遵循二级结构边界(respect_ss=True)，跳过处理 {protein_id}")
            fragment_stats["processing_time"] = time.time() - start_time
            return {}, {}, fragment_stats

        # 3. 计算接触图 - 向量化实现
        contact_map = None
        residue_pairs = None
        contact_computation_failed = False
        ca_coords = None  # 保存CA坐标供后续使用

        try:
            # 向量化提取CA原子索引
            ca_indices = np.array([atom.index for atom in structure.topology.atoms if atom.name == 'CA'])

            if len(ca_indices) > 0:
                # 批量提取CA坐标
                ca_coords = structure.xyz[0, ca_indices]

                # 使用向量化方法计算距离矩阵
                dist_matrix = np.zeros((len(ca_indices), len(ca_indices)), dtype=np.float32)

                # 使用广播机制计算欧氏距离
                for i in range(len(ca_indices)):
                    # 使用NumPy广播避免显式循环
                    diff = ca_coords[i:i + 1] - ca_coords
                    dist_matrix[i] = np.sqrt(np.sum(diff * diff, axis=1))

                # 向量化提取接触残基对
                # 创建所有可能的残基对索引
                i_indices, j_indices = np.triu_indices(len(ca_indices), k=2)  # 上三角矩阵索引，跳过相邻残基

                # 应用距离阈值过滤
                contact_mask = dist_matrix[i_indices, j_indices] <= 0.8  # 0.8 nm 截断值

                if np.any(contact_mask):
                    # 提取接触残基对
                    contact_i = i_indices[contact_mask]
                    contact_j = j_indices[contact_mask]

                    # 提取接触距离
                    contact_distances = dist_matrix[contact_i, contact_j]

                    # 构建接触图
                    contact_map = contact_distances
                    residue_pairs = np.column_stack((contact_i, contact_j))
                else:
                    logger.warning(f"{protein_id} 未找到接触残基对")
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

        # 4. 向量化提取残基编码和计算有效残基数量
        valid_residues = 0
        res_codes = []
        residue_extraction_failed = False

        try:
            # 批量获取残基名称和对应编码
            residue_names = np.array([res.name for res in structure.topology.residues])

            # 向量化转换三字母到单字母编码
            res_codes = np.array([three_to_one(name) for name in residue_names])

            # 向量化计算有效残基数量
            valid_residues = np.sum(res_codes != 'X')

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

        # 6. 创建智能片段 - 使用优化的向量化函数
        fragments = []
        fragment_creation_failed = False

        try:
            # 调用优化的create_intelligent_fragments函数，整合多种边界检测方法
            fragments = create_intelligent_fragments(
                structure,
                ss_array,
                contact_map,
                residue_pairs,
                min_length=min_length,
                max_length=max_length,
                respect_ss=respect_ss,
                respect_domains=respect_domains
            )

            # 检查片段创建结果
            if not fragments:
                logger.warning(f"{protein_id} 未生成有效片段")
                fragment_creation_failed = True
            else:
                # 向量化过滤无效片段
                valid_fragments = []
                for start_idx, end_idx, frag_id in fragments:
                    if end_idx - start_idx >= min_length:
                        valid_fragments.append((start_idx, end_idx, frag_id))

                # 更新片段列表
                fragments = valid_fragments

                if not fragments:
                    logger.warning(f"{protein_id} 过滤后无有效片段")
                    fragment_creation_failed = True

        except Exception as e:
            logger.warning(f"{protein_id} 智能片段创建失败: {str(e)[:100]}")
            fragment_creation_failed = True

        # 如果智能片段创建失败，跳过
        if fragment_creation_failed:
            fragment_stats["processing_time"] = time.time() - start_time
            return {}, {}, fragment_stats

        # 更新统计信息
        fragment_stats["fragments"] = len(fragments)
        successful_fragments = 0

        # 7. 向量化处理每个片段
        from concurrent.futures import ThreadPoolExecutor

        # 创建片段处理函数
        def process_fragment(fragment_info):
            start_idx, end_idx, frag_id = fragment_info
            fragment_id = f"{protein_id}_{frag_id}"

            # 提取序列 (向量化)
            try:
                # 提取片段范围内的残基编码
                fragment_indices = np.arange(start_idx, end_idx)
                valid_indices = fragment_indices[fragment_indices < len(res_codes)]

                if len(valid_indices) == 0:
                    return None

                # 提取有效编码并过滤非标准氨基酸
                fragment_codes = res_codes[valid_indices]
                standard_aa_mask = fragment_codes != 'X'

                if not np.any(standard_aa_mask):
                    return None

                # 合并序列
                sequence = ''.join(fragment_codes[standard_aa_mask])

                # 如果序列太短，跳过
                if len(sequence) < min_length:
                    return None

                # 构建片段数据
                fragment_data = {
                    "protein_id": protein_id,
                    "fragment_id": frag_id,
                    "sequence": sequence,
                    "length": len(sequence),
                    "start_idx": start_idx,
                    "end_idx": end_idx
                }

                # 构建知识图谱
                if (respect_ss and ss_array is None) or (
                        respect_domains and (contact_map is None or residue_pairs is None)):
                    return None

                # 使用默认二级结构数组如果原始计算失败
                ss_data = ss_array if ss_array is not None else np.full((1, structure.n_residues), 'C')

                kg = build_enhanced_residue_graph(
                    structure, ss_data,
                    (start_idx, end_idx, fragment_id),
                    k_neighbors, distance_threshold, plddt_threshold
                )

                # 检查图谱有效性
                if kg is not None and kg.number_of_nodes() >= 2:
                    return fragment_id, fragment_data, nx.node_link_data(kg), kg.number_of_edges()

                return None

            except Exception as e:
                logger.debug(f"片段 {fragment_id} 处理失败: {str(e)[:50]}")
                return None

        # 使用线程池并行处理所有片段
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() * 2, len(fragments))) as executor:
            # 批量提交所有片段处理任务
            future_results = {executor.submit(process_fragment, frag): frag for frag in fragments}

            # 收集处理结果
            for future in concurrent.futures.as_completed(future_results):
                result = future.result()

                if result is None:
                    fragment_stats["failed_fragments"] += 1
                    continue

                fragment_id, fragment_data, graph_data, edge_count = result
                extracted_fragments[fragment_id] = fragment_data
                knowledge_graphs[fragment_id] = graph_data
                fragment_stats["edges"] += edge_count
                successful_fragments += 1

        # 更新统计
        fragment_stats["successful_fragments"] = successful_fragments
        fragment_stats["processing_time"] = time.time() - start_time

        # 如果没有成功处理任何片段，记录警告
        if successful_fragments == 0:
            logger.warning(f"{protein_id} 所有片段处理失败")

        # 释放内存 - 明确删除大数组
        del structure
        if 'ss_array' in locals() and ss_array is not None:
            del ss_array
        if 'contact_map' in locals() and contact_map is not None:
            del contact_map
        if 'residue_pairs' in locals() and residue_pairs is not None:
            del residue_pairs
        if 'ca_coords' in locals() and ca_coords is not None:
            del ca_coords

        return extracted_fragments, knowledge_graphs, fragment_stats

    except Exception as e:
        logger.error(f"处理文件 {file_path} 失败: {str(e)}")
        logger.error(traceback.format_exc())
        fragment_stats["processing_time"] = time.time() - start_time
        return {}, {}, fragment_stats

def find_pdb_files(root_dir):
    """递归搜索所有PDB和CIF文件，并去重"""
    pdb_files = []
    processed_ids = set()  # 记录已处理的蛋白质ID

    logger.info(f"正在搜索目录: {root_dir}")

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.pdb', '.pdb.gz', '.cif', '.cif.gz')):
                # 提取蛋白质ID (去除扩展名)
                protein_id = file.split('.')[0]

                # 跳过已处理的ID
                if protein_id in processed_ids:
                    continue

                processed_ids.add(protein_id)
                pdb_files.append(os.path.join(root, file))

    logger.info(f"找到 {len(pdb_files)} 个不重复的PDB/CIF文件")
    return pdb_files



def process_file_chunk(file_list, min_length=5, max_length=50, k_neighbors=8,
                       distance_threshold=8.0, plddt_threshold=70.0,
                       respect_ss=True, respect_domains=True):
    """优化的文件块处理函数 - 一次处理多个文件，减少进程创建开销"""
    results = []

    for file_path in file_list:
        try:
            start_time = time.time()
            # 处理文件
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
        except Exception as e:
            error_info = {"error": str(e), "file_path": file_path}
            results.append((error_info, {}, {}, None))

    return results

def process_file_parallel(file_list, output_dir, min_length=5, max_length=50,
                                 n_workers=None, batch_size=50000, memory_limit_gb=800,
                                 k_neighbors=8, distance_threshold=8.0, plddt_threshold=70.0,
                                 respect_ss=True, respect_domains=True, format_type="pyg"):
    """
    高性能蛋白质结构批处理系统 - 适用于TB级内存和百核处理器

    参数:
        file_list: 待处理文件列表
        output_dir: 输出目录
        batch_size: 每批处理的文件数量 (默认: 50000，适合TB级内存)
        memory_limit_gb: 内存使用上限(GB) (默认: 800GB，预留200GB系统使用)
        n_workers: 并行工作进程数 (默认: None, 使用CPU核心数-1)
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    # 为大规模处理创建更高效的目录结构
    base_data_dir = os.path.join(output_dir, "ALL")
    temp_dir = os.path.join(output_dir, "TEMP")
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # 预计算总处理批次
    total_files = len(file_list)
    total_batches = (total_files + batch_size - 1) // batch_size
    logger.info(f"总文件数: {total_files}, 划分为 {total_batches} 批次处理，每批次 {batch_size} 文件")
    logger.info(f"使用 {n_workers} 个CPU核心并行处理 (共112核)")
    logger.info(f"内存限制设置为 {memory_limit_gb}GB (总内存约1TB)")

    # 日志文件初始化
    sequences_log_path = os.path.join(output_dir, "sequences.log")
    fragments_log_path = os.path.join(output_dir, "fragments_stats.log")
    processing_log_path = os.path.join(output_dir, "processing.log")

    # 划分批次
    batches = [file_list[i:i + batch_size] for i in range(0, total_files, batch_size)]

    # 全局统计
    global_stats = {
        "processed_files": 0,
        "extracted_fragments": 0,
        "knowledge_graphs": 0,
        "failed_files": 0,
        "total_edges": 0
    }

    # 初始化日志文件
    with open(sequences_log_path, 'w', buffering=1) as s_log:
        s_log.write("fragment_id,protein_id,length,sequence\n")
    with open(fragments_log_path, 'w') as f_log:
        f_log.write("file_name,protein_id,valid_residues,fragments,edges\n")
    with open(processing_log_path, 'w') as p_log:
        p_log.write("timestamp,file_path,status,elapsed,fragments,knowledge_graphs,error\n")

    # 处理每个批次
    for batch_id, batch_files in enumerate(batches):
        logger.info(f"开始处理批次 {batch_id + 1}/{len(batches)} ({len(batch_files)} 文件)")
        batch_output_dir = os.path.join(base_data_dir, f"batch_{batch_id + 1}")
        os.makedirs(batch_output_dir, exist_ok=True)

        # 批次级缓存
        batch_proteins = {}
        batch_graphs = {}

        # 每批处理开始前，确保内存充足
        check_memory_usage(threshold_gb=memory_limit_gb, force_gc=True)

        # 使用优化的进程池和共享内存处理单个批次
        with tqdm(total=len(batch_files), desc=f"批次 {batch_id + 1} 处理进度") as pbar:
            # 采用更高效的并行策略 - 使用更大的任务块，减少进程间通信
            chunk_size = max(1, len(batch_files) // (n_workers * 4))  # 动态计算分块大小

            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=multiprocessing.get_context('spawn')  # 更稳定的进程创建方式
            ) as executor:
                futures = []
                for i in range(0, len(batch_files), chunk_size):
                    chunk_files = batch_files[i:i + chunk_size]
                    future = executor.submit(
                        process_file_chunk, chunk_files, min_length, max_length,
                        k_neighbors, distance_threshold, plddt_threshold, respect_ss, respect_domains
                    )
                    futures.append(future)

                # 收集结果，更新进度条
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results = future.result()
                        for result, proteins, kg, fragment_stats in results:
                            pbar.update(1)

                            if "error" in result:
                                global_stats["failed_files"] += 1
                                with open(processing_log_path, 'a') as p_log:
                                    p_log.write(
                                        f"{time.strftime('%Y-%m-%d %H:%M:%S')},{result['file_path']},FAILED,0,0,0,{result['error']}\n")
                                continue

                            # 更新统计信息
                            global_stats["processed_files"] += 1
                            global_stats["extracted_fragments"] += len(proteins)
                            global_stats["knowledge_graphs"] += len(kg)

                            if fragment_stats:
                                global_stats["total_edges"] += fragment_stats.get('edges', 0)

                                # 记录片段统计
                                with open(fragments_log_path, 'a') as f_log:
                                    f_log.write(f"{fragment_stats['file_name']},{fragment_stats['protein_id']},"
                                                f"{fragment_stats.get('valid_residues', 0)},"
                                                f"{fragment_stats.get('fragments', 0)},"
                                                f"{fragment_stats.get('edges', 0)}\n")

                            # 更新序列日志
                            with open(sequences_log_path, 'a', buffering=1) as s_log:
                                for fragment_id, data in proteins.items():
                                    s_log.write(f"{fragment_id},{data['protein_id']},"
                                                f"{len(data['sequence'])},{data['sequence']}\n")

                            # 累积数据
                            batch_proteins.update(proteins)
                            batch_graphs.update(kg)

                            # 记录处理日志
                            with open(processing_log_path, 'a') as p_log:
                                elapsed = result.get('elapsed', 0)
                                p_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},"
                                            f"{result['file_path']},SUCCESS,{elapsed:.2f},"
                                            f"{len(proteins)},{len(kg)},\n")
                    except Exception as e:
                        logger.error(f"处理批次时出错: {str(e)}")
                        logger.error(traceback.format_exc())

        # 保存批次结果
        logger.info(
            f"批次 {batch_id + 1} 处理完成，保存 {len(batch_proteins)} 个蛋白质片段和 {len(batch_graphs)} 个图谱")

        # 优化保存过程 - 使用多线程并行保存
        save_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as save_executor:
            # 并行保存蛋白质数据
            protein_future = save_executor.submit(
                save_results_chunked, batch_proteins, batch_output_dir,
                base_name="protein_data", chunk_size=10000
            )

            # 并行保存图谱数据
            if batch_graphs:
                graph_future = save_executor.submit(
                    save_knowledge_graphs, batch_graphs, batch_output_dir,
                    base_name="protein_kg", chunk_size=10000, format_type=format_type
                )

        # 等待保存完成
        protein_future.result()
        if batch_graphs:
            graph_future.result()

        logger.info(f"批次 {batch_id + 1} 数据保存完成，耗时 {time.time() - save_start:.2f} 秒")

        # 清理此批次数据并执行垃圾回收
        batch_proteins.clear()
        batch_graphs.clear()
        check_memory_usage(force_gc=True)

        # 当前进度报告
        logger.info(f"当前总体进度: {global_stats['processed_files']}/{total_files} 文件 "
                    f"({global_stats['processed_files'] / total_files * 100:.1f}%), "
                    f"提取片段数: {global_stats['extracted_fragments']}, "
                    f"知识图谱数: {global_stats['knowledge_graphs']}, "
                    f"边总数: {global_stats['total_edges']}")

    # 处理完成，返回统计信息
    return global_stats, base_data_dir


def save_results_chunked(all_proteins, output_dir, base_name="protein_data", chunk_size=100000):
    """分块保存蛋白质序列结果（修复NumPy类型的JSON序列化问题）"""
    os.makedirs(output_dir, exist_ok=True)

    # 将蛋白质数据分块
    protein_ids = list(all_proteins.keys())
    chunks = [protein_ids[i:i + chunk_size] for i in range(0, len(protein_ids), chunk_size)]

    output_files = []
    for i, chunk_ids in enumerate(chunks):
        # 创建一个新的字典，确保所有值都是可JSON序列化的
        chunk_data = {}
        for pid in chunk_ids:
            protein_data = all_proteins[pid]
            # 将所有可能的NumPy类型转换为Python原生类型
            clean_data = {}
            for key, value in protein_data.items():
                if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    clean_data[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                    clean_data[key] = float(value)
                elif isinstance(value, np.ndarray):
                    clean_data[key] = value.tolist()
                else:
                    clean_data[key] = value
            chunk_data[pid] = clean_data

        output_file = os.path.join(output_dir, f"{base_name}_chunk_{i + 1}.json")

        with open(output_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)

        output_files.append(output_file)
        logger.info(f"保存数据块 {i + 1}/{len(chunks)}: {output_file} ({len(chunk_ids)} 个蛋白质)")

    # 保存元数据（确保所有类型都是Python原生类型）
    metadata = {
        "total_proteins": int(len(all_proteins)),
        "chunk_count": int(len(chunks)),
        "chunk_files": [str(file) for file in output_files],
        "created_at": str(time.strftime("%Y-%m-%d %H:%M:%S"))
    }

    metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_files, metadata


def save_knowledge_graphs(kg_data, output_dir, base_name="protein_kg", chunk_size=100000, format_type="pyg"):
    """保存知识图谱数据，支持JSON和PyG格式"""
    if not kg_data:
        logger.warning(f"没有知识图谱数据可保存为{format_type}格式")
        return None

    # 创建输出目录
    kg_dir = os.path.join(output_dir, f"knowledge_graphs_{format_type}")
    os.makedirs(kg_dir, exist_ok=True)

    # 获取所有蛋白质ID并分块
    all_protein_ids = list(kg_data.keys())
    num_chunks = (len(all_protein_ids) + chunk_size - 1) // chunk_size

    logger.info(f"将{len(all_protein_ids)}个知识图谱拆分为{num_chunks}个块，每块最多{chunk_size}个蛋白质")

    # 创建索引字典
    index = {
        "total_proteins": len(all_protein_ids),
        "chunks_count": num_chunks,
        "chunk_size": chunk_size,
        "total_nodes": 0,
        "total_edges": 0,
        "error_count": 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": format_type
    }

    # 处理每个块
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_protein_ids))
        chunk_ids = all_protein_ids[start_idx:end_idx]

        logger.info(f"开始处理知识图谱块 {chunk_id + 1}/{num_chunks}，共{len(chunk_ids)}个蛋白质")

        # 为当前块创建输出文件
        output_file = os.path.join(kg_dir,
                                   f"{base_name}_chunk_{chunk_id + 1}.{'pt' if format_type == 'pyg' else 'json'}")

        if format_type == "pyg":
            # PyG格式保存
            try:
                graphs_data = {}

                for i, pid in enumerate(chunk_ids):
                    if i % 1000 == 0 or i == len(chunk_ids) - 1:
                        logger.info(f"  - 正在处理第{i + 1}/{len(chunk_ids)}个蛋白质图谱")

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

                            # 6. 侧链柔性 (1维)
                            side_chain_flexibility = float(props['flexibility'])

                            # 7. pLDDT值 (1维)
                            plddt = float(node_attrs.get('plddt', 70.0)) / 100.0  # 归一化到0-1

                            # 合并所有特征
                            features = blosum + position + [hydropathy, charge, molecular_weight,
                                                            volume, flexibility, is_aromatic,
                                                            ss_alpha, ss_beta, ss_coil, sasa,
                                                            side_chain_flexibility, plddt]

                            node_features.append(features)

                        # 提取边特征
                        for src, tgt, edge_data in nx_graph.edges(data=True):
                            edge_index[0].append(node_mapping[src])
                            edge_index[1].append(node_mapping[tgt])

                            # 1. 边类型编码 (4维 one-hot)
                            edge_type = edge_data.get('edge_type', 0)
                            edge_type_onehot = [0, 0, 0, 0]
                            if edge_type < len(edge_type_onehot):
                                edge_type_onehot[edge_type] = 1

                            # 2. 空间距离 (1维)
                            distance = float(edge_data.get('distance', 0))

                            # 3. 相互作用强度 (1维)
                            interaction_strength = float(edge_data.get('interaction_strength', 0.5))

                            # 4. 方向性 (2维)
                            direction = edge_data.get('direction', [0, 0])
                            if len(direction) != 2:
                                direction = [0, 0]

                            # 合并边特征
                            edge_features = edge_type_onehot + [distance, interaction_strength] + direction
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
                            index["total_nodes"] += len(node_features)
                            index["total_edges"] += len(edge_attr)
                    except Exception as e:
                        logger.warning(f"转换蛋白质 {pid} 图谱时出错: {str(e)}")
                        index["error_count"] += 1

                # 保存为PyTorch文件
                if graphs_data:
                    torch.save(graphs_data, output_file)
                    logger.info(f"已保存 {len(graphs_data)} 个PyG格式图谱到 {output_file}")
                else:
                    logger.warning(f"块 {chunk_id + 1} 中没有有效的PyG图谱数据")

            except Exception as e:
                logger.error(f"保存PyG格式知识图谱时出错 (块 {chunk_id + 1}): {str(e)}")
                logger.error(traceback.format_exc())
        else:
            # JSON格式保存
            try:
                # 准备当前块的数据
                chunk_data = {pid: kg_data[pid] for pid in chunk_ids}

                # 保存JSON文件
                with open(output_file, 'w') as f:
                    json.dump(chunk_data, f)

                # 更新统计信息
                node_count = 0
                edge_count = 0
                for pid in chunk_ids:
                    if isinstance(kg_data[pid], dict) and 'nodes' in kg_data[pid]:
                        node_count += len(kg_data[pid].get('nodes', []))
                        edge_count += len(kg_data[pid].get('links', []))

                index["total_nodes"] += node_count
                index["total_edges"] += edge_count

                logger.info(f"已保存 {len(chunk_data)} 个JSON格式图谱到 {output_file}")
            except Exception as e:
                logger.error(f"保存JSON格式知识图谱时出错 (块 {chunk_id + 1}): {str(e)}")
                logger.error(traceback.format_exc())

    # 保存索引文件
    index_file = os.path.join(kg_dir, f"{base_name}_index.json")
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)

    return index


# ======================= 主程序和命令行接口 =======================

def main():
    """优化的主函数 - 支持大规模处理"""
    parser = argparse.ArgumentParser(description="大规模蛋白质结构数据处理与知识图谱构建系统")
    parser.add_argument("input", help="输入PDB/CIF文件或包含这些文件的目录")
    parser.add_argument("--output_dir", "-o", default="./kg",
                        help="输出目录 (默认: ./kg)")
    parser.add_argument("--min_length", "-m", type=int, default=5,
                        help="最小序列长度 (默认: 5)")
    parser.add_argument("--max_length", "-M", type=int, default=50,
                        help="最大序列长度 (默认: 50)")
    parser.add_argument("--batch_size", "-b", type=int, default=50000,
                        help="大规模批处理大小 (默认: 50000，适合TB级内存)")
    parser.add_argument("--memory_limit", type=int, default=800,
                        help="内存使用上限GB (默认: 800)")
    parser.add_argument("--workers", "-w", type=int, default=100,
                        help="并行工作进程数 (默认: 100，适合112核CPU)")
    parser.add_argument("--k_neighbors", type=int, default=8,
                        help="空间邻接的K近邻数 (默认: 8)")
    parser.add_argument("--distance_threshold", type=float, default=8.0,
                        help="空间邻接距离阈值 (默认: 8.0埃)")
    parser.add_argument("--plddt_threshold", type=float, default=70.0,
                        help="AlphaFold pLDDT质量得分阈值 (默认: 70.0)")
    parser.add_argument("--respect_ss", action="store_true", default=True,
                        help="是否尊重二级结构边界进行片段划分 (默认: True)")
    parser.add_argument("--respect_domains", action="store_true", default=True,
                        help="是否尊重结构域边界进行片段划分 (默认: True)")
    parser.add_argument("--format", choices=["pyg", "json"], default="pyg",
                        help="知识图谱保存格式 (默认: pyg)")
    parser.add_argument("--limit", type=int,
                        help="限制处理的文件数量 (用于测试)")

    args = parser.parse_args()

    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    global logger
    logger, log_file_path = setup_logging(args.output_dir)
    logger.info(f"日志将写入文件: {log_file_path}")
    logger.info(f"大规模处理模式: 使用最大 {args.memory_limit}GB 内存和 {args.workers} 个CPU核心")

    # 查找输入文件
    if os.path.isdir(args.input):
        start_time = time.time()
        logger.info(f"开始扫描目录: {args.input}")
        pdb_files = find_pdb_files(args.input)
        logger.info(f"扫描完成，耗时: {time.time() - start_time:.1f}秒，找到 {len(pdb_files)} 个PDB/CIF文件")
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

    # 使用大规模处理函数
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
        format_type=args.format
    )

    total_time = time.time() - start_proc_time
    avg_time_per_file = total_time / (stats['processed_files'] + 0.001)

    # 处理结果统计
    logger.info("\n处理完成:")
    logger.info(f"- 总耗时: {total_time / 3600:.2f}小时 ({total_time:.1f}秒)")
    logger.info(f"- 平均每文件: {avg_time_per_file:.3f}秒")
    logger.info(f"- 处理速度: {stats['processed_files'] / total_time:.1f}文件/秒")
    logger.info(f"- 处理的文件总数: {stats['processed_files']}")
    logger.info(f"- 提取的蛋白质片段总数: {stats['extracted_fragments']}")
    logger.info(f"- 生成的知识图谱总数: {stats['knowledge_graphs']}")
    logger.info(f"- 知识图谱边总数: {stats.get('total_edges', 0)}")
    logger.info(f"- 失败的文件数: {stats.get('failed_files', 0)}")
    logger.info(f"- 结果保存在: {all_data_dir}")
    logger.info(f"- 日志文件: {log_file_path}")

    # 添加总结信息
    summary_file = os.path.join(args.output_dir, "extraction_summary.json")
    with open(summary_file, 'w') as f:
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time_seconds": total_time,
            "execution_time_hours": total_time / 3600,
            "files_per_second": stats['processed_files'] / total_time,
            "processed_files": stats['processed_files'],
            "failed_files": stats.get('failed_files', 0),
            "extracted_fragments": stats['extracted_fragments'],
            "knowledge_graphs": stats['knowledge_graphs'],
            "total_edges": stats.get('total_edges', 0),
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
            "all_data_dir": os.path.abspath(all_data_dir)
        }
        json.dump(summary, f, indent=2)

    logger.info(f"摘要信息已保存到: {summary_file}")
    logger.info("大规模蛋白质结构提取与知识图谱构建流程完成！")

if __name__ == "__main__":
    main()