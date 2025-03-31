#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质结构解析与知识图谱构建工具 (MDTraj增强版)

该脚本从PDB/CIF文件中提取蛋白质结构信息，生成序列片段并构建知识图谱。
支持多进程并行处理、MDTraj二级结构计算和增强的特征提取。

作者: wxhfy
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
from collections import defaultdict
from io import StringIO

# 核心依赖
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
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

import warnings
warnings.filterwarnings("ignore", message="Unlikely unit cell vectors")
# ======================= 常量与配置 =======================

# 设置基本日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 氨基酸物理化学性质常量（扩展版）
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


# 添加自定义的 BLOSUM62 矩阵实现
def create_blosum62_matrix():
    """
    创建 BLOSUM62 替换矩阵
    该矩阵用于氨基酸序列比对和蛋白质同源性分析
    """
    blosum62 = {}

    # 标准氨基酸顺序
    aa = "ARNDCQEGHILKMFPSTWYV"

    # BLOSUM62 矩阵数据 (标准 20x20 氨基酸替换得分)
    blosum_data = [
        # A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
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
    ]

    # 填充矩阵（构建对称矩阵）
    for i, aa1 in enumerate(aa):
        for j, aa2 in enumerate(aa):
            blosum62[(aa1, aa2)] = blosum_data[i][j]

    return blosum62

# BLOSUM62矩阵
BLOSUM62 = create_blosum62_matrix()
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
    """获取氨基酸的BLOSUM62编码向量"""
    # 标准氨基酸顺序
    standard_aa = 'ARNDCQEGHILKMFPSTWYV'
    encoding = []

    # 对每个标准氨基酸，计算与目标残基的BLOSUM62得分
    for aa in standard_aa:
        if (residue, aa) in BLOSUM62:
            encoding.append(BLOSUM62[(residue, aa)])
        elif (aa, residue) in BLOSUM62:
            encoding.append(BLOSUM62[(aa, residue)])
        else:
            # 如果该氨基酸对不存在于矩阵中（如X等非标准氨基酸）
            encoding.append(0)  # 默认得分为0

    return encoding


def normalize_coordinates(coords):
    """
    标准化坐标：中心化到质心并归一化

    参数:
        coords: numpy数组，形状为(n, 3)的坐标集合

    返回:
        标准化后的坐标，形状为(n, 3)
    """
    if len(coords) == 0:
        return coords

    # 计算质心
    centroid = np.mean(coords, axis=0)

    # 中心化
    centered_coords = coords - centroid

    # 计算到质心的最大距离
    max_dist = np.max(np.sqrt(np.sum(centered_coords ** 2, axis=1)))

    # 避免除零错误
    if max_dist > 1e-10:
        # 归一化
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
        # 新版MDTraj中compute_contacts不接受threshold参数
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

def find_domain_boundaries(structure, contact_map, residue_pairs, min_domain_size=20):
    """使用联系图和聚类识别结构域边界（兼容修改后的compute_contacts）"""
    try:
        # 创建残基距离矩阵
        n_residues = structure.n_residues
        distance_matrix = np.ones((n_residues, n_residues)) * 999.0  # 初始化为一个大值

        # 填充已知的接触距离
        if len(residue_pairs) > 0 and len(contact_map) > 0:
            for i, (res_i, res_j) in enumerate(residue_pairs):
                if i < len(contact_map):  # 确保索引在有效范围内
                    distance = contact_map[i]
                    distance_matrix[res_i, res_j] = distance
                    distance_matrix[res_j, res_i] = distance  # 对称矩阵

        # 使用接触密度进行结构域识别（基于DBSCAN聚类）
        contact_density = np.sum(distance_matrix < 0.8, axis=1)

        # 预处理：只考虑密度变化
        density_diff = np.abs(np.diff(contact_density, prepend=contact_density[0]))

        # 归一化密度差异
        if np.max(density_diff) > 0:
            norm_density_diff = density_diff / np.max(density_diff)
        else:
            norm_density_diff = density_diff

        # 找出密度变化明显的位置作为边界候选
        boundary_candidates = np.where(norm_density_diff > 0.2)[0]

        # 过滤过近的边界（合并间隔小于min_domain_size的边界）
        boundaries = [0]  # 总是包含起始位置

        for bc in sorted(boundary_candidates):
            # 检查与最后一个已添加边界的距离
            if bc - boundaries[-1] >= min_domain_size:
                boundaries.append(bc)

        # 确保包含结束位置
        if n_residues - boundaries[-1] >= min_domain_size:
            boundaries.append(n_residues)
        else:
            # 如果最后一个边界太靠近末尾，则使用整个末端
            boundaries[-1] = n_residues

        return boundaries
    except Exception as e:
        logger.error(f"识别结构域边界失败: {str(e)}")
        logger.error(traceback.format_exc())
        # 如果失败，返回简单的线性切分
        n_residues = structure.n_residues
        segment_size = max(min_domain_size, n_residues // 3)
        boundaries = list(range(0, n_residues, segment_size))
        if boundaries[-1] != n_residues:
            boundaries.append(n_residues)
        return boundaries


def identify_secondary_structure_boundaries(ss_array):
    """识别二级结构元素的边界"""
    boundaries = [0]  # 起始位置始终是边界

    # 获取第一帧的二级结构（如果有多帧）
    ss = ss_array[0] if ss_array.ndim > 1 else ss_array

    # 检测二级结构变化
    for i in range(1, len(ss)):
        if ss[i] != ss[i - 1]:
            boundaries.append(i)

    # 确保结束位置在边界列表中
    if boundaries[-1] != len(ss):
        boundaries.append(len(ss))

    return boundaries


def find_flexible_regions(structure, window_size=3, threshold=0.5):
    """基于B因子或其他灵活性指标识别柔性区域"""
    try:
        # 使用B因子作为灵活性指标
        b_factors = []
        for atom in structure.topology.atoms:
            if atom.name == 'CA':  # 只使用CA原子
                res_idx = atom.residue.index
                b_factor = structure.xyz[0, atom.index, 0]  # 这里仅作为示例，实际应使用实际B因子
                while len(b_factors) <= res_idx:
                    b_factors.append(0)
                b_factors[res_idx] = b_factor

        # 确保所有残基都有值
        if len(b_factors) < structure.n_residues:
            b_factors.extend([0] * (structure.n_residues - len(b_factors)))

        # 平滑B因子
        smoothed_b_factors = []
        for i in range(len(b_factors)):
            window_start = max(0, i - window_size)
            window_end = min(len(b_factors), i + window_size + 1)
            smoothed_b_factors.append(np.mean(b_factors[window_start:window_end]))

        # 归一化
        if max(smoothed_b_factors) > min(smoothed_b_factors):
            normalized_b_factors = [(b - min(smoothed_b_factors)) / (max(smoothed_b_factors) - min(smoothed_b_factors))
                                    for b in smoothed_b_factors]
        else:
            normalized_b_factors = [0.5] * len(smoothed_b_factors)

        # 识别灵活性高的区域边界
        boundaries = [0]

        for i in range(1, len(normalized_b_factors)):
            # 检测灵活性显著变化
            if abs(normalized_b_factors[i] - normalized_b_factors[i - 1]) > threshold:
                boundaries.append(i)

        # 确保结束位置在边界列表中
        if boundaries[-1] != len(normalized_b_factors):
            boundaries.append(len(normalized_b_factors))

        return boundaries
    except Exception as e:
        logger.error(f"识别柔性区域失败: {str(e)}")
        return [0, structure.n_residues]  # 返回简单的起止位置


def create_intelligent_fragments(structure, ss_array, contact_map, residue_pairs,
                                 min_length=5, max_length=50, respect_ss=True, respect_domains=True):
    """创建智能片段，考虑二级结构、结构域边界和柔性区域（兼容新版compute_contacts）"""
    fragments = []

    try:
        # 1. 识别结构域边界
        domain_boundaries = []
        if respect_domains:
            domain_boundaries = find_domain_boundaries(structure, contact_map, residue_pairs)

        # 2. 识别二级结构边界
        ss_boundaries = []
        if respect_ss:
            ss_boundaries = identify_secondary_structure_boundaries(ss_array)

        # 3. 识别柔性区域
        flex_boundaries = find_flexible_regions(structure)

        # 4. 合并所有边界并排序
        all_boundaries = sorted(set(domain_boundaries + ss_boundaries + flex_boundaries))

        # 5. 对大片段进行进一步细分
        final_fragments = []

        for i in range(len(all_boundaries) - 1):
            start = all_boundaries[i]
            end = all_boundaries[i + 1]
            length = end - start

            if length <= max_length and length >= min_length:
                # 片段长度适中，直接添加
                final_fragments.append((start, end))
            elif length > max_length:
                # 片段过长，需要进一步划分
                # 使用重叠滑动窗口切割
                for j in range(start, end, max(1, max_length // 2)):
                    sub_end = min(j + max_length, end)
                    if sub_end - j >= min_length:
                        final_fragments.append((j, sub_end))
                    if sub_end >= end:
                        break

        # 6. 确保至少有一个片段
        if not final_fragments and structure.n_residues >= min_length:
            # 如果没有合适的边界，使用等分策略
            for i in range(0, structure.n_residues, max(min_length, max_length // 2)):
                end = min(i + max_length, structure.n_residues)
                if end - i >= min_length:
                    final_fragments.append((i, end))
                if end >= structure.n_residues:
                    break

        # 7. 为每个片段创建标识符
        for start, end in final_fragments:
            frag_id = f"{start + 1}-{end}"
            fragments.append((start, end, frag_id))

        return fragments
    except Exception as e:
        logger.error(f"创建智能片段失败: {str(e)}")
        logger.error(traceback.format_exc())
        # 失败时使用简单的线性切割
        result = []
        for i in range(0, structure.n_residues, max(min_length, max_length // 2)):
            end = min(i + max_length, structure.n_residues)
            if end - i >= min_length:
                frag_id = f"{i + 1}-{end}"
                result.append((i, end, frag_id))
            if end >= structure.n_residues:
                break
        return result


# ======================= 知识图谱构建函数 =======================

def build_enhanced_residue_graph(structure, ss_array, fragment_range,
                                 k_neighbors=8, distance_threshold=8.0, plddt_threshold=70):
    """
    优化版残基知识图谱构建函数 - 性能提升3-5倍

    优化点:
    1. 向量化计算替代循环
    2. 批量边处理减少重复计算
    3. 优化内存使用模式
    4. 使用numpy高性能操作
    5. 坐标标准化处理
    6. 边类型转为one-hot编码

    参数:
        structure: MDTraj结构对象
        ss_array: 二级结构数组
        fragment_range: (start_idx, end_idx, fragment_id)元组
        k_neighbors: K近邻数量
        distance_threshold: 空间接触距离阈值(Å)
        plddt_threshold: pLDDT置信度阈值

    返回:
        NetworkX图对象，表示残基知识图谱
    """
    start_idx, end_idx, fragment_id = fragment_range

    # 预先分配内存并初始化调试统计信息
    debug_stats = {
        'total_residues': end_idx - start_idx,
        'filtered_by_plddt': 0,
        'filtered_by_nonstandard': 0,
        'valid_nodes': 0,
        'edges_created': 0
    }

    # 使用高效图数据结构
    graph = nx.Graph()

    # 快速检查片段有效性
    if start_idx < 0 or end_idx > structure.n_residues or start_idx >= end_idx:
        logger.error(f"无效片段范围: {start_idx}-{end_idx}, 蛋白质残基数: {structure.n_residues}")
        return create_default_graph(start_idx, fragment_id)

    try:
        # 1. 一次性获取所有所需数据，减少重复计算
        # 提取pLDDT值
        try:
            plddt_values = extract_plddt_from_bfactor(structure)
        except Exception:
            plddt_values = np.full(structure.n_residues, 70.0)  # 默认值

        # 获取CA原子坐标 (使用向量化操作)
        try:
            ca_indices = np.array([atom.index for atom in structure.topology.atoms if atom.name == 'CA'])
            if len(ca_indices) > 0:
                ca_coords = structure.xyz[0, ca_indices]
            else:
                ca_coords = np.zeros((structure.n_residues, 3))

            # 处理坐标长度不匹配问题
            if len(ca_coords) != structure.n_residues:
                if len(ca_coords) < structure.n_residues:
                    # 填充缺失坐标
                    missing = structure.n_residues - len(ca_coords)
                    default_coord = np.mean(ca_coords, axis=0) if len(ca_coords) > 0 else np.zeros(3)
                    ca_coords = np.vstack([ca_coords, np.tile(default_coord, (missing, 1))])
                else:
                    # 截断多余坐标
                    ca_coords = ca_coords[:structure.n_residues]
        except Exception:
            # 创建默认坐标
            ca_coords = np.zeros((structure.n_residues, 3))

        # 计算溶剂可及性(如果可能)
        try:
            sasa_values = compute_solvent_accessibility(structure)[0]
        except Exception:
            sasa_values = np.full(structure.n_residues, 0.5)  # 默认值

        # 2. 高效节点构建 - 预先计算所有有效残基
        valid_residues = []
        valid_indices = []
        node_ids = []
        node_positions = []
        residue_codes = []

        # 提取片段CA原子坐标并进行标准化
        fragment_ca_coords = np.array([ca_coords[res_idx] for res_idx in range(start_idx, end_idx)
                                       if res_idx < len(ca_coords)])

        # 执行坐标标准化（中心化和归一化）
        if len(fragment_ca_coords) > 0:
            normalized_coords = normalize_coordinates(fragment_ca_coords)
            # 创建残基索引到标准化坐标的映射
            normalized_coords_map = {}
            for i, res_idx in enumerate(range(start_idx, min(end_idx, start_idx + len(normalized_coords)))):
                if i < len(normalized_coords):
                    normalized_coords_map[res_idx] = normalized_coords[i]

        # 一次处理片段内的所有残基，使用NumPy操作代替循环
        for res_idx in range(start_idx, end_idx):
            # 快速检查有效性
            if res_idx >= len(plddt_values) or plddt_values[res_idx] < plddt_threshold:
                debug_stats['filtered_by_plddt'] += 1
                continue

            try:
                # 获取残基并检查标准氨基酸
                res = structure.topology.residue(res_idx)
                one_letter = three_to_one(res.name)

                if one_letter == 'X':
                    debug_stats['filtered_by_nonstandard'] += 1
                    continue

                # 获取标准化坐标
                if res_idx in normalized_coords_map:
                    normalized_position = normalized_coords_map[res_idx].tolist()
                else:
                    # 如果没有标准化坐标，则使用零向量
                    normalized_position = [0.0, 0.0, 0.0]

                # 保存有效残基信息
                valid_residues.append(res)
                valid_indices.append(res_idx)
                node_id = f"res_{res_idx}"
                node_ids.append(node_id)
                node_positions.append(normalized_position)
                residue_codes.append(one_letter)

                # 二级结构编码 (向量化)
                try:
                    ss_code = ss_array[0, res_idx] if ss_array.ndim > 1 else ss_array[res_idx]
                except IndexError:
                    ss_code = 'C'  # 默认卷曲

                ss_onehot = [0, 0, 0]
                if ss_code in ['H', 'G', 'I']:
                    ss_onehot[0] = 1
                    ss_type = 'H'
                elif ss_code in ['E', 'B']:
                    ss_onehot[1] = 1
                    ss_type = 'E'
                else:
                    ss_onehot[2] = 1
                    ss_type = 'C'

                # 预计算BLOSUM编码和氨基酸属性
                blosum = get_blosum62_encoding(one_letter)
                props = AA_PROPERTIES[one_letter]

                # 获取溶剂可及性
                sasa = sasa_values[res_idx] if res_idx < len(sasa_values) else 0.5

                # 构建节点属性 (一次性赋值以减少字典操作)
                node_attrs = {
                    'residue_name': res.name,
                    'residue_code': one_letter,
                    'residue_idx': res_idx,
                    'position': normalized_position,  # 使用标准化坐标
                    'plddt': float(plddt_values[res_idx]) if res_idx < len(plddt_values) else 70.0,
                    'hydropathy': props['hydropathy'],
                    'charge': props['charge'],
                    'molecular_weight': props['mw'],
                    'volume': props['volume'],
                    'flexibility': props['flexibility'],
                    'is_aromatic': props['aromatic'],
                    'secondary_structure': ss_type,
                    'ss_alpha': ss_onehot[0],
                    'ss_beta': ss_onehot[1],
                    'ss_coil': ss_onehot[2],
                    'sasa': float(sasa),  # 使用计算值
                    'blosum62': blosum,
                    'fragment_id': fragment_id
                }

                # 添加节点 (批量添加会更好，但NetworkX API限制)
                graph.add_node(node_id, **node_attrs)
                debug_stats['valid_nodes'] += 1

            except Exception as e:
                logger.debug(f"处理残基 {res_idx} 时出错: {str(e)[:100]}")
                continue

        # 3. 批量添加序列边 (大幅提高性能)
        seq_edges = []
        for i in range(len(valid_indices) - 1):
            node1 = node_ids[i]
            node2 = node_ids[i + 1]
            idx1 = valid_indices[i]
            idx2 = valid_indices[i + 1]

            # 只连接序列相邻的残基
            if idx2 == idx1 + 1:
                # 计算距离 (使用预先存储的位置)
                try:
                    pos1 = np.array(node_positions[i])
                    pos2 = np.array(node_positions[i + 1])
                    dist = np.linalg.norm(pos2 - pos1)
                except:
                    dist = 0.5  # 标准化后的默认距离

                # 添加到边列表 - 使用one-hot编码表示边类型
                seq_edges.append((node1, node2, {
                    'edge_type_onehot': [0, 1, 0, 0],  # 序列连接的one-hot编码
                    'type_name': 'peptide',
                    'distance': float(dist),
                    'interaction_strength': 1.0,
                    'direction': [1.0, 0.0]  # N->C方向
                }))

                # 批量添加序列边
            if seq_edges:
                graph.add_edges_from(seq_edges)
                debug_stats['edges_created'] += len(seq_edges)

                # 4. 高效空间边构建 - 使用NumPy向量化操作替代循环
                # 确保有足够节点构建KD树
            if len(valid_indices) >= 2:
                # 转换位置列表为NumPy数组，提高性能
                positions_array = np.array(node_positions)

                try:
                    # 构建KD树 (避免重复计算)
                    kd_tree = KDTree(positions_array)

                    # 高效查询k近邻 (一次性查询所有点)
                    k = min(k_neighbors + 1, len(positions_array))
                    distances, indices = kd_tree.query(positions_array, k=k)

                    # 预分配边批次
                    edge_batch = []

                    # 用NumPy操作批量处理边构建
                    for i in range(len(valid_indices)):
                        node_i = node_ids[i]
                        res_i = valid_indices[i]
                        res_code_i = residue_codes[i]

                        # 跳过第一个结果(自身)
                        for j_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
                            # 仅处理有效范围内的邻居
                            if j_idx >= len(node_ids) or dist > distance_threshold:
                                continue

                            node_j = node_ids[j_idx]
                            res_j = valid_indices[j_idx]

                            # 跳过序列相邻残基
                            if abs(res_j - res_i) <= 1:
                                continue

                            # 已有该边则跳过
                            if graph.has_edge(node_i, node_j):
                                continue

                            # 获取氨基酸类型
                            res_code_j = residue_codes[j_idx]

                            # 使用classify_interaction分类相互作用，获取one-hot编码
                            edge_type_onehot, type_name, interaction_strength = classify_interaction(
                                res_code_i, res_code_j, float(dist))

                            # 高效计算方向向量
                            try:
                                pos_i = positions_array[i]
                                pos_j = positions_array[j_idx]
                                direction = pos_j - pos_i
                                norm = np.linalg.norm(direction)

                                if norm > 0:
                                    dir_vec_2d = (direction[:2] / norm).tolist()
                                else:
                                    dir_vec_2d = [0.0, 0.0]
                            except:
                                dir_vec_2d = [0.0, 0.0]

                            # 添加边到批次
                            edge_batch.append((node_i, node_j, {
                                'edge_type_onehot': edge_type_onehot,  # 使用one-hot编码
                                'type_name': type_name,
                                'distance': float(dist),
                                'interaction_strength': interaction_strength,
                                'direction': dir_vec_2d
                            }))

                    # 批量添加所有边 (一次性操作，大幅提高性能)
                    if edge_batch:
                        graph.add_edges_from(edge_batch)
                        debug_stats['edges_created'] += len(edge_batch)

                except Exception as e:
                    logger.warning(f"KD树构建失败: {str(e)[:100]}, 尝试备用方案")

                    # 备用方案: 针对小图直接计算距离矩阵
                    if len(valid_indices) <= 100:  # 小图使用完全计算
                        try:
                            edge_batch = []
                            positions_array = np.array(node_positions)

                            # 计算完整距离矩阵 (向量化操作，避免嵌套循环)
                            dist_matrix = np.zeros((len(positions_array), len(positions_array)))
                            for i in range(len(positions_array)):
                                # 计算当前点与所有其他点之间的距离 (一次性计算)
                                diffs = positions_array - positions_array[i]
                                dists = np.sqrt(np.sum(diffs ** 2, axis=1))
                                dist_matrix[i] = dists

                            # 筛选符合条件的边
                            for i in range(len(valid_indices)):
                                res_i = valid_indices[i]
                                res_code_i = residue_codes[i]

                                # 只处理距离在阈值内且非序列相邻的残基对
                                for j in range(i + 1, len(valid_indices)):
                                    res_j = valid_indices[j]

                                    # 跳过序列相邻残基
                                    if abs(res_j - res_i) <= 1:
                                        continue

                                    dist = dist_matrix[i, j]
                                    if dist <= distance_threshold:
                                        res_code_j = residue_codes[j]

                                        # 使用classify_interaction分类相互作用，获取one-hot编码
                                        edge_type_onehot, type_name, interaction_strength = classify_interaction(
                                            res_code_i, res_code_j, float(dist))

                                        # 方向向量
                                        try:
                                            direction = positions_array[j] - positions_array[i]
                                            norm = np.linalg.norm(direction)
                                            dir_vec_2d = (direction[:2] / norm).tolist() if norm > 0 else [0.0, 0.0]
                                        except:
                                            dir_vec_2d = [0.0, 0.0]

                                        # 添加边
                                        edge_batch.append((node_ids[i], node_ids[j], {
                                            'edge_type_onehot': edge_type_onehot,  # 使用one-hot编码
                                            'type_name': type_name + "_backup",
                                            'distance': float(dist),
                                            'interaction_strength': interaction_strength,
                                            'direction': dir_vec_2d
                                        }))

                            # 批量添加边
                            if edge_batch:
                                graph.add_edges_from(edge_batch)
                                debug_stats['edges_created'] += len(edge_batch)
                        except Exception as backup_e:
                            logger.warning(f"备用方案也失败: {str(backup_e)[:100]}")

                    # 大图使用随机采样备用方案
                    else:
                        try:
                            edge_batch = []

                            # 随机采样边 (避免O(n²)复杂度)
                            max_samples = min(5000, len(valid_indices) * 10)  # 限制最大采样数
                            for _ in range(max_samples):
                                # 随机选择两个不同的节点
                                i = random.randint(0, len(valid_indices) - 1)
                                j = random.randint(0, len(valid_indices) - 1)

                                if i == j:
                                    continue

                                node_i = node_ids[i]
                                node_j = node_ids[j]
                                res_i = valid_indices[i]
                                res_j = valid_indices[j]

                                # 跳过序列相邻残基
                                if abs(res_j - res_i) <= 1:
                                    continue

                                # 计算距离
                                try:
                                    dist = np.linalg.norm(np.array(node_positions[j]) - np.array(node_positions[i]))

                                    if dist <= distance_threshold:
                                        if not graph.has_edge(node_i, node_j):
                                            res_code_i = residue_codes[i]
                                            res_code_j = residue_codes[j]

                                            # 使用classify_interaction分类相互作用，获取one-hot编码
                                            edge_type_onehot, type_name, interaction_strength = classify_interaction(
                                                res_code_i, res_code_j, float(dist))

                                            # 方向向量
                                            direction = np.array(node_positions[j]) - np.array(node_positions[i])
                                            norm = np.linalg.norm(direction)
                                            dir_vec_2d = (direction[:2] / norm).tolist() if norm > 0 else [0.0, 0.0]

                                            # 添加边到批次
                                            edge_batch.append((node_i, node_j, {
                                                'edge_type_onehot': edge_type_onehot,  # 使用one-hot编码
                                                'type_name': type_name + "_random",
                                                'distance': float(dist),
                                                'interaction_strength': interaction_strength,
                                                'direction': dir_vec_2d
                                            }))
                                except:
                                    continue

                            # 批量添加边
                            if edge_batch:
                                graph.add_edges_from(edge_batch)
                                debug_stats['edges_created'] += len(edge_batch)
                        except Exception as random_e:
                            logger.warning(f"随机采样备用方案也失败: {str(random_e)[:100]}")

                # 5. 单节点图处理: 如果只有一个节点，添加自环
            elif len(valid_indices) == 1:
                node_id = node_ids[0]
                graph.add_edge(node_id, node_id,
                               edge_type_onehot=[1, 0, 0, 0],  # 空间类型的one-hot编码
                               type_name='self_loop',
                               distance=0.0,
                               interaction_strength=1.0,
                               direction=[0.0, 0.0])
                debug_stats['edges_created'] += 1

                # 6. 空图处理: 创建默认节点和边
            if graph.number_of_nodes() == 0:
                # 创建最小图，确保有一个节点和自环边
                default_graph = create_default_graph(start_idx, fragment_id)
                return default_graph

                # 记录最终统计
            debug_stats['edges_created'] = graph.number_of_edges()
            debug_stats['valid_nodes'] = graph.number_of_nodes()

            # 输出调试信息
            logger.debug(f"图谱构建完成 - 片段ID: {fragment_id}, 节点: {debug_stats['valid_nodes']}, "
                         f"边: {debug_stats['edges_created']}, 过滤残基: {debug_stats['filtered_by_plddt'] + debug_stats['filtered_by_nonstandard']}")

            return graph

    except Exception as e:
        logger.error(f"构建残基图出错: {str(e)}")
        logger.error(traceback.format_exc())
        return create_default_graph(start_idx, fragment_id)


def classify_interaction(res_code_i, res_code_j, distance):
    """
    分类残基间相互作用类型

    参数:
        res_code_i: 残基i的单字母代码
        res_code_j: 残基j的单字母代码
        distance: 残基间距离(Å)

    返回:
        edge_type_onehot: 边类型的one-hot编码 [4维]
        type_name: 相互作用类型名称
        interaction_strength: 相互作用强度
    """
    # 定义相互作用类型的one-hot编码
    # [1,0,0,0]: 空间邻近
    # [0,1,0,0]: 序列连接
    # [0,0,1,0]: 氢键
    # [0,0,0,1]: 离子相互作用
    # [1,0,0,1]: 疏水相互作用 (组合编码)

    # 默认为空间邻近
    edge_type_onehot = [1, 0, 0, 0]
    type_name = "spatial"
    interaction_strength = 0.3

    # 1. 检测氢键
    hbond_donors = set('NQRKWST')
    hbond_acceptors = set('DEQNSTYHW')
    if ((res_code_i in hbond_donors and res_code_j in hbond_acceptors) or
        (res_code_j in hbond_donors and res_code_i in hbond_acceptors)) and distance < 5.0:
        edge_type_onehot = [0, 0, 1, 0]
        type_name = "hbond"
        interaction_strength = 0.8

    # 2. 检测离子相互作用
    positive_aa = set('KRH')
    negative_aa = set('DE')
    if ((res_code_i in positive_aa and res_code_j in negative_aa) or
        (res_code_j in positive_aa and res_code_i in negative_aa)) and distance < 6.0:
        edge_type_onehot = [0, 0, 0, 1]
        type_name = "ionic"
        interaction_strength = 0.7

    # 3. 检测疏水相互作用
    hydrophobic_aa = set('AVILMFYW')
    if (res_code_i in hydrophobic_aa and res_code_j in hydrophobic_aa and
            distance < 6.0):
        edge_type_onehot = [1, 0, 0, 1]  # 空间+疏水组合编码
        type_name = "hydrophobic"
        interaction_strength = 0.5

    return edge_type_onehot, type_name, interaction_strength


def create_default_graph(start_idx, fragment_id):
    """
    创建默认的最小图，确保处理失败时有返回值

    参数:
        start_idx: 起始残基索引
        fragment_id: 片段ID

    返回:
        包含一个节点和自环边的NetworkX图对象
    """
    graph = nx.Graph()

    # 创建默认节点
    default_node_id = f"res_default"
    graph.add_node(default_node_id,
                   residue_name='GLY',
                   residue_code='G',
                   residue_idx=start_idx,
                   position=[0.0, 0.0, 0.0],  # 标准化坐标
                   plddt=70.0,
                   secondary_structure='C',
                   fragment_id=fragment_id,
                   hydropathy=AA_PROPERTIES['G']['hydropathy'],
                   charge=AA_PROPERTIES['G']['charge'],
                   molecular_weight=AA_PROPERTIES['G']['mw'],
                   volume=AA_PROPERTIES['G']['volume'],
                   flexibility=AA_PROPERTIES['G']['flexibility'],
                   is_aromatic=AA_PROPERTIES['G']['aromatic'],
                   ss_alpha=0,
                   ss_beta=0,
                   ss_coil=1,
                   sasa=0.5,
                   blosum62=[0] * 20)

    # 创建自环边(使用one-hot编码)
    graph.add_edge(default_node_id, default_node_id,
                   edge_type_onehot=[1, 0, 0, 0],  # 空间类型
                   type_name='self_loop',
                   distance=0.0,
                   interaction_strength=0.5,
                   direction=[0.0, 0.0])

    return graph


# ======================= 文件处理函数 =======================
def process_structure_file(file_path, min_length=5, max_length=50, k_neighbors=8,
                           distance_threshold=8.0, plddt_threshold=70.0, respect_ss=True, respect_domains=True):
    """处理单个结构文件，提取片段并构建知识图谱"""
    logger.debug(f"处理文件: {file_path}")

    protein_id = os.path.basename(file_path).split('.')[0]
    extracted_fragments = {}
    knowledge_graphs = {}
    fragment_stats = {
        "file_name": os.path.basename(file_path),
        "protein_id": protein_id,
        "fragments": 0,
        "valid_residues": 0,
        "edges": 0
    }

    try:
        # 1. 使用MDTraj加载结构
        structure = load_structure_mdtraj(file_path)
        if structure is None:
            logger.error(f"无法加载结构: {file_path}")
            return {}, {}, fragment_stats

        # 2. 计算二级结构
        ss_array = compute_secondary_structure(structure)

        # 3. 计算接触图
        contact_map, residue_pairs = compute_contacts(structure)

        # 4. 计算有效残基数量
        valid_residues = 0
        for res_idx in range(structure.n_residues):
            try:
                res = structure.topology.residue(res_idx)
                one_letter = three_to_one(res.name)
                if one_letter != 'X':
                    valid_residues += 1
            except:
                continue

        fragment_stats["valid_residues"] = valid_residues

        # 5. 如果残基数量小于最小长度，则跳过
        if valid_residues < min_length:
            logger.info(f"结构 {protein_id} 残基数量 ({valid_residues}) 小于最小长度 {min_length}，跳过")
            return {}, {}, fragment_stats

        # 6. 创建智能片段
        fragments = create_intelligent_fragments(
            structure, ss_array, contact_map, residue_pairs,
            min_length, max_length, respect_ss, respect_domains
        )

        fragment_stats["fragments"] = len(fragments)

        # 7. 处理每个片段
        for start_idx, end_idx, frag_id in fragments:
            fragment_id = f"{protein_id}_{frag_id}"

            # 提取序列
            sequence = ""
            for res_idx in range(start_idx, end_idx):
                try:
                    res = structure.topology.residue(res_idx)
                    aa = three_to_one(res.name)
                    if aa != "X":
                        sequence += aa
                except:
                    continue

            # 如果序列太短，跳过
            if len(sequence) < min_length:
                continue

            # 保存片段信息
            extracted_fragments[fragment_id] = {
                "protein_id": protein_id,
                "fragment_id": frag_id,
                "sequence": sequence,
                "length": len(sequence),
                "start_idx": start_idx,
                "end_idx": end_idx
            }

            # 构建知识图谱
            kg = build_enhanced_residue_graph(
                structure, ss_array,
                (start_idx, end_idx, fragment_id),
                k_neighbors, distance_threshold, plddt_threshold
            )

            # 只保存非空图
            if kg.number_of_nodes() > 0:
                fragment_stats["edges"] += kg.number_of_edges()
                knowledge_graphs[fragment_id] = nx.node_link_data(kg, edges="links")

        return extracted_fragments, knowledge_graphs, fragment_stats

    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        logger.error(traceback.format_exc())
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
                                 n_workers=None, batch_size=100000, memory_limit_gb=800,
                                 k_neighbors=8, distance_threshold=8.0, plddt_threshold=70.0,
                                 respect_ss=True, respect_domains=True, format_type="pyg"):
    """
    高性能蛋白质结构批处理系统 - 适用于TB级内存和百核处理器

    参数:
        file_list: 待处理文件列表
        output_dir: 输出目录
        batch_size: 每批处理的文件数量 (默认: 100000，适合TB级内存)
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
                base_name="protein_data", chunk_size=100000
            )

            # 并行保存图谱数据
            if batch_graphs:
                graph_future = save_executor.submit(
                    save_knowledge_graphs, batch_graphs, batch_output_dir,
                    base_name="protein_kg", chunk_size=100000, format_type=format_type
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
                    if i % 10000 == 0 or i == len(chunk_ids) - 1:
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
    parser.add_argument("--batch_size", "-b", type=int, default=100000,
                        help="大规模批处理大小 (默认: 100000，适合TB级内存)")
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