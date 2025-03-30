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
    """标准化坐标（中心化并缩放）"""
    if len(coords) == 0:
        return []

    # 转换为numpy数组
    coords_array = np.array(coords)

    # 计算质心
    centroid = np.mean(coords_array, axis=0)

    # 中心化坐标
    centered_coords = coords_array - centroid

    # 计算缩放因子（使用最大距离标准化）
    max_dist = np.max(np.sqrt(np.sum(centered_coords ** 2, axis=1)))

    # 避免除以零
    if max_dist > 0:
        normalized_coords = centered_coords / max_dist
    else:
        normalized_coords = centered_coords

    return normalized_coords.tolist()


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


def compute_ionic_interactions(structure, cutoff=0.4):
    """识别离子相互作用（盐桥）"""
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
            distances = md.compute_distances(structure, pairs)[0]

            # 找出小于阈值的接触对
            contacts = pairs[distances < cutoff]
            return contacts
        else:
            return []
    except Exception as e:
        logger.error(f"计算离子相互作用失败: {str(e)}")
        return []


def extract_plddt_from_bfactor(structure):
    """从B因子中提取pLDDT值（适用于AlphaFold模型）"""
    try:
        # 对每个残基，获取其B因子的平均值
        # 在AlphaFold模型中，B因子被用来存储pLDDT值
        b_factors = []

        for atom in structure.topology.atoms:
            res_idx = atom.residue.index
            b_factor = structure.xyz[0, atom.index, 0]  # 仅使用第一个坐标作为索引
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
    """构建残基级增强知识图谱，包含丰富的节点特征和边特征"""
    start_idx, end_idx, fragment_id = fragment_range
    graph = nx.Graph()

    # 检查切片范围有效性
    if start_idx < 0 or end_idx > structure.n_residues:
        logger.error(f"无效的片段范围: {start_idx}-{end_idx}，蛋白质残基数: {structure.n_residues}")
        return graph

    try:
        # 1. 计算所需的特征
        # 溶剂可及性
        sasa = compute_solvent_accessibility(structure)[0]

        # 接触图和残基对
        contact_map, residue_pairs = compute_contacts(structure)

        # 氢键
        hbonds = compute_hydrogen_bonds(structure)

        # 疏水接触
        hydrophobic_contacts = compute_hydrophobic_contacts(structure)

        # 离子相互作用
        ionic_interactions = compute_ionic_interactions(structure)

        # pLDDT值（如果可用，否则使用默认值）
        plddt_values = extract_plddt_from_bfactor(structure)

        # 2. 提取CA原子坐标用于后续操作
        ca_indices = [atom.index for atom in structure.topology.atoms if atom.name == 'CA']
        ca_coords = structure.xyz[0, ca_indices]  # 使用第一帧

        # 确保坐标数量与残基数量匹配
        if len(ca_coords) != structure.n_residues:
            logger.warning(f"CA原子数量 ({len(ca_coords)}) 与残基数量 ({structure.n_residues}) 不匹配")
            # 根据实际情况调整
            ca_coords = ca_coords[:structure.n_residues] if len(ca_coords) > structure.n_residues else \
                np.vstack([ca_coords, np.zeros((structure.n_residues - len(ca_coords), 3))])

        # 3. 添加节点（仅处理指定范围内的残基）
        for res_idx in range(start_idx, end_idx):
            # 跳过pLDDT低于阈值的残基
            if plddt_values[res_idx] < plddt_threshold:
                continue

            # 获取残基信息
            res = structure.topology.residue(res_idx)
            res_name = res.name
            one_letter = three_to_one(res_name)

            # 跳过非标准氨基酸
            if one_letter == 'X':
                continue

            # 计算节点特征
            # BLOSUM62编码
            blosum_encoding = get_blosum62_encoding(one_letter)

            # 相对空间坐标
            ca_coord = ca_coords[res_idx].tolist() if res_idx < len(ca_coords) else [0, 0, 0]

            # 二级结构（确保使用一致的索引）
            ss_code = ss_array[0, res_idx] if ss_array.ndim > 1 else ss_array[res_idx]

            # 二级结构独热编码
            ss_onehot = [0, 0, 0]  # [alpha, beta, coil/other]
            if ss_code in ['H', 'G', 'I']:
                ss_onehot[0] = 1  # alpha
            elif ss_code in ['E', 'B']:
                ss_onehot[1] = 1  # beta
            else:
                ss_onehot[2] = 1  # coil/other

            # 添加节点属性
            node_attrs = {
                # 基本信息
                'residue_name': res_name,
                'residue_code': one_letter,
                'residue_idx': res_idx,
                'position': ca_coord,
                'plddt': float(plddt_values[res_idx]),

                # 氨基酸理化属性
                'hydropathy': AA_PROPERTIES[one_letter]['hydropathy'],
                'charge': AA_PROPERTIES[one_letter]['charge'],
                'molecular_weight': AA_PROPERTIES[one_letter]['mw'],
                'volume': AA_PROPERTIES[one_letter]['volume'],
                'flexibility': AA_PROPERTIES[one_letter]['flexibility'],
                'is_aromatic': AA_PROPERTIES[one_letter]['aromatic'],

                # 结构信息
                'secondary_structure': ss_code,
                'ss_alpha': ss_onehot[0],
                'ss_beta': ss_onehot[1],
                'ss_coil': ss_onehot[2],
                'sasa': float(sasa[res_idx]),

                # 高级特征
                'blosum62': blosum_encoding,
                'fragment_id': fragment_id
            }

            # 添加节点
            node_id = f"res_{res_idx}"
            graph.add_node(node_id, **node_attrs)

        # 4. 添加序列连接边
        all_nodes = list(graph.nodes())
        for i in range(len(all_nodes) - 1):
            node1 = all_nodes[i]
            node2 = all_nodes[i + 1]

            # 确保是序列上相邻的残基
            res_idx1 = graph.nodes[node1]['residue_idx']
            res_idx2 = graph.nodes[node2]['residue_idx']

            if res_idx2 == res_idx1 + 1:
                # 计算CA原子间的实际距离
                pos1 = graph.nodes[node1]['position']
                pos2 = graph.nodes[node2]['position']
                distance = float(np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2)))

                # 序列相邻边 - 类型1
                graph.add_edge(node1, node2,
                               edge_type=1,
                               type_name='peptide',
                               distance=distance,
                               interaction_strength=1.0,
                               direction=[1.0, 0.0])  # N->C方向

        # 5. 添加空间相互作用边
        # 构建KD树用于空间查询
        node_ids = list(graph.nodes())
        node_positions = np.array([graph.nodes[n]['position'] for n in node_ids])

        if len(node_positions) > 1:
            try:
                kd_tree = KDTree(node_positions)

                # 查询k个最近邻
                k = min(k_neighbors + 1, len(node_positions))
                distances, indices = kd_tree.query(node_positions, k=k)

                # 处理查询结果
                for i, neighbors in enumerate(indices):
                    node_i = node_ids[i]
                    res_i = graph.nodes[node_i]['residue_idx']

                    for j_idx, dist in zip(neighbors[1:], distances[i, 1:]):
                        if j_idx < len(node_ids):  # 确保索引有效
                            node_j = node_ids[j_idx]
                            res_j = graph.nodes[node_j]['residue_idx']

                            # 跳过序列相邻的残基（已添加序列边）
                            if abs(res_j - res_i) <= 1:
                                continue

                            # 仅连接距离小于阈值的残基
                            if dist <= distance_threshold:
                                # 检查是否已有边
                                if not graph.has_edge(node_i, node_j):
                                    # 确定相互作用类型和强度
                                    is_hbond = any((res_i, res_j) in hbonds or (res_j, res_i) in hbonds)
                                    is_hydrophobic = any((res_i, res_j) in hydrophobic_contacts or
                                                         (res_j, res_i) in hydrophobic_contacts)
                                    is_ionic = any((res_i, res_j) in ionic_interactions or
                                                   (res_j, res_i) in ionic_interactions)

                                    # 基本边类型：空间邻近 - 类型0
                                    edge_type = 0
                                    type_name = 'spatial'

                                    # 根据相互作用类型细分边
                                    if is_hbond:
                                        edge_type = 2
                                        type_name = 'hbond'
                                        interaction_strength = 0.8
                                    elif is_ionic:
                                        edge_type = 3
                                        type_name = 'ionic'
                                        interaction_strength = 0.7
                                    elif is_hydrophobic:
                                        edge_type = 4
                                        type_name = 'hydrophobic'
                                        interaction_strength = 0.5
                                    else:
                                        interaction_strength = 0.3

                                    # 计算方向向量
                                    pos_i = np.array(graph.nodes[node_i]['position'])
                                    pos_j = np.array(graph.nodes[node_j]['position'])
                                    direction_vector = pos_j - pos_i
                                    norm = np.linalg.norm(direction_vector)

                                    if norm > 0:
                                        direction_vector = direction_vector / norm
                                    else:
                                        direction_vector = np.array([0, 0])

                                    # 添加边
                                    graph.add_edge(node_i, node_j,
                                                   edge_type=edge_type,
                                                   type_name=type_name,
                                                   distance=float(dist),
                                                   interaction_strength=interaction_strength,
                                                   direction=direction_vector[:2].tolist())

            except Exception as e:
                logger.warning(f"空间边构建失败: {str(e)}")

        return graph
    except Exception as e:
        logger.error(f"构建残基图时出错: {str(e)}")
        logger.error(traceback.format_exc())
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


def process_file_parallel(file_path, min_length=5, max_length=50, k_neighbors=8,
                          distance_threshold=8.0, plddt_threshold=70.0, respect_ss=True, respect_domains=True):
    """并行处理单个文件的函数"""
    try:
        start_time = time.time()

        # 处理文件
        proteins, kg, fragment_stats = process_structure_file(
            file_path, min_length, max_length, k_neighbors,
            distance_threshold, plddt_threshold, respect_ss, respect_domains
        )

        elapsed = time.time() - start_time
        file_name = os.path.basename(file_path)

        # 结果统计
        fragment_count = len(proteins)
        kg_count = len(kg)

        result_info = {
            "file_path": file_path,
            "elapsed": elapsed,
            "fragment_count": fragment_count,
            "kg_count": kg_count,
            "success": True
        }
        return result_info, proteins, kg, fragment_stats
    except Exception as e:
        logger.error(f"处理文件失败: {file_path} - {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "file_path": file_path}, {}, {}, None


def process_file_batch_parallel(file_list, output_dir, min_length=5, max_length=50, n_workers=None,
                                k_neighbors=8, distance_threshold=8.0, plddt_threshold=70.0,
                                respect_ss=True, respect_domains=True, format_type="pyg"):
    """并行处理多个文件，每batch_size个文件保存一次结果"""
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    logger.info(f"使用 {n_workers} 个CPU核心并行处理...")
    logger.info(f"处理参数: 最小长度={min_length}, 最大长度={max_length}, "
                f"k近邻={k_neighbors}, 距离阈值={distance_threshold}埃, "
                f"pLDDT阈值={plddt_threshold}, 结构划分方式: 基于二级结构={respect_ss}, 基于结构域={respect_domains}")

    # 创建ALL子文件夹用于存储中间结果
    all_data_dir = os.path.join(output_dir, "ALL")
    os.makedirs(all_data_dir, exist_ok=True)

    # 将日志文件保存在输出目录中
    sequences_log_path = os.path.join(output_dir, "sequences.log")
    fragments_log_path = os.path.join(output_dir, "fragments_stats.log")
    processing_log_path = os.path.join(output_dir, "processing.log")

    all_proteins = {}
    all_knowledge_graphs = {}
    all_fragment_stats = []
    stats = defaultdict(int)
    batch_size = 1000

    # 初始化统计数据键
    stats["processed_files"] = 0
    stats["extracted_fragments"] = 0
    stats["knowledge_graphs"] = 0
    stats["failed_files"] = 0
    stats["total_edges"] = 0

    # 初始化文件计数器和批次ID
    file_counter = 0
    batch_id = 1

    with open(sequences_log_path, 'w', buffering=1) as s_log:
        s_log.write("fragment_id,protein_id,length,sequence\n")

    with open(fragments_log_path, 'w') as f_log:
        f_log.write("file_name,protein_id,valid_residues,fragments,edges\n")

    with open(processing_log_path, 'w') as p_log:
        p_log.write("timestamp,file_path,status,elapsed,fragments,knowledge_graphs,error\n")

    # 使用进程池进行并行处理
    with tqdm(total=len(file_list), desc="处理结构文件") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    process_file_parallel, file_path, min_length, max_length,
                    k_neighbors, distance_threshold, plddt_threshold, respect_ss, respect_domains
                ) for file_path in file_list
            ]

            for future in concurrent.futures.as_completed(futures):
                result, proteins, kg, fragment_stats = future.result()
                pbar.update(1)

                file_counter += 1

                if "error" in result:
                    stats["failed_files"] += 1
                    logger.error(f"处理文件失败: {result['file_path']} - {result['error']}")
                    # 记录失败日志
                    with open(processing_log_path, 'a') as p_log:
                        p_log.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')},{result['file_path']},FAILED,0,0,0,{result['error']}\n")
                    continue

                # 更新统计数据
                stats["processed_files"] += 1
                stats["extracted_fragments"] += len(proteins)
                stats["knowledge_graphs"] += len(kg)
                if fragment_stats:
                    stats["total_edges"] += fragment_stats.get('edges', 0)

                # 记录处理日志
                with open(processing_log_path, 'a') as p_log:
                    elapsed = result.get('elapsed', 0)
                    status = "SUCCESS" if "success" in result and result["success"] else "FAILED"
                    error = result.get('error', '')
                    p_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{result['file_path']},{status},"
                                f"{elapsed:.2f},{len(proteins)},{len(kg)},{error}\n")

                # 记录片段统计
                if fragment_stats:
                    all_fragment_stats.append(fragment_stats)
                    with open(fragments_log_path, 'a') as f_log:
                        f_log.write(f"{fragment_stats['file_name']},{fragment_stats['protein_id']},"
                                    f"{fragment_stats.get('valid_residues', 0)},{fragment_stats.get('fragments', 0)},"
                                    f"{fragment_stats.get('edges', 0)}\n")

                # 更新序列日志
                with open(sequences_log_path, 'a', buffering=1) as s_log:
                    for fragment_id, data in proteins.items():
                        s_log.write(
                            f"{fragment_id},{data['protein_id']},{len(data['sequence'])},{data['sequence']}\n")

                # 累积数据
                all_proteins.update(proteins)
                all_knowledge_graphs.update(kg)

                # 每处理指定数量文件保存一次中间结果
                if file_counter % batch_size == 0:
                    logger.info(f"已处理 {file_counter}/{len(file_list)} 个文件，保存当前批次结果...")

                    batch_output_dir = os.path.join(all_data_dir, f"batch_{batch_id}")
                    os.makedirs(batch_output_dir, exist_ok=True)

                    # 保存蛋白质数据
                    save_results_chunked(
                        all_proteins, batch_output_dir,
                        base_name="protein_data",
                        chunk_size=batch_size
                    )

                    # 保存知识图谱
                    if all_knowledge_graphs:
                        save_knowledge_graphs(
                            all_knowledge_graphs, batch_output_dir,
                            base_name="protein_kg",
                            chunk_size=batch_size,
                            format_type=format_type
                        )

                    # 重置集合并增加批次ID
                    all_proteins = {}
                    all_knowledge_graphs = {}
                    batch_id += 1

                    # 执行垃圾回收
                    check_memory_usage(force_gc=True)

                    # 更新统计信息
                    logger.info(f"当前处理统计: 成功={stats['processed_files']}, 失败={stats['failed_files']}, "
                                f"片段={stats['extracted_fragments']}, 知识图谱={stats['knowledge_graphs']}")

    # 保存最后一批数据
    if all_proteins:
        logger.info(f"保存最后一批结果 (剩余 {len(all_proteins)} 个蛋白质片段)...")

        final_batch_dir = os.path.join(all_data_dir, f"batch_{batch_id}")
        os.makedirs(final_batch_dir, exist_ok=True)

        # 保存蛋白质数据
        save_results_chunked(
            all_proteins, final_batch_dir,
            base_name="protein_data",
            chunk_size=batch_size
        )

        # 保存知识图谱
        if all_knowledge_graphs:
            save_knowledge_graphs(
                all_knowledge_graphs, final_batch_dir,
                base_name="protein_kg",
                chunk_size=batch_size,
                format_type=format_type
            )

    logger.info(f"片段统计数据已写入: {fragments_log_path}")
    logger.info(f"处理日志已写入: {processing_log_path}")
    logger.info(f"序列信息已写入: {sequences_log_path}")
    logger.info(f"创建的片段总数: {stats['extracted_fragments']}")

    return stats, all_data_dir


def save_results_chunked(all_proteins, output_dir, base_name="protein_data", chunk_size=10000):
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


def save_knowledge_graphs(kg_data, output_dir, base_name="protein_kg", chunk_size=10000, format_type="pyg"):
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

                            # 6. 保守性得分 - 暂无，使用默认值 (1维)
                            conservation = 0.5

                            # 7. 侧链柔性 (1维)
                            side_chain_flexibility = float(props['flexibility'])

                            # 8. pLDDT值 (1维)
                            plddt = float(node_attrs.get('plddt', 70.0)) / 100.0  # 归一化到0-1

                            # 合并所有特征
                            features = blosum + position + [hydropathy, charge, molecular_weight,
                                                            volume, flexibility, is_aromatic,
                                                            ss_alpha, ss_beta, ss_coil, sasa,
                                                            conservation, side_chain_flexibility, plddt]

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
    """主函数"""
    parser = argparse.ArgumentParser(description="从PDB/CIF文件中提取蛋白质结构数据并构建知识图谱（MDTraj增强版）")
    parser.add_argument("input", help="输入PDB/CIF文件或包含这些文件的目录")
    parser.add_argument("--output_dir", "-o", default="./kg",
                        help="输出目录 (默认: ./kg)")
    parser.add_argument("--min_length", "-m", type=int, default=5,
                        help="最小序列长度 (默认: 5)")
    parser.add_argument("--max_length", "-M", type=int, default=50,
                        help="最大序列长度 (默认: 50)")
    parser.add_argument("--batch_size", "-b", type=int, default=10000,
                        help="保存批次大小 (默认: 10000)")
    parser.add_argument("--workers", "-w", type=int,
                        help="并行工作进程数 (默认: CPU核心数-1)")
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
    global logger  # 使用全局logger变量
    logger, log_file_path = setup_logging(args.output_dir)
    logger.info(f"日志将写入文件: {log_file_path}")

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

    # 处理文件 - 返回统计信息和中间数据目录
    stats, all_data_dir = process_file_batch_parallel(
        pdb_files,
        args.output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        n_workers=args.workers,
        k_neighbors=args.k_neighbors,
        distance_threshold=args.distance_threshold,
        plddt_threshold=args.plddt_threshold,
        respect_ss=args.respect_ss,
        respect_domains=args.respect_domains,
        format_type=args.format
    )

    # 处理结果统计
    logger.info("\n处理完成:")
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
                "format": args.format
            },
            "output_dir": os.path.abspath(args.output_dir),
            "all_data_dir": os.path.abspath(all_data_dir)
        }
        json.dump(summary, f, indent=2)

    logger.info(f"摘要信息已保存到: {summary_file}")
    logger.info("蛋白质结构提取与知识图谱构建流程完成！")


if __name__ == "__main__":
    main()