#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质结构解析与知识图谱构建工具

该脚本从PDB/CIF文件中提取蛋白质结构信息，生成序列片段并构建知识图谱。
支持多进程并行处理和GPU加速。

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
import torch
import numpy as np
import networkx as nx
from Bio import PDB
from Bio.PDB import DSSP
from Bio.SeqUtils import seq1
from scipy.spatial import KDTree
from collections import defaultdict
from tqdm import tqdm
from torch_geometric.data import Data
import time
import sys
import platform
import traceback

# ======================= 常量与配置 =======================

# 设置基本日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 氨基酸物理化学性质常量
AA_PROPERTIES = {
    'A': {'hydropathy': 1.8, 'charge': 0, 'polar': False, 'mw': 89.09},
    'C': {'hydropathy': 2.5, 'charge': 0, 'polar': False, 'mw': 121.15},
    'D': {'hydropathy': -3.5, 'charge': -1, 'polar': True, 'mw': 133.10},
    'E': {'hydropathy': -3.5, 'charge': -1, 'polar': True, 'mw': 147.13},
    'F': {'hydropathy': 2.8, 'charge': 0, 'polar': False, 'mw': 165.19},
    'G': {'hydropathy': -0.4, 'charge': 0, 'polar': False, 'mw': 75.07},
    'H': {'hydropathy': -3.2, 'charge': 0.1, 'polar': True, 'mw': 155.16},
    'I': {'hydropathy': 4.5, 'charge': 0, 'polar': False, 'mw': 131.17},
    'K': {'hydropathy': -3.9, 'charge': 1, 'polar': True, 'mw': 146.19},
    'L': {'hydropathy': 3.8, 'charge': 0, 'polar': False, 'mw': 131.17},
    'M': {'hydropathy': 1.9, 'charge': 0, 'polar': False, 'mw': 149.21},
    'N': {'hydropathy': -3.5, 'charge': 0, 'polar': True, 'mw': 132.12},
    'P': {'hydropathy': -1.6, 'charge': 0, 'polar': False, 'mw': 115.13},
    'Q': {'hydropathy': -3.5, 'charge': 0, 'polar': True, 'mw': 146.15},
    'R': {'hydropathy': -4.5, 'charge': 1, 'polar': True, 'mw': 174.20},
    'S': {'hydropathy': -0.8, 'charge': 0, 'polar': True, 'mw': 105.09},
    'T': {'hydropathy': -0.7, 'charge': 0, 'polar': True, 'mw': 119.12},
    'V': {'hydropathy': 4.2, 'charge': 0, 'polar': False, 'mw': 117.15},
    'W': {'hydropathy': -0.9, 'charge': 0, 'polar': True, 'mw': 204.23},
    'Y': {'hydropathy': -1.3, 'charge': 0, 'polar': True, 'mw': 181.19}
}


# ======================= 日志和工具函数 =======================
def check_memory_usage(threshold_gb=None, force_gc=False):
    """
    检查内存使用情况，如果超过阈值则进行垃圾回收

    参数:
        threshold_gb: 内存使用阈值（GB），超过此值将执行垃圾回收。如果为None，使用系统内存的80%作为阈值
        force_gc: 是否强制执行垃圾回收，无论内存使用量如何

    返回:
        bool: 如果执行了垃圾回收则返回True，否则返回False
    """
    import gc
    import psutil
    import logging
    import torch

    # 创建logger或使用已有的logger
    try:
        logger = logging.getLogger()
    except:
        # 如果无法获取logger，创建一个基本的
        logging.basicConfig(level=logging.INFO,
                           format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()

    # 首先检查是否需要强制执行垃圾回收
    if force_gc:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        # 获取当前进程
        process = psutil.Process()

        # 获取当前内存使用量（RSS：Resident Set Size，实际使用的物理内存）
        mem_used_bytes = process.memory_info().rss
        mem_used_gb = mem_used_bytes / (1024 ** 3)  # 转换为GB

        # 获取系统总内存
        if threshold_gb is None:
            system_mem = psutil.virtual_memory()
            total_mem_gb = system_mem.total / (1024 ** 3)
            threshold_gb = total_mem_gb * 0.8  # 默认使用系统内存的80%作为阈值

        # 检查是否超过阈值
        if mem_used_gb > threshold_gb:
            logger.warning(f"内存使用已达 {mem_used_gb:.2f} GB，超过阈值 {threshold_gb:.2f} GB，执行垃圾回收")

            # 执行垃圾回收
            collected = gc.collect()

            # 如果有GPU，清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 再次检查内存使用量，查看回收效果
            mem_after_gc = process.memory_info().rss / (1024 ** 3)
            mem_freed = mem_used_gb - mem_after_gc

            if mem_freed > 0:
                logger.info(f"垃圾回收释放了 {mem_freed:.2f} GB 内存，当前使用: {mem_after_gc:.2f} GB")
            else:
                logger.warning(f"垃圾回收未能释放内存，当前使用: {mem_after_gc:.2f} GB")

            return True
        else:
            if force_gc:
                logger.info(f"当前内存使用: {mem_used_gb:.2f} GB (阈值: {threshold_gb:.2f} GB)")
            return False

    except Exception as e:
        logger.warning(f"检查内存使用时出错: {str(e)}")

        # 如果无法获取内存信息，执行垃圾回收作为预防措施
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
        return False


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


# ======================= 序列和片段处理函数 =======================

def create_protein_fragments(chain, residues, dssp=None, max_length=50, step_size=25, respect_ss=False):
    """将长蛋白质链分割为重叠片段，使用步长控制重叠量"""
    fragments = []

    # 过滤出标准氨基酸
    standard_residues = []
    for res in residues:
        try:
            aa = three_to_one(res.get_resname())
            if aa != "X" and 'CA' in res:  # 确保是标准氨基酸且有CA原子
                standard_residues.append(res)
        except Exception:
            logger.debug(f"跳过非标准残基: {res.get_resname() if hasattr(res, 'get_resname') else 'Unknown'}")
            continue

    if not standard_residues:
        logger.debug(f"链 {chain.get_id()} 没有标准残基")
        return []

    if len(standard_residues) <= max_length:
        # 如果蛋白质足够短，直接返回整个链
        return [(standard_residues, "full")]

    # 根据是否尊重二级结构选择分段方法
    if respect_ss and dssp:
        try:
            # 寻找二级结构边界
            ss_boundaries = []
            prev_ss = None

            for i, res in enumerate(standard_residues):
                res_key = (chain.get_id(), res.get_id())
                if res_key in dssp:
                    curr_ss = dssp[res_key][2]  # 二级结构类型
                    if prev_ss != curr_ss:
                        ss_boundaries.append(i)
                        prev_ss = curr_ss

            # 确保添加最后一个边界
            if not ss_boundaries or ss_boundaries[-1] != len(standard_residues):
                ss_boundaries.append(len(standard_residues))

            # 基于二级结构边界创建片段
            for i in range(len(ss_boundaries) - 1):
                start = ss_boundaries[i]
                for j in range(i + 1, min(i + 3, len(ss_boundaries))):
                    end = ss_boundaries[j]
                    if 10 <= end - start <= max_length:
                        fragment = standard_residues[start:end]
                        frag_id = f"{standard_residues[start].get_id()[1]}-{standard_residues[end - 1].get_id()[1]}"
                        fragments.append((fragment, frag_id))

            if fragments:
                return fragments

        except Exception as e:
            logger.warning(f"基于二级结构的分段失败: {str(e)}，回退到常规滑动窗口")
            respect_ss = False

    # 如果二级结构划分失败，使用改进的方法
    if not fragments:
        # 随机选择使用自适应随机窗口或基于局部特征的边界
        if random.choice([True, False]):
            # 方法1：使用自适应随机窗口
            return create_protein_fragments_adaptive(chain, standard_residues, max_length)
        else:
            # 方法2：使用基于局部特征的边界
            boundaries = find_natural_boundaries(standard_residues)
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                # 合并过小的片段
                while end - start < 10 and i < len(boundaries) - 2:
                    i += 1
                    end = boundaries[i + 1]

                # 拆分过大的片段
                if end - start > max_length:
                    for j in range(start, end, step_size):
                        sub_end = min(j + max_length, end)
                        fragment = standard_residues[j:sub_end]
                        frag_id = f"{fragment[0].get_id()[1]}-{fragment[-1].get_id()[1]}"
                        fragments.append((fragment, frag_id))
                else:
                    fragment = standard_residues[start:end]
                    frag_id = f"{fragment[0].get_id()[1]}-{fragment[-1].get_id()[1]}"
                    fragments.append((fragment, frag_id))

    # 常规滑动窗口，如果上述方法都没创建片段
    if not fragments:
        for start in range(0, len(standard_residues) - max_length + 1, step_size):
            fragment = standard_residues[start:start + max_length]
            start_res_num = fragment[0].get_id()[1]
            end_res_num = fragment[-1].get_id()[1]
            frag_id = f"{start_res_num}-{end_res_num}"
            fragments.append((fragment, frag_id))

        # 确保添加最后一个片段（如果不完整）
        last_start = len(standard_residues) - max_length
        if last_start % step_size != 0 and last_start > 0:
            last_fragment = standard_residues[-max_length:]
            start_res_num = last_fragment[0].get_id()[1]
            end_res_num = last_fragment[-1].get_id()[1]
            last_frag_id = f"{start_res_num}-{end_res_num}"
            fragments.append((last_fragment, last_frag_id))

    return fragments


def create_protein_fragments_adaptive(chain, residues, max_length=50, min_length=20):
    """使用自适应随机窗口创建片段"""
    fragments = []
    standard_residues = [res for res in residues if three_to_one(res.get_resname()) != "X" and 'CA' in res]

    if len(standard_residues) <= max_length:
        # 使用链ID作为片段标识符的一部分
        chain_id = chain.get_id()
        return [(standard_residues, f"{chain_id}_full")]

    # 使用随机长度和步长
    current_pos = 0
    while current_pos < len(standard_residues):
        # 随机选择片段长度 (20-50)
        frag_length = random.randint(min_length, max_length)

        if current_pos + frag_length > len(standard_residues):
            # 处理最后一个片段
            frag_length = len(standard_residues) - current_pos

        fragment = standard_residues[current_pos:current_pos + frag_length]
        frag_id = f"{fragment[0].get_id()[1]}-{fragment[-1].get_id()[1]}"
        fragments.append((fragment, frag_id))

        # 确定步长
        if frag_length <= 20:
            # 小片段使用固定步长
            step = max(1, frag_length // 2)
        else:
            # 确保随机范围有效
            min_step = 5  # 最小步长5
            max_step = min(frag_length // 2, 25)  # 最大步长25或片段长度的一半

            # 确保min_step <= max_step
            if min_step > max_step:
                step = max_step
            else:
                step = random.randint(min_step, max_step)

        current_pos += step

    return fragments


def find_natural_boundaries(residues, window=5):
    """基于局部特征变化识别自然分割点"""
    boundaries = [0]  # 起始点

    # 计算滑动窗口的物理化学特性变化
    scores = []
    for i in range(len(residues) - window):
        window1 = residues[i:i + window]
        window2 = residues[i + window:i + 2 * window] if i + 2 * window <= len(residues) else residues[i + window:]

        # 计算窗口内氨基酸特性
        hydro1 = sum(AA_PROPERTIES[three_to_one(r.get_resname())]['hydropathy'] for r in window1) / len(window1)
        hydro2 = sum(AA_PROPERTIES[three_to_one(r.get_resname())]['hydropathy'] for r in window2) / len(window2)

        # 特性变化分数
        scores.append(abs(hydro1 - hydro2))

    # 找出显著变化点 (局部最大值)
    for i in range(1, len(scores) - 1):
        if scores[i] > scores[i - 1] and scores[i] > scores[i + 1] and scores[i] > np.mean(scores) + 0.5 * np.std(
                scores):
            boundaries.append(i + window)

    # 添加结束点
    boundaries.append(len(residues))

    return boundaries


def run_dssp(structure, file_path):
    """运行DSSP分析并获取二级结构信息"""
    try:
        model = structure[0]
        dssp = DSSP(model, file_path, dssp='mkdssp')
        return dssp
    except Exception as e:
        logger.warning(f"DSSP分析失败: {str(e)}")
        return None


# ======================= 知识图谱构建函数 =======================

def build_residue_graph(chain, residues, dssp=None, k_neighbors=8, distance_threshold=8.0,
                        fragment_id=None, plddt_threshold=70):
    """为蛋白质链或片段构建残基知识图谱"""
    graph = nx.Graph()

    if not residues:
        return graph

    # 1. 添加节点
    ca_coords = []
    ca_atoms = []
    standard_aa_set = set('ACDEFGHIKLMNPQRSTVWY')

    for i, res in enumerate(residues):
        try:
            res_id = res.get_id()
            res_num = res_id[1]
            res_name = res.get_resname()
            aa_code = three_to_one(res_name)

            # 跳过非标准氨基酸
            if aa_code not in standard_aa_set:
                continue

            # 尝试获取CA原子坐标和pLDDT值
            ca_atom = res['CA']
            ca_coord = ca_atom.get_coord()

            # 从B-factor获取pLDDT值 (AlphaFold模型特有)
            plddt = ca_atom.get_bfactor()

            # 跳过低质量预测的残基
            if plddt < plddt_threshold:
                continue

            # 添加节点属性
            node_attrs = {
                'residue_name': res_name,
                'residue_code': aa_code,
                'residue_num': res_num,
                'plddt': plddt,
                'position': ca_coord.tolist(),
                'chain_id': chain.get_id()
            }

            # 添加氨基酸物理化学属性
            if aa_code in AA_PROPERTIES:
                props = AA_PROPERTIES[aa_code]
                node_attrs.update({
                    'hydropathy': props['hydropathy'],
                    'charge': props['charge'],
                    'polar': props['polar'],
                    'molecular_weight': props['mw']
                })

            # 获取二级结构(如果可用)
            if dssp:
                res_key = (chain.get_id(), res_id)
                if res_key in dssp:
                    ss = dssp[res_key][2]
                    acc = dssp[res_key][3]
                    phi = dssp[res_key][4]
                    psi = dssp[res_key][5]
                    node_attrs.update({
                        'secondary_structure': ss,
                        'accessible_area': acc,
                        'phi': phi,
                        'psi': psi
                    })
            else:
                node_attrs['secondary_structure'] = 'X'  # 未知二级结构

            # 添加片段ID (如果有)
            if fragment_id:
                node_attrs['fragment_id'] = fragment_id

            # 添加节点
            node_id = f"{chain.get_id()}_{res_num}"
            graph.add_node(node_id, **node_attrs)

            # 存储坐标用于后续空间邻接计算
            ca_coords.append(ca_coord)
            ca_atoms.append(node_id)

        except KeyError:  # CA原子缺失
            continue
        except Exception as e:
            logger.debug(f"添加节点时出错: {str(e)}")

    # 2. 添加序列邻接边
    node_ids = list(graph.nodes())
    for i in range(len(node_ids) - 1):
        node1 = node_ids[i]
        node2 = node_ids[i + 1]
        # 确保是序列上相邻的残基
        res_num1 = int(graph.nodes[node1]['residue_num'])
        res_num2 = int(graph.nodes[node2]['residue_num'])
        if res_num2 == res_num1 + 1:
            graph.add_edge(node1, node2, edge_type='peptide', weight=1.0, distance=1.0)

    # 3. 添加空间邻接边
    if len(ca_coords) > 1:
        try:
            # 构建KD树用于快速空间查询
            kd_tree = KDTree(ca_coords)

            # 对每个残基找到k个最近邻
            _, indices = kd_tree.query(ca_coords, k=min(k_neighbors + 1, len(ca_coords)))

            # 添加空间邻接边
            for i, neighbors in enumerate(indices):
                for j_idx in neighbors[1:]:  # 跳过自身(第一个结果)
                    j = int(j_idx)

                    # 计算残基间距离
                    dist = float(np.linalg.norm(np.array(ca_coords[i]) - np.array(ca_coords[j])))

                    # 只连接距离小于阈值的残基
                    if dist <= distance_threshold:
                        if not graph.has_edge(ca_atoms[i], ca_atoms[j]):
                            graph.add_edge(ca_atoms[i], ca_atoms[j],
                                           weight=dist, edge_type='spatial', distance=dist)
        except Exception as e:
            logger.warning(f"计算空间邻接时出错: {str(e)}")

    return graph


# ======================= 文件处理函数 =======================

def parse_pdb_file(pdb_file_path, max_length=50, step_size=25, build_kg=True,
                   k_neighbors=8, distance_threshold=8.0, plddt_threshold=70, respect_ss=True):
    """解析PDB/CIF文件并提取蛋白质信息"""
    logger.debug(f"解析文件: {pdb_file_path}")

    # 确定文件类型和处理方式
    is_gzipped = pdb_file_path.endswith('.gz')
    is_cif = '.cif' in pdb_file_path
    protein_id = os.path.basename(pdb_file_path).split('.')[0]
    temp_file = None
    extracted_chains = {}
    knowledge_graphs = {}

    # 添加片段统计跟踪
    fragment_stats = {
        "file_name": os.path.basename(pdb_file_path),
        "protein_id": protein_id,
        "chains": {}
    }

    try:
        # 处理文件格式
        if is_gzipped:
            temp_file = f"/tmp/{os.path.basename(pdb_file_path)[:-3]}"
            with gzip.open(pdb_file_path, 'rb') as f_in:
                with open(temp_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            file_to_parse = temp_file
        else:
            file_to_parse = pdb_file_path

        # 检查文件
        if not os.path.exists(file_to_parse) or os.path.getsize(file_to_parse) == 0:
            logger.error(f"文件不存在或为空: {pdb_file_path}")
            return {}, {}, None

        # 选择解析器
        parser = PDB.MMCIFParser() if is_cif else PDB.PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(protein_id, file_to_parse)
        except Exception as e:
            logger.error(f"解析结构失败: {str(e)}")
            return {}, {}, None

        # 尝试运行DSSP
        dssp_dict = None
        try:
            dssp_dict = run_dssp(structure, file_to_parse)
        except Exception as e:
            logger.debug(f"DSSP分析失败: {str(e)}")

        # 处理每个模型和链
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                # 初始化链统计
                fragment_stats["chains"][chain_id] = {
                    "valid_residues": 0,
                    "fragments": 0
                }
                # 收集有效残基
                valid_residues = []
                for res in chain:
                    if res.get_id()[0] == ' ':  # 跳过异常残基
                        try:
                            if 'CA' in res:  # 确保残基有CA原子
                                valid_residues.append(res)
                        except:
                            continue
                # 更新残基数量
                fragment_stats["chains"][chain_id]["valid_residues"] = len(valid_residues)

                # 如果有足够的残基，创建片段
                if len(valid_residues) >= 5:  # 至少需要5个残基
                    fragments = create_protein_fragments(
                        chain, valid_residues, dssp_dict,
                        max_length, step_size, respect_ss
                    )

                    # 更新片段数量
                    fragment_stats["chains"][chain_id]["fragments"] = len(fragments)

                    # 处理每个片段
                    for frag_residues, frag_id in fragments:
                        if not frag_residues:
                            continue

                        fragment_id = f"{protein_id}_{chain_id}_{frag_id}"

                        # 提取序列
                        sequence = ""
                        for res in frag_residues:
                            try:
                                aa = three_to_one(res.get_resname())
                                if aa != "X":
                                    sequence += aa
                            except:
                                continue

                        # 保存片段信息
                        extracted_chains[fragment_id] = {
                            "protein_id": protein_id,
                            "chain_id": chain_id,
                            "fragment_id": frag_id,
                            "sequence": sequence,
                            "length": len(sequence)
                        }

                        # 构建知识图谱
                        if build_kg:
                            kg = build_residue_graph(
                                chain, frag_residues, dssp_dict,
                                k_neighbors, distance_threshold,
                                fragment_id, plddt_threshold
                            )
                            # 只保存非空图
                            if kg.number_of_nodes() > 0:
                                knowledge_graphs[fragment_id] = nx.node_link_data(kg, edges="links")

        return extracted_chains, knowledge_graphs, fragment_stats

    except Exception as e:
        logger.error(f"解析文件 {pdb_file_path} 时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {}, {}, None
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


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


def process_file_parallel(file_path, max_length=50, step_size=25, build_kg=True,
                          k_neighbors=8, distance_threshold=8.0, plddt_threshold=70, respect_ss=True):
    """并行处理单个文件的函数"""
    try:
        start_time = time.time()

        # 文件检查
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return {"error": f"文件不存在或为空: {file_path}"}, {}, {}, None

        # 处理文件
        proteins, kg, fragment_stats = parse_pdb_file(
            file_path, max_length, step_size, build_kg,
            k_neighbors, distance_threshold, plddt_threshold, respect_ss
        )

        elapsed = time.time() - start_time
        file_name = os.path.basename(file_path)

        # 结果统计
        chain_count = len(set([info["chain_id"] for info in proteins.values()])) if proteins else 0
        fragment_count = len(proteins)

        result_info = {
            "file_path": file_path,
            "elapsed": elapsed,
            "protein_count": len(proteins),
            "kg_count": len(kg),
            "success": True
        }
        return result_info, proteins, kg, fragment_stats
    except Exception as e:
        logger.error(f"处理文件失败: {file_path} - {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "file_path": file_path}, {}, {}, None


def process_file_batch_parallel(file_list, output_dir, max_length=50, step_size=25, build_kg=True, n_workers=None,
                                k_neighbors=8, distance_threshold=8.0, plddt_threshold=70.0, respect_ss=True, format_type="pyg"):
    """并行处理多个文件，每1000个文件保存一次结果"""
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    logger.info(f"使用 {n_workers} 个CPU核心并行处理...")
    logger.info(f"处理参数: 最大长度={max_length}, 步长={step_size}, "
                f"k近邻={k_neighbors}, 距离阈值={distance_threshold}埃, "
                f"pLDDT阈值={plddt_threshold}, 遵循二级结构={respect_ss}")

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
    chunk_size = 1000

    # 初始化统计数据键
    stats["processed_files"] = 0
    stats["extracted_chains"] = 0
    stats["knowledge_graphs"] = 0
    stats["failed_files"] = 0
    stats["total_fragments"] = 0

    # 初始化文件计数器和批次ID
    file_counter = 0
    batch_id = 1

    with open(sequences_log_path, 'w', buffering=1) as s_log:
        s_log.write("fragment_id,protein_id,chain_id,length,sequence\n")

    with open(fragments_log_path, 'w') as f_log:
        f_log.write("file_name,protein_id,chain_id,valid_residues,fragments\n")

    with open(processing_log_path, 'w') as p_log:
        p_log.write("timestamp,file_path,status,elapsed,proteins,knowledge_graphs,error\n")

    # 使用进程池进行并行处理
    with tqdm(total=len(file_list), desc="处理PDB/CIF文件") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    process_file_parallel, file_path, max_length, step_size, build_kg,
                    k_neighbors, distance_threshold, plddt_threshold, respect_ss
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
                stats["extracted_chains"] += len(proteins)
                stats["knowledge_graphs"] += len(kg)
                stats["total_fragments"] += len(proteins)

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
                    for chain_id, chain_stats in fragment_stats["chains"].items():
                        with open(fragments_log_path, 'a') as f_log:
                            f_log.write(f"{fragment_stats['file_name']},{fragment_stats['protein_id']},"
                                        f"{chain_id},{chain_stats['valid_residues']},{chain_stats['fragments']}\n")

                # 更新序列日志
                with open(sequences_log_path, 'a', buffering=1) as s_log:
                    for fragment_id, data in proteins.items():
                        s_log.write(
                            f"{fragment_id},{data['protein_id']},{data['chain_id']},{len(data['sequence'])},{data['sequence']}\n")

                # 累积数据
                all_proteins.update(proteins)
                all_knowledge_graphs.update(kg)

                # 每处理指定数量文件保存一次中间结果
                if file_counter % chunk_size == 0:
                    logger.info(f"已处理 {file_counter}/{len(file_list)} 个文件，保存当前批次结果...")

                    batch_output_dir = os.path.join(all_data_dir, f"batch_{batch_id}")
                    os.makedirs(batch_output_dir, exist_ok=True)

                    # 保存蛋白质数据
                    save_results_chunked(
                        all_proteins, batch_output_dir,
                        base_name="protein_data",
                        chunk_size=chunk_size
                    )

                    # 保存知识图谱
                    if build_kg and all_knowledge_graphs:
                        save_knowledge_graphs(
                            all_knowledge_graphs, batch_output_dir,
                            base_name="protein_kg",
                            chunk_size=chunk_size
                        )

                    # 重置集合并增加批次ID
                    all_proteins = {}
                    all_knowledge_graphs = {}
                    batch_id += 1

                    # 更新统计信息
                    logger.info(f"当前处理统计: 成功={stats['processed_files']}, 失败={stats['failed_files']}, "
                                f"片段={stats['total_fragments']}")

    # 保存最后一批数据
    if all_proteins:
        logger.info(f"保存最后一批结果 (剩余 {len(all_proteins)} 个蛋白质片段)...")

        final_batch_dir = os.path.join(all_data_dir, f"batch_{batch_id}")
        os.makedirs(final_batch_dir, exist_ok=True)

        # 保存蛋白质数据
        save_results_chunked(
            all_proteins, final_batch_dir,
            base_name="protein_data",
            chunk_size=chunk_size
        )

        # 保存知识图谱
        if build_kg and all_knowledge_graphs:
            save_knowledge_graphs(
                all_knowledge_graphs, final_batch_dir,
                base_name="protein_kg",
                chunk_size=chunk_size,
                format_type=format_type
            )

    logger.info(f"片段统计数据已写入: {fragments_log_path}")
    logger.info(f"处理日志已写入: {processing_log_path}")
    logger.info(f"创建的片段总数: {stats['total_fragments']}")

    return stats, all_data_dir


def save_results_chunked(all_proteins, output_dir, base_name="kg", chunk_size=1000):
    """分块保存蛋白质序列结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 将蛋白质数据分块
    protein_ids = list(all_proteins.keys())
    chunks = [protein_ids[i:i + chunk_size] for i in range(0, len(protein_ids), chunk_size)]

    output_files = []
    for i, chunk_ids in enumerate(chunks):
        chunk_data = {pid: all_proteins[pid] for pid in chunk_ids}
        output_file = os.path.join(output_dir, f"{base_name}_chunk_{i + 1}.json")

        with open(output_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)

        output_files.append(output_file)
        logger.info(f"保存数据块 {i + 1}/{len(chunks)}: {output_file} ({len(chunk_ids)} 个蛋白质)")

    # 保存元数据
    metadata = {
        "total_proteins": len(all_proteins),
        "chunk_count": len(chunks),
        "chunk_files": output_files,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_files, metadata


def save_knowledge_graphs(kg_data, output_dir, base_name="protein_kg", chunk_size=1000, format_type="pyg"):
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
                        # 获取NetworkX图
                        nx_data = kg_data[pid]
                        # 如果是字典格式，转换为NetworkX图
                        if isinstance(nx_data, dict):
                            nx_graph = nx.node_link_graph(nx_data)
                        else:
                            nx_graph = nx_data

                        # 转换为PyG格式
                        x = []  # 节点特征
                        edge_index = [[], []]  # 边索引
                        edge_attr = []  # 边特性

                        # 创建节点映射
                        node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}

                        # 提取节点特征
                        for node in nx_graph.nodes():
                            node_attrs = nx_graph.nodes[node]
                            # 将节点属性转换为特征向量
                            features = [
                                AA_PROPERTIES.get(node_attrs.get('residue_code', 'X'), {}).get('hydropathy', 0),
                                AA_PROPERTIES.get(node_attrs.get('residue_code', 'X'), {}).get('charge', 0),
                                node_attrs.get('plddt', 50) / 100.0,  # 归一化pLDDT
                            ]
                            x.append(features)

                        # 提取边
                        for src, tgt, edge_data in nx_graph.edges(data=True):
                            edge_index[0].append(node_mapping[src])
                            edge_index[1].append(node_mapping[tgt])

                            # 边属性
                            edge_type = 1 if edge_data.get('edge_type') == 'peptide' else 0
                            distance = edge_data.get('distance', 0)
                            edge_attr.append([edge_type, distance])

                        # 创建PyG数据对象
                        if x and edge_index[0]:  # 确保有节点和边
                            data = Data(
                                x=torch.tensor(x, dtype=torch.float),
                                edge_index=torch.tensor(edge_index, dtype=torch.long),
                                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                                protein_id=pid
                            )
                            graphs_data[pid] = data

                            # 更新统计信息
                            index["total_nodes"] += len(x)
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


def extract_sequence_from_graph(graph):
    """从知识图谱中提取蛋白质序列

    参数:
        graph: 知识图谱数据（可以是节点链接格式或PyG格式）

    返回:
        str: 提取的氨基酸序列，如果无法提取则返回None
    """
    try:
        # 处理节点链接格式
        if isinstance(graph, dict) and 'nodes' in graph:
            # 从节点中获取残基信息
            nodes = graph['nodes']

            # 检查节点是否包含残基信息
            if not nodes or not isinstance(nodes[0], dict):
                return None

            # 提取残基和位置信息
            residue_info = []
            for node in nodes:
                if 'residue_code' in node and 'residue_num' in node:
                    aa_code = node['residue_code']
                    res_num = node.get('residue_num', 0)
                    residue_info.append((res_num, aa_code))

            # 按残基编号排序
            residue_info.sort(key=lambda x: x[0])

            # 检查序列是否连续
            sequence = ''.join([aa for _, aa in residue_info])
            return sequence

        # 处理PyG格式
        elif hasattr(graph, 'x') and hasattr(graph, 'edge_index'):
            # 获取节点属性
            try:
                # 检查是否有节点特征数据
                if not hasattr(graph, 'residue_code'):
                    # 尝试从附加属性中获取残基信息
                    residue_info = []

                    # 如果图有存储残基代码和编号的属性
                    if hasattr(graph, 'residue_codes') and hasattr(graph, 'residue_nums'):
                        codes = graph.residue_codes
                        nums = graph.residue_nums

                        # 转换为CPU张量并转为Python列表
                        if hasattr(codes, 'cpu'):
                            codes = codes.cpu().tolist()
                        if hasattr(nums, 'cpu'):
                            nums = nums.cpu().tolist()

                        for i in range(len(codes)):
                            residue_info.append((nums[i], codes[i]))
                    else:
                        # 如果没有直接的残基信息，检查其他可能的属性
                        return None

                    # 按残基编号排序
                    residue_info.sort(key=lambda x: x[0])
                    sequence = ''.join([aa for _, aa in residue_info])
                    return sequence
                else:
                    # 直接使用残基代码属性
                    codes = graph.residue_code
                    if hasattr(codes, 'cpu'):
                        codes = codes.cpu().tolist()
                    return ''.join(codes)

            except Exception as e:
                # 处理PyG数据提取错误
                return None

        # 其他格式的图谱
        else:
            # 尝试从图的自定义属性中提取
            if hasattr(graph, 'sequence'):
                return graph.sequence
            elif hasattr(graph, 'protein_sequence'):
                return graph.protein_sequence

    except Exception as e:
        # 提取过程中出现任何错误，返回None
        return None

    # 如果无法提取序列，返回None
    return None


# ======================= 主程序和命令行接口 =======================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从PDB/CIF文件中提取蛋白质结构数据并构建知识图谱")
    parser.add_argument("input", help="输入PDB/CIF文件或包含这些文件的目录")
    parser.add_argument("--output_dir", "-o", default="./kg",
                        help="输出目录 (默认: ./kg)")
    parser.add_argument("--max_length", "-m", type=int, default=50,
                        help="最大序列长度 (默认: 50)")
    parser.add_argument("--step_size", "-s", type=int, default=25,
                        help="滑动窗口步长 (默认: 25)")
    parser.add_argument("--chunk_size", "-c", type=int, default=1000,
                        help="输出分块大小 (默认: 1000)")
    parser.add_argument("--build_kg", "-k", action="store_true", default=True,
                        help="构建知识图谱 (默认: True)")
    parser.add_argument("--workers", "-w", type=int,
                        help="并行工作进程数 (默认: CPU核心数-1)")
    parser.add_argument("--k_neighbors", type=int, default=8,
                        help="空间邻接的K近邻数 (默认: 8)")
    parser.add_argument("--distance_threshold", type=float, default=8.0,
                        help="空间邻接距离阈值 (默认: 8.0埃)")
    parser.add_argument("--plddt_threshold", type=float, default=70.0,
                        help="AlphaFold pLDDT质量得分阈值 (默认: 70.0)")
    parser.add_argument("--respect_ss", action="store_true", default=True,
                        help="是否尊重二级结构边界进行片段划分")
    parser.add_argument("--pyg_format", action="store_true", default='pyg',
                        help="保存为什么格式 (默认: pyg，可选: json)")
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
        max_length=args.max_length,
        step_size=args.step_size,
        build_kg=args.build_kg,
        n_workers=args.workers,
        k_neighbors=args.k_neighbors,
        distance_threshold=args.distance_threshold,
        plddt_threshold=args.plddt_threshold,
        respect_ss=args.respect_ss,
        format_type=args.pyg_format
    )

    # 处理结果统计
    logger.info("\n处理完成:")
    logger.info(f"- 处理的文件总数: {stats['processed_files']}")
    logger.info(f"- 提取的蛋白质片段总数: {stats['extracted_chains']}")
    logger.info(f"- 生成的知识图谱总数: {stats['knowledge_graphs']}")
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
            "extracted_fragments": stats['extracted_chains'],
            "knowledge_graphs": stats['knowledge_graphs'],
            "parameters": {
                "max_length": args.max_length,
                "step_size": args.step_size,
                "k_neighbors": args.k_neighbors,
                "distance_threshold": args.distance_threshold,
                "plddt_threshold": args.plddt_threshold,
                "respect_ss": args.respect_ss
            },
            "output_dir": os.path.abspath(args.output_dir),
            "all_data_dir": os.path.abspath(all_data_dir)
        }
        json.dump(summary, f, indent=2)

    logger.info(f"摘要信息已保存到: {summary_file}")
    logger.info("蛋白质结构提取流程完成！")


if __name__ == "__main__":
    main()
