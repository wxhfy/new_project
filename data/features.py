#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图嵌入系统 - 特征提取模块

从蛋白质结构数据中提取节点特征和边特征，用于图神经网络模型训练。
支持多种氨基酸特性编码和边类型特征化。

作者: wxhfy
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

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
    'Y': {'hydropathy': -1.3, 'charge': 0, 'polar': True, 'mw': 181.19},
    'X': {'hydropathy': 0.0, 'charge': 0, 'polar': False, 'mw': 0.0}  # 未知氨基酸
}

# 二级结构映射
SS_MAPPING = {
    'H': 0,  # Alpha helix
    'B': 1,  # Beta bridge
    'E': 2,  # Extended strand
    'G': 3,  # 3-10 helix
    'I': 4,  # Pi helix
    'T': 5,  # Turn
    'S': 6,  # Bend
    'C': 7,  # Coil
    'X': 8  # 未知
}


def extract_node_features(node_data: Dict, feature_types: List[str] = None) -> torch.Tensor:
    """
    从节点数据中提取特征向量

    参数:
        node_data: 包含节点属性的字典
        feature_types: 要提取的特征类型列表，可选:
            - 'one_hot': 氨基酸类型的one-hot编码
            - 'position': 空间坐标特征
            - 'plddt': AlphaFold预测置信度
            - 'physicochemical': 物理化学特性
            - 'ss': 二级结构编码
            - 'all': 所有可用特征

    返回:
        torch.Tensor: 节点特征向量
    """
    if feature_types is None:
        feature_types = ['one_hot', 'position', 'physicochemical', 'ss']
    elif 'all' in feature_types:
        feature_types = ['one_hot', 'position', 'plddt', 'physicochemical', 'ss']

    features = []

    # 获取氨基酸代码
    aa_code = node_data.get('residue_code', 'X')

    # One-hot编码氨基酸类型
    if 'one_hot' in feature_types:
        amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
        one_hot = [1.0 if aa == aa_code else 0.0 for aa in amino_acids]
        features.extend(one_hot)

    # 空间位置编码
    if 'position' in feature_types:
        position = node_data.get('position', [0.0, 0.0, 0.0])
        features.extend(position)

    # pLDDT质量分数
    if 'plddt' in feature_types:
        plddt = node_data.get('plddt', 70.0) / 100.0  # 归一化到0-1
        features.append(plddt)

    # 物理化学特性
    if 'physicochemical' in feature_types:
        props = AA_PROPERTIES.get(aa_code, AA_PROPERTIES['X'])
        features.append(props['hydropathy'] / 5.0)  # 归一化
        features.append(props['charge'] / 2.0)  # 归一化
        features.append(1.0 if props['polar'] else 0.0)
        features.append(props['mw'] / 210.0)  # 归一化

    # 二级结构编码
    if 'ss' in feature_types:
        ss = node_data.get('secondary_structure', 'X')
        ss_idx = SS_MAPPING.get(ss, SS_MAPPING['X'])
        ss_one_hot = [1.0 if i == ss_idx else 0.0 for i in range(len(SS_MAPPING))]
        features.extend(ss_one_hot)

    return torch.tensor(features, dtype=torch.float)


def extract_edge_features(edge_data: Dict, feature_types: List[str] = None) -> torch.Tensor:
    """
    从边数据中提取特征向量

    参数:
        edge_data: 包含边属性的字典
        feature_types: 要提取的特征类型列表，可选:
            - 'distance': 空间距离
            - 'edge_type': 边类型编码
            - 'weight': 边权重
            - 'all': 所有可用特征

    返回:
        torch.Tensor: 边特征向量
    """
    if feature_types is None:
        feature_types = ['distance', 'edge_type']
    elif 'all' in feature_types:
        feature_types = ['distance', 'edge_type', 'weight']

    features = []

    # 空间距离
    if 'distance' in feature_types:
        distance = edge_data.get('distance', 10.0)
        features.append(distance / 20.0)  # 归一化到0-1范围

    # 边类型编码
    if 'edge_type' in feature_types:
        # 边类型映射: 0=肽键, 1=空间邻接, 2=二级结构内部, 3=其他
        edge_type_map = {'peptide': 0, 'spatial': 1, 'ss_internal': 2, 'other': 3}
        edge_type = edge_data.get('edge_type', 'other')
        type_idx = edge_type_map.get(edge_type, 3)
        edge_type_one_hot = [1.0 if i == type_idx else 0.0 for i in range(len(edge_type_map))]
        features.extend(edge_type_one_hot)

    # 边权重
    if 'weight' in feature_types:
        weight = edge_data.get('weight', 1.0)
        features.append(weight)

    return torch.tensor(features, dtype=torch.float)