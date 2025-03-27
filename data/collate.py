#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图嵌入系统 - 数据批处理函数

用于将多个蛋白质图结构批处理为批次数据，
支持节点特征、边特征和图特征的正确处理和批次索引构建。

作者: wxhfy
"""

import torch
from torch_geometric.data import Batch, Data
from typing import Dict, List, Optional, Union
import numpy as np


def protein_collate_fn(data_list: List[Data]) -> Batch:
    """
    将多个蛋白质图数据对象合并成一个批次

    参数:
        data_list: 蛋白质图数据对象列表

    返回:
        Batch: 合并后的批次数据
    """
    # 使用PyG的Batch类合并数据
    batch = Batch.from_data_list(data_list)

    # 添加蛋白质特定的批处理属性
    protein_ids = []
    chain_ids = []
    fragment_ids = []
    sequences = []

    # 收集蛋白质元数据
    for i, data in enumerate(data_list):
        # 获取蛋白质ID
        protein_id = getattr(data, 'protein_id', f'protein_{i}')
        protein_ids.extend([protein_id] * data.num_nodes)

        # 获取链ID
        chain_id = getattr(data, 'chain_id', 'X')
        chain_ids.extend([chain_id] * data.num_nodes)

        # 获取片段ID
        fragment_id = getattr(data, 'fragment_id', f'frag_{i}')
        fragment_ids.extend([fragment_id] * data.num_nodes)

        # 获取序列
        sequence = getattr(data, 'sequence', '')
        sequences.append(sequence)

    # 存储元数据
    if protein_ids:
        batch.protein_ids = protein_ids

    if chain_ids:
        batch.chain_ids = chain_ids

    if fragment_ids:
        batch.fragment_ids = fragment_ids

    if sequences:
        batch.sequences = sequences

    # 如果有节点级标签，确保它们被正确批处理
    if hasattr(data_list[0], 'y') and data_list[0].y is not None:
        if data_list[0].y.dim() == 0 or data_list[0].y.size(0) == 1:
            # 图级标签
            batch.y = torch.cat([d.y.view(1) for d in data_list], dim=0)
        else:
            # 节点级标签
            batch.y = torch.cat([d.y for d in data_list], dim=0)

    # 添加批次大小属性
    batch.num_graphs = len(data_list)

    return batch


def weighted_protein_collate_fn(data_list: List[Data], weight_by_size: bool = True) -> Batch:
    """
    将多个蛋白质图数据对象合并成一个批次，并添加权重信息

    参数:
        data_list: 蛋白质图数据对象列表
        weight_by_size: 是否根据图大小计算样本权重

    返回:
        Batch: 合并后的批次数据
    """
    # 标准合并
    batch = protein_collate_fn(data_list)

    # 添加样本权重
    if weight_by_size:
        # 根据图大小计算权重
        graph_sizes = [data.num_nodes for data in data_list]
        total_nodes = sum(graph_sizes)
        weights = torch.tensor([total_nodes / (len(data_list) * size) for size in graph_sizes],
                               dtype=torch.float)
        batch.weights = weights
    else:
        # 使用均等权重
        batch.weights = torch.ones(len(data_list), dtype=torch.float)

    return batch