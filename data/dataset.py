#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图神经网络 - 数据集模块

提供用于加载和处理蛋白质图数据的PyTorch数据集类，支持多种任务类型和数据格式。

作者: wxhfy
"""

import os
import json
import logging
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional, Callable
import pickle


class ProteinGraphDataset(Dataset):
    """
    蛋白质图数据集

    支持从预处理的知识图谱文件加载蛋白质结构数据，
    可用于节点级别、边级别和图级别的预测任务。
    """

    def __init__(
            self,
            data_dir: str,
            task_type: str = 'graph',  # 'node', 'edge', 'graph'
            split: str = 'train',  # 'train', 'val', 'test'
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            in_memory: bool = True,
            max_samples: Optional[int] = None,
            node_features: List[str] = None,
            use_cache: bool = True,
            cache_dir: Optional[str] = None,
            label_file: Optional[str] = None,
            format_type: str = 'pyg'  # 'pyg' or 'json'
    ):
        """
        参数:
            data_dir: 包含蛋白质图数据的目录
            task_type: 任务类型 ('node', 'edge', 'graph')
            split: 数据集划分 ('train', 'val', 'test')
            transform: 在获取样本时应用的变换
            pre_transform: 在数据加载时应用的预处理变换
            in_memory: 是否将所有数据加载到内存中
            max_samples: 最大样本数量限制 (用于调试)
            node_features: 要使用的节点特征列表
            use_cache: 是否使用缓存加速加载
            cache_dir: 缓存目录，若不指定则使用data_dir/cache
            label_file: 标签文件路径 (JSON格式)
            format_type: 数据格式类型 ('pyg' 或 'json')
        """
        self.data_dir = data_dir
        self.task_type = task_type
        self.split = split
        self.transform = transform
        self.pre_transform = pre_transform
        self.in_memory = in_memory
        self.max_samples = max_samples
        self.format_type = format_type

        # 设置节点特征
        self.node_features = node_features or ['hydropathy', 'charge', 'polar', 'molecular_weight']

        # 设置缓存目录
        self.use_cache = use_cache
        self.cache_dir = cache_dir or os.path.join(data_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 加载标签（如果提供）
        self.labels = {}
        if label_file and os.path.exists(label_file):
            with open(label_file, 'r') as f:
                self.labels = json.load(f)

        # 加载索引文件
        kg_dir = os.path.join(data_dir, f"knowledge_graphs_{format_type}")
        index_files = [f for f in os.listdir(kg_dir) if f.endswith('_index.json')]

        if not index_files:
            raise ValueError(f"在 {kg_dir} 中未找到索引文件")

        with open(os.path.join(kg_dir, index_files[0]), 'r') as f:
            self.index = json.load(f)

        # 确定数据文件
        self.data_files = [
            os.path.join(kg_dir, f)
            for f in os.listdir(kg_dir)
            if (f.endswith('.json') or f.endswith('.pt')) and not f.endswith('_index.json')
        ]

        # 加载或创建样本索引
        self.protein_ids = []
        cache_path = os.path.join(self.cache_dir, f"{self.split}_ids.pkl")

        if self.use_cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.protein_ids = pickle.load(f)
                self.logger.info(f"从缓存加载了 {len(self.protein_ids)} 个蛋白质ID")
        else:
            self._build_protein_index()

            if self.use_cache:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.protein_ids, f)

        # 限制样本数量 (用于调试)
        if self.max_samples is not None:
            self.protein_ids = self.protein_ids[:self.max_samples]

        # 如果完全加载到内存，预加载所有数据
        self.data_cache = {}
        if self.in_memory:
            self.logger.info("预加载所有数据到内存...")
            self._preload_data()

    def _build_protein_index(self):
        """构建蛋白质ID索引"""
        self.logger.info("构建蛋白质索引...")

        # 从所有数据文件中收集蛋白质ID
        all_ids = []
        for file_path in tqdm(self.data_files, desc="读取数据文件"):
            if self.format_type == 'pyg' and file_path.endswith('.pt'):
                # PyG格式
                data_dict = torch.load(file_path)
                all_ids.extend(list(data_dict.keys()))
            elif self.format_type == 'json' and file_path.endswith('.json'):
                # JSON格式
                with open(file_path, 'r') as f:
                    data_dict = json.load(f)
                all_ids.extend(list(data_dict.keys()))

        # 基于split进行划分
        random.seed(42)  # 设置随机种子确保可重复性
        random.shuffle(all_ids)

        if self.split == 'train':
            # 70%用于训练
            self.protein_ids = all_ids[:int(0.7 * len(all_ids))]
        elif self.split == 'val':
            # 20%用于验证
            self.protein_ids = all_ids[int(0.7 * len(all_ids)):int(0.9 * len(all_ids))]
        else:  # test
            # 10%用于测试
            self.protein_ids = all_ids[int(0.9 * len(all_ids)):]

        self.logger.info(f"{self.split}集合包含 {len(self.protein_ids)} 个蛋白质")

    def _preload_data(self):
        """预加载所有数据到内存"""
        # 为每个数据文件创建ID到文件的映射
        id_to_file = {}
        for file_path in self.data_files:
            if self.format_type == 'pyg' and file_path.endswith('.pt'):
                data_dict = torch.load(file_path)
                for pid in data_dict.keys():
                    if pid in self.protein_ids:
                        id_to_file[pid] = file_path
            elif self.format_type == 'json' and file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data_dict = json.load(f)
                for pid in data_dict.keys():
                    if pid in self.protein_ids:
                        id_to_file[pid] = file_path

        # 加载所需的数据
        for pid in tqdm(self.protein_ids, desc="预加载数据"):
            if pid in id_to_file:
                file_path = id_to_file[pid]
                if self.format_type == 'pyg':
                    data_dict = torch.load(file_path)
                    graph_data = data_dict[pid]
                    # 可选地应用预处理
                    if self.pre_transform is not None:
                        graph_data = self.pre_transform(graph_data)
                    self.data_cache[pid] = graph_data
                else:  # json
                    with open(file_path, 'r') as f:
                        data_dict = json.load(f)
                    graph_data = self._json_to_pyg(data_dict[pid])
                    # 可选地应用预处理
                    if self.pre_transform is not None:
                        graph_data = self.pre_transform(graph_data)
                    self.data_cache[pid] = graph_data

    def _json_to_pyg(self, json_graph):
        """将JSON格式的图转换为PyG格式"""
        # 提取节点特征和属性
        x = []  # 节点特征列表
        node_map = {}  # 从JSON节点ID到索引的映射

        # 处理节点
        for i, node in enumerate(json_graph.get('nodes', [])):
            # 保存节点映射
            node_map[node.get('id', str(i))] = i

            # 提取节点特征
            features = []
            for feat in self.node_features:
                if feat in node:
                    value = node[feat]
                    # 转换布尔值为浮点数
                    if isinstance(value, bool):
                        value = float(value)
                    features.append(value)
                else:
                    features.append(0.0)  # 默认值

            x.append(features)

        # 处理边
        edge_index = [[], []]  # 源节点和目标节点列表
        edge_attr = []  # 边属性列表

        for edge in json_graph.get('links', []):
            source = edge.get('source')
            target = edge.get('target')

            # 确保源节点和目标节点存在于映射中
            if source in node_map and target in node_map:
                # 添加边
                edge_index[0].append(node_map[source])
                edge_index[1].append(node_map[target])

                # 边属性
                weight = edge.get('weight', 1.0)
                edge_type = 1.0 if edge.get('edge_type') == 'peptide' else 0.0
                distance = edge.get('distance', 0.0)

                edge_attr.append([edge_type, weight, distance])

        # 创建PyG数据对象
        return Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            num_nodes=len(x)
        )

    def _find_file_for_id(self, protein_id):
        """查找包含指定蛋白质ID的文件"""
        for file_path in self.data_files:
            try:
                if file_path.endswith('.pt'):
                    # PyTorch格式
                    data_dict = torch.load(file_path)
                    if protein_id in data_dict:
                        return file_path, data_dict
                elif file_path.endswith('.json'):
                    # JSON格式
                    with open(file_path, 'r') as f:
                        data_dict = json.load(f)
                    if protein_id in data_dict:
                        return file_path, data_dict
            except Exception as e:
                self.logger.warning(f"读取文件 {file_path} 时出错: {str(e)}")
        return None, None

    def __len__(self):
        """返回数据集大小"""
        return len(self.protein_ids)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        protein_id = self.protein_ids[idx]

        # 检查缓存
        if self.in_memory and protein_id in self.data_cache:
            graph_data = self.data_cache[protein_id]
        else:
            # 查找并加载数据
            file_path, data_dict = self._find_file_for_id(protein_id)

            if file_path is None:
                raise ValueError(f"未找到蛋白质ID: {protein_id}")

            if self.format_type == 'pyg':
                graph_data = data_dict[protein_id]
            else:  # json
                graph_data = self._json_to_pyg(data_dict[protein_id])

            # 应用预处理
            if self.pre_transform is not None:
                graph_data = self.pre_transform(graph_data)

        # 添加蛋白质ID到数据中
        graph_data.protein_id = protein_id

        # 添加标签（如果可用）
        if protein_id in self.labels:
            if self.task_type == 'graph':
                graph_data.y = torch.tensor(self.labels[protein_id], dtype=torch.float)
            elif self.task_type == 'node':
                # 假设标签是每个节点的列表
                graph_data.y = torch.tensor(self.labels[protein_id], dtype=torch.float)
            elif self.task_type == 'edge':
                # 假设标签是每条边的列表
                graph_data.edge_y = torch.tensor(self.labels[protein_id], dtype=torch.float)

        # 应用变换
        if self.transform is not None:
            graph_data = self.transform(graph_data)

        return graph_data


class ProteinGraphDataModule:
    """
    蛋白质图数据模块

    用于管理蛋白质图数据的加载、预处理和批处理
    """

    def __init__(
            self,
            data_dir: str,
            batch_size: int = 32,
            num_workers: int = 4,
            task_type: str = 'graph',
            node_features: List[str] = None,
            in_memory: bool = True,
            use_cache: bool = True,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            label_file: Optional[str] = None,
            format_type: str = 'pyg'
    ):
        """
        参数:
            data_dir: 数据目录
            batch_size: 批处理大小
            num_workers: 数据加载工作进程数
            task_type: 任务类型 ('node', 'edge', 'graph')
            node_features: 要使用的节点特征列表
            in_memory: 是否将所有数据加载到内存
            use_cache: 是否使用缓存
            transform: 数据变换
            pre_transform: 数据预处理
            label_file: 标签文件路径 (JSON格式)
            format_type: 数据格式类型 ('pyg' 或 'json')
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_type = task_type
        self.in_memory = in_memory
        self.use_cache = use_cache
        self.transform = transform
        self.pre_transform = pre_transform
        self.label_file = label_file
        self.format_type = format_type
        self.node_features = node_features

        # 设置数据集
        self._setup()

    def _setup(self):
        """设置数据集"""
        self.train_dataset = ProteinGraphDataset(
            data_dir=self.data_dir,
            task_type=self.task_type,
            split='train',
            transform=self.transform,
            pre_transform=self.pre_transform,
            in_memory=self.in_memory,
            node_features=self.node_features,
            use_cache=self.use_cache,
            label_file=self.label_file,
            format_type=self.format_type
        )

        self.val_dataset = ProteinGraphDataset(
            data_dir=self.data_dir,
            task_type=self.task_type,
            split='val',
            transform=self.transform,
            pre_transform=self.pre_transform,
            in_memory=self.in_memory,
            node_features=self.node_features,
            use_cache=self.use_cache,
            label_file=self.label_file,
            format_type=self.format_type
        )

        self.test_dataset = ProteinGraphDataset(
            data_dir=self.data_dir,
            task_type=self.task_type,
            split='test',
            transform=self.transform,
            pre_transform=self.pre_transform,
            in_memory=self.in_memory,
            node_features=self.node_features,
            use_cache=self.use_cache,
            label_file=self.label_file,
            format_type=self.format_type
        )

    def train_dataloader(self):
        """返回训练数据加载器"""
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """返回验证数据加载器"""
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        """返回测试数据加载器"""
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class ProteinGraphTransform:
    """蛋白质图数据增强和变换"""

    @staticmethod
    def drop_nodes(data, drop_rate=0.1):
        """随机丢弃节点"""
        if drop_rate <= 0 or drop_rate >= 1:
            return data

        num_nodes = data.num_nodes
        num_drop = int(num_nodes * drop_rate)

        if num_drop == 0:
            return data

        # 随机选择要保留的节点
        keep_idx = torch.randperm(num_nodes)[:num_nodes - num_drop]
        keep_idx, _ = torch.sort(keep_idx)

        # 创建新的边索引和节点特征
        row, col = data.edge_index
        mask = (row.view(-1, 1) == keep_idx.view(1, -1)).any(dim=1)
        mask = mask & (col.view(-1, 1) == keep_idx.view(1, -1)).any(dim=1)

        # 更新数据
        edge_index = data.edge_index[:, mask]
        edge_attr = data.edge_attr[mask] if hasattr(data, 'edge_attr') else None

        # 重新映射节点索引
        node_idx = torch.full((num_nodes,), -1, dtype=torch.long)
        node_idx[keep_idx] = torch.arange(keep_idx.size(0))
        edge_index = node_idx[edge_index]

        # 创建新的数据对象
        new_data = data.clone()
        new_data.x = data.x[keep_idx]
        new_data.edge_index = edge_index
        if edge_attr is not None:
            new_data.edge_attr = edge_attr

        return new_data


    @staticmethod
    def perturb_features(data, noise_scale=0.1):
        """给节点特征添加高斯噪声"""
        if noise_scale <= 0:
            return data

        new_data = data.clone()
        noise = torch.randn_like(data.x) * noise_scale
        new_data.x = data.x + noise
        return new_data


    @staticmethod
    def mask_features(data, mask_rate=0.15):
        """随机掩盖部分节点特征（用于自监督学习）"""
        if mask_rate <= 0 or mask_rate >= 1:
            return data

        new_data = data.clone()
        num_nodes = data.num_nodes
        num_features = data.x.size(1)

        # 为每个节点随机选择要掩盖的特征
        mask = torch.rand(num_nodes, num_features) < mask_rate

        # 保存原始特征用于计算自监督损失
        new_data.original_x = data.x.clone()

        # 将被掩盖的特征设为0
        new_data.x = data.x.clone()
        new_data.x[mask] = 0.0

        # 保存掩码位置
        new_data.feature_mask = mask

        return new_data


    @staticmethod
    def drop_edges(data, drop_rate=0.1):
        """随机丢弃边"""
        if drop_rate <= 0 or drop_rate >= 1:
            return data

        num_edges = data.edge_index.size(1)
        num_drop = int(num_edges * drop_rate)

        if num_drop == 0:
            return data

        # 随机选择要保留的边
        keep_indices = torch.randperm(num_edges)[:num_edges - num_drop]
        keep_indices, _ = torch.sort(keep_indices)

        new_data = data.clone()
        new_data.edge_index = data.edge_index[:, keep_indices]

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            new_data.edge_attr = data.edge_attr[keep_indices]

        return new_data


    @staticmethod
    def subgraph(data, center_node_idx=None, max_size=20):
        """提取以某个节点为中心的子图"""
        if center_node_idx is None:
            # 随机选择中心节点
            center_node_idx = torch.randint(0, data.num_nodes, (1,)).item()

        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # BFS搜索找到周围节点
        visited = set([center_node_idx])
        queue = [center_node_idx]
        front = 0

        while front < len(queue) and len(visited) < max_size:
            node = queue[front]
            front += 1

            # 找出与当前节点相连的所有节点
            neighbors = edge_index[1, edge_index[0] == node].tolist()
            for neighbor in neighbors:
                if neighbor not in visited and len(visited) < max_size:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # 转换为列表并排序
        nodes = sorted(list(visited))
        node_tensor = torch.tensor(nodes, dtype=torch.long)

        # 创建子图
        sub_data = data.subgraph(node_tensor)

        return sub_data

