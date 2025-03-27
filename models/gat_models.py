#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图嵌入系统 - GATv2模型

基于GATv2的蛋白质知识图谱嵌入模型，专为蛋白质图谱的结构特点设计，
能够有效提取节点和边的信息，并生成高质量的嵌入向量。

作者: 基于wxhfy的知识图谱处理工作扩展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from .layers import GATv2Conv, SelfAttention
from .readout import GraphReadout


class ProteinGATv2(nn.Module):
    """
    蛋白质知识图谱的GATv2嵌入模型

    特点:
    1. 多层GATv2结构，针对节点级别特征提取
    2. 高级聚合层，捕获全局图级别信息
    3. 适应蛋白质知识图谱的特殊结构和特性
    4. 支持节点级和图级嵌入输出
    5. 残差连接与层归一化，提高训练稳定性
    """

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers=3,
                 heads=4,
                 dropout=0.1,
                 edge_dim=None,
                 add_self_loops=True,
                 readout_type='multihead',
                 jk_mode='cat',
                 use_layer_norm=True,
                 node_level=True,
                 graph_level=True,
                 use_edge_attr=True):
        """
        参数:
            in_channels (int): 输入节点特征维度
            hidden_channels (int): 隐藏层维度
            out_channels (int): 输出特征维度
            num_layers (int): GAT层数
            heads (int): 每层的注意力头数
            dropout (float): Dropout率
            edge_dim (int, optional): 边特征维度，None表示不使用边特征
            add_self_loops (bool): 是否添加自环
            readout_type (str): 图读出类型
            jk_mode (str): 跳跃连接模式，可选['cat', 'lstm', 'max', 'last']
            use_layer_norm (bool): 是否使用层归一化
            node_level (bool): 是否保留节点级嵌入
            graph_level (bool): 是否输出图级嵌入
            use_edge_attr (bool): 是否使用边属性
        """
        super(ProteinGATv2, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.jk_mode = jk_mode
        self.use_layer_norm = use_layer_norm
        self.node_level = node_level
        self.graph_level = graph_level
        self.use_edge_attr = use_edge_attr and edge_dim is not None

        # 输入投影层
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GATv2层
        self.convs = nn.ModuleList()
        # 第一层：in_channels -> hidden_channels
        self.convs.append(
            GATv2Conv(
                hidden_channels,
                hidden_channels // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim if self.use_edge_attr else None,
                add_residual=True
            )
        )

        # 中间层：hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim if self.use_edge_attr else None,
                    add_residual=True
                )
            )

        # 最后一层：hidden_channels -> out_channels (如果不使用跳跃连接)
        if jk_mode == 'last':
            last_dim = out_channels
        else:
            last_dim = hidden_channels

        self.convs.append(
            GATv2Conv(
                hidden_channels,
                last_dim // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim if self.use_edge_attr else None,
                add_residual=True
            )
        )

        # 层归一化
        if use_layer_norm:
            self.layer_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.layer_norms.append(nn.LayerNorm(hidden_channels))
        else:
            self.layer_norms = None

        # 跳跃连接处理
        if jk_mode == 'cat':
            self.jk_proj = nn.Linear(hidden_channels * num_layers, out_channels)
        elif jk_mode == 'lstm':
            self.jk_lstm = nn.LSTM(
                hidden_channels,
                out_channels // 2,
                bidirectional=True,
                batch_first=True
            )
        elif jk_mode == 'max':
            self.jk_proj = nn.Linear(hidden_channels, out_channels)
        elif jk_mode == 'last':
            self.jk_proj = None  # 最后一层已经输出正确维度

        # 图读出层
        if graph_level:
            final_dim = out_channels
            self.readout = GraphReadout(
                in_channels=final_dim,
                out_channels=out_channels,
                readout_type=readout_type,
                dropout=dropout
            )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        """重置模型参数"""
        self.input_proj.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.layer_norms is not None:
            for ln in self.layer_norms:
                ln.reset_parameters()
        if self.jk_proj is not None:
            self.jk_proj.reset_parameters()
        if self.jk_mode == 'lstm':
            for param in self.jk_lstm.parameters():
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
        if self.graph_level:
            self.readout.reset_parameters() if hasattr(self.readout, 'reset_parameters') else None

    def forward(self, data):
        """
        前向传播

        参数:
            data: PyG数据对象，包含:
                - x: 节点特征
                - edge_index: 边索引
                - edge_attr: 边特征 (可选)
                - batch: 批次索引

        返回:
            dict: 包含节点级和/或图级嵌入向量的字典
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 如果无批次信息，默认一个图
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 输入投影
        x = self.input_proj(x)

        # 存储每层的输出，用于跳跃连接
        layer_outputs = []

        # GAT层传播
        for i, conv in enumerate(self.convs):
            # 应用GAT卷积
            x = conv(x, edge_index,
                     edge_attr=edge_attr if self.use_edge_attr else None)

            # 最后一层特殊处理
            if i < self.num_layers - 1 or self.jk_mode != 'last':
                # 层归一化
                if self.use_layer_norm:
                    x = self.layer_norms[i](x)
                # 激活函数 & Dropout
                x = F.relu(x)
                x = self.dropout(x)

            # 存储层输出
            layer_outputs.append(x)

        # 跳跃连接处理
        if self.jk_mode == 'cat':
            # 拼接所有层的输出
            x = torch.cat(layer_outputs, dim=1)
            x = self.jk_proj(x)
        elif self.jk_mode == 'lstm':
            # 将每层输出作为序列送入LSTM
            node_count = x.size(0)
            x_stack = torch.stack(layer_outputs, dim=1)  # [num_nodes, num_layers, hidden_dim]
            x, _ = self.jk_lstm(x_stack)
            x = x[:, -1, :]  # 取最后一个时间步
        elif self.jk_mode == 'max':
            # 取每个特征维度的最大值
            x = torch.stack(layer_outputs, dim=0)
            x, _ = torch.max(x, dim=0)
            x = self.jk_proj(x)
        # 对于'last'模式，直接使用最后一层的输出

        # 准备输出
        result = {}

        # 节点级嵌入
        if self.node_level:
            result['node_embedding'] = x

        # 图级嵌入
        if self.graph_level:
            graph_embedding = self.readout(x, batch)
            result['graph_embedding'] = graph_embedding

        return result


class ProteinGATv2WithPretraining(nn.Module):
    """
    带有预训练任务的蛋白质GATv2模型

    支持:
    1. 节点属性预测（如二级结构预测）
    2. 边关系预测
    3. 图级别分类/回归
    4. 蛋白质功能预测
    """

    def __init__(self, gat_model, task_type='node', num_tasks=1, hidden_dim=None):
        """
        参数:
            gat_model (ProteinGATv2): 基础GAT模型
            task_type (str): 任务类型 ('node', 'edge', 'graph')
            num_tasks (int): 任务数量
            hidden_dim (int, optional): 任务特定隐藏层维度
        """
        super(ProteinGATv2WithPretraining, self).__init__()

        self.gat_model = gat_model
        self.task_type = task_type
        self.num_tasks = num_tasks

        out_dim = gat_model.out_channels
        hidden_dim = hidden_dim or out_dim

        # 根据任务类型创建预测头
        if task_type == 'node':
            self.predictor = nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_tasks)
            )
        elif task_type == 'edge':
            self.predictor = nn.Sequential(
                nn.Linear(out_dim * 2, hidden_dim),  # 拼接源节点和目标节点的嵌入
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_tasks)
            )
        elif task_type == 'graph':
            self.predictor = nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_tasks)
            )

    def forward(self, data):
        """
        前向传播

        参数:
            data: PyG数据对象

        返回:
            dict: 包含嵌入和预测结果的字典
        """
        # 获取GAT模型的输出
        embeddings = self.gat_model(data)

        # 根据任务类型进行预测
        if self.task_type == 'node':
            node_emb = embeddings['node_embedding']
            pred = self.predictor(node_emb)
            embeddings['node_pred'] = pred

        elif self.task_type == 'edge':
            node_emb = embeddings['node_embedding']
            edge_index = data.edge_index
            # 获取边的源节点和目标节点的嵌入
            src_emb = node_emb[edge_index[0]]
            dst_emb = node_emb[edge_index[1]]
            # 拼接源节点和目标节点的嵌入
            edge_emb = torch.cat([src_emb, dst_emb], dim=1)
            pred = self.predictor(edge_emb)
            embeddings['edge_pred'] = pred

        elif self.task_type == 'graph':
            graph_emb = embeddings['graph_embedding']
            pred = self.predictor(graph_emb)
            embeddings['graph_pred'] = pred

        return embeddings

