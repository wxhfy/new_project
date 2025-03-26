#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质知识图谱GATv2嵌入模型

基于GATv2实现，针对蛋白质短链的结构特性进行优化，
用于提取蛋白质知识图谱中丰富的节点特征和结构信息。

作者: 基于wxhfy的知识图谱处理工作扩展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from .layers import GATv2ConvLayer, MLPLayer
from .readout import AttentiveReadout


class ProteinGATv2Encoder(nn.Module):
    """
    蛋白质知识图谱的GATv2编码器，针对蛋白质短链特性设计

    特点：
    1. 多层次GATv2卷积，捕获不同尺度的结构信息
    2. 残差连接增强梯度流动
    3. 层归一化提高训练稳定性
    4. 边类型感知的注意力
    5. 针对蛋白质特征特别设计的特征变换
    """
    def __init__(self,
                 in_channels,
                 hidden_channels=128,
                 out_channels=128,
                 num_layers=3,
                 heads=4,
                 dropout=0.1,
                 edge_dim=None,
                 use_layer_norm=True,
                 use_residual=True,
                 activation="relu",
                 readout_mode="attentive"):
        """
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出嵌入维度
            num_layers: GATv2层数
            heads: 注意力头数
            dropout: Dropout概率
            edge_dim: 边特征维度，None表示不使用边特征
            use_layer_norm: 是否使用层归一化
            use_residual: 是否使用残差连接
            activation: 激活函数类型 ('relu', 'gelu', 'silu')
            readout_mode: 图级别特征池化方式 ('attentive', 'mean', 'max', 'sum')
        """
        super(ProteinGATv2Encoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.readout_mode = readout_mode

        # 激活函数选择
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        # 输入特征转换层 - 将各种蛋白质特征映射到统一的隐藏维度
        self.feature_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # GATv2层堆叠
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        # 第一层卷积
        self.convs.append(
            GATv2ConvLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels // heads,  # 多头注意力，每个头的输出维度
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout
            )
        )

        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

        # 中间层卷积
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2ConvLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout
                )
            )
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_channels))

        # 最后一层卷积
        self.convs.append(
            GATv2ConvLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels // heads,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout
            )
        )
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

        # 输出投影层 - 最终嵌入维度调整
        self.output_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

        # 图级别的Readout函数
        if readout_mode == "attentive":
            self.readout = AttentiveReadout(hidden_channels)
        elif readout_mode == "mean":
            self.readout = global_mean_pool
        elif readout_mode == "max":
            self.readout = global_max_pool
        elif readout_mode == "sum":
            self.readout = global_add_pool
        else:
            raise ValueError(f"不支持的池化方式: {readout_mode}")

        # 蛋白质特异性的特征增强模块（可选）
        self.feature_enhancer = MLPLayer(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels*2,
            out_channels=hidden_channels,
            dropout=dropout,
            activation=activation
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        前向传播

        参数:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim]
            batch: 批处理索引 [num_nodes]

        返回:
            node_embeddings: 节点级别嵌入 [num_nodes, out_channels]
            graph_embedding: 图级别嵌入 [batch_size, out_channels]
        """
        # 特征编码
        x = self.feature_encoder(x)

        # 特征增强 - 蛋白质特异性处理
        x = self.feature_enhancer(x) + x  # 残差连接

        # 多层GATv2传递
        for i, conv in enumerate(self.convs):
            # 消息传递
            x_new = conv(x, edge_index, edge_attr)

            # 残差连接
            if self.use_residual and i > 0:  # 第一层没有残差
                x_new = x_new + x

            # 更新特征
            x = x_new

            # 层归一化
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            # 激活函数（最后一层除外）
            if i < self.num_layers - 1:
                x = self.act(x)

        # 输出投影
        node_embeddings = self.output_encoder(x)

        # 图级别嵌入 - 如果提供了batch信息
        graph_embedding = None
        if batch is not None:
            if self.readout_mode == "attentive":
                graph_embedding = self.readout(node_embeddings, batch)
            else:
                graph_embedding = self.readout(node_embeddings, batch)

        return node_embeddings, graph_embedding


class ProteinGATv2Model(nn.Module):
    """
    完整的蛋白质知识图谱嵌入模型，包含编码器和任务相关的头部
    """
    def __init__(self,
                 in_channels,
                 hidden_channels=128,
                 out_channels=128,
                 num_layers=3,
                 heads=4,
                 dropout=0.1,
                 edge_dim=None,
                 use_layer_norm=True,
                 use_residual=True,
                 activation="relu",
                 readout_mode="attentive",
                 pooling_ratio=0.5,
                 task_type=None,
                 num_tasks=1):
        """
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出嵌入维度
            num_layers: GATv2层数
            heads: 注意力头数
            dropout: Dropout概率
            edge_dim: 边特征维度，None表示不使用边特征
            use_layer_norm: 是否使用层归一化
            use_residual: 是否使用残差连接
            activation: 激活函数类型
            readout_mode: 图级别特征池化方式
            pooling_ratio: 层次池化比例（若使用层次池化）
            task_type: 任务类型，可以是'pretraining'(预训练),'classification'(分类),'regression'(回归)或None
            num_tasks: 任务数量（分类或回归任务的数量）
        """
        super(ProteinGATv2Model, self).__init__()

        # GATv2编码器
        self.encoder = ProteinGATv2Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            activation=activation,
            readout_mode=readout_mode
        )

        # 任务类型
        self.task_type = task_type

        # 根据任务类型添加不同的头部
        if task_type == 'pretraining':
            # 对比学习预训练头
            self.pretraining_head = nn.Sequential(
                nn.Linear(out_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels)
            )
        elif task_type == 'classification':
            # 分类任务头
            self.classifier = nn.Sequential(
                nn.Linear(out_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_tasks)
            )
        elif task_type == 'regression':
            # 回归任务头
            self.regressor = nn.Sequential(
                nn.Linear(out_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_tasks)
            )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        前向传播

        参数:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim]
            batch: 批处理索引 [num_nodes]

        返回:
            根据任务类型不同返回不同的结果:
            - None: (node_embeddings, graph_embedding)
            - 'pretraining': projection_embeddings
            - 'classification'/'regression': task_outputs
        """
        # 获取节点和图嵌入
        node_embeddings, graph_embedding = self.encoder(x, edge_index, edge_attr, batch)

        # 根据任务类型返回不同输出
        if self.task_type is None:
            return node_embeddings, graph_embedding

        elif self.task_type == 'pretraining':
            projection = self.pretraining_head(graph_embedding)
            return F.normalize(projection, p=2, dim=1)  # L2正则化

        elif self.task_type == 'classification':
            logits = self.classifier(graph_embedding)
            return logits

        elif self.task_type == 'regression':
            predictions = self.regressor(graph_embedding)
            return predictions

        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")