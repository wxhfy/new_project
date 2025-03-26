#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自定义图注意力层和辅助模块

包含GATv2注意力机制实现和多层感知机等组件，
针对蛋白质图谱特性进行了优化。

作者: 基于wxhfy的知识图谱处理工作扩展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2ConvLayer(nn.Module):
    """
    改进的GATv2卷积层，为蛋白质图谱特别优化

    特点：
    1. 支持边特征融合到注意力计算
    2. 边类型感知注意力
    3. 氨基酸相对位置编码（可选）
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=4,
                 dropout=0.1,
                 edge_dim=None,
                 use_bias=True,
                 use_edge_features=True):
        """
        参数:
            in_channels: 输入特征维度
            out_channels: 每个头的输出特征维度
            heads: 注意力头数量
            dropout: Dropout概率
            edge_dim: 边特征维度，None表示不使用边特征
            use_bias: 是否使用偏置
            use_edge_features: 是否在注意力计算中使用边特征
        """
        super(GATv2ConvLayer, self).__init__()

        # 设置使用边特征的标志
        self.use_edge_features = use_edge_features and edge_dim is not None

        # GATv2卷积层
        self.conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim if self.use_edge_features else None,
            add_self_loops=True,  # 添加自环以捕获节点自身信息
            bias=use_bias
        )

        # 用于肽键(peptide)和空间连接(spatial)的边类型感知注意力
        if self.use_edge_features:
            # 边类型编码 - 处理不同类型的边（肽键和空间边）
            self.edge_type_encoder = nn.Embedding(2, edge_dim)  # 假设只有两种边类型

            # 边特征增强模块
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Linear(edge_dim, edge_dim)
            )

        def forward(self, x, edge_index, edge_attr=None):
            """
            前向传播

            参数:
                x: 节点特征 [num_nodes, in_channels]
                edge_index: 边索引 [2, num_edges]
                edge_attr: 边特征 [num_edges, edge_dim]

            返回:
                updated_features: 更新后的节点特征 [num_nodes, heads * out_channels]
            """
            if self.use_edge_features and edge_attr is not None:
                # 提取边类型信息并编码（这里假设edge_attr的第一列表示边类型）
                edge_types = edge_attr[:, 0].long()
                edge_type_embedding = self.edge_type_encoder(edge_types)

                # 融合边类型和边特征
                edge_features = edge_attr[:, 1:].float() if edge_attr.size(1) > 1 else torch.zeros_like(
                    edge_type_embedding)
                enhanced_edge_attr = edge_features + edge_type_embedding

                # 边特征增强
                enhanced_edge_attr = self.edge_encoder(enhanced_edge_attr)

                # 使用增强的边特征进行消息传递
                return self.conv(x, edge_index, edge_attr=enhanced_edge_attr)
            else:
                # 不使用边特征的标准消息传递
                return self.conv(x, edge_index)

class MLPLayer(nn.Module):
    """多层感知机层，用于特征转换"""

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 dropout=0.1,
                 activation="relu",
                 use_layer_norm=True):
        """
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出特征维度
            dropout: Dropout概率
            activation: 激活函数类型
            use_layer_norm: 是否使用层归一化
        """
        super(MLPLayer, self).__init__()

        # 激活函数选择
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        # MLP网络
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

        # 层归一化（可选）
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征 [batch_size, in_channels]

        返回:
            transformed_features: 转换后的特征 [batch_size, out_channels]
        """
        x = self.mlp(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        return x

class ResidualConnection(nn.Module):
    """残差连接模块"""

    def __init__(self, dim, dropout=0.0):
        """
        参数:
            dim: 特征维度
            dropout: Dropout概率
        """
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        应用残差连接

        参数:
            x: 输入特征
            sublayer: 子层函数

        返回:
            residual_output: 残差连接输出
        """
        return x + self.dropout(sublayer(self.norm(x)))