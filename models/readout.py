#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图嵌入系统 - 图读出层

实现图级别的嵌入聚合机制，将节点级特征整合为图级表示，
专为蛋白质图谱的结构特点设计。

作者: 基于wxhfy的知识图谱处理工作扩展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_batch


class GraphReadout(nn.Module):
    """
    图级别嵌入聚合模块

    特点:
    1. 支持多种图池化策略（平均、最大值、和、注意力）
    2. 针对蛋白质图谱结构优化
    3. 可选的自注意力聚合机制
    """

    def __init__(self, in_channels, out_channels=None, readout_type='multihead',
                 num_heads=4, dropout=0.1):
        """
        参数:
            in_channels (int): 输入特征维度
            out_channels (int, optional): 输出特征维度，None则与输入维度相同
            readout_type (str): 聚合类型，可选['mean', 'max', 'add', 'attention', 'multihead']
            num_heads (int): 多头注意力的头数 (仅对attention和multihead有效)
            dropout (float): Dropout率
        """
        super(GraphReadout, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.readout_type = readout_type

        # 基本池化函数
        self.global_mean_pool = global_mean_pool
        self.global_max_pool = global_max_pool
        self.global_add_pool = global_add_pool

        # 多种池化方式的融合层
        if readout_type == 'multihead':
            self.attention = nn.Sequential(
                nn.Linear(in_channels * 3, in_channels),
                nn.LayerNorm(in_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(in_channels, num_heads)
            )

        # 注意力池化层
        elif readout_type == 'attention':
            self.attention_layer = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2),
                nn.LayerNorm(in_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(in_channels // 2, 1)
            )

        # 输出映射层
        if self.out_channels != self.in_channels:
            if readout_type == 'multihead':
                self.out_proj = nn.Linear(in_channels, out_channels)
            else:
                self.out_proj = nn.Linear(in_channels, out_channels)
        else:
            self.out_proj = nn.Identity()

    def forward(self, x, batch, mask=None):
        """
        前向传播

        参数:
            x (Tensor): 节点特征 [num_nodes, in_channels]
            batch (Tensor): 节点对应的批次索引
            mask (Tensor, optional): 节点掩码，用于过滤部分节点

        返回:
            Tensor: 图级嵌入向量 [batch_size, out_channels]
        """
        if self.readout_type == 'mean':
            # 平均池化
            return self.out_proj(self.global_mean_pool(x, batch))

        elif self.readout_type == 'max':
            # 最大值池化
            return self.out_proj(self.global_max_pool(x, batch))

        elif self.readout_type == 'add':
            # 求和池化
            return self.out_proj(self.global_add_pool(x, batch))

        elif self.readout_type == 'attention':
            # 注意力池化
            x_dense, mask = to_dense_batch(x, batch)

            # 计算注意力得分
            scores = self.attention_layer(x_dense).squeeze(-1)
            scores = scores.masked_fill(~mask, float('-inf'))
            attention_weights = F.softmax(scores, dim=1)

            # 应用注意力权重
            out = torch.bmm(attention_weights.unsqueeze(1), x_dense).squeeze(1)
            return self.out_proj(out)

        elif self.readout_type == 'multihead':
            # 多头池化 (结合平均、最大值和求和)
            x_mean = self.global_mean_pool(x, batch)
            x_max = self.global_max_pool(x, batch)
            x_add = self.global_add_pool(x, batch)

            # 拼接不同池化结果
            x_cat = torch.cat([x_mean, x_max, x_add], dim=1)

            # 计算不同池化方式的权重
            attention_weights = self.attention(x_cat)
            attention_weights = F.softmax(attention_weights, dim=1)

            # 加权融合
            x_stacked = torch.stack([x_mean, x_max, x_add], dim=1)
            out = torch.bmm(attention_weights.unsqueeze(1), x_stacked).squeeze(1)

            return self.out_proj(out)

        else:
            raise ValueError(f"不支持的聚合类型: {self.readout_type}")