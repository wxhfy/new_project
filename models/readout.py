#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图级别嵌入池化模块

为蛋白质知识图谱提供多种图池化策略，
包括注意力池化、分层池化、差分池化等。

作者: 基于wxhfy的知识图谱处理工作扩展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class AttentiveReadout(nn.Module):
    """
    注意力加权的图级别池化

    学习节点的重要性权重，进行加权求和，得到图级别表示
    """

    def __init__(self, hidden_dim):
        """
        参数:
            hidden_dim: 节点特征维度
        """
        super(AttentiveReadout, self).__init__()

        # 注意力网络，计算每个节点的重要性分数
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x, batch):
        """
        前向传播

        参数:
            x: 节点特征 [num_nodes, hidden_dim]
            batch: 批处理索引 [num_nodes]

        返回:
            graph_embedding: 图级别嵌入 [batch_size, hidden_dim]
        """
        # 计算注意力得分
        attention_scores = self.attention(x)  # [num_nodes, 1]

        # 进行softmax归一化，分别对每个图中的节点进行
        max_scores = torch.zeros_like(attention_scores)
        max_scores.scatter_(0, batch.unsqueeze(-1).expand(-1, attention_scores.size(-1)),
                            attention_scores, reduce='max')
        exp_scores = torch.exp(attention_scores - max_scores[batch])

        # 为每个图计算指数和
        graph_sizes = torch.zeros(batch.max().item() + 1, dtype=torch.float, device=x.device)
        graph_sizes.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))

        # 计算每个节点的注意力权重
        sum_exp_scores = torch.zeros_like(exp_scores)
        sum_exp_scores.scatter_add_(0, batch.unsqueeze(-1).expand(-1, exp_scores.size(-1)),
                                    exp_scores)
        normalized_scores = exp_scores / sum_exp_scores[batch]

        # 加权求和得到图嵌入
        graph_embedding = torch.zeros(batch.max().item() + 1, x.size(-1),
                                      dtype=torch.float, device=x.device)
        graph_embedding.scatter_add_(0, batch.unsqueeze(-1).expand(-1, x.size(-1)),
                                     x * normalized_scores)

        return graph_embedding


class MultiReadout(nn.Module):
    """
    多种池化方法融合的图级别表示

    结合均值池化、最大池化和注意力池化，捕获更丰富的图级别信息
    """

    def __init__(self, hidden_dim, use_attention=True):
        """
        参数:
            hidden_dim: 节点特征维度
            use_attention: 是否使用注意力池化
        """
        super(MultiReadout, self).__init__()

        self.use_attention = use_attention
        if use_attention:
            self.attentive_pool = AttentiveReadout(hidden_dim)

        # 融合不同池化结果的权重
        self.pool_weights = nn.Parameter(torch.ones(3 if use_attention else 2))
        self.pool_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, batch):
        """
        前向传播

        参数:
            x: 节点特征 [num_nodes, hidden_dim]
            batch: 批处理索引 [num_nodes]

        返回:
            graph_embedding: 图级别嵌入 [batch_size, hidden_dim]
        """
        # 均值池化
        mean_pool = global_mean_pool(x, batch)

        # 最大池化
        max_pool = global_max_pool(x, batch)

        # 总和池化
        sum_pool = global_add_pool(x, batch)

        # 注意力池化（可选）
        if self.use_attention:
            att_pool = self.attentive_pool(x, batch)

            # 对各种池化结果进行加权平均
            weights = F.softmax(self.pool_weights, dim=0)
            graph_embedding = (weights[0] * mean_pool +
                               weights[1] * max_pool +
                               weights[2] * att_pool)
        else:
            # 不使用注意力池化的加权平均
            weights = F.softmax(self.pool_weights, dim=0)
            graph_embedding = weights[0] * mean_pool + weights[1] * max_pool

        # 归一化
        return self.pool_norm(graph_embedding)