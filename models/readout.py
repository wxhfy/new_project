#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图神经网络的读出机制实现

该模块提供了将节点级嵌入转换为图级表示的读出机制，专为抗菌肽(AMPs)设计任务优化。
包括注意力读出、多层次池化等方法，能够有效捕获关键残基与整体结构特征。

作者: wxhfy
日期: 2025-03-29
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_scatter import scatter_add, scatter_mean, scatter_max


class AttentiveReadout(nn.Module):
    """
    注意力加权的图读出机制

    使用自注意力机制对节点表示进行加权聚合，能够识别并强调抗菌肽中的关键残基，
    特别适合捕获决定抗菌活性的重要结构特征。

    参数:
        in_features (int): 输入特征维度
        hidden_dim (int, optional): 注意力机制的隐藏维度
        num_heads (int, optional): 注意力头数量
        dropout (float, optional): Dropout率
        gating (bool, optional): 是否使用门控机制
    """

    def __init__(
            self,
            in_features,
            hidden_dim=None,
            num_heads=4,
            dropout=0.1,
            gating=True
    ):
        super(AttentiveReadout, self).__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim or in_features
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads
        self.gating = gating

        # 注意力查询向量 - 可学习的图级上下文
        self.query = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.query)

        # 节点特征投影
        self.key_proj = nn.Linear(in_features, self.hidden_dim)
        self.value_proj = nn.Linear(in_features, self.hidden_dim)

        # 输出投影
        self.output_proj = nn.Linear(self.hidden_dim, in_features)

        # 门控机制
        if gating:
            self.gate_net = nn.Sequential(
                nn.Linear(in_features * 2, in_features),
                nn.Sigmoid()
            )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x, batch):
        """
        使用注意力机制聚合节点特征为图级表示

        参数:
            x (torch.Tensor): 节点特征 [num_nodes, in_features]
            batch (torch.LongTensor): 批处理索引 [num_nodes]

        返回:
            graph_embedding (torch.Tensor): 图级嵌入 [batch_size, in_features]
        """
        batch_size = batch.max().item() + 1
        num_nodes = x.size(0)

        # 计算键和值
        keys = self.key_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        values = self.value_proj(x).view(num_nodes, self.num_heads, self.head_dim)

        # 扩展查询以匹配批次中的每个图
        query = self.query.expand(batch_size, -1, -1)  # [batch_size, num_heads, head_dim]

        # 准备注意力计算
        # 将节点按批次分组
        node_indices = torch.arange(num_nodes, device=x.device)
        graph_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=x.device)
        for b in range(1, batch_size + 1):
            graph_offsets[b] = (batch < b).sum()

        # 计算每个图的节点数量
        nodes_per_graph = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        for b in range(batch_size):
            nodes_per_graph[b] = ((batch == b).sum())

        # 初始化图嵌入
        graph_embedding = torch.zeros(batch_size, self.in_features, device=x.device)

        # 对每个图分别计算注意力权重和加权和
        for b in range(batch_size):
            start_idx = graph_offsets[b].item()
            end_idx = graph_offsets[b + 1].item()

            if start_idx == end_idx:  # 空图跳过
                continue

            # 提取当前图的节点特征
            graph_keys = keys[start_idx:end_idx]  # [nodes_in_graph, num_heads, head_dim]
            graph_values = values[start_idx:end_idx]  # [nodes_in_graph, num_heads, head_dim]

            # 计算注意力分数
            attn_scores = torch.bmm(
                graph_keys.transpose(0, 1),  # [num_heads, nodes_in_graph, head_dim]
                query[b].unsqueeze(-1)  # [num_heads, head_dim, 1]
            ).squeeze(-1)  # [num_heads, nodes_in_graph]

            # 归一化注意力权重
            attn_weights = F.softmax(attn_scores, dim=1)  # [num_heads, nodes_in_graph]
            attn_weights = self.dropout(attn_weights)

            # 加权求和
            weighted_values = torch.bmm(
                attn_weights.unsqueeze(1),  # [num_heads, 1, nodes_in_graph]
                graph_values.transpose(0, 1)  # [num_heads, nodes_in_graph, head_dim]
            ).squeeze(1)  # [num_heads, head_dim]

            # 合并多头注意力
            graph_heads = weighted_values.view(1, -1)  # [1, num_heads * head_dim]
            graph_context = self.output_proj(graph_heads)  # [1, in_features]

            # 补充创建平均池化表示，用于门控机制
            if self.gating:
                graph_avg = global_mean_pool(
                    x[start_idx:end_idx],
                    torch.zeros(end_idx - start_idx, dtype=torch.long, device=x.device)
                )
                gate = self.gate_net(torch.cat([graph_context, graph_avg], dim=1))
                graph_context = gate * graph_context + (1 - gate) * graph_avg

            # 保存结果
            graph_embedding[b] = graph_context

        # 应用层归一化
        graph_embedding = self.layer_norm(graph_embedding)

        return graph_embedding


class MultiLevelPooling(nn.Module):
    """
    多层次图表示聚合模块

    结合多种池化策略，捕获蛋白质结构的不同级别特征，
    对于理解抗菌肽的全局结构特性与局部活性中心特别有效。

    参数:
        in_features (int): 输入特征维度
        hidden_dim (int, optional): 隐藏层维度
        dropout (float, optional): Dropout率
        use_gate (bool, optional): 是否使用门控融合机制
    """

    def __init__(
            self,
            in_features,
            hidden_dim=None,
            dropout=0.1,
            use_gate=True
    ):
        super(MultiLevelPooling, self).__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim or in_features
        self.use_gate = use_gate

        # 池化特征变换
        self.mean_transform = nn.Linear(in_features, self.hidden_dim)
        self.max_transform = nn.Linear(in_features, self.hidden_dim)
        self.sum_transform = nn.Linear(in_features, self.hidden_dim)

        # 门控融合机制
        if use_gate:
            self.gate_mean = nn.Sequential(
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
            self.gate_max = nn.Sequential(
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
            self.gate_sum = nn.Sequential(
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )

        # 输出投影
        self.output_proj = nn.Linear(self.hidden_dim, in_features)

        # Dropout和层归一化
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x, batch):
        """
        使用多种池化方法聚合节点特征为图级表示

        参数:
            x (torch.Tensor): 节点特征 [num_nodes, in_features]
            batch (torch.LongTensor): 批处理索引 [num_nodes]

        返回:
            graph_embedding (torch.Tensor): 图级嵌入 [batch_size, in_features]
        """
        # 应用不同池化方法
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)

        # 特征变换
        mean_repr = self.mean_transform(mean_pool)
        max_repr = self.max_transform(max_pool)
        sum_repr = self.sum_transform(sum_pool)

        # 特征融合
        if self.use_gate:
            # 计算门控权重
            gate_mean = self.gate_mean(mean_repr)
            gate_max = self.gate_max(max_repr)
            gate_sum = self.gate_sum(sum_repr)

            # 归一化门控权重
            gates = torch.cat([gate_mean, gate_max, gate_sum], dim=1)
            gates = F.softmax(gates, dim=1)

            # 加权融合
            pooled = (gates[:, 0:1] * mean_repr +
                      gates[:, 1:2] * max_repr +
                      gates[:, 2:3] * sum_repr)
        else:
            # 简单平均
            pooled = (mean_repr + max_repr + sum_repr) / 3.0

        # 输出投影
        graph_embedding = self.output_proj(pooled)
        graph_embedding = self.dropout(graph_embedding)
        graph_embedding = self.layer_norm(graph_embedding)

        return graph_embedding


class HierarchicalReadout(nn.Module):
    """
    层次化读出机制

    结合注意力读出和多层次池化，并引入特征聚合方法，
    特别适合捕获抗菌肽中的局部活性中心和全局结构特征。

    参数:
        in_features (int): 输入特征维度
        hidden_dim (int, optional): 隐藏层维度
        num_heads (int, optional): 注意力头数量
        dropout (float, optional): Dropout率
    """

    def __init__(
            self,
            in_features,
            hidden_dim=None,
            num_heads=4,
            dropout=0.1
    ):
        super(HierarchicalReadout, self).__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim or in_features

        # 注意力读出
        self.attn_readout = AttentiveReadout(
            in_features,
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            gating=True
        )

        # 多层次池化
        self.multi_pool = MultiLevelPooling(
            in_features,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
            use_gate=True
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(in_features * 2, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features)
        )

    def forward(self, x, batch):
        """
        使用层次化读出机制聚合节点特征为图级表示

        参数:
            x (torch.Tensor): 节点特征 [num_nodes, in_features]
            batch (torch.LongTensor): 批处理索引 [num_nodes]

        返回:
            graph_embedding (torch.Tensor): 图级嵌入 [batch_size, in_features]
        """
        # 注意力读出
        attn_emb = self.attn_readout(x, batch)

        # 多层次池化
        pool_emb = self.multi_pool(x, batch)

        # 特征融合
        combined = torch.cat([attn_emb, pool_emb], dim=1)
        graph_embedding = self.fusion(combined)

        return graph_embedding


class FocalReadout(nn.Module):
    """
    焦点读出机制

    针对抗菌肽(AMPs)的关键功能区域设计的读出机制，
    通过识别并聚焦于关键残基（如荷电残基、疏水残基）来构建更具功能导向的图表示。

    参数:
        in_features (int): 输入特征维度
        hidden_dim (int, optional): 隐藏层维度
        dropout (float, optional): Dropout率
    """

    def __init__(
            self,
            in_features,
            hidden_dim=None,
            dropout=0.1
    ):
        super(FocalReadout, self).__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim or in_features

        # 残基重要性评分网络
        self.importance_net = nn.Sequential(
            nn.Linear(in_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Linear(in_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, in_features)
        )

    def forward(self, x, batch, node_attr=None):
        """
        基于残基重要性聚合节点特征为图级表示

        参数:
            x (torch.Tensor): 节点特征 [num_nodes, in_features]
            batch (torch.LongTensor): 批处理索引 [num_nodes]
            node_attr (torch.Tensor, optional): 额外节点属性 [num_nodes, attr_dim]

        返回:
            graph_embedding (torch.Tensor): 图级嵌入 [batch_size, in_features]
        """
        # 计算残基重要性分数
        importance = self.importance_net(x).squeeze(-1)  # [num_nodes]

        # 对每个图内的分数进行归一化
        batch_size = batch.max().item() + 1
        normalized_importance = torch.zeros_like(importance)

        for i in range(batch_size):
            batch_mask = (batch == i)
            batch_imp = importance[batch_mask]

            # 使用softmax归一化
            batch_norm_imp = F.softmax(batch_imp, dim=0)
            normalized_importance[batch_mask] = batch_norm_imp

        # 加权特征聚合
        transformed_features = self.feature_transform(x)
        weighted_features = transformed_features * normalized_importance.unsqueeze(-1)

        # 按图聚合
        graph_embedding = scatter_add(weighted_features, batch, dim=0, dim_size=batch_size)

        return graph_embedding