#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图嵌入系统 - 图注意力层实现

包含GATv2图注意力层的实现，专为蛋白质知识图谱优化，
支持节点间关系（肽键和空间邻近）的特殊处理。

作者: 基于wxhfy的知识图谱处理工作扩展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn.inits import glorot, zeros


class GATv2Conv(MessagePassing):
    """
    GATv2注意力层实现，针对蛋白质知识图谱优化

    特点:
    1. 支持边特征融合，针对肽键/空间邻近关系进行差异化处理
    2. 基于GATv2机制，解决了原始GAT中的静态注意力问题
    3. 增强了节点间相互作用的表达能力
    """

    def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2,
                 dropout=0.0, edge_dim=None, bias=True, share_weights=False,
                 add_residual=True):
        """
        参数:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            heads (int, optional): 注意力头数量
            negative_slope (float, optional): LeakyReLU负斜率
            dropout (float, optional): Dropout比率
            edge_dim (int, optional): 边特征维度，None表示不使用边特征
            bias (bool, optional): 是否使用偏置
            share_weights (bool, optional): 是否在源节点和目标节点间共享变换权重
            add_residual (bool, optional): 是否添加残差连接
        """
        super(GATv2Conv, self).__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.share_weights = share_weights
        self.add_residual = add_residual

        # 定义线性变换 (针对源节点和目标节点)
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        if self.share_weights:
            self.lin_dst = self.lin_src
        else:
            self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)

        # 注意力机制
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # 边特征处理 (如果有)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化模型参数"""
        glorot(self.lin_src.weight)
        if not self.share_weights:
            glorot(self.lin_dst.weight)
        glorot(self.att)
        if self.lin_edge is not None:
            glorot(self.lin_edge.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        """
        前向传播

        参数:
            x (Tensor): 节点特征矩阵 [num_nodes, in_channels]
            edge_index (LongTensor): 边索引 [2, num_edges]
            edge_attr (Tensor, optional): 边特征 [num_edges, edge_dim]
            return_attention_weights (bool, optional): 是否返回注意力权重

        返回:
            tuple: (输出特征, 注意力权重) 或 输出特征
        """
        # 处理输入
        if isinstance(x, tuple):
            x_src, x_dst = x  # 如果输入是元组，分别处理源节点和目标节点
        else:
            x_src = x_dst = x

        H, C = self.heads, self.out_channels

        # 线性变换
        x_src = self.lin_src(x_src).view(-1, H, C)  # [num_nodes, heads, channels]
        x_dst = self.lin_dst(x_dst).view(-1, H, C)

        # 传播过程
        out, attention_weights = self._propagate(
            edge_index, x=x, x_src=x_src, x_dst=x_dst, edge_attr=edge_attr,
            size=None, return_attention_weights=return_attention_weights
        )

        # 应用残差连接 (如果启用)
        if self.add_residual and x_src.size(-2) == out.size(-2):
            out = out + x_src

        # 重塑输出
        out = out.view(-1, self.heads * self.out_channels)

        # 添加偏置
        if self.bias is not None:
            out = out + self.bias

        # 返回结果
        if return_attention_weights:
            return out, attention_weights
        else:
            return out

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        """
        定义消息传递函数

        参数:
            x_j: 源节点特征
            x_i: 目标节点特征
            edge_attr: 边特征
            index: 边的目标节点索引
            ptr: 稀疏表示的指针
            size_i: 目标节点数量

        返回:
            更新后的消息
        """
        # 计算注意力分数 (GATv2方式): a(Wh_i || Wh_j)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)

        # 合并源节点和目标节点信息
        x = x_i + x_j

        # 如果有边特征，融合进来
        if edge_attr is not None and self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        # 计算注意力分数
        alpha = (x * self.att).sum(dim=-1)

        # 应用LeakyReLU激活函数
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # 归一化
        alpha = softmax(alpha, index, ptr, size_i)

        # Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 应用注意力权重
        return x_j * alpha.unsqueeze(-1)

    def _propagate(self, edge_index, x, x_src, x_dst, edge_attr=None, size=None, return_attention_weights=False):
        """
        自定义传播函数

        参数:
            edge_index: 边索引
            x: 原始节点特征
            x_src: 源节点转换特征
            x_dst: 目标节点转换特征
            edge_attr: 边特征
            size: 图大小
            return_attention_weights: 是否返回注意力权重

        返回:
            (输出特征, 注意力权重)
        """
        # 准备边索引
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.view(2, -1)

        # 获取源节点和目标节点索引
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # 获取源节点和目标节点特征
        x_j = x_src[src_idx]  # 源节点特征
        x_i = x_dst[dst_idx]  # 目标节点特征

        # 准备边特征 (如果有)
        if edge_attr is not None:
            assert edge_attr.size(0) == edge_index.size(1)

        # 计算注意力分数 (GATv2方式): a(Wh_i || Wh_j)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)

        # 合并源节点和目标节点信息
        x_combined = x_i + x_j

        # 如果有边特征，融合进来
        if edge_attr is not None and self.lin_edge is not None:
            edge_embedding = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            x_combined = x_combined + edge_embedding

        # 计算注意力分数
        alpha = (x_combined * self.att).sum(dim=-1)

        # 应用LeakyReLU激活函数
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # 归一化
        if isinstance(edge_index, torch.Tensor):
            alpha = softmax(alpha, dst_idx, num_nodes=x_dst.size(0))
        else:
            # 对于稀疏表示的特殊处理
            alpha = softmax(alpha, dst_idx, ptr=None, num_nodes=x_dst.size(0))

        # Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 计算消息
        out = x_j * alpha.unsqueeze(-1)

        # 聚合消息
        out = scatter_add(out, dst_idx, dim=0, dim_size=x_dst.size(0))

        attention_weights = None
        if return_attention_weights:
            attention_weights = (edge_index, alpha)

        return out, attention_weights


class SelfAttention(nn.Module):
    """
    节点内部特征的自注意力机制，增强不同通道间的交互
    """

    def __init__(self, in_features, hidden_dim=None):
        """
        参数:
            in_features (int): 输入特征维度
            hidden_dim (int, optional): 注意力隐藏层维度
        """
        super(SelfAttention, self).__init__()

        if hidden_dim is None:
            hidden_dim = in_features // 8
            hidden_dim = max(1, hidden_dim)

        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.value = nn.Linear(in_features, in_features)

        # 初始化
        nn.init.xavier_normal_(self.query.weight)
        nn.init.xavier_normal_(self.key.weight)
        nn.init.xavier_normal_(self.value.weight)

    def forward(self, x):
        """
        前向传播

        参数:
            x (Tensor): 输入特征 [batch_size, sequence_length, in_features]

        返回:
            Tensor: 注意力加权后的特征
        """
        # 计算注意力分数
        query = self.query(x)  # [batch, seq_len, hidden_dim]
        key = self.key(x)  # [batch, seq_len, hidden_dim]
        value = self.value(x)  # [batch, seq_len, in_features]

        # 计算注意力权重
        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # 应用注意力权重
        out = torch.matmul(attn, value)

        return out


def scatter_add(src, index, dim=-1, dim_size=None):
    """
    自定义的scatter_add函数，用于消息聚合

    参数:
        src: 源张量
        index: 目标索引
        dim: 维度
        dim_size: 输出维度大小

    返回:
        聚合后的张量
    """
    if torch_geometric.is_compiled():
        return torch_geometric.scatter_add(src, index, dim, dim_size)
    else:
        return torch.scatter_add(
            torch.zeros((dim_size,) + src.shape[1:], dtype=src.dtype, device=src.device),
            dim, index.reshape(-1, 1).expand(-1, src.shape[1:]), src
        )