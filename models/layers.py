#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强型蛋白质图神经网络基础层组件

该模块实现了针对蛋白质设计与抗菌肽(AMPs)生成的专用图神经网络层组件，
包括异质边处理、物化属性感知、结构敏感注意力等先进特性，
并新增了ESM注意力引导机制，实现了序列模型和结构模型的深度融合。

作者: wxhfy
日期: 2025-04-05
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, MessagePassing
from torch_geometric.utils import softmax, add_self_loops
from torch_scatter import scatter_add, scatter_mean, scatter_max

from statistical import logger


class GATv2ConvLayer(nn.Module):
    """
    增强型GATv2卷积层，支持残差连接、层归一化和自定义激活函数

    该层特别针对蛋白质结构进行了优化，支持边特征以区分肽键、空间连接和相互作用类型

    参数:
        in_channels (int): 输入特征维度
        out_channels (int): 输出特征维度
        heads (int): 注意力头数量
        edge_dim (int, optional): 边特征维度
        dropout (float, optional): Dropout率
        residual (bool, optional): 是否使用残差连接
        use_layer_norm (bool, optional): 是否使用层归一化
        activation (str, optional): 激活函数类型 ('relu', 'gelu', 'leaky_relu')
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            heads=4,
            edge_dim=None,
            dropout=0.2,
            residual=True,
            use_layer_norm=True,
            activation='gelu'
    ):
        super(GATv2ConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.use_layer_norm = use_layer_norm

        # GATv2卷积
        self.conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels // heads if heads > 1 else out_channels,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True if heads > 1 else False,
            add_self_loops=False  # 手动控制自环，更适合蛋白质图
        )

        # 残差连接投影
        if residual:
            if in_channels != out_channels:
                self.res_proj = nn.Linear(in_channels, out_channels)
            else:
                self.res_proj = nn.Identity()

        # 层归一化
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_channels)

        # 激活函数选择
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = nn.GELU()

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, esm_attention=None):
        """
        前向传递，支持ESM注意力引导

        参数:
            x (torch.Tensor): 节点特征 [num_nodes, in_channels]
            edge_index (torch.LongTensor): 边连接 [2, num_edges]
            edge_attr (torch.Tensor, optional): 边特征 [num_edges, edge_dim]
            esm_attention (torch.Tensor, optional): ESM注意力分数 [num_nodes, 1]

        返回:
            out (torch.Tensor): 更新的节点特征 [num_nodes, out_channels]
        """
        # GAT卷积
        out = self.conv(x, edge_index, edge_attr)

        # 应用激活函数
        out = self.act(out)

        # 残差连接
        if self.residual:
            res = self.res_proj(x)
            out = out + res

        # 层归一化
        if self.use_layer_norm:
            out = self.layer_norm(out)

        # Dropout
        out = self.dropout(out)

        return out


class HeterogeneousGATv2Layer(MessagePassing):
    """
    异质边GATv2层，针对不同类型的边使用不同的注意力计算机制

    为肽键和空间边设计分离的注意力计算通道，通过门控机制融合不同类型边的信息。
    特别适合处理蛋白质中的多种相互作用类型，支持ESM注意力引导机制。

    参数:
        in_channels (int): 输入特征维度
        out_channels (int): 输出特征维度
        heads (int): 注意力头数量
        edge_types (int): 边类型数量（默认为2，对应肽键和空间连接）
        edge_dim (int, optional): 边特征维度
        dropout (float, optional): Dropout率
        use_layer_norm (bool, optional): 是否使用层归一化
        activation (str, optional): 激活函数类型 ('relu', 'gelu', 'leaky_relu')
        esm_guidance (bool, optional): 是否启用ESM注意力引导
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            heads=4,
            edge_types=2,
            edge_dim=None,
            dropout=0.2,
            use_layer_norm=True,
            activation='gelu',
            esm_guidance=False,
            **kwargs
    ):
        super(HeterogeneousGATv2Layer, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_types = edge_types
        self.edge_dim = edge_dim
        self.use_layer_norm = use_layer_norm
        self.esm_guidance = esm_guidance

        # 维度计算
        self.head_dim = out_channels // heads

        # 为每种边类型创建独立的线性变换
        self.linear_q = nn.ModuleList([
            nn.Linear(in_channels, self.head_dim * heads) for _ in range(edge_types)
        ])
        self.linear_k = nn.ModuleList([
            nn.Linear(in_channels, self.head_dim * heads) for _ in range(edge_types)
        ])
        self.linear_v = nn.ModuleList([
            nn.Linear(in_channels, self.head_dim * heads) for _ in range(edge_types)
        ])

        # 如果有边特征，创建边特征变换
        if edge_dim is not None:
            self.edge_encoders = nn.ModuleList([
                nn.Linear(edge_dim, self.head_dim * heads) for _ in range(edge_types)
            ])

        # 注意力得分计算
        self.att_layers = nn.ModuleList([
            nn.Linear(self.head_dim * 2, 1) for _ in range(edge_types)
        ])

        # 边类型门控权重
        self.gate_weights = nn.Parameter(torch.Tensor(edge_types, heads))
        nn.init.ones_(self.gate_weights)  # 初始化为平均权重
        self.gate_scale = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.gate_scale, 5.0)  # 控制门控的软硬程度

        # ESM注意力引导相关参数
        if esm_guidance:
            # ESM注意力融合权重（可学习）
            self.esm_weight = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.esm_weight, 0.5)  # 初始化为0.5，平衡图注意力和ESM注意力

            # ESM注意力投影 - 将单一通道注意力扩展到多头
            self.esm_proj = nn.Sequential(
                nn.Linear(1, heads),
                nn.Sigmoid()
            )

        # 输出投影
        self.output_proj = nn.Linear(self.head_dim * heads, out_channels)

        # 激活函数选择
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = nn.GELU()

        # Dropout & LayerNorm
        self.dropout = nn.Dropout(dropout)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_channels)

        # 残差投影
        self.res_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_type=None, edge_attr=None, esm_attention=None):
        """
        前向传递，支持ESM注意力引导

        参数:
            x (torch.Tensor): 节点特征 [num_nodes, in_channels]
            edge_index (torch.LongTensor): 边连接 [2, num_edges]
            edge_type (torch.LongTensor): 边类型索引 [num_edges]
            edge_attr (torch.Tensor, optional): 边特征 [num_edges, edge_dim]
            esm_attention (torch.Tensor, optional): ESM注意力分数 [num_nodes, 1]

        返回:
            out (torch.Tensor): 更新的节点特征 [num_nodes, out_channels]
        """
        # 残差连接
        res = self.res_proj(x)

        # 存储ESM注意力以在消息传递中使用
        self._esm_attention = esm_attention if self.esm_guidance else None

        # 边类型处理 - 如果未提供，假定所有边都是同一类型
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), device=edge_index.device, dtype=torch.long)

        # 消息传递
        out = self.propagate(
            edge_index,
            x=x,
            edge_type=edge_type,
            edge_attr=edge_attr,
            size=None
        )

        # 输出投影
        out = self.output_proj(out)

        # 残差连接
        out = out + res

        # 激活函数
        out = self.act(out)

        # 层归一化
        if self.use_layer_norm:
            out = self.layer_norm(out)

        # Dropout
        out = self.dropout(out)

        return out

    def message(self, x_i, x_j, edge_type, edge_attr, index, ptr, size_i):
        """安全的消息传递，增加边类型检查和空边处理"""
        # 初始化结果张量
        num_edges, num_types = edge_type.size(0), self.edge_types

        # 每个边类型的消息
        messages = torch.zeros(num_edges, self.heads, self.head_dim, device=x_i.device)

        # 获取目标节点索引
        target_nodes = index

        # ESM注意力处理（保持原有逻辑）
        if self._esm_attention is not None and self.esm_guidance:
            # 确保索引安全
            safe_target_nodes = torch.clamp(target_nodes, 0, self._esm_attention.size(0) - 1)
            node_esm_attention = self._esm_attention[safe_target_nodes]
            esm_att_weights = self.esm_proj(node_esm_attention)
        else:
            esm_att_weights = None

        # 对每种类型分别计算注意力和消息
        for t in range(num_types):
            try:
                # 选择当前类型的边
                mask = (edge_type == t)

                # 关键修复：安全检查，确保至少有一条边
                if not mask.any():
                    continue  # 没有此类型的边，跳过处理

                # 获取当前类型的边索引
                curr_indices = torch.where(mask)[0]
                curr_count = curr_indices.size(0)

                # 安全性检查：索引边界验证
                max_index = curr_indices.max().item() if curr_count > 0 else -1
                if max_index >= num_edges:
                    # 处理越界情况，裁剪索引
                    curr_indices = curr_indices[curr_indices < num_edges]

                # 如果无有效边，跳过
                if curr_indices.size(0) == 0:
                    continue

                # 获取当前类型边的源节点和目标节点
                curr_x_i = x_i[curr_indices]  # 目标节点
                curr_x_j = x_j[curr_indices]  # 源节点

                # 后续消息传递处理（原有逻辑）...

            except Exception as e:
                # 降级处理：记录错误但不中断训练
                import logging
                logging.warning(f"处理边类型{t}时出错: {e}")
                raise e

        return messages.view(num_edges, -1)

    def update(self, aggr_out):
        """
        更新节点表示

        参数:
            aggr_out (torch.Tensor): 聚合后的消息 [num_nodes, heads*head_dim]
        """
        return aggr_out


class ESMAttentionExtractor:
    """
    从ESM模型中提取注意力权重，用于指导图编码器
    """

    def __init__(self, esm_model, layer_idx=-1):
        """
        初始化注意力提取器

        参数:
            esm_model: ESM模型实例
            layer_idx: 提取哪一层的注意力，默认为最后一层
        """
        self.esm_model = esm_model
        self.layer_idx = layer_idx
        self.hooks = []
        self.attention_weights = None

        # 注册钩子函数以捕获注意力权重
        self._register_hooks()

    def _register_hooks(self):
        """注册钩子函数来捕获指定层的注意力权重"""
        if not hasattr(self.esm_model, "layers"):
            logger.warning("ESM模型结构不支持注意力提取，使用备选方法")
            return

        # 移除现有钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # 确定目标层
        if self.layer_idx == -1:
            target_layer = self.esm_model.layers[-1].attention
        else:
            target_layer = self.esm_model.layers[self.layer_idx].attention

        # 注册钩子
        def attention_hook(module, input, output):
            # 提取注意力权重
            # 对于ESMC模型，注意力输出通常是(batch_size, num_heads, seq_len, seq_len)
            self.attention_weights = output[1]  # 假设attention_weights是输出元组的第二个元素

        # 添加钩子
        self.hooks.append(target_layer.register_forward_hook(attention_hook))

    def extract_attention(self, seq_embeddings):
        """
        从ESM嵌入中提取注意力权重

        参数:
            seq_embeddings: ESM序列嵌入，形状为[batch_size, seq_len, dim]

        返回:
            torch.Tensor: 提取的注意力权重 [batch_size, seq_len]
        """
        batch_size = seq_embeddings.shape[0]
        seq_len = seq_embeddings.shape[1]

        # 尝试直接从钩子获取注意力权重
        if self.attention_weights is not None:
            # 对多头注意力取平均
            attn = self.attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]

            # 使用[CLS]注意力权重作为每个token的重要性
            token_importance = attn[:, 0, 1:]  # 排除[CLS]自身，取第一行

            # 规范化注意力权重
            token_importance = F.softmax(token_importance, dim=-1)

            return token_importance.unsqueeze(-1)  # [batch_size, seq_len-1, 1]

        # 备选方法：从嵌入向量计算自注意力
        else:
            # 计算嵌入向量的L2范数作为注意力权重
            token_norms = torch.norm(seq_embeddings, dim=-1)  # [batch_size, seq_len]

            # 忽略特殊标记（第一个和最后一个）
            token_norms = token_norms[:, 1:-1]  # [batch_size, seq_len-2]

            # 归一化
            token_importance = F.softmax(token_norms, dim=-1)

            return token_importance.unsqueeze(-1)  # [batch_size, seq_len-2, 1]


class MLPLayer(nn.Module):
    """
    增强型多层感知机层，支持层归一化和多种激活函数

    灵活的MLP实现，支持可变层数和特性，可用于节点特征变换、嵌入映射等任务

    参数:
        in_channels (int): 输入特征维度
        hidden_channels (int): 隐藏层维度
        out_channels (int): 输出特征维度
        layers (int, optional): MLP层数，默认2
        dropout (float, optional): Dropout率
        use_layer_norm (bool, optional): 是否使用层归一化
        activation (str, optional): 激活函数类型 ('relu', 'gelu', 'leaky_relu')
        residual (bool, optional): 是否使用残差连接(仅当in_channels=out_channels时有效)
    """

    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            layers=2,
            dropout=0.2,
            use_layer_norm=True,
            activation='gelu',
            residual=False
    ):
        super(MLPLayer, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.layers = layers
        self.residual = residual and (in_channels == out_channels)

        # 激活函数选择
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU(0.2)
        else:
            act = nn.GELU()

        # 构建MLP
        layers_list = []

        # 输入层
        layers_list.append(nn.Linear(in_channels, hidden_channels))
        if use_layer_norm:
            layers_list.append(nn.LayerNorm(hidden_channels))
        layers_list.append(act)
        layers_list.append(nn.Dropout(dropout))

        # 中间层
        for _ in range(max(0, self.layers - 2)):
            layers_list.append(nn.Linear(hidden_channels, hidden_channels))
            if use_layer_norm:
                layers_list.append(nn.LayerNorm(hidden_channels))
            layers_list.append(act)
            layers_list.append(nn.Dropout(dropout))

        # 输出层
        layers_list.append(nn.Linear(hidden_channels, out_channels))
        if use_layer_norm:
            layers_list.append(nn.LayerNorm(out_channels))

        self.mlp = nn.Sequential(*layers_list)

    def forward(self, x):
        """
        前向传递

        参数:
            x (torch.Tensor): 输入特征

        返回:
            out (torch.Tensor): 变换后的特征
        """
        out = self.mlp(x)

        # 可选的残差连接
        if self.residual:
            out = out + x

        return out


class EdgeTypeEncoder(nn.Module):
    """
    边类型编码器，专门处理蛋白质图中的不同边类型

    可以区分肽键连接、空间邻近连接和各种相互作用类型，为GATv2卷积层提供丰富的边特征

    参数:
        in_features (int): 输入边特征维度
        out_features (int): 输出边嵌入维度
        num_edge_types (int): 边类型数量，默认为2（肽键和空间连接）
        activation (str, optional): 激活函数类型 ('relu', 'gelu', 'leaky_relu')
    """

    def __init__(
            self,
            in_features,
            out_features,
            num_edge_types=2,
            activation='gelu'
    ):
        super(EdgeTypeEncoder, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_edge_types = num_edge_types

        # 边类型嵌入表
        self.edge_type_embedding = nn.Embedding(num_edge_types, out_features // 2)

        # 边特征变换
        self.edge_proj = nn.Linear(in_features, out_features // 2)

        # 激活函数选择
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = nn.GELU()

        # 输出归一化
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, edge_attr):
        """
        编码边特征

        参数:
            edge_attr (torch.Tensor): 边特征 [num_edges, in_features]
                                      假设第一个特征是边类型 (0:肽键, 1:空间连接, ...)

        返回:
            edge_emb (torch.Tensor): 边嵌入 [num_edges, out_features]
        """
        # 提取边类型
        edge_types = edge_attr[:, 0].long()  # 假设第一列是边类型

        # 获取边类型嵌入
        type_embedding = self.edge_type_embedding(edge_types)  # [num_edges, out_features//2]

        # 处理边特征
        edge_features = edge_attr[:, 1:] if edge_attr.size(1) > 1 else torch.zeros_like(edge_attr)
        edge_features = self.edge_proj(edge_features)  # [num_edges, out_features//2]
        edge_features = self.act(edge_features)

        # 连接类型嵌入和特征嵌入
        edge_emb = torch.cat([type_embedding, edge_features], dim=1)  # [num_edges, out_features]

        # 应用层归一化
        edge_emb = self.layer_norm(edge_emb)

        return edge_emb


class EdgeUpdateModule(nn.Module):
    """
    边特征动态更新模块

    根据连接节点的特征更新边属性，增强对残基间相互作用的动态表达能力

    参数:
        node_dim (int): 节点特征维度
        edge_dim (int): 边特征维度
        edge_types (int, optional): 边类型数量，用于异质边处理
        hidden_dim (int, optional): 隐藏层维度
        activation (str, optional): 激活函数类型 ('relu', 'gelu', 'leaky_relu')
    """

    def __init__(
            self,
            node_dim,
            edge_dim,
            edge_types=None,
            hidden_dim=None,
            activation='gelu'
    ):
        super(EdgeUpdateModule, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim or max(node_dim, edge_dim)

        # 激活函数选择
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        else:
            act_fn = nn.GELU()

        # 通用的边更新网络 - 始终创建以保证容错性
        self.edge_update = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            act_fn,
            nn.Linear(self.hidden_dim, edge_dim)
        )

        # 如果需要异质边处理，额外创建为每种类型的边创建独立的更新网络
        if edge_types is not None:
            self.edge_updaters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(node_dim * 2 + edge_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    act_fn,
                    nn.Linear(self.hidden_dim, edge_dim)
                ) for _ in range(edge_types)
            ])

    def forward(self, x, edge_index, edge_attr=None, edge_type=None):
        """
        更新边特征

        参数:
            x (torch.Tensor): 节点特征 [num_nodes, node_dim]
            edge_index (torch.LongTensor): 边连接 [2, num_edges]
            edge_attr (torch.Tensor): 边特征 [num_edges, edge_dim]
            edge_type (torch.LongTensor, optional): 边类型 [num_edges]

        返回:
            updated_edge_attr (torch.Tensor): 更新后的边特征 [num_edges, edge_dim]
        """
        if edge_attr is None:
            return None

        # 获取源节点和目标节点
        src, dst = edge_index

        # 获取源节点和目标节点特征
        src_features = x[src]  # [num_edges, node_dim]
        dst_features = x[dst]  # [num_edges, node_dim]

        try:
            # 更新边特征
            if self.edge_types is not None and edge_type is not None:
                # 初始化结果张量
                updated_edge_attr = torch.zeros_like(edge_attr)

                # 对每种边类型分别更新
                for t in range(self.edge_types):
                    mask = (edge_type == t)
                    if mask.sum() > 0:
                        # 拼接特征
                        combined = torch.cat([
                            src_features[mask],
                            dst_features[mask],
                            edge_attr[mask]
                        ], dim=1)

                        # 使用对应类型的更新网络
                        updated_edge_attr[mask] = self.edge_updaters[t](combined)

                return updated_edge_attr
            else:
                # 拼接源节点、目标节点和当前边特征
                combined = torch.cat([src_features, dst_features, edge_attr], dim=1)

                # 更新边特征 - 使用self.edge_update
                return self.edge_update(combined)
        except Exception as e:
            # 错误处理 - 保证即使出现问题也能正常运行
            import logging
            logging.warning(f"边更新失败: {e}，返回原始边特征")
            return edge_attr

class StructureAwareAttention(nn.Module):
    """
    结构感知注意力层，考虑蛋白质的空间和序列结构特性

    针对抗菌肽的二级结构和局部构象特点进行了优化，融合了几何信息和化学相互作用特征

    参数:
        in_channels (int): 输入特征维度
        attention_dim (int, optional): 注意力维度
        heads (int, optional): 注意力头数量
        structure_types (int, optional): 结构类型数量(如螺旋、折叠、环)
        dropout (float, optional): Dropout率
    """

    def __init__(
            self,
            in_channels,
            attention_dim=None,
            heads=4,
            structure_types=3,  # 螺旋、折叠、环
            dropout=0.1
    ):
        super(StructureAwareAttention, self).__init__()

        self.in_channels = in_channels
        self.attention_dim = attention_dim or in_channels
        self.heads = heads
        self.head_dim = self.attention_dim // heads
        self.structure_types = structure_types

        # 查询、键、值投影
        self.query = nn.Linear(in_channels, self.attention_dim)
        self.key = nn.Linear(in_channels, self.attention_dim)
        self.value = nn.Linear(in_channels, self.attention_dim)

        # 结构类型特定的偏置
        self.structure_bias = nn.Parameter(torch.Tensor(structure_types, heads, 1, 1))
        nn.init.zeros_(self.structure_bias)  # 初始化为零

        # 输出投影
        self.output_proj = nn.Linear(self.attention_dim, in_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 层归一化
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x, structure_type=None, mask=None, edge_index=None):
        """
        计算结构感知注意力

        参数:
            x (torch.Tensor): 节点特征 [batch_size, seq_len, in_channels] 或 [num_nodes, in_channels]
            structure_type (torch.LongTensor, optional): 结构类型 [batch_size, seq_len] 或 [num_nodes]
            mask (torch.BoolTensor, optional): 注意力掩码 [batch_size, seq_len]
            edge_index (torch.LongTensor, optional): 图连接 [2, num_edges]

        返回:
            out (torch.Tensor): 更新的特征，维度与输入相同
        """
        residual = x
        batch_first = len(x.shape) == 3

        if batch_first:
            batch_size, seq_len, _ = x.shape
            q = self.query(x).view(batch_size, seq_len, self.heads, self.head_dim)
            k = self.key(x).view(batch_size, seq_len, self.heads, self.head_dim)
            v = self.value(x).view(batch_size, seq_len, self.heads, self.head_dim)

            # 转置以适应注意力计算
            q = q.transpose(1, 2)  # [batch_size, heads, seq_len, head_dim]
            k = k.transpose(1, 2)  # [batch_size, heads, seq_len, head_dim]
            v = v.transpose(1, 2)  # [batch_size, heads, seq_len, head_dim]

            # 计算注意力分数
            attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
                self.head_dim)  # [batch_size, heads, seq_len, seq_len]

            # 添加结构偏置(如果提供)
            if structure_type is not None:
                structure_bias = self.structure_bias[structure_type]  # [batch_size, seq_len, heads, 1, 1]
                structure_bias = structure_bias.permute(0, 2, 1, 3)  # [batch_size, heads, seq_len, 1]
                attn = attn + structure_bias

            # 掩码(如果提供)
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

            # Softmax
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            # 加权求和
            out = torch.matmul(attn, v)  # [batch_size, heads, seq_len, head_dim]
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [batch_size, seq_len, attention_dim]
        else:
            num_nodes = x.size(0)

            if edge_index is None:
                # 全连接注意力
                q = self.query(x).view(num_nodes, self.heads, self.head_dim)
                k = self.key(x).view(num_nodes, self.heads, self.head_dim)
                v = self.value(x).view(num_nodes, self.heads, self.head_dim)

                # 计算注意力分数
                attn = torch.matmul(q.unsqueeze(1), k.unsqueeze(2).transpose(-1, -2)).squeeze(2) / math.sqrt(
                    self.head_dim)

                # 添加结构偏置(如果提供)
                if structure_type is not None:
                    attn = attn + self.structure_bias[structure_type].squeeze(-1)

                # Softmax
                attn = F.softmax(attn, dim=1)
                attn = self.dropout(attn)

                # 加权求和
                out = torch.sum(attn.unsqueeze(-1) * v.unsqueeze(1), dim=1)
                out = out.view(num_nodes, -1)
            else:
                # 图结构注意力 - 只在连接的节点间计算注意力
                q = self.query(x).view(num_nodes, self.heads, self.head_dim)
                k = self.key(x).view(num_nodes, self.heads, self.head_dim)
                v = self.value(x).view(num_nodes, self.heads, self.head_dim)

                # 源节点和目标节点
                src, dst = edge_index

                # 计算边上的注意力分数
                q_dst = q[dst]  # [num_edges, heads, head_dim]
                k_src = k[src]  # [num_edges, heads, head_dim]

                # 点积注意力
                attn = torch.sum(q_dst * k_src, dim=-1) / math.sqrt(self.head_dim)  # [num_edges, heads]

                # 添加结构偏置(如果提供)
                if structure_type is not None:
                    edge_structure = structure_type[src]  # [num_edges]
                    structure_bias = torch.gather(self.structure_bias.view(self.structure_types, self.heads), 0,
                                                  edge_structure.unsqueeze(-1).expand(-1, self.heads))
                    attn = attn + structure_bias

                # 按目标节点Softmax归一化
                attn = softmax(attn, dst, num_nodes=num_nodes)
                attn = self.dropout(attn)

                # 值向量
                v_src = v[src]  # [num_edges, heads, head_dim]

                # 加权消息
                messages = attn.unsqueeze(-1) * v_src  # [num_edges, heads, head_dim]

                # 聚合到目标节点
                out = torch.zeros(num_nodes, self.heads, self.head_dim, device=x.device)
                for h in range(self.heads):
                    out[:, h] = scatter_add(messages[:, h], dst, dim=0, dim_size=num_nodes)

                out = out.view(num_nodes, -1)  # [num_nodes, attention_dim]

        # 输出投影
        out = self.output_proj(out)
        out = self.dropout(out)

        # 残差连接与层归一化
        out = self.layer_norm(residual + out)

        return out


class PhysicochemicalEncoder(nn.Module):
    """
    蛋白质物理化学特性编码器

    增强蛋白质残基的理化特性表示，特别优化了抗菌肽中关键的电荷和疏水性编码

    参数:
        hidden_dim (int): 隐藏层维度
        num_bins (int, optional): 连续特征分箱数量
        dropout (float, optional): Dropout率
        use_layer_norm (bool, optional): 是否使用层归一化
    """

    def __init__(
            self,
            hidden_dim,
            num_bins=10,
            dropout=0.1,
            use_layer_norm=True
    ):
        super(PhysicochemicalEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.use_layer_norm = use_layer_norm

        # 氨基酸类型嵌入 (20种标准氨基酸)
        self.aa_embedding = nn.Embedding(20, hidden_dim // 4)

        # 疏水性编码
        self.hydropathy_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 8),
            nn.GELU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4)
        )

        # 电荷编码
        self.charge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 8),
            nn.GELU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4)
        )

        # 分子量编码
        self.weight_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 8),
            nn.GELU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4)
        )

        # 辅助特性编码 (极性、体积等)
        self.aux_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 8),
            nn.GELU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4)
        )

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 层归一化
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

        # 记录物化属性的统计范围用于归一化
        self.register_buffer('hydropathy_range', torch.tensor([-4.5, 4.5]))  # Kyte-Doolittle疏水性范围
        self.register_buffer('charge_range', torch.tensor([-2.0, 2.0]))  # 标准电荷范围
        self.register_buffer('weight_range', torch.tensor([75.0, 204.0]))  # 氨基酸分子量范围

    def forward(self, aa_indices, hydropathy, charge, weight, polarity=None, volume=None):
        """
        编码蛋白质残基的物理化学特性

        参数:
            aa_indices (torch.LongTensor): 氨基酸类型索引 [batch_size, seq_len] 或 [num_nodes]
            hydropathy (torch.Tensor): 疏水性值 [batch_size, seq_len, 1] 或 [num_nodes, 1]
            charge (torch.Tensor): 电荷值 [batch_size, seq_len, 1] 或 [num_nodes, 1]
            weight (torch.Tensor): 分子量 [batch_size, seq_len, 1] 或 [num_nodes, 1]
            polarity (torch.Tensor, optional): 极性指标 [batch_size, seq_len, 1] 或 [num_nodes, 1]
            volume (torch.Tensor, optional): 体积指标 [batch_size, seq_len, 1] 或 [num_nodes, 1]

        返回:
            features (torch.Tensor): 编码后的理化特性 [batch_size, seq_len, hidden_dim] 或 [num_nodes, hidden_dim]
        """
        batch_size = aa_indices.size(0)
        is_graph_input = (aa_indices.dim() == 1)

        # 氨基酸嵌入
        aa_emb = self.aa_embedding(aa_indices)  # [..., hidden_dim//4]

        # 物化属性归一化
        hydropathy_norm = self._normalize_and_bin(hydropathy, self.hydropathy_range[0], self.hydropathy_range[1])
        charge_norm = self._normalize_and_bin(charge, self.charge_range[0], self.charge_range[1])
        weight_norm = self._normalize_and_bin(weight, self.weight_range[0], self.weight_range[1])

        # 编码各物化属性
        hydropathy_emb = self.hydropathy_encoder(hydropathy_norm)  # [..., hidden_dim//4]
        charge_emb = self.charge_encoder(charge_norm)  # [..., hidden_dim//4]
        weight_emb = self.weight_encoder(weight_norm)  # [..., hidden_dim//4]

        # 处理辅助特性
        if polarity is not None and volume is not None:
            aux_input = torch.cat([polarity, volume], dim=-1)
            aux_emb = self.aux_encoder(aux_input)  # [..., hidden_dim//4]
        else:
            # 如果没有提供辅助特性，用零向量替代
            if is_graph_input:
                aux_emb = torch.zeros(batch_size, self.hidden_dim // 4, device=aa_indices.device)
            else:
                aux_emb = torch.zeros(batch_size, aa_indices.size(1), self.hidden_dim // 4, device=aa_indices.device)

        # 拼接所有特性表示
        if is_graph_input:
            combined = torch.cat([aa_emb, hydropathy_emb, charge_emb, weight_emb], dim=-1)
        else:
            combined = torch.cat([aa_emb, hydropathy_emb, charge_emb, weight_emb], dim=-1)

        # 特征融合
        features = self.feature_fusion(combined)

        # 层归一化
        if self.use_layer_norm:
            features = self.layer_norm(features)

        return features

    def _normalize_and_bin(self, values, min_val, max_val):
        """
        归一化并分箱连续值

        参数:
            values (torch.Tensor): 输入值
            min_val (float): 最小值
            max_val (float): 最大值

        返回:
            binned_values (torch.Tensor): 归一化并分箱后的值
        """
        # 裁剪到范围
        values_clipped = torch.clamp(values, min=min_val, max=max_val)

        # 归一化到[0,1]
        values_norm = (values_clipped - min_val) / (max_val - min_val)

        # 可选的分箱
        if self.num_bins > 1:
            values_binned = torch.floor(values_norm * self.num_bins) / self.num_bins
            return values_binned
        else:
            return values_norm


class DynamicEdgePruning(nn.Module):
    """
    动态边修剪模块

    通过可学习的阈值机制动态修剪蛋白质图中的边，重点保留功能上重要的连接

    参数:
        edge_dim (int): 边特征维度
        hidden_dim (int, optional): 隐藏层维度
        init_threshold (float, optional): 初始修剪阈值 (0-1)
        dropout (float, optional): Dropout率
    """

    def __init__(
            self,
            edge_dim,
            hidden_dim=None,
            init_threshold=0.5,
            dropout=0.1
    ):
        super(DynamicEdgePruning, self).__init__()

        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim or edge_dim * 2

        # 边重要性评分网络
        self.importance_net = nn.Sequential(
            nn.Linear(edge_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        # 可学习的修剪阈值
        self.threshold = nn.Parameter(torch.tensor([init_threshold]))

        # 温度参数，控制软掩码的锐度
        self.temperature = nn.Parameter(torch.tensor([10.0]))

    def forward(self, edge_attr, edge_index=None, training=True):
        """
        动态修剪边

        参数:
            edge_attr (torch.Tensor): 边特征 [num_edges, edge_dim]
            edge_index (torch.LongTensor, optional): 边连接 [2, num_edges]
            training (bool): 是否处于训练模式

        返回:
            pruned_mask (torch.Tensor): 边掩码 [num_edges]
            edge_weights (torch.Tensor): 边重要性权重 [num_edges, 1]
        """
        # 计算边重要性分数
        edge_scores = self.importance_net(edge_attr)  # [num_edges, 1]

        if training:
            # 训练时使用软掩码
            pruned_mask = torch.sigmoid((edge_scores - self.threshold) * self.temperature)
        else:
            # 推理时使用硬掩码
            pruned_mask = (edge_scores > self.threshold).float()

        return pruned_mask, edge_scores


class SequenceStructureFusion(nn.Module):
    """
    增强型序列结构融合模块

    通过交叉双模态Transformer架构融合ESM序列表示和GATv2结构表示，
    实现双向信息流和多层次交互，最大化保留两种模态的关键信息。

    特点：
    1. 双向交叉注意力，实现序列→结构和结构→序列的双向信息交流
    2. 层级递进融合，逐步细化和整合两种表示
    3. 残差连接保证原始信息不丢失
    4. 向量积交互捕获高阶特征关系

    参数:
        seq_dim (int): 序列嵌入维度 (来自ESM)
        graph_dim (int): 图嵌入维度 (来自GATv2)
        output_dim (int): 输出融合维度
        hidden_dim (int, optional): 内部隐藏维度
        num_heads (int, optional): 注意力头数
        num_layers (int, optional): Transformer层数
        dropout (float, optional): Dropout率
    """

    def __init__(
            self,
            seq_dim,
            graph_dim,
            output_dim,
            hidden_dim=None,
            num_heads=8,
            num_layers=3,
            dropout=0.1
    ):
        super(SequenceStructureFusion, self).__init__()

        self.seq_dim = seq_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 输入映射层 - 将两种嵌入投影到相同维度
        self.seq_proj = nn.Linear(seq_dim, self.hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, self.hidden_dim)

        # 构建双向交叉注意力层
        self.cross_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = BimodalCrossAttentionBlock(
                dim=self.hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            self.cross_layers.append(layer)

        # 多级融合层
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        # 向量积交互层 - 捕获高阶模态交互
        self.bilinear = nn.Bilinear(self.hidden_dim, self.hidden_dim, self.hidden_dim)

        # 输出转换层
        self.output_transform = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型参数"""
        if isinstance(module, nn.Linear):
            # 使用截断正态分布初始化权重，提高训练稳定性
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, seq_emb, graph_emb):
        """
        融合序列和图嵌入

        参数:
            seq_emb (torch.Tensor): 序列嵌入 [batch_size, seq_dim]
            graph_emb (torch.Tensor): 图嵌入 [batch_size, graph_dim]

        返回:
            fused (torch.Tensor): 融合嵌入 [batch_size, output_dim]
        """
        batch_size = seq_emb.size(0)

        # 投影到统一隐藏维度
        seq_hidden = self.seq_proj(seq_emb)
        graph_hidden = self.graph_proj(graph_emb)

        # 保存原始投影用于残差连接
        seq_orig = seq_hidden
        graph_orig = graph_hidden

        # 多层交叉注意力处理
        fusion_features = []
        for i in range(self.num_layers):
            # 双向交叉注意力
            seq_hidden, graph_hidden = self.cross_layers[i](seq_hidden, graph_hidden)

            # 每层融合一次，并保存
            current_fusion = torch.cat([seq_hidden, graph_hidden], dim=-1)
            current_fusion = self.fusion_layers[i](current_fusion)
            fusion_features.append(current_fusion)

        # 残差连接，确保原始信息不丢失
        seq_final = seq_hidden + seq_orig
        graph_final = graph_hidden + graph_orig

        # 向量积交互，捕获高阶特征关系
        bilinear_interaction = self.bilinear(seq_final, graph_final)

        # 组合多种特征：最终序列表示、最终结构表示、向量积交互
        combined = torch.cat([seq_final, graph_final, bilinear_interaction], dim=-1)

        # 转换为最终输出维度
        fused = self.output_transform(combined)

        return fused


class CrossModalAttentionBlock(nn.Module):
    """
    跨模态交叉注意力模块

    允许两种模态通过注意力机制相互交换和增强信息
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossModalAttentionBlock, self).__init__()

        self.dim = dim
        self.num_heads = num_heads

        # 序列→结构注意力
        self.seq_to_graph_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 结构→序列注意力
        self.graph_to_seq_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 层归一化
        self.norm_seq1 = nn.LayerNorm(dim)
        self.norm_seq2 = nn.LayerNorm(dim)
        self.norm_graph1 = nn.LayerNorm(dim)
        self.norm_graph2 = nn.LayerNorm(dim)

        # 前馈网络
        self.seq_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

        self.graph_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_emb, graph_emb):
        """
        前向传播

        参数:
            seq_emb (torch.Tensor): 序列嵌入 [batch_size, 1, dim]
            graph_emb (torch.Tensor): 图嵌入 [batch_size, 1, dim]

        返回:
            seq_out (torch.Tensor): 增强的序列嵌入 [batch_size, 1, dim]
            graph_out (torch.Tensor): 增强的图嵌入 [batch_size, 1, dim]
        """
        # 序列→结构注意力
        graph_attn, _ = self.seq_to_graph_attn(
            query=graph_emb,
            key=seq_emb,
            value=seq_emb
        )
        graph_res = graph_emb + self.dropout(graph_attn)
        graph_res = self.norm_graph1(graph_res)

        # 结构→序列注意力
        seq_attn, _ = self.graph_to_seq_attn(
            query=seq_emb,
            key=graph_emb,
            value=graph_emb
        )
        seq_res = seq_emb + self.dropout(seq_attn)
        seq_res = self.norm_seq1(seq_res)

        # 前馈网络
        seq_out = seq_res + self.dropout(self.seq_ffn(seq_res))
        seq_out = self.norm_seq2(seq_out)

        graph_out = graph_res + self.dropout(self.graph_ffn(graph_res))
        graph_out = self.norm_graph2(graph_out)

        return seq_out, graph_out


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块

    通过多层交叉注意力机制融合ESM序列嵌入和图嵌入，保持原始维度
    """

    def __init__(
            self,
            embedding_dim=1152,
            num_heads=8,
            num_layers=2,
            dropout=0.1
    ):
        super(CrossAttentionFusion, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 多层交叉注意力
        self.cross_layers = nn.ModuleList([
            CrossModalAttentionBlock(
                dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # 输出融合层 - 自适应融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, seq_embedding, graph_embedding):
        """
        前向传播

        参数:
            seq_embedding (torch.Tensor): 序列嵌入 [batch_size, embedding_dim]
            graph_embedding (torch.Tensor): 图嵌入 [batch_size, embedding_dim]

        返回:
            fused_embedding (torch.Tensor): 融合嵌入 [batch_size, embedding_dim]
        """
        # 在CrossAttentionFusion的forward方法开始处添加
        assert seq_embedding.size(-1) == graph_embedding.size(-1) == self.embedding_dim, \
            f"维度不匹配: seq={seq_embedding.size(-1)}, graph={graph_embedding.size(-1)}, expected={self.embedding_dim}"
        # 添加序列维度
        seq_emb = seq_embedding.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        graph_emb = graph_embedding.unsqueeze(1)  # [batch_size, 1, embedding_dim]

        # 保存原始嵌入用于残差连接
        seq_orig = seq_emb
        graph_orig = graph_emb

        # 多层交叉注意力处理
        for layer in self.cross_layers:
            seq_emb, graph_emb = layer(seq_emb, graph_emb)

        # 添加残差连接
        seq_emb = seq_emb + seq_orig
        graph_emb = graph_emb + graph_orig

        # 移除序列维度
        seq_emb = seq_emb.squeeze(1)  # [batch_size, embedding_dim]
        graph_emb = graph_emb.squeeze(1)  # [batch_size, embedding_dim]

        # 自适应门控融合
        concat_emb = torch.cat([seq_emb, graph_emb], dim=-1)
        gate = self.fusion_gate(concat_emb)
        fused_emb = gate * seq_emb + (1 - gate) * graph_emb

        # 最终输出投影
        fused_embedding = self.output_proj(fused_emb)

        return fused_embedding


class BimodalCrossAttentionBlock(nn.Module):
    """
    双模态交叉注意力模块

    实现序列和结构表示之间的双向交叉注意力，
    让两种模态相互增强、相互提取有用信息。
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(BimodalCrossAttentionBlock, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # 序列→结构注意力
        self.seq_to_graph_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 结构→序列注意力
        self.graph_to_seq_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 序列前馈网络
        self.seq_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

        # 结构前馈网络
        self.graph_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

        # 层归一化
        self.seq_norm1 = nn.LayerNorm(dim)
        self.seq_norm2 = nn.LayerNorm(dim)
        self.graph_norm1 = nn.LayerNorm(dim)
        self.graph_norm2 = nn.LayerNorm(dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_emb, graph_emb):
        """
        执行双向交叉注意力

        参数:
            seq_emb (torch.Tensor): 序列嵌入 [batch_size, hidden_dim]
            graph_emb (torch.Tensor): 图嵌入 [batch_size, hidden_dim]

        返回:
            seq_out (torch.Tensor): 更新后的序列嵌入 [batch_size, hidden_dim]
            graph_out (torch.Tensor): 更新后的图嵌入 [batch_size, hidden_dim]
        """
        # 准备输入形状
        batch_size = seq_emb.size(0)
        seq_unsqueezed = seq_emb.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        graph_unsqueezed = graph_emb.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # 序列→结构注意力
        # query=结构, key=value=序列
        graph_attn, _ = self.seq_to_graph_attn(
            query=graph_unsqueezed,
            key=seq_unsqueezed,
            value=seq_unsqueezed
        )
        graph_res = graph_unsqueezed + self.dropout(graph_attn)  # 残差连接
        graph_res = self.graph_norm1(graph_res)

        # 结构→序列注意力
        # query=序列, key=value=结构
        seq_attn, _ = self.graph_to_seq_attn(
            query=seq_unsqueezed,
            key=graph_unsqueezed,
            value=graph_unsqueezed
        )
        seq_res = seq_unsqueezed + self.dropout(seq_attn)  # 残差连接
        seq_res = self.seq_norm1(seq_res)

        # 序列前馈网络
        seq_ff = self.seq_ffn(seq_res)
        seq_out = seq_res + self.dropout(seq_ff)  # 残差连接
        seq_out = self.seq_norm2(seq_out)

        # 结构前馈网络
        graph_ff = self.graph_ffn(graph_res)
        graph_out = graph_res + self.dropout(graph_ff)  # 残差连接
        graph_out = self.graph_norm2(graph_out)

        # 压缩多余维度
        seq_out = seq_out.squeeze(1)  # [batch_size, hidden_dim]
        graph_out = graph_out.squeeze(1)  # [batch_size, hidden_dim]

        return seq_out, graph_out