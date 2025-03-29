#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级优化蛋白质知识图谱GATv2编码器模型

该模块实现了针对抗菌肽(AMPs)设计的先进GATv2编码器，融合了结构感知特征和异质边处理机制，
并通过跨模态对齐与ESM序列编码器协同工作，构建功能导向的蛋白质表示。

作者: wxhfy
日期: 2025-03-29
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax, to_dense_batch
from torch_scatter import scatter_add, scatter_mean

from .layers import MLPLayer, HeterogeneousGATv2Layer, EdgeUpdateModule, GATv2ConvLayer, DynamicEdgePruning
from .readout import AttentiveReadout, HierarchicalReadout


class ProteinGATv2Encoder(nn.Module):
    """
    高级优化蛋白质知识图谱GATv2编码器，针对抗菌肽特性与ESM协同设计

    优化特点：
    1. 相对位置编码与局部参考系：增强旋转平移不变性
    2. 物化属性分箱归一化：避免量纲差异影响注意力
    3. 异质边注意力设计：区分肽键与空间连接的不同信息流
    4. 跨模态对齐机制：与ESM编码器深度融合
    5. 注意力引导技术：整合ESM的序列进化信息

    参数:
        node_input_dim (int): 节点特征输入维度
        edge_input_dim (int): 边特征输入维度
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出嵌入维度
        num_layers (int): GATv2卷积层数量
        num_heads (int): 多头注意力的头数
        edge_types (int): 边类型数量（肽键/空间连接等）
        dropout (float): Dropout概率
        use_pos_encoding (bool): 是否使用相对位置编码
        use_heterogeneous_edges (bool): 是否使用异质边处理
        esm_guidance (bool): 是否使用ESM注意力引导
        activation (str): 激活函数类型
    """

    def __init__(
            self,
            node_input_dim,
            edge_input_dim,
            hidden_dim=128,
            output_dim=128,
            num_layers=3,
            num_heads=4,
            edge_types=2,
            dropout=0.2,
            use_pos_encoding=True,
            use_heterogeneous_edges=True,
            use_edge_pruning=False,  # 新增参数
            esm_guidance=True,
            activation='gelu',
    ):
        super(ProteinGATv2Encoder, self).__init__()

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.edge_types = edge_types
        self.use_pos_encoding = use_pos_encoding
        self.use_heterogeneous_edges = use_heterogeneous_edges
        self.use_edge_pruning = use_edge_pruning
        self.esm_guidance = esm_guidance

        # 激活函数选择
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.GELU()

        # =============== 特征处理模块 ===============

        # 相对位置编码 - 新增
        if self.use_pos_encoding:
            self.pos_encoder = RelativePositionEncoder(
                hidden_dim=hidden_dim // 2,
                max_seq_dist=32,
                dropout=dropout
            )
            # 增加节点输入维度以包含位置信息
            node_encoder_input_dim = node_input_dim + hidden_dim // 2
        else:
            node_encoder_input_dim = node_input_dim

        self.use_edge_pruning = use_edge_pruning
        if self.use_edge_pruning:
            self.edge_pruning = DynamicEdgePruning(
                edge_dim=hidden_dim // 2,
                hidden_dim=hidden_dim,
                dropout=dropout
            )

        # 节点特征编码 - 增强为多层处理
        self.node_encoder = MLPLayer(
            node_encoder_input_dim,
            hidden_dim * 2,
            hidden_dim,
            layers=2,
            dropout=dropout,
            activation=activation
        )

        # 物化属性标准化 - 新增
        self.property_normalizer = PropertyNormalizer()

        # 边类型处理 - 区分不同类型边
        if self.use_heterogeneous_edges:
            # 不同类型边的嵌入映射
            self.edge_type_embeddings = nn.Embedding(edge_types, hidden_dim // 4)
            # 针对不同类型边的特征变换
            self.edge_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(edge_input_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    self.activation
                ) for _ in range(edge_types)
            ])
        else:
            # 统一边特征处理
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_input_dim + edge_types, hidden_dim // 2),  # 加入one-hot边类型
                nn.LayerNorm(hidden_dim // 2),
                self.activation
            )


        # =============== 图卷积模块 ===============

        # 堆叠多层图卷积
        self.convs = nn.ModuleList()
        self.edge_updaters = nn.ModuleList()

        # 针对不同类型边的注意力层
        if self.use_heterogeneous_edges:
            for i in range(num_layers):
                in_dim = hidden_dim if i > 0 else hidden_dim
                out_dim = output_dim if i == num_layers - 1 else hidden_dim

                # 异质边注意力层
                self.convs.append(
                    HeterogeneousGATv2Layer(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        heads=1 if i == num_layers - 1 else num_heads,
                        edge_types=edge_types,
                        edge_dim=hidden_dim // 2,
                        dropout=dropout,
                        use_layer_norm=True,
                        activation=activation
                    )
                )

                # 边更新模块
                self.edge_updaters.append(
                    EdgeUpdateModule(
                        in_dim,
                        hidden_dim // 2,
                        edge_types=edge_types
                    )
                )
        else:
            # 标准GATv2层
            for i in range(num_layers):
                in_dim = hidden_dim if i > 0 else hidden_dim
                out_dim = output_dim if i == num_layers - 1 else hidden_dim

                self.convs.append(
                    GATv2ConvLayer(
                        in_dim,
                        out_dim,
                        heads=1 if i == num_layers - 1 else num_heads,
                        edge_dim=hidden_dim // 2,
                        dropout=dropout,
                        residual=True,
                        use_layer_norm=True,
                        activation=activation
                    )
                )

                # 边更新模块
                self.edge_updaters.append(
                    EdgeUpdateModule(
                        in_dim,
                        hidden_dim // 2
                    )
                )

        # =============== 多尺度特征聚合 ===============

        # 多尺度特征聚合 - 改进为注意力加权
        feature_fusion_input_dim = hidden_dim * (num_layers - 1) + output_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )

        # 层特征重要性自适应学习
        self.layer_attention = nn.Sequential(
            nn.Linear(feature_fusion_input_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, num_layers),
            nn.Softmax(dim=1)
        )

        # =============== 读出机制 ===============

        # 层次化读出
        self.readout = HierarchicalReadout(
            output_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # ESM注意力引导模块
        if self.esm_guidance:
            self.esm_attention_guide = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid()
            )

        # 残基重要性评分
        self.residue_importance = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1)
        )

    def forward(
            self,
            x,
            edge_index,
            edge_attr=None,
            edge_type=None,
            pos=None,
            batch=None,
            esm_attention=None
    ):
        """
        前向传递

        参数:
            x (torch.Tensor): 节点特征 [num_nodes, node_input_dim]
            edge_index (torch.LongTensor): 图的边连接 [2, num_edges]
            edge_attr (torch.Tensor): 边特征 [num_edges, edge_input_dim]
            edge_type (torch.LongTensor): 边类型索引 [num_edges]
            pos (torch.Tensor): 节点坐标 [num_nodes, 3]
            batch (torch.LongTensor): 批处理索引 [num_nodes]
            esm_attention (torch.Tensor): ESM注意力分数 [num_nodes, 1]

        返回:
            node_embeddings (torch.Tensor): 节点级嵌入 [num_nodes, output_dim]
            graph_embedding (torch.Tensor): 图级嵌入 [batch_size, output_dim]
            residue_scores (torch.Tensor): 残基重要性分数 [num_nodes, 1]
        """
        # 处理批索引
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # =============== 特征预处理 ===============

        # 物化属性标准化
        x = self.property_normalizer(x)

        # 计算相对位置编码（如果启用）
        if self.use_pos_encoding and pos is not None:
            pos_encoding = self.pos_encoder(pos, edge_index, batch)
            # 拼接位置编码和节点特征
            x = torch.cat([x, pos_encoding], dim=-1)

        # 节点特征初始编码
        h = self.node_encoder(x)


        if edge_attr is not None:
            if self.use_heterogeneous_edges and edge_type is not None:
                # 针对不同类型边，分别处理特征
                edge_features_list = []
                for t in range(self.edge_types):
                    mask = (edge_type == t)
                    if mask.sum() > 0:
                        # 边特征变换
                        edge_feats = self.edge_encoders[t](edge_attr[mask])
                        # 获取边类型嵌入
                        edge_type_emb = self.edge_type_embeddings(torch.tensor([t], device=edge_attr.device))
                        # 拼接边类型嵌入和特征
                        edge_feats = torch.cat([
                            edge_feats,
                            edge_type_emb.expand(edge_feats.size(0), -1)
                        ], dim=-1)
                        edge_features_list.append((mask, edge_feats))

                # 初始化完整边特征张量
                edge_features = torch.zeros(
                    edge_attr.size(0),
                    self.hidden_dim // 2,
                    device=edge_attr.device
                )

                # 填入各类型边的特征
                for mask, feats in edge_features_list:
                    edge_features[mask] = feats
            else:
                # 统一处理所有边
                # 将边类型转为独热向量
                if edge_type is not None:
                    edge_type_onehot = F.one_hot(edge_type, num_classes=self.edge_types).float()
                    # 拼接边特征和类型
                    edge_input = torch.cat([edge_attr, edge_type_onehot], dim=-1)
                else:
                    edge_input = edge_attr

                edge_features = self.edge_encoder(edge_input)
        else:
            edge_features = None

        # 应用边修剪（如果启用）
        if self.use_edge_pruning and edge_features is not None:
            edge_mask, _ = self.edge_pruning(edge_features)
            edge_features = edge_features * edge_mask
        # =============== 图卷积处理 ===============

        # 存储每一层的特征
        layer_features = []

        # 当前边特征
        current_edge_features = edge_features

        # 通过图卷积层
        for i, (conv, edge_updater) in enumerate(zip(self.convs, self.edge_updaters)):
            if self.use_heterogeneous_edges:
                # 使用异质边注意力层
                h_new = conv(h, edge_index, edge_type, current_edge_features)
            else:
                # 使用标准GATv2层
                h_new = conv(h, edge_index, current_edge_features)

            # 边特征更新
            if current_edge_features is not None:
                if self.use_heterogeneous_edges:
                    current_edge_features = edge_updater(h, edge_index, edge_type, current_edge_features)
                else:
                    current_edge_features = edge_updater(h, edge_index, current_edge_features)

            # 存储中间层特征
            if i < len(self.convs) - 1:
                layer_features.append(h_new)

            h = h_new

        # 最后一层特征
        layer_features.append(h)

        # =============== 多尺度特征聚合 ===============

        # 多尺度特征聚合
        if len(layer_features) > 1:
            # 拼接所有层特征
            multi_scale_features = torch.cat(layer_features, dim=-1)

            # 计算层特征重要性
            layer_weights = self.layer_attention(multi_scale_features)

            # 加权聚合各层特征
            weighted_features = 0
            start_idx = 0
            for i, feat in enumerate(layer_features):
                feat_dim = feat.size(-1)
                weight = layer_weights[:, i].unsqueeze(-1)
                weighted_features += weight * feat
                start_idx += feat_dim

            # 融合特征
            h = self.feature_fusion(multi_scale_features) + weighted_features

        # =============== 节点表示和图级表示 ===============

        # 节点级嵌入
        node_embeddings = h

        # 计算残基重要性评分
        residue_scores = self.residue_importance(node_embeddings)

        # ESM注意力引导（如果提供）
        if self.esm_guidance and esm_attention is not None:
            # 融合ESM注意力和模型学习的注意力
            esm_weights = self.esm_attention_guide(node_embeddings)
            guided_attention = esm_weights * esm_attention + (1 - esm_weights) * residue_scores

            # 用引导后的注意力生成图级表示
            graph_embedding = self.readout(node_embeddings, batch, node_weights=guided_attention)
        else:
            # 标准读出
            graph_embedding = self.readout(node_embeddings, batch)

        return node_embeddings, graph_embedding, residue_scores

    @torch.no_grad()
    def encode_proteins(self, protein_graphs, esm_attention=None, return_importance=False):
        """
        批量编码蛋白质图谱

        参数:
            protein_graphs (List[Data]): 蛋白质图数据对象列表
            esm_attention (List[torch.Tensor], optional): 每个蛋白质的ESM注意力分数
            return_importance (bool): 是否返回残基重要性

        返回:
            node_embeddings_list (List[torch.Tensor]): 每个蛋白质的节点嵌入
            graph_embeddings (torch.Tensor): 图嵌入 [batch_size, output_dim]
            residue_scores_list (List[torch.Tensor], optional): 残基重要性分数
        """
        device = next(self.parameters()).device

        # 高效批处理
        batch_data = self._batch_protein_graphs(protein_graphs, device)

        # 合并ESM注意力（如果提供）
        if esm_attention is not None:
            batch_esm_attention = []
            ptr = batch_data['ptr']
            for i, attn in enumerate(esm_attention):
                batch_esm_attention.append(attn.to(device))
            batch_esm_attention = torch.cat(batch_esm_attention, dim=0)
        else:
            batch_esm_attention = None

        # 编码
        node_embeddings, graph_embedding, residue_scores = self.forward(
            batch_data['x'],
            batch_data['edge_index'],
            batch_data.get('edge_attr'),
            batch_data.get('edge_type'),
            batch_data.get('pos'),
            batch_data['batch'],
            batch_esm_attention
        )

        # 分割结果
        node_embeddings_list, residue_scores_list = self._split_batch_outputs(
            node_embeddings,
            residue_scores,
            batch_data['ptr']
        )

        if return_importance:
            return node_embeddings_list, graph_embedding, residue_scores_list
        else:
            return node_embeddings_list, graph_embedding

    def _batch_protein_graphs(self, protein_graphs, device):
        """高效的批处理方法"""
        # 预分配必要容量
        total_nodes = sum(graph.num_nodes for graph in protein_graphs)
        total_edges = sum(graph.num_edges for graph in protein_graphs)

        # 初始化批数据结构
        x = torch.zeros((total_nodes, self.node_input_dim), device=device)
        edge_index = torch.zeros((2, total_edges), dtype=torch.long, device=device)
        batch = torch.zeros(total_nodes, dtype=torch.long, device=device)
        ptr = torch.zeros(len(protein_graphs) + 1, dtype=torch.long, device=device)

        # 可选数据
        edge_attr = torch.zeros((total_edges, self.edge_input_dim), device=device) if hasattr(protein_graphs[0],
                                                                                              'edge_attr') else None
        edge_type = torch.zeros(total_edges, dtype=torch.long, device=device) if hasattr(protein_graphs[0],
                                                                                         'edge_type') else None
        pos = torch.zeros((total_nodes, 3), device=device) if hasattr(protein_graphs[0], 'pos') else None

        # 填充数据
        node_offset = 0
        edge_offset = 0

        for i, graph in enumerate(protein_graphs):
            num_nodes = graph.num_nodes
            num_edges = graph.num_edges
            ptr[i] = node_offset

            # 节点特征
            x[node_offset:node_offset + num_nodes] = graph.x.to(device)
            # 批索引
            batch[node_offset:node_offset + num_nodes] = i

            # 边索引
            edge_index_i = graph.edge_index.clone().to(device)
            edge_index_i[0] += node_offset
            edge_index_i[1] += node_offset
            edge_index[:, edge_offset:edge_offset + num_edges] = edge_index_i

            # 其他可选数据
            if edge_attr is not None and hasattr(graph, 'edge_attr'):
                edge_attr[edge_offset:edge_offset + num_edges] = graph.edge_attr.to(device)

            if edge_type is not None and hasattr(graph, 'edge_type'):
                edge_type[edge_offset:edge_offset + num_edges] = graph.edge_type.to(device)

            if pos is not None and hasattr(graph, 'pos'):
                pos[node_offset:node_offset + num_nodes] = graph.pos.to(device)

            node_offset += num_nodes
            edge_offset += num_edges

        # 设置最后一个指针
        ptr[-1] = node_offset

        # 构建返回字典
        batch_data = {
            'x': x,
            'edge_index': edge_index[:, :edge_offset],
            'batch': batch,
            'ptr': ptr
        }

        if edge_attr is not None:
            batch_data['edge_attr'] = edge_attr[:edge_offset]

        if edge_type is not None:
            batch_data['edge_type'] = edge_type[:edge_offset]

        if pos is not None:
            batch_data['pos'] = pos

        return batch_data

    def _split_batch_outputs(self, node_embeddings, residue_scores, ptr):
        """将批处理输出分割为每个蛋白质的结果"""
        node_embeddings_list = []
        residue_scores_list = []

        for i in range(len(ptr) - 1):
            start_idx = ptr[i].item()
            end_idx = ptr[i + 1].item()

            node_embeddings_list.append(node_embeddings[start_idx:end_idx])
            residue_scores_list.append(residue_scores[start_idx:end_idx])

        return node_embeddings_list, residue_scores_list


# 新增: 相对位置编码器
class RelativePositionEncoder(nn.Module):
    """
    相对位置编码器

    计算残基间的相对位置和方向信息，增强模型的空间感知能力
    """

    def __init__(self, hidden_dim, max_seq_dist=32, dropout=0.1):
        super(RelativePositionEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_dist = max_seq_dist

        # 序列距离编码
        self.seq_distance_embedding = nn.Embedding(max_seq_dist + 1, hidden_dim // 2)

        # 空间向量编码
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pos, edge_index, batch):
        """
        计算相对位置编码

        参数:
            pos (torch.Tensor): 节点坐标 [num_nodes, 3]
            edge_index (torch.LongTensor): 图的边连接 [2, num_edges]
            batch (torch.LongTensor): 批处理索引 [num_nodes]

        返回:
            position_embedding (torch.Tensor): 位置编码 [num_nodes, hidden_dim]
        """
        device = pos.device
        num_nodes = pos.size(0)

        # 初始化位置编码
        position_embeddings = torch.zeros(num_nodes, self.hidden_dim, device=device)

        # 对每个节点计算相对位置编码
        src, dst = edge_index

        # 计算CA-CA向量
        rel_pos = pos[dst] - pos[src]  # [num_edges, 3]

        # 计算向量模长
        dist = torch.norm(rel_pos, dim=1, keepdim=True)  # [num_edges, 1]

        # 归一化方向向量
        rel_pos_norm = rel_pos / (dist + 1e-6)

        # 编码空间方向
        spatial_code = self.spatial_encoder(rel_pos_norm)  # [num_edges, hidden_dim//2]

        # 构建邻接结构，用于聚合边特征到节点
        for i in range(edge_index.size(1)):
            node_idx = dst[i]
            position_embeddings[node_idx] += spatial_code[i]

        # 平均聚合
        node_degrees = torch.zeros(num_nodes, device=device)
        for node_idx in dst:
            node_degrees[node_idx] += 1

        # 避免除零
        node_degrees = torch.clamp(node_degrees, min=1)

        # 归一化
        position_embeddings = position_embeddings / node_degrees.unsqueeze(1)

        return position_embeddings


# 新增: 物化属性标准化器
class PropertyNormalizer(nn.Module):
    """物化属性标准化器，通过分箱和归一化处理氨基酸特性"""

    def __init__(self, num_bins=10):
        super(PropertyNormalizer, self).__init__()
        self.num_bins = num_bins

        # 记录各属性的统计信息，用于归一化
        self.register_buffer('hydropathy_range', torch.tensor([-4.5, 4.5]))
        self.register_buffer('charge_range', torch.tensor([-1.0, 1.0]))
        self.register_buffer('weight_range', torch.tensor([75.0, 204.0]))

    def forward(self, x):
        """
        标准化蛋白质物化属性

        参数:
            x (torch.Tensor): 节点特征 [num_nodes, node_input_dim]

        返回:
            x_normalized (torch.Tensor): 标准化后的节点特征
        """
        # 假设特征的前几列分别是: hydropathy, charge, molecular_weight等
        # 根据实际数据格式调整索引
        batch_size = x.size(0)
        device = x.device

        # 深拷贝输入，避免修改原始数据
        x_normalized = x.clone()

        # 疏水性分箱归一化 (假设在索引0)
        if x.size(1) > 0:
            hydropathy = x[:, 0]
            x_normalized[:, 0] = self._bin_and_normalize(
                hydropathy,
                self.hydropathy_range[0],
                self.hydropathy_range[1]
            )

        # 电荷分箱归一化 (假设在索引1)
        if x.size(1) > 1:
            charge = x[:, 1]
            x_normalized[:, 1] = self._bin_and_normalize(
                charge,
                self.charge_range[0],
                self.charge_range[1]
            )

        # 分子量分箱归一化 (假设在索引3)
        if x.size(1) > 3:
            weight = x[:, 3]
            x_normalized[:, 3] = self._bin_and_normalize(
                weight,
                self.weight_range[0],
                self.weight_range[1]
            )

        return x_normalized

    def _bin_and_normalize(self, values, min_val, max_val):
        """将连续值分箱并归一化"""
        # 裁剪到范围
        values_clipped = torch.clamp(values, min=min_val, max=max_val)

        # 归一化到[0,1]
        values_norm = (values_clipped - min_val) / (max_val - min_val)

        # 分箱 (可选)
        values_binned = torch.floor(values_norm * self.num_bins) / self.num_bins

        return values_binned


# 针对与ESM编码器对齐的潜空间映射器
class ProteinLatentMapper(nn.Module):
    """
    增强型蛋白质图嵌入潜空间映射器

    使用对比学习和多层残差网络将GATv2嵌入映射到与ESM兼容的空间

    参数:
        input_dim (int): 输入维度（GATv2嵌入维度）
        latent_dim (int): 潜在空间维度（ESM兼容维度）
        hidden_dim (int, optional): 隐藏层维度
        dropout (float, optional): Dropout率
    """

    def __init__(
            self,
            input_dim,
            latent_dim,
            hidden_dim=None,
            dropout=0.1
    ):
        super(ProteinLatentMapper, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or max(input_dim, latent_dim) * 2

        # 映射网络 - 使用5层ResNet结构
        self.mapper = nn.Sequential(
            # 输入层
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            # 残差块1
            ResidualBlock(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            )),
            nn.GELU(),

            # 残差块2
            ResidualBlock(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            )),
            nn.GELU(),

            # 输出层
            nn.Linear(self.hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # 特征校准层 - 用于微调输出与ESM特征对齐
        self.calibration = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )

    def forward(self, x, normalize=True):
        """
        将图嵌入映射到潜在空间

        参数:
            x (torch.Tensor): 输入图嵌入 [batch_size, input_dim]
            normalize (bool): 是否L2归一化输出

        返回:
            latent (torch.Tensor): 潜在空间表示 [batch_size, latent_dim]
        """
        # 通过映射网络
        latent = self.mapper(x)

        # 校准（微调与ESM特征对齐）
        latent = self.calibration(latent)

        # 可选的L2归一化
        if normalize:
            latent = F.normalize(latent, p=2, dim=-1)

        return latent

class ResidualBlock(nn.Module):
    """残差连接块，用于深层网络稳定训练"""

    def __init__(self, module):
        super(ResidualBlock, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class CrossModalContrastiveHead(nn.Module):
    """跨模态对比学习头，用于对齐GATv2嵌入和ESM嵌入"""

    def __init__(self, embedding_dim, temperature=0.07):
        super(CrossModalContrastiveHead, self).__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, graph_embeddings, esm_embeddings):
        """
        计算跨模态对比损失

        参数:
            graph_embeddings (torch.Tensor): 图嵌入 [batch_size, embedding_dim]
            esm_embeddings (torch.Tensor): ESM嵌入 [batch_size, embedding_dim]

        返回:
            loss (torch.Tensor): 对比损失
            similarity (torch.Tensor): 模态间相似度矩阵
        """
        # 投影嵌入到统一空间
        graph_proj = self.projection(graph_embeddings)
        esm_proj = self.projection(esm_embeddings)

        # 归一化嵌入
        graph_proj = F.normalize(graph_proj, dim=-1)
        esm_proj = F.normalize(esm_proj, dim=-1)

        # 计算余弦相似度
        similarity = torch.mm(graph_proj, esm_proj.t()) / self.temperature

        # InfoNCE损失
        labels = torch.arange(similarity.size(0), device=similarity.device)
        loss_graph_to_esm = F.cross_entropy(similarity, labels)
        loss_esm_to_graph = F.cross_entropy(similarity.t(), labels)

        # 双向对比损失
        loss = (loss_graph_to_esm + loss_esm_to_graph) / 2

        return loss, similarity

class ESMGuidanceModule(nn.Module):
    """ESM注意力引导模块，用于整合ESM的序列进化信息"""

    def __init__(self, input_dim, hidden_dim=None):
        super(ESMGuidanceModule, self).__init__()
        self.hidden_dim = hidden_dim or input_dim

        self.attention_processor = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_embeddings, esm_attention):
        """
        融合GATv2注意力和ESM注意力

        参数:
            node_embeddings (torch.Tensor): 节点嵌入 [num_nodes, input_dim]
            esm_attention (torch.Tensor): ESM注意力分数 [num_nodes, 1]

        返回:
            guided_attention (torch.Tensor): 引导后的注意力 [num_nodes, 1]
        """
        # 计算自适应混合权重
        alpha = self.attention_processor(node_embeddings)

        # 线性组合
        guided_attention = alpha * esm_attention + (1 - alpha) * torch.sigmoid(
            node_embeddings.mean(dim=-1, keepdim=True)
        )

        return guided_attention