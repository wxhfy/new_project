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
            node_input_dim,  # 现在是35
            edge_input_dim,  # 现在是8
            hidden_dim=128,
            output_dim=128,
            num_layers=3,
            num_heads=4,
            edge_types=4,  # 修改为4种边类型
            dropout=0.2,
            use_pos_encoding=True,
            use_heterogeneous_edges=True,
            use_edge_pruning=False,
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

        # 相对位置编码
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

            # 添加这段代码：为每种边类型创建单独的编码器
            self.edge_encoders = nn.ModuleList([
                nn.Linear(edge_input_dim, hidden_dim // 2) for _ in range(edge_types)
            ])

            # 保留原有的通用边特征处理器
            self.edge_processor = EdgeProcessor(
                edge_dim=edge_input_dim,
                hidden_dim=hidden_dim,
                edge_types=edge_types,
                dropout=dropout,
                use_heterogeneous=self.use_heterogeneous_edges
            )

            # 新增：备用的统一边编码器，避免在非异质边模式下尝试访问时出错
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_input_dim + edge_types, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                self.activation
            )
        else:
            # 统一边特征处理
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_input_dim, hidden_dim // 2),
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

    # 修改图编码器的前向传播函数，支持ESM指导
    def forward(
            self,
            x,
            edge_index,
            edge_attr=None,
            edge_type=None,
            batch=None,
            esm_attention=None  # 新增：ESM注意力参数
    ):
        """
        前向传递，支持ESM序列注意力指导

        参数:
            x (torch.Tensor): 节点特征 [num_nodes, node_input_dim]
            edge_index (torch.LongTensor): 图的边连接 [2, num_edges]
            edge_attr (torch.Tensor): 边特征 [num_edges, edge_input_dim]
            edge_type (torch.LongTensor): 边类型索引 [num_edges]
            batch (torch.LongTensor): 批处理索引 [num_nodes]
            esm_attention (torch.Tensor): ESM注意力分数 [num_nodes, 1]，用于指导图注意力

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

        # 节点特征初始编码
        h = self.node_encoder(x)

        # 处理边特征
        # 在forward方法中:
        if edge_attr is not None:
            edge_features = self.edge_processor(edge_attr, edge_type)
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
            # 如果是最后一层且提供了ESM注意力，则用于指导
            if i == len(self.convs) - 1 and self.esm_guidance and esm_attention is not None:
                # 集成ESM注意力到图结构注意力机制
                if self.use_heterogeneous_edges:
                    h_new = conv(h, edge_index, edge_type, current_edge_features, esm_attention=esm_attention)
                else:
                    h_new = conv(h, edge_index, current_edge_features, esm_attention=esm_attention)
            else:
                # 常规前向传播
                if self.use_heterogeneous_edges:
                    h_new = conv(h, edge_index, edge_type, current_edge_features)
                else:
                    h_new = conv(h, edge_index, current_edge_features)

            # 边特征更新
            if current_edge_features is not None:
                if self.use_heterogeneous_edges and edge_type is not None:
                    current_edge_features = edge_updater(
                        h, edge_index, current_edge_features, edge_type=edge_type
                    )
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
            for i, feat in enumerate(layer_features):
                weight = layer_weights[:, i].unsqueeze(-1)
                weighted_features += weight * feat

            # 特征融合
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
            graph_embedding = self.readout(node_embeddings, batch, node_weights=residue_scores)

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
        """高效的批处理方法，确保正确处理35维节点特征和8维边特征"""
        # 预分配必要容量
        total_nodes = sum(graph.num_nodes for graph in protein_graphs)
        total_edges = sum(graph.num_edges for graph in protein_graphs)

        # 初始化批数据结构 - 确保维度正确
        x = torch.zeros((total_nodes, self.node_input_dim), device=device)  # 35维
        edge_index = torch.zeros((2, total_edges), dtype=torch.long, device=device)
        batch = torch.zeros(total_nodes, dtype=torch.long, device=device)
        ptr = torch.zeros(len(protein_graphs) + 1, dtype=torch.long, device=device)

        # 可选数据
        edge_attr = torch.zeros((total_edges, self.edge_input_dim), device=device) if hasattr(protein_graphs[0],
                                                                                              'edge_attr') else None
        edge_type = torch.zeros(total_edges, dtype=torch.long, device=device) if hasattr(protein_graphs[0],
                                                                                         'edge_type') else None

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


class EdgeFeatureProcessor(nn.Module):
    """
    处理8维边特征的专用模块

    特征维度明细：
    - 相互作用类型（4维one-hot编码）
    - 空间距离（1维）
    - 相互作用强度（1维）
    - 方向向量（2维）
    """

    def __init__(self, edge_dim=8, hidden_dim=64, dropout=0.1):
        super(EdgeFeatureProcessor, self).__init__()

        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # 相互作用类型编码器
        self.interaction_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )

        # 距离和强度编码器
        self.metric_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )

        # 方向编码器
        self.direction_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )

        # 输出变换
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_dim * 3 // 4, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout)
        )

    def forward(self, edge_attr):
        """
        处理8维边特征

        参数:
            edge_attr (torch.Tensor): 边特征 [num_edges, 8]

        返回:
            edge_features (torch.Tensor): 处理后的边特征 [num_edges, hidden_dim//2]
        """
        # 检查输入维度
        if edge_attr.size(1) < self.edge_dim:
            # 处理维度不足的情况
            padding = torch.zeros(edge_attr.size(0), self.edge_dim - edge_attr.size(1),
                                  device=edge_attr.device)
            edge_attr = torch.cat([edge_attr, padding], dim=1)
        elif edge_attr.size(1) > self.edge_dim:
            # 维度过多时截断
            edge_attr = edge_attr[:, :self.edge_dim]

        # 确保至少有4+2+2维特征可用
        min_required = 8
        if edge_attr.size(1) < min_required:
            # 使用简单线性变换
            return nn.Linear(edge_attr.size(1), self.hidden_dim // 2).to(edge_attr.device)(edge_attr)

        # 分离不同类型的特征
        interaction_type = edge_attr[:, :4]  # 相互作用类型
        distance_strength = edge_attr[:, 4:6]  # 空间距离和强度
        direction = edge_attr[:, 6:8]  # 方向向量

        # 编码各部分特征
        interaction_features = self.interaction_encoder(interaction_type)
        metric_features = self.metric_encoder(distance_strength)
        direction_features = self.direction_encoder(direction)

        # 合并所有特征
        combined = torch.cat([interaction_features, metric_features, direction_features], dim=-1)

        # 最终变换
        edge_features = self.output_transform(combined)

        return edge_features


class EdgeProcessor(nn.Module):
    """统一的边处理接口"""

    def __init__(self, edge_dim, hidden_dim, edge_types, dropout=0.1, use_heterogeneous=True):
        super(EdgeProcessor, self).__init__()
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.edge_types = edge_types
        self.use_heterogeneous = use_heterogeneous

        if use_heterogeneous:
            # 异质边处理
            self.edge_type_embeddings = nn.Embedding(edge_types, hidden_dim // 4)
            self.edge_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(edge_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.GELU()
                ) for _ in range(edge_types)
            ])
        else:
            # 同质边处理
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU()
            )

    def forward(self, edge_attr, edge_type=None):
        """处理边特征"""
        if self.use_heterogeneous and edge_type is not None:
            # 处理异质边
            result = torch.zeros(edge_attr.size(0), self.hidden_dim // 2, device=edge_attr.device)

            for t in range(self.edge_types):
                mask = (edge_type == t)
                if not mask.any():
                    continue

                edge_feats = self.edge_encoders[t](edge_attr[mask])
                result[mask] = edge_feats

            return result
        else:
            # 处理同质边
            return self.edge_encoder(edge_attr)

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


class PropertyNormalizer(nn.Module):
    """针对35维蛋白质特征向量的标准化器"""

    def __init__(self, num_bins=10):
        super(PropertyNormalizer, self).__init__()
        self.num_bins = num_bins

        # 为各类属性设置合适的归一化范围
        self.register_buffer('hydropathy_range', torch.tensor([-4.5, 4.5]))
        self.register_buffer('charge_range', torch.tensor([-2.0, 2.0]))
        self.register_buffer('weight_range', torch.tensor([75.0, 204.0]))
        self.register_buffer('volume_range', torch.tensor([60.0, 230.0]))
        self.register_buffer('sasa_range', torch.tensor([0.0, 1.0]))
        self.register_buffer('plddt_range', torch.tensor([0.0, 100.0]))

    def forward(self, x):
        """
        标准化蛋白质理化和结构特性

        参数:
            x (torch.Tensor): 节点特征 [num_nodes, 35]

        返回:
            x_normalized (torch.Tensor): 标准化后的节点特征 [num_nodes, 35]
        """
        # 深拷贝输入，避免修改原始数据
        x_normalized = x.clone()

        # 根据特征布局进行标准化，特征维度分布：
        # [0-19]: BLOSUM62编码 - 无需归一化，已经是标准化分数
        # [20-22]: 空间坐标 - 假设已经归一化
        # [23]: 疏水性
        x_normalized[:, 23] = self._bin_and_normalize(x[:, 23], self.hydropathy_range[0], self.hydropathy_range[1])

        # [24]: 电荷
        x_normalized[:, 24] = self._bin_and_normalize(x[:, 24], self.charge_range[0], self.charge_range[1])

        # [25]: 分子量
        x_normalized[:, 25] = self._bin_and_normalize(x[:, 25], self.weight_range[0], self.weight_range[1])

        # [26]: 体积
        x_normalized[:, 26] = self._bin_and_normalize(x[:, 26], self.volume_range[0], self.volume_range[1])

        # [27-28]: 柔性和芳香性 - 这些已经是归一化值

        # [29-31]: 二级结构编码 - 已经是one-hot编码，无需归一化

        # [32]: 溶剂可及性
        x_normalized[:, 32] = self._bin_and_normalize(x[:, 32], self.sasa_range[0], self.sasa_range[1])

        # [33]: 侧链柔性 - 已经归一化

        # [34]: pLDDT质量评分
        x_normalized[:, 34] = self._bin_and_normalize(x[:, 34], self.plddt_range[0], self.plddt_range[1])

        return x_normalized

    def _bin_and_normalize(self, values, min_val, max_val):
        """将连续值分箱并归一化"""
        # 裁剪到范围
        values_clipped = torch.clamp(values, min=min_val, max=max_val)

        # 归一化到[0,1]
        values_norm = (values_clipped - min_val) / (max_val - min_val + 1e-6)

        # 分箱 (可选)
        if self.num_bins > 1:
            values_binned = torch.floor(values_norm * self.num_bins) / self.num_bins
            return values_binned
        else:
            return values_norm

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