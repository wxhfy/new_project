#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
抗菌肽设计与生成的蛋白质图神经网络框架

该模块实现了基于GATv2的蛋白质图编码器，用于从蛋白质结构中提取高质量的
嵌入表示，并与ESM序列编码器结合形成双编码器-单解码器架构，专为抗菌肽
设计任务优化。

作者: wxhfy
日期: 2025-03-29
版本: 0.1.0
"""

# 版本信息
__version__ = '0.1.0'
__author__ = 'wxhfy'

# 导入模型组件
from .gat_models import (
    ProteinGATv2Encoder,
    ProteinLatentMapper
)

# 导入基础层
from .layers import (
    GATv2ConvLayer,
    MLPLayer,
    EdgeTypeEncoder,
    StructureAwareAttention
)

# 导入读出机制
from .readout import (
    AttentiveReadout,
    MultiLevelPooling,
    HierarchicalReadout,
)

# 定义公开接口
__all__ = [
    # 模型
    'ProteinGATv2Encoder',
    'ProteinLatentMapper',

    # 层
    'GATv2ConvLayer',
    'MLPLayer',
    'EdgeTypeEncoder',
    'StructureAwareAttention',

    # 读出机制
    'AttentiveReadout',
    'MultiLevelPooling',
    'HierarchicalReadout',

    # 便捷函数
    'create_default_amp_encoder',
    'create_fusion_encoder'
]


def create_default_amp_encoder(node_dim, edge_dim, output_dim=128):
    """
    创建适用于抗菌肽(AMPs)设计的默认GATv2编码器

    参数:
        node_dim (int): 节点特征维度
        edge_dim (int): 边特征维度
        output_dim (int, optional): 输出嵌入维度，默认128

    返回:
        encoder (ProteinGATv2Encoder): 预配置的GATv2编码器实例
    """
    return ProteinGATv2Encoder(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=output_dim,
        output_dim=output_dim,
        num_layers=3,
        num_heads=4,
        edge_types=2,
        dropout=0.2,
        residual=True,
        use_layer_norm=True,
        readout_mode='attn',
        activation='gelu'
    )


def create_fusion_encoder(node_dim, edge_dim, output_dim=768, readout='hierarchical'):
    """
    创建用于与ESM序列编码器融合的GATv2编码器

    配置为输出与ESM兼容的嵌入维度，并使用更复杂的读出机制

    参数:
        node_dim (int): 节点特征维度
        edge_dim (int): 边特征维度
        output_dim (int, optional): 输出嵌入维度，默认768(与ESM兼容)
        readout (str, optional): 读出机制类型 ('attn', 'multi', 'hierarchical', 'focal')

    返回:
        encoder (ProteinGATv2Encoder): 预配置的GATv2编码器实例
        mapper (ProteinLatentMapper): 预配置的潜空间映射器实例
    """
    # 创建编码器
    gat_hidden = min(512, output_dim)
    encoder = ProteinGATv2Encoder(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=gat_hidden,
        output_dim=gat_hidden,
        num_layers=4,
        num_heads=8,
        edge_types=2,
        dropout=0.1,
        residual=True,
        use_layer_norm=True,
        readout_mode=readout if readout in ['attn', 'mean', 'sum', 'max'] else 'attn',
        activation='gelu'
    )

    # 创建潜空间映射器
    mapper = ProteinLatentMapper(
        input_dim=gat_hidden,
        latent_dim=output_dim,
        hidden_dim=output_dim * 2,
        num_layers=2,
        dropout=0.1,
        activation='gelu'
    )

    # 替换读出层(如果使用高级读出机制)
    if readout == 'hierarchical':
        encoder.readout = HierarchicalReadout(
            in_features=gat_hidden,
            hidden_dim=gat_hidden,
            num_heads=8,
            dropout=0.1
        )
    elif readout == 'multi':
        encoder.readout = MultiLevelPooling(
            in_features=gat_hidden,
            hidden_dim=gat_hidden,
            dropout=0.1,
            use_gate=True
        )
    elif readout == 'focal':
        encoder.readout = FocalReadout(
            in_features=gat_hidden,
            hidden_dim=gat_hidden,
            dropout=0.1
        )

    return encoder, mapper