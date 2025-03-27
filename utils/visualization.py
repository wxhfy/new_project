#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图嵌入系统 - 可视化工具

为蛋白质图嵌入提供可视化功能，包括嵌入空间分析、
注意力权重可视化和训练过程监控。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torch_geometric.utils import to_networkx
import networkx as nx


def plot_training_curves(train_losses, val_losses, train_metrics=None, val_metrics=None,
                         save_path=None):
    """
    绘制训练和验证曲线

    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_metrics: 训练指标列表 (可选)
        val_metrics: 验证指标列表 (可选)
        save_path: 保存路径 (可选)
    """
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    # 如果有指标，绘制指标曲线
    if train_metrics is not None and val_metrics is not None:
        plt.subplot(1, 2, 2)
        plt.plot(train_metrics, label='Training Metric')
        plt.plot(val_metrics, label='Validation Metric')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.legend()
        plt.title('Metric Curves')

    plt.tight_layout()

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def visualize_embeddings(embeddings, labels=None, method='tsne', save_path=None):
    """
    可视化嵌入空间

    参数:
        embeddings: 嵌入向量 [n_samples, dim]
        labels: 标签 (可选)
        method: 降维方法 ('tsne' 或 'pca')
        save_path: 保存路径 (可选)
    """
    # 将Tensor转换为NumPy数组
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # pca
        reducer = PCA(n_components=2)

    embeddings_2d = reducer.fit_transform(embeddings)

    # 可视化
    plt.figure(figsize=(10, 8))

    if labels is not None:
        # 将Tensor转换为NumPy数组
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # 处理分类和回归情况
        if len(labels.shape) == 1 or labels.shape[1] == 1:  # 标量标签
            if np.issubdtype(labels.dtype, np.number) and len(np.unique(labels)) > 10:
                # 回归情况
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels,
                                     cmap='viridis', alpha=0.8)
                plt.colorbar(scatter, label='Target Value')
            else:
                # 分类情况
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    idx = labels == label
                    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f'Class {label}', alpha=0.7)
                plt.legend()
        else:  # 多标签情况
            # 使用第一个标签进行可视化
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels[:, 0],
                                 cmap='viridis', alpha=0.8)
            plt.colorbar(scatter, label='First Target')
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

    plt.title(f'Embedding Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()

    return embeddings_2d


def visualize_protein_graph(data, node_weights=None, edge_weights=None, save_path=None):
    """
    可视化蛋白质图结构和注意力权重

    参数:
        data: PyG的Data对象
        node_weights: 节点权重 (可选)
        edge_weights: 边权重 (可选)
        save_path: 保存路径 (可选)
    """
    # 将PyG图转换为NetworkX图
    G = to_networkx(data, to_undirected=True)

    plt.figure(figsize=(10, 8))

    # 获取节点坐标 (如果有)
    pos = None
    if hasattr(data, 'pos') and data.pos is not None:
        pos_tensor = data.pos.detach().cpu().numpy()
        # 只使用前两维作为坐标
        if pos_tensor.shape[1] >= 2:
            pos = {i: (pos_tensor[i, 0], pos_tensor[i, 1]) for i in range(len(G.nodes))}

    # 如果没有坐标，使用布局算法
    if pos is None:
        pos = nx.spring_layout(G)

    # 节点权重
    node_color = '#1f78b4'  # 默认节点颜色
    node_cmap = None

    if node_weights is not None:
        # 将Tensor转换为NumPy数组
        if isinstance(node_weights, torch.Tensor):
            node_weights = node_weights.detach().cpu().numpy()
        node_color = node_weights
        node_cmap = 'viridis'

    # 边权重
    edge_color = 'grey'  # 默认边颜色
    edge_cmap = None
    edge_width = 1.0

    if edge_weights is not None:
        # 将Tensor转换为NumPy数组
        if isinstance(edge_weights, torch.Tensor):
            edge_weights = edge_weights.detach().cpu().numpy()
        edge_color = edge_weights
        edge_cmap = 'plasma'
        # 归一化边宽度
        min_width = 1.0
        max_width = 5.0
        if len(edge_weights) > 0:
            edge_width = min_width + (max_width - min_width) * (edge_weights - np.min(edge_weights)) / (np.max(edge_weights) - np.min(edge_weights) + 1e-6)

    # 绘制图
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=node_cmap, alpha=0.8)
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width, edge_cmap=edge_cmap, alpha=0.5)

    # 添加节点标签
    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title('Protein Graph Visualization')
    plt.axis('off')

    # 添加颜色条
    if node_weights is not None:
        plt.colorbar(nodes, label='Node Weight')

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def plot_attention_heatmap(attention_weights, save_path=None):
    """
    绘制注意力权重热图

    参数:
        attention_weights: 注意力权重矩阵
        save_path: 保存路径 (可选)
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='viridis', cbar=True)
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Target Nodes')
    plt.ylabel('Source Nodes')

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()