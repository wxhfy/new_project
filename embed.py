#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图嵌入系统 - 模型评估与嵌入生成脚本

本脚本用于评估已训练模型的性能并生成嵌入向量，支持多种评估方法：
1. 嵌入质量评估（可视化、聚类分析）
2. 模型复杂度分析
3. 预测性能评估
4. 注意力机制解释
5. 模型消融研究

作者: 基于wxhfy的知识图谱处理工作扩展
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split

# 导入模型和工具
from models.gat_models import ProteinGATv2, ProteinGATv2WithPretraining
from utils.visualization import (visualize_embeddings, visualize_protein_graph,
                                 plot_attention_heatmap, plot_training_curves)


def parse_args():
    parser = argparse.ArgumentParser(description='蛋白质图嵌入评估脚本')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--task_type', type=str, default='node', choices=['node', 'edge', 'graph'],
                        help='任务类型')
    parser.add_argument('--eval_type', type=str, default='all',
                        choices=['embedding', 'complexity', 'ablation', 'attention', 'all'],
                        help='评估类型')
    parser.add_argument('--vis_method', type=str, default='tsne', choices=['tsne', 'pca'],
                        help='嵌入可视化方法')
    return parser.parse_args()


def load_dataset(data_path, task_type):
    """加载数据集"""
    # 这里需要根据你的实际数据格式进行修改
    from torch_geometric.datasets import TUDataset  # 替换为你的数据集
    try:
        dataset = torch.load(data_path)
        print(f"成功加载数据集: {data_path}")
        print(f"数据集大小: {len(dataset)}")
        return dataset
    except:
        print(f"无法加载数据集: {data_path}")
        return None


def load_model(model_path, task_type, device):
    """加载训练好的模型"""
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # 重建模型架构
        base_model = ProteinGATv2(
            in_channels=checkpoint['config']['in_channels'],
            hidden_channels=checkpoint['config']['hidden_channels'],
            out_channels=checkpoint['config']['out_channels'],
            num_layers=checkpoint['config']['num_layers'],
            heads=checkpoint['config']['heads']
        ).to(device)

        model = ProteinGATv2WithPretraining(
            gat_model=base_model,
            task_type=task_type,
            num_tasks=checkpoint['config']['num_tasks']
        ).to(device)

        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"成功加载模型: {model_path}")
        return model, checkpoint['config']
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None


def evaluate_embedding_quality(embeddings, labels=None, n_clusters=None):
    """评估嵌入质量"""
    results = {}

    # 如果没有指定聚类数，尝试从标签推断
    if n_clusters is None and labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if len(labels.shape) == 1 or labels.shape[1] == 1:
            n_clusters = len(np.unique(labels))

    # 默认聚类数
    if n_clusters is None or n_clusters < 2:
        n_clusters = min(5, len(embeddings) // 5)  # 默认值

    # 转换为NumPy数组
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # 1. K-Means聚类
    if len(embeddings) > n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # 2. 轮廓系数 (Silhouette Score)
        if len(np.unique(cluster_labels)) > 1:
            results['silhouette_score'] = silhouette_score(embeddings, cluster_labels)

            # 3. Calinski-Harabasz指数
            results['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)

    # 4. 如果提供标签，计算监督指标
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # 拆分数据用于简单分类器评估
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.3, random_state=42)

        # 使用简单线性模型评估嵌入的分类能力
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        try:
            clf.fit(X_train, y_train.ravel())
            y_pred = clf.predict(X_test)
            results['classification_accuracy'] = accuracy_score(y_test, y_pred)

            # 对于二分类问题，计算AUC
            if len(np.unique(labels)) == 2:
                y_score = clf.predict_proba(X_test)[:, 1]
                results['auc_score'] = roc_auc_score(y_test, y_score)
        except Exception as e:
            print(f"分类评估失败: {e}")

    return results


def analyze_model_complexity(model, sample_data):
    """分析模型复杂度"""
    results = {}

    # 1. 参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results['total_parameters'] = total_params
    results['trainable_parameters'] = trainable_params

    # 2. 推理时间
    device = next(model.parameters()).device
    sample_data = sample_data.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_data)

    # 测量时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_data)
    end_time = time.time()

    results['avg_inference_time'] = (end_time - start_time) / 10

    # 3. 内存使用 (需要torch.cuda.memory_stats())
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(sample_data)
        results['peak_memory'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    return results


def extract_attention_weights(model, data):
    """提取注意力权重"""
    # 这个函数需要根据实际模型架构调整
    # 假设我们可以从GATv2Conv层中获取注意力权重
    attention_weights = {}

    # 重置钩子
    hooks = []

    # 定义钩子函数来获取注意力权重
    def get_attention_hook(name):
        def hook(module, input, output):
            # 假设output是元组(tensor, (edge_index, attention))
            if isinstance(output, tuple) and len(output) == 2:
                attention_weights[name] = output[1][1].detach()

        return hook

    # 注册钩子
    for name, module in model.named_modules():
        if isinstance(module, model.gat_model.convs.__class__.__bases__[0]):
            for i, layer in enumerate(module):
                hooks.append(layer.register_forward_hook(get_attention_hook(f"layer_{i}")))

    # 执行前向传播
    model.eval()
    with torch.no_grad():
        _ = model(data)

    # 移除钩子
    for hook in hooks:
        hook.remove()

    return attention_weights


def perform_ablation_study(model_class, config, dataset, device):
    """进行消融实验"""
    results = {}

    # 1. 测试不同注意力头数
    for heads in [1, 4, 8]:
        config_copy = config.copy()
        config_copy['heads'] = heads

        # 创建模型
        model = model_class(
            in_channels=config_copy['in_channels'],
            hidden_channels=config_copy['hidden_channels'],
            out_channels=config_copy['out_channels'],
            num_layers=config_copy['num_layers'],
            heads=heads
        ).to(device)

        # 这里仅计算模型复杂度
        sample_data = dataset[0].to(device)
        complexity = analyze_model_complexity(model, sample_data)
        results[f"heads_{heads}"] = complexity

    # 2. 测试有无边特征
    for use_edge_attr in [True, False]:
        config_copy = config.copy()
        config_copy['use_edge_attr'] = use_edge_attr

        model = model_class(
            in_channels=config_copy['in_channels'],
            hidden_channels=config_copy['hidden_channels'],
            out_channels=config_copy['out_channels'],
            num_layers=config_copy['num_layers'],
            use_edge_attr=use_edge_attr
        ).to(device)

        sample_data = dataset[0].to(device)
        complexity = analyze_model_complexity(model, sample_data)
        results[f"use_edge_attr_{use_edge_attr}"] = complexity

    return results


def generate_embeddings(model, dataloader, device, level='graph'):
    """生成嵌入向量"""
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data)

            if level == 'graph':
                emb = output['graph_embedding']
                # 获取图标签 (如果有)
                if hasattr(data, 'y'):
                    label = data.y
                    labels.append(label)
            elif level == 'node':
                emb = output['node_embedding']
                # 获取节点标签 (如果有)
                if hasattr(data, 'y'):
                    label = data.y
                    labels.append(label)

            embeddings.append(emb.cpu())

    embeddings = torch.cat(embeddings, dim=0)

    if labels:
        labels = torch.cat(labels, dim=0)
        return embeddings, labels

    return embeddings, None


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device)

    # 加载数据集
    dataset = load_dataset(args.data_path, args.task_type)
    if dataset is None:
        return

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 加载模型
    model, config = load_model(args.model_path, args.task_type, device)
    if model is None:
        return

    # 评估类型
    eval_types = [args.eval_type] if args.eval_type != 'all' else [
        'embedding', 'complexity', 'ablation', 'attention'
    ]

    # 性能统计
    performance = {}

    # 1. 嵌入质量评估
    if 'embedding' in eval_types:
        print("\n--- 评估嵌入质量 ---")
        # 生成嵌入
        level = 'graph' if args.task_type == 'graph' else 'node'
        embeddings, labels = generate_embeddings(model, dataloader, device, level)

        # 评估嵌入质量
        embedding_metrics = evaluate_embedding_quality(embeddings, labels)
        performance['embedding_quality'] = embedding_metrics
        print("嵌入质量指标:")
        for k, v in embedding_metrics.items():
            print(f"  {k}: {v:.4f}")

        # 可视化嵌入
        print("可视化嵌入...")
        vis_path = os.path.join(args.output_dir, f"embedding_vis_{args.vis_method}.png")
        visualize_embeddings(
            embeddings,
            labels,
            method=args.vis_method,
            save_path=vis_path
        )
        print(f"嵌入可视化已保存至: {vis_path}")

    # 2. 模型复杂度分析
    if 'complexity' in eval_types:
        print("\n--- 分析模型复杂度 ---")
        sample_data = dataset[0].to(device)
        complexity_metrics = analyze_model_complexity(model, sample_data)
        performance['model_complexity'] = complexity_metrics

        print("模型复杂度指标:")
        print(f"  总参数量: {complexity_metrics['total_parameters']:,}")
        print(f"  可训练参数量: {complexity_metrics['trainable_parameters']:,}")
        print(f"  平均推理时间: {complexity_metrics['avg_inference_time'] * 1000:.2f} ms")
        if 'peak_memory' in complexity_metrics:
            print(f"  峰值内存: {complexity_metrics['peak_memory']:.2f} MB")

    # 3. 注意力机制分析
    if 'attention' in eval_types:
        print("\n--- 分析注意力机制 ---")
        # 选择一个样本数据
        sample_data = dataset[0].to(device)

        # 可视化蛋白质图
        print("可视化蛋白质图结构...")
        graph_path = os.path.join(args.output_dir, "protein_graph.png")
        visualize_protein_graph(sample_data, save_path=graph_path)
        print(f"蛋白质图可视化已保存至: {graph_path}")

        # 提取注意力权重 (需要根据实际模型架构调整)
        try:
            attention_weights = extract_attention_weights(model, sample_data)
            for name, weights in attention_weights.items():
                attn_path = os.path.join(args.output_dir, f"attention_{name}.png")
                plot_attention_heatmap(weights[0].cpu().numpy(), save_path=attn_path)
                print(f"注意力热图已保存至: {attn_path}")
        except Exception as e:
            print(f"注意力权重提取失败: {e}")

    # 4. 模型消融研究
    if 'ablation' in eval_types:
        print("\n--- 执行模型消融研究 ---")

        # 这里需要基于model_class进行消融实验
        model_class = model.gat_model.__class__

        ablation_results = perform_ablation_study(model_class, config, dataset, device)
        performance['ablation_study'] = ablation_results

        print("消融研究结果:")
        for variant, metrics in ablation_results.items():
            print(f"  变种 {variant}:")
            print(f"    总参数量: {metrics['total_parameters']:,}")
            print(f"    平均推理时间: {metrics['avg_inference_time'] * 1000:.2f} ms")

    # 保存所有性能指标
    import json
    perf_path = os.path.join(args.output_dir, "performance_metrics.json")
    with open(perf_path, 'w') as f:
        # 转换不能被JSON序列化的值
        cleaned_performance = {}
        for k, v in performance.items():
            if isinstance(v, dict):
                cleaned_dict = {}
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float, str, bool)) or v2 is None:
                        cleaned_dict[k2] = v2
                    else:
                        cleaned_dict[k2] = str(v2)
                cleaned_performance[k] = cleaned_dict
            else:
                cleaned_performance[k] = str(v) if not isinstance(v, (int, float, str, bool)) and v is not None else v

        json.dump(cleaned_performance, f, indent=2)

    print(f"\n所有性能指标已保存至: {perf_path}")
    print("\n评估完成!")


if __name__ == "__main__":
    main()