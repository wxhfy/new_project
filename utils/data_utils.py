#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图谱数据验证与处理工具

该模块提供蛋白质图谱数据的加载、验证和处理功能，
确保图谱符合预定义的节点特征(35维)和边特征(8维)规范。

作者: wxhfy
日期: 2025-04-02
"""

import os
import torch
import logging
import numpy as np
import random
from tqdm import tqdm

from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 氨基酸映射
AA_MAP = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

# 图谱节点特征维度定义
NODE_FEATURE_DIMS = {
    'blosum62': (0, 20),  # BLOSUM62编码（20维）
    'spatial': (20, 23),  # 空间坐标（3维）
    'physicochemical': (23, 29),  # 理化特性（6维）
    'secondary_structure': (29, 32),  # 二级结构（3维）
    'other': (32, 35)  # 其他特征（3维）
}

# 图谱边特征维度定义
EDGE_FEATURE_DIMS = {
    'interaction_type': (0, 4),  # 相互作用类型（4维）
    'distance': (4, 5),  # 空间距离（1维）
    'strength': (5, 6),  # 相互作用强度（1维）
    'direction': (6, 8)  # 方向向量（2维）
}


def validate_graph_features(graph_id, graph):
    """
    验证蛋白质图谱的节点特征和边特征是否符合规范

    参数:
        graph_id: 图谱ID
        graph: PyG图数据对象

    返回:
        tuple: (是否有效, 错误信息)
    """
    validation_messages = []
    is_valid = True

    # 基本信息
    has_node_features = hasattr(graph, 'x') and graph.x is not None
    has_edge_features = hasattr(graph, 'edge_attr') and graph.edge_attr is not None
    has_edge_index = hasattr(graph, 'edge_index') and graph.edge_index is not None

    validation_messages.append(f"图谱ID: {graph_id}")
    validation_messages.append(f"节点数: {graph.num_nodes if hasattr(graph, 'num_nodes') else 'N/A'}")
    validation_messages.append(f"边数: {graph.num_edges if hasattr(graph, 'num_edges') else 'N/A'}")

    # 1. 检查节点特征
    if not has_node_features:
        validation_messages.append("❌ 节点特征缺失")
        is_valid = False
    else:
        node_feat_dim = graph.x.shape[1]
        if node_feat_dim == 35:
            validation_messages.append(f"✅ 节点特征维度符合要求: {node_feat_dim}")

            # 检查各特征段
            for feat_name, (start, end) in NODE_FEATURE_DIMS.items():
                feat_slice = graph.x[:, start:end]
                validation_messages.append(f"  - {feat_name}: {feat_slice.shape[1]}维 ({start}-{end})")

                # 检查特定特征值的范围
                if feat_name == 'spatial':
                    # 空间坐标应在[-1, 1]范围内
                    coord_min = torch.min(feat_slice).item()
                    coord_max = torch.max(feat_slice).item()
                    validation_messages.append(f"    坐标范围: [{coord_min:.2f}, {coord_max:.2f}]")

                elif feat_name == 'secondary_structure':
                    # 二级结构是one-hot编码，应该每行有且只有一个1
                    valid_ss = torch.sum(torch.round(feat_slice), dim=1) == 1
                    if not torch.all(valid_ss):
                        validation_messages.append(
                            f"    ❌ 二级结构编码有{torch.sum(~valid_ss).item()}行不符合one-hot规范")

        else:
            validation_messages.append(f"❌ 节点特征维度不符: 实际{node_feat_dim}, 预期35")
            is_valid = False

    # 2. 检查边索引
    if not has_edge_index:
        validation_messages.append("❌ 边索引缺失")
        is_valid = False
    else:
        if graph.edge_index.shape[0] != 2:
            validation_messages.append(f"❌ 边索引形状错误: {graph.edge_index.shape}")
            is_valid = False
        else:
            validation_messages.append(f"✅ 边索引格式正确: [2, {graph.edge_index.shape[1]}]")

    # 3. 检查边特征
    if not has_edge_features:
        validation_messages.append("❌ 边特征缺失")
        is_valid = False
    else:
        edge_feat_dim = graph.edge_attr.shape[1]
        if edge_feat_dim == 8:
            validation_messages.append(f"✅ 边特征维度符合要求: {edge_feat_dim}")

            # 检查各特征段
            for feat_name, (start, end) in EDGE_FEATURE_DIMS.items():
                feat_slice = graph.edge_attr[:, start:end]
                validation_messages.append(f"  - {feat_name}: {feat_slice.shape[1]}维 ({start}-{end})")

                # 检查特定特征
                if feat_name == 'interaction_type':
                    # 相互作用类型是one-hot编码
                    valid_types = torch.sum(torch.round(feat_slice), dim=1) == 1
                    if not torch.all(valid_types):
                        validation_messages.append(
                            f"    ❌ 相互作用类型编码有{torch.sum(~valid_types).item()}行不符合one-hot规范")

                elif feat_name == 'distance':
                    # 距离应为正值
                    min_dist = torch.min(feat_slice).item()
                    max_dist = torch.max(feat_slice).item()
                    if min_dist < 0:
                        validation_messages.append(f"    ❌ 存在负距离值: {min_dist:.2f}")
                    validation_messages.append(f"    距离范围: [{min_dist:.2f}, {max_dist:.2f}]")

                elif feat_name == 'strength':
                    # 强度应在[0, 1]范围内
                    min_strength = torch.min(feat_slice).item()
                    max_strength = torch.max(feat_slice).item()
                    if min_strength < 0 or max_strength > 1:
                        validation_messages.append(
                            f"    ❌ 强度值超出范围[0, 1]: [{min_strength:.2f}, {max_strength:.2f}]")
        else:
            validation_messages.append(f"❌ 边特征维度不符: 实际{edge_feat_dim}, 预期8")
            is_valid = False

    return is_valid, "\n".join(validation_messages)


def verify_protein_graph_file(file_path, max_samples=5, visualize=False):
    """
    验证蛋白质图谱文件中的图谱结构

    参数:
        file_path: 图谱文件路径
        max_samples: 最多验证的样本数
        visualize: 是否可视化特征分布

    返回:
        bool: 是否所有样本都有效
    """
    try:
        logger.info(f"开始验证图谱文件: {file_path}")

        # 加载图谱数据
        try:
            data = torch.load(file_path)
            logger.info(f"成功加载图谱文件，数据类型: {type(data)}")
        except Exception as e:
            logger.error(f"加载图谱文件失败: {str(e)}")
            return False

        # 确定数据格式并提取样本
        samples = []
        if isinstance(data, dict):
            logger.info(f"图谱文件包含字典数据，键数量: {len(data)}")
            # 从字典中提取样本
            for i, (key, graph) in enumerate(data.items()):
                if i >= max_samples:
                    break
                samples.append((key, graph))
        elif isinstance(data, list):
            logger.info(f"图谱文件包含列表数据，元素数量: {len(data)}")
            # 从列表中提取样本
            for i, graph in enumerate(data):
                if i >= max_samples:
                    break
                samples.append((f"sample_{i}", graph))
        else:
            logger.error(f"不支持的数据格式: {type(data)}")
            return False

        # 验证样本
        all_valid = True
        node_feature_values = []
        edge_feature_values = []

        for graph_id, graph in samples:
            is_valid, message = validate_graph_features(graph_id, graph)
            logger.info("\n" + message)

            if not is_valid:
                all_valid = False

            # 收集特征数据用于可视化
            if visualize and hasattr(graph, 'x') and graph.x is not None:
                node_feature_values.append(graph.x.numpy())

            if visualize and hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                edge_feature_values.append(graph.edge_attr.numpy())

        # 总结
        if all_valid:
            logger.info(f"✅ 验证通过：所有{len(samples)}个图谱样本都符合规范")
        else:
            logger.warning(f"❌ 验证失败：部分图谱样本不符合规范")

        return all_valid

    except Exception as e:
        logger.error(f"验证过程发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def load_pyg_graphs(file_paths, max_samples_per_length=10000, min_len=5, max_len=50,
                    fix_edge_features=False):
    """
    加载PyG图数据，同时控制每个长度下的样本数量，并过滤长度不在指定范围内的图数据。
    可选择验证图谱结构是否符合规范，并对最终采样的图谱进行边特征修复。

    参数:
        file_paths (list): PyG图数据文件路径列表
        max_samples_per_length (int): 每个长度保留的最大样本数
        min_len (int): 最小序列长度
        max_len (int): 最大序列长度
        validate (bool): 是否验证图谱结构
        fix_edge_features (bool): 是否修复边特征中的one-hot编码问题

    返回:
        list: PyG图数据对象列表
    """
    all_graphs = {}  # 字典：按长度分组
    validation_stats = {
        "total_graphs": 0,
        "valid_graphs": 0,
        "invalid_graphs": 0,
        "validation_errors": []
    }

    # 加载所有文件
    for file_idx, file_path in enumerate(tqdm(file_paths, desc="加载图数据")):
        try:
            # 加载PyTorch文件
            data = torch.load(file_path)

            # 检查数据格式（字典 or 列表）
            if isinstance(data, dict):
                # 将字典值按长度分类
                for protein_id, graph in data.items():
                    if hasattr(graph, 'x') and hasattr(graph, 'edge_index'):
                        seq_len = graph.x.shape[0]  # 使用节点数作为序列长度
                        validation_stats["total_graphs"] += 1

                        # 过滤长度不在[min_len, max_len]范围内的图
                        if seq_len < min_len or seq_len > max_len:
                            continue

                        if seq_len not in all_graphs:
                            all_graphs[seq_len] = []

                        all_graphs[seq_len].append(graph)

            elif isinstance(data, list):
                # 直接处理列表
                for i, graph in enumerate(data):
                    if hasattr(graph, 'x') and hasattr(graph, 'edge_index'):
                        seq_len = graph.x.shape[0]
                        validation_stats["total_graphs"] += 1

                        # 过滤长度不在[min_len, max_len]范围内的图
                        if seq_len < min_len or seq_len > max_len:
                            continue

                        if seq_len not in all_graphs:
                            all_graphs[seq_len] = []

                        all_graphs[seq_len].append(graph)
            else:
                logger.warning(f"文件 {file_path} 中的数据格式不支持")
        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {str(e)}")

    # 对每个长度进行采样
    sampled_graphs = []
    length_stats = {}

    for length, graphs in all_graphs.items():
        # 记录原始数量
        length_stats[length] = len(graphs)

        # 如果超过max_samples_per_length，随机采样
        if len(graphs) > max_samples_per_length:
            sampled = random.sample(graphs, max_samples_per_length)
        else:
            sampled = graphs

        sampled_graphs.extend(sampled)

    logger.info(f"按长度采样后共获取 {len(sampled_graphs)} 个图")
    logger.info(f"各长度数据统计: {length_stats}")

    # 在采样完成后修复边特征（仅处理最终需要使用的图谱）
    if fix_edge_features and sampled_graphs:
        fixed_count = 0
        edges_fixed = 0
        logger.info("开始修复采样图谱的边特征...")

        for i in tqdm(range(len(sampled_graphs)), desc="修复边特征"):
            graph, num_fixed = normalize_edge_features(sampled_graphs[i])
            sampled_graphs[i] = graph
            if num_fixed > 0:
                fixed_count += 1
                edges_fixed += num_fixed

        # 显示修复统计
        if fixed_count > 0:
            logger.info(
                f"边特征修复完成: 修复了 {fixed_count}/{len(sampled_graphs)} 个图谱 ({fixed_count / len(sampled_graphs) * 100:.1f}%)")
            logger.info(f"总共修正了 {edges_fixed} 条边的相互作用类型编码")
        else:
            logger.info("所有图谱边特征均符合规范，无需修复")

    return sampled_graphs


def normalize_edge_features(graph):
    """
    规范化图谱的边特征，确保相互作用类型为one-hot编码

    参数:
        graph: PyG图谱对象

    返回:
        tuple: (修复后的图谱对象, 修复的边数量)
    """
    edges_fixed = 0

    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None and graph.edge_attr.shape[1] >= 4:
        # 提取交互类型编码（前4维）
        interaction_types = graph.edge_attr[:, :4]

        # 检查每行的和是否为1（严格one-hot编码）
        row_sums = torch.sum(interaction_types, dim=1)
        non_onehot_mask = torch.abs(row_sums - 1.0) > 1e-6

        if torch.any(non_onehot_mask):
            # 创建新的规范化one-hot编码
            corrected_types = interaction_types.clone()
            problem_indices = torch.nonzero(non_onehot_mask).squeeze()

            # 对每个有问题的行，将最大值设为1，其他设为0
            if problem_indices.dim() == 0:  # 只有一个问题行
                idx = problem_indices.item()
                max_idx = torch.argmax(interaction_types[idx])
                corrected_types[idx] = torch.zeros_like(corrected_types[idx])
                corrected_types[idx, max_idx] = 1.0
                edges_fixed = 1
            else:  # 多个问题行
                for idx in problem_indices:
                    max_idx = torch.argmax(interaction_types[idx])
                    corrected_types[idx] = torch.zeros_like(corrected_types[idx])
                    corrected_types[idx, max_idx] = 1.0
                edges_fixed = len(problem_indices)

            # 更新边特征
            graph.edge_attr[:, :4] = corrected_types

            # 确保以下情况：如果全为0，则设置第一个类型为1（肽键连接）
            zero_rows = torch.sum(graph.edge_attr[:, :4], dim=1) == 0
            if torch.any(zero_rows):
                zero_indices = torch.nonzero(zero_rows).squeeze()
                if zero_indices.dim() == 0:  # 只有一行全0
                    graph.edge_attr[zero_indices.item(), 0] = 1.0
                    edges_fixed += 1
                else:  # 多行全0
                    graph.edge_attr[zero_indices, 0] = 1.0
                    edges_fixed += len(zero_indices)

    return graph, edges_fixed


def fix_all_graph_edges(graphs):
    """
    修复所有图谱的边特征

    参数:
        graphs: 图谱列表或字典

    返回:
        修复后的图谱数据
    """
    fixed_count = 0

    if isinstance(graphs, list):
        # 处理列表形式的图谱
        for i in range(len(graphs)):
            graphs[i] = normalize_edge_features(graphs[i])
            fixed_count += 1

    elif isinstance(graphs, dict):
        # 处理字典形式的图谱
        for key in graphs:
            graphs[key] = normalize_edge_features(graphs[key])
            fixed_count += 1

    logger.info(f"已修复 {fixed_count} 个图谱的边特征")
    return graphs


def split_data(graphs, split_ratio=None, seed=42):
    """
    将数据集分割为训练、验证和测试集

    参数:
        graphs (list): 图数据对象列表
        split_ratio (list): 分割比例，默认[训练:0.7, 验证:0.2, 测试:0.1]
        seed (int): 随机种子

    返回:
        tuple: (训练集, 验证集, 测试集)
    """
    # 确保分割比例有效
    if split_ratio is None:
        split_ratio = [0.7, 0.2, 0.1]

    total = sum(split_ratio)
    if abs(total - 1.0) > 0.001:  # 允许小误差
        logger.warning(f"分割比例之和为 {total}，与1.0不符，已自动调整")
        split_ratio = [r / total for r in split_ratio]  # 归一化

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 随机打乱数据
    indices = list(range(len(graphs)))
    random.shuffle(indices)

    # 计算分割点
    train_end = int(split_ratio[0] * len(graphs))
    val_end = train_end + int(split_ratio[1] * len(graphs))

    # 分割数据
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # 创建数据集
    train_data = [graphs[i] for i in train_indices]
    val_data = [graphs[i] for i in val_indices]
    test_data = [graphs[i] for i in test_indices]

    logger.info(f"数据集分割完成 - 训练集: {len(train_data)}，验证集: {len(val_data)}，测试集: {len(test_data)}")
    logger.info(
        f"实际使用的分割比例: [{len(train_data) / len(graphs):.2f}, {len(val_data) / len(graphs):.2f}, {len(test_data) / len(graphs):.2f}]")

    return train_data, val_data, test_data


def collate_protein_graphs(batch):
    """
    将蛋白质图数据批处理为批次

    参数:
        batch (list): 图数据对象列表

    返回:
        Batch: 批处理后的图数据
    """
    # 过滤None值（以防某些图无效）
    valid_batch = [graph for graph in batch if graph is not None]

    if not valid_batch:
        logger.warning("批次中无有效图谱数据")
        # 返回一个空批次
        return Batch()

    return Batch.from_data_list(valid_batch)


def save_processed_data(train_data, val_data, test_data, output_dir, report=True):
    """
    保存处理后的数据

    参数:
        train_data (list): 训练数据
        val_data (list): 验证数据
        test_data (list): 测试数据
        output_dir (str): 输出目录
        report (bool): 是否生成数据报告
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存数据
    torch.save(train_data, os.path.join(output_dir, "train_data.pt"))
    torch.save(val_data, os.path.join(output_dir, "val_data.pt"))
    torch.save(test_data, os.path.join(output_dir, "test_data.pt"))

    logger.info(f"数据已保存至 {output_dir}")

    # 生成数据报告
    if report:
        report_path = os.path.join(output_dir, "data_report.txt")

        with open(report_path, 'w') as f:
            f.write("蛋白质图谱数据处理报告\n")
            f.write("====================\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 训练集统计
            f.write("训练集统计:\n")
            f.write(f"- 样本数: {len(train_data)}\n")
            if train_data:
                node_counts = [graph.num_nodes for graph in train_data]
                edge_counts = [graph.num_edges for graph in train_data]
                f.write(f"- 平均节点数: {np.mean(node_counts):.1f} ± {np.std(node_counts):.1f}\n")
                f.write(f"- 平均边数: {np.mean(edge_counts):.1f} ± {np.std(edge_counts):.1f}\n")
                f.write(f"- 节点数范围: {min(node_counts)} - {max(node_counts)}\n")
            f.write("\n")

            # 验证集统计
            f.write("验证集统计:\n")
            f.write(f"- 样本数: {len(val_data)}\n")
            if val_data:
                node_counts = [graph.num_nodes for graph in val_data]
                edge_counts = [graph.num_edges for graph in val_data]
                f.write(f"- 平均节点数: {np.mean(node_counts):.1f} ± {np.std(node_counts):.1f}\n")
                f.write(f"- 平均边数: {np.mean(edge_counts):.1f} ± {np.std(edge_counts):.1f}\n")
                f.write(f"- 节点数范围: {min(node_counts)} - {max(node_counts)}\n")
            f.write("\n")

            # 测试集统计
            f.write("测试集统计:\n")
            f.write(f"- 样本数: {len(test_data)}\n")
            if test_data:
                node_counts = [graph.num_nodes for graph in test_data]
                edge_counts = [graph.num_edges for graph in test_data]
                f.write(f"- 平均节点数: {np.mean(node_counts):.1f} ± {np.std(node_counts):.1f}\n")
                f.write(f"- 平均边数: {np.mean(edge_counts):.1f} ± {np.std(edge_counts):.1f}\n")
                f.write(f"- 节点数范围: {min(node_counts)} - {max(node_counts)}\n")

        logger.info(f"数据报告已保存至 {report_path}")



def process_protein_data(input_dir, output_dir, max_samples_per_length=10000,
                         split_ratio=None, seed=42, validate=False, fix_edge_features=False):
    """
    处理蛋白质数据，加载图数据并保存为训练、验证和测试集

    参数:
        input_dir (str): PyG图数据文件所在目录
        output_dir (str): 输出目录
        max_samples_per_length (int): 每个长度保留的最大样本数
        split_ratio (list): 分割比例，默认[训练:0.7, 验证:0.2, 测试:0.1]
        seed (int): 随机种子
        validate (bool): 是否验证图谱结构
    """
    # 获取所有PT文件
    file_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pt'):
                file_paths.append(os.path.join(root, file))

    logger.info(f"找到 {len(file_paths)} 个PT文件")

    # 加载图数据（可选验证）
    graphs = load_pyg_graphs(file_paths, max_samples_per_length, fix_edge_features=fix_edge_features)

    # 分割数据
    train_data, val_data, test_data = split_data(graphs, split_ratio, seed)

    # 保存处理后的数据（生成报告）
    save_processed_data(train_data, val_data, test_data, output_dir, report=True)

    return train_data, val_data, test_data


def examine_protein_graph(graph_path, graph_id=None):
    """
    详细检查单个蛋白质图谱结构

    参数:
        graph_path: 图谱文件路径
        graph_id: 如果文件包含多个图谱，可以指定ID

    返回:
        bool: 图谱是否有效
    """
    logger.info(f"开始检查图谱文件: {graph_path}")

    try:
        # 加载图谱数据
        data = torch.load(graph_path)

        # 确定图谱对象
        if graph_id is not None and isinstance(data, dict):
            if graph_id in data:
                graph = data[graph_id]
                logger.info(f"成功找到指定图谱: {graph_id}")
            else:
                logger.error(f"未找到指定图谱ID: {graph_id}")
                available_ids = list(data.keys())[:5] + ['...'] if len(data) > 5 else list(data.keys())
                logger.info(f"可用的图谱ID: {available_ids}")
                return False
        elif isinstance(data, dict) and len(data) > 0:
            # 使用第一个图谱
            graph_id = next(iter(data.keys()))
            graph = data[graph_id]
            logger.info(f"未指定图谱ID，使用第一个图谱: {graph_id}")
        elif isinstance(data, list) and len(data) > 0:
            # 使用第一个图谱
            graph = data[0]
            graph_id = "sample_0"
            logger.info(f"数据为列表，使用第一个图谱")
        else:
            # 假设传入的直接是图谱对象
            graph = data
            graph_id = "direct_graph"
            logger.info(f"直接使用传入的图谱对象")

        # 详细验证图谱结构
        is_valid, validation_msg = validate_graph_features(graph_id, graph)

        # 打印验证消息
        print("\n" + validation_msg + "\n")

        # 详细分析图谱结构
        if is_valid:
            # 节点分析
            node_feature_analysis(graph)

            # 边分析
            edge_feature_analysis(graph)

            # 拓扑结构分析
            topology_analysis(graph)

            logger.info(f"✅ 图谱 {graph_id} 验证通过并完成详细分析")
        else:
            logger.error(f"❌ 图谱 {graph_id} 验证失败")

        return is_valid

    except Exception as e:
        logger.error(f"检查图谱时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def node_feature_analysis(graph):
    """详细分析节点特征"""
    if not hasattr(graph, 'x') or graph.x is None:
        logger.warning("图谱缺少节点特征")
        return

    # 基本统计
    node_features = graph.x.numpy() if isinstance(graph.x, torch.Tensor) else graph.x

    logger.info("节点特征分析:")
    logger.info(f"- 节点数量: {node_features.shape[0]}")
    logger.info(f"- 特征维度: {node_features.shape[1]}")

    # 特征分段分析
    for feature_name, (start, end) in NODE_FEATURE_DIMS.items():
        feature_slice = node_features[:, start:end]
        mean_val = np.mean(feature_slice)
        std_val = np.std(feature_slice)
        min_val = np.min(feature_slice)
        max_val = np.max(feature_slice)

        logger.info(f"- {feature_name} 特征 ({start}:{end}):")
        logger.info(f"  - 平均值: {mean_val:.4f} ± {std_val:.4f}")
        logger.info(f"  - 范围: [{min_val:.4f}, {max_val:.4f}]")

        # 特定特征类型的分析
        if feature_name == 'blosum62':
            # 分析氨基酸分布
            residue_analysis(feature_slice)
        elif feature_name == 'secondary_structure':
            # 分析二级结构分布
            ss_analysis(feature_slice)

    # 相关性分析
    logger.info("- 主要特征间相关性:")
    try:
        corr_matrix = np.corrcoef(node_features[:, [0, 20, 23, 29, 32]].T)
        features = ["BLOSUM首维", "空间X坐标", "疏水性", "α螺旋", "溶剂可及性"]
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                logger.info(f"  - {features[i]} 与 {features[j]}: {corr_matrix[i, j]:.4f}")
    except:
        logger.warning("  计算相关性时出错")


def edge_feature_analysis(graph):
    """详细分析边特征"""
    if not hasattr(graph, 'edge_attr') or graph.edge_attr is None:
        logger.warning("图谱缺少边特征")
        return

    # 边特征基本统计
    edge_attrs = graph.edge_attr.numpy() if isinstance(graph.edge_attr, torch.Tensor) else graph.edge_attr
    edge_index = graph.edge_index.numpy() if isinstance(graph.edge_index, torch.Tensor) else graph.edge_index

    logger.info("边特征分析:")
    logger.info(f"- 边数量: {edge_attrs.shape[0]}")
    logger.info(f"- 特征维度: {edge_attrs.shape[1]}")

    # 计算边密度
    num_nodes = graph.num_nodes
    max_edges = num_nodes * (num_nodes - 1) / 2
    edge_density = edge_attrs.shape[0] / max_edges
    logger.info(f"- 边密度: {edge_density:.4f} (边数/最大可能边数)")

    # 相互作用类型分析
    interaction_types = edge_attrs[:, :4]
    type_counts = np.sum(np.round(interaction_types), axis=0)
    type_names = ['肽键连接', '氢键相互作用', '离子相互作用', '疏水相互作用']

    logger.info("- 相互作用类型分布:")
    for i, name in enumerate(type_names):
        percentage = type_counts[i] / edge_attrs.shape[0] * 100
        logger.info(f"  - {name}: {int(type_counts[i])} ({percentage:.1f}%)")

    # 距离分析
    distances = edge_attrs[:, 4]
    logger.info(
        f"- 空间距离: {np.mean(distances):.2f} ± {np.std(distances):.2f} Å (范围: {np.min(distances):.2f}-{np.max(distances):.2f} Å)")

    # 相互作用强度分析
    strength = edge_attrs[:, 5]
    logger.info(
        f"- 相互作用强度: {np.mean(strength):.2f} ± {np.std(strength):.2f} (范围: {np.min(strength):.2f}-{np.max(strength):.2f})")

    # 方向向量分析
    directions = edge_attrs[:, 6:8]
    direction_norm = np.sqrt(np.sum(directions ** 2, axis=1))
    logger.info(f"- 方向向量范数: {np.mean(direction_norm):.4f} ± {np.std(direction_norm):.4f}")


def topology_analysis(graph):
    """分析图谱的拓扑结构"""
    if not hasattr(graph, 'edge_index') or graph.edge_index is None:
        logger.warning("图谱缺少边索引，无法进行拓扑分析")
        return

    # 转换为NetworkX图以便分析
    import networkx as nx

    edge_index = graph.edge_index.numpy() if isinstance(graph.edge_index, torch.Tensor) else graph.edge_index
    G = nx.Graph()

    # 添加节点
    for i in range(graph.num_nodes):
        G.add_node(i)

    # 添加边
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst)

    logger.info("拓扑结构分析:")

    # 基本统计
    logger.info(f"- 节点数: {G.number_of_nodes()}")
    logger.info(f"- 边数: {G.number_of_edges()}")

    # 连通性分析
    is_connected = nx.is_connected(G)
    logger.info(f"- 图是否连通: {'是' if is_connected else '否'}")

    if not is_connected:
        components = list(nx.connected_components(G))
        logger.info(f"- 连通分量数: {len(components)}")

        # 分析每个连通分量的大小
        component_sizes = [len(c) for c in components]
        logger.info(f"- 最大连通分量大小: {max(component_sizes)} 节点")
        logger.info(f"- 连通分量大小分布: {sorted(component_sizes, reverse=True)}")

    # 度分布
    degrees = [d for _, d in G.degree()]
    logger.info(f"- 平均度: {np.mean(degrees):.2f} ± {np.std(degrees):.2f}")
    logger.info(f"- 最大度: {max(degrees)}")
    logger.info(f"- 最小度: {min(degrees)}")

    # 团数和聚类系数
    try:
        clustering = nx.average_clustering(G)
        logger.info(f"- 平均聚类系数: {clustering:.4f}")
    except:
        logger.info("- 无法计算平均聚类系数")

    # 路径长度
    if is_connected:
        try:
            avg_path = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
            logger.info(f"- 平均最短路径长度: {avg_path:.2f}")
            logger.info(f"- 图直径: {diameter}")
        except:
            logger.info("- 无法计算路径长度统计")


def residue_analysis(blosum_features):
    """根据BLOSUM编码分析氨基酸分布"""
    try:
        # 估算氨基酸类型
        if blosum_features.shape[1] != 20:
            logger.warning("BLOSUM特征维度不是20，跳过氨基酸分析")
            return

        # 找出每行最大值的索引，作为氨基酸类型估计
        aa_indices = np.argmax(blosum_features, axis=1)

        # 统计各类氨基酸数量
        aa_counts = np.bincount(aa_indices, minlength=20)

        # 氨基酸索引到单字母代码的映射
        std_aa_order = "ARNDCQEGHILKMFPSTWYV"

        # 计算各类氨基酸百分比
        total_aa = np.sum(aa_counts)
        aa_percentages = aa_counts / total_aa * 100

        # 打印氨基酸分布
        logger.info("  - 估计的氨基酸分布:")
        for i in range(20):
            if aa_counts[i] > 0:
                logger.info(f"    - {std_aa_order[i]}: {aa_counts[i]} ({aa_percentages[i]:.1f}%)")

        # 分析氨基酸属性分布
        hydrophobic = "AVILMFYW"
        charged = "DEKR"
        polar = "NQST"
        special = "CGP"

        hydrophobic_count = sum(aa_counts[std_aa_order.index(aa)] for aa in hydrophobic)
        charged_count = sum(aa_counts[std_aa_order.index(aa)] for aa in charged)
        polar_count = sum(aa_counts[std_aa_order.index(aa)] for aa in polar)
        special_count = sum(aa_counts[std_aa_order.index(aa)] for aa in special)

        logger.info("  - 氨基酸特性分布:")
        logger.info(f"    - 疏水性氨基酸: {hydrophobic_count} ({hydrophobic_count / total_aa * 100:.1f}%)")
        logger.info(f"    - 带电氨基酸: {charged_count} ({charged_count / total_aa * 100:.1f}%)")
        logger.info(f"    - 极性氨基酸: {polar_count} ({polar_count / total_aa * 100:.1f}%)")
        logger.info(f"    - 特殊氨基酸: {special_count} ({special_count / total_aa * 100:.1f}%)")

    except Exception as e:
        logger.warning(f"氨基酸分析出错: {str(e)}")


def ss_analysis(ss_features):
    """分析二级结构分布"""
    try:
        # 检查是否为二级结构特征
        if ss_features.shape[1] != 3:
            logger.warning("二级结构特征维度不是3，跳过分析")
            return

        # 估算二级结构类型
        ss_indices = np.argmax(ss_features, axis=1)

        # 统计各类二级结构数量
        ss_counts = np.bincount(ss_indices, minlength=3)

        # 计算百分比
        total_residues = np.sum(ss_counts)
        ss_percentages = ss_counts / total_residues * 100

        # 打印二级结构分布
        ss_names = ["α-螺旋", "β-折叠", "卷曲"]
        logger.info("  - 二级结构分布:")
        for i in range(3):
            logger.info(f"    - {ss_names[i]}: {ss_counts[i]} ({ss_percentages[i]:.1f}%)")

        # 计算螺旋和折叠的比例
        helix_strand_ratio = ss_counts[0] / max(1, ss_counts[1])
        logger.info(f"  - α-螺旋/β-折叠比例: {helix_strand_ratio:.2f}")

    except Exception as e:
        logger.warning(f"二级结构分析出错: {str(e)}")


if __name__ == "__main__":
    # 测试数据处理与验证
    import time
    import argparse

    parser = argparse.ArgumentParser(description="蛋白质图谱数据验证与处理工具")
    parser.add_argument("--mode", choices=["validate", "process", "examine"], default="process",
                        help="运行模式: validate-验证图谱文件, process-处理数据集, examine-详细检查单个图谱")
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件或目录路径")
    parser.add_argument("--output", type=str, default="../processed_data",
                        help="输出目录路径 (仅在process模式下使用)")
    parser.add_argument("--graph_id", type=str, default=None,
                        help="图谱ID (仅在examine模式下使用)")
    parser.add_argument("--max_samples", type=int, default=5,
                        help="最大样本数 (仅在validate模式下使用)")
    parser.add_argument("--fix_edges", action="store_true" ,
                        help="修复图谱中的边特征编码问题")

    args = parser.parse_args()

    # 配置日志
    logger.info("启动蛋白质图谱数据验证与处理工具")
    logger.info(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"运行模式: {args.mode}")

    try:
        if args.mode == "validate" and not args.fix_edges:
            # 验证图谱文件
            logger.info(f"验证图谱文件: {args.input}")
            verify_protein_graph_file(args.input, max_samples=args.max_samples)
        elif args.mode == "validate" and args.fix_edges:
            # 加载、修复并保存
            data = torch.load(args.input)
            fixed_data = fix_all_graph_edges(data)
            output_path = args.input.replace(".pt", "_fixed.pt")
            torch.save(fixed_data, output_path)
            logger.info(f"已修复并保存至: {output_path}")
        elif args.mode == "process":
            # 处理数据集
            logger.info(f"处理数据集: {args.input} -> {args.output}")

            # 从配置类获取默认参数，使用安全导入方式
            try:
                from utils.config import Config

                config = Config()
                split_ratio = config.TRAIN_VAL_TEST_SPLIT
            except:
                logger.info("未找到配置类，使用默认参数")
                split_ratio = [0.7, 0.2, 0.1]
            # 执行数据处理
            process_protein_data(
                args.input,
                args.output,
                max_samples_per_length=10000,
                split_ratio=split_ratio,
                fix_edge_features=args.fix_edges
            )

        elif args.mode == "examine":
            # 详细检查单个图谱
            logger.info(f"详细检查图谱: {args.input}, ID: {args.graph_id}")
            examine_protein_graph(args.input, args.graph_id)

    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())