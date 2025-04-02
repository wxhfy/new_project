#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质知识图谱去冗余工具 (纯缓存图谱版)

该脚本功能:
1. 直接读取预先缓存的图谱文件
2. 验证图谱缓存的完整性
3. 执行图谱结构相似度计算和去冗余
4. 保存去冗余后的结果

作者: wxhfy
"""

import argparse
import concurrent.futures
import gc
import glob
import io
import json
import logging
import os
import sys
import time
import traceback

import numpy as np
import psutil
import torch
from fsspec.asyn import ResourceError
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_gpu_device(device_id):
    """设置使用的GPU设备ID"""
    if device_id is not None:
        try:
            if torch.cuda.is_available():
                device_id_list = [int(id.strip()) for id in device_id.split(',')]
                devices_available = list(range(torch.cuda.device_count()))
                valid_devices = [id for id in device_id_list if id in devices_available]

                if valid_devices:
                    # 设置环境变量以限制CUDA可见设备
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, valid_devices))
                    torch.cuda.set_device(valid_devices[0])
                    logger.info(f"已设置GPU设备: {valid_devices}")
                else:
                    logger.warning(f"未找到有效的GPU设备ID: {device_id}，将使用默认设备")
            else:
                logger.warning("未检测到可用的CUDA设备，将使用CPU")
        except Exception as e:
            logger.warning(f"设置GPU设备时出错: {str(e)}，将使用默认设备")


def check_memory_usage(threshold_gb=None, force_gc=False):
    """检查内存使用情况，并在需要时执行垃圾回收"""
    try:
        # 强制垃圾回收（如果要求）
        if force_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 获取当前内存使用量
        mem_used = psutil.Process().memory_info().rss / (1024 ** 3)  # 转换为GB

        # 如果设置了阈值且内存超过阈值
        if threshold_gb and mem_used > threshold_gb:
            logger.warning(f"内存使用已达 {mem_used:.2f} GB，超过阈值 {threshold_gb:.2f} GB，执行垃圾回收")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True

        # 内存使用未超过阈值
        return False

    except Exception as e:
        logger.error(f"检查内存使用时出错: {str(e)}")
        gc.collect()  # 出错时仍执行垃圾回收以确保安全
        return False


def log_system_resources():
    """记录系统资源使用情况"""
    try:
        mem = psutil.virtual_memory()
        logger.info(f"系统内存: {mem.used / 1024 ** 3:.1f}GB/{mem.total / 1024 ** 3:.1f}GB ({mem.percent}%)")

        swap = psutil.swap_memory()
        logger.info(f"交换内存: {swap.used / 1024 ** 3:.1f}GB/{swap.total / 1024 ** 3:.1f}GB ({swap.percent}%)")

        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        logger.info(f"CPU核心: {cpu_count}个, 使用率: {sum(cpu_percent) / len(cpu_percent):.1f}%")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                    reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                    max_mem = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                    logger.info(
                        f"GPU {i} ({torch.cuda.get_device_name(i)}): 已分配={allocated:.1f}GB, 已保留={reserved:.1f}GB, 总计={max_mem:.1f}GB")
                except:
                    logger.info(f"GPU {i}: 无法获取内存信息")
    except Exception as e:
        logger.warning(f"无法获取系统资源信息: {str(e)}")


def setup_logging(output_dir):
    """设置日志系统"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"graph_deduplication_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 捕获所有级别的日志

    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 添加控制台处理器 - 仅显示INFO及以上级别
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console)

    # 添加文件处理器 - 记录所有级别（包括DEBUG）
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    # 记录系统信息
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("未检测到GPU")

    return root_logger, log_file


import os
import io
import glob
import time
import json
import torch
import numpy as np
import traceback
import logging
import gc
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_graph_cache_metadata(cache_meta_path):
    """
    加载图谱缓存元数据

    参数:
        cache_meta_path: 图谱缓存元数据的JSON文件路径

    返回:
        dict: 缓存元数据
    """
    if not os.path.exists(cache_meta_path):
        logger.error(f"图谱缓存元数据文件不存在: {cache_meta_path}")
        return {}

    try:
        with open(cache_meta_path, 'r') as f:
            metadata = json.load(f)

        # 获取缓存计数和完整性信息
        cached_count = metadata.get("cached_count", 0) or metadata.get("total_graphs", 0)
        is_complete = metadata.get("is_complete_set", "False")
        expected_count = metadata.get("expected_graph_count", 0) or cached_count

        logger.info(f"图谱缓存元数据: 包含 {cached_count} 个图谱")
        logger.info(f"完整性标记: {is_complete}")
        logger.info(f"预期图谱数: {expected_count}")

        # 获取ID示例信息
        id_sample = metadata.get("id_sample", [])
        if id_sample:
            logger.info(f"ID示例: {', '.join(str(x) for x in id_sample[:3])}...")

        return metadata

    except Exception as e:
        logger.error(f"加载图谱缓存元数据时出错: {str(e)}")
        return {}


def safe_load_graph(file_path, map_location=None):
    """
    安全加载图谱文件，避免内存映射问题并处理所有异常

    参数:
        file_path: 图谱文件路径
        map_location: PyTorch加载时的设备位置

    返回:
        成功时返回加载的图谱，失败时返回空字典
    """
    try:
        # 检查文件是否存在
        if not os.path.isfile(file_path):
            logger.error(f"文件不存在: {file_path}")
            return {}

        # 读取文件内容到内存
        with open(file_path, 'rb') as f:
            file_content = f.read()
            if not file_content:
                logger.error(f"文件为空: {file_path}")
                return {}

            # 创建内存缓冲区并加载
            buffer = io.BytesIO(file_content)
            result = torch.load(buffer, map_location=map_location)
            buffer.close()

            # 确保返回值是有效的
            if result is None:
                logger.error(f"加载结果为None: {file_path}")
                return {}

            return result

    except Exception as e:
        logger.error(f"加载文件失败: {file_path}, 错误: {str(e)}")
        return {}


def graph_feature_extraction(graph):
    """增强版图谱特征提取 - 充分利用节点和边特征"""
    features = []

    # 1. 基本拓扑特征
    num_nodes = graph.x.shape[0] if hasattr(graph, 'x') else 0
    num_edges = graph.edge_index.shape[1] // 2 if hasattr(graph, 'edge_index') else 0
    features.extend([num_nodes, num_edges, num_edges / max(1, num_nodes)])

    # 2. 节点特征统计 - 处理35维特征
    if hasattr(graph, 'x') and graph.x is not None:
        x = graph.x.cpu().numpy() if torch.is_tensor(graph.x) else graph.x

        # 2.1 BLOSUM62编码统计 (前20维)
        blosum_feats = x[:, :20]
        features.extend(np.mean(blosum_feats, axis=0))
        features.extend(np.std(blosum_feats, axis=0))

        # 2.2 空间坐标统计 (3维)
        coord_feats = x[:, 20:23]
        features.extend(np.mean(coord_feats, axis=0))
        features.extend(np.std(coord_feats, axis=0))

        # 2.3 理化特性统计 (6维)
        physchem_feats = x[:, 23:29]
        features.extend(np.mean(physchem_feats, axis=0))
        features.extend(np.std(physchem_feats, axis=0))

        # 2.4 二级结构和其他特征统计 (6维)
        struct_feats = x[:, 29:]
        features.extend(np.mean(struct_feats, axis=0))
        features.extend(np.std(struct_feats, axis=0))

    # 3. 边特征统计 - 处理8维特征
    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        edge_attr = graph.edge_attr.cpu().numpy() if torch.is_tensor(graph.edge_attr) else graph.edge_attr

        # 3.1 相互作用类型分布 (前4维)
        type_dist = np.mean(edge_attr[:, :4], axis=0)
        features.extend(type_dist)

        # 3.2 距离特征统计
        dist_feat = edge_attr[:, 4]
        features.extend([np.mean(dist_feat), np.std(dist_feat), np.min(dist_feat), np.max(dist_feat)])

        # 3.3 相互作用强度统计
        strength_feat = edge_attr[:, 5]
        features.extend([np.mean(strength_feat), np.std(strength_feat)])

        # 3.4 方向向量统计
        dir_feat = edge_attr[:, 6:8]
        features.extend(np.mean(dir_feat, axis=0))

    return np.array(features, dtype=np.float32)

def process_batch(batch_ids, graphs):
    """
    并行处理一批图谱的特征提取

    参数:
        batch_ids: 要处理的图谱ID列表
        graphs: 包含图谱的字典

    返回:
        dict: ID到特征向量的映射
    """
    batch_features = {}
    for graph_id in batch_ids:
        try:
            graph = graphs[graph_id]
            feature = graph_feature_extraction(graph)
            if feature is not None and not np.isnan(feature).any() and len(feature) > 0:
                batch_features[graph_id] = feature
        except Exception as e:
            pass  # 简化错误处理
    return batch_features


def compute_graph_level_features(graph):
    """计算图谱级别特征，包括拓扑特征"""
    features = []

    # 计算连通分量数量
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        import networkx as nx
        # 转换为NetworkX图进行拓扑分析
        G = nx.Graph()
        edge_list = graph.edge_index.t().numpy()
        G.add_edges_from(edge_list)

        # 计算图拓扑特征
        num_connected_components = nx.number_connected_components(G)
        clustering_coef = nx.average_clustering(G)
        try:
            diameter = max(nx.eccentricity(G.subgraph(c)) for c in nx.connected_components(G))
        except:
            diameter = 0

        features.extend([num_connected_components, clustering_coef, diameter])

        # 计算度分布统计
        degrees = [d for _, d in G.degree()]
        features.extend([np.mean(degrees), np.std(degrees), np.max(degrees)])

    return np.array(features, dtype=np.float32)

def load_and_process_graphs_in_batches(cache_dir, cache_id, output_dir, similarity_threshold=0.85,
                                       batch_size=10000, processing_batch_size=1000):
    """
    分批加载和处理图谱，执行特征提取和相似度去冗余

    参数:
        cache_dir: 缓存目录路径
        cache_id: 缓存ID前缀
        output_dir: 输出目录路径
        similarity_threshold: 相似度阈值，默认0.85
        batch_size: 每次从文件加载的图谱数量
        processing_batch_size: 特征提取的批处理大小

    返回:
        list: 处理结果文件列表
    """
    import os
    import torch
    import numpy as np
    import json
    import glob
    import time
    import gc
    import traceback
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有缓存文件
    cache_files = sorted(glob.glob(os.path.join(cache_dir, f"{cache_id}_part_*.pt")))
    if not cache_files:
        logger.error(f"未找到任何缓存文件: {os.path.join(cache_dir, f'{cache_id}_part_*.pt')}")
        return {}

    logger.info(f"找到 {len(cache_files)} 个缓存文件，开始分批处理图谱")

    # 记录处理开始时间
    start_time = time.time()

    # 加载元数据文件
    meta_path = os.path.join(cache_dir, f"{cache_id}_meta.json")
    metadata = load_graph_cache_metadata(meta_path)

    # 初始化FAISS索引
    try:
        import faiss
        # 估计特征维度
        logger.info("估计特征维度...")
        test_file = cache_files[0]
        test_data = safe_load_graph(test_file, map_location='cpu')

        if not test_data:
            logger.error("无法加载测试文件来估计特征维度")
            return {}

        # 提取测试特征确定维度
        test_id = next(iter(test_data))
        test_feature = graph_feature_extraction(test_data[test_id])
        feature_dim = len(test_feature)
        logger.info(f"特征维度: {feature_dim}")

        # 初始化FAISS索引 - 使用L2距离
        index = faiss.IndexFlatL2(feature_dim)

        # 初始化特征标准化器
        scaler = StandardScaler()

        # 从首个文件采样训练标准化器
        sample_features = []
        sample_count = min(2000, len(test_data))
        sample_ids = list(test_data.keys())[:sample_count]

        logger.info(f"从 {len(sample_ids)} 个样本训练标准化器...")
        for graph_id in sample_ids:
            feature = graph_feature_extraction(test_data[graph_id])
            if feature is not None and not np.isnan(feature).any():
                sample_features.append(feature)

        if sample_features:
            sample_features = np.vstack(sample_features)
            scaler.fit(sample_features)
            logger.info(f"标准化器训练完成，使用 {len(sample_features)} 个样本")

        # 释放测试数据
        del test_data
        del sample_features
        gc.collect()

    except ImportError:
        logger.error("未安装FAISS库，无法继续图谱相似度计算")
        return {}

    # 跟踪已处理和已保留的图谱ID
    processed_ids = set()
    retained_graph_ids = set()

    # 计数器
    total_graph_count = 0
    total_retained = 0

    # 为中间结果创建临时目录
    temp_output_dir = os.path.join(output_dir, "temp_similarity_results")
    os.makedirs(temp_output_dir, exist_ok=True)

    # 处理结果文件列表
    result_files = []

    # 处理每个缓存文件
    with tqdm(total=len(cache_files), desc="处理图谱缓存文件") as pbar:
        for file_idx, file_path in enumerate(cache_files):
            batch_start = time.time()
            logger.info(f"处理文件 {file_idx + 1}/{len(cache_files)}: {os.path.basename(file_path)}")

            try:
                # 安全加载图谱
                graphs_batch = safe_load_graph(file_path, map_location='cpu')

                if not graphs_batch:
                    logger.warning(f"文件加载失败或为空: {os.path.basename(file_path)}")
                    pbar.update(1)
                    continue

                # 更新计数器
                total_graph_count += len(graphs_batch)
                logger.info(f"加载了 {len(graphs_batch)} 个图谱，累计处理: {total_graph_count}")

                # 只处理新图谱
                new_graphs = {id: graph for id, graph in graphs_batch.items() if id not in processed_ids}
                logger.info(f"本批次有 {len(new_graphs)} 个新图谱需要处理")

                if not new_graphs:
                    logger.info("无新图谱，跳过当前批次")
                    pbar.update(1)
                    continue

                # 分批提取特征
                features = {}
                graph_ids = list(new_graphs.keys())

                # 特征提取进度跟踪
                feat_start = time.time()
                with tqdm(total=len(graph_ids), desc="提取特征", leave=False) as feat_pbar:
                    for i in range(0, len(graph_ids), processing_batch_size):
                        batch_ids = graph_ids[i:i + processing_batch_size]
                        batch_features = process_batch(batch_ids, new_graphs)
                        features.update(batch_features)
                        feat_pbar.update(len(batch_ids))

                        # 定期执行内存清理
                        if i % (processing_batch_size * 5) == 0:
                            gc.collect()

                logger.info(f"提取了 {len(features)} 个特征向量，用时 {time.time() - feat_start:.2f}秒")

                # 如果有特征，进行去冗余处理
                if features:
                    # 将特征转换为矩阵
                    feature_ids = list(features.keys())
                    feature_matrix = np.vstack([features[gid] for gid in feature_ids])

                    # 标准化特征
                    feature_matrix = scaler.transform(feature_matrix)

                    # 按复杂度排序 (节点数+边数)
                    complexity = np.array(
                        [feature_matrix[i][0] + feature_matrix[i][1] for i in range(len(feature_matrix))])
                    sorted_indices = np.argsort(-complexity)  # 降序排列，复杂结构优先

                    # 构建当前批次相似性索引
                    batch_index = faiss.IndexFlatL2(feature_dim)
                    batch_index.add(feature_matrix)

                    # 执行去冗余操作
                    batch_retained_ids = set()
                    redundant_count = 0

                    sim_start = time.time()
                    for idx in sorted_indices:
                        graph_id = feature_ids[idx]

                        # 跳过已处理过的图谱
                        if graph_id in processed_ids:
                            continue

                        # 记录为已处理
                        processed_ids.add(graph_id)

                        # 先保留当前图谱
                        batch_retained_ids.add(graph_id)
                        retained_graph_ids.add(graph_id)

                        # 查询相似图谱
                        query_vector = feature_matrix[idx:idx + 1]
                        k = min(100, len(feature_matrix))  # 限制查询数量
                        distances, neighbors = batch_index.search(query_vector, k)

                        # 标记相似图谱为已处理
                        for j, dist in zip(neighbors[0][1:], distances[0][1:]):  # 跳过自身
                            if j < len(feature_ids):
                                # 计算相似度得分 (L2距离转换为相似度)
                                similarity = 1.0 / (1.0 + dist / 10.0)  # 调整为更合适的相似度映射
                                if similarity > similarity_threshold:
                                    similar_id = feature_ids[j]
                                    if similar_id not in processed_ids:
                                        processed_ids.add(similar_id)
                                        redundant_count += 1

                    logger.info(
                        f"相似度计算完成，去除 {redundant_count} 个冗余图谱，用时 {time.time() - sim_start:.2f}秒")

                    # 保留非冗余图谱
                    batch_filtered_graphs = {gid: new_graphs[gid] for gid in batch_retained_ids}
                    total_retained += len(batch_filtered_graphs)

                    # 记录批次结果
                    logger.info(
                        f"批次 {file_idx + 1} 处理完成: 保留 {len(batch_filtered_graphs)}/{len(new_graphs)} 个图谱 (总保留: {total_retained})")

                    # 保存当前批次结果
                    if batch_filtered_graphs:
                        result_file = os.path.join(temp_output_dir, f"similar_filtered_{file_idx}.pt")
                        torch.save(batch_filtered_graphs, result_file)
                        result_files.append(result_file)
                        logger.info(f"批次结果已保存至: {result_file}")

                # 释放资源
                del graphs_batch
                del new_graphs
                del features
                if 'feature_matrix' in locals():
                    del feature_matrix
                gc.collect()

                # 更新进度
                pbar.update(1)
                pbar.set_postfix({"保留率": f"{total_retained * 100 / total_graph_count:.1f}%"})
                logger.info(f"批次处理用时: {time.time() - batch_start:.2f}秒")

            except Exception as e:
                logger.error(f"处理文件出错: {os.path.basename(file_path)}")
                logger.error(traceback.format_exc())
                pbar.update(1)

    # 处理完成后，合并和保存最终结果
    total_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"图谱相似度计算与去冗余处理完成")
    logger.info(f"总处理: {total_graph_count} 个图谱")
    logger.info(f"保留: {total_retained} 个图谱 (保留率: {total_retained * 100 / total_graph_count:.1f}%)")
    logger.info(f"去冗余: {total_graph_count - total_retained} 个图谱")
    logger.info(f"总耗时: {total_time:.2f} 秒 (平均 {total_time / total_graph_count * 1000:.2f} 毫秒/图谱)")

    # 保存统计信息
    save_statistics(output_dir, total_graph_count, total_retained)

    # 保存保留的ID列表
    retained_ids_file = os.path.join(output_dir, "similarity_retained_ids.json")
    with open(retained_ids_file, 'w') as f:
        json.dump(list(retained_graph_ids), f)
    logger.info(f"保留的图谱ID已保存至: {retained_ids_file}")

    # 如果需要，合并所有结果到一个文件
    if result_files and len(result_files) > 0:
        merged_file = os.path.join(output_dir, "merged_nonredundant_graphs.pt")
        try:
            logger.info(f"合并 {len(result_files)} 个结果文件...")
            merged_graphs = {}

            for result_file in tqdm(result_files, desc="合并结果"):
                batch_graphs = safe_load_graph(result_file)
                merged_graphs.update(batch_graphs)

            # 保存合并结果
            torch.save(merged_graphs, merged_file)
            logger.info(f"合并完成，保存 {len(merged_graphs)} 个非冗余图谱至: {merged_file}")

            # 更新元数据
            meta_output_path = os.path.join(output_dir, "similarity_meta.json")
            meta_info = {
                "original_count": total_graph_count,
                "filtered_count": total_retained,
                "retention_rate": float(total_retained) / total_graph_count if total_graph_count > 0 else 0,
                "similarity_threshold": similarity_threshold,
                "processing_time": total_time,
                "timestamp": time.time(),
                "date": time.strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(meta_output_path, 'w') as f:
                json.dump(meta_info, f, indent=2)

        except Exception as e:
            logger.error(f"合并结果文件时出错: {str(e)}")

    return result_files


def save_statistics(output_dir, original_count, filtered_count):
    """
    保存去冗余统计信息

    参数:
        output_dir: 输出目录路径
        original_count: 原始图谱数量
        filtered_count: 过滤后图谱数量
    """
    if original_count == 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "图谱相似度去冗余报告.txt")
    retention_rate = filtered_count * 100 / original_count
    redundancy_rate = 100 - retention_rate

    with open(summary_file, 'w') as f:
        f.write("蛋白质图谱相似度去冗余统计报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"原始图谱总数: {original_count}\n")
        f.write(f"保留图谱数量: {filtered_count}\n")
        f.write(f"去冗余图谱数: {original_count - filtered_count}\n\n")
        f.write(f"保留率: {retention_rate:.2f}%\n")
        f.write(f"冗余率: {redundancy_rate:.2f}%\n\n")
        f.write("=" * 40 + "\n")
        f.write("说明: 本系统使用图神经网络特征和FAISS向量相似度进行图谱去冗余\n")

    # 同时保存JSON格式便于程序读取
    json_file = os.path.join(output_dir, "similarity_stats.json")
    stats = {
        "timestamp": time.time(),
        "date": time.strftime('%Y-%m-%d %H:%M:%S'),
        "original_count": original_count,
        "filtered_count": filtered_count,
        "retention_rate": retention_rate,
        "redundancy_rate": redundancy_rate
    }

    with open(json_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"统计信息已保存至: {summary_file} 和 {json_file}")


def load_retained_sequence_ids(seq_ids_path):
    """
    加载已保留的序列ID列表

    参数:
        seq_ids_path: 序列ID文件路径

    返回:
        set: 保留的序列ID集合
    """
    logger.info(f"从文件加载保留的序列ID: {seq_ids_path}")

    if not os.path.exists(seq_ids_path):
        logger.error(f"序列ID文件不存在: {seq_ids_path}")
        return set()

    try:
        with open(seq_ids_path, 'r') as f:
            data = json.load(f)

        # 处理不同格式的ID文件
        if isinstance(data, list):
            # 直接是ID列表
            retained_ids = set(data)
        elif isinstance(data, dict):
            # 包含在字典中
            if "retained_ids" in data:
                retained_ids = set(data["retained_ids"])
            else:
                # 尝试提取所有有效ID
                retained_ids = set()
                for key, value in data.items():
                    if isinstance(value, dict) and "sequence" in value:
                        retained_ids.add(key)
                    elif isinstance(value, str):
                        retained_ids.add(key)
        else:
            logger.error("不支持的ID文件格式")
            return set()

        logger.info(f"成功加载 {len(retained_ids)} 个序列ID")
        return retained_ids

    except Exception as e:
        logger.error(f"加载序列ID文件出错: {str(e)}")
        return set()


def increase_file_limit():
    """增加系统允许打开的文件数量"""
    try:
        import resource
        # 获取当前限制
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info(f"当前文件描述符限制: soft={soft}, hard={hard}")

        # 设置新的软限制（不超过硬限制）
        new_soft = max(hard, 65536)  # 设置为65536或硬限制中较小者
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

        # 验证新限制是否生效
        new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info(f"文件描述符限制: {soft} → {new_soft} (硬限制: {hard})")
        return True
    except (ImportError, ValueError, ResourceError) as e:
        logger.warning(f"无法增加文件描述符限制: {str(e)}")
        return False


def main():
    """主函数"""
    global logger
    parser = argparse.ArgumentParser(description="蛋白质知识图谱去冗余工具")

    # 输入输出参数
    parser.add_argument("--cache_dir", "-c", type=str, required=True,
                        help="图谱缓存目录路径")
    parser.add_argument("--cache_id", "-ci", type=str, default="graph_data_cache",
                        help="图谱缓存ID前缀 (默认: graph_data_cache)")
    parser.add_argument("--output_dir", "-o", type=str, default="./filtered_graphs",
                        help="输出目录路径 (默认: ./filtered_graphs)")

    # 去冗余参数
    parser.add_argument("--similarity_threshold", "-st", type=float, default=0.85,
                        help="图谱相似度阈值 (默认: 0.85)")

    # 性能参数
    parser.add_argument("--processing_batch_size", "-pbs", type=int, default=2000,
                        help="特征提取批处理大小 (默认: 1000)")
    parser.add_argument("--file_batch_size", "-fbs", type=int, default=50000,
                        help="文件加载批处理大小 (默认: 50000)")

    # 采样参数
    parser.add_argument("--use_sampling", "-s", action="store_true", default=False,
                        help="对大数据集进行采样处理 (默认: False)")
    parser.add_argument("--sample_ratio", "-sr", type=float, default=0.1,
                        help="采样比例 (默认: 0.1)")

    args = parser.parse_args()

    # 尝试增加文件描述符限制
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = max(hard, 65536)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        logger.info(f"文件描述符限制: {soft} → {new_soft}")
    except:
        pass

    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    logger, log_file_path = setup_logging(args.output_dir)
    logger.info(f"日志将写入文件: {log_file_path}")

    # 记录系统资源状态
    log_system_resources()

    # 打印运行配置
    logger.info("运行配置:")
    logger.info(f"- 图谱缓存目录: {args.cache_dir}")
    logger.info(f"- 图谱缓存ID前缀: {args.cache_id}")
    logger.info(f"- 输出目录: {args.output_dir}")
    logger.info(f"- 图谱相似度阈值: {args.similarity_threshold}")
    logger.info(f"- 特征提取批处理大小: {args.processing_batch_size}")
    logger.info(f"- 文件加载批处理大小: {args.file_batch_size}")
    logger.info(f"- 使用采样处理: {'是' if args.use_sampling else '否'}")
    if args.use_sampling:
        logger.info(f"- 采样比例: {args.sample_ratio}")

    try:
        # 检查FAISS是否可用
        try:
            import faiss
            has_faiss = True
            logger.info("FAISS库已安装，将用于图谱去冗余")
        except ImportError:
            has_faiss = False
            logger.error("未安装FAISS库，无法进行图谱去冗余")
            return

        # 使用新的批处理方法
        result_files = load_and_process_graphs_in_batches(
            cache_dir=args.cache_dir,
            cache_id=args.cache_id,
            output_dir=args.output_dir,
            similarity_threshold=args.similarity_threshold,
            batch_size=args.file_batch_size,
            processing_batch_size=args.processing_batch_size
        )

        if result_files:
            logger.info(f"处理完成！生成了 {len(result_files)} 个结果文件")
        else:
            logger.error("处理失败，未生成结果文件")

    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")
        logger.error(traceback.format_exc())

    # 再次记录系统资源状态
    log_system_resources()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if 'logger' in globals():
            logger.exception(f"程序执行时出现错误: {str(e)}")
        print(f"错误: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)