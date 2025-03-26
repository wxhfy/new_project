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
            logger.info(f"ID示例: {', '.join(id_sample[:3])}...")

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
            logging.error(f"文件不存在: {file_path}")
            return {}

        # 读取文件内容到内存
        with open(file_path, 'rb') as f:
            file_content = f.read()
            if not file_content:
                logging.error(f"文件为空: {file_path}")
                return {}

            # 创建内存缓冲区
            buffer = io.BytesIO(file_content)

            # 直接从内存缓冲区加载，不使用mmap
            result = torch.load(buffer, map_location=map_location, mmap=False)
            buffer.close()

            # 确保返回值是可迭代的
            if result is None:
                logging.error(f"加载结果为None: {file_path}")
                return {}

            return result

    except Exception as e:
        logging.error(f"加载文件失败: {file_path}, 错误: {str(e)}")
        return {}  # 返回空字典而不是None，避免迭代错误


def load_and_process_graphs_in_batches(cache_dir, cache_id, output_dir, similarity_threshold=0.85,
                                       batch_size=50000, processing_batch_size=1000):
    """
    分批加载和处理图谱，直接进行特征提取和去冗余

    参数:
        cache_dir: 缓存目录路径
        cache_id: 缓存ID前缀
        output_dir: 输出目录路径
        similarity_threshold: 相似度阈值
        batch_size: 每次从文件加载的图谱数量
        processing_batch_size: 特征提取的批处理大小
    """
    # 查找所有缓存文件
    cache_files = sorted(glob.glob(os.path.join(cache_dir, f"{cache_id}_part_*.pt")))
    if not cache_files:
        logger.error(f"未找到任何缓存文件: {os.path.join(cache_dir, f'{cache_id}_part_*.pt')}")
        return {}

    logger.info(f"找到 {len(cache_files)} 个缓存文件，将分批处理")

    # 初始化Faiss索引
    try:
        import faiss
        # 使用较小的图谱先估计特征维度
        test_file = cache_files[0]
        test_data = safe_load_graph(test_file, map_location='cpu')
        if not test_data:
            logger.error("无法加载测试文件估计特征维度")
            return {}

        # 提取一个测试特征来确定维度
        test_id = next(iter(test_data))
        test_feature = extract_graph_feature(test_data[test_id])
        feature_dim = len(test_feature)
        logger.info(f"特征维度估计: {feature_dim}")

        # 初始化FAISS索引
        index = faiss.IndexFlatL2(feature_dim)
        scaler = StandardScaler()

        # 用于训练StandardScaler的样本数据
        sample_features = []

        # 随机采样一些图谱用于训练标准化器
        sample_count = min(500000, len(test_data))
        sample_ids = list(test_data.keys())[:sample_count]

        for graph_id in sample_ids:
            feature = extract_graph_feature(test_data[graph_id])
            if feature is not None and not np.isnan(feature).any():
                sample_features.append(feature)

        # 如果有足够的样本，训练标准化器
        if len(sample_features) > 0:
            sample_features = np.vstack(sample_features)
            scaler.fit(sample_features)
            logger.info(f"使用 {len(sample_features)} 个样本训练了标准化器")

        # 释放测试数据
        del test_data
        del sample_features
        gc.collect()

    except ImportError:
        logger.error("未安装FAISS库，无法继续处理")
        return {}

    # 用于存储已保留和已处理的图谱ID
    retained_graph_ids = set()
    processed_ids = set()

    # 总图谱计数
    total_graph_count = 0
    total_retained = 0

    # 为保存结果创建目录
    os.makedirs(os.path.join(output_dir, "graph_cache"), exist_ok=True)

    # 创建临时文件用于保存中间结果
    temp_output_dir = os.path.join(output_dir, "temp_results")
    os.makedirs(temp_output_dir, exist_ok=True)

    # 保存中间结果的文件列表
    result_files = []

    # 处理每个缓存文件
    for file_idx, file_path in enumerate(cache_files):
        logger.info(f"处理文件 {file_idx + 1}/{len(cache_files)}: {os.path.basename(file_path)}")

        try:
            # 加载当前文件中的图谱
            graphs_batch = safe_load_graph(file_path, map_location='cpu')

            if not graphs_batch:
                logger.warning(f"文件为空或加载失败: {os.path.basename(file_path)}")
                continue

            total_graph_count += len(graphs_batch)
            logger.info(f"加载了 {len(graphs_batch)} 个图谱，总计: {total_graph_count}")

            # 提取特征并进行去冗余
            batch_retained_ids = set()

            # 只处理尚未处理的图谱
            new_graphs = {id: graph for id, graph in graphs_batch.items() if id not in processed_ids}
            logger.info(f"本批次需要处理 {len(new_graphs)} 个新图谱")

            if not new_graphs:
                logger.info("本批次没有新图谱，跳过处理")
                continue

            # 提取特征
            features = {}
            graph_ids = list(new_graphs.keys())

            # 分批提取特征，减少内存压力
            for i in range(0, len(graph_ids), processing_batch_size):
                batch_ids = graph_ids[i:i + processing_batch_size]

                for graph_id in batch_ids:
                    try:
                        graph = new_graphs[graph_id]
                        feature = extract_graph_feature(graph)
                        if feature is not None and not np.isnan(feature).any():
                            features[graph_id] = feature
                    except Exception as e:
                        pass

                # 定期执行垃圾回收
                if i % (processing_batch_size * 10) == 0:
                    gc.collect()

            logger.info(f"提取了 {len(features)} 个特征向量")

            # 如果有特征，进行标准化和去冗余
            if features:
                # 将特征转换为矩阵
                feature_ids = list(features.keys())
                feature_matrix = np.array([features[gid] for gid in feature_ids])

                # 标准化特征
                feature_matrix = scaler.transform(feature_matrix)

                # 计算复杂度作为排序依据
                complexity = np.array([feature_matrix[i][0] + feature_matrix[i][1] for i in range(len(feature_matrix))])
                sorted_indices = np.argsort(-complexity)  # 降序

                # 添加到索引
                index.add(feature_matrix)

                # 进行去冗余
                for idx in sorted_indices:
                    graph_id = feature_ids[idx]

                    # 如果已处理，跳过
                    if graph_id in processed_ids:
                        continue

                    # 将当前图谱标记为已处理
                    processed_ids.add(graph_id)

                    # 保留当前图谱
                    batch_retained_ids.add(graph_id)
                    retained_graph_ids.add(graph_id)

                    # 查询相似图谱
                    query_vector = feature_matrix[idx:idx + 1]
                    k = min(100, len(feature_matrix))
                    distances, neighbors = index.search(query_vector, k)

                    # 将相似图谱标记为已处理
                    for j, dist in zip(neighbors[0][1:], distances[0][1:]):
                        if j < len(feature_ids):
                            similarity = 1.0 - (dist / 2.0)
                            if similarity > similarity_threshold:
                                similar_id = feature_ids[j]
                                processed_ids.add(similar_id)

            # 为当前批次构建去冗余后的图谱
            batch_filtered_graphs = {gid: new_graphs[gid] for gid in batch_retained_ids if gid in new_graphs}
            total_retained += len(batch_filtered_graphs)

            logger.info(f"本批次保留 {len(batch_filtered_graphs)}/{len(new_graphs)} 个图谱，总保留: {total_retained}")

            # 保存当前批次结果
            if batch_filtered_graphs:
                result_file = os.path.join(temp_output_dir, f"filtered_batch_{file_idx}.pt")
                torch.save(batch_filtered_graphs, result_file)
                result_files.append(result_file)

                logger.info(f"保存批次结果到: {result_file}")

            # 释放内存
            del graphs_batch
            del new_graphs
            del features
            del feature_matrix
            gc.collect()

        except Exception as e:
            logger.error(f"处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
            logger.error(traceback.format_exc())

    # 处理完所有文件后，合并结果
    logger.info(f"所有文件处理完成，合并最终结果...")
    logger.info(
        f"总共有 {total_graph_count} 个图谱，保留了 {total_retained} 个，保留率: {total_retained * 100 / total_graph_count:.1f}%")

    # 保存最终统计结果
    save_statistics(output_dir, total_graph_count, total_retained)

    # 保存ID列表
    retained_ids_file = os.path.join(output_dir, "retained_graph_ids.json")
    with open(retained_ids_file, 'w') as f:
        json.dump(list(retained_graph_ids), f)

    logger.info(f"保留的图谱ID已保存到: {retained_ids_file}")

    # 返回结果文件列表，可用于后续处理
    return result_files


def extract_graph_feature(graph):
    """从图谱中提取特征向量 - 优化版本"""
    # 初始化基本特征
    num_nodes = 0
    num_edges = 0
    avg_degree = 0

    try:
        # 预先检查图谱类型
        is_pyg_format = hasattr(graph, 'x') and hasattr(graph, 'edge_index')
        is_dict_format = isinstance(graph, dict)

        if is_pyg_format:
            # PyG格式处理
            if hasattr(graph, 'x') and graph.x is not None:
                num_nodes = graph.x.shape[0]

            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                num_edges = graph.edge_index.shape[1] // 2

            # 计算平均度
            avg_degree = num_edges / num_nodes if num_nodes > 0 else 0

            # 提取节点特征
            if num_nodes > 0 and hasattr(graph, 'x') and graph.x is not None:
                with torch.no_grad():  # 避免创建计算图
                    x = graph.x.cpu().numpy() if torch.is_tensor(graph.x) else graph.x

                    # 计算统计量
                    if len(x) > 0:
                        node_feat_mean = np.nanmean(x, axis=0)
                        node_feat_std = np.nanstd(x, axis=0)

                        # 处理可能的NaN值
                        node_feat_mean = np.nan_to_num(node_feat_mean)
                        node_feat_std = np.nan_to_num(node_feat_std)

                        # 合并特征
                        return np.concatenate([[num_nodes, num_edges, avg_degree],
                                               node_feat_mean, node_feat_std]).astype(np.float32)

        elif is_dict_format:
            # 字典格式处理
            if 'nodes' in graph:
                num_nodes = len(graph['nodes'])
            if 'edges' in graph:
                num_edges = len(graph['edges'])

            avg_degree = num_edges / num_nodes if num_nodes > 0 else 0

            if 'node_features' in graph and graph['node_features'] is not None:
                with torch.no_grad():  # 避免创建计算图
                    x = graph['node_features'].cpu().numpy() if torch.is_tensor(graph['node_features']) else graph[
                        'node_features']

                    if len(x) > 0:
                        node_feat_mean = np.nanmean(x, axis=0)
                        node_feat_std = np.nanstd(x, axis=0)

                        node_feat_mean = np.nan_to_num(node_feat_mean)
                        node_feat_std = np.nan_to_num(node_feat_std)

                        return np.concatenate([[num_nodes, num_edges, avg_degree],
                                               node_feat_mean, node_feat_std]).astype(np.float32)

        # 如果无法提取详细特征，返回基本特征
        return np.array([num_nodes, num_edges, avg_degree, 0, 0], dtype=np.float32)

    except Exception as e:
        # 返回默认特征
        return np.array([0, 0, 0, 0, 0], dtype=np.float32)


# 定义批处理函数
def process_batch(batch_ids, graphs):
    batch_features = {}
    for graph_id in batch_ids:
        try:
            graph = graphs[graph_id]
            feature = extract_graph_feature(graph)
            if feature is not None and not np.isnan(feature).any():
                batch_features[graph_id] = feature
        except Exception as e:
            pass  # 简化错误处理
    return batch_features


def parallel_extract_features(graphs, num_workers=32, batch_size=1000):
    """
    内存安全的并行图谱特征提取 - 增加内存监控和自动调整
    """
    features = {}
    graph_ids = list(graphs.keys())
    total_graphs = len(graph_ids)

    # 动态调整参数
    # 1. 根据图谱数量调整工作进程数和批处理大小
    if total_graphs > 1000000:  # 非常大的数据集
        suggested_workers = min(num_workers, 4)
        suggested_batch = min(batch_size, 200)
    elif total_graphs > 100000:  # 大型数据集
        suggested_workers = min(num_workers, 8)
        suggested_batch = min(batch_size, 500)
    else:  # 适度大小的数据集
        suggested_workers = min(num_workers, 16)
        suggested_batch = min(batch_size, 1000)

    # 2. 确保工作进程数不超过系统资源限制
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        # 估算每个进程需要的文件描述符数量
        safe_workers = max(2, min(suggested_workers, (soft // 4) // 20))
        if safe_workers < suggested_workers:
            logger.warning(f"调整工作进程数: {suggested_workers} → {safe_workers} (文件描述符限制)")
            suggested_workers = safe_workers
    except:
        # 如果无法获取系统限制，采用保守设置
        if suggested_workers > 8:
            suggested_workers = 8
            logger.warning(f"未能确定系统资源限制，设置保守的工作进程数: {suggested_workers}")

    # 3. 监控系统内存使用情况并调整参数
    mem = psutil.virtual_memory()
    if mem.percent > 80:  # 内存使用率高
        logger.warning(f"系统内存使用率高 ({mem.percent}%)，减小批处理大小和工作进程数")
        suggested_batch = max(50, suggested_batch // 2)
        suggested_workers = max(2, suggested_workers // 2)

    logger.info(f"使用 {suggested_workers} 个工作进程，批处理大小: {suggested_batch}")

    # 创建更合理大小的批次
    batches = [graph_ids[i:i + suggested_batch] for i in range(0, total_graphs, suggested_batch)]
    logger.info(f"将任务分为 {len(batches)} 个批次")

    # 计算分组大小，控制同时运行的进程数
    max_concurrent = max(1, min(suggested_workers, 4))
    batch_groups = [batches[i:i + max_concurrent] for i in range(0, len(batches), max_concurrent)]

    total_extracted = 0
    for group_idx, group_batches in enumerate(batch_groups):
        logger.info(f"处理批次组 {group_idx + 1}/{len(batch_groups)}")

        # 为当前组创建进程池并执行
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(group_batches)) as executor:
            future_to_batch = {executor.submit(process_batch, batch, graphs): i
                               for i, batch in enumerate(group_batches)}

            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_features = future.result()
                    total_extracted += len(batch_features)
                    features.update(batch_features)
                except Exception as e:
                    logger.error(f"批次处理失败: {str(e)}")

        # 每组完成后强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 计算进度并报告
        progress = (group_idx + 1) * 100 / len(batch_groups)
        logger.info(f"特征提取进度: {progress:.1f}%, 已提取 {total_extracted}/{total_graphs} 个特征")

        # 再次检查内存使用，如果内存使用率过高，调整参数
        mem = psutil.virtual_memory()
        if mem.percent > 85:
            logger.warning(f"内存使用率高 ({mem.percent}%)，减小后续组的大小")
            max_concurrent = max(1, max_concurrent - 1)

    success_rate = len(features) * 100 / len(graphs) if graphs else 0
    logger.info(f"特征提取完成: 成功率 {success_rate:.1f}% ({len(features)}/{len(graphs)})")

    return features


def sequential_extract_features(graphs, batch_size=5000):
    """单进程提取图谱特征 - 当并行处理失败时的备用方案"""
    features = {}
    graph_ids = list(graphs.keys())

    # 创建批次，但不并行处理
    batches = [graph_ids[i:i + batch_size] for i in range(0, len(graph_ids), batch_size)]
    logger.info(f"使用单进程模式处理 {len(batches)} 个批次")

    with tqdm(total=len(batches), desc="单进程特征提取") as pbar:
        for i, batch in enumerate(batches):
            # 处理单个批次
            batch_features = {}
            for graph_id in batch:
                try:
                    graph = graphs[graph_id]
                    feature = extract_graph_feature(graph)
                    if feature is not None and not np.isnan(feature).any():
                        batch_features[graph_id] = feature
                except Exception as e:
                    pass

            # 更新结果
            features.update(batch_features)
            pbar.update(1)
            pbar.set_postfix({"已处理": len(features)})

            # 定期垃圾回收
            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info(f"单进程特征提取完成: 共提取 {len(features)}/{len(graphs)} 个图谱特征")
    return features



def save_statistics(output_dir, original_count, filtered_count):
    """
    保存过滤统计信息

    参数:
        output_dir: 输出目录路径
        original_count: 原始图谱数量
        filtered_count: 过滤后图谱数量
    """
    if original_count == 0:
        return

    summary_file = os.path.join(output_dir, "graph_deduplication_summary.txt")
    retention_rate = filtered_count * 100 / original_count
    redundancy_rate = 100 - retention_rate

    with open(summary_file, 'w') as f:
        f.write(f"图谱去冗余统计信息 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
        f.write(f"原始图谱数量: {original_count}\n")
        f.write(f"过滤后图谱数量: {filtered_count}\n")
        f.write(f"保留率: {retention_rate:.2f}%\n")
        f.write(f"冗余率: {redundancy_rate:.2f}%\n")
        f.write("-" * 50 + "\n")

    logger.info(f"统计信息已保存至: {summary_file}")

    # 同时保存JSON格式的统计信息，方便程序读取
    json_file = os.path.join(output_dir, "deduplication_stats.json")
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
                # 可能是其他格式，尝试提取所有值
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