#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图谱处理与去冗余后处理工具

该脚本负责处理由removal.py生成的缓存结果，执行以下操作：
1. 加载sequence去冗余后的序列数据
2. 加载对应的图谱数据
3. 执行图谱去冗余
4. 保存最终的过滤结果

作者: wxhfy
"""

import argparse
import glob
import io
import json
import logging
import os
import time
import gc
import concurrent.futures
import pickle
import sys
import traceback

import psutil
import torch
import numpy as np
from tqdm import tqdm
import faiss

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging(output_dir):
    """设置日志系统"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"graph_postprocess_{time.strftime('%Y%m%d_%H%M%S')}.log")

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


def check_memory_usage(threshold_gb=None, force_gc=False):
    """
    检查内存使用情况，并在需要时执行垃圾回收

    参数:
        threshold_gb: 内存使用阈值(GB)，超过此值返回True
        force_gc: 是否强制执行垃圾回收

    返回:
        bool: 内存使用是否超过阈值
    """
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
                    logger.info(f"GPU {i}: 已分配 {allocated:.1f}GB, 已保留 {reserved:.1f}GB, 总计 {max_mem:.1f}GB")
                except:
                    logger.info(f"GPU {i}: 无法获取内存信息")
    except Exception as e:
        logger.warning(f"无法获取系统资源信息: {str(e)}")


def find_batch_directories(input_path):
    """查找所有批次目录"""
    batch_dirs = []
    if os.path.isdir(input_path):
        # 查找所有批次目录
        for item in os.listdir(input_path):
            if item.startswith("batch_") and os.path.isdir(os.path.join(input_path, item)):
                batch_dirs.append(os.path.join(input_path, item))

        # 如果没有找到批次目录，将输入路径作为单个目录
        if not batch_dirs:
            batch_dirs = [input_path]

    return sorted(batch_dirs)


def check_graph_files_exist(batch_dirs):
    """检查是否存在知识图谱文件"""
    for batch_dir in batch_dirs:
        kg_pyg_dir = os.path.join(batch_dir, "knowledge_graphs_pyg")
        if os.path.exists(kg_pyg_dir):
            pt_files = glob.glob(os.path.join(kg_pyg_dir, "protein_kg_chunk_*.pt"))
            if pt_files:
                return True
    return False


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


def get_file_stats(file_path):
    """获取文件统计信息"""
    try:
        stats = os.stat(file_path)
        return {
            'path': file_path,
            'size': stats.st_size,
            'mtime': stats.st_mtime
        }
    except Exception as e:
        logger.error(f"获取文件统计信息失败 {file_path}: {str(e)}")
        return {'path': file_path, 'size': 0, 'mtime': 0}


def check_pt_file_content(file_path):
    """检查PT文件包含的图谱ID（禁用mmap）"""
    try:
        # 显式禁用内存映射加载
        data = safe_load_graph(file_path, map_location='cpu')
        if isinstance(data, dict):
            file_ids = set(data.keys())
            return file_path, file_ids
    except Exception as e:
        logger.debug(f"检查文件 {os.path.basename(file_path)} 失败: {str(e)}")
    return file_path, set()


def process_file_batch(batch_files, target_ids, id_mapping):
    """处理文件批次，加载包含目标ID的图谱（禁用mmap）"""
    batch_start_time = time.time()
    batch_graphs = {}
    processed_count = 0

    # 处理每个文件
    for file_path in batch_files:
        try:
            # 检查此文件是否包含目标ID
            file_ids = id_mapping.get(file_path, set())
            common_ids = set(target_ids) & file_ids

            if not common_ids:
                continue

            # 只加载包含目标ID的图谱
            graphs_data = safe_load_graph(file_path, map_location='cpu')

            if not graphs_data or not isinstance(graphs_data, dict):
                logger.warning(f"文件格式不正确或为空: {os.path.basename(file_path)}")
                continue

            # 只提取所需的ID
            for graph_id in common_ids:
                if graph_id in graphs_data:
                    batch_graphs[graph_id] = graphs_data[graph_id]
                    processed_count += 1

            # 释放内存
            del graphs_data

            # 定期检查内存使用情况
            if processed_count % 10000 == 0:
                check_memory_usage(threshold_gb=900, force_gc=True)

        except Exception as e:
            logger.debug(f"处理文件失败: {os.path.basename(file_path)}, 错误: {str(e)}")

    batch_time = time.time() - batch_start_time

    # 直接返回三个值
    return batch_graphs, processed_count, batch_time


def load_cached_graphs(cache_dir, cache_id, selected_ids_set):
    """
    超高速缓存加载器 - 无mmap版本
    """
    cached_graphs = {}
    cache_meta_file = os.path.join(cache_dir, f"{cache_id}_meta.json")

    if not os.path.exists(cache_meta_file):
        return cached_graphs

    try:
        # 查找所有缓存文件并按修改时间排序 - 修改扩展名为pt
        cache_files = glob.glob(os.path.join(cache_dir, f"{cache_id}_part_*.pt"))
        if not cache_files:
            return cached_graphs

        cache_files.sort(key=os.path.getmtime, reverse=True)
        logger.info(f"发现 {len(cache_files)} 个图谱缓存文件")

        # 将文件分成4批处理以平衡内存使用
        batch_size = max(1, len(cache_files) // 4)
        batches = [cache_files[i:i + batch_size] for i in range(0, len(cache_files), batch_size)]

        total_loaded = 0
        with tqdm(total=len(cache_files), desc="加载缓存文件", ncols=100) as pbar:
            # 处理每个批次
            for batch_idx, batch_files in enumerate(batches):
                # 统计当前批次加载信息
                batch_loaded = 0

                # 创建线程池处理当前批次的文件
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(batch_files))) as executor:
                    batch_results = {}
                    future_to_file = {
                        executor.submit(safe_load_graph, file_path, 'cpu'): file_path
                        for file_path in batch_files
                    }

                    # 处理完成的任务
                    for future in concurrent.futures.as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            file_data = future.result()

                            # 保存当前文件的数据
                            if file_data and isinstance(file_data, dict):
                                # 仅保留目标ID的图谱
                                for graph_id, graph in file_data.items():
                                    if graph_id in selected_ids_set and graph_id not in cached_graphs:
                                        cached_graphs[graph_id] = graph
                                        batch_loaded += 1

                            # 更新进度条
                            pbar.update(1)

                        except Exception as e:
                            logger.debug(f"处理缓存文件失败: {os.path.basename(file_path)}, 错误: {str(e)}")

                # 更新总计数
                total_loaded += batch_loaded
                logger.info(f"批次 {batch_idx + 1}/{len(batches)} 完成，此批次加载 {batch_loaded} 个图谱")

                # 每批次后强制垃圾回收
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 计算覆盖率
        coverage = len(cached_graphs) * 100 / len(selected_ids_set) if selected_ids_set else 0
        logger.info(f"缓存加载完成: 命中率 {coverage:.1f}% ({len(cached_graphs)}/{len(selected_ids_set)})")

    except Exception as e:
        logger.error(f"加载缓存失败: {str(e)}")
        traceback.print_exc()

    return cached_graphs


def process_file_mapping_chunk(chunk, remaining_ids):
    local_id_to_files = {}
    local_matched_files = set()

    for file_path, file_ids in chunk.items():
        # 计算与目标ID集合的交集
        common_ids = remaining_ids.intersection(file_ids)
        if common_ids:  # 如果有交集，则此文件需要加载
            local_matched_files.add(file_path)
            for graph_id in common_ids:
                if graph_id not in local_id_to_files:
                    local_id_to_files[graph_id] = []
                local_id_to_files[graph_id].append(file_path)

    return local_id_to_files, local_matched_files


def accelerated_graph_loader(input_path, filtered_ids, num_workers=128, memory_limit_gb=900,
                             use_cache=True, cache_dir=None, cache_id="graphs_cache", use_mmap=False):
    """
    加速版图谱加载器 - 优先从缓存加载，然后按需从源文件加载缺失部分

    参数:
        input_path: 输入目录路径
        filtered_ids: 已过滤的ID列表
        num_workers: 工作进程数
        memory_limit_gb: 内存使用限制(GB)
        use_cache: 是否使用缓存
        cache_dir: 缓存目录路径
        cache_id: 缓存ID前缀
        use_mmap: 是否使用内存映射加载

    返回:
        dict: 加载的图谱字典
    """
    # 构建ID集合以加速查找
    filtered_ids_set = set(filtered_ids)
    logger.info(f"图谱加载目标: {len(filtered_ids_set)}个ID")

    # 初始化结果容器
    loaded_graphs = {}
    remaining_ids = filtered_ids_set.copy()  # 跟踪尚未加载的ID

    # 第一步: 尝试从缓存加载
    if use_cache and cache_dir and os.path.exists(cache_dir):
        logger.info(f"从缓存目录加载图谱: {cache_dir}")
        cache_start = time.time()

        # 从缓存加载图谱
        cached_graphs = load_cached_graphs(cache_dir, cache_id, filtered_ids_set)

        if cached_graphs:
            # 更新已加载和待加载的ID集合
            loaded_graphs.update(cached_graphs)
            cached_ids = set(cached_graphs.keys())
            remaining_ids = filtered_ids_set - cached_ids

            logger.info(f"从缓存中加载了 {len(cached_graphs)} 个图谱，剩余 {len(remaining_ids)} 个需要从源文件加载")
            logger.info(f"缓存加载耗时: {time.time() - cache_start:.1f}秒")

            # 释放内存
            del cached_graphs
            check_memory_usage(force_gc=True)
        else:
            logger.info("缓存中未找到有效图谱数据")

    # 如果所有ID已从缓存加载，直接返回
    if not remaining_ids:
        logger.info("所有请求的图谱已从缓存加载完成")
        return loaded_graphs

    # 第二步: 从源文件加载剩余ID
    logger.info(f"将从源文件加载 {len(remaining_ids)} 个图谱")

    # 查找所有批次目录
    batch_dirs = find_batch_directories(input_path)
    logger.info(f"找到 {len(batch_dirs)} 个批次目录")

    # 收集所有图谱文件
    pt_files = []
    for batch_dir in batch_dirs:
        kg_pyg_dir = os.path.join(batch_dir, "knowledge_graphs_pyg")
        if os.path.exists(kg_pyg_dir):
            files = glob.glob(os.path.join(kg_pyg_dir, "protein_kg_chunk_*.pt"))
            pt_files.extend(files)

    if not pt_files:
        logger.warning("未找到任何图谱文件")
        return loaded_graphs

    logger.info(f"找到 {len(pt_files)} 个图谱文件")

    # 为提高效率，首先扫描所有文件以确定ID到文件的映射
    logger.info("扫描文件内容以确定ID到文件的映射...")
    file_to_ids = {}  # 文件路径 -> ID集合

    # 并行扫描所有文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(check_pt_file_content, file_path) for file_path in pt_files]

        with tqdm(total=len(pt_files), desc="扫描文件", ncols=100) as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    file_path, file_ids = future.result()
                    if file_ids:
                        file_to_ids[file_path] = file_ids
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"扫描文件时出错: {str(e)}")

    # 构建ID到文件的映射和需要加载的文件集合
    id_to_files = {}  # ID -> 文件路径列表
    matched_files = set()  # 包含目标ID的文件路径集合

    # 将映射任务分成多个块并行处理
    chunk_size = max(1, len(file_to_ids) // num_workers)
    file_to_ids_items = list(file_to_ids.items())
    chunks = [dict(file_to_ids_items[i:i + chunk_size]) for i in range(0, len(file_to_ids_items), chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_workers, len(chunks))) as executor:
        futures = [executor.submit(process_file_mapping_chunk, chunk, remaining_ids) for chunk in chunks]

        for future in concurrent.futures.as_completed(futures):
            try:
                local_id_to_files, local_matched_files = future.result()
                id_to_files.update(local_id_to_files)
                matched_files.update(local_matched_files)
            except Exception as e:
                logger.error(f"处理文件映射时出错: {str(e)}")

    logger.info(f"找到 {len(matched_files)} 个包含目标ID的文件")

    # 将文件划分成批次并行加载
    batch_size = max(1, len(matched_files) // num_workers)
    matched_files_list = list(matched_files)
    batches = [matched_files_list[i:i + batch_size] for i in range(0, len(matched_files_list), batch_size)]

    logger.info(f"文件分为 {len(batches)} 个批次进行加载")

    # 并行加载每个批次
    total_processed = 0
    total_start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_workers, len(batches))) as executor:
        futures = [executor.submit(process_file_batch, batch, list(remaining_ids), file_to_ids) for batch in batches]

        with tqdm(total=len(batches), desc="加载文件批次", ncols=100) as pbar:
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    batch_graphs, processed_count, batch_time = future.result()

                    # 更新结果
                    loaded_graphs.update(batch_graphs)
                    total_processed += processed_count

                    # 更新剩余ID
                    remaining_ids -= set(batch_graphs.keys())

                    # 更新进度
                    pbar.update(1)
                    pbar.set_postfix({
                        'loaded': len(loaded_graphs),
                        'remaining': len(remaining_ids),
                        'elapsed': f"{(time.time() - total_start_time):.1f}s"
                    })

                    # 检查内存使用
                    check_memory_usage(threshold_gb=memory_limit_gb)

                except Exception as e:
                    logger.error(f"处理批次时出错: {str(e)}")

    # 计算统计信息
    total_time = time.time() - total_start_time
    coverage = len(loaded_graphs) * 100 / len(filtered_ids_set) if filtered_ids_set else 0

    logger.info(f"图谱加载完成:")
    logger.info(f"- 总共加载: {len(loaded_graphs)}/{len(filtered_ids_set)} 个图谱 (覆盖率: {coverage:.1f}%)")
    logger.info(
        f"- 总耗时: {total_time:.1f}秒 (平均 {total_time / len(loaded_graphs):.3f}秒/图谱)" if loaded_graphs else "- 总耗时: 0秒")
    logger.info(f"- 未找到的图谱数: {len(remaining_ids)}")

    return loaded_graphs


def save_filtered_data(filtered_sequences, filtered_graphs, output_dir, chunk_size=60000):
    """保存去冗余后的数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存序列数据
    logger.info("保存过滤后的序列数据...")
    save_results_chunked(filtered_sequences, output_dir, base_name="filtered_proteins", chunk_size=chunk_size)

    # 保存图谱数据
    if filtered_graphs:
        logger.info("保存过滤后的图谱数据...")
        save_results_chunked(filtered_graphs, output_dir, base_name="filtered_graphs", chunk_size=chunk_size)

    # 保存统计信息
    save_statistics(output_dir, len(filtered_sequences), len(filtered_graphs))


def save_results_chunked(data, output_dir, base_name="filtered_data", chunk_size=60000):
    """分块保存数据到JSON文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 将数据分块
    data_ids = list(data.keys())
    chunks = [data_ids[i:i + chunk_size] for i in range(0, len(data_ids), chunk_size)]

    output_files = []
    for i, chunk_ids in enumerate(tqdm(chunks, desc=f"保存{base_name}")):
        chunk_data = {id: data[id] for id in chunk_ids}
        output_file = os.path.join(output_dir, f"{base_name}_chunk_{i + 1}.pt")

        try:
            torch.save(chunk_data, output_file)
            output_files.append(output_file)
        except Exception as e:
            logger.error(f"保存数据块 {i + 1} 时出错: {str(e)}")

    # 保存元数据
    metadata = {
        "base_name": base_name,
        "total_items": len(data),
        "chunk_size": chunk_size,
        "chunks": len(chunks),
        "files": output_files,
        "timestamp": time.time()
    }

    metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"保存了 {len(data)} 个{base_name}到 {len(chunks)} 个数据块")
    return output_files, metadata


def save_statistics(output_dir, filtered_sequences_count, filtered_graphs_count):
    """保存过滤统计信息"""
    summary_file = os.path.join(output_dir, "filtering_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"过滤统计信息 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
        f.write(f"过滤后序列数量: {filtered_sequences_count}\n")
        f.write(f"过滤后图谱数量: {filtered_graphs_count}\n")
        f.write("-" * 50 + "\n")

    logger.info(f"统计信息已保存至: {summary_file}")


def load_sequence_results(input_dir):
    """加载序列处理的结果"""
    logger.info(f"从目录加载序列处理结果: {input_dir}")

    # 查找序列检查点文件
    seq_checkpoint_dir = os.path.join(input_dir, "seq_checkpoints")
    seq_checkpoint_file = os.path.join(seq_checkpoint_dir, "seq_clustering_results.pkl")

    if not os.path.exists(seq_checkpoint_file):
        logger.error(f"序列处理结果文件不存在: {seq_checkpoint_file}")
        return None

    try:
        # 加载序列聚类结果
        with open(seq_checkpoint_file, 'rb') as f:
            seq_results = pickle.load(f)

        logger.info(f"成功加载序列处理结果，包含 {len(seq_results)} 个序列ID")
        return seq_results
    except Exception as e:
        logger.error(f"加载序列处理结果出错: {str(e)}")
        return None


def perform_graph_similarity_filtering(filtered_seq_ids, graph_data, threshold=0.85, batch_size=10000, use_gpu=True):
    """
    基于图谱相似度进行过滤

    参数:
        filtered_seq_ids: 经过序列过滤的ID列表
        graph_data: 图谱数据字典
        threshold: 相似度阈值
        batch_size: 批处理大小
        use_gpu: 是否使用GPU

    返回:
        过滤后的图谱ID列表
    """
    logger.info(f"开始图谱相似度过滤，阈值：{threshold}")

    # 检查数据
    if not filtered_seq_ids or not graph_data:
        logger.warning("没有可用的图谱数据或序列ID进行过滤")
        return list(graph_data.keys()) if graph_data else []

    # 提取图谱特征
    graph_ids = []
    features = []

    logger.info("提取图谱特征...")
    for graph_id, graph in tqdm(graph_data.items(), desc="提取特征"):
        if graph_id in filtered_seq_ids:
            try:
                # 提取图谱特征向量 - 假设图谱对象有适当的方法
                feature = extract_graph_feature(graph)
                if feature is not None:
                    graph_ids.append(graph_id)
                    features.append(feature)
            except Exception as e:
                logger.debug(f"提取特征时出错 (ID: {graph_id}): {str(e)}")

    if not features:
        logger.warning("未能提取到任何有效特征")
        return graph_ids

    # 转换为numpy数组
    features_array = np.stack(features)
    logger.info(f"提取了 {len(features)} 个特征向量，形状: {features_array.shape}")

    # 初始化FAISS索引
    d = features_array.shape[1]  # 特征维度
    index = faiss.IndexFlatL2(d)

    # 如果可用并且请求使用GPU
    if use_gpu and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("使用GPU加速FAISS")
        except Exception as e:
            logger.warning(f"GPU加速失败，回退到CPU: {str(e)}")

    # 添加特征到索引
    index.add(features_array)

    # 执行相似度搜索
    logger.info("执行相似度搜索...")

    # 保留的图谱ID
    retained_graph_ids = set()
    processed_ids = set()

    # 分批处理以节省内存
    for i in range(0, len(features_array), batch_size):
        batch_features = features_array[i:i + batch_size]
        batch_ids = graph_ids[i:i + batch_size]

        # 查询每个图谱与其他图谱的相似度
        k = min(100, len(features_array))  # 搜索最近的k个邻居
        D, I = index.search(batch_features, k)

        for j, (distances, indices) in enumerate(zip(D, I)):
            graph_id = batch_ids[j]

            # 如果已处理，跳过
            if graph_id in processed_ids:
                continue

            # 将当前图谱标记为已处理
            processed_ids.add(graph_id)

            # 保留当前图谱
            retained_graph_ids.add(graph_id)

            # 标记所有相似的图谱为已处理
            for idx, dist in zip(indices[1:], distances[1:]):  # 从1开始跳过自身
                if idx < len(graph_ids) and dist < threshold:
                    processed_ids.add(graph_ids[idx])

    logger.info(f"图谱过滤完成，保留 {len(retained_graph_ids)}/{len(graph_ids)} 个图谱")
    return list(retained_graph_ids)


def extract_graph_feature(graph):
    """从图谱中提取特征向量"""
    try:
        # 这里应根据实际图谱结构实现特征提取
        # 例如，可以使用图谱的节点嵌入、边特征等

        # 假设图谱对象有node_features属性
        if hasattr(graph, 'node_features') and graph.node_features is not None:
            # 简单平均所有节点特征作为图谱特征
            return np.mean(graph.node_features.detach().cpu().numpy(), axis=0)

        # 如果没有现成的特征，可以返回一个随机特征
        # 这里仅作示例，实际应用中应根据图谱结构提取有意义的特征
        return np.random.random(128).astype(np.float32)

    except Exception as e:
        logger.debug(f"提取图谱特征时出错: {str(e)}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图谱处理与去冗余后处理工具")

    # 输入输出参数
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入目录路径，包含序列聚类结果")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="输出目录路径")

    # 过滤参数
    parser.add_argument("--graph_similarity", type=float, default=0.85,
                        help="图谱相似度阈值 (默认: 0.85)")
    parser.add_argument("--chunk_size", type=int, default=60000,
                        help="分块大小 (默认: 60000)")

    # 性能参数
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="批处理大小 (默认: 10000)")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="并行工作进程数 (默认: 64)")
    parser.add_argument("--memory_limit", type=int, default=900,
                        help="内存使用上限(GB) (默认: 900)")

    # GPU相关参数
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="使用GPU加速FAISS (默认: 是)")
    parser.add_argument("--gpu_device", "-g", type=str, default=None,
                        help="指定要使用的GPU设备ID，多个ID用逗号分隔，例如 '0,1' (默认: 使用所有可用GPU)")

    # 缓存参数
    parser.add_argument("--use_cache", action="store_true", default=True,
                        help="使用缓存加速图谱加载 (默认: 是)")
    parser.add_argument("--cache_dir", type=str, default="./pdb_kg_data/graph_cache/",
                        help="图谱缓存目录路径，默认为input目录下的graph_cache")

    args = parser.parse_args()

    # 如果未指定缓存目录，使用默认位置
    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.input, "graph_cache")

    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    global logger  # 使用全局logger变量
    logger, log_file_path = setup_logging(args.output_dir)
    logger.info(f"日志将写入文件: {log_file_path}")

    # 记录系统资源状态
    log_system_resources()

    # 打印运行配置
    logger.info("运行配置:")
    logger.info(f"- 输入目录: {args.input}")
    logger.info(f"- 输出目录: {args.output_dir}")
    logger.info(f"- 图谱相似度阈值: {args.graph_similarity}")
    logger.info(f"- 分块大小: {args.chunk_size}")
    logger.info(f"- 批处理大小: {args.batch_size}")
    logger.info(f"- 并行工作进程数: {args.num_workers}")
    logger.info(f"- 内存使用上限: {args.memory_limit}GB")
    logger.info(f"- 使用GPU加速: {'是' if args.use_gpu else '否'}")
    logger.info(f"- 指定GPU设备: {args.gpu_device if args.gpu_device else '全部'}")
    logger.info(f"- 使用图谱缓存: {'是' if args.use_cache else '否'}")
    logger.info(f"- 缓存目录: {args.cache_dir}")

    try:
        # 1. 加载序列处理结果
        seq_results = load_sequence_results(args.input)
        if not seq_results:
            logger.error("无法加载序列处理结果，退出")
            return

        filtered_seq_ids = list(seq_results.keys())
        logger.info(f"加载了 {len(filtered_seq_ids)} 个经过序列过滤的ID")

        # 2. 加载图谱数据
        logger.info("开始加载图谱数据...")
        graph_data = accelerated_graph_loader(
            args.input,
            filtered_seq_ids,
            num_workers=args.num_workers,
            memory_limit_gb=args.memory_limit,
            use_cache=args.use_cache,
            cache_dir=args.cache_dir
        )

        logger.info(f"成功加载了 {len(graph_data)} 个图谱")

        # 3. 图谱相似度过滤
        retained_graph_ids = perform_graph_similarity_filtering(
            filtered_seq_ids,
            graph_data,
            threshold=args.graph_similarity,
            batch_size=args.batch_size,
            use_gpu=args.use_gpu
        )

        # 4. 构建最终结果
        filtered_graphs = {id: graph_data[id] for id in retained_graph_ids if id in graph_data}

        # 5. 保存结果
        save_filtered_data(
            seq_results,
            filtered_graphs,
            args.output_dir,
            chunk_size=args.chunk_size
        )

        logger.info("处理完成!")
        logger.info(f"最终保留 {len(seq_results)} 个序列和 {len(filtered_graphs)} 个图谱")

    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")
        logger.error(traceback.format_exc())

    # 再次记录系统资源状态
    log_system_resources()


if __name__ == "__main__":
    # 捕获全局异常，确保程序不会崩溃而没有日志
    try:
        main()
    except Exception as e:
        if 'logger' in globals():
            logger.exception(f"程序执行时出现错误: {str(e)}")
        # 确保错误也输出到控制台
        print(f"错误: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)