import argparse
import concurrent.futures
import gzip
import json
import multiprocessing
import os
import random
import shutil
import logging
import faiss
from transformers import AutoTokenizer, AutoModel
import hdbscan

from torch_geometric.data import Data, Dataset
import networkx as nx
from Bio import PDB
from Bio.PDB import DSSP
from scipy.spatial import KDTree
import time
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 氨基酸物理化学性质常量
AA_PROPERTIES = {
    'A': {'hydropathy': 1.8, 'charge': 0, 'polar': False, 'mw': 89.09},
    'C': {'hydropathy': 2.5, 'charge': 0, 'polar': False, 'mw': 121.15},
    'D': {'hydropathy': -3.5, 'charge': -1, 'polar': True, 'mw': 133.10},
    'E': {'hydropathy': -3.5, 'charge': -1, 'polar': True, 'mw': 147.13},
    'F': {'hydropathy': 2.8, 'charge': 0, 'polar': False, 'mw': 165.19},
    'G': {'hydropathy': -0.4, 'charge': 0, 'polar': False, 'mw': 75.07},
    'H': {'hydropathy': -3.2, 'charge': 0.1, 'polar': True, 'mw': 155.16},
    'I': {'hydropathy': 4.5, 'charge': 0, 'polar': False, 'mw': 131.17},
    'K': {'hydropathy': -3.9, 'charge': 1, 'polar': True, 'mw': 146.19},
    'L': {'hydropathy': 3.8, 'charge': 0, 'polar': False, 'mw': 131.17},
    'M': {'hydropathy': 1.9, 'charge': 0, 'polar': False, 'mw': 149.21},
    'N': {'hydropathy': -3.5, 'charge': 0, 'polar': True, 'mw': 132.12},
    'P': {'hydropathy': -1.6, 'charge': 0, 'polar': False, 'mw': 115.13},
    'Q': {'hydropathy': -3.5, 'charge': 0, 'polar': True, 'mw': 146.15},
    'R': {'hydropathy': -4.5, 'charge': 1, 'polar': True, 'mw': 174.20},
    'S': {'hydropathy': -0.8, 'charge': 0, 'polar': True, 'mw': 105.09},
    'T': {'hydropathy': -0.7, 'charge': 0, 'polar': True, 'mw': 119.12},
    'V': {'hydropathy': 4.2, 'charge': 0, 'polar': False, 'mw': 117.15},
    'W': {'hydropathy': -0.9, 'charge': 0, 'polar': True, 'mw': 204.23},
    'Y': {'hydropathy': -1.3, 'charge': 0, 'polar': True, 'mw': 181.19}
}


def three_to_one(residue_name):
    """将三字母氨基酸代码转换为单字母代码"""
    try:
        from Bio.SeqUtils import seq1
        return seq1(residue_name)
    except ImportError:
        d = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return d.get(residue_name.upper(), 'X')


def create_protein_fragments(chain, residues, dssp=None, max_length=50, step_size=25, respect_ss=False):
    """将长蛋白质链分割为重叠片段，使用步长控制重叠量"""
    fragments = []

    # 过滤出标准氨基酸
    standard_residues = []
    for res in residues:
        try:
            aa = three_to_one(res.get_resname())
            if aa != "X" and 'CA' in res:  # 确保是标准氨基酸且有CA原子
                standard_residues.append(res)
        except:
            logger.debug(f"跳过非标准残基: {res.get_resname() if hasattr(res, 'get_resname') else 'Unknown'}")
            continue

    if not standard_residues:
        logger.debug(f"链 {chain.get_id()} 没有标准残基")
        return []

    if len(standard_residues) <= max_length:
        # 如果蛋白质足够短，直接返回整个链
        return [(standard_residues, "full")]

    # 根据是否尊重二级结构选择分段方法
    if respect_ss and dssp:
        try:
            # 寻找二级结构边界
            ss_boundaries = []
            prev_ss = None

            for i, res in enumerate(standard_residues):
                res_key = (chain.get_id(), res.get_id())
                if res_key in dssp:
                    curr_ss = dssp[res_key][2]  # 二级结构类型
                    if prev_ss != curr_ss:
                        ss_boundaries.append(i)
                        prev_ss = curr_ss

            # 确保添加最后一个边界
            if not ss_boundaries or ss_boundaries[-1] != len(standard_residues):
                ss_boundaries.append(len(standard_residues))

            # 基于二级结构边界创建片段
            for i in range(len(ss_boundaries) - 1):
                start = ss_boundaries[i]
                for j in range(i + 1, min(i + 3, len(ss_boundaries))):
                    end = ss_boundaries[j]
                    if 10 <= end - start <= max_length:
                        fragment = standard_residues[start:end]
                        frag_id = f"{standard_residues[start].get_id()[1]}-{standard_residues[end - 1].get_id()[1]}"
                        fragments.append((fragment, frag_id))

            if fragments:
                return fragments

        except Exception as e:
            logger.warning(f"基于二级结构的分段失败: {str(e)}，回退到常规滑动窗口")
            respect_ss = False

    # 如果二级结构划分失败，使用改进的方法
    if not fragments:
        # 方法1：使用自适应随机窗口
        if random.choice([True, False]):  # 50%概率使用随机窗口
            return create_protein_fragments_adaptive(chain, standard_residues, max_length)
        # 方法2：使用基于局部特征的边界
        else:
            boundaries = find_natural_boundaries(standard_residues)
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                # 合并过小的片段
                while end - start < 10 and i < len(boundaries) - 2:
                    i += 1
                    end = boundaries[i + 1]

                # 拆分过大的片段
                if end - start > max_length:
                    for j in range(start, end, step_size):
                        sub_end = min(j + max_length, end)
                        fragment = standard_residues[j:sub_end]
                        frag_id = f"{fragment[0].get_id()[1]}-{fragment[-1].get_id()[1]}"
                        fragments.append((fragment, frag_id))
                else:
                    fragment = standard_residues[start:end]
                    frag_id = f"{fragment[0].get_id()[1]}-{fragment[-1].get_id()[1]}"
                    fragments.append((fragment, frag_id))

    # 常规滑动窗口
    if not fragments:
        for start in range(0, len(standard_residues) - max_length + 1, step_size):
            fragment = standard_residues[start:start + max_length]
            start_res_num = fragment[0].get_id()[1]
            end_res_num = fragment[-1].get_id()[1]
            frag_id = f"{start_res_num}-{end_res_num}"
            fragments.append((fragment, frag_id))

        # 确保添加最后一个片段（如果不完整）
        last_start = len(standard_residues) - max_length
        if last_start % step_size != 0 and last_start > 0:
            last_fragment = standard_residues[-max_length:]
            start_res_num = last_fragment[0].get_id()[1]
            end_res_num = last_fragment[-1].get_id()[1]
            last_frag_id = f"{start_res_num}-{end_res_num}"
            fragments.append((last_fragment, last_frag_id))

    return fragments


def setup_logging(output_dir):
    """设置日志系统：控制台显示高级摘要，文件保存完整处理细节"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"extraction_{time.strftime('%Y%m%d_%H%M%S')}.log")

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

    # 过滤器 - 排除处理文件的详细信息
    class ProcessingFilter(logging.Filter):
        def filter(self, record):
            # 排除特定处理消息
            return not any(msg in record.getMessage() for msg in
                           ["解析文件", "处理文件", "添加节点", "处理残基", "跳过"])

    console.addFilter(ProcessingFilter())
    root_logger.addHandler(console)

    # 添加文件处理器 - 记录所有级别（包括DEBUG）
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    return root_logger, log_file

def calculate_sequence_similarity(seq1, seq2):
    """使用 MSA 库计算序列相似度，更高效且无废弃警告"""

    # 跳过空序列
    if not seq1 or not seq2:
        return 0.0

    # 如果序列长度差异显著，快速计算相似度
    if len(seq1) < 0.7 * len(seq2) or len(seq2) < 0.7 * len(seq1):
        return min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))

    try:
        # 使用 Needleman-Wunsch 算法 (全局比对)
        from msa import Aligner, Protein

        # 创建蛋白质序列对象
        p1 = Protein.from_string(seq1)
        p2 = Protein.from_string(seq2)

        # 设置比对器并执行全局比对
        aligner = Aligner(mode='global', gap_open=-10, gap_extend=-0.5)
        alignment = aligner.align(p1, p2)

        # 计算相同位点数
        matches = 0
        align_len = 0

        for a, b in zip(alignment.a, alignment.b):
            if a != '-' and b != '-':  # 不比较空位
                align_len += 1
                if a == b:
                    matches += 1

        return matches / max(len(seq1), len(seq2))

    except ImportError:
        # 如果没有安装 msa 库，使用简单的比对算法作为后备
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))

        # 计算相同字符数
        matches = sum(c1 == c2 for c1, c2 in zip(seq1[:min_len], seq2[:min_len]))
        return matches / max_len

def filter_redundant_fragments(extracted_chains, knowledge_graphs, similarity_threshold=0.9):
    """Filter highly similar fragments, keeping longer or earlier fragments"""
    logger.debug("Starting redundancy removal...")

    # Collect fragment data
    fragments_data = []
    for frag_id, data in extracted_chains.items():
        fragments_data.append({
            'id': frag_id,
            'sequence': data['sequence'],
            'protein_id': data['protein_id'],
            'chain_id': data['chain_id'],
            'length': len(data['sequence'])
        })

    if not fragments_data:
        logger.warning("No fragment data to filter")
        return {}, {}

    # Group by protein and chain
    grouped_fragments = {}
    for frag in fragments_data:
        key = (frag['protein_id'], frag['chain_id'])
        if key not in grouped_fragments:
            grouped_fragments[key] = []
        grouped_fragments[key].append(frag)

    # Process each group to remove redundancy
    to_keep = set()
    to_remove = set()

    # 使用tqdm显示进度
    progress_bar = tqdm(grouped_fragments.items(), desc="冗余过滤中", total=len(grouped_fragments))

    for protein_chain, frags in progress_bar:
        progress_bar.set_postfix({"蛋白质": protein_chain[0], "链": protein_chain[1], "片段": len(frags)})
        logger.debug(f"Processing {len(frags)} fragments from {protein_chain[0]} chain {protein_chain[1]}")

        # Sort by length (descending), then by fragment ID (for consistent results)
        frags.sort(key=lambda x: (-x['length'], x['id']))

        # Greedy selection
        for i, frag in enumerate(frags):
            # Skip if already decided
            if frag['id'] in to_remove:
                continue

            # Keep this fragment
            to_keep.add(frag['id'])

            # 标记相似片段进行移除
            for j in range(i + 1, len(frags)):
                other_frag = frags[j]
                # 跳过已决定的片段
                if other_frag['id'] in to_remove:
                    continue

                # 修复这里的比较
                if args.gpu:
                    # 计算单对序列相似度，并从矩阵中提取标量值
                    similarity_matrix = calculate_sequence_similarity_gpu(
                        [frag['sequence']], [other_frag['sequence']])
                    # 从矩阵中提取单个值
                    similarity = similarity_matrix[0][0]
                else:
                    # 使用CPU计算相似度（这部分返回标量，没问题）
                    similarity = calculate_sequence_similarity(frag['sequence'], other_frag['sequence'])

                # 现在similarity是标量，可以安全地比较
                if similarity >= similarity_threshold:
                    to_remove.add(other_frag['id'])

    # Filter collections
    filtered_chains = {k: v for k, v in extracted_chains.items() if k in to_keep}
    filtered_graphs = {k: v for k, v in knowledge_graphs.items() if k in to_keep}

    # Log statistics
    removed = len(extracted_chains) - len(filtered_chains)
    logger.info(
        f"Redundancy removal: {removed}/{len(extracted_chains)} fragments filtered ({removed / len(extracted_chains) * 100:.1f}%)")

    return filtered_chains, filtered_graphs

def create_protein_fragments_adaptive(chain, residues, max_length=50, min_length=20):
    """使用自适应随机窗口创建片段"""
    fragments = []
    standard_residues = [res for res in residues if three_to_one(res.get_resname()) != "X" and 'CA' in res]

    if len(standard_residues) <= max_length:
        # 使用链ID作为片段标识符的一部分
        chain_id = chain.get_id()
        return [(standard_residues, f"{chain_id}_full")]

    # 使用随机长度和步长
    current_pos = 0
    while current_pos < len(standard_residues):
        # 随机选择片段长度 (20-50)
        frag_length = random.randint(min_length, max_length)

        if current_pos + frag_length > len(standard_residues):
            # 处理最后一个片段
            frag_length = len(standard_residues) - current_pos

        fragment = standard_residues[current_pos:current_pos + frag_length]
        frag_id = f"{fragment[0].get_id()[1]}-{fragment[-1].get_id()[1]}"
        fragments.append((fragment, frag_id))

        # 修复随机范围错误
        if frag_length <= 20:
            # 小片段使用固定步长
            step = max(1, frag_length // 2)
        else:
            # 确保随机范围有效
            min_step = 5  # 最小步长5
            max_step = min(frag_length // 2, 25)  # 最大步长25或片段长度的一半

            # 确保min_step <= max_step
            if min_step > max_step:
                step = max_step
            else:
                step = random.randint(min_step, max_step)

        current_pos += step

    return fragments


def calculate_sequence_similarity_gpu(sequences, sequences2=None, batch_size=1000):
    """使用GPU加速批量计算序列相似度矩阵"""

    if not torch.cuda.is_available():
        logger.warning("未找到可用GPU，将使用CPU进行处理")
        print("\n>>> 未找到可用GPU，将使用CPU进行处理 <<<\n")
        return np.array([[calculate_sequence_similarity(s1, s2) for s2 in (sequences2 or sequences)]
                         for s1 in sequences])

    # 获取GPU设备信息并显示
    device = torch.device('cuda')

    # 自比较模式
    self_compare = sequences2 is None
    if self_compare:
        sequences2 = sequences

    n_seqs1, n_seqs2 = len(sequences), len(sequences2)
    similarity_matrix = torch.zeros((n_seqs1, n_seqs2), device=device)

    # 氨基酸字母表和编码映射
    aa_chars = "ACDEFGHIKLMNPQRSTVWY-X"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_chars)}

    # 分批计算以节省内存
    for i in range(0, n_seqs1, batch_size):
        i_end = min(i + batch_size, n_seqs1)

        for j in range(0, n_seqs2, batch_size):
            j_end = min(j + batch_size, n_seqs2)

            # 计算当前批次的相似度
            for idx1 in range(i, i_end):
                seq1 = sequences[idx1]

                for idx2 in range(j, j_end):
                    # 对角线以上元素已计算(自比较模式)
                    if self_compare and idx1 > idx2:
                        similarity_matrix[idx1, idx2] = similarity_matrix[idx2, idx1]
                        continue

                    seq2 = sequences2[idx2]

                    # 长度差异过大时的快速计算
                    len1, len2 = len(seq1), len(seq2)
                    max_len = max(len1, len2)
                    min_len = min(len1, len2)

                    if min_len < 0.7 * max_len:
                        similarity_matrix[idx1, idx2] = min_len / max_len
                        continue

                    # 使用GPU计算序列相似度
                    tensor1 = torch.tensor([aa_to_idx.get(aa, aa_to_idx['X']) for aa in seq1[:min_len]],
                                           device=device)
                    tensor2 = torch.tensor([aa_to_idx.get(aa, aa_to_idx['X']) for aa in seq2[:min_len]],
                                           device=device)

                    # 计算匹配数量
                    matches = (tensor1 == tensor2).sum().item()
                    similarity = matches / max_len

                    similarity_matrix[idx1, idx2] = similarity

    return similarity_matrix.cpu().numpy()


def filter_redundancy_mode():
    """独立的去冗余模式，从保存的序列数据中去除冗余"""
    logger.info("启动独立去冗余模式")

    # GPU usage message - add this check
    if args.gpu:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"使用GPU: {torch.cuda.current_device()} - {gpu_name}")
        else:
            logger.info("GPU不可用，将使用CPU进行处理")
            print("\nGPU不可用，将使用CPU进行处理\n")
    else:
        logger.info("未启用GPU加速，使用CPU进行处理")
        print("\n未启用GPU加速，使用CPU进行处理\n")

    # 加载序列和图谱数据
    proteins_data = {}
    graphs_data = {}

    # 检查输入文件或目录
    input_path = args.input
    if os.path.isdir(input_path):
        # 查找序列数据文件
        seq_files = [f for f in os.listdir(input_path) if f.startswith("pdb_data_") and f.endswith(".json")]
        graph_files = [f for f in os.listdir(input_path) if f.startswith("kg_data_") and f.endswith(".json")]

        if not seq_files:
            logger.error(f"未在 {input_path} 中找到序列数据文件")
            return

        # 加载序列数据
        for seq_file in seq_files:
            file_path = os.path.join(input_path, seq_file)
            logger.info(f"加载序列数据: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    # 确保数据格式正确
                    for key, value in file_data.items():
                        if isinstance(value, dict) and 'sequence' in value:
                            proteins_data[key] = value
                        else:
                            logger.warning(f"跳过无效的蛋白质数据: {key}")
            except Exception as e:
                logger.error(f"加载序列数据失败: {str(e)}")

        # 加载图谱数据
        for graph_file in graph_files:
            file_path = os.path.join(input_path, graph_file)
            logger.info(f"加载图谱数据: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    graphs_data.update(json.load(f))
            except Exception as e:
                logger.error(f"加载图谱数据失败: {str(e)}")

    elif os.path.isfile(input_path):
        # 直接加载单个文件
        logger.info(f"加载数据文件: {input_path}")
        try:
            with open(input_path, 'r') as f:
                file_data = json.load(f)
                if "pdb_data" in input_path:
                    # 确保数据格式正确
                    for key, value in file_data.items():
                        if isinstance(value, dict) and 'sequence' in value:
                            proteins_data[key] = value
                        else:
                            logger.warning(f"跳过无效的蛋白质数据: {key}")
                elif "kg_data" in input_path:
                    graphs_data.update(file_data)
        except Exception as e:
            logger.error(f"加载数据文件失败: {str(e)}")
            return

    else:
        logger.error(f"无效的输入路径: {input_path}")
        return

    # 检查是否成功加载数据
    if not proteins_data:
        logger.error("未找到有效的序列数据")
        return

    logger.info(f"已加载 {len(proteins_data)} 个蛋白质序列和 {len(graphs_data)} 个知识图谱")

    # 执行去冗余处理
    logger.info(f"开始使用相似度阈值 {args.redundancy_threshold} 进行冗余过滤...")
    filtered_proteins, filtered_graphs = remove_redundancy_pipeline(
        proteins_data,
        graphs_data,
        similarity_threshold=args.redundancy_threshold,
        min_cluster_size=10  # 可配置参数
    )

    # 保存结果
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 保存序列数据
    output_seq_file = os.path.join(output_dir, "filtered_sequences.json")
    with open(output_seq_file, 'w') as f:
        json.dump(filtered_proteins, f)
    logger.info(f"已保存过滤后的序列数据至: {output_seq_file}")

    # 保存图谱数据
    if filtered_graphs:
        output_graph_file = os.path.join(output_dir, "filtered_graphs.json")
        with open(output_graph_file, 'w') as f:
            json.dump(filtered_graphs, f)
        logger.info(f"已保存过滤后的图谱数据至: {output_graph_file}")

    # 如果需要也保存PyG格式
    if args.pyg_format and filtered_graphs:
        save_knowledge_graphs(filtered_graphs, output_dir,
                              base_name="filtered_kg", format_type="pyg")

    return filtered_proteins, filtered_graphs

def run_dssp(structure, file_path):
    """运行DSSP分析并获取二级结构信息"""
    try:
        model = structure[0]
        dssp = DSSP(model, file_path, dssp='mkdssp')
        return dssp
    except Exception as e:
        logger.warning(f"DSSP分析失败: {str(e)}")
        return None


def find_natural_boundaries(residues, window=5):
    """基于局部特征变化识别自然分割点"""
    boundaries = [0]  # 起始点

    # 计算滑动窗口的物理化学特性变化
    scores = []
    for i in range(len(residues) - window):
        window1 = residues[i:i + window]
        window2 = residues[i + window:i + 2 * window] if i + 2 * window <= len(residues) else residues[i + window:]

        # 计算窗口内氨基酸特性
        hydro1 = sum(AA_PROPERTIES[three_to_one(r.get_resname())]['hydropathy'] for r in window1) / len(window1)
        hydro2 = sum(AA_PROPERTIES[three_to_one(r.get_resname())]['hydropathy'] for r in window2) / len(window2)

        # 特性变化分数
        scores.append(abs(hydro1 - hydro2))

    # 找出显著变化点 (局部最大值)
    for i in range(1, len(scores) - 1):
        if scores[i] > scores[i - 1] and scores[i] > scores[i + 1] and scores[i] > np.mean(scores) + 0.5 * np.std(
                scores):
            boundaries.append(i + window)

    # 添加结束点
    boundaries.append(len(residues))

    return boundaries

def build_residue_graph(chain, residues, dssp=None, k_neighbors=8, distance_threshold=8.0,
                        fragment_id=None, plddt_threshold=70):
    """为蛋白质链或片段构建残基知识图谱"""
    graph = nx.Graph()

    if not residues:
        return graph

    # 1. 添加节点
    ca_coords = []
    ca_atoms = []
    standard_aa_set = set('ACDEFGHIKLMNPQRSTVWY')

    for i, res in enumerate(residues):
        try:
            res_id = res.get_id()
            res_num = res_id[1]
            res_name = res.get_resname()
            aa_code = three_to_one(res_name)

            # 跳过非标准氨基酸
            if aa_code not in standard_aa_set:
                continue

            # 尝试获取CA原子坐标和pLDDT值
            ca_atom = res['CA']
            ca_coord = ca_atom.get_coord()

            # 从B-factor获取pLDDT值 (AlphaFold模型特有)
            plddt = ca_atom.get_bfactor()

            # 跳过低质量预测的残基
            if plddt < plddt_threshold:
                continue

            # 添加节点属性
            node_attrs = {
                'residue_name': res_name,
                'residue_code': aa_code,
                'residue_num': res_num,
                'plddt': plddt,
                'position': ca_coord.tolist(),
                'chain_id': chain.get_id()
            }

            # 添加氨基酸物理化学属性
            if aa_code in AA_PROPERTIES:
                props = AA_PROPERTIES[aa_code]
                node_attrs.update({
                    'hydropathy': props['hydropathy'],
                    'charge': props['charge'],
                    'polar': props['polar'],
                    'molecular_weight': props['mw']
                })

            # 获取二级结构(如果可用)
            if dssp:
                res_key = (chain.get_id(), res_id)
                if res_key in dssp:
                    ss = dssp[res_key][2]
                    acc = dssp[res_key][3]
                    phi = dssp[res_key][4]
                    psi = dssp[res_key][5]
                    node_attrs.update({
                        'secondary_structure': ss,
                        'accessible_area': acc,
                        'phi': phi,
                        'psi': psi
                    })
            else:
                node_attrs['secondary_structure'] = 'X'  # 未知二级结构

            # 添加片段ID (如果有)
            if fragment_id:
                node_attrs['fragment_id'] = fragment_id

            # 添加节点
            node_id = f"{chain.get_id()}_{res_num}"
            graph.add_node(node_id, **node_attrs)

            # 存储坐标用于后续空间邻接计算
            ca_coords.append(ca_coord)
            ca_atoms.append(node_id)

        except KeyError:  # CA原子缺失
            continue
        except Exception as e:
            logger.debug(f"添加节点时出错: {str(e)}")

    # 2. 添加序列邻接边
    node_ids = list(graph.nodes())
    for i in range(len(node_ids) - 1):
        node1 = node_ids[i]
        node2 = node_ids[i + 1]
        # 确保是序列上相邻的残基
        res_num1 = int(graph.nodes[node1]['residue_num'])
        res_num2 = int(graph.nodes[node2]['residue_num'])
        if res_num2 == res_num1 + 1:
            graph.add_edge(node1, node2, edge_type='peptide', weight=1.0, distance=1.0)

    # 3. 添加空间邻接边
    if len(ca_coords) > 1:
        try:
            # 构建KD树用于快速空间查询
            kd_tree = KDTree(ca_coords)

            # 对每个残基找到k个最近邻
            _, indices = kd_tree.query(ca_coords, k=min(k_neighbors + 1, len(ca_coords)))

            # 添加空间邻接边
            for i, neighbors in enumerate(indices):
                for j_idx in neighbors[1:]:  # 跳过自身(第一个结果)
                    j = int(j_idx)

                    # 计算残基间距离
                    dist = float(np.linalg.norm(np.array(ca_coords[i]) - np.array(ca_coords[j])))

                    # 只连接距离小于阈值的残基
                    if dist <= distance_threshold:
                        if not graph.has_edge(ca_atoms[i], ca_atoms[j]):
                            graph.add_edge(ca_atoms[i], ca_atoms[j],
                                           weight=dist, edge_type='spatial', distance=dist)
        except Exception as e:
            logger.warning(f"计算空间邻接时出错: {str(e)}")

    return graph


def remove_redundancy_pipeline(sequences, graphs=None, similarity_threshold=0.9, min_cluster_size=10,
                               gpu_id=0, embedding_batch_size=1000):
    """
    三步冗余去除流程:
    1. HDBSCAN密度聚类 (FAISS加速)
    2. 类内序列相似度过滤
    3. 最终验证过滤

    参数:
        sequences: 字典 {id: sequence_data} 或序列列表
        graphs: 可选的图谱字典 {id: graph_data}
        similarity_threshold: 相似度阈值
        min_cluster_size: HDBSCAN最小聚类大小
        gpu_id: GPU设备ID
        embedding_batch_size: 嵌入计算批次大小
    返回:
        非冗余序列和对应图谱
    """

    # 添加必要的依赖
    import time
    import torch
    import faiss
    import hdbscan
    import numpy as np
    from collections import defaultdict
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm

    start_time = time.time()

    # 标准化输入数据
    if isinstance(sequences, dict):
        seq_dict = sequences
        seq_ids = list(sequences.keys())
        seq_list = [sequences[i]['sequence'] if isinstance(sequences[i], dict) else sequences[i] for i in seq_ids]
    else:
        seq_list = sequences
        seq_ids = [f"seq_{i}" for i in range(len(sequences))]
        seq_dict = {seq_ids[i]: {'sequence': seq_list[i]} for i in range(len(seq_list))}

    logger.info(f"开始三步去冗余流程，处理{len(seq_list)}个序列...")

    # 步骤1: HDBSCAN密度聚类
    logger.info("步骤1: 执行HDBSCAN密度聚类...")

    # 检查GPU可用性
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device(f'cuda:{gpu_id}')
        gpu_name = torch.cuda.get_device_name(gpu_id)
        logger.info(f"使用GPU加速聚类: {gpu_name}")
    else:
        device = torch.device('cpu')
        logger.info("未检测到GPU，将使用CPU进行处理")

    # 初始化ESM-2模型
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)
        model.eval()
    except Exception as e:
        logger.error(f"ESM2模型加载失败: {str(e)}")
        logger.info("尝试使用备用模型...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
            model.eval()
        except Exception as e2:
            logger.error(f"备用模型也加载失败: {str(e2)}")
            raise RuntimeError("无法加载序列编码模型，请检查网络连接或安装transformers库")

    # 生成序列嵌入
    logger.info("生成序列嵌入向量...")
    all_embeddings = []

    for i in tqdm(range(0, len(seq_list), embedding_batch_size), desc="生成嵌入"):
        batch = seq_list[i:i + embedding_batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # 使用[CLS]令牌作为序列表示
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

    # 合并所有嵌入
    embeddings = np.vstack(all_embeddings)

    # 释放GPU内存
    if use_gpu:
        torch.cuda.empty_cache()

    # 使用HDBSCAN+FAISS进行聚类
    logger.info(f"使用HDBSCAN进行密度聚类 (min_cluster_size={min_cluster_size})...")

    if use_gpu and faiss.get_num_gpus() > 0:
        # 初始化FAISS GPU资源
        res = faiss.StandardGpuResources()

        # 配置FAISS
        config = faiss.GpuIndexFlatConfig()
        config.device = gpu_id

        # HDBSCAN配置
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric='euclidean',
            core_dist_n_jobs=1,
            algorithm='best',
            cluster_selection_method='eom',
            prediction_data=True
        )
    else:
        # CPU版本
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric='euclidean',
            core_dist_n_jobs=-1,
            algorithm='best',
            cluster_selection_method='eom',
            prediction_data=True
        )

    # 执行聚类
    cluster_labels = clusterer.fit_predict(embeddings)

    # 处理离群点(-1标签)
    outliers = np.where(cluster_labels == -1)[0]
    if len(outliers) > 0:
        logger.info(f"处理{len(outliers)}个离群点 ({len(outliers) / len(cluster_labels) * 100:.1f}%)...")

        # 将离群点分配到最近的聚类
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]

        if len(unique_clusters) > 0:
            # 计算聚类中心
            centers = np.zeros((len(unique_clusters), embeddings.shape[1]))
            for i, cluster_id in enumerate(unique_clusters):
                centers[i] = embeddings[cluster_labels == cluster_id].mean(axis=0)

            # 使用FAISS快速找到最近的聚类中心
            if use_gpu:
                index = faiss.GpuIndexFlatL2(res, embeddings.shape[1], config)
            else:
                index = faiss.IndexFlatL2(embeddings.shape[1])

            index.add(centers.astype(np.float32))
            _, nearest = index.search(embeddings[outliers].astype(np.float32), 1)

            # 重新分配离群点标签
            for i, idx in enumerate(outliers):
                cluster_labels[idx] = unique_clusters[nearest[i, 0]]

    # 将序列分组到各个聚类
    clusters = defaultdict(list)
    cluster_seq_ids = defaultdict(list)

    for i, cluster_id in enumerate(cluster_labels):
        clusters[cluster_id].append(seq_list[i])
        cluster_seq_ids[cluster_id].append(seq_ids[i])

    n_clusters = len(clusters)
    logger.info(f"HDBSCAN聚类完成，识别出{n_clusters}个聚类")

    # 步骤2: 类内序列相似度去冗余
    logger.info("步骤2: 执行类内序列相似度去冗余...")

    all_to_keep = set()

    for cluster_id, cluster_seqs in tqdm(clusters.items(), desc="类内去冗余"):
        ids_in_cluster = cluster_seq_ids[cluster_id]

        # 跳过小聚类
        if len(cluster_seqs) <= 1:
            all_to_keep.add(ids_in_cluster[0])
            continue

        # 计算聚类内序列相似度矩阵
        similarity_matrix = calculate_sequence_similarity_gpu(cluster_seqs)

        # 贪心选择算法
        selected = [True] * len(cluster_seqs)

        # 按序列长度降序排序
        len_indices = sorted(range(len(cluster_seqs)),
                             key=lambda i: (-len(cluster_seqs[i]), ids_in_cluster[i]))

        # 贪心移除冗余序列
        for i in range(len(len_indices)):
            idx_i = len_indices[i]

            if not selected[idx_i]:  # 已被移除
                continue

            # 检查与已选序列的相似度
            for j in range(i + 1, len(len_indices)):
                idx_j = len_indices[j]

                if not selected[idx_j]:  # 已被移除
                    continue

                # 如果相似度高于阈值，移除序列j
                if similarity_matrix[idx_i, idx_j] >= similarity_threshold:
                    selected[idx_j] = False

        # 收集该聚类中要保留的序列ID
        for i, keep in enumerate(selected):
            if keep:
                all_to_keep.add(ids_in_cluster[i])

    # 步骤3：构建结果
    filtered_sequences = {seq_id: seq_dict[seq_id] for seq_id in all_to_keep}

    # 如果提供了图谱数据，也进行过滤
    filtered_graphs = None
    if graphs:
        filtered_graphs = {seq_id: graphs[seq_id] for seq_id in all_to_keep if seq_id in graphs}
        logger.info(f"图谱过滤: 从{len(graphs)}个图谱中保留{len(filtered_graphs)}个")

    logger.info(f"去冗余流程完成: 从{len(seq_dict)}个序列中保留{len(filtered_sequences)}个 "
               f"({len(filtered_sequences)/len(seq_dict)*100:.1f}%)")
    logger.info(f"处理时间: {(time.time() - start_time)/60:.1f}分钟")

    if graphs:
        return filtered_sequences, filtered_graphs
    return filtered_sequences

def parse_pdb_file(pdb_file_path, max_length=50, step_size=25, build_kg=True,
                   k_neighbors=8, distance_threshold=8.0, plddt_threshold=70, respect_ss=True):
    """解析PDB/CIF文件并提取蛋白质信息"""
    logger.debug(f"解析文件: {pdb_file_path}")

    # 确定文件类型和处理方式
    is_gzipped = pdb_file_path.endswith('.gz')
    is_cif = '.cif' in pdb_file_path
    protein_id = os.path.basename(pdb_file_path).split('.')[0]
    temp_file = None
    extracted_chains = {}
    knowledge_graphs = {}

    # 添加片段统计跟踪
    fragment_stats = {
        "file_name": os.path.basename(pdb_file_path),
        "protein_id": protein_id,
        "chains": {}
    }

    try:
        # 处理文件格式
        if is_gzipped:
            temp_file = f"/tmp/{os.path.basename(pdb_file_path)[:-3]}"
            with gzip.open(pdb_file_path, 'rb') as f_in:
                with open(temp_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            file_to_parse = temp_file
        else:
            file_to_parse = pdb_file_path

        # 检查文件
        if not os.path.exists(file_to_parse) or os.path.getsize(file_to_parse) == 0:
            logger.error(f"文件不存在或为空: {pdb_file_path}")
            return {}, {}

        # 选择解析器
        parser = PDB.MMCIFParser() if is_cif else PDB.PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(protein_id, file_to_parse)
        except Exception as e:
            logger.error(f"解析结构失败: {str(e)}")
            return {}, {}

        # 尝试运行DSSP
        dssp_dict = None
        try:
            dssp_dict = run_dssp(structure, file_to_parse)
        except Exception as e:
            logger.debug(f"DSSP分析失败: {str(e)}")

        # 处理每个模型和链
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                # 初始化链统计
                fragment_stats["chains"][chain_id] = {
                    "valid_residues": 0,
                    "fragments": 0
                }
                # 收集有效残基
                valid_residues = []
                for res in chain:
                    if res.get_id()[0] == ' ':  # 跳过异常残基
                        try:
                            if 'CA' in res:  # 确保残基有CA原子
                                valid_residues.append(res)
                        except:
                            continue
                # 更新残基数量
                fragment_stats["chains"][chain_id]["valid_residues"] = len(valid_residues)

                # 如果有足够的残基，创建片段
                if len(valid_residues) >= 5:  # 至少需要5个残基
                    fragments = create_protein_fragments(
                        chain, valid_residues, dssp_dict,
                        max_length, step_size, respect_ss
                    )

                    # 更新片段数量
                    fragment_stats["chains"][chain_id]["fragments"] = len(fragments)

                    # 处理每个片段
                    for frag_residues, frag_id in fragments:
                        if not frag_residues:
                            continue

                        fragment_id = f"{protein_id}_{chain_id}_{frag_id}"

                        # 提取序列
                        sequence = ""
                        for res in frag_residues:
                            try:
                                aa = three_to_one(res.get_resname())
                                if aa != "X":
                                    sequence += aa
                            except:
                                continue

                        # 保存片段信息
                        extracted_chains[fragment_id] = {
                            "protein_id": protein_id,
                            "chain_id": chain_id,
                            "fragment_id": frag_id,
                            "sequence": sequence,
                            "length": len(sequence)
                        }

                        # 构建知识图谱
                        if build_kg:
                            kg = build_residue_graph(
                                chain, frag_residues, dssp_dict,
                                k_neighbors, distance_threshold,
                                fragment_id, plddt_threshold
                            )
                            # 只保存非空图
                            if kg.number_of_nodes() > 0:
                                knowledge_graphs[fragment_id] = nx.node_link_data(kg, edges="links")  # 保持现有行为

        return extracted_chains, knowledge_graphs, fragment_stats

    except Exception as e:
        logger.error(f"解析文件 {pdb_file_path} 时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}, {}, None
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def find_pdb_files(root_dir):
    """递归搜索所有PDB和CIF文件，并去重"""
    pdb_files = []
    processed_ids = set()  # 记录已处理的蛋白质ID

    logger.info(f"正在搜索目录: {root_dir}")

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.pdb', '.pdb.gz', '.cif', '.cif.gz')):
                # 提取蛋白质ID (去除扩展名)
                protein_id = file.split('.')[0]

                # 跳过已处理的ID
                if protein_id in processed_ids:
                    continue

                processed_ids.add(protein_id)
                pdb_files.append(os.path.join(root, file))

    logger.info(f"找到 {len(pdb_files)} 个不重复的PDB/CIF文件")
    return pdb_files


def save_knowledge_graphs(kg_data, output_dir, base_name="protein_kg", chunk_size=1000, format_type="pyg"):
    """将知识图谱保存为选择的格式，定期保存以防数据丢失"""
    if not kg_data:
        logger.warning("没有知识图谱数据可保存")
        return None

    logger.info(f"\n保存知识图谱数据 ({format_type} 格式)...")

    # 创建输出目录
    kg_dir = os.path.join(output_dir, f"knowledge_graphs_{format_type}")
    os.makedirs(kg_dir, exist_ok=True)

    # 获取所有蛋白质ID并分块
    all_protein_ids = list(kg_data.keys())
    num_chunks = (len(all_protein_ids) + chunk_size - 1) // chunk_size

    logger.info(f"将知识图谱拆分为{num_chunks}个块，每块最多{chunk_size}个蛋白质")

    # 创建索引字典
    index = {
        "total_proteins": len(all_protein_ids),
        "chunks_count": num_chunks,
        "chunk_size": chunk_size,
        "total_nodes": 0,
        "total_edges": 0,
        "error_count": 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": format_type
    }

    # 处理每个块
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_protein_ids))
        chunk_ids = all_protein_ids[start_idx:end_idx]

        logger.info(f"开始处理知识图谱块 {chunk_id + 1}/{num_chunks}，共{len(chunk_ids)}个蛋白质")

        # 为当前块创建输出文件
        output_file = os.path.join(kg_dir,
                                   f"{base_name}_chunk_{chunk_id + 1}.{'pt' if format_type == 'pyg' else 'json'}")

        if format_type == "pyg":
            # PyG格式保存
            try:
                graphs_data = {}

                # 分批处理该块内的图，每100000个报告一次进度
                for i, pid in enumerate(chunk_ids):
                    if i % 10000 == 0 or i == len(chunk_ids) - 1:
                        logger.info(f"  - 正在处理第{i + 1}/{len(chunk_ids)}个蛋白质图谱")

                    try:
                        # 获取NetworkX图 - 修复这里，确保是NetworkX图对象
                        nx_data = kg_data[pid]
                        # 如果是字典格式，转换为NetworkX图
                        if isinstance(nx_data, dict):
                            nx_graph = nx.node_link_graph(nx_data)
                        else:
                            nx_graph = nx_data

                        # 转换为PyG格式
                        x = []  # 节点特征
                        edge_index = [[], []]  # 边索引
                        edge_attr = []  # 边特性

                        # 创建节点映射
                        node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}

                        # 提取节点特征
                        for node in nx_graph.nodes():
                            node_attrs = nx_graph.nodes[node]
                            # 将节点属性转换为特征向量
                            features = [
                                AA_PROPERTIES.get(node_attrs.get('residue_code', 'X'), {}).get('hydropathy', 0),
                                AA_PROPERTIES.get(node_attrs.get('residue_code', 'X'), {}).get('charge', 0),
                                node_attrs.get('plddt', 50) / 100.0,  # 归一化pLDDT
                            ]
                            x.append(features)

                        # 提取边
                        for src, tgt, edge_data in nx_graph.edges(data=True):
                            edge_index[0].append(node_mapping[src])
                            edge_index[1].append(node_mapping[tgt])

                            # 边属性
                            edge_type = 1 if edge_data.get('edge_type') == 'peptide' else 0
                            distance = edge_data.get('distance', 0)
                            edge_attr.append([edge_type, distance])

                        # 创建PyG数据对象
                        if x and edge_index[0]:  # 确保有节点和边
                            data = Data(
                                x=torch.tensor(x, dtype=torch.float),
                                edge_index=torch.tensor(edge_index, dtype=torch.long),
                                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                                protein_id=pid
                            )
                            graphs_data[pid] = data

                            # 更新统计信息
                            index["total_nodes"] += len(x)
                            index["total_edges"] += len(edge_attr)
                    except Exception as e:
                        logger.warning(f"转换蛋白质 {pid} 图谱时出错: {str(e)}")
                        index["error_count"] += 1

                # 保存为PyTorch文件
                if graphs_data:
                    torch.save(graphs_data, output_file)
                    logger.info(f"已保存 {len(graphs_data)} 个PyG格式图谱到 {output_file}")
                else:
                    logger.warning(f"块 {chunk_id + 1} 中没有有效的PyG图谱数据")

            except Exception as e:
                logger.error(f"保存PyG格式知识图谱时出错 (块 {chunk_id + 1}): {str(e)}")
                import traceback
                logger.error(traceback.format_exc())


def save_results_chunked(all_proteins, output_dir, base_name="pdb_data", chunk_size=1000):
    """分块保存蛋白质序列结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 将蛋白质数据分块
    protein_ids = list(all_proteins.keys())
    chunks = [protein_ids[i:i + chunk_size] for i in range(0, len(protein_ids), chunk_size)]

    output_files = []
    for i, chunk_ids in enumerate(chunks):
        chunk_data = {pid: all_proteins[pid] for pid in chunk_ids}
        output_file = os.path.join(output_dir, f"{base_name}_chunk_{i + 1}.json")

        with open(output_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)

        output_files.append(output_file)
        logger.info(f"保存数据块 {i + 1}/{len(chunks)}: {output_file} ({len(chunk_ids)} 个蛋白质)")

    # 保存元数据
    metadata = {
        "total_proteins": len(all_proteins),
        "chunk_count": len(chunks),
        "chunk_files": output_files,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_files, metadata


def process_file_parallel(file_path, max_length=50, step_size=25, build_kg=True,
                          k_neighbors=8, distance_threshold=8.0, plddt_threshold=70, respect_ss=True):
    """Function to process a single file in parallel"""
    try:
        start_time = time.time()

        # File check
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return {"error": f"文件不存在或为空: {file_path}"}, {}, {}, None

        # Process file
        proteins, kg, fragment_stats = parse_pdb_file(
            file_path, max_length, step_size, build_kg,
            k_neighbors, distance_threshold, plddt_threshold, respect_ss
        )

        elapsed = time.time() - start_time
        file_name = os.path.basename(file_path)

        # Result statistics
        chain_count = len(set([info["chain_id"] for info in proteins.values()])) if proteins else 0
        fragment_count = len(proteins)

        result_info = {
            "file_path": file_path,
            "elapsed": elapsed,
            "protein_count": len(proteins),
            "kg_count": len(kg),
            "success": True
        }
        return result_info, proteins, kg, fragment_stats
    except Exception as e:
        return {"error": str(e), "file_path": file_path}, {}, {}, None


def process_file_batch_parallel(file_list, output_dir, max_length=50, step_size=25, build_kg=True, n_workers=None,
                                k_neighbors=8, distance_threshold=8.0, plddt_threshold=70.0, respect_ss=True):
    """并行处理多个文件，每1000个文件保存一次结果"""
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    logger.info(f"使用 {n_workers} 个CPU核心并行处理...")
    logger.info(f"处理参数: 最大长度={max_length}, 步长={step_size}, "
                f"k近邻={k_neighbors}, 距离阈值={distance_threshold}埃, "
                f"pLDDT阈值={plddt_threshold}, 遵循二级结构={respect_ss}")

    # 创建ALL子文件夹用于存储中间结果
    all_data_dir = os.path.join(output_dir, "ALL")
    os.makedirs(all_data_dir, exist_ok=True)

    # 将日志文件保存在输出目录中
    sequences_log_path = os.path.join(output_dir, "sequences.log")
    fragments_log_path = os.path.join(output_dir, "fragments_stats.log")
    processing_log_path = os.path.join(output_dir, "processing.log")

    all_proteins = {}
    all_knowledge_graphs = {}
    all_fragment_stats = []
    stats = defaultdict(int)
    chunk_size = args.chunk_size if hasattr(args, 'chunk_size') else 1000

    # 初始化统计数据键
    stats["processed_files"] = 0
    stats["extracted_chains"] = 0
    stats["knowledge_graphs"] = 0
    stats["failed_files"] = 0
    stats["total_fragments"] = 0

    # 初始化文件计数器和批次ID
    file_counter = 0
    batch_id = 1

    with open(sequences_log_path, 'w', buffering=1) as s_log:
        s_log.write("fragment_id,protein_id,chain_id,length,sequence\n")

    with open(fragments_log_path, 'w') as f_log:
        f_log.write("file_name,protein_id,chain_id,valid_residues,fragments\n")

    with open(processing_log_path, 'w') as p_log:
        p_log.write("timestamp,file_path,status,elapsed,proteins,knowledge_graphs,error\n")

    # 使用进程池进行并行处理
    with tqdm(total=len(file_list), desc="处理PDB/CIF文件") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    process_file_parallel, file_path, max_length, step_size, build_kg,
                    k_neighbors, distance_threshold, plddt_threshold, respect_ss
                ) for file_path in file_list
            ]

            for future in concurrent.futures.as_completed(futures):
                result, proteins, kg, fragment_stats = future.result()
                pbar.update(1)

                file_counter += 1

                if "error" in result:
                    stats["failed_files"] += 1
                    logger.error(f"处理文件失败: {result['file_path']} - {result['error']}")
                    # 记录失败日志
                    with open(processing_log_path, 'a') as p_log:
                        p_log.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')},{result['file_path']},FAILED,0,0,0,{result['error']}\n")
                    continue

                # 更新统计数据
                stats["processed_files"] += 1
                stats["extracted_chains"] += len(proteins)
                stats["knowledge_graphs"] += len(kg)
                stats["total_fragments"] += len(proteins)

                # 记录处理日志
                with open(processing_log_path, 'a') as p_log:
                    elapsed = result.get('elapsed', 0)
                    status = "SUCCESS" if "success" in result and result["success"] else "FAILED"
                    error = result.get('error', '')
                    p_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{result['file_path']},{status},"
                                f"{elapsed:.2f},{len(proteins)},{len(kg)},{error}\n")

                # 记录片段统计
                if fragment_stats:
                    all_fragment_stats.append(fragment_stats)
                    for chain_id, chain_stats in fragment_stats["chains"].items():
                        with open(fragments_log_path, 'a') as f_log:
                            f_log.write(f"{fragment_stats['file_name']},{fragment_stats['protein_id']},"
                                        f"{chain_id},{chain_stats['valid_residues']},{chain_stats['fragments']}\n")

                # 更新序列日志
                with open(sequences_log_path, 'a', buffering=1) as s_log:
                    for fragment_id, data in proteins.items():
                        s_log.write(
                            f"{fragment_id},{data['protein_id']},{data['chain_id']},{len(data['sequence'])},{data['sequence']}\n")

                # 累积数据
                all_proteins.update(proteins)
                all_knowledge_graphs.update(kg)

                # 每处理1000个文件保存一次中间结果
                if file_counter % 100000 == 0:
                    logger.info(f"已处理 {file_counter}/{len(file_list)} 个文件，保存当前批次结果...")

                    batch_output_dir = os.path.join(all_data_dir, f"batch_{batch_id}")
                    os.makedirs(batch_output_dir, exist_ok=True)

                    # 保存蛋白质数据
                    save_results_chunked(
                        all_proteins, batch_output_dir,
                        base_name="protein_data",
                        chunk_size=chunk_size
                    )

                    # 保存知识图谱
                    if build_kg and all_knowledge_graphs:
                        save_knowledge_graphs(
                            all_knowledge_graphs, batch_output_dir,
                            base_name="protein_kg",
                            chunk_size=chunk_size
                        )

                    # 重置集合并增加批次ID
                    all_proteins = {}
                    all_knowledge_graphs = {}
                    batch_id += 1

                    # 更新统计信息
                    logger.info(f"当前处理统计: 成功={stats['processed_files']}, 失败={stats['failed_files']}, "
                                f"片段={stats['total_fragments']}")

    # 保存最后一批数据
    if all_proteins:
        logger.info(f"保存最后一批结果 (剩余 {len(all_proteins)} 个蛋白质片段)...")

        final_batch_dir = os.path.join(all_data_dir, f"batch_{batch_id}")
        os.makedirs(final_batch_dir, exist_ok=True)

        # 保存蛋白质数据
        save_results_chunked(
            all_proteins, final_batch_dir,
            base_name="protein_data",
            chunk_size=chunk_size
        )

        # 保存知识图谱
        if build_kg and all_knowledge_graphs:
            save_knowledge_graphs(
                all_knowledge_graphs, final_batch_dir,
                base_name="protein_kg",
                chunk_size=chunk_size
            )

    logger.info(f"片段统计数据已写入: {fragments_log_path}")
    logger.info(f"处理日志已写入: {processing_log_path}")
    logger.info(f"创建的片段总数: {stats['total_fragments']}")

    return stats, all_data_dir

def save_chunks(data_dict, base_path, chunk_size):
    """将字典数据分块保存为JSON文件"""
    keys = list(data_dict.keys())
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i:i + chunk_size]
        chunk_data = {k: data_dict[k] for k in chunk_keys}
        chunk_file = f"{base_path}_{i//chunk_size + 1}.json"
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f)


def save_pyg_graphs(graph_data, output_dir, chunk_size=1000):
    """将知识图谱保存为PyG格式"""
    if not graph_data:
        logger.warning("没有知识图谱数据可保存为PyG格式")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"开始将{len(graph_data)}个图保存为PyG格式...")

    # 分块保存
    keys = list(graph_data.keys())
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i:i + chunk_size]
        chunk_file = os.path.join(output_dir, f"filtered_pyg_chunk_{i // chunk_size + 1}.pt")

        # 转换为PyG格式并保存
        pyg_graphs = {}
        for k in chunk_keys:
            if k in graph_data:
                nx_graph = graph_data[k]
                # NetworkX转PyG格式
                edge_index = []
                edge_attr = []
                for u, v, data in nx_graph.edges(data=True):
                    edge_index.append([u, v])
                    edge_attr.append([data.get('weight', 1.0), data.get('bond_type', 0)])

                # 节点特征
                x = []
                for node, attrs in nx_graph.nodes(data=True):
                    feat = [attrs.get(f, 0.0) for f in ['hydropathy', 'charge', 'mw']]
                    x.append(feat)

                # 创建PyG Data对象
                if len(edge_index) > 0:
                    edge_index = torch.tensor(edge_index).t().contiguous()
                    edge_attr = torch.tensor(edge_attr)
                    x = torch.tensor(x)
                    pyg_graphs[k] = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # 保存当前块
        torch.save(pyg_graphs, chunk_file)
        logger.info(f"已保存PyG图块 {i // chunk_size + 1}: {chunk_file}")

    logger.info(f"PyG格式图保存完成，共{len(keys)}个图，分为{(len(keys) - 1) // chunk_size + 1}个块")


def filter_and_save_results(all_data_dir, output_dir, redundancy_threshold=0.9, chunk_size=1000, pyg_format=True):
    """加载所有中间结果，进行去冗余并保存到FILTERED目录"""
    logger.info("开始加载所有处理结果...")

    # 创建去冗余子文件夹
    filtered_dir = os.path.join(output_dir, "FILTERED")
    os.makedirs(filtered_dir, exist_ok=True)

    # 加载所有批次的数据
    all_proteins = {}
    all_knowledge_graphs = {}

    try:
        batch_dirs = [d for d in os.listdir(all_data_dir) if d.startswith("batch_")]
        for batch_dir in tqdm(batch_dirs, desc="加载批次数据"):
            full_batch_dir = os.path.join(all_data_dir, batch_dir)

            # 加载蛋白质数据
            protein_files = [f for f in os.listdir(full_batch_dir) if
                             f.startswith("protein_data_chunk_") and f.endswith(".json")]
            for pf in protein_files:
                try:
                    with open(os.path.join(full_batch_dir, pf), 'r') as f:
                        protein_data = json.load(f)
                        all_proteins.update(protein_data)
                except Exception as e:
                    logger.warning(f"无法加载蛋白质数据文件 {pf}: {str(e)}")

            # 加载图谱数据
            kg_dir = os.path.join(full_batch_dir, "knowledge_graphs_json")
            if os.path.exists(kg_dir):
                kg_files = [f for f in os.listdir(kg_dir) if f.startswith("protein_kg_chunk_") and f.endswith(".json")]
                for kf in kg_files:
                    try:
                        with open(os.path.join(kg_dir, kf), 'r') as f:
                            kg_data = json.load(f)
                            all_knowledge_graphs.update(kg_data)
                    except Exception as e:
                        logger.warning(f"无法加载图谱数据文件 {kf}: {str(e)}")

        logger.info(f"已加载 {len(all_proteins)} 个蛋白质片段和 {len(all_knowledge_graphs)} 个知识图谱")

        if not all_proteins:
            logger.error("未找到任何有效的蛋白质数据，无法进行去冗余")
            return {}, {}

        # 使用remove_redundancy_pipeline进行去冗余
        logger.info(f"开始使用相似度阈值 {redundancy_threshold} 进行冗余过滤...")
        filtered_result = remove_redundancy_pipeline(
            all_proteins,
            all_knowledge_graphs,
            similarity_threshold=redundancy_threshold
        )

        # 根据remove_redundancy_pipeline的返回值类型处理结果
        if isinstance(filtered_result, tuple) and len(filtered_result) == 2:
            filtered_proteins, filtered_graphs = filtered_result
        else:
            filtered_proteins = filtered_result
            filtered_graphs = {k: all_knowledge_graphs[k] for k in filtered_proteins if k in all_knowledge_graphs}

        logger.info(f"去冗余完成: 从 {len(all_proteins)} 个片段中保留了 {len(filtered_proteins)} 个 "
                    f"({len(filtered_proteins) / len(all_proteins) * 100:.1f}%)")

        # 保存序列数据
        logger.info("保存去冗余后的序列数据...")
        save_chunks(filtered_proteins, os.path.join(filtered_dir, "filtered_proteins_chunk"), chunk_size)

        # 保存图谱数据 - JSON格式
        if filtered_graphs:
            logger.info("保存去冗余后的图谱数据...")
            kg_json_dir = os.path.join(filtered_dir, "knowledge_graphs_json")
            os.makedirs(kg_json_dir, exist_ok=True)
            save_chunks(filtered_graphs, os.path.join(kg_json_dir, "filtered_kg_chunk"), chunk_size)

        # 如果需要也保存PyG格式
        if pyg_format and filtered_graphs:
            logger.info("保存去冗余后的PyG格式图谱数据...")
            kg_pyg_dir = os.path.join(filtered_dir, "knowledge_graphs_pyg")
            os.makedirs(kg_pyg_dir, exist_ok=True)
            save_pyg_graphs(filtered_graphs, kg_pyg_dir, chunk_size)

        # 保存过滤摘要信息
        summary_file = os.path.join(filtered_dir, "filtering_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"过滤时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"相似度阈值: {redundancy_threshold}\n")
            f.write(f"原始片段数量: {len(all_proteins)}\n")
            f.write(f"过滤后片段数量: {len(filtered_proteins)}\n")
            f.write(f"保留比例: {len(filtered_proteins) / len(all_proteins) * 100:.1f}%\n")
            if filtered_graphs:
                f.write(f"过滤后图谱数量: {len(filtered_graphs)}\n")

        logger.info(f"过滤结果已保存到: {filtered_dir}")
        return filtered_proteins, filtered_graphs

    except Exception as e:
        logger.error(f"去冗余处理失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}, {}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从PDB/CIF文件中提取蛋白质结构数据并构建知识图谱")
    parser.add_argument("input", help="输入PDB/CIF文件或包含这些文件的目录")
    parser.add_argument("--output_dir", "-o", default="./pdb_data",
                        help="输出目录 (默认: ./pdb_data)")
    parser.add_argument("--mode", choices=["extract", "filter"], default="extract",
                        help="处理模式: extract(提取)或filter(去冗余)")
    parser.add_argument("--max_length", "-m", type=int, default=50,
                        help="最大序列长度 (默认: 50)")
    parser.add_argument("--step_size", "-s", type=int, default=25,
                        help="滑动窗口步长 (默认: 25)")
    parser.add_argument("--chunk_size", "-c", type=int, default=1000,
                        help="输出分块大小 (默认: 1000)")
    parser.add_argument("--build_kg", "-k", action="store_true", default=True,
                        help="构建知识图谱 (默认: True)")
    parser.add_argument("--workers", "-w", type=int,
                        help="并行工作进程数 (默认: CPU核心数-1)")
    parser.add_argument("--k_neighbors", type=int, default=8,
                        help="空间邻接的K近邻数 (默认: 8)")
    parser.add_argument("--distance_threshold", type=float, default=8.0,
                        help="空间邻接距离阈值 (默认: 8.0埃)")
    parser.add_argument("--plddt_threshold", type=float, default=70.0,
                        help="AlphaFold pLDDT质量得分阈值 (默认: 70.0)")
    parser.add_argument("--respect_ss", action="store_true", default=True,
                        help="是否尊重二级结构边界进行片段划分")
    parser.add_argument("--pyg_format", action="store_true", default=True,
                        help="是否保存为PyTorch Geometric格式")
    parser.add_argument("--limit", type=int,
                        help="限制处理的文件数量 (用于测试)")
    parser.add_argument("--redundancy_threshold", type=float, default=0.9,
                        help="Similarity threshold for redundancy filtering (0.0-1.0)")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="是否使用GPU加速相似度计算")
    global args
    args = parser.parse_args()

    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    global logger  # 使用全局logger变量
    logger, log_file_path = setup_logging(args.output_dir)
    logger.info(f"日志将写入文件: {log_file_path}")

    # 根据模式选择运行方式
    if args.mode == "filter":
        # 运行独立去冗余模式
        filter_redundancy_mode(args)
        return

    # 查找输入文件
    if os.path.isdir(args.input):
        pdb_files = find_pdb_files(args.input)
    else:
        pdb_files = [args.input] if os.path.isfile(args.input) else []

    if args.limit:
        pdb_files = pdb_files[:args.limit]
        logger.info(f"限制处理文件数量: {args.limit}")

    if not pdb_files:
        logger.error(f"未找到PDB/CIF文件，请检查输入路径: {args.input}")
        return

    logger.info(f"开始处理 {len(pdb_files)} 个PDB/CIF文件...")

    # 处理文件 - 修改后的函数现在返回统计信息和中间数据目录
    stats, all_data_dir = process_file_batch_parallel(
        pdb_files,
        args.output_dir,
        max_length=args.max_length,
        step_size=args.step_size,
        build_kg=args.build_kg,
        n_workers=args.workers,
        k_neighbors=args.k_neighbors,
        distance_threshold=args.distance_threshold,
        plddt_threshold=args.plddt_threshold,
        respect_ss=args.respect_ss,
    )

    # 处理结果统计
    logger.info("\n处理完成:")
    logger.info(f"- 处理的文件总数: {stats['processed_files']}")
    logger.info(f"- 提取的蛋白质片段总数: {stats['extracted_chains']}")
    logger.info(f"- 生成的知识图谱总数: {stats['knowledge_graphs']}")
    logger.info(f"- 失败的文件数: {stats.get('failed_files', 0)}")

    # 从保存的中间结果中加载数据并进行冗余过滤
    logger.info("开始进行冗余过滤...")
    filtered_proteins, filtered_graphs = filter_and_save_results(
        all_data_dir,
        args.output_dir,
        redundancy_threshold=args.redundancy_threshold,
        chunk_size=args.chunk_size,
        pyg_format=args.pyg_format
    )

    logger.info("处理完成!")




if __name__ == "__main__":
    main()