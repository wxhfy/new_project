#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统计PDB文件中序列长度分布的工具

此脚本用于检查目录中所有PDB.gz文件，统计其中序列长度≤50的序列数量
"""

import os
import glob
import gzip
import logging
import multiprocessing
from collections import defaultdict
from Bio import PDB
from Bio.PDB.Polypeptide import is_aa
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_sequence_from_pdb(pdb_file):
    """
    从PDB文件中提取氨基酸序列

    参数:
        pdb_file: PDB文件路径

    返回:
        dict: 包含链ID和对应序列长度的字典
    """
    try:
        sequences = {}

        # 创建PDB解析器并读取临时解压文件
        parser = PDB.PDBParser(QUIET=True)

        # 对gzip文件进行处理
        if pdb_file.endswith('.gz'):
            with gzip.open(pdb_file, 'rt') as f:
                content = f.read()

            # 创建临时文件名
            temp_file = os.path.join('/tmp', os.path.basename(pdb_file)[:-3])

            # 将内容写入临时文件
            with open(temp_file, 'w') as f:
                f.write(content)

            structure = parser.get_structure('protein', temp_file)

            # 处理完毕后删除临时文件
            try:
                os.remove(temp_file)
            except:
                pass
        else:
            structure = parser.get_structure('protein', pdb_file)

        # 提取每条链的序列
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                residues = []

                # 只收集标准氨基酸残基
                for residue in chain:
                    if is_aa(residue.get_resname(), standard=True):
                        residues.append(residue)

                # 记录序列长度
                if residues:
                    sequences[chain_id] = len(residues)

        return sequences

    except Exception as e:
        logger.error(f"处理文件 {pdb_file} 时出错: {str(e)}")
        return {}


def process_pdb_file(pdb_file):
    """处理单个PDB文件并返回结果"""
    try:
        file_name = os.path.basename(pdb_file)
        sequences = extract_sequence_from_pdb(pdb_file)

        # 统计序列长度≤50的序列数量
        short_seq_count = sum(1 for length in sequences.values() if length <= 50)
        total_seq_count = len(sequences)

        return {
            'file': file_name,
            'total_sequences': total_seq_count,
            'short_sequences': short_seq_count,
            'sequence_lengths': sequences
        }
    except Exception as e:
        logger.error(f"处理文件 {pdb_file} 失败: {str(e)}")
        return {
            'file': os.path.basename(pdb_file),
            'total_sequences': 0,
            'short_sequences': 0,
            'sequence_lengths': {},
            'error': str(e)
        }


def main():
    """主函数"""
    # 目标目录
    target_dir = '/home/20T-1/fyh0106/swiss_prot/'

    # 查找所有.pdb.gz文件
    pdb_files = glob.glob(os.path.join(target_dir, "*.pdb.gz"))

    if not pdb_files:
        logger.warning(f"在 {target_dir} 中未找到任何.pdb.gz文件")
        return

    logger.info(f"找到 {len(pdb_files)} 个.pdb.gz文件，开始分析...")

    # 使用多进程处理
    num_processes = multiprocessing.cpu_count()-20  # 限制最多使用16个核心

    results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_pdb_file, pdb_files),
                           total=len(pdb_files),
                           desc="处理PDB文件"):
            results.append(result)

    # 汇总统计
    total_files = len(results)
    total_sequences = sum(r['total_sequences'] for r in results)
    short_sequences = sum(r['short_sequences'] for r in results)

    # 计算长度分布
    length_distribution = defaultdict(int)
    for result in results:
        for length in result['sequence_lengths'].values():
            length_distribution[length] += 1

    # 输出结果
    logger.info("=" * 60)
    logger.info(f"处理完成! 分析了 {total_files} 个PDB文件")
    logger.info(f"总序列数: {total_sequences}")
    logger.info(f"长度≤50的序列数: {short_sequences}")
    logger.info(f"占比: {short_sequences / total_sequences * 100:.2f}%")

    # 详细的文件统计（仅显示前10个有短序列的文件）
    short_seq_files = [r for r in results if r['short_sequences'] > 0]
    short_seq_files.sort(key=lambda x: x['short_sequences'], reverse=True)

    if short_seq_files:
        logger.info("=" * 60)
        logger.info("包含短序列的文件统计 (前10个):")
        for i, result in enumerate(short_seq_files[:10]):
            logger.info(
                f"{i + 1}. {result['file']}: 总序列数={result['total_sequences']}, 短序列数={result['short_sequences']}")

    # 序列长度分布统计
    length_ranges = [
        (0, 10), (11, 20), (21, 30), (31, 40), (41, 50),
        (51, 100), (101, 200), (201, 300), (301, 500), (501, float('inf'))
    ]

    logger.info("=" * 60)
    logger.info("序列长度分布:")
    for start, end in length_ranges:
        if end == float('inf'):
            range_str = f">= {start}"
        else:
            range_str = f"{start}-{end}"

        count = sum(length_distribution[l] for l in length_distribution if start <= l <= end)
        percentage = count / total_sequences * 100 if total_sequences > 0 else 0
        logger.info(f"{range_str:10s}: {count:5d} ({percentage:.2f}%)")


if __name__ == "__main__":
    main()