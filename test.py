#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图神经网络分布式训练测试脚本

基于原始train_embed.py流程，使用少量数据进行分布式训练测试，
保持与原始训练脚本完全一致的行为，确保代码在分布式环境中正常工作。

作者: wxhfy
日期: 2025-04-08
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import logging
import importlib.util
import random
import numpy as np
from contextlib import nullcontext

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GNN_DIST_TEST")


def parse_args():
    """解析命令行参数，兼容分布式启动"""
    parser = argparse.ArgumentParser(description="蛋白质图神经网络分布式测试")
    parser.add_argument("--config", type=str, default="utils/config.py", help="配置文件路径")
    parser.add_argument("--data_path", type=str, default="data/train_data.pt", help="训练数据路径")
    parser.add_argument("--val_path", type=str, default="data/val_data.pt", help="验证数据路径")
    parser.add_argument("--sample_size", type=int, default=100, help="训练样本数量限制")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 同时支持两种格式的local_rank参数
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地进程排名")
    parser.add_argument("--local-rank", type=int, default=-1, help=argparse.SUPPRESS)

    parser.add_argument("--nodes", type=int, default=1, help="节点数")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="每节点GPU数")
    parser.add_argument("--nr", type=int, default=0, help="节点排名")

    args = parser.parse_args()

    # 使用环境变量作为备选
    if args.local_rank == -1 and args.__dict__["local-rank"] != -1:
        args.local_rank = args.__dict__["local-rank"]

    # 尝试从环境变量获取
    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    return args


def setup_distributed(args):
    """正确设置分布式环境"""
    # 优先从环境变量获取分布式信息
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # 回退到计算值
        world_size = args.gpus_per_node * args.nodes
        rank = args.nr * args.gpus_per_node + args.local_rank
        local_rank = args.local_rank

    # 设置设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 初始化进程组
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        world_size=world_size,
        rank=rank
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return device, rank, world_size


def load_module_from_path(path, module_name):
    """从路径加载Python模块"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"找不到模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def set_random_seed(seed, deterministic=False):
    """设置随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def train_test(gpu, args):
    """分布式训练测试函数"""
    # 1. 设置分布式环境
    is_distributed, global_rank, world_size = setup_distributed(args)
    is_master = global_rank == 0

    # 2. 设置设备
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # 3. 设置日志
    if is_master:
        # 创建输出目录
        debug_dir = os.path.join("outputs", f"dist_test_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(debug_dir, exist_ok=True)
        log_file = os.path.join(debug_dir, f"rank_{global_rank}.log")

        # 添加文件处理器到日志
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info(f"分布式训练设置: 世界大小={world_size}, 当前排名={global_rank}")
        logger.info(f"调试日志将保存到: {log_file}")

    # 4. 设置随机种子
    set_random_seed(args.seed + global_rank)

    # 5. 加载配置
    try:
        config_module = load_module_from_path(args.config, "config")
        config = config_module.Config()

        # 更新配置
        config.TRAIN_CACHE = args.data_path
        config.VAL_CACHE = args.val_path
        config.BATCH_SIZE = args.batch_size
        config.EPOCHS = args.epochs
        config.SEED = args.seed
        config.USE_DISTRIBUTED = True
        config.WORLD_SIZE = world_size
        config.GLOBAL_RANK = global_rank
        config.LOCAL_RANK = gpu
        config.DEVICE = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

        # 设置输出目录
        if is_master:
            config.LOG_DIR = os.path.join(debug_dir, "logs")
            config.MODEL_DIR = os.path.join(debug_dir, "models")
            config.RESULT_DIR = os.path.join(debug_dir, "results")

            os.makedirs(config.LOG_DIR, exist_ok=True)
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            os.makedirs(config.RESULT_DIR, exist_ok=True)

            logger.info(f"成功加载配置: {args.config}")
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        dist.destroy_process_group()
        return

    # 6. 加载训练脚本模块
    try:
        train_module = load_module_from_path("train_embed.py", "train_embed")
        logger.info("成功加载训练模块") if is_master else None
    except Exception as e:
        logger.error(f"加载训练模块失败: {e}")
        dist.destroy_process_group()
        return

    # 7. 准备数据加载器
    try:
        # 使用训练模块的数据集类
        ProteinDataset = train_module.ProteinDataset

        # 创建小数据集
        train_dataset = ProteinDataset(
            data_path=args.data_path,
            max_samples=args.sample_size if hasattr(args, 'sample_size') else None,
            seed=args.seed
        )

        val_dataset = ProteinDataset(
            data_path=args.val_path,
            max_samples=args.sample_size // 5 if hasattr(args, 'sample_size') else None,
            seed=args.seed
        )

        # 创建分布式采样器
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            seed=args.seed
        )

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False,
            seed=args.seed
        )

        # 创建数据加载器
        train_loader = train_module.ProteinDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # 不需要，已使用DistributedSampler
            sampler=train_sampler
        )

        val_loader = train_module.ProteinDataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler
        )

        if is_master:
            logger.info(f"数据集创建成功 - 训练: {len(train_dataset)}个样本, 验证: {len(val_dataset)}个样本")
            logger.info(f"批处理大小: {args.batch_size}, 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
    except Exception as e:
        logger.error(f"创建数据集失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        dist.destroy_process_group()
        return

    # 8. 创建训练器
    try:
        # 解决分布式设置问题
        # 确保设备是 torch.device 类型
        config.DEVICE = torch.device(config.DEVICE)

        # 创建训练器
        ProteinMultiModalTrainer = train_module.ProteinMultiModalTrainer
        with nullcontext():
            trainer = ProteinMultiModalTrainer(config)

        if is_master:
            logger.info("成功创建训练器")
    except Exception as e:
        logger.error(f"创建训练器失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        dist.destroy_process_group()
        return

    # 9. 执行训练
    try:
        if is_master:
            logger.info("开始训练...")
        trainer.train(train_loader, val_loader)
        if is_master:
            logger.info("训练完成")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # 10. 清理分布式环境
    dist.destroy_process_group()


def main():
    """主函数"""
    args = parse_args()

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，将使用CPU进行训练")
        args.gpus_per_node = 0

    # 当使用torch.distributed.launch启动时
    if args.local_rank != -1:
        # 由torch.distributed.launch设置的环境
        logger.info(f"使用torch.distributed.launch模式，本地排名: {args.local_rank}")
        train_test(args.local_rank, args)
    else:
        # 多进程启动
        ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
        args.gpus_per_node = min(args.gpus_per_node, ngpus_per_node)

        if args.gpus_per_node > 1:
            logger.info(f"使用多进程方式启动，每节点GPU数: {args.gpus_per_node}")
            mp.spawn(train_test, nprocs=args.gpus_per_node, args=(args,))
        else:
            # 单GPU或CPU训练
            logger.info("使用单进程方式启动")
            train_test(0, args)


if __name__ == "__main__":
    main()