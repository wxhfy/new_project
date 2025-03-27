#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图嵌入系统 - 配置模块

管理系统配置参数，包括模型超参数、训练设置和数据处理选项。
"""

import os
import yaml
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """模型配置"""
    in_channels: int = 21  # 氨基酸类型特征 + 额外特征
    hidden_channels: int = 128
    out_channels: int = 64
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    edge_dim: Optional[int] = 8
    add_self_loops: bool = True
    readout_type: str = "multihead"  # ['mean', 'max', 'add', 'attention', 'multihead']
    jk_mode: str = "cat"  # ['cat', 'lstm', 'max', 'last']
    use_layer_norm: bool = True
    node_level: bool = True
    graph_level: bool = True
    use_edge_attr: bool = True


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "./data/processed"
    task_type: str = "graph"  # ['node', 'edge', 'graph']
    batch_size: int = 32
    num_workers: int = 4
    in_memory: bool = False
    use_cache: bool = True
    format_type: str = "pdb"  # ['pdb', 'fasta', 'mmcif']
    label_file: Optional[str] = None
    node_features: List[str] = field(default_factory=lambda: ["amino_acid", "position"])
    transform_type: Optional[str] = None
    transform_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """训练配置"""
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0001
    scheduler_type: str = "cosine"  # ['step', 'cosine', 'plateau', 'none']
    warmup_epochs: int = 5
    early_stopping: int = 20
    num_tasks: int = 1
    loss_type: str = "auto"  # ['bce', 'mse', 'ce', 'auto']
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    device: str = "cuda"
    seed: int = 42
    fp16: bool = False
    logging_steps: int = 100


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(config_path: str) -> Config:
    """从YAML文件加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = Config(
        model=ModelConfig(**config_dict.get('model', {})),
        data=DataConfig(**config_dict.get('data', {})),
        train=TrainConfig(**config_dict.get('train', {}))
    )
    return config


def save_config(config: Config, config_path: str):
    """保存配置到YAML文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    config_dict = {
        'model': {k: v for k, v in config.model.__dict__.items()},
        'data': {k: v for k, v in config.data.__dict__.items()},
        'train': {k: v for k, v in config.train.__dict__.items()}
    }

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='蛋白质图嵌入系统')
    parser.add_argument('--config', type=str, default='./config.yaml', help='配置文件路径')
    parser.add_argument('--data_dir', type=str, help='数据目录')
    parser.add_argument('--save_dir', type=str, help='保存目录')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--device', type=str, help='设备')
    parser.add_argument('--seed', type=int, help='随机种子')

    return parser.parse_args()


def update_config_with_args(config: Config, args):
    """使用命令行参数更新配置"""
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.save_dir:
        config.train.save_dir = args.save_dir
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.train.epochs = args.epochs
    if args.lr:
        config.train.lr = args.lr
    if args.device:
        config.train.device = args.device
    if args.seed:
        config.train.seed = args.seed

    return config