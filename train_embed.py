#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图-序列双模态融合表示训练脚本

该模块实现了蛋白质图嵌入与序列嵌入的融合训练流程,
基于对比学习优化表示空间的一致性，为后续扩散模型提供高质量嵌入基础。

作者: wxhfy
日期: 2025-03-29
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import argparse
import logging
import random
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from models.gat_models import (
    ProteinGATv2Encoder,
    ProteinLatentMapper,
    CrossModalContrastiveHead,
    ResidualBlock
)
from models.layers import SequenceStructureFusion
from utils.config import Config
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProteinTensor, LogitsConfig

os.environ["INFRA_PROVIDER"] = "True"
os.environ["ESM_CACHE_DIR"] = "data/weight"  # 指定包含模型文件的目录

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 新增：自定义蛋白质数据类，兼容新旧版本PyG并正确处理字符串字段
class ProteinData:
    """
    兼容新旧版本PyG的蛋白质数据类，分离处理字符串等非张量字段
    """

    def __init__(self, **kwargs):
        self.x = None  # 节点特征
        self.edge_index = None  # 边索引
        self.edge_attr = None  # 边特征（可选）
        self.pos = None  # 坐标（可选）
        self.batch = None  # 批次信息（在批处理中使用）

        # 非张量字段
        self.string_data = {}

        # 处理所有输入字段
        for key, value in kwargs.items():
            if isinstance(value, (str, list)) and not isinstance(value, (torch.Tensor, np.ndarray)):
                # 字符串和列表类型存入string_data
                self.string_data[key] = value
            else:
                # 张量类型直接设为属性
                setattr(self, key, value)

    def __getattr__(self, key):
        # 优先从对象属性中查找
        if key in self.__dict__:
            return self.__dict__[key]
        # 然后从string_data字典中查找
        elif key in self.string_data:
            return self.string_data[key]
        # 最后报错
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "string_data":
            self.__dict__[key] = value
        elif isinstance(value, (str, list)) and not isinstance(value, (torch.Tensor, np.ndarray)):
            # 确保string_data已初始化
            if "string_data" not in self.__dict__:
                self.__dict__["string_data"] = {}
            self.string_data[key] = value
        else:
            self.__dict__[key] = value

    def keys(self):
        """获取所有键名"""
        keys = list(self.__dict__.keys())
        if "string_data" in keys:
            keys.remove("string_data")
            keys.extend(self.string_data.keys())
        return keys

    def items(self):
        """获取所有键值对"""
        result = {}
        for k, v in self.__dict__.items():
            if k != "string_data":
                result[k] = v
        result.update(self.string_data)
        return result.items()

    def to(self, device):
        """将张量移至指定设备"""
        for key, value in self.__dict__.items():
            if key != "string_data" and hasattr(value, "to") and callable(value.to):
                self.__dict__[key] = value.to(device)
        return self

    def clone(self):
        """创建对象的副本"""
        new_obj = ProteinData()
        for key, value in self.__dict__.items():
            if key != "string_data":
                if hasattr(value, "clone") and callable(value.clone):
                    setattr(new_obj, key, value.clone())
                else:
                    setattr(new_obj, key, value)
        new_obj.string_data = self.string_data.copy()
        return new_obj


# 新增：批处理类
class ProteinBatch:
    """处理批次中的多个ProteinData对象"""

    def __init__(self, data_list):
        self.num_graphs = len(data_list)
        self.batch = torch.zeros(0, dtype=torch.long)

        # 合并张量属性
        for key in ["x", "edge_index", "edge_attr", "pos"]:
            self._merge_tensor_attr(data_list, key)

        # 处理字符串属性 - 保留为列表
        self._merge_string_attrs(data_list)

        # 创建批次索引
        offset = 0
        for i, data in enumerate(data_list):
            num_nodes = data.x.size(0)
            self.batch = torch.cat([self.batch, torch.full((num_nodes,), i, dtype=torch.long)])

            # 更新边索引的偏移量
            if i < len(data_list) - 1:
                if hasattr(self, "edge_index") and self.edge_index is not None:
                    data_list[i + 1].edge_index = data_list[i + 1].edge_index + offset

            offset += num_nodes

    def _merge_tensor_attr(self, data_list, key):
        # 合并指定张量属性
        tensors = [getattr(data, key) for data in data_list if hasattr(data, key) and getattr(data, key) is not None]
        if tensors:
            setattr(self, key, torch.cat(tensors, dim=0 if key != "edge_index" else 1))
        else:
            setattr(self, key, None)

    def _merge_string_attrs(self, data_list):
        # 收集所有出现的字符串属性
        string_keys = set()
        for data in data_list:
            if hasattr(data, "string_data"):
                string_keys.update(data.string_data.keys())

        # 为每个字符串属性创建合并列表
        for key in string_keys:
            values = []
            for data in data_list:
                if hasattr(data, "string_data") and key in data.string_data:
                    values.append(data.string_data[key])
                else:
                    values.append(None)  # 保持索引对齐
            setattr(self, key, values)

    def to(self, device):
        """将批次中的张量移至指定设备"""
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device)
        return self


# 新增：版本无关的数据转换函数
def convert_data_format(data):
    """将PyG的Data对象或字典转换为我们自定义的ProteinData格式"""
    if isinstance(data, ProteinData):
        return data

    # 创建新的ProteinData对象
    protein_data = ProteinData()

    # 处理各种可能的输入类型
    if hasattr(data, "keys"):  # 类字典对象
        for key in data.keys():
            try:
                value = data[key] if hasattr(data, "__getitem__") else getattr(data, key)
                setattr(protein_data, key, value)
            except:
                logger.warning(f"无法获取属性: {key}")
    elif isinstance(data, dict):  # 普通字典
        for key, value in data.items():
            setattr(protein_data, key, value)
    else:
        # 未知类型，尝试提取常见属性
        for key in ["x", "edge_index", "edge_attr", "pos", "batch", "sequence"]:
            if hasattr(data, key):
                try:
                    setattr(protein_data, key, getattr(data, key))
                except:
                    pass

    # 验证必要字段
    if protein_data.x is None or protein_data.edge_index is None:
        logger.warning("转换后的数据缺少x或edge_index字段")

    return protein_data


# 新增：自定义数据集类
class ProteinDataset:
    """处理蛋白质图数据的数据集，兼容不同版本的PyG"""

    def __init__(self, data_path, subset_ratio=None, max_samples=None, seed=42):
        self.data_path = data_path
        self.subset_ratio = subset_ratio
        self.max_samples = max_samples
        self.seed = seed
        self.data_list = self._load_and_process()

        # 确保 data_list 不为空
        if not self.data_list:
            logger.warning(f"从 {data_path} 加载的数据为空，创建一个包含占位元素的列表")
            # 创建一个仅包含默认元素的数据列表作为后备
            self.data_list = [self._create_default_data()]

    def _create_default_data(self):
        """创建一个默认的空数据对象"""
        # 创建一个最小的图对象
        protein_data = ProteinData()
        protein_data.x = torch.zeros((1, 20), dtype=torch.float)  # 一个节点，20维特征
        protein_data.edge_index = torch.zeros((2, 0), dtype=torch.long)  # 空边
        protein_data.string_data["sequence"] = "A"  # 最小序列
        return protein_data

    def _load_and_process(self):
        """加载数据并处理成兼容格式"""
        try:
            # 检查文件是否存在
            if not os.path.exists(self.data_path):
                logger.error(f"数据文件不存在: {self.data_path}")
                return []

            # 加载数据
            logger.info(f"加载数据: {self.data_path}")
            data = torch.load(self.data_path)

            # 确保是列表格式
            if not isinstance(data, list):
                data = [data]

            # 检查数据是否为空
            if not data:
                logger.warning(f"从 {self.data_path} 加载的数据为空")
                return []

            # 可选：使用数据子集
            if self.subset_ratio is not None or self.max_samples is not None:
                random.seed(self.seed)

                if self.subset_ratio is not None:
                    sample_size = max(int(len(data) * self.subset_ratio), 10)
                else:
                    sample_size = self.max_samples

                sample_size = min(sample_size, len(data))
                indices = random.sample(range(len(data)), sample_size)
                data = [data[i] for i in indices]
                logger.info(f"使用数据子集: {len(data)}个样本")

            # 转换为自定义ProteinData格式
            converted_data = []
            success_count = 0
            for i, item in enumerate(data):
                try:
                    protein_data = convert_data_format(item)
                    if protein_data is not None:  # 确保转换成功
                        converted_data.append(protein_data)
                        success_count += 1
                except Exception as e:
                    logger.warning(f"处理第{i}个样本时出错: {e}")

            logger.info(f"成功加载并转换{success_count}/{len(data)}个样本")
            return converted_data

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def __len__(self):
        """返回数据集长度"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """获取指定索引的数据项"""
        if 0 <= idx < len(self.data_list):
            return self.data_list[idx]
        else:
            logger.warning(f"索引越界: {idx}，返回默认数据")
            return self._create_default_data()


# 修复 ProteinDataLoader 类
class ProteinDataLoader:
    """蛋白质图数据的批处理加载器"""

    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, sampler=None):
        self.dataset = dataset
        # 确保 batch_size 不为 None，设置默认值
        self.batch_size = batch_size if batch_size is not None else 32
        self.shuffle = shuffle and sampler is None
        self.sampler = sampler
        self.num_workers = num_workers  # 当前版本不使用多进程

        # 验证参数
        if not isinstance(self.dataset, (list, tuple)) and not hasattr(self.dataset, '__len__'):
            raise ValueError("dataset 必须支持 __len__ 方法或是列表/元组类型")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数，当前值: {self.batch_size}")

        # 创建索引列表
        self.indices = list(range(len(dataset) if hasattr(dataset, '__len__') else len(list(dataset))))

        # 记录加载器状态
        logger.info(f"创建 ProteinDataLoader - 数据集大小: {len(self.indices)}, "
                    f"批大小: {self.batch_size}, 批次数: {len(self)}")

    def __len__(self):
        dataset_size = len(self.indices)
        # 安全地计算批次数
        return (dataset_size + self.batch_size - 1) // self.batch_size if dataset_size > 0 else 0

    def __iter__(self):
        # 随机打乱索引（如果需要）
        if self.shuffle:
            random.shuffle(self.indices)
        elif self.sampler is not None:
            # 使用提供的采样器
            try:
                self.indices = list(self.sampler)
            except Exception as e:
                logger.warning(f"使用采样器时出错: {e}，使用默认索引")

        # 批次处理
        dataset_size = len(self.indices)
        if dataset_size == 0:
            logger.warning("数据集为空，不产生批次")
            return

        for i in range(0, dataset_size, self.batch_size):
            # 安全地获取当前批次的索引
            batch_indices = self.indices[i:min(i + self.batch_size, dataset_size)]

            try:
                # 获取每个索引对应的数据项
                batch_data = []
                for j in batch_indices:
                    try:
                        item = self.dataset[j]
                        if item is not None:
                            batch_data.append(item)
                    except Exception as e:
                        logger.warning(f"获取数据项 {j} 时出错: {e}")

                if not batch_data:
                    logger.warning(f"批次 {i // self.batch_size} 中没有有效数据，跳过")
                    continue

                # 创建批次
                yield ProteinBatch(batch_data)
            except Exception as e:
                logger.warning(f"创建批次 {i // self.batch_size} 失败: {e}")

                # 尝试提供详细的错误信息以便调试
                import traceback
                logger.debug(f"创建批次详细错误: {traceback.format_exc()}")

                # 对于严重错误，可能需要直接中断迭代
                if "内存不足" in str(e) or "CUDA out of memory" in str(e):
                    logger.error("检测到内存不足错误，终止批次生成")
                    break

                continue


class ProteinMultiModalTrainer:
    """蛋白质多模态融合训练器"""

    def __init__(self, config):
        """
        初始化训练器

        参数:
            config: 配置对象
        """
        self.embedding_cache = None
        self.config = config
        self.device = config.DEVICE
        self.fp16_training = config.FP16_TRAINING
        # 安全获取批处理大小，并提供默认值
        self.batch_size = config.BATCH_SIZE
        if self.batch_size is None or self.batch_size <= 0:
            self.batch_size = 64
            logger.warning(f"无效的批处理大小，使用默认值: {self.batch_size}")
        self.log_dir = config.LOG_DIR

        # 是否为分布式训练主进程
        self.is_master = not config.USE_DISTRIBUTED or config.GLOBAL_RANK == 0

        # 设置随机种子
        self._set_seed(config.SEED)

        # 在主进程上创建TensorBoard写入器
        if self.is_master:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            logger.info(f"TensorBoard日志保存到: {self.log_dir}")
        else:
            self.writer = None

        # 初始化模型
        self._init_models()

        # 初始化优化器
        self._init_optimizers()

        # 初始化混合精度训练
        if self.fp16_training:
            self.scaler = GradScaler()

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 新增: 最佳验证分数，用于早停
        self.best_validation_score = float('inf')

    def _set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    def _init_models(self):
        """初始化模型组件"""
        logger.info("初始化模型组件...")

        # 1. 初始化GAT图编码器 - 根据数据结构优化参数
        try:
            # 尝试动态确定输入维度
            node_input_dim = self.config.NODE_INPUT_DIM
            edge_input_dim = self.config.EDGE_INPUT_DIM

            # 是否使用位置编码 - 根据数据结构调整
            use_pos_encoding = False  # 默认关闭，因为数据中不包含pos信息

            self.graph_encoder = ProteinGATv2Encoder(
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                hidden_dim=self.config.HIDDEN_DIM,
                output_dim=self.config.OUTPUT_DIM,
                num_layers=self.config.NUM_LAYERS,
                num_heads=self.config.NUM_HEADS,
                edge_types=self.config.EDGE_TYPES,
                dropout=self.config.DROPOUT,
                use_pos_encoding=use_pos_encoding,  # 禁用位置编码，因为数据中没有pos信息
                use_heterogeneous_edges=self.config.USE_HETEROGENEOUS_EDGES,
                use_edge_pruning=self.config.USE_EDGE_PRUNING,
                esm_guidance=self.config.ESM_GUIDANCE,  # 启用ESM注意力引导
                activation='gelu'
            ).to(self.device)

            logger.info(f"图编码器初始化成功 - 节点特征维度: {node_input_dim}, 边特征维度: {edge_input_dim}")
        except Exception as e:
            logger.error(f"图编码器初始化失败: {e}")
            raise

        # 2. ESM模型初始化
        logger.info(f"初始化ESM模型: {self.config.ESM_MODEL_NAME}")
        try:
            # 加载模型
            self.esm_model = ESMC.from_pretrained(self.config.ESM_MODEL_NAME, device=self.device)
            logger.info(f"ESM模型成功加载: {self.config.ESM_MODEL_NAME}")

            # 保存ESM引导标志
            self.use_esm_guidance = self.config.ESM_GUIDANCE

            # 初始化ESM注意力存储属性
            self.current_esm_attention = None

        except Exception as e:
            logger.error(f"ESM模型加载失败: {e}")
            logger.error(f"环境变量: ESM_CACHE_DIR={os.environ.get('ESM_CACHE_DIR')}")
            logger.error(f"模型名称: {self.config.ESM_MODEL_NAME}")
            # 降级处理 - 禁用ESM引导
            self.use_esm_guidance = False
            logger.warning("由于ESM模型加载失败，已禁用ESM引导功能")
            raise

        # 3. 潜空间映射器 - 将图嵌入映射到与ESM兼容的空间
        self.latent_mapper = ProteinLatentMapper(
            input_dim=self.config.OUTPUT_DIM,
            latent_dim=self.config.ESM_EMBEDDING_DIM,
            hidden_dim=self.config.HIDDEN_DIM * 2,
            dropout=self.config.DROPOUT
        ).to(self.device)

        # 4. 对比学习头 - 促进模态间对齐
        self.contrast_head = CrossModalContrastiveHead(
            embedding_dim=self.config.ESM_EMBEDDING_DIM,
            temperature=self.config.TEMPERATURE
        ).to(self.device)

        # 5. 序列-结构融合模块 - 整合两种模态的信息
        self.fusion_module = SequenceStructureFusion(
            seq_dim=self.config.ESM_EMBEDDING_DIM,
            graph_dim=self.config.ESM_EMBEDDING_DIM,  # 使用映射后的维度
            output_dim=self.config.FUSION_OUTPUT_DIM,
            hidden_dim=self.config.FUSION_HIDDEN_DIM,
            num_heads=self.config.FUSION_NUM_HEADS,
            num_layers=self.config.FUSION_NUM_LAYERS,
            dropout=self.config.FUSION_DROPOUT
        ).to(self.device)

        # 分布式训练设置
        if self.config.USE_DISTRIBUTED:
            # 同步BN层
            if self.config.SYNC_BN and self.config.NUM_GPUS > 1:
                self.graph_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.graph_encoder)
                self.latent_mapper = nn.SyncBatchNorm.convert_sync_batchnorm(self.latent_mapper)
                self.fusion_module = nn.SyncBatchNorm.convert_sync_batchnorm(self.fusion_module)

            # 封装为DDP模型
            self.graph_encoder = DDP(
                self.graph_encoder,
                device_ids=[self.config.LOCAL_RANK],
                output_device=self.config.LOCAL_RANK,
                find_unused_parameters=False
            )

            self.latent_mapper = DDP(
                self.latent_mapper,
                device_ids=[self.config.LOCAL_RANK],
                output_device=self.config.LOCAL_RANK,
                find_unused_parameters=False
            )

            self.contrast_head = DDP(
                self.contrast_head,
                device_ids=[self.config.LOCAL_RANK],
                output_device=self.config.LOCAL_RANK,
                find_unused_parameters=False
            )

            self.fusion_module = DDP(
                self.fusion_module,
                device_ids=[self.config.LOCAL_RANK],
                output_device=self.config.LOCAL_RANK,
                find_unused_parameters=False
            )

        # 打印模型信息
        if self.is_master:
            pytorch_total_params = sum(p.numel() for p in self.graph_encoder.parameters() if p.requires_grad)
            pytorch_total_params += sum(p.numel() for p in self.latent_mapper.parameters() if p.requires_grad)
            pytorch_total_params += sum(p.numel() for p in self.contrast_head.parameters() if p.requires_grad)
            pytorch_total_params += sum(p.numel() for p in self.fusion_module.parameters() if p.requires_grad)
            logger.info(f"模型总参数量: {pytorch_total_params / 1e6:.2f}M")

            # 打印ESM引导状态
            if self.use_esm_guidance:
                logger.info("已启用ESM注意力引导机制")
            else:
                logger.info("未启用ESM注意力引导机制")

    def _init_optimizers(self):
        """初始化优化器和学习率调度器"""
        # 收集所有参数
        params = [
            {'params': self.graph_encoder.parameters()},
            {'params': self.latent_mapper.parameters()},
            {'params': self.contrast_head.parameters()},
            {'params': self.fusion_module.parameters()}
        ]

        # 创建优化器
        if self.config.OPTIMIZER == "Adam":
            self.optimizer = optim.Adam(
                params,
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == "AdamW":
            self.optimizer = optim.AdamW(
                params,
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(
                params,
                lr=self.config.LEARNING_RATE,
                momentum=0.9,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.config.OPTIMIZER}")

        # 创建学习率调度器
        if self.config.LR_SCHEDULER == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif self.config.LR_SCHEDULER == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.EPOCHS
            )
        elif self.config.LR_SCHEDULER == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.EPOCHS
            )
        else:
            self.scheduler = None

        # 创建预热调度器
        if self.config.WARMUP_STEPS > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.WARMUP_STEPS
            )
        else:
            self.warmup_scheduler = None

    def _get_esm_embedding(self, sequences, sequence_ids=None):
        """
        获取序列的ESM嵌入与注意力，优先使用缓存，无需序列长度对齐

        参数:
            sequences (list): 氨基酸序列列表
            sequence_ids (list, optional): 序列ID列表，用于缓存查找

        返回:
            torch.Tensor: ESM嵌入矩阵，附带attention属性
        """
        batch_embeddings = []
        batch_attentions = []

        # 检查是否启用缓存
        use_cache = hasattr(self, 'embedding_cache') and self.embedding_cache is not None

        # 准备序列ID
        if sequence_ids is None and hasattr(self, 'current_batch_ids'):
            sequence_ids = self.current_batch_ids
        elif sequence_ids is None:
            # 使用序列哈希作为ID
            sequence_ids = [self._create_seq_hash(seq) for seq in sequences]

        # ESM氨基酸编码映射表
        ESM_AA_MAP = {
            'A': 5, 'C': 23, 'D': 13, 'E': 9, 'F': 18,
            'G': 6, 'H': 21, 'I': 12, 'K': 15, 'L': 4,
            'M': 20, 'N': 17, 'P': 14, 'Q': 16, 'R': 10,
            'S': 8, 'T': 11, 'V': 7, 'W': 22, 'Y': 19,
            '_': 32, 'X': 32
        }

        # 逐个处理序列
        for seq_idx, (seq, seq_id) in enumerate(zip(sequences, sequence_ids)):
            try:
                # 尝试从缓存获取嵌入
                if use_cache:
                    cached_data = self.embedding_cache.get_embedding(seq_id=seq_id, sequence=seq)
                    if cached_data is not None:
                        # 成功从缓存获取
                        embeddings = cached_data["embedding"]
                        attention = cached_data.get("attention", None)

                        # 处理嵌入维度
                        if embeddings.dim() == 4:
                            if embeddings.shape[0] == 1 and embeddings.shape[1] == 1:
                                embeddings = embeddings.squeeze(1)
                            else:
                                b, extra, s, d = embeddings.shape
                                embeddings = embeddings.reshape(b, s, d)

                        # 移动到当前设备
                        embeddings = embeddings.to(self.device)
                        if attention is not None:
                            attention = attention.to(self.device)

                        batch_embeddings.append(embeddings)
                        if attention is not None:
                            batch_attentions.append(attention)

                        # 跳过后续计算
                        continue

                # 缓存未命中，需计算嵌入
                # 序列预处理
                if not seq or len(seq) == 0:
                    seq = "A"

                cleaned_seq = ''.join(aa for aa in seq if aa in ESM_AA_MAP)
                if not cleaned_seq:
                    cleaned_seq = "A"

                # 序列编码
                token_ids = [0]  # BOS标记
                for aa in cleaned_seq:
                    token_ids.append(ESM_AA_MAP.get(aa, ESM_AA_MAP['X']))
                token_ids.append(2)  # EOS标记

                # 转换为张量 - 不添加额外维度，避免4D输入问题
                token_tensor = torch.tensor(token_ids, device=self.device)

                # 创建ESM输入
                protein_tensor = ESMProteinTensor(sequence=token_tensor)

                with torch.no_grad():
                    try:
                        # 使用安全计算方法
                        logits_output = self._safe_compute_embeddings(
                            self.esm_model,
                            protein_tensor,
                            LogitsConfig(sequence=True, return_embeddings=True)
                        )

                        # 提取嵌入
                        embeddings = logits_output.embeddings

                        # 处理四维输出
                        if embeddings.dim() == 4:
                            if embeddings.shape[0] == 1 and embeddings.shape[1] == 1:
                                embeddings = embeddings.squeeze(1)
                            else:
                                b, extra, s, d = embeddings.shape
                                embeddings = embeddings.reshape(b, s, d)

                        # 提取注意力信息
                        attention = None
                        if hasattr(logits_output, 'attentions'):
                            attention = self._extract_attention(logits_output.attentions)
                        elif hasattr(logits_output, 'attention_weights'):
                            attention = self._extract_attention(logits_output.attention_weights)

                        # 保存到缓存（如果启用）
                        if use_cache and attention is not None:
                            # 创建缓存数据
                            cache_data = {
                                "embedding": embeddings.cpu().half(),  # 半精度存储
                                "attention": attention.cpu().half(),
                                "sequence": seq,
                                "id": seq_id
                            }

                            # 生成哈希并保存
                            seq_hash = self._create_seq_hash(seq)
                            embedding_file = os.path.join(self.embedding_cache.cache_dir, f"{seq_hash}.pt")
                            torch.save(cache_data, embedding_file, _use_new_zipfile_serialization=True)

                            # 更新索引
                            self.embedding_cache.embedding_index[seq_id] = {
                                "hash": seq_hash,
                                "sequence": seq,
                                "file": embedding_file
                            }

                        # 保存嵌入和注意力
                        batch_embeddings.append(embeddings)
                        if attention is not None:
                            batch_attentions.append(attention)

                    except Exception as e:
                        logger.warning(f"ESM嵌入计算失败: {e}")
                        # 创建备用嵌入
                        backup_embed = torch.zeros(1, len(token_ids), self.config.ESM_EMBEDDING_DIM, device=self.device)
                        batch_embeddings.append(backup_embed)
                        # 创建备用注意力
                        if batch_attentions:  # 只有当至少有一个成功提取的注意力时才添加备用
                            backup_attn = torch.ones(1, len(token_ids) - 2, 1, device=self.device) / (
                                        len(token_ids) - 2)
                            batch_attentions.append(backup_attn)

            except Exception as outer_e:
                logger.error(f"序列处理完全失败(索引:{seq_idx}): {outer_e}")
                # 生成最小备用嵌入
                backup_embed = torch.zeros(1, 10, self.config.ESM_EMBEDDING_DIM, device=self.device)
                batch_embeddings.append(backup_embed)

        # 返回处理结果 - 不进行长度对齐
        if len(batch_embeddings) == 1:
            result = batch_embeddings[0]
            # 添加注意力属性（如果有）
            if batch_attentions:
                result.attention = batch_attentions[0]
            return result
        else:
            # 注意：此处不进行长度对齐，直接返回嵌入列表
            result_list = batch_embeddings

            # 如果提取了注意力，添加到相应的嵌入中
            if batch_attentions and len(batch_attentions) == len(batch_embeddings):
                for i, emb in enumerate(result_list):
                    emb.attention = batch_attentions[i]

            return result_list

    def _get_fused_embedding(self, graph_batch, sequences=None):
        """优化后的融合嵌入方法，支持缓存机制"""

        # 1. 提取序列ID (用于缓存查找)
        sequence_ids = None
        if hasattr(graph_batch, 'protein_id'):
            if isinstance(graph_batch.protein_id, list):
                sequence_ids = graph_batch.protein_id
            else:
                sequence_ids = [graph_batch.protein_id]

        # 保存当前批次ID（用于其他方法）
        self.current_batch_ids = sequence_ids

        # 2. 提取序列
        if sequences is None:
            sequences = self._extract_sequences_from_graphs(graph_batch)

        # 3. 获取序列ESM嵌入和注意力权重，优先使用缓存
        seq_embeddings = self._get_esm_embedding(sequences, sequence_ids)

        # 4. 提取当前序列的注意力用于图编码
        current_esm_attention = None
        if self.config.ESM_GUIDANCE and hasattr(seq_embeddings, 'attention'):
            current_esm_attention = seq_embeddings.attention

        # 5. 使用当前序列注意力指导图编码
        node_embeddings, graph_embedding, attention_weights = self.graph_encoder(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index,
            edge_attr=graph_batch.edge_attr if hasattr(graph_batch, 'edge_attr') else None,
            edge_type=graph_batch.edge_type if hasattr(graph_batch, 'edge_type') else None,
            batch=graph_batch.batch,
            esm_attention=current_esm_attention  # 使用当前批次的注意力
        )

        # 6. 处理序列嵌入 (池化等)
        if seq_embeddings.dim() == 3:  # [batch_size, seq_len, dim]
            # 使用首尾池化: [CLS] token + [EOS] token + 平均池化
            cls_tokens = seq_embeddings[:, 0, :]  # [CLS] token
            eos_tokens = seq_embeddings[:, -1, :]  # [EOS] token
            avg_tokens = seq_embeddings.mean(dim=1)  # 平均池化

            # 组合表示 (平均聚合)
            pooled_seq_emb = (cls_tokens + eos_tokens + avg_tokens) / 3.0  # [batch_size, dim]
        else:
            # 单个序列处理
            cls_token = seq_embeddings[0, 0, :]
            eos_token = seq_embeddings[0, -1, :]
            avg_token = seq_embeddings[0].mean(dim=0)

            pooled_seq_emb = (cls_token + eos_token + avg_token) / 3.0
            pooled_seq_emb = pooled_seq_emb.unsqueeze(0)  # [1, dim]

        # 7. 映射与融合
        graph_latent = self.latent_mapper(graph_embedding)
        fused_embedding = self.fusion_module(pooled_seq_emb, graph_latent)

        # 8. 如果启用缓存统计，记录命中率
        if hasattr(self, 'embedding_cache') and self.embedding_cache is not None:
            if hasattr(self, 'global_step') and self.global_step % 10 == 0:
                stats = self.embedding_cache.get_stats()
                hit_rate = stats["hit_rate"] * 100
                logger.info(
                    f"嵌入缓存命中率: {hit_rate:.1f}%, 命中/总数: {stats['cache_hits']}/{stats['total_requests']}")

        return graph_embedding, pooled_seq_emb, graph_latent, fused_embedding

    def _extract_sequences_from_graphs(self, graphs_batch):
        """
        增强版：从图批次中提取序列，提高健壮性和准确性

        参数:
            graphs_batch: 图数据批次
        返回:
            list: 氨基酸序列列表
        """
        sequences = []

        # 策略1：直接从sequence属性获取（最优先）
        if hasattr(graphs_batch, "sequence") and isinstance(graphs_batch.sequence, list):
            return [seq if seq and isinstance(seq, str) else 'A' for seq in graphs_batch.sequence]

        # 策略2：从节点特征重建序列
        num_graphs = getattr(graphs_batch, "num_graphs", 1)

        # 定义氨基酸映射表（从索引到字母）
        aa_map = 'ACDEFGHIKLMNPQRSTVWY'

        for i in range(num_graphs):
            # 获取当前图的节点掩码
            if hasattr(graphs_batch, "batch") and graphs_batch.batch is not None:
                mask = graphs_batch.batch == i
                # 提取节点特征
                x_i = graphs_batch.x[mask]
            else:
                # 如果没有batch属性，假设只有一个图
                x_i = graphs_batch.x

            # 尝试多种方法从特征中提取氨基酸序列
            sequence = None
            try:
                # 方法1：假设前20维是氨基酸的one-hot编码
                if x_i.shape[1] >= 20:
                    aa_indices = torch.argmax(x_i[:, :20], dim=1).cpu().numpy()
                    sequence = ''.join([aa_map[idx] if idx < len(aa_map) else 'X' for idx in aa_indices])

                # 方法2：如果方法1失败，使用全部特征的最大值索引
                if not sequence or len(sequence) == 0:
                    aa_indices = torch.argmax(x_i, dim=1).cpu().numpy() % len(aa_map)
                    sequence = ''.join([aa_map[idx] for idx in aa_indices])

                # 验证序列的合理性
                if not sequence or len(sequence) != x_i.shape[0]:
                    # 长度不匹配，使用占位符
                    sequence = 'A' * x_i.shape[0]

            except Exception as e:
                logger.warning(f"序列重建失败: {e}，使用默认序列")
                # 使用占位符
                sequence = 'A' * max(1, x_i.shape[0])  # 至少有一个氨基酸

            sequences.append(sequence)

        return sequences

    def protein_fusion_loss(self, graph_embedding, seq_embedding, graph_latent, fused_embedding, batch=None):
        """
        简化高效的蛋白质多模态融合损失函数

        参数:
            graph_embedding: 原始图嵌入 [batch, dim]
            seq_embedding: 序列嵌入 [batch, dim]
            graph_latent: 图嵌入映射到序列空间后的表示 [batch, dim]
            fused_embedding: 融合后的嵌入 [batch, dim]
            batch: 批次数据，可能包含额外信息

        返回:
            tuple: (总损失, 损失字典)
        """
        batch_size = graph_embedding.size(0)

        # 1. InfoNCE对比损失 - 促进模态对齐
        temperature = 0.1  # 温度参数
        sim_matrix = torch.mm(graph_latent, seq_embedding.t()) / temperature
        labels = torch.arange(batch_size, device=graph_latent.device)

        # 双向对比损失
        loss_g2s = F.cross_entropy(sim_matrix, labels)  # 图→序列方向
        loss_s2g = F.cross_entropy(sim_matrix.t(), labels)  # 序列→图方向
        contrast_loss = (loss_g2s + loss_s2g) / 2.0

        # 2. 融合一致性损失 - 确保融合表示保留原始信息
        fusion_g_sim = F.cosine_similarity(fused_embedding, graph_embedding).mean()
        fusion_s_sim = F.cosine_similarity(fused_embedding, seq_embedding).mean()
        consistency_loss = (2.0 - fusion_g_sim - fusion_s_sim) * 0.5

        # 3. 结构感知的散度损失 - 保持样本间的结构关系
        # 计算嵌入空间中样本对之间的距离关系
        def pairwise_distances(x):
            # 计算批次内所有样本对之间的欧氏距离
            return torch.cdist(x, x, p=2)

        # 获取原始模态的距离矩阵
        g_dist = pairwise_distances(graph_embedding)
        s_dist = pairwise_distances(seq_embedding)

        # 获取融合表示的距离矩阵
        f_dist = pairwise_distances(fused_embedding)

        # 归一化距离矩阵
        g_dist = g_dist / (g_dist.max() + 1e-8)
        s_dist = s_dist / (s_dist.max() + 1e-8)
        f_dist = f_dist / (f_dist.max() + 1e-8)

        # 结构保持损失 - 融合空间应保持原始空间的距离关系
        structure_loss = (F.mse_loss(f_dist, g_dist) + F.mse_loss(f_dist, s_dist)) * 0.5

        # 总损失 - 加权组合
        # 基于经验设置权重，对比损失权重最大
        total_loss = contrast_loss + 0.5 * consistency_loss + 0.3 * structure_loss

        # 创建损失字典用于记录
        loss_dict = {
            'total_loss': total_loss,
            'contrast_loss': contrast_loss,
            'consistency_loss': consistency_loss,
            'structure_loss': structure_loss,
            'similarity': sim_matrix.detach()  # 保存相似度矩阵用于可视化
        }

        return total_loss, loss_dict

    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch，内部使用鲁棒的错误处理

        参数:
            train_loader: 训练数据加载器
            epoch: 当前epoch数

        返回:
            dict: 训练统计信息
        """
        self.graph_encoder.train()
        self.latent_mapper.train()
        self.contrast_head.train()
        self.fusion_module.train()

        epoch_stats = {
            'loss': 0.0,
            'contrast_loss': 0.0,
            'consistency_loss': 0.0,
            'structure_loss': 0.0,
            'graph_latent_norm': 0.0,
            'seq_emb_norm': 0.0,
            'batch_count': 0,
            'skipped_batches': 0  # 记录跳过的批次数
        }

        # 设置进度条
        if self.is_master:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{self.config.EPOCHS}")

        # 训练循环
        for batch_idx, batch in enumerate(train_loader):
            try:
                # 基本数据验证
                if batch is None:
                    logger.warning(f"批次 {batch_idx} 为空，跳过")
                    if self.is_master:
                        pbar.update(1)
                    epoch_stats['skipped_batches'] += 1
                    continue

                # 确保批次包含必要的图数据属性
                required_attrs = ['x', 'edge_index', 'batch']
                missing_attrs = [attr for attr in required_attrs if not hasattr(batch, attr)]
                if missing_attrs:
                    logger.warning(f"批次 {batch_idx} 缺少必要属性: {missing_attrs}，跳过")
                    if self.is_master:
                        pbar.update(1)
                    epoch_stats['skipped_batches'] += 1
                    continue

                # 将数据移到设备上
                batch = batch.to(self.device)

                # 清零梯度
                self.optimizer.zero_grad()

                # 混合精度训练
                with autocast(enabled=self.fp16_training):
                    # 获取图嵌入和序列嵌入
                    graph_embedding, seq_embedding, graph_latent, fused_embedding = self._get_fused_embedding(batch)

                    # 计算融合损失
                    loss, loss_dict = self.protein_fusion_loss(
                        graph_embedding,
                        seq_embedding,
                        graph_latent,
                        fused_embedding,
                        batch
                    )

                # 反向传播
                if self.fp16_training:
                    self.scaler.scale(loss).backward()

                    # 梯度裁剪
                    if self.config.GRADIENT_CLIP > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.graph_encoder.parameters(), self.config.GRADIENT_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.latent_mapper.parameters(), self.config.GRADIENT_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.contrast_head.parameters(), self.config.GRADIENT_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.fusion_module.parameters(), self.config.GRADIENT_CLIP)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()

                    # 梯度裁剪
                    if self.config.GRADIENT_CLIP > 0:
                        torch.nn.utils.clip_grad_norm_(self.graph_encoder.parameters(), self.config.GRADIENT_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.latent_mapper.parameters(), self.config.GRADIENT_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.contrast_head.parameters(), self.config.GRADIENT_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.fusion_module.parameters(), self.config.GRADIENT_CLIP)

                    self.optimizer.step()

                # 累积统计信息
                epoch_stats['loss'] += loss.item()
                epoch_stats['contrast_loss'] += loss_dict['contrast_loss'].item()
                epoch_stats['consistency_loss'] += loss_dict['consistency_loss'].item()
                epoch_stats['structure_loss'] += loss_dict['structure_loss'].item()
                epoch_stats['graph_latent_norm'] += torch.norm(graph_latent.detach(), dim=1).mean().item()
                epoch_stats['seq_emb_norm'] += torch.norm(seq_embedding.detach(), dim=1).mean().item()
                epoch_stats['batch_count'] += 1

                # 更新全局步数
                self.global_step += 1

                # 学习率预热
                if self.warmup_scheduler is not None and self.global_step < self.config.WARMUP_STEPS:
                    self.warmup_scheduler.step()

                # 更新进度条
                if self.is_master:
                    pbar.update(1)
                    if batch_idx % self.config.LOG_INTERVAL == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'c_loss': f"{loss_dict['contrast_loss'].item():.4f}",
                            'cons_loss': f"{loss_dict['consistency_loss'].item():.4f}",
                            'lr': f"{current_lr:.6f}"
                        })

                        # 记录到TensorBoard
                        self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                        self.writer.add_scalar('train/contrast_loss', loss_dict['contrast_loss'].item(),
                                               self.global_step)
                        self.writer.add_scalar('train/consistency_loss', loss_dict['consistency_loss'].item(),
                                               self.global_step)
                        self.writer.add_scalar('train/structure_loss', loss_dict['structure_loss'].item(),
                                               self.global_step)
                        self.writer.add_scalar('train/lr', current_lr, self.global_step)

            except Exception as e:
                # 捕获并记录批次处理过程中的任何错误
                logger.error(f"处理批次 {batch_idx} 时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                epoch_stats['skipped_batches'] += 1

                # 更新进度条
                if self.is_master:
                    pbar.update(1)

                # 如果错误率太高，提前终止epoch
                if epoch_stats['skipped_batches'] > len(train_loader) * 0.3:  # 超过30%批次出错
                    logger.error(f"错误批次过多 ({epoch_stats['skipped_batches']}), 提前结束epoch")
                    break

        # 关闭进度条
        if self.is_master:
            pbar.close()

        # 计算平均统计信息
        if epoch_stats['batch_count'] > 0:
            for key in epoch_stats:
                if key not in ['batch_count', 'skipped_batches']:
                    epoch_stats[key] /= epoch_stats['batch_count']

        return epoch_stats

    def validate(self, val_loader):
        """
        在验证集上评估模型

        参数:
            val_loader: 验证数据加载器

        返回:
            dict: 验证统计信息
        """
        self.graph_encoder.eval()
        self.latent_mapper.eval()
        self.contrast_head.eval()
        self.fusion_module.eval()

        val_stats = {
            'loss': 0.0,
            'contrast_loss': 0.0,
            'consistency_loss': 0.0,
            'structure_loss': 0.0,
            'batch_count': 0,
            'skipped_batches': 0
        }

        # 验证循环
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="验证中", disable=not self.is_master)):
                try:
                    # 验证批次数据
                    required_attrs = ['x', 'edge_index', 'batch']
                    missing_attrs = [attr for attr in required_attrs if not hasattr(batch, attr)]
                    if missing_attrs:
                        logger.warning(f"验证批次 {batch_idx} 缺少必要属性: {missing_attrs}，跳过")
                        val_stats['skipped_batches'] += 1
                        continue

                    # 将数据移到设备上
                    batch = batch.to(self.device)

                    # 获取图嵌入和序列嵌入
                    graph_embedding, seq_embedding, graph_latent, fused_embedding = self._get_fused_embedding(batch)

                    # 计算融合损失
                    loss, loss_dict = self.protein_fusion_loss(
                        graph_embedding,
                        seq_embedding,
                        graph_latent,
                        fused_embedding
                    )

                    # 累积统计信息
                    val_stats['loss'] += loss.item()
                    val_stats['contrast_loss'] += loss_dict['contrast_loss'].item()
                    val_stats['consistency_loss'] += loss_dict['consistency_loss'].item()
                    val_stats['structure_loss'] += loss_dict['structure_loss'].item()
                    val_stats['batch_count'] += 1

                except Exception as e:
                    logger.error(f"验证批次 {batch_idx} 处理出错: {e}")
                    val_stats['skipped_batches'] += 1
                    continue

        # 计算平均统计信息
        if val_stats['batch_count'] > 0:
            for key in val_stats:
                if key not in ['batch_count', 'skipped_batches']:
                    val_stats[key] /= val_stats['batch_count']

        # 记录验证结果
        logger.info(f"验证完成: {val_stats['batch_count']}个批次, 跳过{val_stats['skipped_batches']}个批次")
        logger.info(f"验证损失: {val_stats['loss']:.4f}, 对比损失: {val_stats['contrast_loss']:.4f}")

        return val_stats

    def test(self, test_loader):
        """
        在测试集上评估模型

        参数:
            test_loader: 测试数据加载器

        返回:
            dict: 测试统计信息
        """
        # 与验证过程类似
        self.graph_encoder.eval()
        self.latent_mapper.eval()
        self.contrast_head.eval()
        self.fusion_module.eval()

        test_stats = {
            'loss': 0.0,
            'contrast_loss': 0.0,
            'consistency_loss': 0.0,
            'structure_loss': 0.0,
            'batch_count': 0,
            'skipped_batches': 0,
            'embeddings': [],
            'sequences': []
        }

        # 测试循环
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="测试中", disable=not self.is_master)):
                try:
                    # 验证批次数据
                    required_attrs = ['x', 'edge_index', 'batch']
                    missing_attrs = [attr for attr in required_attrs if not hasattr(batch, attr)]
                    if missing_attrs:
                        logger.warning(f"测试批次 {batch_idx} 缺少必要属性: {missing_attrs}，跳过")
                        test_stats['skipped_batches'] += 1
                        continue

                    # 将数据移到设备上
                    batch = batch.to(self.device)

                    # 提取序列
                    sequences = self._extract_sequences_from_graphs(batch)

                    # 获取图嵌入和序列嵌入
                    graph_embedding, seq_embedding, graph_latent, fused_embedding = self._get_fused_embedding(batch,
                                                                                                              sequences)

                    # 计算融合损失
                    loss, loss_dict = self.protein_fusion_loss(
                        graph_embedding,
                        seq_embedding,
                        graph_latent,
                        fused_embedding
                    )

                    # 保存嵌入和序列（仅在主进程上）
                    if self.is_master:
                        test_stats['embeddings'].append({
                            'graph': graph_embedding.cpu(),
                            'seq': seq_embedding.cpu(),
                            'graph_latent': graph_latent.cpu(),
                            'fused': fused_embedding.cpu()
                        })
                        test_stats['sequences'].extend(sequences)

                    # 累积统计信息
                    test_stats['loss'] += loss.item()
                    test_stats['contrast_loss'] += loss_dict['contrast_loss'].item()
                    test_stats['consistency_loss'] += loss_dict['consistency_loss'].item()
                    test_stats['structure_loss'] += loss_dict['structure_loss'].item()
                    test_stats['batch_count'] += 1

                except Exception as e:
                    logger.error(f"测试批次 {batch_idx} 处理出错: {e}")
                    test_stats['skipped_batches'] += 1
                    continue

        # 计算平均统计信息
        if test_stats['batch_count'] > 0:
            for key in test_stats:
                if key not in ['batch_count', 'skipped_batches', 'embeddings', 'sequences']:
                    test_stats[key] /= test_stats['batch_count']

        # 可视化测试结果（仅在主进程上）
        if self.is_master and test_stats['embeddings']:
            self._visualize_test_results(test_stats)

        return test_stats

    def _plot_similarity_matrix(self, similarity_matrix):
        """
        绘制相似度矩阵热图

        参数:
            similarity_matrix: 相似度矩阵

        返回:
            matplotlib.figure.Figure: 热图图形
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            similarity_matrix,
            annot=False,
            cmap='viridis',
            vmin=-1,
            vmax=1,
            ax=ax
        )
        ax.set_title("Graph-Sequence Similarity Matrix")
        ax.set_xlabel("Sequence Embeddings")
        ax.set_ylabel("Graph Latent Embeddings")

        return fig

    def _visualize_embeddings(self, embeddings, method='tsne', n_components=2):
        """
        可视化嵌入向量

        参数:
            embeddings (dict): 包含不同类型嵌入的字典
            method (str): 降维方法，'tsne'或'pca'
            n_components (int): 降维后的维度

        返回:
            tuple: (图表, 嵌入2D投影)
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        # 设置字体为黑体，支持显示中文
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 选择降维方法
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            reducer = PCA(n_components=n_components)

        # 创建大图表
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs = axs.flatten()

        # 存储投影结果
        projections = {}

        # 为每种嵌入类型创建子图
        for i, (name, data) in enumerate([
            ('图结构嵌入', embeddings['graph']),
            ('序列嵌入', embeddings['seq']),
            ('图潜空间嵌入', embeddings['graph_latent']),
            ('融合嵌入', embeddings['fused'])
        ]):
            # 降维
            embedding_2d = reducer.fit_transform(data)
            projections[name] = embedding_2d

            # 绘制散点图
            scatter = axs[i].scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=range(len(data)),  # 使用索引作为颜色
                cmap='viridis',
                alpha=0.7,
                s=50
            )

            axs[i].set_title(f"{name}投影 ({method.upper()})")
            axs[i].set_xlabel("分量1")
            axs[i].set_ylabel("分量2")
            axs[i].grid(True, linestyle='--', alpha=0.7)

            # 添加颜色条
            plt.colorbar(scatter, ax=axs[i], label='样本索引')

        plt.tight_layout()

        return fig, projections

    def _visualize_test_results(self, test_stats):
        """
        可视化测试结果并保存图表

        参数:
            test_stats (dict): 测试统计信息
        """
        # 如果没有嵌入数据，返回
        if not test_stats['embeddings']:
            logger.warning("无嵌入数据可供可视化")
            return

        try:
            # 合并所有批次的嵌入
            all_embeddings = {
                'graph': torch.cat([e['graph'] for e in test_stats['embeddings']]),
                'seq': torch.cat([e['seq'] for e in test_stats['embeddings']]),
                'graph_latent': torch.cat([e['graph_latent'] for e in test_stats['embeddings']]),
                'fused': torch.cat([e['fused'] for e in test_stats['embeddings']])
            }

            # 转换为NumPy数组
            for key in all_embeddings:
                all_embeddings[key] = all_embeddings[key].numpy()

            # 限制可视化样本数，避免过多消耗资源
            max_viz_samples = 1000
            if all_embeddings['graph'].shape[0] > max_viz_samples:
                indices = np.random.choice(all_embeddings['graph'].shape[0], max_viz_samples, replace=False)
                for key in all_embeddings:
                    all_embeddings[key] = all_embeddings[key][indices]
                test_stats['sequences'] = [test_stats['sequences'][i] for i in indices]

                # TSNE可视化
                logger.info("生成t-SNE可视化...")
                tsne_fig, tsne_proj = self._visualize_embeddings(all_embeddings, method='tsne')
                self.writer.add_figure('test/tsne_visualization', tsne_fig, self.global_step)

                # PCA可视化
                logger.info("生成PCA可视化...")
                pca_fig, pca_proj = self._visualize_embeddings(all_embeddings, method='pca')
                self.writer.add_figure('test/pca_visualization', pca_fig, self.global_step)

                # 计算相似度矩阵
                logger.info("计算模态间相似度矩阵...")
                # 计算图-序列相似度
                graph_seq_sim = np.corrcoef(all_embeddings['graph'], all_embeddings['seq'])
                # 计算潜空间-序列相似度
                latent_sim = np.corrcoef(all_embeddings['graph_latent'], all_embeddings['seq'])

                # 绘制相似度矩阵
                sim_fig, axs = plt.subplots(1, 2, figsize=(18, 8))

                # 只显示交叉相关部分
                graph_seq_part = graph_seq_sim[:len(all_embeddings['graph']), len(all_embeddings['graph']):]
                latent_seq_part = latent_sim[:len(all_embeddings['graph_latent']), len(all_embeddings['graph_latent']):]

                sns.heatmap(graph_seq_part, cmap='coolwarm', ax=axs[0], vmin=-1, vmax=1)
                axs[0].set_title("图结构-序列相似度矩阵")

                sns.heatmap(latent_seq_part, cmap='coolwarm', ax=axs[1], vmin=-1, vmax=1)
                axs[1].set_title("潜空间-序列相似度矩阵")

                self.writer.add_figure('test/similarity_matrices', sim_fig, self.global_step)

                # 保存测试结果
                logger.info("保存测试结果和嵌入表示...")
                embeddings_dir = os.path.join(self.config.RESULT_DIR, f"embeddings_epoch_{self.current_epoch}")
                os.makedirs(embeddings_dir, exist_ok=True)

                np.savez(
                    os.path.join(embeddings_dir, "embeddings.npz"),
                    graph=all_embeddings['graph'],
                    seq=all_embeddings['seq'],
                    graph_latent=all_embeddings['graph_latent'],
                    fused=all_embeddings['fused'],
                    sequences=np.array(test_stats['sequences'], dtype=object)
                )

                logger.info(f"嵌入和可视化结果保存到 {embeddings_dir}")

        except Exception as e:
            logger.error(f"可视化过程出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def save_checkpoint(self, metrics, is_best=False):
        """
        保存检查点

        参数:
            metrics (dict): 训练/验证指标
            is_best (bool): 是否为最佳模型
        """
        # 仅在主进程上保存
        if not self.is_master:
            return

        # 创建检查点
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_validation_score': self.best_validation_score,  # 更新为组合验证分数
            'patience_counter': self.patience_counter,
            'metrics': metrics,
            'graph_encoder': self.graph_encoder.state_dict() if not self.config.USE_DISTRIBUTED else self.graph_encoder.module.state_dict(),
            'latent_mapper': self.latent_mapper.state_dict() if not self.config.USE_DISTRIBUTED else self.latent_mapper.module.state_dict(),
            'contrast_head': self.contrast_head.state_dict() if not self.config.USE_DISTRIBUTED else self.contrast_head.module.state_dict(),
            'fusion_module': self.fusion_module.state_dict() if not self.config.USE_DISTRIBUTED else self.fusion_module.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'config': {k: v for k, v in vars(self.config).items() if not k.startswith('_')}  # 避免保存不可序列化的对象
        }

        # 保存检查点
        checkpoint_path = os.path.join(self.config.MODEL_DIR, f"checkpoint_epoch_{self.current_epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"保存检查点到 {checkpoint_path}")

        # 保存最新检查点（覆盖）
        latest_path = os.path.join(self.config.MODEL_DIR, "checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)

        # 保存最佳模型（如果是最佳）
        if is_best:
            best_path = os.path.join(self.config.MODEL_DIR, "checkpoint_best.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型到 {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点

        参数:
            checkpoint_path (str): 检查点路径

        返回:
            dict: 加载的指标
        """
        logger.info(f"从 {checkpoint_path} 加载检查点")

        try:
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 恢复模型状态
            if self.config.USE_DISTRIBUTED:
                self.graph_encoder.module.load_state_dict(checkpoint['graph_encoder'])
                self.latent_mapper.module.load_state_dict(checkpoint['latent_mapper'])
                self.contrast_head.module.load_state_dict(checkpoint['contrast_head'])
                self.fusion_module.module.load_state_dict(checkpoint['fusion_module'])
            else:
                self.graph_encoder.load_state_dict(checkpoint['graph_encoder'])
                self.latent_mapper.load_state_dict(checkpoint['latent_mapper'])
                self.contrast_head.load_state_dict(checkpoint['contrast_head'])
                self.fusion_module.load_state_dict(checkpoint['fusion_module'])

            # 恢复优化器和调度器状态
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler and checkpoint.get('scheduler'):
                self.scheduler.load_state_dict(checkpoint['scheduler'])

            # 恢复训练状态
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']

            # 恢复新的验证分数
            if 'best_validation_score' in checkpoint:
                self.best_validation_score = checkpoint['best_validation_score']
            else:
                # 向后兼容：如果没有组合验证分数，则使用旧的验证损失
                self.best_validation_score = checkpoint.get('best_val_loss', float('inf'))

            self.patience_counter = checkpoint.get('patience_counter', 0)

            logger.info(f"成功加载检查点，恢复于 epoch {self.current_epoch}")

            return checkpoint.get('metrics', {})

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def export_model(self):
        """导出训练好的模型用于推理"""
        if not self.is_master:
            return

        try:
            # 创建导出目录
            export_dir = os.path.join(self.config.MODEL_DIR, "export")
            os.makedirs(export_dir, exist_ok=True)

            # 导出各个组件
            torch.save(
                self.graph_encoder.state_dict() if not self.config.USE_DISTRIBUTED else self.graph_encoder.module.state_dict(),
                os.path.join(export_dir, "graph_encoder.pth")
            )

            torch.save(
                self.latent_mapper.state_dict() if not self.config.USE_DISTRIBUTED else self.latent_mapper.module.state_dict(),
                os.path.join(export_dir, "latent_mapper.pth")
            )

            torch.save(
                self.fusion_module.state_dict() if not self.config.USE_DISTRIBUTED else self.fusion_module.module.state_dict(),
                os.path.join(export_dir, "fusion_module.pth")
            )

            # 导出配置
            import json
            with open(os.path.join(export_dir, "config.json"), "w") as f:
                # 过滤掉无法序列化的对象
                config_dict = {k: v for k, v in vars(self.config).items()
                               if not k.startswith('_') and isinstance(v, (
                    int, float, str, bool, list, dict, type(None)))}
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # 导出模型架构描述
            with open(os.path.join(export_dir, "model_description.txt"), "w") as f:
                f.write("蛋白质图-序列双模态融合表示模型\n")
                f.write("==========================\n\n")
                f.write("1. 图编码器架构:\n")
                f.write(f"   - 输入维度: {self.config.NODE_INPUT_DIM}\n")
                f.write(f"   - 隐藏维度: {self.config.HIDDEN_DIM}\n")
                f.write(f"   - 输出维度: {self.config.OUTPUT_DIM}\n")
                f.write(f"   - 层数: {self.config.NUM_LAYERS}\n")
                f.write(f"   - 注意力头数: {self.config.NUM_HEADS}\n\n")

                f.write("2. 序列编码器:\n")
                f.write(f"   - 模型名称: {self.config.ESM_MODEL_NAME}\n")
                f.write(f"   - 嵌入维度: {self.config.ESM_EMBEDDING_DIM}\n\n")

                f.write("3. 融合模块:\n")
                f.write(f"   - 输出维度: {self.config.FUSION_OUTPUT_DIM}\n")
                f.write(f"   - 隐藏维度: {self.config.FUSION_HIDDEN_DIM}\n")
                f.write(f"   - 注意力头数: {self.config.FUSION_NUM_HEADS}\n")
                f.write(f"   - 层数: {self.config.FUSION_NUM_LAYERS}\n\n")

                f.write("导出日期: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")

            logger.info(f"模型导出到 {export_dir}")

        except Exception as e:
            logger.error(f"导出模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def train(self, train_loader, val_loader, test_loader=None, resume_from=None):
        """
        训练主循环

        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            resume_from: 恢复训练的检查点路径
        """
        # 如果需要恢复训练
        if resume_from:
            metrics = self.load_checkpoint(resume_from)
            logger.info(f"从检查点恢复训练，当前epoch: {self.current_epoch}")

        # 初始化最佳验证指标（如果还未设置）
        if not hasattr(self, 'best_validation_score'):
            self.best_validation_score = float('inf')

        # 训练循环
        for epoch in range(self.current_epoch, self.config.EPOCHS):
            self.current_epoch = epoch

            # 设置分布式采样器的epoch（如果适用）
            if self.config.USE_DISTRIBUTED:
                if hasattr(train_loader, 'sampler') and hasattr(train_loader, 'sampler') and hasattr(
                        train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)

            # 训练一个epoch
            train_start_time = time.time()
            train_stats = self.train_epoch(train_loader, epoch)
            train_time = time.time() - train_start_time

            if self.is_master:
                logger.info(f"Epoch {epoch + 1}/{self.config.EPOCHS} - "
                            f"训练损失: {train_stats['loss']:.4f}, "
                            f"对比损失: {train_stats['contrast_loss']:.4f}, "
                            f"一致性损失: {train_stats['consistency_loss']:.4f}, "
                            f"结构损失: {train_stats['structure_loss']:.4f}, "
                            f"用时: {train_time:.2f}秒")

                # 记录训练指标到TensorBoard
                self.writer.add_scalar('epoch/train_loss', train_stats['loss'], epoch)
                self.writer.add_scalar('epoch/train_contrast_loss', train_stats['contrast_loss'], epoch)
                self.writer.add_scalar('epoch/train_consistency_loss', train_stats['consistency_loss'], epoch)
                self.writer.add_scalar('epoch/train_structure_loss', train_stats['structure_loss'], epoch)
                self.writer.add_scalar('epoch/graph_latent_norm', train_stats['graph_latent_norm'], epoch)
                self.writer.add_scalar('epoch/seq_emb_norm', train_stats['seq_emb_norm'], epoch)
                self.writer.add_scalar('epoch/skipped_batches', train_stats['skipped_batches'], epoch)

            # 验证
            if epoch % self.config.EVAL_INTERVAL == 0:
                val_start_time = time.time()
                val_stats = self.validate(val_loader)
                val_time = time.time() - val_start_time

                if self.is_master:
                    logger.info(f"验证 - 损失: {val_stats['loss']:.4f}, "
                                f"对比损失: {val_stats['contrast_loss']:.4f}, "
                                f"一致性损失: {val_stats['consistency_loss']:.4f}, "
                                f"结构损失: {val_stats['structure_loss']:.4f}, "
                                f"用时: {val_time:.2f}秒")

                    # 记录验证指标到TensorBoard
                    self.writer.add_scalar('epoch/val_loss', val_stats['loss'], epoch)
                    self.writer.add_scalar('epoch/val_contrast_loss', val_stats['contrast_loss'], epoch)
                    self.writer.add_scalar('epoch/val_consistency_loss', val_stats['consistency_loss'], epoch)
                    self.writer.add_scalar('epoch/val_structure_loss', val_stats['structure_loss'], epoch)
                    self.writer.add_scalar('epoch/val_skipped_batches', val_stats['skipped_batches'], epoch)

                # 更新早停策略：使用加权组合的验证指标
                # 对比损失权重最高，一致性和结构损失为辅助损失
                validation_score = val_stats['loss'] * 0.6 - val_stats['consistency_loss'] * 0.2 - val_stats[
                    'structure_loss'] * 0.2
                is_best = validation_score < self.best_validation_score

                if is_best:
                    self.best_validation_score = validation_score
                    self.patience_counter = 0
                    logger.info(f"发现新的最佳模型 (分数: {validation_score:.4f})")
                else:
                    self.patience_counter += 1
                    logger.info(f"未改善模型 ({self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE})")

                # 保存检查点
                if epoch % self.config.SAVE_INTERVAL == 0 or is_best:
                    metrics = {**train_stats, **{f'val_{k}': v for k, v in val_stats.items()}}
                    self.save_checkpoint(metrics, is_best)

                # 早停检查
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    if self.is_master:
                        logger.info(f"早停触发 - {self.patience_counter}个epoch未改善")
                    break

            # 学习率调度器
            if self.scheduler:
                self.scheduler.step()

        # 训练结束，在测试集上评估
        if test_loader and self.is_master:
            logger.info("在测试集上评估模型...")
            # 加载最佳模型
            best_ckpt_path = os.path.join(self.config.MODEL_DIR, "checkpoint_best.pth")
            if os.path.exists(best_ckpt_path):
                self.load_checkpoint(best_ckpt_path)
                logger.info("加载最佳模型进行测试")
            else:
                logger.warning("找不到最佳模型检查点，使用当前模型进行测试")

            test_stats = self.test(test_loader)
            logger.info(f"测试 - 损失: {test_stats['loss']:.4f}, "
                        f"对比损失: {test_stats['contrast_loss']:.4f}, "
                        f"一致性损失: {test_stats['consistency_loss']:.4f}, "
                        f"结构损失: {test_stats['structure_loss']:.4f}")

        # 导出模型
        self.export_model()

        if self.is_master:
            logger.info("训练完成！")
            self.writer.close()


def setup_data_loaders(config):
    """
    设置数据加载器

    参数:
        config: 配置对象

    返回:
        tuple: (训练加载器, 验证加载器, 测试加载器)
    """
    # 确保批处理大小有效 - 使用局部变量而不是修改配置
    config_batch_size = getattr(config, "BATCH_SIZE", 256)
    effective_batch_size = 128  # 默认值

    if config_batch_size is not None and config_batch_size > 0:
        effective_batch_size = config_batch_size
    else:
        logger.warning(f"无效的批处理大小: {config_batch_size}，使用默认值 {effective_batch_size}")

    # 处理评估批处理大小
    config_eval_batch_size = getattr(config, "EVAL_BATCH_SIZE", effective_batch_size)
    effective_eval_batch_size = effective_batch_size

    if config_eval_batch_size is not None and config_eval_batch_size > 0:
        effective_eval_batch_size = config_eval_batch_size
    else:
        logger.warning(f"无效的评估批处理大小: {config_eval_batch_size}，使用默认值 {effective_eval_batch_size}")

    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = ProteinDataset(
        data_path=config.TRAIN_CACHE,
        subset_ratio=config.SUBSET_RATIO if getattr(config, "USE_SUBSET", False) else None,
        seed=config.SEED
    )

    val_dataset = ProteinDataset(
        data_path=config.VAL_CACHE,
        subset_ratio=config.SUBSET_RATIO if getattr(config, "USE_SUBSET", False) else None,
        seed=config.SEED
    )

    test_dataset = ProteinDataset(
        data_path=config.TEST_CACHE,
        subset_ratio=config.SUBSET_RATIO if getattr(config, "USE_SUBSET", False) else None,
        seed=config.SEED
    )

    logger.info(f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")

    # 检查数据集是否有效
    if len(train_dataset) == 0:
        logger.error("训练数据集为空！请检查数据路径和加载流程")

    # 设置分布式采样器（如果启用）
    if getattr(config, "USE_DISTRIBUTED", False):
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=getattr(config, "WORLD_SIZE", 1),
            rank=getattr(config, "GLOBAL_RANK", 0)
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=getattr(config, "WORLD_SIZE", 1),
            rank=getattr(config, "GLOBAL_RANK", 0)
        )

        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=getattr(config, "WORLD_SIZE", 1),
            rank=getattr(config, "GLOBAL_RANK", 0)
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    # 创建数据加载器，使用局部变量而不是config.BATCH_SIZE
    try:
        train_loader = ProteinDataLoader(
            train_dataset,
            batch_size=effective_batch_size,  # 使用局部变量
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=getattr(config, "NUM_WORKERS", 0)
        )

        val_loader = ProteinDataLoader(
            val_dataset,
            batch_size=effective_eval_batch_size,  # 使用局部变量
            shuffle=False,
            sampler=val_sampler,
            num_workers=getattr(config, "NUM_WORKERS", 0)
        )

        test_loader = ProteinDataLoader(
            test_dataset,
            batch_size=effective_eval_batch_size,  # 使用局部变量
            shuffle=False,
            sampler=test_sampler,
            num_workers=getattr(config, "NUM_WORKERS", 0)
        )

        logger.info(
            f"数据加载器创建成功 - 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")
        logger.info(f"使用的批处理大小 - 训练: {effective_batch_size}, 评估: {effective_eval_batch_size}")

        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"创建数据加载器失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # 返回一个有默认值的加载器，以避免程序崩溃
        logger.warning("使用空数据集创建备用数据加载器")

        # 创建空数据集
        empty_dataset = ProteinDataset(config.TRAIN_CACHE)
        empty_loader = ProteinDataLoader(empty_dataset, batch_size=effective_batch_size, shuffle=False)

        return empty_loader, empty_loader, empty_loader

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="蛋白质图-序列多模态融合训练")
    parser.add_argument("--config", type=str, default="utils/config.py", help="配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地进程排名")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的检查点路径")
    parser.add_argument("--test_only", action="store_true", help="仅进行测试，不进行训练")
    parser.add_argument("--clean_cache", action="store_true", help="清除处理缓存目录")
    args = parser.parse_args()

    # 加载配置
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config = config_module.Config()

    # 如果清除缓存
    if args.clean_cache:
        import shutil
        logger.info("清除处理缓存目录...")
        if os.path.exists(config.CACHE_DIR):
            shutil.rmtree(config.CACHE_DIR)
            logger.info(f"已删除缓存目录: {config.CACHE_DIR}")
        # 不要删除已处理的数据

    # 设置本地进程排名
    if args.local_rank != -1:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
        config.LOCAL_RANK = args.local_rank

    # 设置分布式环境
    if config.USE_DISTRIBUTED:
        config.setup_distributed()

    is_master = not config.USE_DISTRIBUTED or config.GLOBAL_RANK == 0

    # 设置日志级别（仅主进程显示INFO及以上级别）
    if is_master:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 显示配置信息
    if is_master:
        logger.info(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        logger.info(f"使用设备: {config.DEVICE}")
        logger.info(f"分布式训练: {config.USE_DISTRIBUTED}")
        if config.USE_DISTRIBUTED:
            logger.info(f"世界大小: {config.WORLD_SIZE}")
            logger.info(f"全局进程排名: {config.GLOBAL_RANK}")
            logger.info(f"本地进程排名: {config.LOCAL_RANK}")
        logger.info(f"混合精度训练: {config.FP16_TRAINING}")
        if config._batch_size is None:
            if config.USE_DISTRIBUTED and config.NUM_GPUS > 0:
                config._batch_size = max(1, config.GLOBAL_BATCH_SIZE // config.NUM_GPUS)
            else:
                config._batch_size = config.GLOBAL_BATCH_SIZE
        logger.info(f"批大小: {config.BATCH_SIZE or config.GLOBAL_BATCH_SIZE}")
        logger.info(f"GPU数量: {config.NUM_GPUS}")

    # 设置种子确保可重复性
    seed = config.SEED + config.GLOBAL_RANK if config.USE_DISTRIBUTED else config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # 创建必要的目录
    if is_master:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.RESULT_DIR, exist_ok=True)

    # 创建数据加载器
    train_loader, val_loader, test_loader = setup_data_loaders(config)

    # 创建训练器
    trainer = ProteinMultiModalTrainer(config)

    # 测试或训练
    if args.test_only:
        # 加载最佳模型
        if args.resume:
            trainer.load_checkpoint(args.resume)
        else:
            best_ckpt_path = os.path.join(config.MODEL_DIR, "checkpoint_best.pth")
            if os.path.exists(best_ckpt_path):
                trainer.load_checkpoint(best_ckpt_path)
            else:
                logger.error("测试模式需要提供检查点路径或已有最佳模型")
                return

        # 测试
        test_stats = trainer.test(test_loader)
        if is_master:
            logger.info(f"测试 - 损失: {test_stats['loss']:.4f}, "
                        f"对比损失: {test_stats['contrast_loss']:.4f}, "
                        f"一致性损失: {test_stats['consistency_loss']:.4f}, "
                        f"结构损失: {test_stats['structure_loss']:.4f}")
    else:
        # 训练
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            resume_from=args.resume
        )

    # 清理分布式进程组
    if config.USE_DISTRIBUTED and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # 在分布式环境中，确保所有进程都退出
        if dist.is_initialized():
            dist.destroy_process_group()

        raise