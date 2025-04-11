#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图-序列双模态融合表示训练脚本

该模块实现了蛋白质图嵌入与序列嵌入的融合训练流程,
基于对比学习优化表示空间的一致性，为后续扩散模型提供高质量嵌入基础。

作者: wxhfy
日期: 2025-03-29
"""
import hashlib
import os
import pickle
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

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from models.gat_models import (
    ProteinGATv2Encoder,
    ProteinLatentMapper,
    CrossModalContrastiveHead,

)
from models.layers import CrossAttentionFusion
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


class ProteinBatch:
    """处理批次中的多个ProteinData对象"""

    def __init__(self, data_list):
        self.num_graphs = len(data_list)

        # 按照正确顺序处理节点和边
        node_attributes = self._merge_node_attributes(data_list)
        edge_attributes = self._merge_edge_attributes(data_list, node_attributes['ptr'])

        # 将所有属性设置为类属性
        for k, v in {**node_attributes, **edge_attributes}.items():
            setattr(self, k, v)

    def _merge_node_attributes(self, data_list):
        """先处理所有节点属性，确保一致性"""
        result = {
            'x': [],
            'batch': [],
            'ptr': [0]
        }

        node_offset = 0

        for i, data in enumerate(data_list):
            if not hasattr(data, 'x') or data.x is None:
                continue

            num_nodes = data.x.size(0)

            # 添加节点特征
            result['x'].append(data.x)

            # 添加批次索引
            result['batch'].append(torch.full((num_nodes,), i, dtype=torch.long))

            # 更新节点偏移
            node_offset += num_nodes
            result['ptr'].append(node_offset)

        # 合并张量
        if result['x']:
            result['x'] = torch.cat(result['x'], dim=0)
            result['batch'] = torch.cat(result['batch'], dim=0)
        else:
            # 空批次处理
            result['x'] = torch.zeros((0, 35))  # 假设特征是35维
            result['batch'] = torch.zeros(0, dtype=torch.long)

        result['ptr'] = torch.tensor(result['ptr'], dtype=torch.long)

        return result

    def _merge_edge_attributes(self, data_list, ptr):
        """处理边属性，使用正确的节点偏移"""
        result = {
            'edge_index': [],
            'edge_attr': [],
            'edge_type': []
        }

        has_edge_attr = all(hasattr(d, 'edge_attr') and d.edge_attr is not None
                            for d in data_list if hasattr(d, 'edge_index'))

        has_edge_type = all(hasattr(d, 'edge_type') and d.edge_type is not None
                            for d in data_list if hasattr(d, 'edge_index'))

        # 处理每个图的边
        for i, data in enumerate(data_list):
            if not hasattr(data, 'edge_index') or data.edge_index is None:
                continue

            # 获取节点偏移
            node_offset = ptr[i].item()

            # 本地边验证和过滤
            num_nodes_i = ptr[i + 1] - ptr[i]
            edge_index = data.edge_index.clone()
            valid_mask = (edge_index[0] < num_nodes_i) & (edge_index[1] < num_nodes_i) & \
                         (edge_index[0] >= 0) & (edge_index[1] >= 0)

            if not valid_mask.all():
                # 过滤无效边
                edge_index = edge_index[:, valid_mask]
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    edge_attr_i = data.edge_attr[valid_mask]
                else:
                    edge_attr_i = None

                if hasattr(data, 'edge_type') and data.edge_type is not None:
                    edge_type_i = data.edge_type[valid_mask]
                else:
                    edge_type_i = None
            else:
                edge_attr_i = data.edge_attr if hasattr(data, 'edge_attr') else None
                edge_type_i = data.edge_type if hasattr(data, 'edge_type') else None

            # 应用节点偏移
            edge_index[0] += node_offset
            edge_index[1] += node_offset

            # 添加到结果
            result['edge_index'].append(edge_index)

            if edge_attr_i is not None:
                result['edge_attr'].append(edge_attr_i)

            if edge_type_i is not None:
                result['edge_type'].append(edge_type_i)

        # 合并边张量
        if result['edge_index']:
            result['edge_index'] = torch.cat(result['edge_index'], dim=1)

            if result['edge_attr']:
                result['edge_attr'] = torch.cat(result['edge_attr'], dim=0)
            else:
                del result['edge_attr']

            if result['edge_type']:
                result['edge_type'] = torch.cat(result['edge_type'], dim=0)
            else:
                del result['edge_type']
        else:
            del result['edge_index']
            del result['edge_attr']
            del result['edge_type']

        return result

    def to(self, device):
        """将所有张量移至指定设备"""
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
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
        self.use_esm_guidance = None
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
        self.fusion_module = CrossAttentionFusion(
            embedding_dim=self.config.ESM_EMBEDDING_DIM,  # 1152
            num_heads=self.config.FUSION_NUM_HEADS,  # 建议使用8或16
            num_layers=self.config.FUSION_NUM_LAYERS,  # 建议使用2或3
            dropout=self.config.FUSION_DROPOUT
        ).to(self.device)

        # 新增: 预定义ESM注意力维度适配器，避免动态创建
        self.esm_attention_adapter = nn.Linear(1, self.config.HIDDEN_DIM).to(self.device)

        # 分布式训练设置
        if self.config.USE_DISTRIBUTED:
            # 同步BN层
            if self.config.SYNC_BN and self.config.NUM_GPUS > 1:
                self.graph_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.graph_encoder)
                self.latent_mapper = nn.SyncBatchNorm.convert_sync_batchnorm(self.latent_mapper)
                self.fusion_module = nn.SyncBatchNorm.convert_sync_batchnorm(self.fusion_module)

            # 为融合模块添加前向钩子，确保所有参数都参与计算
            def ensure_all_params_used(module, input,output):
                # 获取所有参数并执行一个不影响结果的操作
                params_sum = 0
                for param in module.parameters():
                    if param.requires_grad:
                        params_sum = params_sum + param.mean() * 0  # 零乘不影响结果

                # 如果是张量，添加参数项
                if isinstance(output, torch.Tensor):
                    return output + params_sum
                else:
                    return output

            # 获取实际模块（穿透DDP封装）
            actual_fusion_module = self.fusion_module

            # 注册钩子
            actual_fusion_module.register_forward_hook(ensure_all_params_used)
            # 封装为DDP模型
            self.graph_encoder = DDP(
                self.graph_encoder,
                device_ids=[self.config.LOCAL_RANK],
                output_device=self.config.LOCAL_RANK,
            )

            self.latent_mapper = DDP(
                self.latent_mapper,
                device_ids=[self.config.LOCAL_RANK],
                output_device=self.config.LOCAL_RANK,
            )

            self.contrast_head = DDP(
                self.contrast_head,
                device_ids=[self.config.LOCAL_RANK],
                output_device=self.config.LOCAL_RANK,
            )

            self.fusion_module = DDP(
                self.fusion_module,
                device_ids=[self.config.LOCAL_RANK],
                output_device=self.config.LOCAL_RANK,
            )

            # 新增: 将注意力适配器也包装为DDP模型
            self.esm_attention_adapter = DDP(
                self.esm_attention_adapter,
                device_ids=[self.config.LOCAL_RANK],
                output_device=self.config.LOCAL_RANK,
            )

            # 新增: 启用静态图模式，解决参数重用问题
            logger.info("为DDP模型启用静态图模式...")
            for model_name, model in [
                ("graph_encoder", self.graph_encoder),
                ("latent_mapper", self.latent_mapper),
                ("contrast_head", self.contrast_head),
                ("fusion_module", self.fusion_module),
                ("esm_attention_adapter", self.esm_attention_adapter)
            ]:
                if hasattr(model, "_set_static_graph"):
                    model._set_static_graph()
                    logger.info(f"已为{model_name}启用静态图模式")

        # 打印模型信息
        if self.is_master:
            pytorch_total_params = sum(p.numel() for p in self.graph_encoder.parameters() if p.requires_grad)
            pytorch_total_params += sum(p.numel() for p in self.latent_mapper.parameters() if p.requires_grad)
            pytorch_total_params += sum(p.numel() for p in self.contrast_head.parameters() if p.requires_grad)
            pytorch_total_params += sum(p.numel() for p in self.fusion_module.parameters() if p.requires_grad)
            pytorch_total_params += sum(
                p.numel() for p in self.esm_attention_adapter.parameters() if p.requires_grad)  # 新增
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

    def _get_dataset_name(self):
        """从训练配置中获取当前数据集名称，用于定位分布式预计算的缓存目录"""
        # 尝试从配置中获取数据集名称
        if hasattr(self.config, "DATASET_NAME"):
            return self.config.DATASET_NAME

        # 尝试从数据路径中提取数据集名称
        for attr in ["TRAIN_CACHE", "VAL_CACHE", "TEST_CACHE"]:
            if hasattr(self.config, attr):
                path = getattr(self.config, attr)
                if path:
                    # 提取文件名（不含扩展名）
                    filename = os.path.basename(path)
                    dataset_name = os.path.splitext(filename)[0]
                    return dataset_name

        return None

    def _update_cache_stats(self, hits, requests):
        """更新缓存统计信息"""
        # 如果有内存缓存，更新其统计信息
        if hasattr(self, 'embedding_cache') and self.embedding_cache is not None:
            if not hasattr(self.embedding_cache, 'total_requests'):
                self.embedding_cache.total_requests = 0
            if not hasattr(self.embedding_cache, 'cache_hits'):
                self.embedding_cache.cache_hits = 0

            self.embedding_cache.total_requests += requests
            self.embedding_cache.cache_hits += hits

        # 保存全局统计
        if not hasattr(self, '_cache_stats'):
            self._cache_stats = {
                "total_requests": 0,
                "cache_hits": 0
            }

        self._cache_stats["total_requests"] += requests
        self._cache_stats["cache_hits"] += hits

    def _safe_compute_embeddings(self, esm_model, protein_tensor, config):
        """安全计算ESMC嵌入，处理维度不匹配问题"""
        try:
            # 尝试直接计算
            return esm_model.logits(protein_tensor, config)
        except Exception as e:
            error_msg = str(e)

            # 处理维度错误
            if "Wrong shape: expected 3 dims" in error_msg and "Received 4-dim tensor" in error_msg:
                # 修复序列维度
                sequence = protein_tensor.sequence

                if sequence.dim() == 4:
                    # [b, 1, s, d] -> [b, s, d]
                    if sequence.shape[1] == 1:
                        fixed_sequence = sequence.squeeze(1)
                    else:
                        # 对于其他形状，尝试更通用的方法
                        b = sequence.shape[0]
                        s = sequence.shape[2]
                        fixed_sequence = sequence.reshape(b, s, -1)

                    # 创建新的蛋白质张量
                    fixed_tensor = ESMProteinTensor(sequence=fixed_sequence)
                    logger.info(f"修复维度形状: {sequence.shape} -> {fixed_sequence.shape}")

                    # 重试
                    return esm_model.logits(fixed_tensor, config)

                # 处理其他维度问题
                if sequence.dim() == 3:
                    b, s, d = sequence.shape
                    if s == 1:
                        # [b, 1, d] -> [b, d]
                        fixed_sequence = sequence.squeeze(1)
                        fixed_tensor = ESMProteinTensor(sequence=fixed_sequence)
                        logger.info(f"修复维度形状: {sequence.shape} -> {fixed_sequence.shape}")
                        return esm_model.logits(fixed_tensor, config)

            # 无法处理的错误，重新抛出
            raise

    def _get_esm_embedding(self, sequences, sequence_ids=None):
        """
        获取序列的ESM嵌入与注意力，优先从缓存加载

        参数:
            sequences (list): 氨基酸序列列表
            sequence_ids (list, optional): 序列ID列表，用于缓存查找

        返回:
            torch.Tensor: 格式统一的ESM嵌入
        """
        import traceback

        # 准备批次容器
        batch_embeddings = []
        batch_attentions = []
        batch_size = len(sequences)

        # 1. 检查缓存可用性
        use_cache = hasattr(self, 'embedding_cache') and self.embedding_cache is not None
        cache_hit_count = 0

        # 2. 准备序列ID
        if sequence_ids is None:
            if hasattr(self, 'current_batch_ids') and self.current_batch_ids:
                sequence_ids = self.current_batch_ids
            else:
                # 使用序列哈希作为ID
                sequence_ids = [hashlib.md5(seq.encode()).hexdigest() for seq in sequences]

        # 确保序列ID长度匹配
        if len(sequence_ids) != len(sequences):
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]

        # 3. ESM氨基酸编码映射表
        ESM_AA_MAP = {
            'A': 5, 'C': 23, 'D': 13, 'E': 9, 'F': 18,
            'G': 6, 'H': 21, 'I': 12, 'K': 15, 'L': 4,
            'M': 20, 'N': 17, 'P': 14, 'Q': 16, 'R': 10,
            'S': 8, 'T': 11, 'V': 7, 'W': 22, 'Y': 19,
            '_': 32, 'X': 32
        }

        # 4. 逐个序列处理
        for i, (seq, seq_id) in enumerate(zip(sequences, sequence_ids)):
            try:
                # 尝试从缓存获取嵌入
                cached_data = None
                if use_cache:
                    # 生成序列哈希
                    seq_hash = hashlib.md5(seq.encode()).hexdigest()

                    # 先通过ID查找缓存
                    if seq_id in self.embedding_cache.embedding_index:
                        cache_info = self.embedding_cache.embedding_index[seq_id]
                        if os.path.exists(cache_info["file"]):
                            try:
                                cached_data = torch.load(cache_info["file"], map_location='cpu')
                                cache_hit_count += 1
                            except Exception as e:
                                logger.warning(f"从ID加载缓存失败: {e}")

                    # 尝试通过哈希直接查找
                    if cached_data is None:
                        cache_file = os.path.join(self.embedding_cache.cache_dir, f"{seq_hash}.pt")
                        if os.path.exists(cache_file):
                            try:
                                cached_data = torch.load(cache_file, map_location='cpu')
                                cache_hit_count += 1
                                # 更新索引
                                self.embedding_cache.embedding_index[seq_id] = {
                                    "hash": seq_hash,
                                    "sequence": seq,
                                    "file": cache_file
                                }
                            except Exception as e:
                                logger.warning(f"从哈希加载缓存失败: {e}")

                # 5. 处理缓存命中情况
                if cached_data is not None:
                    # 从缓存提取数据
                    embedding = cached_data["embedding"].to(self.device)
                    attention = None
                    if "attention" in cached_data and cached_data["attention"] is not None:
                        attention = cached_data["attention"].to(self.device)

                    # 处理维度，确保统一性
                    if embedding.dim() == 4:
                        # [1, 1, seq_len, dim] -> [1, seq_len, dim]
                        if embedding.shape[0] == 1 and embedding.shape[1] == 1:
                            embedding = embedding.squeeze(1)
                        else:
                            # 其他4D形状，重塑为3D
                            embedding = embedding.reshape(embedding.shape[0], embedding.shape[2], embedding.shape[3])

                    # 将提取的嵌入添加到批次
                    batch_embeddings.append(embedding)
                    if attention is not None:
                        batch_attentions.append(attention)
                    continue

                # 6. 缓存未命中，使用ESM模型计算嵌入
                if not seq:
                    seq = "A"  # 确保非空序列

                # 清理序列，只保留有效氨基酸
                cleaned_seq = ''.join(aa for aa in seq if aa in ESM_AA_MAP)
                if not cleaned_seq:
                    cleaned_seq = "A"

                # 编码序列
                token_ids = [0]  # BOS标记
                for aa in cleaned_seq:
                    token_ids.append(ESM_AA_MAP.get(aa, ESM_AA_MAP['X']))
                token_ids.append(2)  # EOS标记

                # 转换为张量
                token_tensor = torch.tensor(token_ids, device=self.device)
                protein_tensor = ESMProteinTensor(sequence=token_tensor)

                # 使用ESM模型计算嵌入
                with torch.no_grad():
                    try:
                        # 安全计算嵌入
                        logits_output = self._safe_compute_embeddings(
                            self.esm_model,
                            protein_tensor,
                            LogitsConfig(sequence=True, return_embeddings=True)
                        )

                        # 提取嵌入
                        embedding = logits_output.embeddings

                        # 处理维度
                        if embedding.dim() == 4:
                            if embedding.shape[0] == 1 and embedding.shape[1] == 1:
                                embedding = embedding.squeeze(1)
                            else:
                                embedding = embedding.reshape(embedding.shape[0], -1, embedding.shape[-1])

                        # 提取注意力信息
                        attention = None
                        if hasattr(logits_output, 'attentions'):
                            try:
                                attn_data = logits_output.attentions
                                if attn_data.dim() == 4:  # [batch, heads, seq_len, seq_len]
                                    cls_attention = attn_data.mean(dim=1)[:, 0, 1:-1]
                                    attention = F.softmax(cls_attention, dim=-1).unsqueeze(-1)
                            except Exception as e:
                                logger.warning(f"提取注意力失败: {e}")

                        # 保存到缓存
                        if use_cache:
                            self._save_to_cache(seq, seq_id, embedding, attention)

                        # 添加到批次
                        batch_embeddings.append(embedding)
                        if attention is not None:
                            batch_attentions.append(attention)

                    except Exception as e:
                        logger.error(f"ESM嵌入计算失败: {e}")
                        # 创建备用嵌入
                        emb_dim = self.config.ESM_EMBEDDING_DIM
                        backup_embed = torch.zeros(1, len(token_ids), emb_dim, device=self.device)
                        batch_embeddings.append(backup_embed)

            except Exception as e:
                logger.error(f"处理序列 {i} 失败: {e}")
                logger.error(traceback.format_exc())
                # 创建备用嵌入
                emb_dim = self.config.ESM_EMBEDDING_DIM
                backup_embed = torch.zeros(1, 5, emb_dim, device=self.device)
                batch_embeddings.append(backup_embed)

        # 7. 更新缓存统计信息
        if use_cache:
            # 增加缓存命中和总请求计数
            if not hasattr(self.embedding_cache, 'cache_hits'):
                self.embedding_cache.cache_hits = 0
            if not hasattr(self.embedding_cache, 'total_requests'):
                self.embedding_cache.total_requests = 0

            self.embedding_cache.cache_hits += cache_hit_count
            self.embedding_cache.total_requests += batch_size

        # 8. 返回结果，确保格式统一
        if len(batch_embeddings) == 1:
            # 单个序列情况
            result = batch_embeddings[0]
            if batch_attentions and batch_attentions[0] is not None:
                # 添加注意力属性
                setattr(result, 'attention', batch_attentions[0])
            return result
        else:
            # 批次情况
            if len(set(emb.shape[-1] for emb in batch_embeddings)) > 1:
                # 嵌入维度不一致，进行统一
                target_dim = self.config.ESM_EMBEDDING_DIM
                for i, emb in enumerate(batch_embeddings):
                    if emb.shape[-1] != target_dim:
                        # 创建临时投影层
                        temp_proj = nn.Linear(emb.shape[-1], target_dim).to(self.device)
                        # 重塑为2D进行投影，然后恢复原始维度
                        orig_shape = emb.shape
                        emb_2d = emb.reshape(-1, emb.shape[-1])
                        emb_proj = temp_proj(emb_2d)
                        batch_embeddings[i] = emb_proj.reshape(*orig_shape[:-1], target_dim)

            return batch_embeddings

    def _save_to_cache(self, embedding_cache, seq, seq_id, embeddings, attention=None):
        """
        保存嵌入和注意力到缓存

        参数:
            embedding_cache: 嵌入缓存对象
            seq: 蛋白质序列
            seq_id: 序列ID
            embeddings: 嵌入张量
            attention: 注意力张量
        """
        try:
            # 创建缓存数据 - 使用半精度降低存储空间
            cache_data = {
                "embedding": embeddings.detach().cpu().half(),  # 降低存储空间
                "sequence": seq,
                "id": seq_id,
                "timestamp": time.time()  # 添加时间戳以便缓存管理
            }

            # 只有当注意力可用时才保存
            if attention is not None:
                cache_data["attention"] = attention.detach().cpu().half()

            # 生成序列哈希和文件路径
            seq_hash = self._create_seq_hash(seq)
            embedding_file = os.path.join(embedding_cache.cache_dir, f"{seq_hash}.pt")

            # 保存到文件
            torch.save(cache_data, embedding_file, _use_new_zipfile_serialization=True)

            # 更新索引
            embedding_cache.embedding_index[seq_id] = {
                "hash": seq_hash,
                "sequence": seq,
                "file": embedding_file,
                "timestamp": cache_data["timestamp"]
            }

            # 定期保存索引
            if random.random() < 0.05:  # 5%概率保存索引，避免过于频繁的IO
                self._save_cache_index(embedding_cache)
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def _save_cache_index(self, embedding_cache):
        """安全地保存缓存索引"""
        try:
            index_file = os.path.join(embedding_cache.cache_dir, "embedding_index.pkl")
            with open(index_file, "wb") as f:
                pickle.dump(embedding_cache.embedding_index, f)
        except Exception as e:
            logger.warning(f"索引保存失败: {e}")

    def _get_fused_embedding(self, graph_batch, sequences=None):
        """
        优化的序列-结构融合嵌入获取函数，解决维度不匹配问题

        参数:
            graph_batch: 图批次数据
            sequences: 可选的序列列表

        返回:
            图嵌入，序列嵌入，图潜变量，融合嵌入
        """
        try:
            import traceback

            # 1. 提取序列ID (用于缓存查找)
            sequence_ids = None
            if hasattr(graph_batch, 'protein_id'):
                if isinstance(graph_batch.protein_id, list):
                    sequence_ids = graph_batch.protein_id
                else:
                    sequence_ids = [graph_batch.protein_id]

            # 保存当前批次ID
            self.current_batch_ids = sequence_ids

            # 2. 提取序列
            if sequences is None:
                sequences = self._extract_sequences_from_graphs(graph_batch)

            # 3. 获取序列ESM嵌入
            seq_embeddings = self._get_esm_embedding(sequences, sequence_ids)

            # 4. 提取并预处理ESM注意力
            current_esm_attention = None
            if self.use_esm_guidance:
                # 提取注意力信息
                if isinstance(seq_embeddings, list):
                    if len(seq_embeddings) > 0 and hasattr(seq_embeddings[0], 'attention'):
                        current_esm_attention = seq_embeddings[0].attention
                elif hasattr(seq_embeddings, 'attention'):
                    current_esm_attention = seq_embeddings.attention

                # 处理注意力维度 - 使用预定义适配器
                if current_esm_attention is not None:
                    attention_dim = current_esm_attention.size(-1)  # 获取原始维度
                    target_dim = self.config.HIDDEN_DIM  # 目标维度（GAT隐藏层维度）

                    # 使用预定义的适配器进行维度转换
                    if attention_dim != target_dim:
                        logger.info(f"使用预定义适配器处理ESM注意力: {attention_dim} -> {target_dim}")
                        # 保存原始形状
                        orig_shape = current_esm_attention.shape
                        # 分离注意力梯度
                        with torch.no_grad():
                            # 重塑为二维张量
                            flattened = current_esm_attention.reshape(-1, attention_dim)
                            # 使用预定义适配器转换维度
                            adapted = self.esm_attention_adapter(flattened)
                            # 恢复原始形状
                            current_esm_attention = adapted.reshape(*orig_shape[:-1], target_dim)

            # 5. 使用图编码器处理图数据 - 一次性执行前向传播
            try:
                with torch.set_grad_enabled(True):  # 确保梯度流动
                    node_embeddings, graph_embedding, attention_weights = self.graph_encoder(
                        x=graph_batch.x,
                        edge_index=graph_batch.edge_index,
                        edge_attr=graph_batch.edge_attr if hasattr(graph_batch, 'edge_attr') else None,
                        edge_type=graph_batch.edge_type if hasattr(graph_batch, 'edge_type') else None,
                        batch=graph_batch.batch,
                        esm_attention=current_esm_attention  # 传入预处理后的注意力
                    )
            except Exception as e:
                logger.error(f"图编码器处理失败: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"图编码器处理失败: {e}")

            # 6. 池化序列嵌入
            try:
                pooled_seq_emb = self._pool_sequence_embeddings(seq_embeddings)
            except Exception as e:
                logger.error(f"序列嵌入池化失败: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"序列嵌入池化失败: {e}")

            # 7. 映射与融合
            try:
                # 映射图嵌入到潜变量空间
                graph_latent = self.latent_mapper(graph_embedding)

                # 确保维度正确
                if pooled_seq_emb.dim() > 2:
                    pooled_seq_emb = pooled_seq_emb.reshape(pooled_seq_emb.size(0), -1)

                if graph_latent.dim() > 2:
                    graph_latent = graph_latent.reshape(graph_latent.size(0), -1)

                # logger.info(f"融合前维度 - 图潜变量: {graph_latent.shape}, 序列嵌入: {pooled_seq_emb.shape}")

                # 使用交叉注意力融合模块执行融合
                fused_embedding = self.fusion_module(pooled_seq_emb, graph_latent)

            except Exception as e:
                logger.error(f"嵌入融合失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise RuntimeError(f"嵌入融合失败: {e}")

            return graph_embedding, pooled_seq_emb, graph_latent, fused_embedding
        except Exception as e:
            logger.error(f"融合嵌入获取失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"嵌入融合失败: {e}")

    def _ensure_edge_type_consistency(self, edge_index, edge_type):
        """确保边类型与边索引的一致性"""
        if edge_type is None or edge_index.size(1) == 0:
            return edge_type

        # 确保边类型数量与边索引匹配
        if edge_type.size(0) != edge_index.size(1):
            import logging
            logging.warning(f"边类型数量({edge_type.size(0)})与边数量({edge_index.size(1)})不匹配")

            # 如果边类型少于边数量，使用默认类型补齐
            if edge_type.size(0) < edge_index.size(1):
                padding = torch.zeros(
                    edge_index.size(1) - edge_type.size(0),
                    dtype=edge_type.dtype,
                    device=edge_type.device
                )
                edge_type = torch.cat([edge_type, padding], dim=0)
            else:
                # 如果边类型多于边数量，截断多余部分
                edge_type = edge_type[:edge_index.size(1)]

        # 确保边类型在有效范围内
        if self.config.EDGE_TYPES > 0:  # 如果配置了边类型数量
            if edge_type.max() >= self.config.EDGE_TYPES:
                import logging
                logging.warning(f"边类型最大值({edge_type.max().item()})超出配置的边类型数量({self.config.EDGE_TYPES})")
                # 将超出范围的边类型设为0
                edge_type = torch.clamp(edge_type, 0, self.config.EDGE_TYPES - 1)

        return edge_type

    def _init_embedding_cache(self):
        """初始化嵌入缓存"""
        try:
            # 获取缓存目录
            cache_dir = getattr(self.config, 'EMBEDDING_CACHE_DIR', 'data/esm_embeddings')

            # 创建缓存目录
            os.makedirs(cache_dir, exist_ok=True)

            # 创建缓存结构
            self.embedding_cache = type('EmbeddingCache', (), {
                'cache_dir': cache_dir,
                'embedding_index': {},
                'cache_hits': 0,
                'total_requests': 0
            })

            # 加载缓存索引
            index_file = os.path.join(cache_dir, "embedding_index.pkl")
            if os.path.exists(index_file):
                try:
                    with open(index_file, "rb") as f:
                        self.embedding_cache.embedding_index = pickle.load(f)
                    logger.info(f"加载了嵌入缓存索引，包含{len(self.embedding_cache.embedding_index)}条记录")
                except Exception as e:
                    logger.warning(f"加载缓存索引失败: {e}, 使用空索引")

            # 设置缓存验证标志
            self.embedding_cache.validated = False

            logger.info(f"初始化ESM嵌入缓存: {cache_dir}")
            return True
        except Exception as e:
            logger.warning(f"初始化缓存失败: {e}")
            self.embedding_cache = None
            return False

    def _log_cache_stats(self):
        """记录缓存统计信息"""
        # 避免频繁记录，仅在特定步骤记录
        should_log = (hasattr(self, 'global_step') and
                      self.global_step % getattr(self.config, 'LOG_INTERVAL', 10) == 0 and
                      getattr(self, 'is_master', True))

        if should_log and hasattr(self, 'embedding_cache') and self.embedding_cache is not None:
            hits = getattr(self.embedding_cache, 'cache_hits', 0)
            total = getattr(self.embedding_cache, 'total_requests', 0)

            if total > 0:
                hit_rate = (hits / total) * 100
                logger.info(f"ESM嵌入缓存命中率: {hit_rate:.1f}%, 命中/总数: {hits}/{total}")

                # 记录到TensorBoard
                if hasattr(self, 'writer') and self.writer is not None:
                    self.writer.add_scalar('cache/hit_rate', hit_rate, self.global_step)
                    self.writer.add_scalar('cache/hits', hits, self.global_step)
                    self.writer.add_scalar('cache/total_requests', total, self.global_step)

    def _extract_sequence_ids(self, graph_batch):
        """从图批次中提取序列ID，支持多种数据格式"""
        sequence_ids = None

        # 检查protein_id字段
        if hasattr(graph_batch, 'protein_id'):
            if isinstance(graph_batch.protein_id, list):
                sequence_ids = graph_batch.protein_id
            else:
                sequence_ids = [graph_batch.protein_id]

        # 检查其他可能的ID字段
        elif hasattr(graph_batch, 'id') or hasattr(graph_batch, 'ids'):
            id_field = getattr(graph_batch, 'id', None) or getattr(graph_batch, 'ids', None)
            if isinstance(id_field, list):
                sequence_ids = id_field
            else:
                sequence_ids = [id_field]

        # 检查string_data中的ID
        elif hasattr(graph_batch, 'string_data'):
            for key in ['protein_id', 'id', 'ids', 'sequence_id']:
                if key in graph_batch.string_data:
                    value = graph_batch.string_data[key]
                    if isinstance(value, list):
                        sequence_ids = value
                    else:
                        sequence_ids = [value]
                    break

        # 如果仍然没有ID，使用序列哈希
        if sequence_ids is None and hasattr(graph_batch, 'sequence'):
            sequences = graph_batch.sequence
            if isinstance(sequences, list):
                sequence_ids = [self._create_seq_hash(seq) for seq in sequences]
            elif isinstance(sequences, str):
                sequence_ids = [self._create_seq_hash(sequences)]

        return sequence_ids

    def _pool_sequence_embeddings(self, seq_embeddings):
        """序列嵌入池化，处理不同维度的输入"""
        try:
            if isinstance(seq_embeddings, list):
                # 处理嵌入列表
                pooled_list = []
                for emb in seq_embeddings:
                    if emb.dim() == 3:  # [batch, seq_len, dim]
                        # 使用首尾和平均池化
                        cls_token = emb[:, 0, :]
                        eos_token = emb[:, -1, :]
                        avg_token = emb.mean(dim=1)
                        pooled = (cls_token + eos_token + avg_token) / 3.0
                        pooled_list.append(pooled)
                    elif emb.dim() == 2:  # [seq_len, dim]
                        cls_token = emb[0:1, :]
                        eos_token = emb[-1:, :]
                        avg_token = emb.mean(dim=0, keepdim=True)
                        pooled = (cls_token + eos_token + avg_token) / 3.0
                        pooled_list.append(pooled)
                    else:
                        pooled_list.append(emb)

                # 合并所有池化嵌入
                return torch.cat(pooled_list, dim=0)

            elif seq_embeddings.dim() == 3:  # [batch, seq_len, dim]
                # 标准三维张量池化
                cls_tokens = seq_embeddings[:, 0, :]
                eos_tokens = seq_embeddings[:, -1, :]
                avg_tokens = seq_embeddings.mean(dim=1)
                return (cls_tokens + eos_tokens + avg_tokens) / 3.0

            elif seq_embeddings.dim() == 2:
                # 检查是批次维度还是序列维度在前
                if seq_embeddings.size(0) > seq_embeddings.size(1):
                    # 可能已经是池化后的结果 [batch, dim]
                    return seq_embeddings
                else:
                    # 单个序列 [seq_len, dim]
                    cls_token = seq_embeddings[0]
                    eos_token = seq_embeddings[-1]
                    avg_token = seq_embeddings.mean(dim=0)
                    pooled = (cls_token + eos_token + avg_token) / 3.0
                    return pooled.unsqueeze(0)  # [1, dim]

            # 其他情况，直接返回
            return seq_embeddings

        except Exception as e:
            logger.error(f"序列池化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # 返回备用嵌入
            batch_size = 1
            if isinstance(seq_embeddings, list):
                batch_size = len(seq_embeddings)
            elif seq_embeddings.dim() > 0:
                batch_size = seq_embeddings.size(0)

            return torch.zeros(batch_size, self.config.ESM_EMBEDDING_DIM, device=self.device)
    def _create_seq_hash(self, sequence):
        """为序列创建哈希值"""
        if not sequence:
            return hashlib.md5("empty".encode()).hexdigest()
        return hashlib.md5(sequence.encode()).hexdigest()

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
        """增强的蛋白质多模态融合损失函数"""
        batch_size = graph_embedding.size(0)

        # 1. InfoNCE对比损失 - 促进模态对齐
        # 确保维度兼容性
        if graph_latent.size(1) != seq_embedding.size(1):
            target_dim = min(graph_latent.size(1), seq_embedding.size(1))
            graph_latent_proj = graph_latent[:, :target_dim]
            seq_embedding_proj = seq_embedding[:, :target_dim]
        else:
            graph_latent_proj = graph_latent
            seq_embedding_proj = seq_embedding

        temperature = self.config.TEMPERATURE
        sim_matrix = torch.mm(graph_latent_proj, seq_embedding_proj.t()) / temperature
        labels = torch.arange(batch_size, device=graph_latent.device)

        # 双向对比损失
        loss_g2s = F.cross_entropy(sim_matrix, labels)
        loss_s2g = F.cross_entropy(sim_matrix.t(), labels)
        contrast_loss = (loss_g2s + loss_s2g) / 2.0

        # 2. 融合一致性损失 - 增加维度适配
        # 进行潜在维度投影，确保维度匹配
        if fused_embedding.size(1) != graph_embedding.size(1):
            if not hasattr(self, '_fusion_to_graph_proj'):
                self._fusion_to_graph_proj = nn.Linear(
                    fused_embedding.size(1), graph_embedding.size(1)).to(fused_embedding.device)
            fused_for_graph = self._fusion_to_graph_proj(fused_embedding)
        else:
            fused_for_graph = fused_embedding

        if fused_embedding.size(1) != seq_embedding.size(1):
            if not hasattr(self, '_fusion_to_seq_proj'):
                self._fusion_to_seq_proj = nn.Linear(
                    fused_embedding.size(1), seq_embedding.size(1)).to(fused_embedding.device)
            fused_for_seq = self._fusion_to_seq_proj(fused_embedding)
        else:
            fused_for_seq = fused_embedding

        # 计算余弦相似度
        fusion_g_sim = F.cosine_similarity(fused_for_graph, graph_embedding).mean()
        fusion_s_sim = F.cosine_similarity(fused_for_seq, seq_embedding).mean()
        consistency_loss = (2.0 - fusion_g_sim - fusion_s_sim) * 0.5

        # 3. 结构感知的散度损失
        # 定义二阶关系保持损失函数
        def pairwise_distances(x):
            return torch.cdist(x, x, p=2)

        # 获取原始模态的距离矩阵
        g_dist = pairwise_distances(graph_embedding)
        s_dist = pairwise_distances(seq_embedding)
        f_dist = pairwise_distances(fused_embedding)

        # 归一化距离矩阵
        g_dist = g_dist / (g_dist.max() + 1e-8)
        s_dist = s_dist / (s_dist.max() + 1e-8)
        f_dist = f_dist / (f_dist.max() + 1e-8)

        # 结构保持损失
        structure_loss = (F.mse_loss(f_dist, g_dist) + F.mse_loss(f_dist, s_dist)) * 0.5

        # 4. 创建一个虚拟损失，确保所有参数都有梯度
        # 添加微小的参数正则化项
        param_loss = 0
        for module in [self.graph_encoder, self.latent_mapper, self.fusion_module]:
            if hasattr(module, 'parameters'):
                for p in module.parameters():
                    if p.requires_grad:
                        param_loss = param_loss + (p * 0).sum() * 1e-8

        # 总损失 - 加权组合
        total_loss = (contrast_loss +
                      0.5 * consistency_loss +
                      0.3 * structure_loss +
                      param_loss)

        # 创建损失字典用于记录
        loss_dict = {
            'total_loss': total_loss,
            'contrast_loss': contrast_loss,
            'consistency_loss': consistency_loss,
            'structure_loss': structure_loss,
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


        return test_stats


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