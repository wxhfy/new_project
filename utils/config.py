#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质多模态融合学习框架配置文件

该模块定义了训练、模型和评估的所有超参数，
专门针对抗菌肽(AMPs)设计的图神经网络与序列模型融合优化。
支持多GPU分布式训练加速。

作者: wxhfy
日期: 2025-03-29
"""

import os
import torch
from datetime import datetime


class Config:
    """配置类，包含所有训练和模型参数"""

    # =============== 基础配置 ===============
    PROJECT_NAME = "MultiModal"
    EXPERIMENT_NAME = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # =============== GPU配置 ===============
    # 多GPU支持
    USE_DISTRIBUTED = True  # 是否使用分布式训练
    WORLD_SIZE = -1  # 总进程数，-1表示自动检测
    NUM_GPUS = torch.cuda.device_count()  # 可用GPU数量
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))  # 当前进程在节点内的排名
    GLOBAL_RANK = int(os.environ.get("RANK", 0))  # 当前进程在全局的排名
    MASTER_ADDR = os.environ.get("MASTER_ADDR", "localhost")  # 主节点地址
    MASTER_PORT = os.environ.get("MASTER_PORT", "12355")  # 主节点端口
    DIST_BACKEND = "nccl"  # 分布式后端，GPU用nccl，CPU用gloo

    # 设备选择逻辑
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() and NUM_GPUS > 0 else "cpu")

    SEED = 42
    NUM_WORKERS = 32  # 每个GPU的数据加载器工作进程数

    # =============== 路径配置 ===============
    # 使用实际项目根目录
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 然后定义其他依赖于BASE_DIR的路径
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", EXPERIMENT_NAME)
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
    CACHE_DIR = os.path.join(BASE_DIR, "data")

    # 创建必要的目录
    for directory in [MODEL_DIR, LOG_DIR, RESULT_DIR, CACHE_DIR]:
        os.makedirs(directory, exist_ok=True)

    # =============== 数据配置 ===============
    # --- 预计算嵌入配置 ---
    USE_PRECOMPUTED_EMBEDDINGS = True  # 是否使用预计算嵌入，必须为True以启用预计算嵌入管理器
    FALLBACK_TO_COMPUTE = False  # 不启用回退机制，完全依赖预计算嵌入
    PRECOMPUTED_FORMAT = "pt"  # 预计算嵌入格式，'pt', 'hdf5'或'auto'(自动检测)
    PRECOMPUTED_EMBEDDINGS_PATH = "/home/20T-1/fyh0106/kg/embeddings/"  # 预计算嵌入文件路径，支持{dataset_name}占位符

    # 数据处理参数
    MAX_SEQ_LENGTH = 512
    PLDDT_THRESHOLD = 70.0  # AlphaFold质量阈值
    SAMPLES_PER_LENGTH = 10000  # 每个长度取样本数量
    TRAIN_VAL_TEST_SPLIT = [0.7, 0.2, 0.1]  # 训练:验证:测试比例
    USE_SUBSET = True
    SUBSET_RATIO = 0.1

    # 训练数据缓存文件
    TRAIN_CACHE = os.path.join(CACHE_DIR, "train_data.pt")
    VAL_CACHE = os.path.join(CACHE_DIR, "val_data.pt")
    TEST_CACHE = os.path.join(CACHE_DIR, "test_data.pt")

    # =============== 图结构配置 ===============
    # 边类型定义
    EDGE_TYPES = 4  # 肽键(0), 氢键(1), 离子(2), 疏水(3)
    EDGE_TYPE_NAMES = ["peptide", "hydrogen_bond", "ionic", "hydrophobic"]

    # 边类型映射字典
    EDGE_TYPE_MAP = {
        "peptide": 0,  # 肽键连接
        "hydrogen_bond": 1,  # 氢键相互作用
        "ionic": 2,  # 离子相互作用(盐桥)
        "hydrophobic": 3  # 疏水相互作用
    }

    # 图数据结构参数
    NODE_INPUT_DIM = 35  # 节点特征维度
    EDGE_INPUT_DIM = 8  # 边特征维度

    # 节点特征维度说明 (35维)
    NODE_FEATURE_INDICES = {
        "blosum62": slice(0, 20),  # BLOSUM62编码：20维
        "position": slice(20, 23),  # 3D坐标：3维
        "hydropathy": 23,  # 疏水性：1维
        "charge": 24,  # 电荷：1维
        "molecular_weight": 25,  # 分子量：1维
        "volume": 26,  # 体积：1维
        "flexibility": 27,  # 柔性：1维
        "aromaticity": 28,  # 芳香性：1维
        "secondary_structure": slice(29, 32),  # 二级结构编码：3维
        "sasa": 32,  # 溶剂可及性：1维
        "side_chain_flexibility": 33,  # 侧链柔性：1维
        "plddt": 34  # pLDDT质量评分：1维
    }

    # 边特征维度说明 (8维)
    EDGE_FEATURE_INDICES = {
        "interaction_type": slice(0, 4),  # 相互作用类型：4维one-hot
        "distance": 4,  # 空间距离：1维
        "strength": 5,  # 相互作用强度：1维
        "direction": slice(6, 8)  # 方向向量：2维
    }

    # 分布式ESM设置
    ESM_SHARDED = False  # 是否在多个GPU上分片ESM模型（未使用）
    # ESM_MODEL_NAME = "esm2_t36_3B_UR50D"  # ESM模型名称，已移除，因为不使用实时计算
    ESM_EMBEDDING_DIM = 2560  # ESM模型嵌入维度，保留以确保模型架构兼容
    ESM_GUIDANCE = True  # 不启用ESM注意力引导机制，因为不使用ESM模型

    # =============== 图模型配置 ===============
    HIDDEN_DIM = 512
    OUTPUT_DIM = 512  # 最终图表示维度
    NUM_LAYERS = 3  # GAT层数
    NUM_HEADS = 16  # 注意力头数
    DROPOUT = 0.2

    # 特性开关
    USE_POS_ENCODING = True  # 相对位置编码
    USE_HETEROGENEOUS_EDGES = True  # 异质边处理
    USE_EDGE_PRUNING = True  # 动态边修剪

    # =============== 融合模块配置 ===============
    FUSION_HIDDEN_DIM = 512
    FUSION_OUTPUT_DIM = 512
    FUSION_NUM_HEADS = 16
    FUSION_NUM_LAYERS = 3
    FUSION_DROPOUT = 0.1

    # =============== 训练配置 ===============
    EPOCHS = 50

    # 批次大小设置（分布式训练）
    GLOBAL_BATCH_SIZE = 512  # 全局批次大小
    _batch_size = None  # 添加私有变量存储计算结果

    # 每个GPU的批次大小会自动计算: GLOBAL_BATCH_SIZE // NUM_GPUS (如果NUM_GPUS > 0)
    @property
    def BATCH_SIZE(self):
        """计算每个GPU的批处理大小"""
        if self._batch_size is None:
            if self.USE_DISTRIBUTED and self.NUM_GPUS > 0:
                self._batch_size = max(1, self.GLOBAL_BATCH_SIZE // self.NUM_GPUS)
            else:
                self._batch_size = self.GLOBAL_BATCH_SIZE
        return self._batch_size

    # 添加setter方法允许外部设置批处理大小
    @BATCH_SIZE.setter
    def BATCH_SIZE(self, value):
        if value <= 0:
            raise ValueError(f"批处理大小必须为正整数，当前值: {value}")
        self._batch_size = value

    EVAL_BATCH_SIZE = 256  # 验证和测试时的批次大小
    ACCUMULATION_STEPS = 1  # 梯度累积步数，可进一步增大有效批次大小

    # 优化器配置
    OPTIMIZER = "AdamW"  # 'Adam', 'AdamW', 'SGD'
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    LR_SCHEDULER = "cosine"  # 'step', 'cosine', 'linear', 'constant'
    WARMUP_STEPS = 1000
    GRADIENT_CLIP = 1.0

    # 分布式训练优化
    FP16_TRAINING = True  # 是否使用混合精度训练
    SYNC_BN = True  # 是否使用跨GPU同步BatchNorm

    EARLY_STOPPING_PATIENCE = 3

    # 对比学习参数
    TEMPERATURE = 0.07
    CONTRASTIVE_WEIGHT = 0.5

    # 模型权重
    GRAPH_MODEL_WEIGHT = 0.3  # 图模型权重
    SEQ_MODEL_WEIGHT = 0.7  # 序列模型权重

    # 训练配置 - 内存管理
    MEMORY_CLEAN_INTERVAL = 50  # 每隔多少批次清理一次GPU内存，0表示不清理

    # =============== 任务配置 ===============
    # 抗菌肽活性预测
    TASK_TYPE = "classification"  # 'classification' 或 'regression'
    NUM_CLASSES = 2  # 如果是分类任务
    TASK_METRICS = ["accuracy", "f1", "auc"]  # 评估指标

    # =============== 日志配置 ===============
    LOG_INTERVAL = 5  # 每隔多少batch记录一次训练信息
    EVAL_INTERVAL = 5  # 每隔多少epoch进行一次验证
    SAVE_INTERVAL = 5  # 每隔多少epoch保存一次模型
    DIST_LOGGING = True  # 分布式训练时是否只在主进程中记录日志

    # =============== 氨基酸特性映射 ===============
    # 疏水性值 (Kyte & Doolittle)
    HYDROPATHY = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }

    # 电荷值
    CHARGE = {
        'A': 0.0, 'C': 0.0, 'D': -1.0, 'E': -1.0, 'F': 0.0,
        'G': 0.0, 'H': 0.5, 'I': 0.0, 'K': 1.0, 'L': 0.0,
        'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0, 'R': 1.0,
        'S': 0.0, 'T': 0.0, 'V': 0.0, 'W': 0.0, 'Y': 0.0
    }

    def __init__(self, **kwargs):
        """允许通过kwargs覆盖默认配置"""
        self._batch_size = None
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"配置项 {key} 不存在")

        # 自动调整分布式训练参数
        if self.USE_DISTRIBUTED and self.WORLD_SIZE == -1:
            if "WORLD_SIZE" in os.environ:
                self.WORLD_SIZE = int(os.environ["WORLD_SIZE"])
            else:
                self.WORLD_SIZE = self.NUM_GPUS

    def save(self, filepath=None):
        """保存配置到文件"""
        # 仅主进程保存配置
        if self.USE_DISTRIBUTED and self.GLOBAL_RANK != 0:
            return

        if filepath is None:
            filepath = os.path.join(self.OUTPUT_DIR, "config.pkl")

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

        # 同时保存为可读文本
        txt_path = filepath.replace(".pkl", ".txt")
        with open(txt_path, 'w') as f:
            for key, value in sorted(self.__dict__.items()):
                if key.isupper() or not key.startswith("_"):  # 排除私有变量
                    f.write(f"{key} = {value}\n")

    @classmethod
    def load(cls, filepath):
        """从文件加载配置"""
        import pickle
        with open(filepath, 'rb') as f:
            config_dict = pickle.load(f)

        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)

        return config

    def setup_distributed(self):
        """设置分布式训练环境"""
        if not self.USE_DISTRIBUTED or self.NUM_GPUS <= 1:
            return

        # 设置环境变量
        if not "MASTER_ADDR" in os.environ:
            os.environ["MASTER_ADDR"] = self.MASTER_ADDR
        if not "MASTER_PORT" in os.environ:
            os.environ["MASTER_PORT"] = str(self.MASTER_PORT)

        # 初始化进程组
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self.DIST_BACKEND,
                world_size=self.WORLD_SIZE,
                rank=self.GLOBAL_RANK
            )

        # 设置当前设备
        torch.cuda.set_device(self.LOCAL_RANK)

        # 同步
        torch.distributed.barrier()

        print(f"分布式进程初始化完成: rank {self.GLOBAL_RANK}, world_size {self.WORLD_SIZE}")
