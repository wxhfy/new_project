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
from pandas.io.formats.style import subset_args
from torch_geometric.data import Data, Batch, InMemoryDataset
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


class ProteinData(Data):
    def __init__(self, **kwargs):
        # 分离字符串字段
        self.string_data = {
            k: v for k, v in kwargs.items()
            if isinstance(v, (str, list)) and not isinstance(v, (torch.Tensor, np.ndarray))
        }

        # 仅保留数值字段给父类
        super().__init__(**{
            k: v for k, v in kwargs.items()
            if k not in self.string_data
        })

    def __getitem__(self, key):
        # 优先从字符串数据中查找
        if key in self.string_data:
            return self.string_data[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(value, (str, list)) and not isinstance(value, (torch.Tensor, np.ndarray)):
            self.string_data[key] = value
        else:
            super().__setitem__(key, value)

    def keys(self):
        return list(super().keys()) + list(self.string_data.keys())

    def to_dict(self):
        return {**super().to_dict(), **self.string_data}

class ProteinMultiModalTrainer:
    """蛋白质多模态融合训练器"""

    def __init__(self, config):
        """
        初始化训练器

        参数:
            config: 配置对象
        """
        self.config = config
        self.device = config.DEVICE
        self.fp16_training = config.FP16_TRAINING
        self.batch_size = config.BATCH_SIZE
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

        # 1. 初始化GAT图编码器
        self.graph_encoder = ProteinGATv2Encoder(
            node_input_dim=self.config.NODE_INPUT_DIM,
            edge_input_dim=self.config.EDGE_INPUT_DIM,
            hidden_dim=self.config.HIDDEN_DIM,
            output_dim=self.config.OUTPUT_DIM,
            num_layers=self.config.NUM_LAYERS,
            num_heads=self.config.NUM_HEADS,
            edge_types=self.config.EDGE_TYPES,
            dropout=self.config.DROPOUT,
            use_pos_encoding=self.config.USE_POS_ENCODING,
            use_heterogeneous_edges=self.config.USE_HETEROGENEOUS_EDGES,
            use_edge_pruning=self.config.USE_EDGE_PRUNING,
            esm_guidance=self.config.ESM_GUIDANCE,
            activation='gelu'
        ).to(self.device)

        # 2. ESM模型初始化（修改此部分）
        logger.info(f"初始化ESM模型: {self.config.ESM_MODEL_NAME}")
        try:
            # 加载模型
            self.esm_model = ESMC.from_pretrained(self.config.ESM_MODEL_NAME, device=self.device)
            logger.info(f"ESM模型成功加载: {self.config.ESM_MODEL_NAME}")


        except Exception as e:
            logger.error(f"ESM模型加载失败: {e}")
            logger.error(f"环境变量: ESM_CACHE_DIR={os.environ.get('ESM_CACHE_DIR')}")
            logger.error(f"模型名称: {self.config.ESM_MODEL_NAME}")
            raise

        # 3. 潜空间映射器
        self.latent_mapper = ProteinLatentMapper(
            input_dim=self.config.OUTPUT_DIM,
            latent_dim=self.config.ESM_EMBEDDING_DIM,
            hidden_dim=self.config.HIDDEN_DIM * 2,
            dropout=self.config.DROPOUT
        ).to(self.device)

        # 4. 对比学习头
        self.contrast_head = CrossModalContrastiveHead(
            embedding_dim=self.config.ESM_EMBEDDING_DIM,
            temperature=self.config.TEMPERATURE
        ).to(self.device)

        # 5. 序列-结构融合模块
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

    def _get_esm_embedding(self, sequences):
        """
        获取序列的ESM嵌入，优化处理流程

        参数:
            sequences (list): 氨基酸序列列表

        返回:
            torch.Tensor: ESM嵌入矩阵
        """
        batch_embeddings = []

        ESM_AA_MAP = {
            'A': 5, 'C': 23, 'D': 13, 'E': 9, 'F': 18,
            'G': 6, 'H': 21, 'I': 12, 'K': 15, 'L': 4,
            'M': 20, 'N': 17, 'P': 14, 'Q': 16, 'R': 10,
            'S': 8, 'T': 11, 'V': 7, 'W': 22, 'Y': 19,
            '_': 32, 'X': 32
        }

        # 逐个处理序列
        for seq_idx, seq in enumerate(sequences):
            try:
                # 确保序列有效
                if not seq or len(seq) == 0:
                    seq = "A"  # 使用单个"A"作为默认序列

                # 清理序列
                cleaned_seq = ''.join(aa for aa in seq if aa in ESM_AA_MAP)
                if not cleaned_seq:
                    cleaned_seq = "A"

                # 限制序列长度
                max_length = 512  # 更保守的长度限制
                if len(cleaned_seq) > max_length:
                    cleaned_seq = cleaned_seq[:max_length]
                    logger.warning(f"序列(索引:{seq_idx})已截断至{max_length}个氨基酸")

                # 确保最小序列长度
                if len(cleaned_seq) < 8:
                    padding = "A" * (8 - len(cleaned_seq))
                    cleaned_seq = cleaned_seq + padding

                # 编码序列
                token_ids = [0]  # BOS标记
                for aa in cleaned_seq:
                    token_ids.append(ESM_AA_MAP.get(aa, ESM_AA_MAP['X']))
                token_ids.append(2)  # EOS标记

                # 转换为张量
                token_tensor = torch.tensor(token_ids, device=self.device).unsqueeze(0)

                # 创建恰当的输入格式
                protein_tensor = ESMProteinTensor(sequence=token_tensor)

                # 使用捕获异常的包装器处理前向传播
                try:
                    with torch.no_grad():
                        # 简单的包装逻辑防止可能的维度错误
                        def safe_forward(tensor):
                            try:
                                # 标准前向传播
                                logits_output = self.esm_model.logits(
                                    tensor,
                                    LogitsConfig(sequence=True, return_embeddings=True)
                                )
                                return logits_output.embeddings
                            except Exception as forward_error:
                                logger.warning(f"ESM模型前向传播失败，尝试备用方法: {forward_error}")
                                # 如果标准方法失败，尝试直接调用ESM模型的表征层
                                try:
                                    if hasattr(self.esm_model, "get_sequence_representations"):
                                        return self.esm_model.get_sequence_representations(tensor.sequence)
                                    else:
                                        raise AttributeError("ESM模型缺少get_sequence_representations方法")
                                except Exception:
                                    # 生成备用嵌入
                                    return torch.zeros(1, len(token_ids), self.config.ESM_EMBEDDING_DIM,
                                                       device=self.device)

                        # 获取嵌入
                        embeddings = safe_forward(protein_tensor)

                        # 验证嵌入维度
                        if embeddings.dim() != 3:
                            logger.warning(f"嵌入维度异常: {embeddings.shape}，重新调整维度")
                            # 尝试调整维度
                            if embeddings.dim() == 2:
                                embeddings = embeddings.unsqueeze(0)
                            elif embeddings.dim() == 4:
                                b, _, s, d = embeddings.shape
                                embeddings = embeddings.reshape(b, s, d)

                        batch_embeddings.append(embeddings)

                except Exception as e:
                    logger.warning(f"处理序列(索引:{seq_idx})时出错: {e}")
                    # 生成备用嵌入
                    token_length = len(token_ids)
                    backup_embed = torch.zeros(1, token_length, self.config.ESM_EMBEDDING_DIM, device=self.device)
                    batch_embeddings.append(backup_embed)
            except Exception as outer_e:
                logger.error(f"序列处理完全失败(索引:{seq_idx}): {outer_e}")
                # 生成最小备用嵌入
                backup_embed = torch.zeros(1, 10, self.config.ESM_EMBEDDING_DIM, device=self.device)
                batch_embeddings.append(backup_embed)

        # 合并批次嵌入
        if len(batch_embeddings) > 1:
            # 尝试处理可能的维度不匹配
            try:
                return torch.cat(batch_embeddings, dim=0)
            except RuntimeError as e:
                logger.warning(f"合并批次嵌入失败: {e}，尝试修复维度不匹配")
                # 找到最常见的序列长度
                seq_lengths = [emb.size(1) for emb in batch_embeddings]
                most_common_length = max(set(seq_lengths), key=seq_lengths.count)

                # 调整所有嵌入到相同长度
                aligned_embeddings = []
                for emb in batch_embeddings:
                    if emb.size(1) != most_common_length:
                        if emb.size(1) < most_common_length:
                            # 填充短序列
                            padding = torch.zeros(
                                emb.size(0), most_common_length - emb.size(1), emb.size(2),
                                device=emb.device
                            )
                            aligned_emb = torch.cat([emb, padding], dim=1)
                        else:
                            # 裁剪长序列
                            aligned_emb = emb[:, :most_common_length, :]
                        aligned_embeddings.append(aligned_emb)
                    else:
                        aligned_embeddings.append(emb)

                return torch.cat(aligned_embeddings, dim=0)
        else:
            return batch_embeddings[0]

    def _extract_sequences_from_graphs(self, graphs_batch):
        """
        从PyG图批次中提取序列 - 兼容自定义ProteinData

        参数:
            graphs_batch: PyG图数据批次

        返回:
            list: 氨基酸序列列表
        """
        sequences = []

        # 检查是否有sequence字段且为列表
        if hasattr(graphs_batch, 'sequence'):
            if isinstance(graphs_batch.sequence, list):
                return graphs_batch.sequence

        # 遍历批次中的每个图
        for i in range(graphs_batch.num_graphs):
            # 获取当前图的掩码
            mask = graphs_batch.batch == i

            # 提取节点特征
            x = graphs_batch.x[mask]

            # 获取序列
            sequence = None

            # 尝试从图属性中提取
            if hasattr(graphs_batch, 'sequence'):
                # 对于ProteinData，sequence已转换为列表
                if isinstance(graphs_batch.sequence, list) and i < len(graphs_batch.sequence):
                    sequence = graphs_batch.sequence[i]

            # 如果无法从属性获取，尝试从节点特征生成
            if sequence is None:
                try:
                    # 假设前20维是氨基酸的one-hot编码
                    aa_indices = torch.argmax(x[:, :20], dim=1).cpu().numpy()
                    aa_map = 'ACDEFGHIKLMNPQRSTVWY'
                    sequence = ''.join([aa_map[idx] for idx in aa_indices])
                except:
                    # 使用占位符
                    sequence = 'A' * x.shape[0]

            sequences.append(sequence)

        return sequences

    def _get_fused_embedding(self, graph_batch, sequences=None):
        """
        获取图嵌入和序列嵌入的融合表示 - 适应字符串字段版本

        参数:
            graph_batch: PyG图数据批次
            sequences (list, optional): 氨基酸序列列表，如果为None则从图中提取

        返回:
            tuple: (图嵌入, 序列嵌入, 图潜空间嵌入, 融合嵌入)
        """
        # 1. 获取图嵌入
        node_embeddings, graph_embedding, _ = self.graph_encoder(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index,
            edge_attr=graph_batch.edge_attr if hasattr(graph_batch, 'edge_attr') else None,
            edge_type=graph_batch.edge_type if hasattr(graph_batch, 'edge_type') else None,
            pos=graph_batch.pos if hasattr(graph_batch, 'pos') else None,
            batch=graph_batch.batch
        )

        # 2. 从图中提取序列或使用提供的序列
        if sequences is None:
            sequences = self._extract_sequences_from_graphs(graph_batch)

        # 3. 获取序列ESM嵌入
        seq_embeddings = self._get_esm_embedding(sequences)

        # 如果是批次中有多个序列，取每个序列的表示
        if seq_embeddings.dim() == 3:  # [batch_size, seq_len, dim]
            # 使用首尾池化: [CLS] token + [EOS] token + 平均池化
            cls_tokens = seq_embeddings[:, 0, :]  # [CLS] token
            eos_tokens = seq_embeddings[:, -1, :]  # [EOS] token
            avg_tokens = seq_embeddings.mean(dim=1)  # 平均池化

            # 组合表示
            pooled_seq_emb = (cls_tokens + eos_tokens + avg_tokens) / 3.0  # [batch_size, dim]
        else:
            # 单个序列
            cls_token = seq_embeddings[0, 0, :]
            eos_token = seq_embeddings[0, -1, :]
            avg_token = seq_embeddings[0].mean(dim=0)

            pooled_seq_emb = (cls_token + eos_token + avg_token) / 3.0
            pooled_seq_emb = pooled_seq_emb.unsqueeze(0)  # [1, dim]

        # 4. 将图嵌入映射到序列空间
        graph_latent = self.latent_mapper(graph_embedding)

        # 5. 序列-结构融合
        fused_embedding = self.fusion_module(pooled_seq_emb, graph_latent)

        return graph_embedding, pooled_seq_emb, graph_latent, fused_embedding

    # 新增融合损失函数 ▼
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
        训练一个epoch - 优化版本

        参数:
            train_loader: 训练数据加载器，包含ProteinData对象的批次
            epoch: 当前epoch数

        返回:
            dict: 训练统计信息
        """
        # 设置模型为训练模式
        self.graph_encoder.train()
        self.latent_mapper.train()
        self.contrast_head.train()
        self.fusion_module.train()

        # 初始化统计信息
        epoch_stats = {
            'loss': 0.0,  # 总损失
            'contrast_loss': 0.0,  # 对比损失
            'consistency_loss': 0.0,  # 一致性损失
            'structure_loss': 0.0,  # 结构损失
            'graph_latent_norm': 0.0,  # 图潜空间嵌入范数
            'seq_emb_norm': 0.0,  # 序列嵌入范数
            'batch_count': 0,  # 成功处理的批次数
            'skipped_batches': 0  # 跳过的批次数
        }

        # 设置进度条（仅在主进程上）
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

                # 数据预处理 - 确保字符串字段正确处理
                if hasattr(batch, 'sequence') and not isinstance(batch.sequence, list):
                    # 确保sequence是列表形式，便于后续处理
                    logger.warning(f"批次 {batch_idx} 的sequence字段不是列表，尝试修复")
                    try:
                        if isinstance(batch.sequence, str):
                            # 单个序列转换为列表
                            batch.sequence = [batch.sequence] * batch.num_graphs
                    except Exception as e:
                        logger.error(f"修复sequence字段失败: {e}")

                # 将数据移到指定设备
                batch = batch.to(self.device)

                # 清零梯度
                self.optimizer.zero_grad()

                # 使用混合精度训练（如果启用）
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

                # 反向传播 - 混合精度版本
                if self.fp16_training:
                    self.scaler.scale(loss).backward()

                    # 梯度裁剪（如果启用）
                    if self.config.GRADIENT_CLIP > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.graph_encoder.parameters(), self.config.GRADIENT_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.latent_mapper.parameters(), self.config.GRADIENT_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.contrast_head.parameters(), self.config.GRADIENT_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.fusion_module.parameters(), self.config.GRADIENT_CLIP)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 标准反向传播
                    loss.backward()

                    # 梯度裁剪（如果启用）
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

                # 学习率预热（如果启用）
                if self.warmup_scheduler is not None and self.global_step < self.config.WARMUP_STEPS:
                    self.warmup_scheduler.step()

                # 更新进度条和日志（仅在主进程上）
                if self.is_master:
                    pbar.update(1)
                    if batch_idx % self.config.LOG_INTERVAL == 0:
                        # 获取当前学习率
                        current_lr = self.optimizer.param_groups[0]['lr']

                        # 更新进度条状态
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

                        # 记录梯度范数（可选）
                        if self.config.LOG_GRAD_NORM and batch_idx % (self.config.LOG_INTERVAL * 5) == 0:
                            for name, param in self.graph_encoder.named_parameters():
                                if param.requires_grad and param.grad is not None:
                                    self.writer.add_histogram(f'grad/graph_encoder.{name}', param.grad,
                                                              self.global_step)

                        # 记录相似度矩阵（定期）
                        if batch_idx % (self.config.LOG_INTERVAL * 10) == 0 and 'similarity' in loss_dict:
                            self.writer.add_figure(
                                'train/similarity_matrix',
                                self._plot_similarity_matrix(loss_dict['similarity'].cpu().numpy()),
                                self.global_step
                            )

            except Exception as e:
                # 捕获并记录训练过程中的任何错误
                logger.error(f"处理批次 {batch_idx} 时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())

                # 增加跳过批次计数
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

        # 记录epoch级别的统计信息
        if self.is_master and epoch_stats['batch_count'] > 0:
            # 记录跳过的批次比例
            skip_ratio = epoch_stats['skipped_batches'] / len(train_loader)
            self.writer.add_scalar('train/skipped_batch_ratio', skip_ratio, epoch)

            # 记录训练进度
            progress = (epoch + 1) / self.config.EPOCHS
            self.writer.add_scalar('train/progress', progress, epoch)

            # 添加训练统计信息的摘要
            logger.info(f"Epoch {epoch + 1} 统计: 损失={epoch_stats['loss']:.4f}, "
                        f"对比损失={epoch_stats['contrast_loss']:.4f}, "
                        f"跳过批次率={skip_ratio:.2%}")

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
            'batch_count': 0
        }

        # 验证循环
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中", disable=not self.is_master):
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

        # 计算平均统计信息
        for key in val_stats:
            if key != 'batch_count':
                val_stats[key] /= val_stats['batch_count']

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
            'embeddings': [],
            'sequences': []
        }

        # 测试循环
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="测试中", disable=not self.is_master):
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

        # 计算平均统计信息
        for key in test_stats:
            if key not in ['batch_count', 'embeddings', 'sequences']:
                test_stats[key] /= test_stats['batch_count']

        # 可视化测试结果（仅在主进程上）
        if self.is_master:
            self._visualize_test_results(test_stats)

        return test_stats

    def _analyze_embedding_correlations(self, embeddings, sequences):
        """
        分析嵌入相关性

        参数:
            embeddings (dict): 各种嵌入类型字典
            sequences (list): 序列列表
        """
        # 创建特性字典
        seq_properties = {}

        # 计算序列属性
        for i, seq in enumerate(sequences):
            hydrophobic_count = sum(1 for aa in seq if aa in 'AILMFWYV')
            charged_count = sum(1 for aa in seq if aa in 'DEKRH')
            polar_count = sum(1 for aa in seq if aa in 'STNQ')

            # 序列特性
            seq_properties[i] = {
                'length': len(seq),
                'hydrophobic_ratio': hydrophobic_count / len(seq) if len(seq) > 0 else 0,
                'charged_ratio': charged_count / len(seq) if len(seq) > 0 else 0,
                'polar_ratio': polar_count / len(seq) if len(seq) > 0 else 0
            }

        # 样本量太大时随机选择一部分
        max_samples = 1000
        if len(sequences) > max_samples:
            selected_indices = np.random.choice(len(sequences), max_samples, replace=False)
        else:
            selected_indices = np.arange(len(sequences))

        # 特性与嵌入维度相关性分析
        for embed_name in ['fused']:
            embed = embeddings[embed_name][selected_indices]

            # 主成分分析
            from sklearn.decomposition import PCA
            pca = PCA(n_components=10)  # 只看前10个主成分
            pca_embed = pca.fit_transform(embed)

            # 计算与序列特性的相关性
            import pandas as pd

            # 创建特性数据框
            df_props = pd.DataFrame([seq_properties[i] for i in selected_indices])

            # 创建主成分数据框
            df_pca = pd.DataFrame(
                pca_embed,
                columns=[f'PC{i + 1}' for i in range(pca_embed.shape[1])]
            )

            # 合并数据并计算相关系数
            df = pd.concat([df_props, df_pca], axis=1)
            corr = df.corr()

            # 绘制相关性热图
            plt.figure(figsize=(12, 10))
            corr_plot = sns.heatmap(corr.iloc[:4, 4:], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f"{embed_name}嵌入与序列特性相关性")

            self.writer.add_figure(f'test/{embed_name}_property_correlation', corr_plot.figure,
                                   self.global_step)

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
            return

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

        # TSNE可视化
        tsne_fig, tsne_proj = self._visualize_embeddings(all_embeddings, method='tsne')
        self.writer.add_figure('test/tsne_visualization', tsne_fig, self.global_step)

        # PCA可视化
        pca_fig, pca_proj = self._visualize_embeddings(all_embeddings, method='pca')
        self.writer.add_figure('test/pca_visualization', pca_fig, self.global_step)

        # 保存各种嵌入的相似度图
        graph_seq_sim = np.corrcoef(all_embeddings['graph'], all_embeddings['seq'])
        latent_sim = np.corrcoef(all_embeddings['graph_latent'], all_embeddings['seq'])

        # 绘制相似度矩阵
        sim_fig, axs = plt.subplots(1, 2, figsize=(18, 8))

        sns.heatmap(graph_seq_sim[:len(all_embeddings['graph']), len(all_embeddings['graph']):],
                    cmap='coolwarm', ax=axs[0], vmin=-1, vmax=1)
        axs[0].set_title("图结构-序列相似度矩阵")

        sns.heatmap(latent_sim[:len(all_embeddings['graph_latent']), len(all_embeddings['graph_latent']):],
                    cmap='coolwarm', ax=axs[1], vmin=-1, vmax=1)
        axs[1].set_title("潜空间-序列相似度矩阵")

        self.writer.add_figure('test/similarity_matrices', sim_fig, self.global_step)

        # 嵌入相关性分析
        self._analyze_embedding_correlations(all_embeddings, test_stats['sequences'])

        # 保存投影结果用于后续分析
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
            'config': vars(self.config)
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
        """
        logger.info(f"从 {checkpoint_path} 加载检查点")

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 恢复模型状态
        self.graph_encoder.load_state_dict(checkpoint['graph_encoder'])
        self.latent_mapper.load_state_dict(checkpoint['latent_mapper'])
        self.contrast_head.load_state_dict(checkpoint['contrast_head'])
        self.fusion_module.load_state_dict(checkpoint['fusion_module'])

        # 恢复优化器和调度器状态
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and checkpoint['scheduler']:
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

        self.patience_counter = checkpoint['patience_counter']

        logger.info(f"成功加载检查点，恢复于 epoch {self.current_epoch}")

        return checkpoint.get('metrics', {})

    def export_model(self):
        """导出训练好的模型用于推理"""
        if not self.is_master:
            return

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
            json.dump(vars(self.config), f, indent=2)

        logger.info(f"模型导出到 {export_dir}")

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

        # 初始化最佳验证指标
        if not hasattr(self, 'best_validation_score'):
            self.best_validation_score = float('inf')

        # 训练循环
        for epoch in range(self.current_epoch, self.config.EPOCHS):
            self.current_epoch = epoch

            # 设置分布式采样器的epoch
            if self.config.USE_DISTRIBUTED:
                if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
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

                # 更新早停策略：使用加权组合的验证指标
                validation_score = val_stats['loss'] * 0.6 - val_stats['consistency_loss'] * 0.2 - val_stats[
                    'structure_loss'] * 0.2
                is_best = validation_score < self.best_validation_score

                if is_best:
                    self.best_validation_score = validation_score
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

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


class ProteinGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(None, None, None, None)

        # 确保data_list是列表类型，且每个元素是Data对象
        if isinstance(data_list, list):
            self.data_list = data_list
        else:
            # 如果不是列表，尝试转换为列表
            self.data_list = [data_list]

        # 正确设置数据和切片
        self.data, self.slices = self.collate(self.data_list)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        # 确保返回正确的Data对象
        if idx < len(self.data_list):
            return self.data_list[idx]
        else:
            raise IndexError(f"索引 {idx} 超出范围 (0-{len(self.data_list) - 1})")


def setup_data_loaders(config):
    """
    设置数据加载器 - 使用自定义ProteinData处理字符串字段

    参数:
        config: 配置对象

    返回:
        tuple: (训练加载器, 验证加载器, 测试加载器)
    """
    from torch_geometric.data import InMemoryDataset
    from torch_geometric.loader import DataLoader
    import random

    # 创建数据集类
    class ProteinGraphDataset(InMemoryDataset):
        def __init__(self, data_list):
            super().__init__(None, None, None, None)
            # 确保所有数据都是ProteinData类型
            self.data_list = [
                convert_to_protein_data(item) if not isinstance(item, ProteinData) else item
                for item in data_list
            ]

        def len(self):
            return len(self.data_list)

        def get(self, idx):
            return self.data_list[idx]

    # 提取数据
    try:
        logger.info(f"从 {config.CACHE_DIR} 加载数据...")
        train_data = torch.load(config.TRAIN_CACHE)
        val_data = torch.load(config.VAL_CACHE)
        test_data = torch.load(config.TEST_CACHE)

        # 确保数据是列表格式
        train_data = train_data if isinstance(train_data, list) else [train_data]
        val_data = val_data if isinstance(val_data, list) else [val_data]
        test_data = test_data if isinstance(test_data, list) else [test_data]

        # 使用子集（如配置）
        if config.USE_SUBSET:
            random.seed(42)  # 保证可复现性
            subset_ratio = config.SUBSET_RATIO

            train_size = max(int(len(train_data) * subset_ratio), 100)
            val_size = max(int(len(val_data) * subset_ratio), 50)
            test_size = max(int(len(test_data) * subset_ratio), 50)

            train_indices = random.sample(range(len(train_data)), train_size)
            val_indices = random.sample(range(len(val_data)), val_size)
            test_indices = random.sample(range(len(test_data)), test_size)

            train_data = [train_data[i] for i in train_indices]
            val_data = [val_data[i] for i in val_indices]
            test_data = [test_data[i] for i in test_indices]

            logger.info(f"使用数据子集 - 采样比例: {subset_ratio}")

        logger.info(f"数据加载成功 - 训练: {len(train_data)}, 验证: {len(val_data)}, 测试: {len(test_data)}")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise

    # 创建数据集
    train_dataset = ProteinGraphDataset(train_data)
    val_dataset = ProteinGraphDataset(val_data)
    test_dataset = ProteinGraphDataset(test_data)

    # 分布式训练设置
    if config.USE_DISTRIBUTED:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.WORLD_SIZE,
            rank=config.GLOBAL_RANK
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=config.WORLD_SIZE,
            rank=config.GLOBAL_RANK
        )

        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=config.WORLD_SIZE,
            rank=config.GLOBAL_RANK
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    # 创建数据加载器 - 不需要自定义collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        sampler=test_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def convert_to_protein_data(data):
    """确保完全分离字符串字段"""
    if isinstance(data, ProteinData):
        return data

    # 获取所有字段
    attrs = {k: data[k] for k in data.keys()}

    # 创建ProteinData实例
    protein_data = ProteinData(**attrs)

    # 验证转换
    assert hasattr(protein_data, 'x'), "缺少必要字段 x"
    assert hasattr(protein_data, 'edge_index'), "缺少必要字段 edge_index"
    return protein_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="蛋白质图-序列多模态融合训练")
    parser.add_argument("--config", type=str, default="utils/config.py", help="配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地进程排名")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的检查点路径")
    parser.add_argument("--test_only", action="store_true", help="仅进行测试，不进行训练")
    args = parser.parse_args()

    # 加载配置
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config = config_module.Config()

    # 设置本地进程排名
    if args.local_rank != -1:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
        config.LOCAL_RANK = args.local_rank

    # 设置分布式环境
    if config.USE_DISTRIBUTED:
        config.setup_distributed()

    is_master = not config.USE_DISTRIBUTED or config.GLOBAL_RANK == 0

    # 设置日志级别
    if is_master:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 显示配置信息
    if is_master:
        logger.info(f"使用设备: {config.DEVICE}")
        logger.info(f"分布式训练: {config.USE_DISTRIBUTED}")
        if config.USE_DISTRIBUTED:
            logger.info(f"世界大小: {config.WORLD_SIZE}")
            logger.info(f"全局进程排名: {config.GLOBAL_RANK}")
            logger.info(f"本地进程排名: {config.LOCAL_RANK}")
        logger.info(f"混合精度训练: {config.FP16_TRAINING}")
        logger.info(f"批大小: {config.BATCH_SIZE}")
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

    # 验证一个数据样本（仅主进程）
    if is_master:
        try:
            logger.info("验证数据格式...")
            # 加载一个样本并检查
            sample_data = torch.load(config.TRAIN_CACHE)
            if isinstance(sample_data, list) and len(sample_data) > 0:
                sample = sample_data[0]
            else:
                sample = sample_data

            logger.info(f"数据类型: {type(sample)}")
            logger.info(f"数据字段: {sample.keys()}")  # 调用keys()方法

            # 测试转换为ProteinData
            protein_sample = convert_to_protein_data(sample)
            logger.info(f"转换后类型: {type(protein_sample)}")
            logger.info(f"转换后字段: {protein_sample.keys()}")  # 调用keys()方法

            # 创建小批次测试
            from torch_geometric.data import Batch
            mini_batch = Batch.from_data_list([protein_sample, protein_sample])
            logger.info(f"批次测试成功: {mini_batch.num_graphs}个图")
            logger.info("数据验证通过 ✓")
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # 验证失败时直接终止程序

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
                        f"对比损失: {test_stats['contrast_loss']:.4f}")
    else:
        # 训练
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            resume_from=args.resume
        )

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