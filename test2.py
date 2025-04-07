#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质多模态融合模型分层测试工具

该模块提供分层测试和可视化功能，验证各层输出维度，
避免运行整个训练过程才发现问题。

作者: wxhfy
日期: 2025-04-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from torchviz import make_dot
from utils.config import Config
from models.gat_models import ProteinGATv2Encoder
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelArchitectureTester:
    """模型架构测试与可视化工具"""

    def __init__(self, config=None):
        """初始化测试器"""
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.hooks = []
        self.activation_maps = {}
        self.output_dir = os.path.join(self.config.OUTPUT_DIR, "model_tests")
        os.makedirs(self.output_dir, exist_ok=True)

    def _register_hooks(self, model):
        """为模型所有层注册前向传播钩子"""
        self.hooks = []
        self.activation_maps = {}

        def hook_fn(name):
            def fn(module, input, output):
                # 记录每层输出的形状和统计信息
                if isinstance(output, tuple):
                    self.activation_maps[name] = {
                        "shape": [o.shape if isinstance(o, torch.Tensor) else type(o) for o in output],
                        "stats": [self._get_stats(o) if isinstance(o, torch.Tensor) else None for o in output]
                    }
                else:
                    self.activation_maps[name] = {
                        "shape": output.shape,
                        "stats": self._get_stats(output)
                    }

            return fn

        # 为所有模块注册钩子
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.MultiheadAttention)) or \
                    "Conv" in module.__class__.__name__ or \
                    "Attention" in module.__class__.__name__:
                self.hooks.append(module.register_forward_hook(hook_fn(name)))

    def _get_stats(self, tensor):
        """获取张量的基本统计信息"""
        if not isinstance(tensor, torch.Tensor):
            return None

        with torch.no_grad():
            stats = {
                "min": tensor.min().item() if tensor.numel() > 0 else None,
                "max": tensor.max().item() if tensor.numel() > 0 else None,
                "mean": tensor.mean().item() if tensor.numel() > 0 else None,
                "std": tensor.std().item() if tensor.numel() > 0 else None,
                "norm": torch.norm(tensor).item() if tensor.numel() > 0 else None,
                "has_nan": torch.isnan(tensor).any().item()
            }
        return stats

    def test_graph_encoder(self, node_dim=35, edge_dim=8, batch_size=2, num_nodes=20):
        """测试图编码器各层输出维度"""
        logger.info("测试图编码器架构...")

        # 创建模型
        model = ProteinGATv2Encoder(
            node_input_dim=node_dim,
            edge_input_dim=edge_dim,
            hidden_dim=self.config.HIDDEN_DIM,
            output_dim=self.config.OUTPUT_DIM,
            num_layers=self.config.NUM_LAYERS,
            num_heads=self.config.NUM_HEADS,
            edge_types=self.config.EDGE_TYPES,
            dropout=0.0,  # 测试时禁用dropout
            use_heterogeneous_edges=self.config.USE_HETEROGENEOUS_EDGES,
            esm_guidance=self.config.ESM_GUIDANCE
        ).to(self.device)

        # 注册钩子
        self._register_hooks(model)

        # 创建测试输入
        x = torch.randn(num_nodes, node_dim).to(self.device)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2)).to(self.device)
        edge_attr = torch.randn(edge_index.size(1), edge_dim).to(self.device)
        edge_type = torch.randint(0, self.config.EDGE_TYPES, (edge_index.size(1),)).to(self.device)
        batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes // batch_size).to(self.device)
        esm_attention = torch.rand(num_nodes, 1).to(self.device) if self.config.ESM_GUIDANCE else None

        # 前向传递
        try:
            with torch.no_grad():
                node_embeddings, graph_embedding, attention_weights = model(
                    x, edge_index, edge_attr, edge_type, batch, esm_attention
                )
            logger.info(f"图编码器前向传递成功!")
            logger.info(f"节点嵌入维度: {node_embeddings.shape}")
            logger.info(f"图嵌入维度: {graph_embedding.shape}")

            # 打印各层输出维度
            self._print_layer_dimensions()

            # 可视化模型架构
            self._visualize_model(model, (x, edge_index, edge_attr, edge_type, batch, esm_attention),
                                  "graph_encoder_architecture")

            # 释放钩子
            for hook in self.hooks:
                hook.remove()

            return True
        except Exception as e:
            logger.error(f"图编码器测试失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 释放钩子
            for hook in self.hooks:
                hook.remove()

            return False

    def test_full_pipeline(self, node_dim=35, edge_dim=8, batch_size=2, num_nodes=20, seq_len=50):
        """测试完整的多模态融合管道"""
        logger.info("测试完整多模态融合管道...")

        # 这部分需要依据您的具体实现编写，以下是示例框架
        try:
            # 在实际实现中替换为您自己的代码
            from train_embed import ProteinMultiModalTrainer

            # 初始化训练器
            trainer = ProteinMultiModalTrainer(self.config)

            # 创建测试批次数据
            from train_embed import ProteinBatch

            # 这里创建测试数据
            # ... (根据您的实际数据结构创建)

            logger.info("管道测试成功")
            return True
        except Exception as e:
            logger.error(f"完整管道测试失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _print_layer_dimensions(self):
        """打印各层输出维度"""
        logger.info("=" * 50)
        logger.info("模型各层输出维度:")
        logger.info("=" * 50)

        for name, info in self.activation_maps.items():
            if isinstance(info["shape"], list):
                shapes_str = ", ".join(str(s) for s in info["shape"])
                logger.info(f"{name}: {shapes_str}")
            else:
                logger.info(f"{name}: {info['shape']}")

        logger.info("=" * 50)

    def _visualize_model(self, model, inputs, filename):
        """使用torchviz可视化模型架构"""
        try:
            # 前向传递
            outputs = model(*inputs)

            # 使用make_dot生成计算图
            if isinstance(outputs, tuple):
                dot = make_dot(outputs[0], params=dict(model.named_parameters()))
            else:
                dot = make_dot(outputs, params=dict(model.named_parameters()))

            # 保存为PDF和PNG
            dot_path = os.path.join(self.output_dir, f"{filename}")
            dot.format = "pdf"
            dot.render(dot_path)
            dot.format = "png"
            dot.render(dot_path)

            logger.info(f"模型架构可视化已保存至: {dot_path}.pdf 和 {dot_path}.png")

            # 可视化维度流
            self._visualize_dimension_flow(filename)
        except Exception as e:
            logger.error(f"模型可视化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def _visualize_dimension_flow(self, filename):
        """可视化模型中的维度流向"""
        # 提取各层的输出维度
        layers = []
        dims = []

        for name, info in self.activation_maps.items():
            if isinstance(info["shape"], list):
                # 处理多输出层
                for i, shape in enumerate(info["shape"]):
                    if isinstance(shape, torch.Size):
                        layers.append(f"{name}[{i}]")
                        dims.append(shape[-1] if len(shape) > 1 else shape[0])
            elif isinstance(info["shape"], torch.Size):
                # 处理单输出层
                layers.append(name)
                dims.append(info["shape"][-1] if len(info["shape"]) > 1 else info["shape"][0])

        # 绘制维度流向图
        plt.figure(figsize=(12, 8))
        plt.plot(dims, marker='o', linestyle='-', linewidth=2)
        plt.title("模型维度流向图", fontsize=15)
        plt.xlabel("层索引", fontsize=12)
        plt.ylabel("特征维度", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加层名标签
        if len(layers) <= 20:  # 当层数较少时显示所有层名
            plt.xticks(range(len(dims)), layers, rotation=90)
        else:  # 层数过多时只显示部分层名
            step = len(layers) // 10
            indices = list(range(0, len(layers), step))
            labels = [layers[i] for i in indices]
            plt.xticks(indices, labels, rotation=90)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{filename}_dimension_flow.png")
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"维度流向图已保存至: {plot_path}")


if __name__ == "__main__":
    # 从命令行参数加载配置文件
    import argparse

    parser = argparse.ArgumentParser(description="模型架构测试工具")
    parser.add_argument("--config", type=str, default="utils/config.py", help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.Config()

    # 创建测试器
    tester = ModelArchitectureTester(config)

    # 测试图编码器
    tester.test_graph_encoder()

    # 测试完整管道
    # tester.test_full_pipeline()