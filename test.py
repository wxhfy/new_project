#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质多模态融合模型架构可视化工具

该模块提供直观的模型架构可视化功能，生成模型结构图，
可视化各层特征维度流向和关键组件连接关系。

作者: wxhfy
日期: 2025-04-07
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torchviz import make_dot
import pydot
import hiddenlayer as hl
from utils.config import Config
from models.gat_models import ProteinGATv2Encoder
from models.layers import SequenceStructureFusion
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelVisualizer:
    """模型架构可视化工具"""

    def __init__(self, config=None):
        """初始化可视化器"""
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.output_dir = os.path.join(self.config.OUTPUT_DIR, "model_visualizations")
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_graph_encoder(self, save_formats=["pdf", "png"]):
        """可视化图编码器架构"""
        logger.info("可视化图编码器架构...")

        # 创建模型
        model = ProteinGATv2Encoder(
            node_input_dim=self.config.NODE_INPUT_DIM,
            edge_input_dim=self.config.EDGE_INPUT_DIM,
            hidden_dim=self.config.HIDDEN_DIM,
            output_dim=self.config.OUTPUT_DIM,
            num_layers=self.config.NUM_LAYERS,
            num_heads=self.config.NUM_HEADS,
            edge_types=self.config.EDGE_TYPES,
            use_heterogeneous_edges=self.config.USE_HETEROGENEOUS_EDGES,
            use_edge_pruning=self.config.USE_EDGE_PRUNING,
            esm_guidance=self.config.ESM_GUIDANCE
        ).to(self.device)

        # 创建测试输入
        num_nodes = 20
        batch_size = 2
        x = torch.randn(num_nodes, self.config.NODE_INPUT_DIM).to(self.device)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2)).to(self.device)
        edge_attr = torch.randn(edge_index.size(1), self.config.EDGE_INPUT_DIM).to(self.device)
        edge_type = torch.randint(0, self.config.EDGE_TYPES, (edge_index.size(1),)).to(self.device)
        batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes // batch_size).to(self.device)
        esm_attention = torch.rand(num_nodes, 1).to(self.device) if self.config.ESM_GUIDANCE else None

        # 前向传递
        try:
            outputs = model(x, edge_index, edge_attr, edge_type, batch, esm_attention)

            # 使用torchviz可视化
            if isinstance(outputs, tuple):
                dot = make_dot(outputs[0], params=dict(model.named_parameters()))
            else:
                dot = make_dot(outputs, params=dict(model.named_parameters()))

            # 保存为请求的格式
            base_path = os.path.join(self.output_dir, "graph_encoder")
            for fmt in save_formats:
                dot_path = f"{base_path}.{fmt}"
                dot.format = fmt
                dot.render(base_path, cleanup=True)
                logger.info(f"图编码器架构已保存为: {dot_path}")

            # 使用hiddenlayer尝试更简洁的可视化
            try:
                graph = hl.build_graph(model, (x, edge_index, edge_attr, edge_type, batch, esm_attention))
                hl_path = os.path.join(self.output_dir, "graph_encoder_hl.png")
                graph.save(hl_path, format="png")
                logger.info(f"简化的图编码器架构已保存为: {hl_path}")
            except Exception as e:
                logger.warning(f"hiddenlayer可视化失败，使用备用方法: {str(e)}")

            # 创建自定义维度流图
            self._create_dimension_flow_diagram(model, "graph_encoder_dim_flow.png")

            return True
        except Exception as e:
            logger.error(f"图编码器可视化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def visualize_fusion_module(self, save_formats=["pdf", "png"]):
        """可视化融合模块架构"""
        logger.info("可视化融合模块架构...")

        # 创建融合模块
        fusion = SequenceStructureFusion(
            seq_dim=self.config.ESM_EMBEDDING_DIM,
            graph_dim=self.config.OUTPUT_DIM,
            output_dim=self.config.FUSION_OUTPUT_DIM,
            hidden_dim=self.config.FUSION_HIDDEN_DIM,
            num_heads=self.config.FUSION_NUM_HEADS,
            num_layers=self.config.FUSION_NUM_LAYERS
        ).to(self.device)

        # 创建测试输入
        batch_size = 4
        seq_emb = torch.randn(batch_size, self.config.ESM_EMBEDDING_DIM).to(self.device)
        graph_emb = torch.randn(batch_size, self.config.OUTPUT_DIM).to(self.device)

        # 前向传递
        try:
            output = fusion(seq_emb, graph_emb)

            # 使用torchviz可视化
            dot = make_dot(output, params=dict(fusion.named_parameters()))

            # 保存为请求的格式
            base_path = os.path.join(self.output_dir, "fusion_module")
            for fmt in save_formats:
                dot_path = f"{base_path}.{fmt}"
                dot.format = fmt
                dot.render(base_path, cleanup=True)
                logger.info(f"融合模块架构已保存为: {dot_path}")

            return True
        except Exception as e:
            logger.error(f"融合模块可视化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def visualize_full_pipeline(self, save_formats=["pdf", "png"]):
        """可视化完整的多模态融合管道"""
        logger.info("可视化完整多模态融合管道...")

        # 这部分需要依据您的具体实现编写，以下是示例框架
        try:
            # 在实际实现中替换为您自己的代码
            from train_embed import ProteinMultiModalTrainer

            # 初始化训练器
            trainer = ProteinMultiModalTrainer(self.config)

            # 创建自定义架构图
            self._create_custom_pipeline_diagram()

            logger.info("完整管道可视化成功")
            return True
        except Exception as e:
            logger.error(f"完整管道可视化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_dimension_flow_diagram(self, model, filename):
        """创建维度流向图"""
        # 提取模型中的关键层
        dim_info = []

        # 收集维度信息
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                dim_info.append({
                    "name": name,
                    "in_features": module.in_features,
                    "out_features": module.out_features
                })

        # 排序层
        dim_info.sort(key=lambda x: x["name"])

        # 提取维度
        layer_names = [info["name"].split(".")[-1] for info in dim_info]
        in_dims = [info["in_features"] for info in dim_info]
        out_dims = [info["out_features"] for info in dim_info]

        # 绘制维度流向图
        plt.figure(figsize=(15, 10))

        # 输入维度
        plt.plot(in_dims, marker='o', linestyle='-', linewidth=2, label="输入维度")
        # 输出维度
        plt.plot(out_dims, marker='x', linestyle='--', linewidth=2, label="输出维度")

        plt.title("模型维度流向图", fontsize=16)
        plt.xlabel("层索引", fontsize=14)
        plt.ylabel("维度", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        # 添加层名标签
        if len(layer_names) <= 20:  # 当层数较少时显示所有层名
            plt.xticks(range(len(dim_info)), layer_names, rotation=90)
        else:  # 层数过多时只显示部分层名
            step = len(layer_names) // 10
            indices = list(range(0, len(layer_names), step))
            labels = [layer_names[i] for i in indices]
            plt.xticks(indices, labels, rotation=90)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"维度流向图已保存至: {plot_path}")

    def _create_custom_pipeline_diagram(self):
        """创建多模态融合管道的架构示意图"""
        try:
            import pygraphviz as pgv

            # 创建有向图
            G = pgv.AGraph(directed=True, rankdir="LR", splines="ortho")

            # 定义节点样式
            G.node_attr.update(shape="box", style="filled", fontname="SimHei", fontsize="14")

            # 添加节点并设置颜色
            G.add_node("蛋白质序列", fillcolor="#AED6F1")
            G.add_node("蛋白质图结构", fillcolor="#AED6F1")

            G.add_node("ESM模型", fillcolor="#F5CBA7")
            G.add_node(f"ESM嵌入\n({self.config.ESM_EMBEDDING_DIM}维)", fillcolor="#F5CBA7")
            G.add_node(f"ESM投影\n({self.config.OUTPUT_DIM}维)", fillcolor="#F5CBA7")

            G.add_node("GATv2图编码器", fillcolor="#D5F5E3")
            G.add_node(f"图嵌入\n({self.config.OUTPUT_DIM}维)", fillcolor="#D5F5E3")
            G.add_node("序列-结构融合模块", fillcolor="#FADBD8")
            G.add_node(f"融合嵌入\n({self.config.FUSION_OUTPUT_DIM}维)", fillcolor="#FADBD8")
            G.add_node("生成模型", fillcolor="#E8DAEF")

            # 添加边
            G.add_edge("蛋白质序列", "ESM模型")
            G.add_edge("ESM模型", f"ESM嵌入\n({self.config.ESM_EMBEDDING_DIM}维)")
            G.add_edge(f"ESM嵌入\n({self.config.ESM_EMBEDDING_DIM}维)", f"ESM投影\n({self.config.OUTPUT_DIM}维)")

            G.add_edge("蛋白质图结构", "GATv2图编码器")
            G.add_edge("GATv2图编码器", f"图嵌入\n({self.config.OUTPUT_DIM}维)")

            # 添加ESM注意力引导（如果启用）
            if self.config.ESM_GUIDANCE:
                G.add_edge("ESM模型", "GATv2图编码器", color="red", label="注意力引导")

            G.add_edge(f"ESM投影\n({self.config.OUTPUT_DIM}维)", "序列-结构融合模块")
            G.add_edge(f"图嵌入\n({self.config.OUTPUT_DIM}维)", "序列-结构融合模块")
            G.add_edge("序列-结构融合模块", f"融合嵌入\n({self.config.FUSION_OUTPUT_DIM}维)")
            G.add_edge(f"融合嵌入\n({self.config.FUSION_OUTPUT_DIM}维)", "生成模型")

            # 布局和保存
            G.layout(prog="dot")
            output_path = os.path.join(self.output_dir, "pipeline_diagram")
            G.draw(f"{output_path}.png")
            G.draw(f"{output_path}.pdf")

            logger.info(f"融合管道架构图已保存至: {output_path}.png 和 {output_path}.pdf")

            return True
        except Exception as e:
            logger.error(f"自定义架构图创建失败: {str(e)}")

            # 备用方案：创建简单的matplotlib图
            self._create_simple_pipeline_diagram()
            return False

    def _create_simple_pipeline_diagram(self):
        """创建简化的管道架构图（matplotlib备用方案）"""
        plt.figure(figsize=(12, 8))

        # 使用matplotlib画简单框图
        components = ["蛋白质序列", "ESM模型", "ESM嵌入", "ESM投影",
                      "蛋白质图结构", "GATv2图编码器", "图嵌入",
                      "序列-结构融合模块", "融合嵌入", "生成模型"]

        # 设置位置
        positions = {
            "蛋白质序列": (0, 3),
            "ESM模型": (1, 3),
            "ESM嵌入": (2, 3),
            "ESM投影": (3, 3),
            "蛋白质图结构": (0, 1),
            "GATv2图编码器": (1, 1),
            "图嵌入": (2, 1),
            "序列-结构融合模块": (4, 2),
            "融合嵌入": (5, 2),
            "生成模型": (6, 2)
        }

        # 设置颜色
        colors = {
            "蛋白质序列": "#AED6F1",
            "ESM模型": "#F5CBA7",
            "ESM嵌入": "#F5CBA7",
            "ESM投影": "#F5CBA7",
            "蛋白质图结构": "#AED6F1",
            "GATv2图编码器": "#D5F5E3",
            "图嵌入": "#D5F5E3",
            "序列-结构融合模块": "#FADBD8",
            "融合嵌入": "#FADBD8",
            "生成模型": "#E8DAEF"
        }

        # 绘制框和标签
        for comp in components:
            x, y = positions[comp]
            plt.text(x, y, comp, ha='center', va='center',
                     bbox=dict(facecolor=colors[comp], edgecolor='black', boxstyle='round,pad=0.5'))

        # 绘制连接线
        connections = [
            ("蛋白质序列", "ESM模型"),
            ("ESM模型", "ESM嵌入"),
            ("ESM嵌入", "ESM投影"),
            ("蛋白质图结构", "GATv2图编码器"),
            ("GATv2图编码器", "图嵌入"),
            ("ESM投影", "序列-结构融合模块"),
            ("图嵌入", "序列-结构融合模块"),
            ("序列-结构融合模块", "融合嵌入"),
            ("融合嵌入", "生成模型")
        ]

        for start, end in connections:
            x1, y1 = positions[start]
            x2, y2 = positions[end]
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.7)

        # 添加ESM注意力引导（如果启用）
        if self.config.ESM_GUIDANCE:
            plt.plot([1, 1], [3, 1], 'r--', alpha=0.7, label="注意力引导")
            plt.legend()

        plt.axis('off')
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "simple_pipeline_diagram.png")
        plt.savefig(output_path)
        plt.close()

        logger.info(f"简化融合管道架构图已保存至: {output_path}")


if __name__ == "__main__":
    # 从命令行参数加载配置文件
    import argparse

    parser = argparse.ArgumentParser(description="模型架构可视化工具")
    parser.add_argument("--config", type=str, default="utils/config.py", help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.Config()

    # 创建可视化器
    visualizer = ModelVisualizer(config)

    # 运行可视化
    visualizer.visualize_graph_encoder()
    visualizer.visualize_fusion_module()
    visualizer.visualize_full_pipeline()