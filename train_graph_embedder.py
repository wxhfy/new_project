import os
import argparse
import json
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from protein_gnn_embedding import GCNEncoder, GATEncoder, MultiModalFusion, set_seed, setup_logging
from protein_gnn_embedding import load_protein_graph_dataset, visualize_embeddings, check_memory_usage, save_model, \
    load_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="蛋白质知识图谱嵌入训练工具")

    # 数据参数
    parser.add_argument("--input_path", "-i", type=str, required=True,
                        help="输入目录或文件路径")
    parser.add_argument("--output_dir", "-o", type=str, default="./results",
                        help="输出目录路径 (默认: ./results)")
    parser.add_argument("--min_nodes", type=int, default=5,
                        help="图谱最小节点数 (默认: 5)")
    parser.add_argument("--max_nodes", type=int, default=500,
                        help="图谱最大节点数 (默认: 500)")

    # 训练参数
    parser.add_argument("--model_type", "-m", type=str, default="gcn",
                        choices=["gcn", "gat"], help="模型类型: gcn 或 gat (默认: gcn)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="隐藏层维度 (默认: 256)")
    parser.add_argument("--output_dim", type=int, default=128,
                        help="输出嵌入维度 (默认: 128)")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="GNN层数 (默认: 3)")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="批处理大小 (默认: 32)")
    parser.add_argument("--epochs", "-e", type=int, default=30,
                        help="训练轮数 (默认: 30)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率 (默认: 0.001)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout比例 (默认: 0.3)")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "add", "attention"],
                        help="池化方法 (默认: mean)")
    parser.add_argument("--heads", type=int, default=4,
                        help="GAT注意力头数 (默认: 4)")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU设备ID (默认: 0)")
    parser.add_argument("--test_mode", action="store_true",
                        help="测试模式，只加载少量数据")
    parser.add_argument("--max_test_files", type=int, default=5,
                        help="测试模式下最多处理的文件数 (默认: 5)")
    parser.add_argument("--max_test_graphs", type=int, default=100,
                        help="测试模式下最多处理的图谱数 (默认: 100)")
    parser.add_argument("--visualize", action="store_true",
                        help="是否生成嵌入可视化")

    return parser.parse_args()


def train_embedder(args):
    """训练图嵌入模型"""
    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"使用设备: {device}")

    # 设置日志
    logger, log_file = setup_logging(args.output_dir)
    logger.info(f"开始训练图嵌入模型, 参数: {args}")

    # 加载数据集
    dataset = load_protein_graph_dataset(
        args.input_path,
        args.output_dir,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        test_mode=args.test_mode,
        max_test_files=args.max_test_files,
        max_test_graphs=args.max_test_graphs
    )

    if dataset is None or len(dataset) == 0:
        logger.error("数据集为空，无法训练模型")
        return

    # 划分训练集和验证集
    train_idx, valid_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=args.seed
    )

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]

    logger.info(f"数据集大小: 总共{len(dataset)}个图谱, 训练集{len(train_dataset)}个, 验证集{len(valid_dataset)}个")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # 确定输入特征维度
    input_dim = dataset[0].x.size(1)
    logger.info(f"输入特征维度: {input_dim}")

    # 创建模型
    if args.model_type.lower() == "gcn":
        model = GCNEncoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_norm=True,
            use_edge_attr=True,
            residual=True,
            pooling=args.pooling
        )
    else:  # GAT
        model = GATEncoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_norm=True,
            heads=args.heads,
            residual=True,
            pooling=args.pooling
        )

    model = model.to(device)
    logger.info(f"创建模型: {args.model_type.upper()}, 参数数量: {sum(p.numel() for p in model.parameters())}")

    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 训练跟踪变量
    best_loss = float('inf')
    best_epoch = 0
    patience = 7
    patience_counter = 0

    # 创建模型保存目录
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # 模型配置
    model_config = {
        "model_type": args.model_type,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "heads": args.heads if args.model_type == "gat" else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 训练循环
    logger.info("开始训练...")
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 前向传播
            graph_emb, node_emb = model(batch)

            # 这里使用对比损失，让同一批次中的图谱互相远离
            # 使用余弦相似度计算批次内的图谱相似度
            sim_matrix = F.cosine_similarity(graph_emb.unsqueeze(1), graph_emb.unsqueeze(0), dim=2)

            # 对角线上是自己和自己的相似度，应该是1
            # 非对角线上是与其他图谱的相似度，应该尽可能小
            identity_matrix = torch.eye(sim_matrix.size(0), device=device)
            loss = F.mse_loss(sim_matrix, identity_matrix)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        train_loss /= train_steps

        # 验证阶段
        model.eval()
        valid_loss = 0.0
        valid_steps = 0

        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)

                # 前向传播
                graph_emb, node_emb = model(batch)

                # 计算对比损失
                sim_matrix = F.cosine_similarity(graph_emb.unsqueeze(1), graph_emb.unsqueeze(0), dim=2)
                identity_matrix = torch.eye(sim_matrix.size(0), device=device)
                loss = F.mse_loss(sim_matrix, identity_matrix)

                valid_loss += loss.item()
                valid_steps += 1

        valid_loss /= valid_steps

        # 打印进度
        logger.info(f"Epoch {epoch + 1}/{args.epochs}, 训练损失: {train_loss:.6f}, 验证损失: {valid_loss:.6f}")

        # 保存最佳模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            patience_counter = 0

            # 保存模型
            model_path = os.path.join(model_dir, f"{args.model_type}_embedding_best.pt")
            save_model(model, model_path, model_config)
            logger.info(f"保存最佳模型，验证损失: {valid_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"{patience}个epoch内验证损失未改善，提前停止训练")
                break

        # 定期检查内存使用
        check_memory_usage(force_gc=True)

    logger.info(f"训练完成，最佳模型在第{best_epoch + 1}个epoch，验证损失: {best_loss:.6f}")

    # 加载最佳模型
    best_model_path = os.path.join(model_dir, f"{args.model_type}_embedding_best.pt")
    load_model(model, best_model_path)

    # 如果需要可视化
    if args.visualize:
        logger.info("生成嵌入可视化...")

        # 收集所有嵌入
        all_embeddings = []

        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                graph_emb, _ = model(batch)
                all_embeddings.append(graph_emb.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)

        # 可视化
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        visualize_embeddings(
            all_embeddings,
            method='tsne',
            output_path=os.path.join(viz_dir, f"{args.model_type}_embeddings_tsne.png")
        )

        visualize_embeddings(
            all_embeddings,
            method='pca',
            output_path=os.path.join(viz_dir, f"{args.model_type}_embeddings_pca.png")
        )

    return model, best_model_path


def main():
    """主函数"""
    args = parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 训练模型
    model, model_path = train_embedder(args)

    print(f"训练完成，最佳模型已保存至: {model_path}")


if __name__ == "__main__":
    main()