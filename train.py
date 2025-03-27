#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蛋白质图嵌入系统 - 模型训练脚本

本脚本用于训练蛋白质知识图谱嵌入模型，支持多种任务类型:
1. 节点级任务 (如蛋白质二级结构预测)
2. 边级任务 (如蛋白质相互作用预测)
3. 图级任务 (如蛋白质功能预测)

支持多种训练策略和优化技术，包括早停、学习率调度和模型检查点。

作者: 基于wxhfy的知识图谱处理工作扩展
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

# 导入模型和工具
from models.gat_models import ProteinGATv2, ProteinGATv2WithPretraining
from utils.visualization import plot_training_curves


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='蛋白质图嵌入训练脚本')

    # 基本参数
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='输出目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--task_type', type=str, default='node',
                        choices=['node', 'edge', 'graph'], help='任务类型')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='任务数量(分类类别数或回归输出维度)')
    parser.add_argument('--task_mode', type=str, default='classification',
                        choices=['classification', 'regression'], help='任务模式')

    # 模型参数
    parser.add_argument('--in_channels', type=int, default=21, help='输入特征维度')
    parser.add_argument('--hidden_channels', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--out_channels', type=int, default=64, help='输出特征维度')
    parser.add_argument('--num_layers', type=int, default=3, help='GAT层数')
    parser.add_argument('--heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--edge_dim', type=int, default=None, help='边特征维度')
    parser.add_argument('--readout_type', type=str, default='multihead',
                        choices=['mean', 'max', 'add', 'attention', 'multihead'],
                        help='图读出类型')
    parser.add_argument('--use_edge_attr', action='store_true', help='是否使用边特征')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--early_stop', type=int, default=20, help='早停轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--fp16', action='store_true', help='是否使用半精度训练')
    parser.add_argument('--scheduler', action='store_true', help='是否使用学习率调度器')

    # 数据增强和训练策略
    parser.add_argument('--augmentation', action='store_true', help='是否使用数据增强')
    parser.add_argument('--augment_prob', type=float, default=0.5, help='数据增强概率')
    parser.add_argument('--eval_interval', type=int, default=1, help='验证间隔')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')

    return parser.parse_args()


def load_dataset(data_path, task_type):
    """加载数据集并划分训练/验证/测试集"""
    try:
        # 尝试加载PyG格式数据集
        dataset = torch.load(data_path)
        print(f"成功加载数据集: {data_path}")
        print(f"数据集大小: {len(dataset)}")

        # 随机划分数据集
        torch.manual_seed(42)  # 确保可重复性

        # 对数据集进行洗牌
        indices = torch.randperm(len(dataset))

        # 划分比例: 80% 训练, 10% 验证, 10% 测试
        train_idx = indices[:int(0.8 * len(dataset))]
        val_idx = indices[int(0.8 * len(dataset)):int(0.9 * len(dataset))]
        test_idx = indices[int(0.9 * len(dataset)):]

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]

        print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    except Exception as e:
        print(f"加载数据集失败: {e}")
        return None, None, None


def create_model(args):
    """创建模型"""
    # 创建基础GAT模型
    base_model = ProteinGATv2(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        edge_dim=args.edge_dim,
        readout_type=args.readout_type,
        node_level=(args.task_type == 'node'),
        graph_level=(args.task_type == 'graph'),
        use_edge_attr=args.use_edge_attr
    )

    # 创建带预训练任务的模型
    model = ProteinGATv2WithPretraining(
        gat_model=base_model,
        task_type=args.task_type,
        num_tasks=args.num_tasks
    )

    return model


def get_loss_fn(task_mode):
    """获取损失函数"""
    if task_mode == 'classification':
        return nn.CrossEntropyLoss()  # 多分类
    else:  # 回归
        return nn.MSELoss()




def train_epoch(model, loader, optimizer, loss_fn, device, task_mode, use_augmentation=False, aug_prob=0.5):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    # 用于统计结果
    all_preds = []
    all_targets = []

    for data in loader:

        # 将数据移到设备
        data = data.to(device)
        optimizer.zero_grad()

        # 前向传播
        results = model(data)

        # 根据任务类型获取预测和目标
        if model.task_type == 'node':
            pred = results['node_pred']
            target = data.y
        elif model.task_type == 'edge':
            pred = results['edge_pred']
            target = data.edge_label if hasattr(data, 'edge_label') else data.y
        elif model.task_type == 'graph':
            pred = results['graph_pred']
            target = data.y

        # 计算损失
        if task_mode == 'classification':
            if len(target.shape) > 1:
                target = target.squeeze(-1)  # 去除最后一个维度
            loss = loss_fn(pred, target)
        else:  # 回归
            loss = loss_fn(pred, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item() * data.num_graphs

        # 保存预测和目标
        if task_mode == 'classification':
            pred_cls = pred.argmax(dim=1) if pred.size(1) > 1 else (pred > 0).float()
        else:
            pred_cls = pred

        all_preds.append(pred_cls.detach().cpu())
        all_targets.append(target.detach().cpu())

    # 计算平均损失
    avg_loss = total_loss / len(loader.dataset)

    # 计算指标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if task_mode == 'classification':
        accuracy = accuracy_score(all_targets.numpy(), all_preds.numpy())
        return avg_loss, accuracy
    else:
        rmse = mean_squared_error(all_targets.numpy(), all_preds.numpy(), squared=False)
        r2 = r2_score(all_targets.numpy(), all_preds.numpy())
        return avg_loss, rmse, r2


def evaluate(model, loader, loss_fn, device, task_mode):
    """评估模型"""
    model.eval()
    total_loss = 0

    # 用于统计结果
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # 前向传播
            results = model(data)

            # 根据任务类型获取预测和目标
            if model.task_type == 'node':
                pred = results['node_pred']
                target = data.y
            elif model.task_type == 'edge':
                pred = results['edge_pred']
                target = data.edge_label if hasattr(data, 'edge_label') else data.y
            elif model.task_type == 'graph':
                pred = results['graph_pred']
                target = data.y

            # 计算损失
            if task_mode == 'classification':
                if len(target.shape) > 1:
                    target = target.squeeze(-1)  # 去除最后一个维度
                loss = loss_fn(pred, target)
            else:  # 回归
                loss = loss_fn(pred, target)

            # 累计损失
            total_loss += loss.item() * data.num_graphs

            # 保存预测和目标
            if task_mode == 'classification':
                pred_cls = pred.argmax(dim=1) if pred.size(1) > 1 else (pred > 0).float()
            else:
                pred_cls = pred

            all_preds.append(pred_cls.detach().cpu())
            all_targets.append(target.detach().cpu())

    # 计算平均损失
    avg_loss = total_loss / len(loader.dataset)

    # 计算指标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if task_mode == 'classification':
        accuracy = accuracy_score(all_targets.numpy(), all_preds.numpy())

        # 对于二分类，计算AUC
        if model.num_tasks == 1 or (all_targets.max().item() == 1 and all_targets.min().item() == 0):
            try:
                auc = roc_auc_score(all_targets.numpy(), all_preds.numpy())
                return avg_loss, accuracy, auc
            except:
                return avg_loss, accuracy
        return avg_loss, accuracy
    else:
        rmse = mean_squared_error(all_targets.numpy(), all_preds.numpy(), squared=False)
        r2 = r2_score(all_targets.numpy(), all_preds.numpy())
        return avg_loss, rmse, r2


def save_checkpoint(model, optimizer, epoch, args, metrics, scheduler=None, is_best=False):
    """保存检查点"""
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
    best_path = os.path.join(args.output_dir, "best_model.pt")

    # 构建检查点
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'in_channels': args.in_channels,
            'hidden_channels': args.hidden_channels,
            'out_channels': args.out_channels,
            'num_layers': args.num_layers,
            'heads': args.heads,
            'dropout': args.dropout,
            'edge_dim': args.edge_dim,
            'readout_type': args.readout_type,
            'task_type': args.task_type,
            'task_mode': args.task_mode,
            'num_tasks': args.num_tasks,
            'use_edge_attr': args.use_edge_attr
        }
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # 保存检查点
    torch.save(checkpoint, checkpoint_path)

    # 如果是最优模型，额外保存一份
    if is_best:
        torch.save(checkpoint, best_path)
        print(f"保存最优模型到 {best_path}")

    return checkpoint_path


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cuda'):
    """加载检查点"""
    checkpoint = torch.load(path, map_location=device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 如果提供了优化器，加载优化器状态
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 如果提供了调度器且检查点中存在调度器状态，加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 返回检查点中的轮次和指标
    return checkpoint['epoch'], checkpoint['metrics']


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device)

    # 加载数据集
    train_dataset, val_dataset, test_dataset = load_dataset(args.data_path, args.task_type)
    if train_dataset is None:
        return

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # 创建模型
    model = create_model(args)
    model = model.to(device)

    # 损失函数
    loss_fn = get_loss_fn(args.task_mode)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器
    scheduler = None
    if args.scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    # 设置TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # 训练状态追踪
    best_val_metric = float('inf') if args.task_mode == 'regression' else 0.0
    best_epoch = 0
    early_stop_count = 0
    start_epoch = 0

    # 如果提供了检查点，从检查点恢复
    if args.resume is not None:
        print(f"从检查点恢复: {args.resume}")
        start_epoch, metrics = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch += 1  # 从下一个epoch开始
        print(f"恢复至epoch {start_epoch}, 指标: {metrics}")

    # 如果使用半精度训练
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    print("开始训练...")

    # 保存训练曲线数据
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # 训练一个epoch
        if args.task_mode == 'classification':
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, loss_fn, device, args.task_mode,
                args.augmentation, args.augment_prob
            )
            train_losses.append(train_loss)
            train_metrics.append(train_acc)

            print(f"Epoch {epoch + 1}/{args.epochs}, 耗时: {time.time() - start_time:.2f}s, "
                  f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}", end="")
        else:  # 回归任务
            train_loss, train_rmse, train_r2 = train_epoch(
                model, train_loader, optimizer, loss_fn, device, args.task_mode,
                args.augmentation, args.augment_prob
            )
            train_losses.append(train_loss)
            train_metrics.append(train_rmse)

            print(f"Epoch {epoch + 1}/{args.epochs}, 耗时: {time.time() - start_time:.2f}s, "
                  f"训练损失: {train_loss:.4f}, 训练RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}", end="")

        # 验证
        if (epoch + 1) % args.eval_interval == 0:
            if args.task_mode == 'classification':
                val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, args.task_mode)
                val_metric = val_acc
                val_losses.append(val_loss)
                val_metrics.append(val_metric)

                print(f", 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")

                # 记录到TensorBoard
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)

                # 更新学习率
                if args.scheduler:
                    scheduler.step(val_loss)

                # 检查是否为最佳模型
                is_best = val_metric > best_val_metric
            else:  # 回归任务
                val_loss, val_rmse, val_r2 = evaluate(model, val_loader, loss_fn, device, args.task_mode)
                val_metric = val_rmse  # 使用RMSE作为主要指标
                val_losses.append(val_loss)
                val_metrics.append(val_metric)

                print(f", 验证损失: {val_loss:.4f}, 验证RMSE: {val_rmse:.4f}, 验证R²: {val_r2:.4f}")

                # 记录到TensorBoard
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('RMSE/train', train_rmse, epoch)
                writer.add_scalar('RMSE/val', val_rmse, epoch)
                writer.add_scalar('R2/train', train_r2, epoch)
                writer.add_scalar('R2/val', val_r2, epoch)

                # 更新学习率
                if args.scheduler:
                    scheduler.step(val_loss)

                # 检查是否为最佳模型（对于回归，较小的RMSE更好）
                is_best = val_metric < best_val_metric

            # 保存检查点
            if args.task_mode == 'classification':
                metrics = {'loss': val_loss, 'accuracy': val_acc}
            else:
                metrics = {'loss': val_loss, 'rmse': val_rmse, 'r2': val_r2}

            save_path = save_checkpoint(
                model, optimizer, epoch, args, metrics, scheduler, is_best
            )

            # 更新最佳指标和早停计数
            if is_best:
                best_val_metric = val_metric
                best_epoch = epoch
                early_stop_count = 0
                print(f"新的最佳模型! 指标: {val_metric:.4f}")
            else:
                early_stop_count += 1
                print(f"早停计数: {early_stop_count}/{args.early_stop}")

                if early_stop_count >= args.early_stop:
                    print(f"连续 {args.early_stop} 个epoch没有改进，提前停止训练")
                    break
        else:
            print()  # 换行

    print(f"训练完成! 最佳epoch: {best_epoch + 1}, 最佳验证指标: {best_val_metric:.4f}")

    # 绘制训练曲线
    curves_path = os.path.join(args.output_dir, 'training_curves.png')
    if args.task_mode == 'classification':
        plot_training_curves(
            train_losses, val_losses, train_metrics, val_metrics,
            'Epoch', 'Loss', 'Accuracy', curves_path
        )
    else:
        plot_training_curves(
            train_losses, val_losses, train_metrics, val_metrics,
            'Epoch', 'Loss', 'RMSE', curves_path
        )

    # 在测试集上评估最终模型
    print("在测试集上评估最终模型...")
    # 加载最佳模型
    best_path = os.path.join(args.output_dir, "best_model.pt")
    load_checkpoint(best_path, model, device=device)

    if args.task_mode == 'classification':
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device, args.task_mode)
        print(f"测试集结果: 损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")
    else:
        test_loss, test_rmse, test_r2 = evaluate(model, test_loader, loss_fn, device, args.task_mode)
        print(f"测试集结果: 损失: {test_loss:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    # 关闭TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()