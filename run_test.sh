#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2  # 根据可用GPU数量调整

# 检查CUDA可用性
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$CUDA_AVAILABLE" == "True" ]; then
    # 获取可用的GPU数量
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "检测到 $NUM_GPUS 个可用的GPU"

    if [ $NUM_GPUS -gt 1 ]; then
        # 多GPU分布式启动
        python -m torch.distributed.launch \
            --nproc_per_node=$NUM_GPUS \
            test.py \
            --config utils/config.py \
            --data_path data/train_data.pt \
            --val_path data/val_data.pt \
            --sample_size 100 \
            --batch_size 16 \
            --epochs 2
    else
        # 单GPU分布式启动
        python -m torch.distributed.launch \
            --nproc_per_node=1 \
            test.py \
            --config utils/config.py \
            --data_path data/train_data.pt \
            --val_path data/val_data.pt \
            --sample_size 100 \
            --batch_size 16 \
            --epochs 2
    fi
else
    echo "CUDA不可用，将使用CPU进行训练"
    python test.py \
        --config utils/config.py \
        --data_path data/train_data.pt \
        --val_path data/val_data.pt \
        --sample_size 100 \
        --batch_size 16 \
        --epochs 2 \
        --gpus_per_node 0
fi