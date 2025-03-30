#!/bin/bash
# 蛋白质图-序列多模态融合训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,3  # 使用的GPU设备
export OMP_NUM_THREADS=20  # 设置OpenMP线程数

# 训练参数
CONFIG_FILE="utils/config.py"  # 配置文件路径
OUTPUT_DIR="outputs/multimodal"  # 输出目录
NUM_GPUS=2  # GPU数量
LOG_FILE="${OUTPUT_DIR}/train_log_$(date '+%Y%m%d_%H%M%S').log"  # 日志文件



# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/models"
mkdir -p "${OUTPUT_DIR}/results"

# 日志函数
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1" | tee -a "${LOG_FILE}"
}

# 单GPU训练
run_single() {
    log "启动单GPU训练..."
    python train_embed.py \
        --config $CONFIG_FILE \
        --local_rank -1 \
        2>&1 | tee -a "${LOG_FILE}"
}

# 多GPU训练 - 使用torchrun替代torch.distributed.launch
run_multi() {
    log "启动 $NUM_GPUS GPU分布式训练..."

    # 使用torchrun（推荐方式）
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$(( 29500 + RANDOM % 1000 )) \
        train_embed.py \
        --config $CONFIG_FILE \
        2>&1 | tee -a "${LOG_FILE}"

    # 如果torchrun不可用或出错，回退到原始方法并修正参数格式
    if [ $? -ne 0 ]; then
        log "torchrun失败，尝试使用torch.distributed.launch替代方案..."
        python -m torch.distributed.launch \
            --nproc_per_node=$NUM_GPUS \
            --master_port=$(( 29500 + RANDOM % 1000 )) \
            train_embed.py \
            --config $CONFIG_FILE \
            --local_rank=\$LOCAL_RANK \
            2>&1 | tee -a "${LOG_FILE}"
    fi
}

# 多GPU训练 - 原始分布式启动方法（备用）
run_multi_legacy() {
    log "启动 $NUM_GPUS GPU分布式训练（原始方法）..."
    PYTHONPATH=$(pwd) python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$(( 29500 + RANDOM % 1000 )) \
        train_embed.py \
        --config $CONFIG_FILE \
        2>&1 | tee -a "${LOG_FILE}"
}

# 测试模式
run_test() {
    log "启动测试模式..."
    python train_embed.py \
        --config $CONFIG_FILE \
        --test_only \
        --resume "${OUTPUT_DIR}/models/checkpoint_best.pth" \
        2>&1 | tee -a "${LOG_FILE}"
}

# 分析嵌入
analyze_embeddings() {
    log "分析嵌入..."
    # 查找最新的嵌入文件
    LATEST_EMBED_DIR=$(find "${OUTPUT_DIR}/results" -name "embeddings_epoch_*" -type d | sort -r | head -n 1)

    if [ -z "$LATEST_EMBED_DIR" ]; then
        log "错误: 未找到嵌入文件目录"
        exit 1
    fi

    EMBED_FILE="${LATEST_EMBED_DIR}/embeddings.npz"

    if [ ! -f "$EMBED_FILE" ]; then
        log "错误: 未找到嵌入文件: $EMBED_FILE"
        exit 1
    fi

    log "使用嵌入文件: $EMBED_FILE"

    python visualize_embeddings.py \
        --embeddings $EMBED_FILE \
        --output_dir "${OUTPUT_DIR}/analysis" \
        --methods "tsne,pca,umap" \
        --subsample 2000 \
        2>&1 | tee -a "${LOG_FILE}"
}

# 检查训练环境
check_environment() {
    log "检查训练环境..."
    log "Python版本: $(python --version 2>&1)"
    log "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
    log "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
    log "CUDA设备数量: $(python -c 'import torch; print(torch.cuda.device_count())')"
    log "CUDA设备列表:"
    for i in $(seq 0 $(($(python -c 'import torch; print(torch.cuda.device_count())') - 1))); do
        log "  GPU $i: $(python -c "import torch; print(torch.cuda.get_device_name($i))")"
    done
}

# 清理工作
cleanup() {
    log "执行清理工作..."
    # 删除临时文件等
}

# 捕获退出信号
trap cleanup EXIT

# 根据参数执行不同操作
case "$1" in
    "single")
        check_environment
        run_single
        ;;
    "multi")
        check_environment
        run_multi
        ;;
    "multi_legacy")
        check_environment
        run_multi_legacy
        ;;
    "test")
        check_environment
        run_test
        ;;
    "analyze")
        analyze_embeddings
        ;;
    "env")
        check_environment
        ;;
    *)
        echo "用法: $0 {single|multi|multi_legacy|test|analyze|env}"
        echo "  single: 单GPU训练"
        echo "  multi: 多GPU训练（使用torchrun，推荐）"
        echo "  multi_legacy: 多GPU训练（使用torch.distributed.launch，备用方法）"
        echo "  test: 测试模式"
        echo "  analyze: 分析嵌入"
        echo "  env: 仅检查训练环境"
        exit 1
esac

log "完成！"