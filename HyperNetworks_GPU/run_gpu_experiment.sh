#!/bin/bash

# 确保脚本中的错误会导致整个脚本停止执行
set -e

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: 没有找到conda命令，请确保已安装Anaconda或Miniconda"
    exit 1
fi

# 显示当前激活的conda环境
echo "当前conda环境: $(conda info --envs | grep '*' | awk '{print $1}')"

# 检查mod环境是否存在
if ! conda env list | grep -q "mod"; then
    echo "错误: 没有找到名为'mod'的conda环境，请先创建它"
    exit 1
fi

# 获取脚本参数
MODE=${1:-"test"}  # 默认为test模式，可选: test, train
BATCH_SIZE=${2:-128}  # 默认批次大小为128

echo "==================================================================="
echo "  HyperNetworks GPU 实验 - 运行在'mod'环境中"
echo "==================================================================="
echo "模式: $MODE"
echo "批次大小: $BATCH_SIZE"
echo "==================================================================="

# 进行实验
# 使用eval来确保conda activate正常工作（在非交互式shell中需要这样做）
echo "激活conda mod环境..."
eval "$(conda shell.bash hook)"
conda activate mod

# 进入GPU目录
cd "$(dirname "$0")"
echo "当前工作目录: $(pwd)"

# 检查CUDA是否可用
echo "检查CUDA可用性..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'当前GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

if [ "$MODE" = "train" ]; then
    # 训练模式
    echo "开始训练模式，批次大小: $BATCH_SIZE"
    python train.py --batch_size $BATCH_SIZE
elif [ "$MODE" = "test" ]; then
    # 测试模式
    echo "开始测试模式，批次大小: $BATCH_SIZE"
    python test_gpu.py --batch_size $BATCH_SIZE
else
    echo "无效模式: $MODE，请使用'train'或'test'"
    exit 1
fi

echo "==================================================================="
echo "  实验完成"
echo "==================================================================="

# 回到原始目录
cd - > /dev/null
