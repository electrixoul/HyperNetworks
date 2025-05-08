# HyperNetworks GPU 运行指南

本指南描述了如何在远程开发机上使用 `sae_eeg` conda 环境运行 HyperNetworks 的 GPU 模式。

## 环境配置

当前远程开发机配置如下：

- **CUDA 版本**: 12.6
- **PyTorch 版本**: 2.7.0+cu126
- **GPU**: 2 × NVIDIA H100 80GB HBM3
- **cuDNN 版本**: 9.5.1
- **Conda 环境**: `sae_eeg`

## 可用脚本

本设置中包含以下脚本文件:

1. **`run_hypernetworks_gpu.sh`** - 用于在 `sae_eeg` conda 环境中运行 HyperNetworks GPU 训练的主脚本
2. **`check_gpu.py`** - 验证 GPU 环境配置的脚本

## 如何使用

### 检查 GPU 环境

在开始训练前，你可以运行以下命令来验证 GPU 环境是否配置正确：

```bash
conda run -n sae_eeg python check_gpu.py
```

这个脚本将会输出你的 PyTorch 版本、CUDA 版本和可用的 GPU 信息。

### 运行训练

使用 `run_hypernetworks_gpu.sh` 脚本可以在 `sae_eeg` 环境中运行 GPU 训练：

```bash
# 使用默认参数运行
./run_hypernetworks_gpu.sh

# 自定义批次大小和训练轮数
./run_hypernetworks_gpu.sh --batch_size 64 --epochs 100

# 使用较低的学习率
./run_hypernetworks_gpu.sh --lr 0.001

# 从之前的检查点恢复训练
./run_hypernetworks_gpu.sh --resume

# 指定保存检查点的路径
./run_hypernetworks_gpu.sh --checkpoint_path ./my_model_checkpoint.pth
```

### 可用参数

`run_hypernetworks_gpu.sh` 脚本支持以下参数：

- `--batch_size` - 训练批次大小（默认：128）
- `--epochs` - 训练轮数（默认：200）
- `--lr` - 学习率（默认：0.002）
- `--weight_decay` - 权重衰减（默认：0.0005）
- `--checkpoint_path` - 保存检查点的路径（默认：./hypernetworks_cifar_gpu.pth）
- `--resume` - 是否从检查点恢复训练（无需额外参数）

## 训练日志

训练日志将在终端中显示，包含以下信息：

- 每 50 个批次的训练损失和准确率
- 每个 epoch 结束后的训练和测试准确率
- 当前学习率
- 每个 epoch 的运行时间
- 最佳准确率和模型保存信息

## 模型结构

HyperNetworks 模型由以下核心组件组成：

1. **超网络 (HyperNetwork)** - 用于生成主网络权重的网络，在 `hypernetwork_modules.py` 中实现
2. **主网络 (PrimaryNetwork)** - 使用超网络生成的权重进行图像分类的网络，在 `primary_net.py` 中实现
3. **ResNet 块** - 使用动态生成的权重的残差网络块，在 `resnet_blocks.py` 中实现

## 目录结构

```
HyperNetworks/
├── HyperNetworks_GPU/         # GPU 优化版本的源代码
│   ├── hypernetwork_modules.py
│   ├── primary_net.py
│   ├── README.md
│   ├── resnet_blocks.py
│   └── train.py
├── run_hypernetworks_gpu.sh   # GPU 训练脚本
├── check_gpu.py               # GPU 环境检查脚本
└── README_GPU_SETUP.md        # 本文件
```

## 注意事项

1. 确保在 `sae_eeg` conda 环境中运行所有命令
2. 首次运行训练时，CIFAR-10 数据集将被自动下载到 `./data` 目录
3. 如果训练中断，可以使用 `--resume` 参数从之前的检查点继续训练
4. 当前配置已针对双 H100 GPU 进行优化，会自动使用 DataParallel 进行多 GPU 训练
