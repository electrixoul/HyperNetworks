# HyperNetworks GPU 实现

这是 [HyperNetworks](https://arxiv.org/abs/1609.09106) (Ha 等人，ICLR 2017) 的 GPU 优化 PyTorch 实现。该实现针对现代 GPU 做了优化，用于 CIFAR-10 数据集上的 ResNet 训练。

## 与CPU实现的主要区别

1. **默认使用CUDA**：
   - 模型参数默认初始化在 GPU 上
   - 自动检测并使用可用的 GPU 资源
   - 支持多 GPU 训练（通过 DataParallel）

2. **性能优化**：
   - 使用 pin_memory 加速 CPU 到 GPU 的数据传输
   - 增加了 DataLoader 的 num_workers 提高数据加载性能
   - 优化的学习率调度策略

3. **额外功能**：
   - 更详细的训练记录和性能指标
   - 支持测试 GPU 性能的专用脚本
   - 改进的命令行参数支持

## 环境要求

- PyTorch（支持CUDA）
- torchvision
- CUDA 工具包（如果使用 GPU）
- numpy
- Python 3.6+

确保使用 `conda` 环境运行：

```bash
conda activate mod  # 使用指定的mod环境
```

## 数据集

该实现使用 CIFAR-10 数据集，包含以下类别：
- 飞机（'plane'）
- 汽车（'car'）
- 鸟（'bird'）
- 猫（'cat'）
- 鹿（'deer'）
- 狗（'dog'）
- 青蛙（'frog'）
- 马（'horse'）
- 船（'ship'）
- 卡车（'truck'）

数据集将在首次运行时自动下载到 `./data` 目录。

## 使用方法

### 训练模型

```bash
# 基本训练
python train.py

# 使用自定义参数
python train.py --batch_size 64 --lr 0.001 --epochs 100

# 从检查点恢复训练
python train.py --resume
```

### 训练参数

- `--resume`：从检查点恢复训练
- `--batch_size`：训练批次大小（默认：128）
- `--epochs`：训练轮数（默认：200）
- `--lr`：初始学习率（默认：0.002）
- `--weight_decay`：权重衰减系数（默认：0.0005）
- `--checkpoint_path`：检查点保存路径（默认：'./hypernetworks_cifar_gpu.pth'）

### 测试 GPU 性能

```bash
# 基本性能测试
python test_gpu.py

# 使用不同批次大小测试
python test_gpu.py --batch_size 32

# 指定模型路径
python test_gpu.py --model_path ./your_model_checkpoint.pth
```

### 测试参数

- `--batch_size`：测试批次大小（默认：100）
- `--model_path`：模型检查点路径（默认：'./hypernetworks_cifar_gpu.pth'）

## 模型架构

HyperNetwork 是一个网络生成网络的实现，它包含：

1. **超网络（Hypernetwork）**：
   - 用于生成主网络的权重
   - 由 `hypernetwork_modules.py` 实现

2. **主网络（Primary Network）**：
   - 使用生成的权重进行图像分类
   - 由 `primary_net.py` 实现
   - 基于 ResNet 架构

3. **ResNet 块**：
   - 使用超网络生成的权重
   - 由 `resnet_blocks.py` 实现

## 性能参考

基于 GPU 实现，在 CIFAR-10 数据集上的 ResNet 变体：

- 参数数量：约 0.148M（相比传统 Wide ResNet 40-2 的 2.2M）
- 预期测试错误率：约 7-8%
- 训练时间：因 GPU 类型而异，但显著快于 CPU 实现

## 引用

如果您使用此代码，请考虑引用原始论文和此实现：

```
@article{ha2016hypernetworks,
  title={HyperNetworks},
  author={Ha, David and Dai, Andrew and Le, Quoc V},
  journal={arXiv preprint arXiv:1609.09106},
  year={2016}
}

@article{gaurav2018hypernetsgithub,
  title={HyperNetworks(Github)},
  author={{Mittal}, G.},
  howpublished = {\url{https://github.com/g1910/HyperNetworks}},
  year={2018}
}
