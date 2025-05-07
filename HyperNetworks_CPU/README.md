# 超网络 (HyperNetworks) - CPU 版本

这是 [HyperNetworks](https://arxiv.org/abs/1609.09106) (Ha 等人，ICLR 2017) 实现的 CPU 版本。原始项目主要设计用于在 GPU 上运行，但此版本已经修改为可以在 CPU 上运行，解决了 NVIDIA 驱动版本兼容性问题。

## 修改内容

相比原始版本，CPU 版本做了以下修改：

1. **移除 CUDA 依赖**：
   - 从 `hypernetwork_modules.py` 移除 `.cuda()` 调用
   - 从 `primary_net.py` 移除 `.cuda()` 调用
   - 注释掉 `train.py` 中的 `net.cuda()`
   - 从训练和评估循环中移除 `.cuda()` 调用

2. **更新过时的 PyTorch 语法**：
   - 将 `loss.data[0]` 改为 `loss.item()`
   - 将 `.sum()` 更新为 `.sum().item()`

这些更改使得代码能够在 CPU 上运行，无需特定的 NVIDIA 驱动版本。

## 如何运行

```bash
python train.py
```

要从检查点恢复：

```bash
python train.py -r
```

## 环境要求

- PyTorch
- torchvision
- Python 3

## 注意事项

- 训练在 CPU 上比在 GPU 上慢得多
- 可以在 `train.py` 中修改 `max_iter` 值以减少训练迭代次数，从而进行快速测试

## 原始项目

这是基于 [https://github.com/g1910/HyperNetworks](https://github.com/g1910/HyperNetworks) 的修改版本。关于超网络的更多信息，请参考原始 README 和论文。
