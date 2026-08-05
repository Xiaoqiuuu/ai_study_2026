# CIFAR-10 ResNet18 微调项目

> 对应计划 Day 6（超前进度）—— CNN 图像分类

## 特性

- ✅ **ResNet18 预训练微调**：ImageNet → CIFAR-10 迁移学习
- ✅ **冻结策略**：默认冻结 layer1-3，只训练 layer4 + fc（省显存）
- ✅ **AMP 混合精度**：自动开启 `torch.cuda.amp`，6GB 显存友好
- ✅ **数据增强**：RandomCrop + RandomHorizontalFlip + Normalize
- ✅ **CosineAnnealingLR**：比 StepLR 更适合小数据集微调
- ✅ **Label Smoothing**：0.1，防止过拟合
- ✅ **梯度裁剪**：max_norm=1.0，稳定训练
- ✅ **同套工程规范**：和 MNIST 项目完全一致的模块化结构

## 快速开始

```bash
cd cifar10_cnn
pip install -r requirements.txt
python main.py
```

## 显存占用参考（RTX 4050 6GB）

| 配置 | 显存占用 | 单 epoch 时间 |
|------|---------|--------------|
| batch=128, AMP=ON, freeze_backbone=True | ~3.8 GB | ~35s |
| batch=128, AMP=OFF, freeze_backbone=True | ~5.2 GB | ~50s |
| batch=256, AMP=ON, freeze_backbone=True | ~5.5 GB | ~28s |

> 建议保持默认配置，留足显存余量。

## 关键代码对比（MNIST vs CIFAR-10）

| 维度 | MNIST | CIFAR-10 |
|------|-------|----------|
| 输入 | 1×28×28 (flatten→784) | 3×32×32 (保持空间结构) |
| 模型 | 自定义 MLP | ResNet18 预训练 |
| 数据增强 | 无 | RandomCrop + Flip |
| 精度优化 | 无 | AMP + Label Smoothing + Grad Clip |
| 学习率调度 | StepLR | CosineAnnealingLR |

## 你可以做的改进

- [ ] 把 ResNet18 第一层 `conv1` 从 `7x7 stride=2` 改成 `3x3 stride=1`，更适合 32x32 小图
- [ ] 加入 Cutout / RandAugment 数据增强
- [ ] 尝试不冻结 backbone（需调小 lr，batch_size 降为 64）
- [ ] 画混淆矩阵，分析哪类最容易错（飞机 vs 鸟？）
- [ ] 可视化 ResNet 的 feature map（hook 机制）
