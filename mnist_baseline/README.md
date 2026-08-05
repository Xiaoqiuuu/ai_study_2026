# MNIST 工业级训练框架

> 对应计划 Day 5 工程化要求 + Day 8-9 PyTorch 基础

## 特性

- ✅ **配置驱动**：`config.yaml` 管理所有超参，命令行可覆盖
- ✅ **可复现性**：固定 seed + deterministic mode
- ✅ **模块化**：dataset / model / trainer / utils 分离
- ✅ **日志系统**：文件 + 控制台双输出，非 print
- ✅ **TensorBoard**：自动记录 loss/accuracy 曲线
- ✅ **Checkpoint**：自动保存最佳模型，支持早停
- ✅ **类型注解 + Docstring**：符合 PEP 规范
- ✅ **Kaiming 初始化**：适合 ReLU 网络的参数初始化

## 快速开始

```bash
# 1. 进入项目目录
cd mnist_baseline

# 2. 安装依赖
pip install -r requirements.txt

# 3. 直接训练（自动下载 MNIST 到 ./data）
python main.py

# 4. 命令行覆盖超参
python main.py --epochs 30 --lr 0.001 --batch_size 256

# 5. 仅评估（加载 best_model.pt）
python main.py --eval_only

# 6. 查看 TensorBoard
tensorboard --logdir=./runs
```

## 项目结构说明

| 文件 | 职责 |
|------|------|
| `config.yaml` | 所有超参数集中管理 |
| `main.py` | 入口：解析参数 → 加载数据 → 构建模型 → 启动训练 |
| `src/dataset.py` | 数据预处理、归一化、flatten、DataLoader |
| `src/model.py` | FeedForwardNet：Linear + BN + ReLU + Dropout |
| `src/trainer.py` | 训练/验证/测试循环 + 早停 + Checkpoint |
| `src/utils.py` | seed、logger、device 选择、配置加载 |

## 关键设计决策

1. **为什么用 AdamW 而非 SGD？**
   - AdamW 解耦了权重衰减和梯度更新，默认首选优化器。

2. **为什么加 BatchNorm？**
   - 加速收敛，允许更大学习率，有轻微正则化效果。

3. **为什么 val_ratio=0.1？**
   - MNIST 训练集 60k，取 6k 做验证足够统计显著。

4. **早停 patience=5 的含义？**
   - 连续 5 个 epoch 验证集不提升就停止，防止过拟合。

## 你可以做的改进（作为练习）

- [ ] 把网络换成 CNN（LeNet-5），对比全连接 vs CNN 参数量和准确率
- [ ] 加入学习率 warmup
- [ ] 加入混淆矩阵可视化
- [ ] 用 argparse 支持选择优化器（adamw / sgd）
