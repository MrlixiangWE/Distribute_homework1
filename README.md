# Distribute Homework 1

本仓库是“单机版 / 分布式 / 深度学习系统”多系统算法对比实验，使用 Python 实现并给出实测结果。

## 一、实验内容

### 1) 非视觉基线实验
- 脚本：`multi_system_compare.py`
- 任务：工业设备故障风格的二分类（合成数据）
- 对比系统：
  - 单机：`scikit-learn LogisticRegression`
  - 分布式：`Dask + dask-ml LogisticRegression`
  - 深度学习：`PyTorch MLP`

### 2) 视觉任务实验
- 脚本：`vision_multi_system_compare.py`
- 任务：`CIFAR-10` 图像分类
- 对比内容：
  - 三系统统一 Softmax 回归（图像特征）
  - 深度学习附加 ResNet18 迁移学习基线

### 3) ResNet18 跨系统统一对比（最终版）
- 脚本：`resnet_cross_system_compare.py`
- 任务：`CIFAR-10` 图像分类
- 三个系统均使用 `ResNet18`，仅调整系统配置与训练策略：
  - 单机系统：单进程迁移学习（冻结主干，仅训练 FC）
  - 分布式系统：`PyTorch DDP` 本机多进程模拟（可切换远程多机）
  - 深度学习系统：单进程全参数微调

## 二、核心结果（ResNet18 统一对比）

结果来源：`outputs_resnet/comparison_results.csv`

| system | algorithm | accuracy | macro_f1 | macro_auc_ovr | top3_acc | train_time_sec | infer_time_sec | peak_memory_mb | model_size_mb |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_machine_system | ResNet18 | 0.7220 | 0.7189 | 0.9568 | 0.9267 | 27.2742 | 2.8090 | 805.8789 | 42.7307 |
| distributed_system | ResNet18 | 0.6960 | 0.6958 | 0.9560 | 0.9280 | 82.7907 | 14.6165 | 2796.0508 | 42.7309 |
| deep_learning_system | ResNet18 | 0.8547 | 0.8538 | 0.9881 | 0.9753 | 35.6942 | 2.8229 | 1116.3281 | 42.7306 |

## 三、结果分析（简要）

1. **模型指标**  
深度学习系统（全参数微调）表现最佳，`accuracy=0.8547`，显著高于单机与分布式迁移学习配置。

2. **计算资源（内存）**  
分布式本机模拟内存峰值最高（约 `2796 MB`），约为单机的 `3.47x`，主要来自 DDP 多进程开销。

3. **执行时间**  
在单机模拟分布式场景下，分布式训练最慢（`82.79s`），单机迁移学习训练更快（`27.27s`）。  
深度学习系统全量微调训练时间高于单机迁移学习，但在指标上收益最大。

4. **存储资源**  
三者模型体积几乎相同（约 `42.73 MB`），因为核心结构均为 ResNet18。

## 四、目录说明

- `outputs/`：非视觉基线实验结果
- `outputs_vision/`：视觉任务综合实验结果
- `outputs_resnet/`：ResNet18 跨系统最终对比结果

每个输出目录一般包含：
- `comparison_results.csv`
- `comparison_results.json`
- `comparison_results.md`
- `installation_config.json`
- `experiment_config.json`（若该实验脚本有记录）

## 五、运行方式

```bash
python3 multi_system_compare.py
python3 vision_multi_system_compare.py
python3 resnet_cross_system_compare.py
```

## 六、注意事项

- 数据集会在运行时下载到本地，不纳入 Git 管理。
- 大模型权重文件不提交到仓库，已通过 `.gitignore` 排除。
