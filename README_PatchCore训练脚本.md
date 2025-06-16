# PatchCore 训练脚本使用说明

本文档说明如何使用提供的脚本来训练 PatchCore 异常检测模型。

## 文件说明

### 主要脚本

1. **`flexible_patchcore_train.py`** - 灵活的命令行训练脚本
   - 支持命令行参数配置
   - 可以自定义所有训练参数
   - 适合自动化和批量训练

2. **`simple_patchcore_train.py`** - 简化版训练脚本  
   - 通过修改脚本顶部的配置参数来设置
   - 易于理解和快速修改
   - 适合初学者和快速测试

3. **`training_examples.py`** - 使用示例脚本
   - 展示如何使用灵活训练脚本
   - 包含多种配置示例
   - 可以交互式运行示例

### 辅助文件

- **`train_custom_data.py`** - 原始训练脚本
- **`reorganize_data.py`** - 数据重组脚本

## 数据集结构要求

训练脚本期望以下数据集结构：

```
数据集目录/
├── train/
│   └── normal/          # 训练用正常样本
│       ├── image1.png
│       ├── image2.png
│       └── ...
└── test/
    ├── normal/          # 测试用正常样本  
    │   ├── image1.png
    │   └── ...
    └── abnormal/        # 测试用异常样本
        ├── defect1.png
        ├── defect2.png
        └── ...
```

## 使用方法

### 方法1：使用简化版脚本（推荐新手）

1. 打开 `simple_patchcore_train.py`
2. 修改顶部的配置参数：

```python
# 数据路径配置
DATA_ROOT = "./your_dataset"              # 改为你的数据集路径
NORMAL_TRAIN_DIR = "train/normal"         # 训练正常样本目录
NORMAL_TEST_DIR = "test/normal"           # 测试正常样本目录  
ABNORMAL_TEST_DIR = "test/abnormal"       # 测试异常样本目录

# 模型配置
BACKBONE = "wide_resnet50_2"              # 骨干网络
LAYERS = ["layer2", "layer3"]             # 特征提取层
```

3. 运行脚本：

```bash
python simple_patchcore_train.py
```

### 方法2：使用灵活脚本（推荐进阶用户）

#### 基础使用（使用默认参数）

```bash
python flexible_patchcore_train.py
```

#### 自定义数据路径

```bash
python flexible_patchcore_train.py \
    --data_root ./my_dataset \
    --normal_train_dir train/good \
    --normal_test_dir test/good \
    --abnormal_test_dir test/defect
```

#### 自定义模型配置

```bash
python flexible_patchcore_train.py \
    --backbone resnet18 \
    --layers layer1 layer2 layer3 \
    --coreset_sampling_ratio 0.05 \
    --num_neighbors 5
```

#### 完整自定义示例

```bash
python flexible_patchcore_train.py \
    --data_root ./project_test_organized \
    --backbone wide_resnet50_2 \
    --layers layer2 layer3 \
    --coreset_sampling_ratio 0.1 \
    --train_batch_size 16 \
    --output_dir ./patchcore_results \
    --dataset_name my_dataset
```

### 方法3：运行示例

```bash
python training_examples.py
```

这会显示帮助信息和可用的示例，你可以选择运行其中任何一个。

## 参数说明

### 数据路径参数

- `--data_root`: 数据集根目录路径
- `--normal_train_dir`: 训练用正常样本目录（相对于data_root）
- `--normal_test_dir`: 测试用正常样本目录（相对于data_root）
- `--abnormal_test_dir`: 测试用异常样本目录（相对于data_root）

### 模型参数

- `--backbone`: 特征提取骨干网络
  - 选项：`resnet18`, `resnet50`, `wide_resnet50_2`
  - 推荐：`wide_resnet50_2`（精度更高但速度较慢）
- `--layers`: 用于特征提取的层
  - 默认：`["layer2", "layer3"]`
  - 可选：`layer1`, `layer2`, `layer3`, `layer4`
- `--coreset_sampling_ratio`: 核心集采样比例（0.0-1.0）
  - 默认：`0.1`
  - 较小值：训练更快但可能精度降低
- `--num_neighbors`: 最近邻数量
  - 默认：`9`

### 训练参数

- `--train_batch_size`: 训练批次大小（默认：16）
- `--eval_batch_size`: 评估批次大小（默认：16）
- `--max_epochs`: 最大训练轮数（默认：1，PatchCore通常只需要1轮）
- `--num_workers`: 数据加载工作进程数（Windows推荐设为0）

### 输出参数

- `--output_dir`: 输出目录（默认：`./results`）
- `--dataset_name`: 数据集名称（默认：`custom_dataset`）

## 模型配置建议

### 快速测试

```python
BACKBONE = "resnet18"
LAYERS = ["layer2"]
CORESET_SAMPLING_RATIO = 0.05
```

### 平衡配置（推荐）

```python
BACKBONE = "wide_resnet50_2"  
LAYERS = ["layer2", "layer3"]
CORESET_SAMPLING_RATIO = 0.1
```

### 高精度配置

```python
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer1", "layer2", "layer3"]
CORESET_SAMPLING_RATIO = 0.2
```

## 常见问题

### 1. 数据路径错误

**错误信息**: `FileNotFoundError: 数据根目录不存在`

**解决方法**: 检查数据路径是否正确，确保目录存在。

### 2. 内存不足

**错误信息**: `CUDA out of memory` 或类似内存错误

**解决方法**: 
- 减小批次大小：`--train_batch_size 8 --eval_batch_size 8`
- 使用更小的骨干网络：`--backbone resnet18`
- 减少特征层：`--layers layer2`

### 3. 训练很慢

**解决方法**:
- 使用更小的核心集采样比例：`--coreset_sampling_ratio 0.05`
- 使用更小的骨干网络：`--backbone resnet18`
- 减少特征层数量

### 4. 依赖项错误

**解决方法**: 确保已正确安装anomalib和相关依赖：

```bash
conda activate anomalib
pip install -e .
```

## 输出文件

训练完成后，以下文件会保存在输出目录：

- `checkpoints/`: 模型检查点文件
- `logs/`: 训练日志
- `results/`: 测试结果和可视化
- `config.yaml`: 训练配置文件

## 下一步

训练完成后，你可以：

1. 使用 `inference_custom_data.py` 进行推理
2. 检查保存的模型检查点
3. 分析测试结果和性能指标
4. 调整参数进行进一步优化

## 示例命令总结

```bash
# 快速开始
python simple_patchcore_train.py

# 查看帮助
python flexible_patchcore_train.py --help

# 运行示例
python training_examples.py

# 自定义训练
python flexible_patchcore_train.py \
    --data_root ./my_data \
    --backbone wide_resnet50_2 \
    --output_dir ./my_results
``` 