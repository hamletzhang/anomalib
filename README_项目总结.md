# Anomalib 异常检测项目总结

## 📋 项目概述

本项目使用 Intel 的 Anomalib 框架实现了基于 Patchcore 模型的工业异常检测系统。通过训练正常样本来学习正常模式，然后检测偏离正常模式的异常样本。

### 🎯 最终成果
- **总体准确率：87.88%** (145/165张)
- **正常样本检测：65.85%** (27/41张)
- **异常样本检测：95.16%** (118/124张)
- **模型特点：** 高异常检测敏感度，适合工业质检场景

---

## 🗂️ 文件结构

```
anomalib/
├── project_test/                    # 原始数据
│   ├── OK/                         # 原始正常样本
│   └── NG/                         # 原始异常样本
├── project_test_organized/          # 重组后数据
│   ├── train/normal/               # 训练用正常样本 (189个)
│   └── test/
│       ├── normal/                 # 测试用正常样本 (41个)
│       └── abnormal/               # 测试用异常样本 (124个)
├── reorganize_data.py              # 数据重组脚本
├── train_custom_data.py            # 训练脚本
├── inference_custom_data.py        # 推理测试脚本
└── results/                        # 训练结果和模型
    └── Patchcore/project_test_organized/v2/
        └── weights/lightning/model.ckpt
```

---

## 🔧 训练代码详解 (`train_custom_data.py`)

### 核心组件

#### 1. 数据模块配置
```python
datamodule = Folder(
    name="project_test_organized",          # 数据集名称
    root="./project_test_organized",        # 数据集根目录
    normal_dir="train/normal",              # 训练用正常样本路径
    normal_test_dir="test/normal",          # 测试用正常样本路径  
    abnormal_dir="test/abnormal",           # 测试用异常样本路径
    train_batch_size=32,                    # 训练批次大小
    eval_batch_size=32,                     # 验证批次大小
    num_workers=0,                          # Windows环境建议设为0
)
```

**关键点说明：**
- `Folder` 类是 Anomalib 处理自定义数据集的标准方式
- 必须明确指定 `normal_dir`、`normal_test_dir`、`abnormal_dir`
- Windows 环境下 `num_workers=0` 避免多进程问题

#### 2. 模型初始化
```python
model = Patchcore(
    backbone="wide_resnet50_2",             # 特征提取骨干网络
    layers=["layer2", "layer3"],           # 提取特征的层
    pre_trained=True,                       # 使用预训练权重
    num_neighbors=9,                        # 最近邻数量
)
```

#### 3. 训练引擎
```python
engine = Engine(
    max_epochs=1,                           # Patchcore只需1个epoch
    accelerator="auto",                     # 自动检测GPU/CPU
    devices=1,                              # 设备数量
)
```

### 训练流程
1. **数据准备** → 2. **模型初始化** → 3. **训练执行** → 4. **模型保存** → 5. **性能评估**

---

## 🔍 测试代码详解 (`inference_custom_data.py`)

### 核心逻辑

#### 1. 模型加载
```python
model = Patchcore()                         # 初始化相同的模型架构
ckpt_path = "results/Patchcore/project_test_organized/v2/weights/lightning/model.ckpt"
```

#### 2. 数据模块配置（与训练相同）
```python
datamodule = Folder(
    name="project_test_organized",
    root="./project_test_organized",
    normal_dir="train/normal",
    normal_test_dir="test/normal", 
    abnormal_dir="test/abnormal",
    # ... 其他参数与训练时相同
)
```

**重要：** 推理时必须使用与训练完全相同的数据模块配置！

#### 3. 推理执行
```python
predictions = engine.predict(
    model=model,
    datamodule=datamodule,                  # 使用完整数据模块
    ckpt_path=ckpt_path,                   # 模型权重路径
    return_predictions=True,                # 返回详细预测结果
)
```

#### 4. 结果分析
脚本会自动分析每个样本的预测结果，生成：
- 详细的样本级别预测结果
- 正常/异常样本的准确率统计
- 混淆矩阵
- 总体性能指标

---

## 🔄 如何替换数据集

### 方法一：替换现有数据（推荐）

1. **备份当前数据**
```bash
cp -r project_test project_test_backup
```

2. **清空现有数据文件夹**
```bash
rm -rf project_test/OK/*
rm -rf project_test/NG/*
```

3. **放入新数据**
- 将新的正常样本放入 `project_test/OK/` 文件夹
- 将新的异常样本放入 `project_test/NG/` 文件夹

4. **重新运行数据重组**
```bash
python reorganize_data.py
```

5. **重新训练**
```bash
python train_custom_data.py
```

### 方法二：创建新的数据集

1. **创建新的数据文件夹**
```bash
mkdir my_new_dataset
mkdir my_new_dataset/OK
mkdir my_new_dataset/NG
```

2. **修改脚本中的路径**

在 `reorganize_data.py` 中修改：
```python
# 修改这些路径
source_ok_dir = "my_new_dataset/OK"      # 改为新路径
source_ng_dir = "my_new_dataset/NG"     # 改为新路径
target_root = "my_new_dataset_organized" # 改为新的目标路径
```

在 `train_custom_data.py` 中修改：
```python
datamodule = Folder(
    name="my_new_dataset_organized",        # 改为新名称
    root="./my_new_dataset_organized",      # 改为新路径
    # ... 其他参数保持不变
)
```

在 `inference_custom_data.py` 中做相应修改。

### 数据格式要求

✅ **支持的图像格式：**
- PNG, JPG, JPEG, BMP, TIFF
- 脚本会自动检测并添加正确的扩展名

✅ **数据量建议：**
- 正常样本：至少50张，推荐100+张
- 异常样本：用于测试验证，数量可以灵活调整
- 训练/测试比例：建议7:3或8:2

---

## 🔧 如何更换模型

Anomalib 支持多种异常检测模型，每种模型有不同的特点：

### 1. 可选模型类型

#### 🚀 **快速模型（适合实时检测）**
```python
# EfficientAd - 快速且准确
from anomalib.models import EfficientAd
model = EfficientAd(
    teacher_out_channels=384,
    model_size="m",                         # s, m, l
    lr=1e-4,
)

# FastFlow - 超快速度
from anomalib.models import Fastflow
model = Fastflow(
    backbone="resnet18",                    # 更轻量的骨干网络
    flow_steps=8,
)
```

#### 🎯 **高精度模型（适合离线分析）**
```python
# PaDiM - 高精度，内存友好
from anomalib.models import Padim
model = Padim(
    backbone="wide_resnet50_2",
    layers=["layer1", "layer2", "layer3"],
    n_features=100,
)

# Reverse Distillation - 最新的高性能模型
from anomalib.models import ReverseDistillation
model = ReverseDistillation(
    backbone="wide_resnet50_2",
    layers=["layer1", "layer2", "layer3"],
)
```

#### 🧠 **深度学习模型**
```python
# DRAEM - 基于重构的模型
from anomalib.models import Draem
model = Draem(
    backbone="resnet18",
    beta=(0.1, 1.0),
)

# STFPM - 学生-教师框架
from anomalib.models import Stfpm
model = Stfpm(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
)
```

### 2. 更换模型的步骤

#### 步骤1：修改训练脚本
在 `train_custom_data.py` 中：
```python
# 替换这部分
# from anomalib.models import Patchcore
# model = Patchcore()

# 改为想要的模型
from anomalib.models import EfficientAd  # 例如换成EfficientAd
model = EfficientAd()
```

#### 步骤2：调整训练参数
不同模型可能需要不同的训练设置：
```python
# 对于需要多个epochs的模型
engine = Engine(max_epochs=10)           # PatchCore用1，其他可能需要更多

# 对于内存敏感的模型
datamodule = Folder(
    # ...
    train_batch_size=16,                  # 减少批次大小
    eval_batch_size=16,
    # ...
)
```

#### 步骤3：修改推理脚本
在 `inference_custom_data.py` 中：
```python
# 确保使用相同的模型
from anomalib.models import EfficientAd  # 与训练时保持一致
model = EfficientAd()
```

#### 步骤4：更新模型路径
不同模型的保存路径会不同：
```python
# 检查results文件夹中的实际路径
ckpt_path = "results/EfficientAd/project_test_organized/v1/weights/lightning/model.ckpt"
```

### 3. 模型选择建议

| 使用场景 | 推荐模型 | 特点 |
|---------|---------|------|
| 🏭 **实时工业检测** | EfficientAd, FastFlow | 速度快，资源占用少 |
| 🔬 **高精度分析** | Patchcore, PaDiM | 精度高，内存友好 |
| 🧪 **研究实验** | Reverse Distillation | 最新技术，性能优异 |
| 💻 **资源受限环境** | EfficientAd (small) | 内存和计算要求低 |

---

## 📊 性能优化建议

### 1. 数据层面
- **数据质量**：确保正常样本真正代表"正常"状态
- **数据平衡**：正常样本数量要充足，覆盖各种正常变化
- **图像质量**：保持一致的光照、角度、分辨率

### 2. 模型层面
- **超参数调优**：根据验证结果调整学习率、批次大小等
- **模型集成**：可以尝试多个模型的组合
- **阈值优化**：根据业务需求调整异常检测阈值

### 3. 部署层面
- **模型压缩**：使用量化或剪枝技术减少模型大小
- **批处理**：对多张图像同时处理提高效率
- **硬件加速**：利用GPU或专用推理硬件

---

## 🚀 快速开始指南

### 新数据集训练
1. 准备数据 → 放入 `project_test/OK` 和 `project_test/NG`
2. 运行 `python reorganize_data.py`
3. 运行 `python train_custom_data.py`
4. 运行 `python inference_custom_data.py`

### 更换模型
1. 修改 `train_custom_data.py` 中的模型导入和初始化
2. 调整训练参数（epochs, batch_size等）
3. 重新训练和测试

### 调试技巧
- 检查 `results/` 文件夹中的日志和可视化结果
- 使用小数据集快速验证流程
- 关注训练过程中的损失函数变化

---

## 🔗 相关资源

- [Anomalib 官方文档](https://anomalib.readthedocs.io/)
- [Anomalib GitHub](https://github.com/openvinotoolkit/anomalib)
- [论文：PatchCore](https://arxiv.org/abs/2106.08265)

---

*创建时间：2024年*  
*适用版本：Anomalib 1.0+* 