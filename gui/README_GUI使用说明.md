# PatchCore训练GUI使用说明

这是一个基于PyQt5的图形界面，用于方便地配置和运行PatchCore异常检测模型训练。

## 文件结构

```
gui/
├── patchcore_gui.py          # GUI主程序
├── patchcore_trainer.py      # 训练逻辑封装
├── flexible_patchcore_train.py  # 原始训练脚本
├── run_gui.py               # GUI启动脚本
├── start_gui.bat            # Windows批处理启动脚本
├── requirements.txt         # 依赖项列表
└── README_GUI使用说明.md     # 本文档
```

## 安装和启动

### 方法1：使用批处理脚本（推荐Windows用户）

1. 双击 `start_gui.bat` 文件
2. 脚本会自动激活anaconda环境并启动GUI

### 方法2：手动启动

1. 打开终端并切换到gui目录：
   ```bash
   cd gui
   ```

2. 激活anomalib环境：
   ```bash
   conda activate anomalib
   ```

3. 安装PyQt5（如果未安装）：
   ```bash
   pip install PyQt5
   ```

4. 启动GUI：
   ```bash
   python run_gui.py
   ```

## 界面功能

### 1. 数据集配置

- **数据根目录**: 选择包含训练和测试数据的根目录
- **训练正常样本**: 指定训练用正常样本的相对路径（如 `train/normal`）
- **测试正常样本**: 指定测试用正常样本的相对路径（如 `test/normal`）
- **测试异常样本**: 指定测试用异常样本的相对路径（如 `test/abnormal`）
- **数据集名称**: 为数据集指定一个名称

### 2. 模型配置

- **骨干网络**: 选择特征提取的骨干网络
  - `wide_resnet50_2`: 高精度，较慢（推荐）
  - `resnet18`: 快速，较低精度
  - `resnet50`: 平衡选择
  
- **特征层**: 选择用于特征提取的网络层
  - 默认选择：`layer2` 和 `layer3`
  - 可选：`layer1`, `layer2`, `layer3`, `layer4`
  
- **核心集采样比例**: 控制训练效率和精度的平衡（0.01-1.0）
  - 较小值：训练更快，可能精度稍低
  - 较大值：训练较慢，精度更高
  
- **最近邻数量**: K-近邻算法的邻居数量（1-50）

### 3. 训练配置

- **训练批次大小**: 每个训练批次的样本数量
- **评估批次大小**: 每个评估批次的样本数量
- **最大训练轮数**: PatchCore通常只需要1轮
- **工作进程数**: 数据加载的并行进程数（Windows推荐设为0）

### 4. 输出配置

- **输出目录**: 训练结果保存的目录

### 5. 操作控制

- **加载配置**: 从JSON文件加载之前保存的配置
- **保存配置**: 将当前配置保存为JSON文件
- **检查数据**: 验证数据路径是否正确和存在
- **开始训练**: 启动训练流程
- **停止训练**: 中断正在进行的训练

### 6. 训练日志

- 实时显示训练过程中的日志信息
- 进度条显示训练进度
- 清除日志按钮清空显示内容

## 使用步骤

### 快速开始

1. 启动GUI：双击 `start_gui.bat`
2. 检查默认配置是否符合需求
3. 点击"检查数据"确认数据路径正确
4. 点击"开始训练"启动训练流程

### 详细配置

1. **配置数据路径**：
   - 点击"数据根目录"旁的"浏览"按钮选择数据目录
   - 或直接输入路径（如 `../project_test_organized`）
   - 确认子目录路径正确

2. **选择模型参数**：
   - 根据需求选择骨干网络
   - 勾选要使用的特征层
   - 调整核心集采样比例和最近邻数量

3. **设置训练参数**：
   - 根据内存大小调整批次大小
   - 设置输出目录

4. **保存配置**（可选）：
   - 点击"保存配置"将设置保存为JSON文件
   - 方便下次使用相同配置

5. **开始训练**：
   - 点击"检查数据"确认一切就绪
   - 点击"开始训练"启动训练

## 配置建议

### 快速测试配置
```
骨干网络: resnet18
特征层: layer2
核心集采样比例: 0.05
批次大小: 16
```

### 推荐配置
```
骨干网络: wide_resnet50_2
特征层: layer2, layer3
核心集采样比例: 0.1
批次大小: 16
```

### 高精度配置
```
骨干网络: wide_resnet50_2
特征层: layer1, layer2, layer3
核心集采样比例: 0.2
批次大小: 8
```

## 配置文件

GUI支持配置的保存和加载功能。配置文件为JSON格式，包含所有训练参数：

```json
{
    "data_root": "./project_test_organized",
    "normal_train_dir": "train/normal",
    "normal_test_dir": "test/normal",
    "abnormal_test_dir": "test/abnormal",
    "dataset_name": "custom_dataset",
    "backbone": "wide_resnet50_2",
    "layers": ["layer2", "layer3"],
    "coreset_sampling_ratio": 0.1,
    "num_neighbors": 9,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "max_epochs": 1,
    "num_workers": 0,
    "output_dir": "./results"
}
```

## 常见问题

### 1. GUI无法启动

**可能原因**: 缺少PyQt5依赖
**解决方法**: 
```bash
conda activate anomalib
pip install PyQt5
```

### 2. 训练失败

**可能原因**: 数据路径不正确或anomalib环境问题
**解决方法**: 
- 使用"检查数据"功能验证路径
- 确认已激活anomalib环境
- 检查日志中的具体错误信息

### 3. 内存不足

**解决方法**: 
- 减小批次大小（如改为8）
- 使用更小的骨干网络（如resnet18）
- 减少特征层数量

### 4. 训练很慢

**解决方法**: 
- 使用更小的核心集采样比例（如0.05）
- 选择更快的骨干网络（如resnet18）

## 技术架构

- **GUI层**: `patchcore_gui.py` - 负责用户界面和交互
- **训练层**: `patchcore_trainer.py` - 封装训练逻辑
- **后端**: `flexible_patchcore_train.py` - 实际的训练实现

这种分层架构确保了界面逻辑与训练逻辑的分离，便于维护和扩展。

## 输出结果

训练完成后，结果会保存在指定的输出目录中：
- `checkpoints/`: 模型检查点文件
- `logs/`: 训练日志文件  
- `results/`: 测试结果和可视化
- `config.yaml`: 训练配置文件

训练完成后可以使用推理脚本对新数据进行异常检测。 