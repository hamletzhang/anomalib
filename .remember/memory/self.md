# 超模记忆 - 错误修正记录

## Anomalib数据集配置相关

### 自定义数据集配置
- 使用 `Folder` 数据模块而不是 `MVTecAD` 来配置自定义数据集
- 需要指定 `root`, `normal_dir`, `abnormal_dir` 参数
- 对于正常样本使用 `normal_dir=\"OK\"`，异常样本使用 `abnormal_dir=\"NG\"`

### Folder数据模块正确配置
- `Folder` 构造函数需要 `normal_dir` 参数（必需）
- 对于有独立test目录的数据集，应使用：
  - `normal_dir=\"train/normal\"` - 训练用正常样本
  - `normal_test_dir=\"test/normal\"` - 测试用正常样本  
  - `abnormal_dir=\"test/abnormal\"` - 测试用异常样本
- Windows环境下设置 `num_workers=0` 避免多进程问题

### Patchcore模型训练
- Patchcore 模型训练非常高效，通常只需要 `max_epochs=1`
- 训练成功后会显示测试结果，包括 AUROC 和 F1Score 指标
- 模型会自动保存训练结果

### 推理脚本的数据加载错误
- **重要错误**：使用 `PredictDataset` 分别处理正常和异常样本时，会导致所有样本都被混合在一起
- **错误原因**：`PredictDataset` 会加载指定目录下的所有图像，不区分子目录
- **解决方案**：需要使用训练好的数据模块进行推理，而不是单独创建 `PredictDataset`
- **正确做法**：使用 `datamodule.test_dataloader()` 进行推理，这样可以保持正确的标签信息

### 依赖项安装问题
- 在conda环境中安装anomalib依赖时，可能需要逐个安装：
  - `FrEIA` - 流归一化模型依赖
  - `python-dotenv` - 环境变量配置
  - `open_clip_torch` - CLIP模型支持
- 使用 `pip install -e .` 以开发模式安装anomalib 