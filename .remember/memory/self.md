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

## PatchCore推理引擎错误修正

### InferenceBatch对象访问错误

**Mistake**: 将模型推理结果当作字典处理

Wrong: 
```python
anomaly_score = predictions["anomaly_score"].cpu().numpy()[0]
anomaly_map = predictions["anomaly_map"].cpu().numpy()[0]
pred_label = predictions["pred_label"].cpu().numpy()[0]
```

Correct:
```python
# InferenceBatch对象需要用属性访问
anomaly_score = predictions.pred_score.cpu().numpy()
if anomaly_score.ndim > 0:
    anomaly_score = anomaly_score[0] if len(anomaly_score) > 0 else 0.0

anomaly_map = predictions.anomaly_map.cpu().numpy()[0]

# pred_label可能不存在，需要检查
if hasattr(predictions, 'pred_label'):
    pred_label = predictions.pred_label.cpu().numpy()
    if pred_label.ndim > 0:
        pred_label = pred_label[0] if len(pred_label) > 0 else 0
else:
    pred_label = int(anomaly_score > threshold)
```

### 正常图片异常检测问题

**Problem**: 训练好的模型将正常图片也识别为异常（分数1.0）

**Analysis**: 
1. 可能模型训练不充分或训练数据质量问题
2. 默认阈值0.5可能不适合实际数据分布
3. 需要检查模型训练过程和验证指标

**Solution**: 
- 检查模型训练日志和指标
- 基于验证集数据动态调整异常检测阈值
- 验证训练数据集的质量和标注正确性

## GUI界面改进问题

### 字体显示问题
- **问题**: GUI界面字体过小，特别是文本框中的字体
- **解决方案**: 增大字体大小至少一倍，改善整体可读性
- **涉及组件**: QTextEdit日志框、QLineEdit输入框、QLabel标签等

### 训练进度反馈问题
- **问题**: 训练过程缺乏详细进度信息和时间估计
- **当前状况**: 进度条简单，缺乏细粒度更新
- **需要改进**:
  - 添加预期时间估计
  - 更详细的训练步骤日志
  - 实时的进度百分比显示
  - 训练各阶段的时间统计

### 推理界面需求
- **需求**: 创建独立的推理界面模块
- **功能要求**:
  - 单图片预测功能
  - 文件夹批量预测功能
  - 可视化结果展示（热力图、分割图等）
  - 前后端分离架构
  - 在窗口中直接显示预测结果图像
- **参考**: 使用results/images中生成的可视化图片作为展示样式 