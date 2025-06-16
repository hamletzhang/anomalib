# 超模记忆 - 项目规则和用户偏好

## 用户偏好
- 使用中文简体回复
- 参考已有示例代码进行修改，不重新创建
- 优先使用Patchcore模型进行异常检测训练
- 将逻辑与前端分离，创建模块化架构

## GUI开发经验 (新增)
- **GUI框架选择**: 使用PyQt5创建桌面应用界面
- **架构设计**: 采用三层架构
  - GUI层: `patchcore_gui.py` - 负责用户界面和交互
  - 训练层: `patchcore_trainer.py` - 封装训练逻辑，提供回调接口
  - 后端: `flexible_patchcore_train.py` - 实际的训练实现
- **线程处理**: 使用QThread防止GUI界面在训练时卡顿
- **配置管理**: 支持JSON格式的配置保存和加载
- **路径处理**: 提供文件夹浏览对话框，支持相对和绝对路径

## 项目特定规则 - 新数据集 (已更新)
- **新数据源位置**：
  - 全部OK样本：`C:\Users\Administrator\Desktop\自监督测试\ALL_OK` (890张)
  - 全部NG样本：`C:\Users\Administrator\Desktop\自监督测试\ALL_NG` (406张)
- **重组后数据**：`project_test_organized` 文件夹
  - `train/normal/` - 训练用正常样本 (623张，70%分割)
  - `test/normal/` - 测试用正常样本 (267张，30%分割) 
  - `test/abnormal/` - 测试用异常样本 (406张)
  - `test/ALL_OK/` - 全部正常样本 (890张，用于全量测试)
  - `test/ALL_NG/` - 全部异常样本 (406张，用于全量测试)

## 数据集统计 (新数据)
- **原始数据总计**: 1296张 (890 OK + 406 NG)
- **训练集**: 623张正常样本
- **测试集**: 267张正常样本 + 406张异常样本 = 673张
- **数据分割比例**: 约7:3 (训练:测试)

## 成功的配置经验
- 使用conda activate anomalib环境
- 数据重组脚本自动检测图像格式并添加正确扩展名
- 支持随机分割数据集，确保可重复性(random_seed=42)
- Patchcore模型训练配置：
  - `max_epochs=1` 
  - `num_workers=0` (Windows环境)

## GUI工具特点 (新增)
- **易用性**: 图形界面简化配置过程，无需命令行操作
- **实时反馈**: 训练日志实时显示，进度条显示训练进度
- **配置验证**: 自动检查数据路径和参数有效性
- **多种启动方式**: 
  - Windows批处理脚本 (`start_gui.bat`)
  - Python直接启动 (`run_gui.py`)
- **配置持久化**: JSON格式保存和加载训练配置
- **错误处理**: 友好的错误信息和解决建议

## GUI依赖管理 (新增)
- **主要依赖**: PyQt5, anomalib, torch, torchvision
- **安装方式**: 
  ```bash
  conda activate anomalib
  pip install PyQt5
  ```
- **环境检查**: 启动时自动检查所有必需的依赖项

## 推理脚本功能 (已更新)
- 支持两种测试模式：
  1. **部分测试数据**：使用train/test分割的测试集 (267 OK + 406 NG)
  2. **全量测试数据**：使用所有样本 (890 OK + 406 NG)
- 自动询问用户选择测试模式
- 详细的样本级别分析和统计报告
- 生成混淆矩阵和准确率指标

## 技术要点
- 推理时必须使用与训练相同的数据模块配置
- 数据重组脚本使用PIL库检测图像格式，比imghdr更准确
- 支持多种图像格式：PNG, JPG, BMP, TIFF等
- 文件命名规范：OK_N.ext, NG_N.ext, OK_test_N.ext
- GUI使用多线程避免界面冻结
- 训练器类提供回调接口实现GUI与训练逻辑的通信

## 项目文档
- 已创建详细的项目总结文档：`README_项目总结.md`
- 包含完整的训练、测试、数据替换、模型替换指南
- 包含多种异常检测模型的选择建议和配置示例
- 新增GUI使用说明：`gui/README_GUI使用说明.md`

## 后续使用指南

### 替换数据集的标准流程 (已更新)
1. 将新数据放入指定位置：
   - 正常样本 → `C:\Users\Administrator\Desktop\自监督测试\ALL_OK`
   - 异常样本 → `C:\Users\Administrator\Desktop\自监督测试\ALL_NG`
2. 运行 `python reorganize_data.py`
3. 使用GUI工具进行训练：
   - 双击 `gui/start_gui.bat`
   - 或运行 `python train_custom_data.py`
4. 运行 `python inference_custom_data.py` (选择测试模式)

### GUI使用流程 (新增)
1. **启动**: 双击 `gui/start_gui.bat` 或运行 `python gui/run_gui.py`
2. **配置**: 使用图形界面设置所有训练参数
3. **验证**: 点击"检查数据"确认配置正确
4. **训练**: 点击"开始训练"启动训练流程
5. **保存**: 可选择保存配置为JSON文件供下次使用

### 数据重组脚本功能
- 自动检测和转换图像格式
- 智能7:3分割正常样本用于训练和测试
- 创建全量测试目录支持完整评估
- 提供详细的统计信息和处理日志

### 更换模型的标准流程
1. 修改 `train_custom_data.py` 中的模型导入和初始化
2. 根据新模型调整训练参数（epochs, batch_size等）
3. 同步修改 `inference_custom_data.py` 中的模型加载
4. 更新模型权重文件路径
5. 重新训练和测试
6. 如需要，更新GUI中的模型选项

### 推荐的模型选择
- **实时检测**: EfficientAd, FastFlow
- **高精度分析**: Patchcore, PaDiM  
- **研究实验**: Reverse Distillation
- **资源受限**: EfficientAd (small)

### GUI配置建议 (新增)
- **快速测试**: resnet18 + layer2 + 0.05采样比例
- **推荐配置**: wide_resnet50_2 + layer2,layer3 + 0.1采样比例  
- **高精度**: wide_resnet50_2 + layer1,layer2,layer3 + 0.2采样比例 