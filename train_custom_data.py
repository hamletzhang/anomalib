# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""使用自定义数据集训练异常检测模型

这个示例展示了如何使用project_test_organized文件夹中的数据来训练异常检测模型。
参考了basic_training.py的基本结构，使用Folder数据模块来处理自定义数据集，
使用Patchcore模型进行训练。

数据集结构:
project_test_organized/
├── train/
│   └── normal/     # 训练用的正常样本 (189个)
└── test/
    ├── normal/     # 测试用的正常样本 (81个)
    └── abnormal/   # 测试用的异常样本 (248个)
"""

# 1. 导入所需模块
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

# 2. 创建数据集
# 使用Folder数据模块来处理重组后的自定义文件夹结构
print("配置数据模块...")
datamodule = Folder(
    name="project_test_organized",  # 数据集名称
    root="./project_test_organized",  # 数据集根目录
    normal_dir="train/normal",  # 正常样本训练目录
    normal_test_dir="test/normal",  # 正常样本测试目录
    abnormal_dir="test/abnormal",  # 异常样本测试目录
    train_batch_size=16,  # 每个训练批次的图像数量
    eval_batch_size=16,  # 每个验证/测试批次的图像数量
    num_workers=0,  # Windows上设置为0避免多进程问题
)

# 3. 初始化模型
# Patchcore 是一个优秀的异常检测模型
print("初始化Patchcore模型...")
model = Patchcore()

# 4. 创建训练引擎
print("配置训练引擎...")
engine = Engine(max_epochs=1)  # Patchcore通常只需要1个epoch

# 5. 训练模型
print("开始训练...")
try:
    # 首先设置数据模块
    datamodule.setup()
    
    # 输出数据集信息
    print(f"训练样本: {len(datamodule.train_dataloader().dataset)}个")
    print(f"测试样本: {len(datamodule.test_dataloader().dataset)}个")
    
    print("=" * 50)
    
    # 开始训练
    engine.fit(datamodule=datamodule, model=model)
    
    print("训练完成！模型已保存。")
    
    # 测试模型
    print("开始测试模型...")
    engine.test(datamodule=datamodule, model=model)
    
except Exception as e:
    print(f"训练过程中出现错误: {e}")
    print("请检查数据路径和依赖项是否正确安装。") 