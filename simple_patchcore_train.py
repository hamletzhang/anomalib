# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""简化版PatchCore训练脚本

这是一个简化版本的训练脚本，方便快速修改和使用。
只需要修改下面的配置参数就可以开始训练。
"""

from pathlib import Path
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ========================配置参数========================
# 数据路径配置
DATA_ROOT = "./project_test_organized"        # 数据集根目录
NORMAL_TRAIN_DIR = "train/normal"             # 训练正常样本目录
NORMAL_TEST_DIR = "test/normal"               # 测试正常样本目录  
ABNORMAL_TEST_DIR = "test/abnormal"           # 测试异常样本目录

# 模型配置
BACKBONE = "wide_resnet50_2"                  # 骨干网络: resnet18, resnet50, wide_resnet50_2
LAYERS = ["layer2", "layer3"]                 # 特征提取层
CORESET_SAMPLING_RATIO = 0.1                  # 核心集采样比例
NUM_NEIGHBORS = 9                             # 最近邻数量

# 训练配置
TRAIN_BATCH_SIZE = 16                         # 训练批次大小
EVAL_BATCH_SIZE = 16                          # 评估批次大小
MAX_EPOCHS = 1                                # 训练轮数（PatchCore通常只需要1轮）
NUM_WORKERS = 0                               # 数据加载工作进程数（Windows设为0）

# 输出配置
OUTPUT_DIR = "./results"                      # 输出目录
DATASET_NAME = "custom_dataset"               # 数据集名称
# =====================================================


def check_data_paths():
    """检查数据路径"""
    data_root = Path(DATA_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")
    
    paths_to_check = [
        (data_root / NORMAL_TRAIN_DIR, "训练正常样本"),
        (data_root / NORMAL_TEST_DIR, "测试正常样本"),
        (data_root / ABNORMAL_TEST_DIR, "测试异常样本"),
    ]
    
    for path, description in paths_to_check:
        if not path.exists():
            raise FileNotFoundError(f"{description}目录不存在: {path}")
        file_count = len(list(path.glob("*")))
        print(f"✓ {description}: {path} ({file_count} 个文件)")


def main():
    """主函数"""
    print("=" * 60)
    print("简化版 PatchCore 训练脚本")
    print("=" * 60)
    
    # 1. 检查数据路径
    print("\n1. 检查数据路径...")
    try:
        check_data_paths()
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("请检查配置参数中的数据路径设置")
        return
    
    # 2. 创建数据模块
    print("\n2. 配置数据模块...")
    datamodule = Folder(
        name=DATASET_NAME,
        root=DATA_ROOT,
        normal_dir=NORMAL_TRAIN_DIR,
        normal_test_dir=NORMAL_TEST_DIR,
        abnormal_dir=ABNORMAL_TEST_DIR,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    print(f"✓ 数据模块配置完成")
    
    # 3. 初始化模型
    print("\n3. 初始化PatchCore模型...")
    print(f"   骨干网络: {BACKBONE}")
    print(f"   特征层: {LAYERS}")
    print(f"   核心集采样比例: {CORESET_SAMPLING_RATIO}")
    print(f"   最近邻数量: {NUM_NEIGHBORS}")
    
    model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        pre_trained=True,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        num_neighbors=NUM_NEIGHBORS,
    )
    print(f"✓ 模型初始化完成")
    
    # 4. 创建训练引擎
    print("\n4. 配置训练引擎...")
    print(f"   最大轮数: {MAX_EPOCHS}")
    print(f"   输出目录: {OUTPUT_DIR}")
    
    engine = Engine(
        max_epochs=MAX_EPOCHS,
        enable_checkpointing=True,
        default_root_dir=OUTPUT_DIR,
        accelerator="auto",
        devices=1,
    )
    print(f"✓ 训练引擎配置完成")
    
    # 5. 开始训练
    print("\n5. 开始训练...")
    try:
        # 设置数据模块
        datamodule.setup()
        
        # 显示数据集信息
        train_size = len(datamodule.train_dataloader().dataset)
        test_size = len(datamodule.test_dataloader().dataset)
        print(f"   训练样本数: {train_size}")
        print(f"   测试样本数: {test_size}")
        
        # 开始训练
        print("\n开始训练...")
        engine.fit(datamodule=datamodule, model=model)
        print("\n✅ 训练完成！")
        
        # 6. 测试模型
        print("\n6. 开始测试模型...")
        test_results = engine.test(datamodule=datamodule, model=model)
        
        print("\n✅ 测试完成！")
        print("\n📊 测试结果:")
        if test_results:
            for result in test_results:
                for key, value in result.items():
                    print(f"   {key}: {value:.4f}")
        
        print(f"\n📁 结果保存在: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("请检查数据路径和依赖项是否正确安装。")
        raise


if __name__ == "__main__":
    print(__doc__)
    main() 