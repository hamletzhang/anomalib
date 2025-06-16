#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchCore训练脚本使用示例

这个文件展示了如何使用 flexible_patchcore_train.py 脚本进行训练。
包含不同的配置示例和使用场景。
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """运行命令并打印结果"""
    print(f"\n{'='*60}")
    print(f"运行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("标准输出:")
        print(result.stdout)
        if result.stderr:
            print("标准错误:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        print("标准输出:")
        print(e.stdout)
        print("标准错误:")
        print(e.stderr)
        return False


def example_1_basic_training():
    """示例1: 基础训练（使用默认参数）"""
    cmd = [
        sys.executable, "flexible_patchcore_train.py",
    ]
    return run_command(cmd, "基础训练（使用默认参数）")


def example_2_custom_data_paths():
    """示例2: 自定义数据路径"""
    cmd = [
        sys.executable, "flexible_patchcore_train.py",
        "--data_root", "./project_test_organized",
        "--normal_train_dir", "train/normal",
        "--normal_test_dir", "test/normal", 
        "--abnormal_test_dir", "test/abnormal",
        "--dataset_name", "project_test",
    ]
    return run_command(cmd, "自定义数据路径")


def example_3_custom_model_config():
    """示例3: 自定义模型配置"""
    cmd = [
        sys.executable, "flexible_patchcore_train.py",
        "--backbone", "resnet18",
        "--layers", "layer1", "layer2", "layer3",
        "--coreset_sampling_ratio", "0.05",
        "--num_neighbors", "5",
    ]
    return run_command(cmd, "自定义模型配置（ResNet18）")


def example_4_custom_training_params():
    """示例4: 自定义训练参数"""
    cmd = [
        sys.executable, "flexible_patchcore_train.py",
        "--train_batch_size", "32",
        "--eval_batch_size", "32",
        "--output_dir", "./custom_results",
    ]
    return run_command(cmd, "自定义训练参数")


def example_5_complete_custom():
    """示例5: 完全自定义配置"""
    cmd = [
        sys.executable, "flexible_patchcore_train.py",
        "--data_root", "./project_test_organized",
        "--normal_train_dir", "train/normal",
        "--normal_test_dir", "test/normal",
        "--abnormal_test_dir", "test/abnormal",
        "--backbone", "wide_resnet50_2",
        "--layers", "layer2", "layer3",
        "--coreset_sampling_ratio", "0.1",
        "--num_neighbors", "9",
        "--train_batch_size", "16", 
        "--eval_batch_size", "16",
        "--output_dir", "./patchcore_results",
        "--dataset_name", "custom_anomaly_detection",
    ]
    return run_command(cmd, "完全自定义配置")


def show_help():
    """显示帮助信息"""
    cmd = [sys.executable, "flexible_patchcore_train.py", "--help"]
    return run_command(cmd, "显示帮助信息")


def main():
    """主函数"""
    print("PatchCore训练脚本使用示例")
    print("=" * 60)
    
    # 检查脚本是否存在
    if not Path("flexible_patchcore_train.py").exists():
        print("错误: 找不到 flexible_patchcore_train.py 脚本")
        print("请确保脚本在当前目录中")
        return
    
    # 显示帮助信息
    print("\n1. 首先查看帮助信息:")
    show_help()
    
    print("\n\n" + "="*60)
    print("可用的训练示例:")
    print("="*60)
    
    examples = [
        ("基础训练", example_1_basic_training),
        ("自定义数据路径", example_2_custom_data_paths),
        ("自定义模型配置", example_3_custom_model_config), 
        ("自定义训练参数", example_4_custom_training_params),
        ("完全自定义配置", example_5_complete_custom),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\n选择一个示例运行 (1-5), 或按 Enter 跳过:")
    choice = input().strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(examples):
        _, func = examples[int(choice) - 1]
        success = func()
        if success:
            print("\n✅ 示例运行成功!")
        else:
            print("\n❌ 示例运行失败!")
    else:
        print("跳过示例运行")
    
    print("\n\n" + "="*60)
    print("常用命令示例:")
    print("="*60)
    print("""
# 基础训练（使用默认参数）
python flexible_patchcore_train.py

# 指定不同的数据路径
python flexible_patchcore_train.py \\
    --data_root ./my_dataset \\
    --normal_train_dir train/good \\
    --normal_test_dir test/good \\
    --abnormal_test_dir test/defect

# 使用不同的骨干网络
python flexible_patchcore_train.py \\
    --backbone resnet18 \\
    --layers layer1 layer2

# 调整训练参数
python flexible_patchcore_train.py \\
    --train_batch_size 32 \\
    --eval_batch_size 32 \\
    --output_dir ./results

# 完整示例
python flexible_patchcore_train.py \\
    --data_root ./project_test_organized \\
    --backbone wide_resnet50_2 \\
    --layers layer2 layer3 \\
    --coreset_sampling_ratio 0.1 \\
    --train_batch_size 16 \\
    --output_dir ./patchcore_results \\
    --dataset_name my_dataset
""")


if __name__ == "__main__":
    main() 