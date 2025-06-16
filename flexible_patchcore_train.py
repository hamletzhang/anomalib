# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""灵活的PatchCore训练脚本

这个脚本允许你方便地指定训练集和测试集的位置，使用PatchCore模型进行异常检测训练。
支持多种数据组织方式和灵活的参数配置。
"""

import argparse
import os
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用PatchCore模型训练异常检测")
    
    # 数据路径参数
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="./project_test_organized",
        help="数据集根目录路径"
    )
    parser.add_argument(
        "--normal_train_dir", 
        type=str, 
        default="train/normal",
        help="训练用正常样本目录（相对于data_root）"
    )
    parser.add_argument(
        "--normal_test_dir", 
        type=str, 
        default="test/normal",
        help="测试用正常样本目录（相对于data_root）"
    )
    parser.add_argument(
        "--abnormal_test_dir", 
        type=str, 
        default="test/abnormal",
        help="测试用异常样本目录（相对于data_root）"
    )
    
    # 模型参数
    parser.add_argument(
        "--backbone", 
        type=str, 
        default="wide_resnet50_2",
        choices=["resnet18", "resnet50", "wide_resnet50_2"],
        help="特征提取骨干网络"
    )
    parser.add_argument(
        "--layers", 
        type=str, 
        nargs="+",
        default=["layer2", "layer3"],
        help="用于特征提取的层"
    )
    parser.add_argument(
        "--coreset_sampling_ratio", 
        type=float, 
        default=0.1,
        help="核心集采样比例"
    )
    parser.add_argument(
        "--num_neighbors", 
        type=int, 
        default=9,
        help="最近邻数量"
    )
    
    # 训练参数
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=16,
        help="训练批次大小"
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=16,
        help="评估批次大小"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=0,
        help="数据加载器工作进程数"
    )
    parser.add_argument(
        "--max_epochs", 
        type=int, 
        default=1,
        help="最大训练轮数（PatchCore通常只需要1轮）"
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="输出目录"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="custom_dataset",
        help="数据集名称"
    )
    
    return parser.parse_args()


def check_data_paths(args):
    """检查数据路径是否存在"""
    data_root = Path(args.data_root)
    
    # 检查根目录
    if not data_root.exists():
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")
    
    # 检查各个子目录
    normal_train_path = data_root / args.normal_train_dir
    normal_test_path = data_root / args.normal_test_dir
    abnormal_test_path = data_root / args.abnormal_test_dir
    
    if not normal_train_path.exists():
        raise FileNotFoundError(f"训练正常样本目录不存在: {normal_train_path}")
    
    if not normal_test_path.exists():
        raise FileNotFoundError(f"测试正常样本目录不存在: {normal_test_path}")
    
    if not abnormal_test_path.exists():
        raise FileNotFoundError(f"测试异常样本目录不存在: {abnormal_test_path}")
    
    # 输出目录信息
    print(f"数据根目录: {data_root}")
    print(f"训练正常样本: {normal_train_path} ({len(list(normal_train_path.glob('*')))} 个文件)")
    print(f"测试正常样本: {normal_test_path} ({len(list(normal_test_path.glob('*')))} 个文件)")
    print(f"测试异常样本: {abnormal_test_path} ({len(list(abnormal_test_path.glob('*')))} 个文件)")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    print("=" * 60)
    print("PatchCore 异常检测训练脚本")
    print("=" * 60)
    
    # 检查数据路径
    print("\n1. 检查数据路径...")
    try:
        check_data_paths(args)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 创建数据模块
    print("\n2. 配置数据模块...")
    datamodule = Folder(
        name=args.dataset_name,
        root=args.data_root,
        normal_dir=args.normal_train_dir,
        normal_test_dir=args.normal_test_dir,
        abnormal_dir=args.abnormal_test_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    
    # 初始化PatchCore模型
    print("\n3. 初始化PatchCore模型...")
    print(f"   骨干网络: {args.backbone}")
    print(f"   特征层: {args.layers}")
    print(f"   核心集采样比例: {args.coreset_sampling_ratio}")
    print(f"   最近邻数量: {args.num_neighbors}")
    
    model = Patchcore(
        backbone=args.backbone,
        layers=args.layers,
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
    )
    
    # 创建训练引擎
    print("\n4. 配置训练引擎...")
    print(f"   最大轮数: {args.max_epochs}")
    print(f"   输出目录: {args.output_dir}")
    
    engine = Engine(
        max_epochs=args.max_epochs,
        enable_checkpointing=True,
        default_root_dir=args.output_dir,
        accelerator="auto",
        devices=1,
    )
    
    # 开始训练
    print("\n5. 开始训练...")
    try:
        # 设置数据模块
        datamodule.setup()
        
        # 输出数据集信息
        train_size = len(datamodule.train_dataloader().dataset)
        test_size = len(datamodule.test_dataloader().dataset)
        print(f"   训练样本数: {train_size}")
        print(f"   测试样本数: {test_size}")
        
        # 开始训练
        engine.fit(datamodule=datamodule, model=model)
        
        print("\n✅ 训练完成！")
        
        # 测试模型
        print("\n6. 开始测试模型...")
        test_results = engine.test(datamodule=datamodule, model=model)
        
        print("\n✅ 测试完成！")
        print("\n测试结果:")
        if test_results:
            for result in test_results:
                for key, value in result.items():
                    print(f"   {key}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("请检查数据路径和依赖项是否正确安装。")
        raise


if __name__ == "__main__":
    main() 