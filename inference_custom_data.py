# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""使用训练好的Patchcore模型进行推理

这个脚本展示了如何使用训练好的模型对测试数据进行推理，
并提供更直观的结果展示，包括OK和NG样本的判断准确率。
支持部分测试数据和全量测试数据两种模式。
"""

import os
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

def count_files_in_directory(directory):
    """计算目录中的文件数量"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def analyze_predictions(predictions, threshold=0.5):
    """分析预测结果并提供详细统计"""
    normal_correct = 0
    normal_total = 0
    abnormal_correct = 0
    abnormal_total = 0
    
    print("=" * 60)
    print("🔍 详细预测结果分析")
    print("=" * 60)
    
    for batch in predictions:
        for i in range(len(batch.image_path)):
            image_path = batch.image_path[i]
            pred_score = batch.pred_score[i].item()
            pred_label = batch.pred_label[i].item()  # 0: normal, 1: abnormal
            gt_label = batch.gt_label[i].item()  # 实际标签：0: normal, 1: abnormal
            
            # 根据实际标签判断
            if gt_label == 0:  # 实际为正常样本
                normal_total += 1
                if pred_label == 0:  # 预测也为正常
                    normal_correct += 1
                    status = "✅ 正确"
                else:
                    status = "❌ 错误"
                print(f"正常样本 {normal_total:3d}: {Path(image_path).name:20s} | 分数: {pred_score:.4f} | {status}")
                
            else:  # 实际为异常样本
                abnormal_total += 1
                if pred_label == 1:  # 预测也为异常
                    abnormal_correct += 1
                    status = "✅ 正确"
                else:
                    status = "❌ 错误"
                print(f"异常样本 {abnormal_total:3d}: {Path(image_path).name:20s} | 分数: {pred_score:.4f} | {status}")
    
    return normal_correct, normal_total, abnormal_correct, abnormal_total

def main():
    print("🚀 开始使用训练好的Patchcore模型进行推理...")
    
    # 询问用户选择测试模式
    print("\n请选择测试模式:")
    print("1. 部分测试数据 (train/test 分割的测试集)")
    print("2. 全量测试数据 (所有 OK 和 NG 样本)")
    
    while True:
        choice = input("请输入选择 (1 或 2): ").strip()
        if choice in ['1', '2']:
            break
        print("❌ 无效输入，请输入 1 或 2")
    
    use_full_data = (choice == '2')
    
    # 1. 初始化模型
    print("📝 加载Patchcore模型...")
    model = Patchcore()
    engine = Engine()
    
    # 2. 模型检查点路径
    ckpt_path = "results/Patchcore/project_test_organized/v3/weights/lightning/model.ckpt"
    
    if not os.path.exists(ckpt_path):
        print(f"❌ 未找到模型文件: {ckpt_path}")
        print("请先运行训练脚本 train_custom_data.py")
        return
    
    print(f"✅ 找到模型文件: {ckpt_path}")
    
    # 3. 创建数据模块（与训练时相同的配置）
    print("📂 创建数据模块...")
    
    if use_full_data:
        # 全量测试模式
        print("🔍 使用全量测试数据...")
        datamodule = Folder(
            name="project_test_organized",
            root="./project_test_organized",
            normal_dir="train/normal",
            normal_test_dir="test/ALL_OK",
            abnormal_dir="test/ALL_NG",
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,
        )
        
        # 统计测试数据 - 全部样本
        normal_count = count_files_in_directory("project_test_organized/test/ALL_OK")
        abnormal_count = count_files_in_directory("project_test_organized/test/ALL_NG")
    else:
        # 部分测试模式（与训练时完全相同）
        print("🔍 使用部分测试数据...")
        datamodule = Folder(
            name="project_test_organized",
            root="./project_test_organized",
            normal_dir="train/normal",
            normal_test_dir="test/normal",
            abnormal_dir="test/abnormal",
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,
        )
        
        # 统计测试数据 - 部分样本
        normal_count = count_files_in_directory("project_test_organized/test/normal")
        abnormal_count = count_files_in_directory("project_test_organized/test/abnormal")
    
    # 4. 准备数据
    datamodule.setup()
    
    print(f"📊 测试数据统计:")
    print(f"   正常样本 (OK): {normal_count} 张")
    print(f"   异常样本 (NG): {abnormal_count} 张")
    print(f"   总计: {normal_count + abnormal_count} 张")
    
    # 5. 进行推理
    print("🔮 开始推理...")
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        return_predictions=True,
    )
    
    # 6. 分析结果
    if predictions:
        normal_correct, normal_total, abnormal_correct, abnormal_total = analyze_predictions(predictions)
        
        # 计算准确率
        normal_accuracy = (normal_correct / normal_total * 100) if normal_total > 0 else 0
        abnormal_accuracy = (abnormal_correct / abnormal_total * 100) if abnormal_total > 0 else 0
        overall_accuracy = ((normal_correct + abnormal_correct) / (normal_total + abnormal_total) * 100) if (normal_total + abnormal_total) > 0 else 0
        
        print("\n" + "=" * 60)
        print("📈 最终结果统计")
        print("=" * 60)
        print(f"✨ 正常样本 (OK):")
        print(f"   判断正确: {normal_correct:3d} / {normal_total:3d} 张")
        print(f"   准确率: {normal_accuracy:6.2f}%")
        print()
        print(f"⚠️  异常样本 (NG):")
        print(f"   判断正确: {abnormal_correct:3d} / {abnormal_total:3d} 张")
        print(f"   准确率: {abnormal_accuracy:6.2f}%")
        print()
        print(f"🎯 总体准确率: {overall_accuracy:6.2f}% ({normal_correct + abnormal_correct}/{normal_total + abnormal_total})")
        print("=" * 60)
        
        # 混淆矩阵风格的输出
        normal_wrong = normal_total - normal_correct
        abnormal_wrong = abnormal_total - abnormal_correct
        
        print("\n📊 混淆矩阵:")
        print("                实际标签")
        print("              正常   异常")
        print(f"预测   正常   {normal_correct:4d}   {abnormal_wrong:4d}")
        print(f"       异常   {normal_wrong:4d}   {abnormal_correct:4d}")
        
    else:
        print("❌ 推理失败！")

if __name__ == "__main__":
    main() 