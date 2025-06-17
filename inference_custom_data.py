# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""使用训练好的Patchcore模型进行推理

这个脚本展示了如何使用训练好的模型对测试数据进行推理，
并提供更直观的结果展示，包括OK和NG样本的判断准确率。
支持部分测试数据和全量测试数据两种模式。
新增得分分布可视化功能，帮助选择合适的阈值。
"""

import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 支持负号显示

def count_files_in_directory(directory):
    """计算目录中的文件数量"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def collect_score_data(predictions):
    """收集所有样本的得分数据，按标签分类"""
    normal_scores = []  # OK样本分数
    abnormal_scores = []  # NG样本分数
    all_scores = []  # 所有样本分数
    
    for batch in predictions:
        for i in range(len(batch.image_path)):
            pred_score = batch.pred_score[i].item()
            gt_label = batch.gt_label[i].item()
            
            all_scores.append(pred_score)
            
            if gt_label == 0:  # 正常样本
                normal_scores.append(pred_score)
            else:  # 异常样本
                abnormal_scores.append(pred_score)
    
    return normal_scores, abnormal_scores, all_scores

def create_score_distribution_plot(normal_scores, abnormal_scores, output_dir="./"):
    """创建得分分布图，类似用户提供的图片"""
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置分数范围和分箱
    score_min = 0.0
    score_max = 1.0
    num_bins = 100
    bins = np.linspace(score_min, score_max, num_bins + 1)
    
    # 计算直方图数据
    normal_hist, _ = np.histogram(normal_scores, bins=bins)
    abnormal_hist, _ = np.histogram(abnormal_scores, bins=bins)
    
    # 计算分箱中心点
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 绘制直方图
    ax.plot(bin_centers, normal_hist, 'g-', linewidth=2, label='标注(OK)', marker='o', markersize=3)
    ax.plot(bin_centers, abnormal_hist, 'r-', linewidth=2, label='标注(NG)', marker='s', markersize=3)
    
    # 设置图表属性
    ax.set_xlabel('得分', fontsize=14)
    ax.set_ylabel('图片数量', fontsize=14)
    ax.set_title('得分分布图', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(max(normal_hist), max(abnormal_hist)) * 1.1)
    
    # 添加统计信息文本
    stats_text = f"""样本统计:
正常样本 (OK): {len(normal_scores)} 张
异常样本 (NG): {len(abnormal_scores)} 张
总计: {len(normal_scores) + len(abnormal_scores)} 张

得分统计:
OK样本平均分: {np.mean(normal_scores):.4f}
NG样本平均分: {np.mean(abnormal_scores):.4f}
OK样本中位数: {np.median(normal_scores):.4f}
NG样本中位数: {np.median(abnormal_scores):.4f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"score_distribution_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"📊 得分分布图已保存到: {output_file}")
    
    # 显示图片（如果在支持的环境中）
    try:
        plt.show()
    except:
        print("💡 提示: 无法显示图片，但已保存到文件")
    
    plt.close()
    
    return output_file

def analyze_threshold_performance(normal_scores, abnormal_scores, threshold_range=(0.1, 0.9), step=0.05):
    """分析不同阈值下的性能表现"""
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    results = []
    
    for threshold in thresholds:
        # 计算混淆矩阵
        tp = sum(1 for score in abnormal_scores if score >= threshold)  # 正确检测的异常
        fn = sum(1 for score in abnormal_scores if score < threshold)   # 漏检的异常
        tn = sum(1 for score in normal_scores if score < threshold)     # 正确检测的正常
        fp = sum(1 for score in normal_scores if score >= threshold)    # 误检的正常
        
        # 计算指标
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率/敏感性
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0     # 精确率
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp
        })
    
    return results

def create_threshold_analysis_plot(threshold_results, output_dir="./"):
    """创建阈值分析图"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    thresholds = [r['threshold'] for r in threshold_results]
    accuracies = [r['accuracy'] for r in threshold_results]
    sensitivities = [r['sensitivity'] for r in threshold_results]
    specificities = [r['specificity'] for r in threshold_results]
    f1_scores = [r['f1_score'] for r in threshold_results]
    
    # 第一个子图：准确率和F1分数
    ax1.plot(thresholds, accuracies, 'b-', linewidth=2, label='准确率', marker='o')
    ax1.plot(thresholds, f1_scores, 'purple', linewidth=2, label='F1分数', marker='s')
    ax1.set_xlabel('阈值', fontsize=12)
    ax1.set_ylabel('指标值', fontsize=12)
    ax1.set_title('不同阈值下的性能指标', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 第二个子图：敏感性和特异性
    ax2.plot(thresholds, sensitivities, 'r-', linewidth=2, label='敏感性(异常检出率)', marker='^')
    ax2.plot(thresholds, specificities, 'g-', linewidth=2, label='特异性(正常识别率)', marker='v')
    ax2.set_xlabel('阈值', fontsize=12)
    ax2.set_ylabel('指标值', fontsize=12)
    ax2.set_title('敏感性vs特异性', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"threshold_analysis_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"📈 阈值分析图已保存到: {output_file}")
    
    try:
        plt.show()
    except:
        print("💡 提示: 无法显示图片，但已保存到文件")
    
    plt.close()
    
    return output_file

def recommend_optimal_threshold(threshold_results):
    """推荐最优阈值"""
    
    # 找到F1分数最高的阈值
    best_f1 = max(threshold_results, key=lambda x: x['f1_score'])
    
    # 找到准确率最高的阈值
    best_acc = max(threshold_results, key=lambda x: x['accuracy'])
    
    # 找到平衡敏感性和特异性的阈值
    balanced_scores = []
    for result in threshold_results:
        balance_score = min(result['sensitivity'], result['specificity'])
        balanced_scores.append((result['threshold'], balance_score))
    
    best_balanced = max(balanced_scores, key=lambda x: x[1])
    
    print("\n" + "=" * 60)
    print("🎯 阈值推荐")
    print("=" * 60)
    print(f"最佳F1分数阈值: {best_f1['threshold']:.3f} (F1: {best_f1['f1_score']:.4f})")
    print(f"最佳准确率阈值: {best_acc['threshold']:.3f} (准确率: {best_acc['accuracy']:.4f})")
    print(f"最平衡阈值: {best_balanced[0]:.3f} (平衡分数: {best_balanced[1]:.4f})")
    
    # 选择F1分数最高的作为推荐阈值
    recommended = best_f1
    print(f"\n💡 推荐使用阈值: {recommended['threshold']:.3f}")
    print(f"   该阈值下的性能:")
    print(f"   - 准确率: {recommended['accuracy']:.4f}")
    print(f"   - 敏感性: {recommended['sensitivity']:.4f}")
    print(f"   - 特异性: {recommended['specificity']:.4f}")
    print(f"   - F1分数: {recommended['f1_score']:.4f}")
    
    return recommended['threshold']

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

def analyze_predictions_with_threshold(predictions, threshold=0.5):
    """使用指定阈值分析预测结果并提供详细统计"""
    normal_correct = 0
    normal_total = 0
    abnormal_correct = 0
    abnormal_total = 0
    
    print("=" * 60)
    print(f"🔍 使用阈值 {threshold:.3f} 重新分析结果")
    print("=" * 60)
    
    for batch in predictions:
        for i in range(len(batch.image_path)):
            image_path = batch.image_path[i]
            pred_score = batch.pred_score[i].item()
            gt_label = batch.gt_label[i].item()  # 实际标签：0: normal, 1: abnormal
            
            # 基于新阈值重新计算预测标签
            new_pred_label = 1 if pred_score >= threshold else 0
            
            # 根据实际标签判断
            if gt_label == 0:  # 实际为正常样本
                normal_total += 1
                if new_pred_label == 0:  # 预测也为正常
                    normal_correct += 1
                    status = "✅ 正确"
                else:
                    status = "❌ 错误"
                print(f"正常样本 {normal_total:3d}: {Path(image_path).name:20s} | 分数: {pred_score:.4f} | 新预测: {'异常' if new_pred_label else '正常'} | {status}")
                
            else:  # 实际为异常样本
                abnormal_total += 1
                if new_pred_label == 1:  # 预测也为异常
                    abnormal_correct += 1
                    status = "✅ 正确"
                else:
                    status = "❌ 错误"
                print(f"异常样本 {abnormal_total:3d}: {Path(image_path).name:20s} | 分数: {pred_score:.4f} | 新预测: {'异常' if new_pred_label else '正常'} | {status}")
    
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
    ckpt_path = r"C:\Users\Administrator\Desktop\anomalib\gui\model.ckpt"
    
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
        # 收集得分数据
        print("\n📊 收集得分数据...")
        normal_scores, abnormal_scores, all_scores = collect_score_data(predictions)
        
        # 创建得分分布图
        print("🎨 生成得分分布图...")
        create_score_distribution_plot(normal_scores, abnormal_scores)
        
        # 进行阈值分析
        print("🔍 进行阈值性能分析...")
        threshold_results = analyze_threshold_performance(normal_scores, abnormal_scores)
        
        # 创建阈值分析图
        create_threshold_analysis_plot(threshold_results)
        
        # 推荐最优阈值
        optimal_threshold = recommend_optimal_threshold(threshold_results)
        
        # 使用推荐阈值重新分析（如果用户同意）
        print(f"\n是否使用推荐阈值 {optimal_threshold:.3f} 重新分析结果？")
        use_optimal = input("输入 'y' 或 'yes' 使用，其他键跳过: ").strip().lower()
        
        if use_optimal in ['y', 'yes']:
            print(f"\n使用推荐阈值 {optimal_threshold:.3f} 重新分析...")
            
            # 使用新阈值重新计算准确率
            new_normal_correct, new_normal_total, new_abnormal_correct, new_abnormal_total = analyze_predictions_with_threshold(predictions, optimal_threshold)
            
            # 计算新的准确率
            new_normal_accuracy = (new_normal_correct / new_normal_total * 100) if new_normal_total > 0 else 0
            new_abnormal_accuracy = (new_abnormal_correct / new_abnormal_total * 100) if new_abnormal_total > 0 else 0
            new_overall_accuracy = ((new_normal_correct + new_abnormal_correct) / (new_normal_total + new_abnormal_total) * 100) if (new_normal_total + new_abnormal_total) > 0 else 0
            
            print("\n" + "=" * 60)
            print(f"📈 新阈值 ({optimal_threshold:.3f}) 下的结果统计")
            print("=" * 60)
            print(f"✨ 正常样本 (OK):")
            print(f"   判断正确: {new_normal_correct:3d} / {new_normal_total:3d} 张")
            print(f"   准确率: {new_normal_accuracy:6.2f}%")
            print()
            print(f"⚠️  异常样本 (NG):")
            print(f"   判断正确: {new_abnormal_correct:3d} / {new_abnormal_total:3d} 张")
            print(f"   准确率: {new_abnormal_accuracy:6.2f}%")
            print()
            print(f"🎯 总体准确率: {new_overall_accuracy:6.2f}% ({new_normal_correct + new_abnormal_correct}/{new_normal_total + new_abnormal_total})")
            
            # 新的混淆矩阵
            new_normal_wrong = new_normal_total - new_normal_correct
            new_abnormal_wrong = new_abnormal_total - new_abnormal_correct
            
            print("\n📊 新阈值下的混淆矩阵:")
            print("                实际标签")
            print("              正常   异常")
            print(f"预测   正常   {new_normal_correct:4d}   {new_abnormal_wrong:4d}")
            print(f"       异常   {new_normal_wrong:4d}   {new_abnormal_correct:4d}")
            print("=" * 60)
        else:
            # 原始分析（使用模型默认阈值）
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