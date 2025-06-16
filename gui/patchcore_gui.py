#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore训练GUI界面

基于PyQt5创建的图形界面，用于方便地配置和运行PatchCore异常检测模型训练。
将界面逻辑与训练逻辑分离，调用flexible_patchcore_train.py模块进行实际训练。
"""

import sys
import os
import json
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QGroupBox, QLabel, 
                             QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
                             QComboBox, QTextEdit, QProgressBar, QFileDialog,
                             QMessageBox, QScrollArea, QFrame, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

# 导入训练模块
from patchcore_trainer import PatchCoreTrainer


class TrainingThread(QThread):
    """训练线程，防止界面卡顿"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def run(self):
        """运行训练"""
        try:
            trainer = PatchCoreTrainer(self.config)
            trainer.log_callback = self.log_signal.emit
            trainer.progress_callback = self.progress_signal.emit
            
            success = trainer.train()
            if success:
                self.finished_signal.emit(True, "训练完成！")
            else:
                self.finished_signal.emit(False, "训练失败")
                
        except Exception as e:
            self.finished_signal.emit(False, f"训练出错: {str(e)}")


class ScoreDistributionThread(QThread):
    """得分分布图生成线程"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str, list)  # success, message, output_files
    
    def __init__(self, model_path, use_full_data=True):
        super().__init__()
        self.model_path = model_path
        self.use_full_data = use_full_data
        
    def run(self):
        """运行得分分布图生成"""
        try:
            self.log_signal.emit("📊 开始加载模型和数据...")
            self.progress_signal.emit(10)
            
            # 导入必要的模块
            import numpy as np
            import matplotlib.pyplot as plt
            from anomalib.data import Folder
            from anomalib.engine import Engine
            from anomalib.models import Patchcore
            from datetime import datetime
            import lightning as L
            from anomalib.callbacks import LoadModelCallback
            
            # 设置matplotlib中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            self.log_signal.emit("🤖 初始化模型...")
            self.progress_signal.emit(20)
            
            # 初始化模型和引擎
            model = Patchcore()
            engine = Engine()
            
            # 创建数据模块
            if self.use_full_data:
                self.log_signal.emit("📂 加载全量测试数据...")
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
            else:
                self.log_signal.emit("📂 加载部分测试数据...")
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
                
            datamodule.setup()
            self.progress_signal.emit(40)
            
            self.log_signal.emit("🔮 开始推理...")
            
            # 创建自定义trainer，禁用可视化回调
            trainer = L.Trainer(
                accelerator="auto",
                devices="auto",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[LoadModelCallback()]
            )
            
            # 进行推理
            predictions = engine.predict(
                model=model,
                datamodule=datamodule,
                ckpt_path=self.model_path,
                return_predictions=True,
                trainer=trainer,
            )
            
            self.progress_signal.emit(70)
            
            if not predictions:
                self.finished_signal.emit(False, "推理失败，无法生成得分分布图", [])
                return
                
            self.log_signal.emit("📊 收集得分数据...")
            
            # 收集得分数据
            normal_scores = []
            abnormal_scores = []
            
            for batch in predictions:
                for i in range(len(batch.image_path)):
                    pred_score = batch.pred_score[i].item()
                    gt_label = batch.gt_label[i].item()
                    
                    if gt_label == 0:  # 正常样本
                        normal_scores.append(pred_score)
                    else:  # 异常样本
                        abnormal_scores.append(pred_score)
            
            self.progress_signal.emit(80)
            self.log_signal.emit("🎨 生成可视化图表...")
            
            # 生成得分分布图
            output_files = []
            
            # 1. 得分分布图
            dist_file = self.create_score_distribution_plot(normal_scores, abnormal_scores)
            if dist_file:
                output_files.append(dist_file)
                
            # 2. 阈值分析图
            self.log_signal.emit("📈 生成阈值分析图...")
            threshold_results = self.analyze_threshold_performance(normal_scores, abnormal_scores)
            analysis_file = self.create_threshold_analysis_plot(threshold_results)
            if analysis_file:
                output_files.append(analysis_file)
                
            self.progress_signal.emit(100)
            
            # 推荐最优阈值
            optimal_threshold = self.recommend_optimal_threshold(threshold_results)
            
            message = f"得分分布图生成完成！推荐阈值: {optimal_threshold:.3f}"
            self.finished_signal.emit(True, message, output_files)
            
        except Exception as e:
            self.finished_signal.emit(False, f"生成得分分布图时出错: {str(e)}", [])
            
    def create_score_distribution_plot(self, normal_scores, abnormal_scores):
        """创建得分分布图"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from datetime import datetime
            
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
            ax.set_ylim(0, max(max(normal_hist) if normal_hist.size > 0 else 0, 
                              max(abnormal_hist) if abnormal_hist.size > 0 else 0) * 1.1)
            
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
            output_file = f"score_distribution_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_signal.emit(f"📊 得分分布图已保存: {output_file}")
            return output_file
            
        except Exception as e:
            self.log_signal.emit(f"❌ 生成得分分布图失败: {str(e)}")
            return None
            
    def analyze_threshold_performance(self, normal_scores, abnormal_scores, threshold_range=(0.1, 0.9), step=0.05):
        """分析不同阈值下的性能表现"""
        import numpy as np
        
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        results = []
        
        for threshold in thresholds:
            # 计算混淆矩阵
            tp = sum(1 for score in abnormal_scores if score >= threshold)
            fn = sum(1 for score in abnormal_scores if score < threshold)
            tn = sum(1 for score in normal_scores if score < threshold)
            fp = sum(1 for score in normal_scores if score >= threshold)
            
            # 计算指标
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
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
        
    def create_threshold_analysis_plot(self, threshold_results):
        """创建阈值分析图"""
        try:
            import matplotlib.pyplot as plt
            from datetime import datetime
            
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
            output_file = f"threshold_analysis_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_signal.emit(f"📈 阈值分析图已保存: {output_file}")
            return output_file
            
        except Exception as e:
            self.log_signal.emit(f"❌ 生成阈值分析图失败: {str(e)}")
            return None
            
    def recommend_optimal_threshold(self, threshold_results):
        """推荐最优阈值"""
        if not threshold_results:
            return 0.5
            
        # 找到F1分数最高的阈值
        best_f1 = max(threshold_results, key=lambda x: x['f1_score'])
        
        # 输出推荐信息到日志
        self.log_signal.emit(f"🎯 推荐阈值: {best_f1['threshold']:.3f}")
        self.log_signal.emit(f"   F1分数: {best_f1['f1_score']:.4f}")
        self.log_signal.emit(f"   准确率: {best_f1['accuracy']:.4f}")
        self.log_signal.emit(f"   敏感性: {best_f1['sensitivity']:.4f}")
        self.log_signal.emit(f"   特异性: {best_f1['specificity']:.4f}")
        
        return best_f1['threshold']


class PatchCoreGUI(QMainWindow):
    """PatchCore训练GUI主界面"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchCore 异常检测训练工具")
        self.setGeometry(100, 100, 900, 1000)  # 增大窗口尺寸
        
        # 设置全局字体
        self.setup_fonts()
        
        # 设置样式
        self.setStyleSheet(self.get_stylesheet())
        
        # 初始化变量
        self.training_thread = None
        self.current_config = {}
        
        # 创建界面
        self.init_ui()
        
        # 加载默认配置
        self.load_default_config()
        
    def setup_fonts(self):
        """设置全局字体大小"""
        # 创建基础字体
        base_font = QFont()
        base_font.setFamily("Microsoft YaHei")  # 使用微软雅黑字体
        base_font.setPointSize(12)  # 基础字体大小
        
        # 设置应用程序默认字体
        QApplication.instance().setFont(base_font)
        
        # 为不同组件设置特定字体大小
        self.label_font = QFont("Microsoft YaHei", 11)
        self.input_font = QFont("Microsoft YaHei", 11)
        self.button_font = QFont("Microsoft YaHei", 12, QFont.Bold)
        self.log_font = QFont("Consolas", 11)  # 日志使用等宽字体
        self.title_font = QFont("Microsoft YaHei", 13, QFont.Bold)
        
    def get_stylesheet(self):
        """返回界面样式"""
        return """
        QMainWindow {
            background-color: #f0f0f0;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 13px;
            border: 2px solid #cccccc;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            font-size: 12px;
            font-weight: bold;
            border-radius: 4px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:pressed {
            background-color: #3d8b40;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-size: 11px;
            min-height: 15px;
        }
        QTextEdit {
            border: 1px solid #ddd;
            border-radius: 3px;
            font-family: Consolas, Monaco, monospace;
            font-size: 11px;
            padding: 5px;
        }
        QLabel {
            font-size: 11px;
            padding: 2px;
        }
        QCheckBox {
            font-size: 11px;
            spacing: 5px;
        }
        QProgressBar {
            border: 1px solid #ddd;
            border-radius: 3px;
            text-align: center;
            font-size: 11px;
            font-weight: bold;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 3px;
        }
        """
        
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(scroll)
        
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(10)  # 增加间距
        
        # 创建各个配置组
        layout.addWidget(self.create_data_config_group())
        layout.addWidget(self.create_model_config_group())
        layout.addWidget(self.create_training_config_group())
        layout.addWidget(self.create_output_config_group())
        layout.addWidget(self.create_control_group())
        layout.addWidget(self.create_log_group())
        
    def create_data_config_group(self):
        """创建数据配置组"""
        group = QGroupBox("数据集配置")
        group.setFont(self.title_font)
        layout = QGridLayout(group)
        layout.setSpacing(8)  # 增加间距
        
        # 数据根目录
        label = QLabel("数据根目录:")
        label.setFont(self.label_font)
        layout.addWidget(label, 0, 0)
        
        self.data_root_edit = QLineEdit("./project_test_organized")
        self.data_root_edit.setFont(self.input_font)
        layout.addWidget(self.data_root_edit, 0, 1)
        
        self.data_root_btn = QPushButton("浏览")
        self.data_root_btn.setFont(self.button_font)
        self.data_root_btn.clicked.connect(self.browse_data_root)
        layout.addWidget(self.data_root_btn, 0, 2)
        
        # 训练正常样本目录
        label = QLabel("训练正常样本:")
        label.setFont(self.label_font)
        layout.addWidget(label, 1, 0)
        
        self.normal_train_edit = QLineEdit("train/normal")
        self.normal_train_edit.setFont(self.input_font)
        layout.addWidget(self.normal_train_edit, 1, 1, 1, 2)
        
        # 测试正常样本目录
        label = QLabel("测试正常样本:")
        label.setFont(self.label_font)
        layout.addWidget(label, 2, 0)
        
        self.normal_test_edit = QLineEdit("test/normal")
        self.normal_test_edit.setFont(self.input_font)
        layout.addWidget(self.normal_test_edit, 2, 1, 1, 2)
        
        # 测试异常样本目录
        label = QLabel("测试异常样本:")
        label.setFont(self.label_font)
        layout.addWidget(label, 3, 0)
        
        self.abnormal_test_edit = QLineEdit("test/abnormal")
        self.abnormal_test_edit.setFont(self.input_font)
        layout.addWidget(self.abnormal_test_edit, 3, 1, 1, 2)
        
        # 数据集名称
        label = QLabel("数据集名称:")
        label.setFont(self.label_font)
        layout.addWidget(label, 4, 0)
        
        self.dataset_name_edit = QLineEdit("custom_dataset")
        self.dataset_name_edit.setFont(self.input_font)
        layout.addWidget(self.dataset_name_edit, 4, 1, 1, 2)
        
        return group
        
    def create_model_config_group(self):
        """创建模型配置组"""
        group = QGroupBox("模型配置")
        group.setFont(self.title_font)
        layout = QGridLayout(group)
        layout.setSpacing(8)
        
        # 骨干网络
        label = QLabel("骨干网络:")
        label.setFont(self.label_font)
        layout.addWidget(label, 0, 0)
        
        self.backbone_combo = QComboBox()
        self.backbone_combo.setFont(self.input_font)
        self.backbone_combo.addItems(["wide_resnet50_2", "resnet18", "resnet50"])
        layout.addWidget(self.backbone_combo, 0, 1)
        
        # 特征层
        label = QLabel("特征层:")
        label.setFont(self.label_font)
        layout.addWidget(label, 1, 0)
        
        layer_widget = QWidget()
        layer_layout = QHBoxLayout(layer_widget)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        
        self.layer1_cb = QCheckBox("layer1")
        self.layer2_cb = QCheckBox("layer2")
        self.layer3_cb = QCheckBox("layer3")
        self.layer4_cb = QCheckBox("layer4")
        
        # 设置复选框字体
        for cb in [self.layer1_cb, self.layer2_cb, self.layer3_cb, self.layer4_cb]:
            cb.setFont(self.label_font)
        
        # 默认选择layer2和layer3
        self.layer2_cb.setChecked(True)
        self.layer3_cb.setChecked(True)
        
        layer_layout.addWidget(self.layer1_cb)
        layer_layout.addWidget(self.layer2_cb)
        layer_layout.addWidget(self.layer3_cb)
        layer_layout.addWidget(self.layer4_cb)
        layer_layout.addStretch()
        
        layout.addWidget(layer_widget, 1, 1)
        
        # 核心集采样比例
        label = QLabel("核心集采样比例:")
        label.setFont(self.label_font)
        layout.addWidget(label, 2, 0)
        
        self.coreset_ratio_spin = QDoubleSpinBox()
        self.coreset_ratio_spin.setFont(self.input_font)
        self.coreset_ratio_spin.setRange(0.01, 1.0)
        self.coreset_ratio_spin.setSingleStep(0.01)
        self.coreset_ratio_spin.setValue(0.1)
        self.coreset_ratio_spin.setDecimals(2)
        layout.addWidget(self.coreset_ratio_spin, 2, 1)
        
        # 最近邻数量
        label = QLabel("最近邻数量:")
        label.setFont(self.label_font)
        layout.addWidget(label, 3, 0)
        
        self.num_neighbors_spin = QSpinBox()
        self.num_neighbors_spin.setFont(self.input_font)
        self.num_neighbors_spin.setRange(1, 50)
        self.num_neighbors_spin.setValue(9)
        layout.addWidget(self.num_neighbors_spin, 3, 1)
        
        return group
        
    def create_training_config_group(self):
        """创建训练配置组"""
        group = QGroupBox("训练配置")
        group.setFont(self.title_font)
        layout = QGridLayout(group)
        layout.setSpacing(8)
        
        # 训练批次大小
        label = QLabel("训练批次大小:")
        label.setFont(self.label_font)
        layout.addWidget(label, 0, 0)
        
        self.train_batch_spin = QSpinBox()
        self.train_batch_spin.setFont(self.input_font)
        self.train_batch_spin.setRange(1, 256)
        self.train_batch_spin.setValue(16)
        layout.addWidget(self.train_batch_spin, 0, 1)
        
        # 评估批次大小
        label = QLabel("评估批次大小:")
        label.setFont(self.label_font)
        layout.addWidget(label, 1, 0)
        
        self.eval_batch_spin = QSpinBox()
        self.eval_batch_spin.setFont(self.input_font)
        self.eval_batch_spin.setRange(1, 256)
        self.eval_batch_spin.setValue(16)
        layout.addWidget(self.eval_batch_spin, 1, 1)
        
        # 最大训练轮数
        label = QLabel("最大训练轮数:")
        label.setFont(self.label_font)
        layout.addWidget(label, 2, 0)
        
        self.max_epochs_spin = QSpinBox()
        self.max_epochs_spin.setFont(self.input_font)
        self.max_epochs_spin.setRange(1, 1000)
        self.max_epochs_spin.setValue(1)
        layout.addWidget(self.max_epochs_spin, 2, 1)
        
        # 工作进程数
        label = QLabel("工作进程数:")
        label.setFont(self.label_font)
        layout.addWidget(label, 3, 0)
        
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setFont(self.input_font)
        self.num_workers_spin.setRange(0, 16)
        self.num_workers_spin.setValue(0)
        layout.addWidget(self.num_workers_spin, 3, 1)
        
        return group
        
    def create_output_config_group(self):
        """创建输出配置组"""
        group = QGroupBox("输出配置")
        group.setFont(self.title_font)
        layout = QGridLayout(group)
        layout.setSpacing(8)
        
        # 输出目录
        label = QLabel("输出目录:")
        label.setFont(self.label_font)
        layout.addWidget(label, 0, 0)
        
        self.output_dir_edit = QLineEdit("./results")
        self.output_dir_edit.setFont(self.input_font)
        layout.addWidget(self.output_dir_edit, 0, 1)
        
        self.output_dir_btn = QPushButton("浏览")
        self.output_dir_btn.setFont(self.button_font)
        self.output_dir_btn.clicked.connect(self.browse_output_dir)
        layout.addWidget(self.output_dir_btn, 0, 2)
        
        return group
        
    def create_control_group(self):
        """创建控制按钮组"""
        group = QGroupBox("操作控制")
        group.setFont(self.title_font)
        layout = QHBoxLayout(group)
        layout.setSpacing(10)
        
        # 加载配置按钮
        self.load_config_btn = QPushButton("加载配置")
        self.load_config_btn.setFont(self.button_font)
        self.load_config_btn.clicked.connect(self.load_config)
        layout.addWidget(self.load_config_btn)
        
        # 保存配置按钮
        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.setFont(self.button_font)
        self.save_config_btn.clicked.connect(self.save_config)
        layout.addWidget(self.save_config_btn)
        
        # 检查数据按钮
        self.check_data_btn = QPushButton("检查数据")
        self.check_data_btn.setFont(self.button_font)
        self.check_data_btn.clicked.connect(self.check_data)
        layout.addWidget(self.check_data_btn)
        
        # 得分分布图按钮
        self.score_dist_btn = QPushButton("生成得分分布图")
        self.score_dist_btn.setFont(self.button_font)
        self.score_dist_btn.clicked.connect(self.generate_score_distribution)
        self.score_dist_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                font-weight: bold;
                font-size: 12px;
                padding: 10px 16px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
        """)
        layout.addWidget(self.score_dist_btn)
        
        layout.addStretch()
        
        # 开始训练按钮
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                font-weight: bold;
                font-size: 14px;
                padding: 12px 24px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        layout.addWidget(self.start_btn)
        
        # 停止训练按钮
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                font-weight: bold;
                font-size: 14px;
                padding: 12px 24px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        layout.addWidget(self.stop_btn)
        
        return group
        
    def create_log_group(self):
        """创建日志显示组"""
        group = QGroupBox("训练日志")
        group.setFont(self.title_font)
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFont(self.label_font)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)  # 增加进度条高度
        layout.addWidget(self.progress_bar)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setFont(self.log_font)
        self.log_text.setMinimumHeight(250)  # 增加日志框高度
        self.log_text.setMaximumHeight(350)
        layout.addWidget(self.log_text)
        
        # 清除日志按钮
        clear_btn = QPushButton("清除日志")
        clear_btn.setFont(self.button_font)
        clear_btn.clicked.connect(self.clear_log)
        layout.addWidget(clear_btn)
        
        return group
        
    def browse_data_root(self):
        """浏览数据根目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据根目录")
        if dir_path:
            self.data_root_edit.setText(dir_path)
            
    def browse_output_dir(self):
        """浏览输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
            
    def get_selected_layers(self):
        """获取选中的特征层"""
        layers = []
        if self.layer1_cb.isChecked():
            layers.append("layer1")
        if self.layer2_cb.isChecked():
            layers.append("layer2")
        if self.layer3_cb.isChecked():
            layers.append("layer3")
        if self.layer4_cb.isChecked():
            layers.append("layer4")
        return layers
        
    def get_current_config(self):
        """获取当前配置"""
        return {
            "data_root": self.data_root_edit.text(),
            "normal_train_dir": self.normal_train_edit.text(),
            "normal_test_dir": self.normal_test_edit.text(),
            "abnormal_test_dir": self.abnormal_test_edit.text(),
            "dataset_name": self.dataset_name_edit.text(),
            "backbone": self.backbone_combo.currentText(),
            "layers": self.get_selected_layers(),
            "coreset_sampling_ratio": self.coreset_ratio_spin.value(),
            "num_neighbors": self.num_neighbors_spin.value(),
            "train_batch_size": self.train_batch_spin.value(),
            "eval_batch_size": self.eval_batch_spin.value(),
            "max_epochs": self.max_epochs_spin.value(),
            "num_workers": self.num_workers_spin.value(),
            "output_dir": self.output_dir_edit.text(),
        }
        
    def load_default_config(self):
        """加载默认配置"""
        self.log("加载默认配置...")
        
    def load_config(self):
        """加载配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载配置文件", "", "JSON文件 (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.apply_config(config)
                self.log(f"成功加载配置: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载配置失败: {e}")
                
    def save_config(self):
        """保存配置文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存配置文件", "patchcore_config.json", "JSON文件 (*.json)"
        )
        if file_path:
            try:
                config = self.get_current_config()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                self.log(f"成功保存配置: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存配置失败: {e}")
                
    def apply_config(self, config):
        """应用配置到界面"""
        self.data_root_edit.setText(config.get("data_root", ""))
        self.normal_train_edit.setText(config.get("normal_train_dir", ""))
        self.normal_test_edit.setText(config.get("normal_test_dir", ""))
        self.abnormal_test_edit.setText(config.get("abnormal_test_dir", ""))
        self.dataset_name_edit.setText(config.get("dataset_name", ""))
        
        backbone = config.get("backbone", "wide_resnet50_2")
        index = self.backbone_combo.findText(backbone)
        if index >= 0:
            self.backbone_combo.setCurrentIndex(index)
            
        # 设置特征层
        layers = config.get("layers", ["layer2", "layer3"])
        self.layer1_cb.setChecked("layer1" in layers)
        self.layer2_cb.setChecked("layer2" in layers)
        self.layer3_cb.setChecked("layer3" in layers)
        self.layer4_cb.setChecked("layer4" in layers)
        
        self.coreset_ratio_spin.setValue(config.get("coreset_sampling_ratio", 0.1))
        self.num_neighbors_spin.setValue(config.get("num_neighbors", 9))
        self.train_batch_spin.setValue(config.get("train_batch_size", 16))
        self.eval_batch_spin.setValue(config.get("eval_batch_size", 16))
        self.max_epochs_spin.setValue(config.get("max_epochs", 1))
        self.num_workers_spin.setValue(config.get("num_workers", 0))
        self.output_dir_edit.setText(config.get("output_dir", "./results"))
        
    def check_data(self):
        """检查数据路径"""
        config = self.get_current_config()
        data_root = Path(config["data_root"])
        
        self.log("开始检查数据路径...")
        
        if not data_root.exists():
            self.log(f"❌ 数据根目录不存在: {data_root}")
            return
            
        paths_to_check = [
            (data_root / config["normal_train_dir"], "训练正常样本"),
            (data_root / config["normal_test_dir"], "测试正常样本"),
            (data_root / config["abnormal_test_dir"], "测试异常样本"),
        ]
        
        all_good = True
        for path, description in paths_to_check:
            if path.exists():
                file_count = len(list(path.glob("*")))
                self.log(f"✅ {description}: {path} ({file_count} 个文件)")
            else:
                self.log(f"❌ {description}目录不存在: {path}")
                all_good = False
                
        if all_good:
            self.log("✅ 所有数据路径检查通过！")
        else:
            self.log("❌ 数据路径检查失败，请修正后再训练")
            
    def start_training(self):
        """开始训练"""
        # 检查配置
        config = self.get_current_config()
        if not config["layers"]:
            QMessageBox.warning(self, "错误", "请至少选择一个特征层！")
            return
            
        # 更新界面状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.log("开始训练...")
        self.log(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
        
        # 启动训练线程
        self.training_thread = TrainingThread(config)
        self.training_thread.log_signal.connect(self.log)
        self.training_thread.progress_signal.connect(self.update_progress)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()
        
    def stop_training(self):
        """停止训练"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
            
        self.training_finished(False, "训练已被用户停止")
        
    def training_finished(self, success, message):
        """训练完成回调"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if success:
            self.log(f"✅ {message}")
            
            # 训练成功后询问是否生成得分分布图
            reply = QMessageBox.question(
                self, 
                "训练完成", 
                f"{message}\n\n是否要生成得分分布图来评估模型性能？\n(这将对测试数据进行推理并生成可视化图表)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.generate_score_distribution()
            else:
                QMessageBox.information(self, "训练完成", "训练完成！可以稍后点击'生成得分分布图'按钮来评估模型性能。")
        else:
            self.log(f"❌ {message}")
            QMessageBox.warning(self, "训练失败", message)
            
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def log(self, message):
        """添加日志"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # 自动滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)
        
    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
        
    def generate_score_distribution(self):
        """生成得分分布图"""
        # 检查是否存在训练好的模型
        config = self.get_current_config()
        output_dir = config.get("output_dir", "./results")
        dataset_name = config.get("dataset_name", "custom_dataset")
        
        # 查找最新的模型文件
        model_pattern = f"{output_dir}/Patchcore/{dataset_name}/*/weights/lightning/model.ckpt"
        import glob
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            QMessageBox.warning(self, "错误", 
                "未找到训练好的模型！\n请先完成模型训练。")
            return
            
        # 选择最新的模型
        latest_model = max(model_files, key=os.path.getctime)
        self.log(f"🔍 找到模型文件: {latest_model}")
        
        # 询问用户选择测试模式
        reply = QMessageBox.question(
            self,
            "选择测试模式",
            "请选择要分析的数据集:\n\n"
            "Yes - 全量测试数据 (所有 OK 和 NG 样本)\n"
            "No - 部分测试数据 (train/test 分割的测试集)\n\n"
            "建议使用全量数据获得更全面的分析结果。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        use_full_data = (reply == QMessageBox.Yes)
        mode_text = "全量测试数据" if use_full_data else "部分测试数据"
        
        self.log(f"📊 开始生成得分分布图 - 使用{mode_text}...")
        
        # 启动得分分布生成线程
        self.score_dist_thread = ScoreDistributionThread(latest_model, use_full_data)
        self.score_dist_thread.log_signal.connect(self.log)
        self.score_dist_thread.progress_signal.connect(self.update_progress)
        self.score_dist_thread.finished_signal.connect(self.score_distribution_finished)
        
        # 禁用按钮，显示进度条
        self.score_dist_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.score_dist_thread.start()
        
    def score_distribution_finished(self, success, message, output_files=None):
        """得分分布图生成完成回调"""
        # 恢复界面状态
        self.score_dist_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.log(f"✅ {message}")
            
            # 显示生成的文件
            if output_files:
                files_text = "\n".join([f"• {os.path.basename(f)}" for f in output_files])
                QMessageBox.information(
                    self,
                    "得分分布图生成完成",
                    f"{message}\n\n生成的文件:\n{files_text}\n\n"
                    "图片已保存到当前目录，可以用图片查看器打开查看。"
                )
            else:
                QMessageBox.information(self, "完成", message)
        else:
            self.log(f"❌ {message}")
            QMessageBox.warning(self, "生成失败", message)


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用图标和信息
    app.setApplicationName("PatchCore训练工具")
    app.setApplicationVersion("1.0")
    
    # 创建并显示主窗口
    window = PatchCoreGUI()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 