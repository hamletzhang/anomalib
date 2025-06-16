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


class PatchCoreGUI(QMainWindow):
    """PatchCore训练GUI主界面"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchCore 异常检测训练工具")
        self.setGeometry(100, 100, 800, 900)
        
        # 设置样式
        self.setStyleSheet(self.get_stylesheet())
        
        # 初始化变量
        self.training_thread = None
        self.current_config = {}
        
        # 创建界面
        self.init_ui()
        
        # 加载默认配置
        self.load_default_config()
        
    def get_stylesheet(self):
        """返回界面样式"""
        return """
        QMainWindow {
            background-color: #f0f0f0;
        }
        QGroupBox {
            font-weight: bold;
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
            padding: 8px 16px;
            text-align: center;
            font-size: 14px;
            border-radius: 4px;
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
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
            font-size: 12px;
        }
        QTextEdit {
            border: 1px solid #ddd;
            border-radius: 3px;
            font-family: Consolas, Monaco, monospace;
            font-size: 10px;
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
        layout = QGridLayout(group)
        
        # 数据根目录
        layout.addWidget(QLabel("数据根目录:"), 0, 0)
        self.data_root_edit = QLineEdit("./project_test_organized")
        layout.addWidget(self.data_root_edit, 0, 1)
        self.data_root_btn = QPushButton("浏览")
        self.data_root_btn.clicked.connect(self.browse_data_root)
        layout.addWidget(self.data_root_btn, 0, 2)
        
        # 训练正常样本目录
        layout.addWidget(QLabel("训练正常样本:"), 1, 0)
        self.normal_train_edit = QLineEdit("train/normal")
        layout.addWidget(self.normal_train_edit, 1, 1, 1, 2)
        
        # 测试正常样本目录
        layout.addWidget(QLabel("测试正常样本:"), 2, 0)
        self.normal_test_edit = QLineEdit("test/normal")
        layout.addWidget(self.normal_test_edit, 2, 1, 1, 2)
        
        # 测试异常样本目录
        layout.addWidget(QLabel("测试异常样本:"), 3, 0)
        self.abnormal_test_edit = QLineEdit("test/abnormal")
        layout.addWidget(self.abnormal_test_edit, 3, 1, 1, 2)
        
        # 数据集名称
        layout.addWidget(QLabel("数据集名称:"), 4, 0)
        self.dataset_name_edit = QLineEdit("custom_dataset")
        layout.addWidget(self.dataset_name_edit, 4, 1, 1, 2)
        
        return group
        
    def create_model_config_group(self):
        """创建模型配置组"""
        group = QGroupBox("模型配置")
        layout = QGridLayout(group)
        
        # 骨干网络
        layout.addWidget(QLabel("骨干网络:"), 0, 0)
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(["wide_resnet50_2", "resnet18", "resnet50"])
        layout.addWidget(self.backbone_combo, 0, 1)
        
        # 特征层
        layout.addWidget(QLabel("特征层:"), 1, 0)
        layer_widget = QWidget()
        layer_layout = QHBoxLayout(layer_widget)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        
        self.layer1_cb = QCheckBox("layer1")
        self.layer2_cb = QCheckBox("layer2")
        self.layer3_cb = QCheckBox("layer3")
        self.layer4_cb = QCheckBox("layer4")
        
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
        layout.addWidget(QLabel("核心集采样比例:"), 2, 0)
        self.coreset_ratio_spin = QDoubleSpinBox()
        self.coreset_ratio_spin.setRange(0.01, 1.0)
        self.coreset_ratio_spin.setSingleStep(0.01)
        self.coreset_ratio_spin.setValue(0.1)
        self.coreset_ratio_spin.setDecimals(2)
        layout.addWidget(self.coreset_ratio_spin, 2, 1)
        
        # 最近邻数量
        layout.addWidget(QLabel("最近邻数量:"), 3, 0)
        self.num_neighbors_spin = QSpinBox()
        self.num_neighbors_spin.setRange(1, 50)
        self.num_neighbors_spin.setValue(9)
        layout.addWidget(self.num_neighbors_spin, 3, 1)
        
        return group
        
    def create_training_config_group(self):
        """创建训练配置组"""
        group = QGroupBox("训练配置")
        layout = QGridLayout(group)
        
        # 训练批次大小
        layout.addWidget(QLabel("训练批次大小:"), 0, 0)
        self.train_batch_spin = QSpinBox()
        self.train_batch_spin.setRange(1, 256)
        self.train_batch_spin.setValue(16)
        layout.addWidget(self.train_batch_spin, 0, 1)
        
        # 评估批次大小
        layout.addWidget(QLabel("评估批次大小:"), 1, 0)
        self.eval_batch_spin = QSpinBox()
        self.eval_batch_spin.setRange(1, 256)
        self.eval_batch_spin.setValue(16)
        layout.addWidget(self.eval_batch_spin, 1, 1)
        
        # 最大训练轮数
        layout.addWidget(QLabel("最大训练轮数:"), 2, 0)
        self.max_epochs_spin = QSpinBox()
        self.max_epochs_spin.setRange(1, 1000)
        self.max_epochs_spin.setValue(1)
        layout.addWidget(self.max_epochs_spin, 2, 1)
        
        # 工作进程数
        layout.addWidget(QLabel("工作进程数:"), 3, 0)
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 16)
        self.num_workers_spin.setValue(0)
        layout.addWidget(self.num_workers_spin, 3, 1)
        
        return group
        
    def create_output_config_group(self):
        """创建输出配置组"""
        group = QGroupBox("输出配置")
        layout = QGridLayout(group)
        
        # 输出目录
        layout.addWidget(QLabel("输出目录:"), 0, 0)
        self.output_dir_edit = QLineEdit("./results")
        layout.addWidget(self.output_dir_edit, 0, 1)
        self.output_dir_btn = QPushButton("浏览")
        self.output_dir_btn.clicked.connect(self.browse_output_dir)
        layout.addWidget(self.output_dir_btn, 0, 2)
        
        return group
        
    def create_control_group(self):
        """创建控制按钮组"""
        group = QGroupBox("操作控制")
        layout = QHBoxLayout(group)
        
        # 加载配置按钮
        self.load_config_btn = QPushButton("加载配置")
        self.load_config_btn.clicked.connect(self.load_config)
        layout.addWidget(self.load_config_btn)
        
        # 保存配置按钮
        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.clicked.connect(self.save_config)
        layout.addWidget(self.save_config_btn)
        
        # 检查数据按钮
        self.check_data_btn = QPushButton("检查数据")
        self.check_data_btn.clicked.connect(self.check_data)
        layout.addWidget(self.check_data_btn)
        
        layout.addStretch()
        
        # 开始训练按钮
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self.start_training)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                font-weight: bold;
                font-size: 16px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        layout.addWidget(self.start_btn)
        
        # 停止训练按钮
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                font-weight: bold;
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
        layout = QVBoxLayout(group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setMinimumHeight(200)
        self.log_text.setMaximumHeight(300)
        layout.addWidget(self.log_text)
        
        # 清除日志按钮
        clear_btn = QPushButton("清除日志")
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
            QMessageBox.information(self, "训练完成", message)
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