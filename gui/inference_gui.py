#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore推理界面

前后端分离的推理GUI，支持单图和批量推理，可视化结果展示。
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QGroupBox, QLabel, 
                             QLineEdit, QPushButton, QProgressBar, QFileDialog,
                             QTextEdit, QTabWidget, QScrollArea, QFrame,
                             QDoubleSpinBox, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QMessageBox

# 导入推理引擎
from inference_engine import InferenceEngine


class InferenceThread(QThread):
    """推理线程，防止界面卡顿"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(object)  # 推理结果
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, engine, mode, input_path, output_dir=None):
        super().__init__()
        self.engine = engine
        self.mode = mode  # 'single' 或 'batch'
        self.input_path = input_path
        self.output_dir = output_dir
        
    def run(self):
        """运行推理"""
        try:
            if self.mode == 'single':
                # 单图推理
                result = self.engine.predict_single_image(self.input_path)
                if result:
                    self.result_signal.emit(result)
                    self.finished_signal.emit(True, "单图推理完成")
                else:
                    self.finished_signal.emit(False, "单图推理失败")
                    
            elif self.mode == 'batch':
                # 批量推理
                results = self.engine.predict_folder(self.input_path, self.output_dir)
                if results:
                    self.result_signal.emit(results)
                    self.finished_signal.emit(True, f"批量推理完成，共处理 {len(results)} 张图片")
                else:
                    self.finished_signal.emit(False, "批量推理失败")
                    
        except Exception as e:
            self.finished_signal.emit(False, f"推理出错: {str(e)}")


class ImageDisplayWidget(QLabel):
    """图片显示组件"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
        """)
        self.setAlignment(Qt.AlignCenter)
        self.setText("点击选择图片或\n将在此显示推理结果")
        self.setWordWrap(True)
        
    def display_image(self, image_path: str):
        """显示图片"""
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 缩放图片以适应显示区域
                scaled_pixmap = pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.setPixmap(scaled_pixmap)
            else:
                self.setText("无法加载图片")
        except Exception as e:
            self.setText(f"图片加载错误: {str(e)}")
            
    def clear_display(self):
        """清除显示"""
        self.clear()
        self.setText("点击选择图片或\n将在此显示推理结果")


class InferenceGUI(QMainWindow):
    """推理界面主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchCore 异常检测推理工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 设置字体
        self.setup_fonts()
        
        # 初始化变量
        self.inference_engine = InferenceEngine()
        self.inference_thread = None
        self.current_results = []
        
        # 创建界面
        self.init_ui()
        
        # 尝试自动加载模型
        self.auto_load_model()
        
    def setup_fonts(self):
        """设置字体"""
        base_font = QFont("Microsoft YaHei", 11)
        QApplication.instance().setFont(base_font)
        
        self.label_font = QFont("Microsoft YaHei", 11)
        self.input_font = QFont("Microsoft YaHei", 11)
        self.button_font = QFont("Microsoft YaHei", 12, QFont.Bold)
        self.log_font = QFont("Consolas", 10)
        self.title_font = QFont("Microsoft YaHei", 13, QFont.Bold)
        
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget_layout = QVBoxLayout(central_widget)
        central_widget_layout.addWidget(main_splitter)
        
        # 左侧控制面板
        left_panel = self.create_control_panel()
        main_splitter.addWidget(left_panel)
        
        # 右侧显示面板
        right_panel = self.create_display_panel()
        main_splitter.addWidget(right_panel)
        
        # 设置分割器比例
        main_splitter.setSizes([400, 800])
        
    def create_control_panel(self) -> QWidget:
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 模型配置组
        layout.addWidget(self.create_model_config_group())
        
        # 推理配置组
        layout.addWidget(self.create_inference_config_group())
        
        # 单图推理组
        layout.addWidget(self.create_single_inference_group())
        
        # 批量推理组
        layout.addWidget(self.create_batch_inference_group())
        
        # 控制按钮组
        layout.addWidget(self.create_control_buttons_group())
        
        layout.addStretch()
        
        return panel
        
    def create_model_config_group(self) -> QGroupBox:
        """创建模型配置组"""
        group = QGroupBox("模型配置")
        group.setFont(self.title_font)
        layout = QGridLayout(group)
        layout.setSpacing(8)
        
        # 模型路径
        label = QLabel("模型路径:")
        label.setFont(self.label_font)
        layout.addWidget(label, 0, 0)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setFont(self.input_font)
        self.model_path_edit.setPlaceholderText("自动查找最新模型或手动选择")
        layout.addWidget(self.model_path_edit, 0, 1)
        
        self.model_browse_btn = QPushButton("浏览")
        self.model_browse_btn.setFont(self.button_font)
        self.model_browse_btn.clicked.connect(self.browse_model)
        layout.addWidget(self.model_browse_btn, 0, 2)
        
        # 配置文件路径
        label = QLabel("配置文件:")
        label.setFont(self.label_font)
        layout.addWidget(label, 1, 0)
        
        self.config_path_edit = QLineEdit()
        self.config_path_edit.setFont(self.input_font)
        self.config_path_edit.setPlaceholderText("可选，使用默认配置")
        layout.addWidget(self.config_path_edit, 1, 1)
        
        self.config_browse_btn = QPushButton("浏览")
        self.config_browse_btn.setFont(self.button_font)
        self.config_browse_btn.clicked.connect(self.browse_config)
        layout.addWidget(self.config_browse_btn, 1, 2)
        
        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.setFont(self.button_font)
        self.load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_btn, 2, 0, 1, 3)
        
        return group
        
    def create_inference_config_group(self) -> QGroupBox:
        """创建推理配置组"""
        group = QGroupBox("推理配置")
        group.setFont(self.title_font)
        layout = QGridLayout(group)
        layout.setSpacing(8)
        
        # 异常检测阈值
        label = QLabel("异常阈值:")
        label.setFont(self.label_font)
        layout.addWidget(label, 0, 0)
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setFont(self.input_font)
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setDecimals(2)
        layout.addWidget(self.threshold_spin, 0, 1)
        
        return group
        
    def create_single_inference_group(self) -> QGroupBox:
        """创建单图推理组"""
        group = QGroupBox("单图推理")
        group.setFont(self.title_font)
        layout = QGridLayout(group)
        layout.setSpacing(8)
        
        # 图片路径
        label = QLabel("图片路径:")
        label.setFont(self.label_font)
        layout.addWidget(label, 0, 0)
        
        self.single_image_edit = QLineEdit()
        self.single_image_edit.setFont(self.input_font)
        layout.addWidget(self.single_image_edit, 0, 1)
        
        self.single_browse_btn = QPushButton("选择图片")
        self.single_browse_btn.setFont(self.button_font)
        self.single_browse_btn.clicked.connect(self.browse_single_image)
        layout.addWidget(self.single_browse_btn, 0, 2)
        
        # 开始推理按钮
        self.single_infer_btn = QPushButton("开始推理")
        self.single_infer_btn.setFont(self.button_font)
        self.single_infer_btn.clicked.connect(self.start_single_inference)
        layout.addWidget(self.single_infer_btn, 1, 0, 1, 3)
        
        return group
        
    def create_batch_inference_group(self) -> QGroupBox:
        """创建批量推理组"""
        group = QGroupBox("批量推理")
        group.setFont(self.title_font)
        layout = QGridLayout(group)
        layout.setSpacing(8)
        
        # 输入文件夹
        label = QLabel("输入文件夹:")
        label.setFont(self.label_font)
        layout.addWidget(label, 0, 0)
        
        self.batch_input_edit = QLineEdit()
        self.batch_input_edit.setFont(self.input_font)
        layout.addWidget(self.batch_input_edit, 0, 1)
        
        self.batch_input_btn = QPushButton("选择文件夹")
        self.batch_input_btn.setFont(self.button_font)
        self.batch_input_btn.clicked.connect(self.browse_batch_input)
        layout.addWidget(self.batch_input_btn, 0, 2)
        
        # 输出文件夹
        label = QLabel("输出文件夹:")
        label.setFont(self.label_font)
        layout.addWidget(label, 1, 0)
        
        self.batch_output_edit = QLineEdit("./inference_results")
        self.batch_output_edit.setFont(self.input_font)
        layout.addWidget(self.batch_output_edit, 1, 1)
        
        self.batch_output_btn = QPushButton("选择文件夹")
        self.batch_output_btn.setFont(self.button_font)
        self.batch_output_btn.clicked.connect(self.browse_batch_output)
        layout.addWidget(self.batch_output_btn, 1, 2)
        
        # 开始批量推理按钮
        self.batch_infer_btn = QPushButton("开始批量推理")
        self.batch_infer_btn.setFont(self.button_font)
        self.batch_infer_btn.clicked.connect(self.start_batch_inference)
        layout.addWidget(self.batch_infer_btn, 2, 0, 1, 3)
        
        return group
        
    def create_control_buttons_group(self) -> QGroupBox:
        """创建控制按钮组"""
        group = QGroupBox("操作控制")
        group.setFont(self.title_font)
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFont(self.label_font)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 停止推理按钮
        self.stop_btn = QPushButton("停止推理")
        self.stop_btn.setFont(self.button_font)
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        # 清除结果按钮
        self.clear_btn = QPushButton("清除结果")
        self.clear_btn.setFont(self.button_font)
        self.clear_btn.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_btn)
        
        layout.addLayout(button_layout)
        
        return group
        
    def create_display_panel(self) -> QWidget:
        """创建右侧显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(self.label_font)
        layout.addWidget(self.tab_widget)
        
        # 结果显示标签页
        self.result_tab = self.create_result_tab()
        self.tab_widget.addTab(self.result_tab, "推理结果")
        
        # 日志标签页
        self.log_tab = self.create_log_tab()
        self.tab_widget.addTab(self.log_tab, "日志信息")
        
        return panel
        
    def create_result_tab(self) -> QWidget:
        """创建结果显示标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 图片显示区域
        self.image_display = ImageDisplayWidget()
        layout.addWidget(self.image_display)
        
        # 结果信息显示
        self.result_info = QTextEdit()
        self.result_info.setFont(self.log_font)
        self.result_info.setMaximumHeight(150)
        self.result_info.setPlaceholderText("推理结果信息将在此显示...")
        layout.addWidget(self.result_info)
        
        return widget
        
    def create_log_tab(self) -> QWidget:
        """创建日志标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setFont(self.log_font)
        layout.addWidget(self.log_text)
        
        # 清除日志按钮
        clear_log_btn = QPushButton("清除日志")
        clear_log_btn.setFont(self.button_font)
        clear_log_btn.clicked.connect(self.clear_log)
        layout.addWidget(clear_log_btn)
        
        return widget
        
    def browse_model(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.ckpt *.pth)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            
    def browse_config(self):
        """浏览配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件", "", "JSON文件 (*.json)"
        )
        if file_path:
            self.config_path_edit.setText(file_path)
            
    def browse_single_image(self):
        """浏览单张图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "", 
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if file_path:
            self.single_image_edit.setText(file_path)
            self.image_display.display_image(file_path)
            
    def browse_batch_input(self):
        """浏览批量输入文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if folder_path:
            self.batch_input_edit.setText(folder_path)
            
    def browse_batch_output(self):
        """浏览批量输出文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder_path:
            self.batch_output_edit.setText(folder_path)
            
    def auto_load_model(self):
        """自动加载最新模型"""
        self.log("🔍 查找模型...")
        latest_model = self.inference_engine.find_latest_model()
        if latest_model:
            self.model_path_edit.setText(latest_model)
            self.log(f"🤖 找到最新模型: {Path(latest_model).name}")
        else:
            self.log("⚠️ 未找到训练好的模型")
            
    def load_model(self):
        """加载模型"""
        model_path = self.model_path_edit.text().strip()
        config_path = self.config_path_edit.text().strip()
        
        if model_path:
            self.inference_engine.model_path = model_path
            
        # 设置阈值
        threshold = self.threshold_spin.value()
        self.inference_engine.set_threshold(threshold)
        
        # 设置回调函数
        self.inference_engine.log_callback = self.log
        
        # 加载模型
        success = self.inference_engine.load_model(config_path if config_path else None)
        
        if success:
            QMessageBox.information(self, "成功", "模型加载成功！")
        else:
            QMessageBox.warning(self, "错误", "模型加载失败！")
            
    def start_single_inference(self):
        """开始单图推理"""
        if not self.inference_engine.model:
            QMessageBox.warning(self, "错误", "请先加载模型！")
            return
            
        image_path = self.single_image_edit.text().strip()
        if not image_path or not Path(image_path).exists():
            QMessageBox.warning(self, "错误", "请选择有效的图片文件！")
            return
            
        # 更新阈值
        threshold = self.threshold_spin.value()
        self.inference_engine.set_threshold(threshold)
        
        # 启动推理线程
        self.start_inference_thread('single', image_path)
        
    def start_batch_inference(self):
        """开始批量推理"""
        if not self.inference_engine.model:
            QMessageBox.warning(self, "错误", "请先加载模型！")
            return
            
        input_path = self.batch_input_edit.text().strip()
        output_path = self.batch_output_edit.text().strip()
        
        if not input_path or not Path(input_path).exists():
            QMessageBox.warning(self, "错误", "请选择有效的输入文件夹！")
            return
            
        if not output_path:
            QMessageBox.warning(self, "错误", "请指定输出文件夹！")
            return
            
        # 更新阈值
        threshold = self.threshold_spin.value()
        self.inference_engine.set_threshold(threshold)
        
        # 启动推理线程
        self.start_inference_thread('batch', input_path, output_path)
        
    def start_inference_thread(self, mode, input_path, output_dir=None):
        """启动推理线程"""
        # 更新界面状态
        self.single_infer_btn.setEnabled(False)
        self.batch_infer_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建推理线程
        self.inference_thread = InferenceThread(
            self.inference_engine, mode, input_path, output_dir
        )
        
        # 连接信号
        self.inference_thread.log_signal.connect(self.log)
        self.inference_thread.progress_signal.connect(self.update_progress)
        self.inference_thread.result_signal.connect(self.display_results)
        self.inference_thread.finished_signal.connect(self.inference_finished)
        
        # 启动线程
        self.inference_thread.start()
        
    def stop_inference(self):
        """停止推理"""
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.terminate()
            self.inference_thread.wait()
            self.inference_finished(False, "推理已被用户停止")
            
    def inference_finished(self, success, message):
        """推理完成回调"""
        # 恢复界面状态
        self.single_infer_btn.setEnabled(True)
        self.batch_infer_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if success:
            self.log(f"✅ {message}")
        else:
            self.log(f"❌ {message}")
            QMessageBox.warning(self, "推理完成", message)
            
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def display_results(self, results):
        """显示推理结果"""
        if isinstance(results, dict):
            # 单图结果
            self.display_single_result(results)
        elif isinstance(results, list):
            # 批量结果
            self.display_batch_results(results)
            
        # 切换到结果标签页
        self.tab_widget.setCurrentIndex(0)
        
    def display_single_result(self, result):
        """显示单图推理结果"""
        # 显示图片
        image_path = result["image_path"]
        self.image_display.display_image(image_path)
        
        # 显示结果信息
        info = f"""📷 图片: {Path(image_path).name}
🔍 异常分数: {result['anomaly_score']:.6f}
🎯 检测阈值: {result['threshold']:.2f}
📊 预测结果: {'🔴 异常' if result['is_anomaly'] else '🟢 正常'}
📐 图片尺寸: {result['original_size'][0]} × {result['original_size'][1]}
"""
        self.result_info.setPlainText(info)
        
        # 保存当前结果
        self.current_results = [result]
        
    def display_batch_results(self, results):
        """显示批量推理结果"""
        # 清除图片显示
        self.image_display.clear_display()
        self.image_display.setText(f"批量推理完成\n共处理 {len(results)} 张图片")
        
        # 统计结果
        normal_count = sum(1 for r in results if not r["is_anomaly"])
        anomaly_count = len(results) - normal_count
        avg_score = sum(r["anomaly_score"] for r in results) / len(results)
        
        info = f"""📁 批量推理结果
📊 总计图片: {len(results)} 张
🟢 正常图片: {normal_count} 张
🔴 异常图片: {anomaly_count} 张
📈 异常率: {anomaly_count/len(results)*100:.1f}%
📊 平均分数: {avg_score:.6f}
🎯 检测阈值: {results[0]['threshold']:.2f}

异常图片列表:
"""
        # 添加异常图片列表
        anomaly_images = [r for r in results if r["is_anomaly"]]
        for i, r in enumerate(anomaly_images[:10], 1):  # 最多显示10个
            info += f"{i}. {Path(r['image_path']).name} (分数: {r['anomaly_score']:.4f})\n"
            
        if len(anomaly_images) > 10:
            info += f"... 还有 {len(anomaly_images) - 10} 个异常图片"
            
        self.result_info.setPlainText(info)
        
        # 保存当前结果
        self.current_results = results
        
    def clear_results(self):
        """清除结果"""
        self.image_display.clear_display()
        self.result_info.clear()
        self.current_results = []
        self.log("🧹 结果已清除")
        
    def log(self, message):
        """添加日志"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
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
    
    # 设置应用信息
    app.setApplicationName("PatchCore推理工具")
    app.setApplicationVersion("1.0")
    
    # 创建并显示主窗口
    window = InferenceGUI()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 