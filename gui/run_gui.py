#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore训练GUI启动脚本

启动PatchCore训练的图形用户界面
"""

import sys
import os
from pathlib import Path

# 确保PyQt5可用
try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from PyQt5.QtCore import Qt
except ImportError:
    print("错误: 无法导入PyQt5")
    print("请安装PyQt5:")
    print("pip install PyQt5")
    sys.exit(1)

# 确保anomalib可用
try:
    import anomalib
    print(f"使用anomalib版本: {anomalib.__version__}")
except ImportError:
    print("错误: 无法导入anomalib")
    print("请确保已正确安装anomalib并激活相应的conda环境")
    print("conda activate anomalib")
    sys.exit(1)

# 导入GUI主程序
try:
    from patchcore_gui import PatchCoreGUI
except ImportError as e:
    print(f"错误: 无法导入GUI模块: {e}")
    sys.exit(1)


def check_dependencies():
    """检查依赖项"""
    dependencies = [
        ("PyQt5", "PyQt5"),
        ("anomalib", "anomalib"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
    ]
    
    missing = []
    for name, module in dependencies:
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    if missing:
        print(f"缺少依赖项: {', '.join(missing)}")
        print("请安装缺少的依赖项后重试")
        return False
    
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("PatchCore 训练 GUI 工具")
    print("=" * 60)
    
    # 检查依赖项
    print("检查依赖项...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ 所有依赖项已安装")
    
    # 检查数据目录
    parent_dir = Path(__file__).parent.parent
    data_dir = parent_dir / "project_test_organized"
    if data_dir.exists():
        print(f"✅ 找到数据目录: {data_dir}")
    else:
        print(f"⚠️  未找到默认数据目录: {data_dir}")
        print("请确保数据目录存在，或在界面中指定正确的数据路径")
    
    # 创建并运行应用
    app = QApplication(sys.argv)
    
    # 设置应用属性
    app.setApplicationName("PatchCore训练工具")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Anomalib")
    
    # 设置高DPI缩放
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    try:
        # 创建主窗口
        window = PatchCoreGUI()
        window.show()
        
        print("✅ GUI界面已启动")
        print("请在图形界面中配置训练参数")
        
        # 运行应用
        sys.exit(app.exec_())
        
    except Exception as e:
        QMessageBox.critical(None, "启动错误", f"GUI启动失败: {e}")
        print(f"❌ GUI启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 