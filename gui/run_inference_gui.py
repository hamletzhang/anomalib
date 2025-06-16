#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore推理GUI启动脚本

检查运行环境和依赖项，然后启动推理界面。
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    """检查依赖项是否安装"""
    missing_deps = []
    
    # 检查PyQt5
    try:
        import PyQt5
        print("✅ PyQt5 已安装")
    except ImportError:
        missing_deps.append("PyQt5")
        print("❌ PyQt5 未安装")
    
    # 检查anomalib
    try:
        import anomalib
        print("✅ anomalib 已安装")
    except ImportError:
        missing_deps.append("anomalib")
        print("❌ anomalib 未安装")
        
    # 检查torch
    try:
        import torch
        print(f"✅ PyTorch 已安装 (版本: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"🚀 CUDA 可用，GPU数量: {torch.cuda.device_count()}")
        else:
            print("💻 使用CPU模式")
    except ImportError:
        missing_deps.append("torch")
        print("❌ PyTorch 未安装")
        
    # 检查其他依赖
    other_deps = [
        ('opencv-python', 'cv2'),
        ('matplotlib', 'matplotlib'),
        ('numpy', 'numpy'),
        ('Pillow', 'PIL')
    ]
    
    for dep_name, import_name in other_deps:
        try:
            __import__(import_name)
            print(f"✅ {dep_name} 已安装")
        except ImportError:
            missing_deps.append(dep_name)
            print(f"❌ {dep_name} 未安装")
    
    return missing_deps


def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"🐍 Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("⚠️ 警告: Python版本过低，建议使用Python 3.7+")
        
    # 检查当前工作目录
    current_dir = Path.cwd()
    print(f"📁 当前目录: {current_dir}")
    
    # 检查是否在正确的目录
    gui_dir = current_dir / "gui" if current_dir.name != "gui" else current_dir
    if not gui_dir.exists():
        print("❌ 错误: 请在项目根目录或gui目录下运行此脚本")
        return False
        
    # 检查必要文件
    required_files = [
        "inference_gui.py",
        "inference_engine.py",
    ]
    
    for file_name in required_files:
        file_path = gui_dir / file_name
        if file_path.exists():
            print(f"✅ 找到文件: {file_name}")
        else:
            print(f"❌ 缺少文件: {file_name}")
            return False
            
    # 检查模型文件
    results_dir = current_dir / "results"
    if results_dir.exists():
        print(f"✅ 结果目录存在: {results_dir}")
    else:
        print("⚠️ 警告: 结果目录不存在，请先训练模型")
        
    return True


def install_missing_dependencies(missing_deps):
    """提示安装缺失的依赖"""
    if not missing_deps:
        return
        
    print("\n" + "="*50)
    print("⚠️ 发现缺失的依赖项，请先安装:")
    print("="*50)
    
    # 根据环境给出安装建议
    print("使用conda环境 (推荐):")
    print("conda activate anomalib")
    for dep in missing_deps:
        if dep == "anomalib":
            print("pip install anomalib")
        elif dep == "PyQt5":
            print("pip install PyQt5")
        elif dep == "torch":
            print("conda install pytorch torchvision torchaudio -c pytorch")
        else:
            print(f"pip install {dep}")
            
    print("\n或使用pip:")
    print("pip install " + " ".join(missing_deps))
    print("="*50)


def setup_environment():
    """设置环境变量和路径"""
    # 添加gui目录到Python路径
    gui_dir = Path(__file__).parent
    if str(gui_dir) not in sys.path:
        sys.path.insert(0, str(gui_dir))
        
    # 设置工作目录
    if gui_dir.name == "gui":
        os.chdir(gui_dir)
        print(f"📂 切换到gui目录: {gui_dir}")


def main():
    """主函数"""
    print("🚀 PatchCore推理GUI启动器")
    print("="*50)
    
    # 检查运行环境
    if not check_environment():
        print("❌ 环境检查失败")
        input("按回车键退出...")
        return 1
        
    # 检查依赖项
    print("\n🔍 检查依赖项...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        install_missing_dependencies(missing_deps)
        print("\n❌ 请安装缺失的依赖项后重新运行")
        input("按回车键退出...")
        return 1
        
    print("\n✅ 所有依赖项检查通过")
    
    # 设置环境
    setup_environment()
    
    # 启动GUI
    print("\n🎯 启动推理界面...")
    try:
        # 这里需要确保能导入inference_gui模块
        from inference_gui import main as gui_main
        gui_main()
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序退出")
        return 0
        
    except Exception as e:
        print(f"\n❌ 启动失败: {str(e)}")
        print("\n调试信息:")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")
        return 1


if __name__ == "__main__":
    sys.exit(main())