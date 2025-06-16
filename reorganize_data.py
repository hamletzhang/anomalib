# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""重新组织自定义数据集以适应 Anomalib 的 Folder 数据模块

这个脚本将原始的 OK 和 NG 数据重新组织成 Anomalib 期望的目录结构：
- train/normal/: 训练用的正常样本
- test/normal/: 测试用的正常样本  
- test/abnormal/: 测试用的异常样本
- test/ALL_OK/: 全部正常样本（用于全量测试）
- test/ALL_NG/: 全部异常样本（用于全量测试）
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import random

def detect_image_format(file_path):
    """检测图像文件的实际格式"""
    try:
        with Image.open(file_path) as img:
            return img.format.lower()
    except Exception as e:
        print(f"无法打开图像文件 {file_path}: {e}")
        return None

def copy_and_rename_files(source_dir, target_dir, prefix=""):
    """复制文件并添加正确的扩展名"""
    if not os.path.exists(source_dir):
        print(f"⚠️  源目录不存在: {source_dir}")
        return 0
    
    os.makedirs(target_dir, exist_ok=True)
    
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    print(f"📁 处理 {source_dir} -> {target_dir} ({len(files)} 个文件)")
    
    copied_count = 0
    for filename in files:
        source_path = os.path.join(source_dir, filename)
        
        # 检测图像格式
        image_format = detect_image_format(source_path)
        if image_format is None:
            continue
            
        # 根据检测到的格式添加扩展名
        if image_format == 'jpeg':
            extension = '.jpg'
        elif image_format == 'png':
            extension = '.png'
        elif image_format == 'bitmap':
            extension = '.bmp'
        elif image_format == 'tiff':
            extension = '.tiff'
        else:
            extension = f'.{image_format}'
        
        # 创建目标文件名
        if prefix:
            target_filename = f"{prefix}_{copied_count + 1}{extension}"
        else:
            base_name = Path(filename).stem
            target_filename = f"{base_name}{extension}"
        
        target_path = os.path.join(target_dir, target_filename)
        
        # 复制文件
        try:
            shutil.copy2(source_path, target_path)
            copied_count += 1
        except Exception as e:
            print(f"❌ 复制文件失败 {source_path}: {e}")
    
    print(f"✅ 成功复制 {copied_count} 个文件到 {target_dir}")
    return copied_count

def main():
    print("🚀 开始重新组织数据集...")
    
    # 新的数据源路径
    source_ok_dir = r"C:\Users\Administrator\Desktop\自监督测试\ALL_OK"
    source_ng_dir = r"C:\Users\Administrator\Desktop\自监督测试\ALL_NG"
    target_root = "project_test_organized"
    
    # 检查源数据是否存在
    if not os.path.exists(source_ok_dir):
        print(f"❌ 正常样本目录不存在: {source_ok_dir}")
        return
    
    if not os.path.exists(source_ng_dir):
        print(f"❌ 异常样本目录不存在: {source_ng_dir}")
        return
    
    # 统计原始数据
    ok_files = [f for f in os.listdir(source_ok_dir) if os.path.isfile(os.path.join(source_ok_dir, f))]
    ng_files = [f for f in os.listdir(source_ng_dir) if os.path.isfile(os.path.join(source_ng_dir, f))]
    
    print(f"📊 原始数据统计:")
    print(f"   正常样本 (OK): {len(ok_files)} 张")
    print(f"   异常样本 (NG): {len(ng_files)} 张")
    print(f"   总计: {len(ok_files) + len(ng_files)} 张")
    
    # 清理旧的目标目录
    if os.path.exists(target_root):
        print(f"🗑️  清理旧的目标目录: {target_root}")
        shutil.rmtree(target_root)
    
    # 创建目标目录结构
    train_normal_dir = os.path.join(target_root, "train", "normal")
    test_normal_dir = os.path.join(target_root, "test", "normal") 
    test_abnormal_dir = os.path.join(target_root, "test", "abnormal")
    test_all_ok_dir = os.path.join(target_root, "test", "ALL_OK")
    test_all_ng_dir = os.path.join(target_root, "test", "ALL_NG")
    
    # 创建目录
    for dir_path in [train_normal_dir, test_normal_dir, test_abnormal_dir, test_all_ok_dir, test_all_ng_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("📁 创建目录结构完成")
    
    # 随机分割正常样本：70% 训练，30% 测试
    random.seed(42)  # 设置随机种子以确保可重复性
    random.shuffle(ok_files)
    
    train_split = int(len(ok_files) * 0.7)
    train_ok_files = ok_files[:train_split]
    test_ok_files = ok_files[train_split:]
    
    print(f"📊 正常样本分割:")
    print(f"   训练集: {len(train_ok_files)} 张")
    print(f"   测试集: {len(test_ok_files)} 张")
    
    # 复制训练用正常样本
    print("\n🔄 复制训练用正常样本...")
    train_count = 0
    for filename in train_ok_files:
        source_path = os.path.join(source_ok_dir, filename)
        image_format = detect_image_format(source_path)
        if image_format is None:
            continue
            
        if image_format == 'jpeg':
            extension = '.jpg'
        elif image_format == 'png':
            extension = '.png'
        elif image_format == 'bitmap':
            extension = '.bmp'
        elif image_format == 'tiff':
            extension = '.tiff'
        else:
            extension = f'.{image_format}'
        
        target_filename = f"OK_{train_count + 1}{extension}"
        target_path = os.path.join(train_normal_dir, target_filename)
        
        try:
            shutil.copy2(source_path, target_path)
            train_count += 1
        except Exception as e:
            print(f"❌ 复制文件失败 {source_path}: {e}")
    
    # 复制测试用正常样本
    print("\n🔄 复制测试用正常样本...")
    test_normal_count = 0
    for filename in test_ok_files:
        source_path = os.path.join(source_ok_dir, filename)
        image_format = detect_image_format(source_path)
        if image_format is None:
            continue
            
        if image_format == 'jpeg':
            extension = '.jpg'
        elif image_format == 'png':
            extension = '.png'
        elif image_format == 'bitmap':
            extension = '.bmp'
        elif image_format == 'tiff':
            extension = '.tiff'
        else:
            extension = f'.{image_format}'
        
        target_filename = f"OK_test_{test_normal_count + 1}{extension}"
        target_path = os.path.join(test_normal_dir, target_filename)
        
        try:
            shutil.copy2(source_path, target_path)
            test_normal_count += 1
        except Exception as e:
            print(f"❌ 复制文件失败 {source_path}: {e}")
    
    # 复制测试用异常样本
    print("\n🔄 复制测试用异常样本...")
    abnormal_count = copy_and_rename_files(source_ng_dir, test_abnormal_dir, "NG")
    
    # 复制全部正常样本（用于全量测试）
    print("\n🔄 复制全部正常样本到 ALL_OK...")
    all_ok_count = copy_and_rename_files(source_ok_dir, test_all_ok_dir, "OK")
    
    # 复制全部异常样本（用于全量测试）
    print("\n🔄 复制全部异常样本到 ALL_NG...")
    all_ng_count = copy_and_rename_files(source_ng_dir, test_all_ng_dir, "NG")
    
    print("\n" + "=" * 60)
    print("✅ 数据重组完成!")
    print("=" * 60)
    print(f"📂 目标目录: {target_root}")
    print(f"📊 最终统计:")
    print(f"   train/normal/: {train_count} 张")
    print(f"   test/normal/: {test_normal_count} 张")
    print(f"   test/abnormal/: {abnormal_count} 张")
    print(f"   test/ALL_OK/: {all_ok_count} 张 (全部正常样本)")
    print(f"   test/ALL_NG/: {all_ng_count} 张 (全部异常样本)")
    print()
    print("现在可以运行训练脚本: python train_custom_data.py")
    print("=" * 60)

if __name__ == "__main__":
    main() 