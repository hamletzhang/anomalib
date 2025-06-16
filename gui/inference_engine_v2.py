#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore推理引擎 V2

基于anomalib Engine.predict()方法的正确实现，确保分数和热力图解码的准确性。
"""

import os
import sys
import cv2
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple
from PIL import Image
import torch
import tempfile

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from anomalib.data import PredictDataset, Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"警告: 无法导入部分模块: {e}")


class InferenceEngineV2:
    """PatchCore推理引擎V2 - 基于Engine.predict()的正确实现"""
    
    def __init__(self, model_path: str = None, debug: bool = False):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型路径，如果为None则使用最新模型
            debug: 是否启用调试模式
        """
        self.model_path = model_path
        self.model = None
        self.engine = None
        self.log_callback: Optional[Callable[[str], None]] = None
        self.progress_callback: Optional[Callable[[int], None]] = None
        self.debug = debug
        
        # 推理配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = 0.5  # 异常检测阈值（用于GUI显示，实际阈值由模型决定）
        
    def debug_log(self, message: str):
        """调试日志输出"""
        if self.debug:
            print(f"[DEBUG] {message}")
        
    def log(self, message: str):
        """记录日志消息"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
            
    def update_progress(self, value: int):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(value)
            
    def find_latest_model(self) -> Optional[str]:
        """查找最新的训练模型"""
        try:
            results_dir = Path("./results")
            if not results_dir.exists():
                # 检查gui/results目录
                gui_results_dir = Path("./gui/results")
                if gui_results_dir.exists():
                    results_dir = gui_results_dir
                else:
                    self.log("❌ 结果目录不存在")
                    return None
                
            # 查找PatchCore模型目录
            patchcore_dirs = list(results_dir.glob("**/Patchcore/**/weights"))
            if not patchcore_dirs:
                self.log("❌ 未找到PatchCore模型")
                return None
                
            # 找到最新的模型
            latest_dir = max(patchcore_dirs, key=lambda x: x.stat().st_mtime)
            model_files = list(latest_dir.glob("*.ckpt"))
            
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                self.log(f"🤖 找到模型: {latest_model}")
                return str(latest_model)
            else:
                self.log("❌ 模型目录中未找到.ckpt文件")
                return None
                
        except Exception as e:
            self.log(f"❌ 查找模型失败: {str(e)}")
            return None
            
    def load_model(self, config_path: str = None) -> bool:
        """加载训练好的模型"""
        try:
            self.log("🔄 加载模型...")
            
            # 确定模型路径
            if not self.model_path:
                self.model_path = self.find_latest_model()
                
            if not self.model_path or not Path(self.model_path).exists():
                self.log("❌ 模型文件不存在")
                return False
                
            self.debug_log(f"模型路径: {self.model_path}")
            
            # **关键修复：使用标准的anomalib方式加载模型**
            # 1. 创建Patchcore模型实例
            self.model = Patchcore()
            self.debug_log("已创建Patchcore模型实例")
            
            # 2. 创建Engine实例
            self.engine = Engine()
            self.debug_log("已创建Engine实例")
            
            # 3. 验证模型文件
            if not Path(self.model_path).exists():
                self.log(f"❌ 模型文件不存在: {self.model_path}")
                return False
                
            self.debug_log("模型验证通过")
            self.log("✅ 模型加载成功")
            return True
            
        except Exception as e:
            self.log(f"❌ 模型加载失败: {str(e)}")
            self.debug_log(f"模型加载错误详情: {str(e)}")
            import traceback
            if self.debug:
                traceback.print_exc()
            return False
            
    def predict_folder(self, folder_path: str, output_dir: str = "./inference_results") -> List[Dict]:
        """
        对文件夹中的所有图片进行批量推理
        
        Args:
            folder_path: 图片文件夹路径
            output_dir: 结果输出目录
            
        Returns:
            List[Dict]: 所有图片的预测结果列表
        """
        try:
            self.log(f"📁 批量推理文件夹: {folder_path}")
            
            folder_path = Path(folder_path)
            if not folder_path.exists():
                self.log("❌ 文件夹不存在")
                return []
                
            # 支持的图片格式
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(folder_path.glob(f"*{ext}"))
                image_files.extend(folder_path.glob(f"*{ext.upper()}"))
                
            if not image_files:
                self.log("❌ 文件夹中没有找到图片文件")
                return []
                
            self.log(f"📊 找到 {len(image_files)} 张图片")
            
            # 创建输出目录
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results = []
            normal_count = 0
            anomaly_count = 0
            
            for i, image_file in enumerate(image_files):
                # 更新进度
                progress = int((i / len(image_files)) * 100)
                self.update_progress(progress)
                
                # 推理单张图片
                result = self.predict_single_image(str(image_file))
                if result:
                    results.append(result)
                    
                    if result["is_anomaly"]:
                        anomaly_count += 1
                    else:
                        normal_count += 1
                        
                    # 生成可视化结果
                    self.generate_visualization(result, output_dir)
                    
            # 完成统计
            self.update_progress(100)
            self.log("=" * 50)
            self.log("📊 批量推理完成!")
            self.log(f"   总计: {len(results)} 张图片")
            self.log(f"   正常: {normal_count} 张")
            self.log(f"   异常: {anomaly_count} 张")
            self.log(f"   异常率: {anomaly_count/len(results)*100:.1f}%")
            self.log(f"   结果保存在: {output_dir}")
            self.log("=" * 50)
            
            # 保存结果摘要
            self.save_summary(results, output_dir)
            
            return results
            
        except Exception as e:
            self.log(f"❌ 批量推理失败: {str(e)}")
            return []

    def save_summary(self, results: List[Dict], output_dir: Path):
        """保存推理结果摘要"""
        try:
            import json
            import time
            import numpy as np
            
            summary_file = output_dir / "inference_summary.json"
            
            # 统计信息
            total_count = len(results)
            anomaly_count = sum(1 for r in results if r["is_anomaly"])
            normal_count = total_count - anomaly_count
            
            # 分数统计
            scores = [r["anomaly_score"] for r in results]
            avg_score = np.mean(scores) if scores else 0
            max_score = np.max(scores) if scores else 0
            min_score = np.min(scores) if scores else 0
            
            summary = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_images": total_count,
                "normal_count": normal_count,
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_count / total_count * 100 if total_count > 0 else 0,
                "threshold": self.threshold,
                "score_statistics": {
                    "average": float(avg_score),
                    "maximum": float(max_score),
                    "minimum": float(min_score)
                },
                "detailed_results": [
                    {
                        "image_name": Path(r["image_path"]).name,
                        "anomaly_score": r["anomaly_score"],
                        "is_anomaly": r["is_anomaly"],
                        "pred_label": r["pred_label"]
                    }
                    for r in results
                ]
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            self.log(f"📋 摘要已保存: {summary_file}")
            
        except Exception as e:
            self.log(f"⚠️ 保存摘要失败: {str(e)}")

    def predict_single_image(self, image_path: str) -> Dict:
        """
        对单张图片进行推理 - 使用Engine.predict()方法
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict: 包含预测结果的字典
        """
        try:
            self.log(f"🔍 推理图片: {image_path}")
            
            if not self.model or not self.engine:
                self.log("❌ 模型未加载")
                return {}
                
            # 检查图片是否存在
            if not Path(image_path).exists():
                self.log(f"❌ 图片不存在: {image_path}")
                return {}
                
            self.debug_log(f"开始处理图片: {image_path}")
            
            # **关键修复：使用PredictDataset和Engine.predict()方法**
            # 1. 创建PredictDataset
            dataset = PredictDataset(path=image_path)
            dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn)
            self.debug_log("已创建PredictDataset和DataLoader")
            
            # 2. 使用Engine.predict进行推理
            self.debug_log("开始Engine.predict推理...")
            predictions = self.engine.predict(
                model=self.model,
                dataloaders=[dataloader],
                ckpt_path=self.model_path,
                return_predictions=True,
            )
            
            if not predictions or len(predictions) == 0:
                self.log("❌ 推理返回空结果")
                return {}
                
            # 3. 解析推理结果
            batch = predictions[0]  # 获取第一个batch
            self.debug_log(f"推理结果batch类型: {type(batch)}")
            self.debug_log(f"batch属性: {dir(batch)}")
            
            # 获取结果
            pred_score = batch.pred_score[0].item()  # 获取第一张图片的分数
            pred_label = batch.pred_label[0].item()  # 获取第一张图片的标签
            anomaly_map = batch.anomaly_map[0].cpu().numpy()  # 获取异常图
            
            self.debug_log(f"pred_score: {pred_score}")
            self.debug_log(f"pred_label: {pred_label}")
            self.debug_log(f"anomaly_map形状: {anomaly_map.shape}")
            self.debug_log(f"anomaly_map范围: {anomaly_map.min():.4f} ~ {anomaly_map.max():.4f}")
            
            # 获取原始图像尺寸
            image = Image.open(image_path)
            original_size = image.size
            
            # 处理异常图：确保是2D数组
            if anomaly_map.ndim == 3 and anomaly_map.shape[0] == 1:
                anomaly_map = anomaly_map[0]  # 移除第一个维度
            elif anomaly_map.ndim == 3:
                anomaly_map = anomaly_map.mean(axis=0)  # 如果有多个通道，取平均
            
            self.debug_log(f"处理后anomaly_map形状: {anomaly_map.shape}")
            
            # 构建结果字典
            result = {
                "image_path": image_path,
                "anomaly_score": float(pred_score),
                "pred_label": int(pred_label),
                "is_anomaly": bool(pred_label),  # pred_label直接表示是否异常
                "anomaly_map": anomaly_map,
                "original_size": original_size,
                "threshold": self.threshold,  # GUI显示用的阈值
            }
            
            self.log(f"   异常分数: {pred_score:.4f}")
            self.log(f"   预测结果: {'异常' if pred_label else '正常'}")
            
            return result
            
        except Exception as e:
            self.log(f"❌ 图片推理失败: {str(e)}")
            self.debug_log(f"推理错误详情: {str(e)}")
            import traceback
            if self.debug:
                traceback.print_exc()
            return {}
            
    def generate_visualization(self, result: Dict, output_dir: Path):
        """生成可视化结果图"""
        try:
            image_path = Path(result["image_path"])
            image_name = image_path.stem
            
            self.debug_log(f"开始生成可视化: {image_name}")
            
            # 读取原始图片
            original_image = cv2.imread(str(image_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # 获取异常热力图
            anomaly_map = result["anomaly_map"]
            self.debug_log(f"异常图原始形状: {anomaly_map.shape}")
            
            # 确保异常图是2D
            if anomaly_map.ndim != 2:
                self.debug_log(f"警告: 异常图维度不是2D: {anomaly_map.shape}")
                return
            
            # 调整热力图大小到原始图片尺寸
            h, w = original_image.shape[:2]
            self.debug_log(f"原始图片尺寸: {h}x{w}")
                
            anomaly_map_resized = cv2.resize(anomaly_map, (w, h))
            self.debug_log(f"缩放后异常图形状: {anomaly_map_resized.shape}")
            
            # 创建可视化图像
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始图片
            axes[0].imshow(original_image)
            axes[0].set_title("原始图片")
            axes[0].axis('off')
            
            # 异常热力图
            im = axes[1].imshow(anomaly_map_resized, cmap='jet')
            axes[1].set_title(f"异常热力图 (分数: {result['anomaly_score']:.4f})")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # 叠加图
            alpha = 0.4
            overlay = original_image.copy().astype(np.float32) / 255.0
            
            # 标准化异常图到[0,1]范围
            anomaly_map_norm = (anomaly_map_resized - anomaly_map_resized.min()) / (anomaly_map_resized.max() - anomaly_map_resized.min() + 1e-8)
            heatmap_colored = plt.cm.jet(anomaly_map_norm)[:, :, :3]  # 只取RGB通道
            
            overlay = (1-alpha) * overlay + alpha * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            
            axes[2].imshow(overlay)
            
            # 添加预测结果文本
            status = "异常" if result["is_anomaly"] else "正常"
            color = "red" if result["is_anomaly"] else "green"
            axes[2].set_title(f"叠加结果: {status}", color=color, fontweight='bold')
            axes[2].axis('off')
            
            # 保存图片
            output_file = output_dir / f"{image_name}_result.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"💾 可视化结果已保存: {output_file.name}")
            
        except Exception as e:
            self.log(f"⚠️ 生成可视化失败: {str(e)}")
            self.debug_log(f"可视化错误详情: {str(e)}")
            import traceback
            if self.debug:
                traceback.print_exc()
            
    def set_threshold(self, threshold: float):
        """设置异常检测阈值（仅用于GUI显示）"""
        self.threshold = threshold
        self.log(f"🎚️ GUI显示阈值设置为: {threshold}")
        
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "threshold": self.threshold,
            "model_type": "PatchCore-V2"
        }


def main():
    """测试函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="PatchCore推理引擎V2测试")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--test-viz", action="store_true", help="测试可视化功能")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 PatchCore 推理引擎V2测试")
    if args.debug:
        print("🐛 调试模式已启用")
    print("=" * 60)
    
    # 指定模型和测试图片路径
    model_path = r"C:\Users\Administrator\Desktop\anomalib\gui\results\Patchcore\custom_dataset\v0\weights\lightning\model.ckpt"
    
    # 测试两张图片：一张异常，一张正常
    test_images = [
        r"C:\Users\Administrator\Desktop\anomalib\project_test_organized\test\abnormal\NG_7.png",
        r"C:\Users\Administrator\Desktop\anomalib\project_test_organized\test\normal\OK_test_3.png"
    ]
    
    print(f"📍 模型路径: {model_path}")
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 创建推理引擎V2
    engine = InferenceEngineV2(model_path=model_path, debug=args.debug)
    
    # 设置回调函数
    def log_callback(message):
        print(f"[LOG] {message}")
        
    def progress_callback(percent):
        print(f"[PROGRESS] {percent}%")
        
    engine.log_callback = log_callback
    engine.progress_callback = progress_callback
    
    print("\n🔄 开始加载模型...")
    # 加载模型
    if engine.load_model():
        print("\n🎯 开始多图推理测试...")
        
        # 创建测试输出目录
        test_output_dir = Path("./test_inference_results_v2")
        test_output_dir.mkdir(exist_ok=True)
        
        for i, test_image in enumerate(test_images, 1):
            print(f"\n{'='*50}")
            print(f"📷 测试图片 {i}: {Path(test_image).name}")
            print(f"   路径: {test_image}")
            
            if not Path(test_image).exists():
                print(f"❌ 测试图片不存在: {test_image}")
                continue
                
            result = engine.predict_single_image(test_image)
            
            if result:
                print("\n📊 推理结果:")
                print("-" * 30)
                print(f"异常分数: {result['anomaly_score']:.6f}")
                print(f"预测标签: {result['pred_label']}")
                print(f"是否异常: {'是' if result['is_anomaly'] else '否'}")
                print(f"原始尺寸: {result['original_size']}")
                print(f"异常图形状: {result['anomaly_map'].shape}")
                print("-" * 30)
                
                # 测试可视化功能
                if args.test_viz:
                    print("\n🎨 测试可视化功能...")
                    engine.generate_visualization(result, test_output_dir)
                    
            else:
                print("❌ 推理失败")
                
        print("\n" + "="*60)
        print("✅ 多图推理测试完成!")
        if args.test_viz:
            print(f"📁 可视化结果保存在: {test_output_dir}")
    else:
        print("❌ 模型加载失败")
        
    print("\n🏁 测试结束")


if __name__ == "__main__":
    main()
