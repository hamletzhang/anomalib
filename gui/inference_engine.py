#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore推理引擎

实现单图片和批量推理功能，支持可视化结果生成和展示。
"""

import os
import sys
import cv2
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple
from PIL import Image
import torch

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from anomalib.data import PredictDataset
    from anomalib.deploy import ExportType, OpenVINOInferencer
    from anomalib.models import Patchcore
    from anomalib.data import Folder
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError as e:
    print(f"警告: 无法导入部分模块: {e}")


class InferenceEngine:
    """PatchCore推理引擎类"""
    
    def __init__(self, model_path: str = None):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型路径，如果为None则使用最新模型
        """
        self.model_path = model_path
        self.model = None
        self.datamodule = None
        self.log_callback: Optional[Callable[[str], None]] = None
        self.progress_callback: Optional[Callable[[int], None]] = None
        self.prediction_results = []
        
        # 推理配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = 0.5  # 异常检测阈值
        
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
                
            # 加载配置
            if config_path and Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.log(f"📋 加载配置: {config_path}")
            else:
                # 使用默认配置
                config = {
                    "backbone": "wide_resnet50_2",
                    "layers": ["layer2", "layer3"],
                    "coreset_sampling_ratio": 0.1,
                    "num_neighbors": 9
                }
                self.log("📋 使用默认配置")
                
            # 创建模型
            self.model = Patchcore(
                backbone=config["backbone"],
                layers=config["layers"],
                pre_trained=True,
                coreset_sampling_ratio=config["coreset_sampling_ratio"],
                num_neighbors=config["num_neighbors"],
            )
            
            # 加载模型权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self.model.to(self.device)
            
            self.log("✅ 模型加载成功")
            return True
            
        except Exception as e:
            self.log(f"❌ 模型加载失败: {str(e)}")
            return False
            
    def predict_single_image(self, image_path: str) -> Dict:
        """
        对单张图片进行推理
        
        Args:
            image_path: 图片路径
            
        Returns:
            Dict: 包含预测结果的字典
        """
        try:
            self.log(f"🔍 推理图片: {image_path}")
            
            if not self.model:
                self.log("❌ 模型未加载")
                return {}
                
            # 加载图片
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            
            # 预处理图片（这里需要与训练时的预处理保持一致）
            # 通常包括resize、normalize等
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            tensor_image = transform(image).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                predictions = self.model(tensor_image)
                
            # 处理结果
            anomaly_score = predictions["anomaly_score"].cpu().numpy()[0]
            anomaly_map = predictions["anomaly_map"].cpu().numpy()[0]
            pred_label = predictions["pred_label"].cpu().numpy()[0]
            
            # 判断是否异常
            is_anomaly = anomaly_score > self.threshold
            
            result = {
                "image_path": image_path,
                "anomaly_score": float(anomaly_score),
                "pred_label": int(pred_label),
                "is_anomaly": bool(is_anomaly),
                "anomaly_map": anomaly_map,
                "original_size": original_size,
                "threshold": self.threshold
            }
            
            self.log(f"   异常分数: {anomaly_score:.4f}")
            self.log(f"   预测结果: {'异常' if is_anomaly else '正常'}")
            
            return result
            
        except Exception as e:
            self.log(f"❌ 图片推理失败: {str(e)}")
            return {}
            
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
            self.log(f"📁 结果保存在: {output_dir}")
            self.log("=" * 50)
            
            # 保存结果摘要
            self.save_summary(results, output_dir)
            
            return results
            
        except Exception as e:
            self.log(f"❌ 批量推理失败: {str(e)}")
            return []
            
    def generate_visualization(self, result: Dict, output_dir: Path):
        """生成可视化结果图"""
        try:
            image_path = Path(result["image_path"])
            image_name = image_path.stem
            
            # 读取原始图片
            original_image = cv2.imread(str(image_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # 获取异常热力图
            anomaly_map = result["anomaly_map"]
            
            # 调整热力图大小到原始图片尺寸
            h, w = original_image.shape[:2]
            anomaly_map_resized = cv2.resize(anomaly_map, (w, h))
            
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
            overlay = original_image.copy()
            heatmap_colored = plt.cm.jet(anomaly_map_resized)[:,:,:3]
            overlay = (1-alpha) * original_image/255 + alpha * heatmap_colored
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
            
    def save_summary(self, results: List[Dict], output_dir: Path):
        """保存推理结果摘要"""
        try:
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
            
    def set_threshold(self, threshold: float):
        """设置异常检测阈值"""
        self.threshold = threshold
        self.log(f"🎚️ 异常检测阈值设置为: {threshold}")
        
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if not self.model:
            return {}
            
        return {
            "model_path": self.model_path,
            "device": self.device,
            "threshold": self.threshold,
            "model_type": "PatchCore"
        }


def main():
    """测试函数"""
    # 创建推理引擎
    engine = InferenceEngine()
    
    # 设置回调函数
    def log_callback(message):
        print(f"[LOG] {message}")
        
    def progress_callback(percent):
        print(f"[PROGRESS] {percent}%")
        
    engine.log_callback = log_callback
    engine.progress_callback = progress_callback
    
    # 加载模型
    if engine.load_model():
        # 测试单图推理
        test_image = "path/to/test/image.jpg"
        if Path(test_image).exists():
            result = engine.predict_single_image(test_image)
            print(f"单图推理结果: {result}")
            
        # 测试批量推理
        test_folder = "path/to/test/folder"
        if Path(test_folder).exists():
            results = engine.predict_folder(test_folder)
            print(f"批量推理完成，共处理 {len(results)} 张图片")
    else:
        print("模型加载失败")


if __name__ == "__main__":
    main() 