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
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple
from PIL import Image
import torch
from torchvision import transforms

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
    
    def __init__(self, model_path: str = None, debug: bool = False):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型路径，如果为None则使用最新模型
            debug: 是否启用调试模式
        """
        self.model_path = model_path
        self.model = None
        self.datamodule = None
        self.log_callback: Optional[Callable[[str], None]] = None
        self.progress_callback: Optional[Callable[[int], None]] = None
        self.prediction_results = []
        self.debug = debug
        
        # 推理配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = 0.5  # 异常检测阈值
        
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
            self.debug_log(f"加载checkpoint: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.debug_log(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # 检查模型权重
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                self.debug_log(f"State dict keys: {list(state_dict.keys())[:10]}...")  # 只显示前10个
                
                # 检查关键组件是否存在
                has_memory_bank = any("memory_bank" in key for key in state_dict.keys())
                has_feature_extractor = any("feature_extractor" in key for key in state_dict.keys())
                
                self.debug_log(f"模型组件检查:")
                self.debug_log(f"  - 特征提取器: {'✓' if has_feature_extractor else '✗'}")
                self.debug_log(f"  - 记忆库: {'✓' if has_memory_bank else '✗'}")
                
                if not has_memory_bank:
                    self.log("⚠️ 警告: 模型缺少记忆库，可能未完成训练")
                
                # 加载权重
                self.model.load_state_dict(state_dict)
            else:
                self.log("❌ checkpoint中缺少state_dict")
                return False
            
            # 设置模型为评估模式
            self.model.eval()
            self.model.to(self.device)
            
            # 检查模型是否有记忆库
            if hasattr(self.model, 'memory_bank') and hasattr(self.model.memory_bank, 'memory_bank'):
                memory_bank_size = self.model.memory_bank.memory_bank.shape if self.model.memory_bank.memory_bank is not None else None
                self.debug_log(f"记忆库大小: {memory_bank_size}")
                
                if memory_bank_size is None or memory_bank_size[0] == 0:
                    self.log("⚠️ 警告: 记忆库为空，模型可能未正确训练")
                    self.log("💡 建议: 重新训练模型或检查训练过程")
            
            # 检查模型的_is_fitted状态
            if hasattr(self.model, '_is_fitted'):
                fitted_status = self.model._is_fitted
                self.debug_log(f"模型fitted状态: {fitted_status}")
                if not fitted_status:
                    self.log("⚠️ 警告: 模型未处于fitted状态")
                    # 尝试设置为fitted状态
                    self.model._is_fitted = True
                    self.debug_log("已手动设置模型为fitted状态")
            
            # 检查后处理器状态
            if hasattr(self.model, 'post_processor'):
                pp = self.model.post_processor
                if hasattr(pp, '_image_threshold') and hasattr(pp, '_pixel_threshold'):
                    img_thresh = getattr(pp, '_image_threshold', None)
                    pix_thresh = getattr(pp, '_pixel_threshold', None)
                    self.debug_log(f"后处理器阈值: image={img_thresh}, pixel={pix_thresh}")
                    
                    # 如果阈值异常，尝试修复
                    if img_thresh is None or (hasattr(img_thresh, 'item') and img_thresh.item() == 1.0):
                        self.log("⚠️ 发现图像阈值异常，尝试重新设置")
                        # 设置一个合理的阈值
                        if hasattr(pp, '_image_threshold'):
                            pp._image_threshold = torch.tensor(0.5)
                        if hasattr(pp, '_pixel_threshold'):
                            pp._pixel_threshold = torch.tensor(0.5)
                        self.debug_log("已重新设置后处理器阈值为0.5")
            
            # **修复：初始化图像预处理transform**
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            self.debug_log("图像预处理transform已初始化")
            
            self.log("✅ 模型加载成功")
            return True
            
        except Exception as e:
            self.log(f"❌ 模型加载失败: {str(e)}")
            self.debug_log(f"模型加载错误详情: {str(e)}")
            import traceback
            if self.debug:
                traceback.print_exc()
            return False
            
    def infer_single_image(self, image_path, debug=False):
        """单个图像推理"""
        try:
            self.debug_log(f"开始推理图像: {image_path}")
            
            # 检查模型是否已加载
            if self.model is None:
                return None, "模型未加载"
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            self.debug_log(f"原始图像尺寸: {image.size}")
            
            # 转换为tensor
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            self.debug_log(f"输入张量形状: {image_tensor.shape}")
            self.debug_log(f"输入张量范围: {image_tensor.min().item():.4f} ~ {image_tensor.max().item():.4f}")
            
            # 推理
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            self.debug_log(f"模型输出类型: {type(predictions)}")
            
            # 获取原始异常分数（未经后处理）
            raw_anomaly_score = predictions.pred_score.cpu().numpy()
            if hasattr(raw_anomaly_score, 'ndim') and raw_anomaly_score.ndim > 0:
                raw_anomaly_score = raw_anomaly_score[0] if len(raw_anomaly_score) > 0 else 0.0
            
            self.debug_log(f"原始异常分数: {raw_anomaly_score:.6f}")
            
            # **关键修复：应用后处理器进行normalization和threshold**
            if hasattr(self.model, 'post_processor') and self.model.post_processor is not None:
                self.debug_log("应用后处理器进行normalization")
                
                # 检查后处理器属性
                image_threshold = getattr(self.model.post_processor, 'image_threshold', None)
                image_min = getattr(self.model.post_processor, 'image_min', None)
                image_max = getattr(self.model.post_processor, 'image_max', None)
                
                self.debug_log(f"后处理器配置: threshold={image_threshold}, min={image_min}, max={image_max}")
                
                # 应用后处理
                processed_predictions = self.model.post_processor(predictions)
                
                self.debug_log(f"后处理输出类型: {type(processed_predictions)}")
                
                # 获取处理后的分数
                if hasattr(processed_predictions, 'pred_score'):
                    processed_score = processed_predictions.pred_score.cpu().numpy()
                    if hasattr(processed_score, 'ndim') and processed_score.ndim > 0:
                        processed_score = processed_score[0] if len(processed_score) > 0 else 0.0
                    self.debug_log(f"后处理pred_score: {processed_score:.6f}")
                else:
                    processed_score = raw_anomaly_score
                    self.debug_log("后处理输出中未找到pred_score，使用原始分数")
                
                # 获取预测标签
                if hasattr(processed_predictions, 'pred_label'):
                    pred_label = processed_predictions.pred_label.cpu().numpy()
                    if pred_label.ndim > 0:
                        pred_label = pred_label[0] if len(pred_label) > 0 else 0
                    self.debug_log(f"后处理pred_label: {pred_label}")
                else:
                    pred_label = int(processed_score > 0.5)  # 使用默认阈值
                    self.debug_log(f"未找到pred_label，基于分数计算: {pred_label}")
                
                anomaly_score = processed_score
                
            else:
                self.debug_log("未找到后处理器，使用原始分数")
                anomaly_score = raw_anomaly_score
                # 基于原始分数和默认阈值计算标签
                pred_label = int(anomaly_score > 0.5)
            
            # 获取异常图
            anomaly_map = predictions.anomaly_map.cpu().numpy()
            self.debug_log(f"异常图形状: {anomaly_map.shape}")
            self.debug_log(f"异常图数值范围: {anomaly_map.min():.4f} ~ {anomaly_map.max():.4f}")
            
            # 检查异常图的形状，移除批次维度
            if anomaly_map.ndim == 4:  # (1, 1, H, W)
                anomaly_map = anomaly_map[0, 0]  # 取出 (H, W)
            elif anomaly_map.ndim == 3:  # (1, H, W) 
                anomaly_map = anomaly_map[0]  # 取出 (H, W)
            elif anomaly_map.ndim == 2:  # 已经是 (H, W)
                pass  # 无需处理
            else:
                self.debug_log(f"警告：异常维度的异常图形状: {anomaly_map.shape}")
                return None, f"异常的anomaly_map维度: {anomaly_map.shape}"
            
            self.debug_log(f"处理后异常图形状: {anomaly_map.shape}")
            
            # 显示分数对比
            self.debug_log(f"分数对比: 原始={raw_anomaly_score:.6f}, 后处理={anomaly_score:.6f}")
            self.debug_log(f"最终预测: 分数={anomaly_score:.6f}, 标签={pred_label}")
            
            result = {
                'anomaly_score': float(anomaly_score),
                'pred_label': int(pred_label),
                'anomaly_map': anomaly_map,
                'raw_score': float(raw_anomaly_score)  # 保存原始分数用于调试
            }
            
            return result, "推理成功"
            
        except Exception as e:
            error_msg = f"推理失败: {str(e)}"
            self.debug_log(f"推理异常: {e}")
            import traceback
            self.debug_log(f"异常堆栈: {traceback.format_exc()}")
            return None, error_msg
            
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
                
            # 检查图片是否存在
            if not Path(image_path).exists():
                self.log(f"❌ 图片不存在: {image_path}")
                return {}
                
            # **关键修复：使用新的infer_single_image方法**
            result, message = self.infer_single_image(image_path)
            
            if result is None:
                self.log(f"❌ 推理失败: {message}")
                return {}
            
            # 获取原始图像尺寸
            image = Image.open(image_path)
            original_size = image.size
            
            # 判断是否异常
            is_anomaly = result['anomaly_score'] > self.threshold
            
            # 构建返回结果
            final_result = {
                "image_path": image_path,
                "anomaly_score": result['anomaly_score'],
                "pred_label": result['pred_label'],
                "is_anomaly": bool(is_anomaly),
                "anomaly_map": result['anomaly_map'],
                "original_size": original_size,
                "threshold": self.threshold,
                "raw_score": result.get('raw_score', result['anomaly_score'])  # 保存原始分数
            }
            
            self.log(f"   异常分数: {result['anomaly_score']:.4f}")
            self.log(f"   预测结果: {'异常' if is_anomaly else '正常'}")
            
            return final_result
            
        except Exception as e:
            self.log(f"❌ 图片推理失败: {str(e)}")
            import traceback
            traceback.print_exc()
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
            self.log(f"   结果保存在: {output_dir}")
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
            
            self.debug_log(f"开始生成可视化: {image_name}")
            
            # 读取原始图片
            original_image = cv2.imread(str(image_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # 获取异常热力图
            anomaly_map = result["anomaly_map"]
            self.debug_log(f"异常图原始形状: {anomaly_map.shape}")
            
            # 处理异常图形状 - 从(1, H, W)变为(H, W)
            if anomaly_map.ndim == 3 and anomaly_map.shape[0] == 1:
                anomaly_map = anomaly_map[0]  # 移除batch维度
            elif anomaly_map.ndim == 3:
                # 如果是(H, W, C)格式，取第一个通道
                anomaly_map = anomaly_map[:, :, 0]
            
            self.debug_log(f"处理后异常图形状: {anomaly_map.shape}")
            
            # 调整热力图大小到原始图片尺寸
            h, w = original_image.shape[:2]
            self.debug_log(f"原始图片尺寸: {h}x{w}")
            
            # 确保anomaly_map是2D数组
            if anomaly_map.ndim != 2:
                raise ValueError(f"异常图维度错误: {anomaly_map.shape}, 期望2D数组")
                
            anomaly_map_resized = cv2.resize(anomaly_map, (w, h))
            self.debug_log(f"缩放后异常图形状: {anomaly_map_resized.shape}")
            
            # 创建可视化图像
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始图片
            axes[0].imshow(original_image)
            axes[0].set_title("原始图片")
            axes[0].axis('off')
            
            # 异常热力图
            im = axes[1].imshow(anomaly_map_resized, cmap='jet', vmin=0, vmax=1)
            axes[1].set_title(f"异常热力图 (分数: {result['anomaly_score']:.4f})")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # 叠加图
            alpha = 0.4
            overlay = original_image.copy().astype(np.float32)
            
            # 确保异常图数值在[0,1]范围内
            anomaly_map_norm = (anomaly_map_resized - anomaly_map_resized.min()) / (anomaly_map_resized.max() - anomaly_map_resized.min() + 1e-8)
            heatmap_colored = plt.cm.jet(anomaly_map_norm)[:, :, :3]  # 只取RGB通道
            
            overlay = (1-alpha) * overlay/255 + alpha * heatmap_colored
            overlay = np.clip(overlay, 0, 1)  # 确保值在[0,1]范围内
            
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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="PatchCore推理引擎测试")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--test-viz", action="store_true", help="测试可视化功能")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 PatchCore 推理引擎测试")
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
    
    # 创建推理引擎，传入debug参数
    engine = InferenceEngine(model_path=model_path, debug=args.debug)
    
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
        test_output_dir = Path("./test_inference_results")
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
                print(f"检测阈值: {result['threshold']}")
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