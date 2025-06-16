# -*- coding: utf-8 -*-
"""
PatchCore训练器模块

包装flexible_patchcore_train.py的功能，提供面向对象的接口供GUI调用。
将训练逻辑与界面逻辑分离。
"""

import sys
import os
from pathlib import Path
from typing import Callable, Optional

# 添加当前目录到Python路径，以便导入训练模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入anomalib模块
try:
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
except ImportError as e:
    print(f"错误: 无法导入anomalib模块: {e}")
    print("请确保已正确安装anomalib并激活相应的conda环境")
    raise


class PatchCoreTrainer:
    """PatchCore训练器类
    
    包装训练逻辑，提供回调接口与GUI通信
    """
    
    def __init__(self, config: dict):
        """初始化训练器
        
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.log_callback: Optional[Callable[[str], None]] = None
        self.progress_callback: Optional[Callable[[int], None]] = None
        
        # 训练组件
        self.datamodule = None
        self.model = None
        self.engine = None
        
    def log(self, message: str):
        """记录日志"""
        print(message)  # 控制台输出
        if self.log_callback:
            self.log_callback(message)
            
    def update_progress(self, value: int):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(value)
            
    def check_data_paths(self) -> bool:
        """检查数据路径是否存在"""
        data_root = Path(self.config["data_root"])
        
        self.log("检查数据路径...")
        
        # 检查根目录
        if not data_root.exists():
            self.log(f"❌ 数据根目录不存在: {data_root}")
            return False
        
        # 检查各个子目录
        paths_to_check = [
            (data_root / self.config["normal_train_dir"], "训练正常样本"),
            (data_root / self.config["normal_test_dir"], "测试正常样本"),
            (data_root / self.config["abnormal_test_dir"], "测试异常样本"),
        ]
        
        all_good = True
        for path, description in paths_to_check:
            if path.exists():
                file_count = len(list(path.glob("*")))
                self.log(f"✅ {description}: {path} ({file_count} 个文件)")
            else:
                self.log(f"❌ {description}目录不存在: {path}")
                all_good = False
                
        return all_good
        
    def create_datamodule(self) -> bool:
        """创建数据模块"""
        try:
            self.log("配置数据模块...")
            self.update_progress(10)
            
            self.datamodule = Folder(
                name=self.config["dataset_name"],
                root=self.config["data_root"],
                normal_dir=self.config["normal_train_dir"],
                normal_test_dir=self.config["normal_test_dir"],
                abnormal_dir=self.config["abnormal_test_dir"],
                train_batch_size=self.config["train_batch_size"],
                eval_batch_size=self.config["eval_batch_size"],
                num_workers=self.config["num_workers"],
            )
            
            self.log("✅ 数据模块配置完成")
            return True
            
        except Exception as e:
            self.log(f"❌ 数据模块创建失败: {e}")
            return False
            
    def create_model(self) -> bool:
        """创建模型"""
        try:
            self.log("初始化PatchCore模型...")
            self.log(f"   骨干网络: {self.config['backbone']}")
            self.log(f"   特征层: {self.config['layers']}")
            self.log(f"   核心集采样比例: {self.config['coreset_sampling_ratio']}")
            self.log(f"   最近邻数量: {self.config['num_neighbors']}")
            
            self.update_progress(20)
            
            self.model = Patchcore(
                backbone=self.config["backbone"],
                layers=self.config["layers"],
                pre_trained=True,
                coreset_sampling_ratio=self.config["coreset_sampling_ratio"],
                num_neighbors=self.config["num_neighbors"],
            )
            
            self.log("✅ 模型初始化完成")
            return True
            
        except Exception as e:
            self.log(f"❌ 模型创建失败: {e}")
            return False
            
    def create_engine(self) -> bool:
        """创建训练引擎"""
        try:
            self.log("配置训练引擎...")
            self.log(f"   最大轮数: {self.config['max_epochs']}")
            self.log(f"   输出目录: {self.config['output_dir']}")
            
            self.update_progress(30)
            
            self.engine = Engine(
                max_epochs=self.config["max_epochs"],
                enable_checkpointing=True,
                default_root_dir=self.config["output_dir"],
                accelerator="auto",
                devices=1,
            )
            
            self.log("✅ 训练引擎配置完成")
            return True
            
        except Exception as e:
            self.log(f"❌ 训练引擎创建失败: {e}")
            return False
            
    def setup_data(self) -> bool:
        """设置数据模块"""
        try:
            self.log("设置数据模块...")
            self.update_progress(40)
            
            # 设置数据模块
            self.datamodule.setup()
            
            # 显示数据集信息
            train_size = len(self.datamodule.train_dataloader().dataset)
            test_size = len(self.datamodule.test_dataloader().dataset)
            self.log(f"   训练样本数: {train_size}")
            self.log(f"   测试样本数: {test_size}")
            
            self.log("✅ 数据模块设置完成")
            return True
            
        except Exception as e:
            self.log(f"❌ 数据模块设置失败: {e}")
            return False
            
    def run_training(self) -> bool:
        """执行训练"""
        try:
            self.log("开始训练...")
            self.update_progress(50)
            
            # 开始训练
            self.engine.fit(datamodule=self.datamodule, model=self.model)
            
            self.update_progress(80)
            self.log("✅ 训练完成！")
            return True
            
        except Exception as e:
            self.log(f"❌ 训练失败: {e}")
            return False
            
    def run_testing(self) -> bool:
        """执行测试"""
        try:
            self.log("开始测试模型...")
            self.update_progress(90)
            
            # 测试模型
            test_results = self.engine.test(datamodule=self.datamodule, model=self.model)
            
            self.log("✅ 测试完成！")
            self.log("📊 测试结果:")
            
            if test_results:
                for result in test_results:
                    for key, value in result.items():
                        self.log(f"   {key}: {value:.4f}")
            
            self.update_progress(100)
            return True
            
        except Exception as e:
            self.log(f"❌ 测试失败: {e}")
            return False
            
    def validate_config(self) -> bool:
        """验证配置"""
        required_keys = [
            "data_root", "normal_train_dir", "normal_test_dir", "abnormal_test_dir",
            "dataset_name", "backbone", "layers", "coreset_sampling_ratio",
            "num_neighbors", "train_batch_size", "eval_batch_size", "max_epochs",
            "num_workers", "output_dir"
        ]
        
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            self.log(f"❌ 配置缺少必需的键: {missing_keys}")
            return False
            
        # 检查特征层
        if not self.config["layers"]:
            self.log("❌ 请至少选择一个特征层")
            return False
            
        # 检查数值范围
        if not (0 < self.config["coreset_sampling_ratio"] <= 1):
            self.log("❌ 核心集采样比例必须在0-1之间")
            return False
            
        if self.config["num_neighbors"] < 1:
            self.log("❌ 最近邻数量必须大于0")
            return False
            
        return True
        
    def train(self) -> bool:
        """执行完整的训练流程"""
        try:
            self.log("=" * 60)
            self.log("开始PatchCore训练流程")
            self.log("=" * 60)
            
            # 1. 验证配置
            if not self.validate_config():
                return False
                
            # 2. 检查数据路径
            if not self.check_data_paths():
                return False
                
            # 3. 创建数据模块
            if not self.create_datamodule():
                return False
                
            # 4. 创建模型
            if not self.create_model():
                return False
                
            # 5. 创建训练引擎
            if not self.create_engine():
                return False
                
            # 6. 设置数据
            if not self.setup_data():
                return False
                
            # 7. 执行训练
            if not self.run_training():
                return False
                
            # 8. 执行测试
            if not self.run_testing():
                return False
                
            self.log("=" * 60)
            self.log("✅ 训练流程完成！")
            self.log(f"📁 结果保存在: {self.config['output_dir']}")
            self.log("=" * 60)
            
            return True
            
        except Exception as e:
            self.log(f"❌ 训练流程出错: {e}")
            return False


def main():
    """测试函数"""
    # 测试配置
    test_config = {
        "data_root": "../project_test_organized",
        "normal_train_dir": "train/normal",
        "normal_test_dir": "test/normal",
        "abnormal_test_dir": "test/abnormal",
        "dataset_name": "test_dataset",
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "max_epochs": 1,
        "num_workers": 0,
        "output_dir": "./test_results",
    }
    
    # 创建训练器并训练
    trainer = PatchCoreTrainer(test_config)
    success = trainer.train()
    
    if success:
        print("✅ 测试训练成功！")
    else:
        print("❌ 测试训练失败！")


if __name__ == "__main__":
    main() 