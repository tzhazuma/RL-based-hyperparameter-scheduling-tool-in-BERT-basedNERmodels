#!/usr/bin/env python
# coding: utf-8
"""
基于高斯过程汤普森采样的NER模型超参数优化

此脚本使用高斯过程和汤普森采样来自动调整命名实体识别(NER)模型的超参数
"""

import os
import sys
import json
import time
import argparse
import subprocess
import random
import numpy as np
import torch
import gpytorch
import logging
import shutil
from pathlib import Path
import re
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import traceback

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gp_ts_ner_optimizer.log")
    ]
)
logger = logging.getLogger(__name__)

# 高斯过程模型类 - 用于超参数优化
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# 超参数空间定义类
class HyperparameterSpace:
    def __init__(self):
        # 定义超参数及其取值范围
        self.params = {
            "learning_rate": {"type": "log_float", "min": 1e-6, "max": 1e-4, "default": 1e-5},
            "weight_decay": {"type": "log_float", "min": 1e-5, "max": 1e-3, "default": 1e-4},
            "adam_beta1": {"type": "float", "min": 0.8, "max": 0.99, "default": 0.9},
            "adam_beta2": {"type": "float", "min": 0.95, "max": 0.999, "default": 0.98},
            "warmup_steps": {"type": "int", "min": 50, "max": 500, "default": 200},
            "train_batch": {"type": "int", "min": 8, "max": 32, "default": 16},
        }
        
        # 超参数维度计数
        self.dim = len(self.params)
        
        # 缓存已探索的配置
        self.explored_configs = {}
    
    def default_config(self) -> Dict[str, Any]:
        """返回默认超参数配置"""
        return {name: param_info["default"] for name, param_info in self.params.items()}
    
    def sample_random(self) -> Dict[str, Any]:
        """随机采样一组超参数"""
        config = {}
        for name, param_info in self.params.items():
            if param_info["type"] == "int":
                config[name] = random.randint(param_info["min"], param_info["max"])
            elif param_info["type"] == "float":
                config[name] = random.uniform(param_info["min"], param_info["max"])
            elif param_info["type"] == "log_float":
                log_min = np.log(param_info["min"])
                log_max = np.log(param_info["max"])
                log_value = random.uniform(log_min, log_max)
                config[name] = np.exp(log_value)
        return config
    
    def normalize_config(self, config: Dict[str, Any]) -> torch.Tensor:
        """将配置归一化到[0,1]空间中的向量表示"""
        normalized = []
        for name, param_info in self.params.items():
            value = config[name]
            if param_info["type"] == "log_float":
                log_min = np.log(param_info["min"])
                log_max = np.log(param_info["max"])
                norm_value = (np.log(value) - log_min) / (log_max - log_min)
            else:
                norm_value = (value - param_info["min"]) / (param_info["max"] - param_info["min"])
            normalized.append(norm_value)
        return torch.tensor(normalized, dtype=torch.float32)
    
    def denormalize_vector(self, vector: torch.Tensor) -> Dict[str, Any]:
        """将[0,1]空间中的向量转换回实际的超参数配置"""
        config = {}
        for i, (name, param_info) in enumerate(self.params.items()):
            norm_value = vector[i].item()
            if param_info["type"] == "int":
                unnorm_value = int(round(norm_value * (param_info["max"] - param_info["min"]) + param_info["min"]))
            elif param_info["type"] == "float":
                unnorm_value = norm_value * (param_info["max"] - param_info["min"]) + param_info["min"]
            elif param_info["type"] == "log_float":
                log_min = np.log(param_info["min"])
                log_max = np.log(param_info["max"])
                unnorm_value = np.exp(norm_value * (log_max - log_min) + log_min)
            config[name] = unnorm_value
        return config
    
    def config_to_args(self, config: Dict[str, Any]) -> List[str]:
        """将配置转换为命令行参数"""
        args = []
        
        # 将超参数添加到命令行参数
        for name, value in config.items():
            if name == "learning_rate":
                args.extend(["--learning_rate", str(value)])
            elif name == "weight_decay":
                args.extend(["--weight_decay", str(value)])
            elif name == "adam_beta1":
                args.extend(["--adam_beta1", str(value)])
            elif name == "adam_beta2":
                args.extend(["--adam_beta2", str(value)])
            elif name == "warmup_steps":
                args.extend(["--warmup_steps", str(int(value))])
            elif name == "train_batch":
                args.extend(["--per_gpu_train_batch_size", str(int(value))])
                
        return args
    
    def get_fingerprint(self, config: Dict[str, Any]) -> str:
        """生成配置的唯一指纹"""
        return json.dumps([(k, round(float(v), 6) if isinstance(v, float) else v) 
                           for k, v in sorted(config.items())])

# 高斯过程-汤普森采样优化器
class GPTSOptimizer:
    def __init__(
        self,
        hyperparameter_space: HyperparameterSpace,
        output_dir: str,
        n_trials: int = 20,
        random_trials: int = 5,
        seed: int = 42,
        data_dir: str = None,
        model_type: str = "roberta",
        model_name: str = "roberta-base",
        n_epochs: int = 5,
        max_seq_length: int = 128,
        cache_dir: str = None,
        run_ner_path: str = None,
        verbose: bool = True,
        skip_model_saving: bool = False
    ):
        self.hyperparameter_space = hyperparameter_space
        self.output_dir = Path(output_dir)
        self.n_trials = n_trials
        self.random_trials = random_trials
        self.seed = seed
        self.data_dir = data_dir
        self.model_type = model_type
        self.model_name = model_name
        self.n_epochs = n_epochs
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir
        self.run_ner_path = run_ner_path
        self.verbose = verbose
        self.skip_model_saving = skip_model_saving
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # 初始化高斯过程模型
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = None
        
        # 存储训练数据(配置和结果)
        self.train_x = torch.tensor([])
        self.train_y = torch.tensor([])
        
        # 跟踪最佳结果
        self.best_f1 = 0.0
        self.best_config = None
        self.best_trial = -1
        
        # 存储每次试验的结果
        self.results = []
        
        # 结果文件路径
        self.results_file = self.output_dir / "optimization_results.json"
        
        # 创建可视化文件夹
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        self.log_file = self.output_dir / "optimization_log.txt"
        
    def run_trial(self, config: Dict[str, Any], trial: int) -> Dict[str, Any]:
        """运行单次试验，使用给定的超参数配置训练和评估NER模型"""
        trial_dir = self.output_dir / f"trial_{trial}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # 优先使用GPU
        use_cuda = True
        
        try:
            # 记录当前试验的配置
            self._log_config(config, trial)
            
            # 构建命令行参数
            base_args = [
                "--data_dir", str(self.data_dir),
                "--model_type", self.model_type,
                "--model_name_or_path", self.model_name,
                "--output_dir", str(trial_dir),
                "--num_train_epochs", str(self.n_epochs),
                "--max_seq_length", str(self.max_seq_length),
                "--do_train",
                "--do_eval",
                "--do_predict",
                "--evaluate_during_training",
                "--overwrite_output_dir",
                "--seed", str(self.seed),
                # 添加不保存checkpoint的参数
                "--save_steps", "500",
                #"--skip_model_saving",  # 新增参数，跳过模型保存
                # 添加显示训练进度的参数
                "--logging_steps", "200"
            ]
            
            if self.cache_dir:
                base_args.extend(["--cache_dir", str(self.cache_dir)])
            
            # 添加超参数
            config_args = self.hyperparameter_space.config_to_args(config)
            all_args = base_args + config_args
            
            # 构建完整命令
            cmd = [sys.executable, self.run_ner_path] + all_args
            logger.info(f"Trial {trial}: Running command: {' '.join(cmd)}")
            
            # 设置环境变量以抑制警告和修复CUDA问题
            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "ignore"
            # 修复CUDA内存分配器问题
            env["PYTORCH_CUDA_ALLOC_CONF"] = ""  # 清空CUDA分配器配置
            env["CUDA_LAUNCH_BLOCKING"] = "1"
            
            print(f"\n{'='*80}")
            print(f"开始试验 {trial}/{self.n_trials} (使用CUDA)")
            print(f"超参数配置:")
            for name, value in config.items():
                print(f"  {name}: {value}")
            print(f"{'='*80}\n")
            
            # 直接在控制台显示输出，不使用管道
            with open(trial_dir / "training_log.txt", "w") as log_file:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    # 不使用PIPE，直接让输出显示在控制台
                    stdout=None,
                    stderr=None,
                    universal_newlines=True,
                    bufsize=1  # 行缓冲
                )
                
                # 等待进程完成
                process.wait()
                returncode = process.returncode
                
                if returncode != 0:
                    # 如果失败，重试一次使用CPU
                    if use_cuda:
                        print(f"\n{'='*80}")
                        print(f"CUDA训练失败，切换到CPU模式重试...")
                        print(f"{'='*80}\n")
                        
                        # 添加no_cuda参数
                        all_args.append("--no_cuda")
                        cmd = [sys.executable, self.run_ner_path] + all_args
                        
                        # 直接在控制台显示输出
                        process = subprocess.run(
                            cmd,
                            env=env,
                            check=False
                        )
                        returncode = process.returncode
                        use_cuda = False
            
            # 读取日志文件来获取输出内容
            log_content = ""
            if os.path.exists(trial_dir / "training_log.txt"):
                with open(trial_dir / "training_log.txt", "r") as f:
                    log_content = f.read()
            
            # 检查返回码
            if returncode != 0:
                logger.error(f"Trial {trial} failed with return code {returncode}")
                
                # 创建一个空的结果文件，记录配置信息
                with open(trial_dir / "test_results.txt", "w") as f:
                    f.write(f"Trial {trial} failed with return code {returncode}\n")
                    f.write(f"Configuration:\n")
                    for key, value in config.items():
                        f.write(f"  {key}: {value}\n")
                
                return {
                    "f1": 0.0, 
                    "precision": 0.0, 
                    "recall": 0.0, 
                    "status": "error",
                    "config": config,
                    "trial": trial,
                    "used_gpu": use_cuda,
                    "error_message": log_content
                }
            
            # 解析结果，从test_results.txt中提取F1, precision, recall
            result_file = trial_dir / "test_results.txt"
            if not result_file.exists():
                logger.error(f"Result file not found: {result_file}")
                return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "status": "error"}
            
            metrics = self._parse_result_file(result_file)
            metrics["status"] = "success"
            
            # 添加配置信息
            metrics["config"] = config
            metrics["trial"] = trial
            metrics["used_gpu"] = use_cuda
            
            # 更新最佳结果
            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]
                self.best_config = config
                self.best_trial = trial
                logger.info(f"New best F1: {self.best_f1:.4f} (Trial {trial})")
                print(f"\n{'*'*80}")
                print(f"新的最佳F1分数: {self.best_f1:.4f} (试验 {trial})")
                print(f"{'*'*80}\n")
                
                # 保存最佳配置的可视化
                self._visualize_best_config()
            
            # 清理模型文件以节省空间（只保留结果和日志）
            for item in trial_dir.glob("*"):
                if item.is_file() and item.name not in ["test_results.txt", "training_log.txt"]:
                    try:
                        os.remove(item)
                    except Exception as e:
                        logger.warning(f"无法删除文件 {item}: {e}")
                elif item.is_dir() and item.name not in ["visualizations"]:
                    try:
                        shutil.rmtree(item)
                    except Exception as e:
                        logger.warning(f"无法删除目录 {item}: {e}")
            
            # 打印摘要
            print(f"\n{'='*50}")
            print(f"试验 {trial} 结果:")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  使用GPU: {use_cuda}")
            print(f"{'='*50}\n")
            
            return metrics
        
        except Exception as e:
            logger.exception(f"Error running trial {trial}: {e}")
            traceback.print_exc()
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "status": "error", "error_message": str(e)}
    
    def _log_config(self, config: Dict[str, Any], trial: int) -> None:
        """记录当前试验的配置信息"""
        config_str = f"Trial {trial} Configuration:\n"
        for name, value in config.items():
            config_str += f"  {name}: {value}\n"
        
        logger.info(config_str)
        
        # 追加到日志文件
        with open(self.log_file, "a") as f:
            f.write(f"\n{'-'*50}\n")
            f.write(f"Trial {trial} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(config_str)
            f.write(f"{'-'*50}\n")
    
    def _parse_result_file(self, result_file: Path) -> Dict[str, float]:
        """解析结果文件以提取评估指标"""
        metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        try:
            with open(result_file, "r") as f:
                for line in f:
                    match = re.search(r"(precision|recall|f1) = ([0-9.]+)", line)
                    if match:
                        metric_name = match.group(1)
                        metric_value = float(match.group(2))
                        metrics[metric_name] = metric_value
        except Exception as e:
            logger.error(f"Error parsing result file {result_file}: {e}")
        
        return metrics
    
    def _update_gp_model(self):
        """更新高斯过程模型"""
        if len(self.train_y) == 0:
            return
        
        # 初始化或更新模型
        if self.model is None:
            self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood)
        else:
            self.model.set_train_data(self.train_x, self.train_y, strict=False)
        
        # 训练模型
        self.model.train()
        self.likelihood.train()
        
        # 使用Adam优化器
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=0.1)
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # 训练循环
        for i in range(100):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
        
    def optimize(self) -> Dict[str, Any]:
        """运行优化流程"""
        logger.info(f"Starting optimization with {self.n_trials} trials ({self.random_trials} random trials)")
        
        # 清除并初始化日志文件
        with open(self.log_file, "w") as f:
            f.write(f"GP-TS NER Hyperparameter Optimization\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total trials: {self.n_trials}, Random trials: {self.random_trials}\n")
            f.write(f"Model: {self.model_type} ({self.model_name})\n")
            f.write("-" * 50 + "\n\n")
        
        # 加载之前的结果，如果存在
        if self.results_file.exists():
            with open(self.results_file, "r") as f:
                self.results = json.load(f)
            logger.info(f"Loaded {len(self.results)} previous results")
            
            # 恢复训练数据和最佳结果
            for result in self.results:
                if result["status"] == "success":
                    config = result["config"]
                    config_tensor = self.hyperparameter_space.normalize_config(config)
                    self.train_x = torch.cat([self.train_x, config_tensor.unsqueeze(0)])
                    self.train_y = torch.cat([self.train_y, torch.tensor([result["f1"]], dtype=torch.float32)])
                    
                    if result["f1"] > self.best_f1:
                        self.best_f1 = result["f1"]
                        self.best_config = config
                        self.best_trial = result["trial"]
        
        # 从上一次中断的地方继续
        start_trial = len(self.results)
        
        # 运行剩余的试验
        for trial in range(start_trial, self.n_trials):
            logger.info(f"Starting trial {trial}/{self.n_trials}")
            
            # 采样新的配置
            if trial < self.random_trials:
                # 前几次使用随机采样
                config = self.hyperparameter_space.sample_random()
                logger.info(f"Trial {trial}: Random sampling")
            else:
                # 使用GP-TS采样
                config = self._sample_next_config()
                logger.info(f"Trial {trial}: GP-TS sampling")
            
            # 检查是否重复的配置
            config_fingerprint = self.hyperparameter_space.get_fingerprint(config)
            if config_fingerprint in self.hyperparameter_space.explored_configs:
                logger.info(f"Trial {trial}: Config already explored, slightly perturbing")
                # 微小扰动避免完全重复
                for name in config:
                    if self.hyperparameter_space.params[name]["type"] in ["float", "log_float"]:
                        config[name] *= (1 + np.random.normal(0, 0.01))
            
            self.hyperparameter_space.explored_configs[config_fingerprint] = True
            
            # 运行试验
            logger.info(f"Trial {trial}: Using config: {config}")
            result = self.run_trial(config, trial)
            self.results.append(result)
            
            # 保存当前结果
            with open(self.results_file, "w") as f:
                json.dump(self.results, f, indent=2)
            
            # 更新GP模型
            if result["status"] == "success":
                config_tensor = self.hyperparameter_space.normalize_config(config)
                self.train_x = torch.cat([self.train_x, config_tensor.unsqueeze(0)])
                self.train_y = torch.cat([self.train_y, torch.tensor([result["f1"]], dtype=torch.float32)])
                self._update_gp_model()
                
                # 记录详细结果
                with open(self.log_file, "a") as f:
                    f.write(f"\nTrial {trial} Results:\n")
                    f.write(f"  F1: {result['f1']:.4f}\n")
                    f.write(f"  Precision: {result['precision']:.4f}\n")
                    f.write(f"  Recall: {result['recall']:.4f}\n")
                    if self.best_trial == trial:
                        f.write("  ** New Best Configuration **\n")
            
            # 画图
            self._plot_results()
            
            # 保存GP模型状态图
            if trial >= self.random_trials and len(self.train_y) >= 2:
                self._visualize_gp_model(trial)
        
        # 记录优化完成信息
        with open(self.log_file, "a") as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"Optimization Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Best F1: {self.best_f1:.4f} (Trial {self.best_trial})\n")
            if self.best_config:
                f.write("Best Configuration:\n")
                for name, value in self.best_config.items():
                    f.write(f"  {name}: {value}\n")
        
        # 返回最佳配置和结果
        return {
            "best_config": self.best_config,
            "best_f1": self.best_f1,
            "best_trial": self.best_trial,
            "all_results": self.results
        }
    
    def _sample_next_config(self) -> Dict[str, Any]:
        """使用汤普森采样选择下一个要尝试的超参数配置"""
        # 如果还没有足够的数据，返回随机配置
        if len(self.train_y) < 2:
            return self.hyperparameter_space.sample_random()
        
        # 更新GP模型
        self._update_gp_model()
        
        # 采样策略：使用网格搜索+汤普森采样
        self.model.eval()
        self.likelihood.eval()
        
        # 创建搜索网格
        n_points = 1000
        search_points = torch.rand(n_points, self.hyperparameter_space.dim)
        
        # 从后验预测中采样
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(search_points)
            # 从后验分布中抽样 - 这就是汤普森采样
            sampled_values = posterior.sample()
        
        # 找到样本中得分最高的点
        best_idx = torch.argmax(sampled_values)
        best_point = search_points[best_idx]
        
        # 转换回实际超参数
        config = self.hyperparameter_space.denormalize_vector(best_point)
        
        return config
    
    def _plot_results(self):
        """绘制所有试验的F1分数和超参数关系"""
        if not self.results:
            return
        
        # 提取有效的试验和F1分数
        trials = []
        f1_scores = []
        configs = []
        
        for result in self.results:
            if result["status"] == "success":
                trials.append(result["trial"])
                f1_scores.append(result["f1"])
                configs.append(result["config"])
        
        if not trials:
            return
        
        # 1. 绘制F1分数随试验变化的曲线
        plt.figure(figsize=(10, 6))
        plt.plot(trials, f1_scores, 'o-', color='blue')
        plt.axhline(y=self.best_f1, color='r', linestyle='--', label=f'Best F1: {self.best_f1:.4f}')
        plt.xlabel('Trial')
        plt.ylabel('F1 Score')
        plt.title('NER Model F1 Score by Trial')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / "optimization_progress.png")
        plt.close()
        
        # 2. 为每个超参数创建单独的图，展示其与F1的关系
        param_names = list(self.hyperparameter_space.params.keys())
        
        for param_name in param_names:
            plt.figure(figsize=(10, 6))
            
            # 提取每个试验的参数值
            param_values = [config[param_name] for config in configs]
            
            # 绘制散点图
            sc = plt.scatter(param_values, f1_scores, c=trials, cmap='viridis', 
                          s=100, alpha=0.7, edgecolors='k')
            
            # 添加颜色条以表示试验顺序
            cbar = plt.colorbar(sc)
            cbar.set_label('Trial Number')
            
            # 标出最佳点
            if self.best_config:
                best_idx = trials.index(self.best_trial) if self.best_trial in trials else -1
                if best_idx >= 0:
                    plt.scatter([param_values[best_idx]], [f1_scores[best_idx]], 
                             s=200, c='red', marker='*', label='Best')
            
            # 设置标题和标签
            plt.title(f'F1 Score vs {param_name}')
            plt.xlabel(param_name)
            plt.ylabel('F1 Score')
            plt.grid(True)
            plt.legend()
            
            # 保存图表
            plt.savefig(self.viz_dir / f"param_{param_name}_vs_f1.png")
            plt.close()
        
        # 3. 绘制超参数配对关系图（如果有足够的数据点）
        if len(trials) >= 5 and len(param_names) > 1:
            # 创建配对网格
            fig, axes = plt.subplots(len(param_names), len(param_names), figsize=(15, 15))
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            
            # 准备数据
            param_data = {name: [config[name] for config in configs] for name in param_names}
            
            # 绘制每对参数的关系
            for i, param1 in enumerate(param_names):
                for j, param2 in enumerate(param_names):
                    ax = axes[i, j]
                    
                    if i == j:  # 对角线上，绘制参数分布直方图
                        ax.hist(param_data[param1], bins=min(10, len(trials)), alpha=0.7)
                        ax.set_title(param1)
                    else:  # 非对角线，绘制两个参数的散点图
                        sc = ax.scatter(param_data[param2], param_data[param1], c=f1_scores, 
                                    cmap='viridis', s=50, alpha=0.7)
                        
                        # 标出最佳点
                        if self.best_config:
                            ax.scatter(self.best_config[param2], self.best_config[param1], 
                                    s=100, c='red', marker='*')
                    
                    # 设置轴标签
                    if i == len(param_names) - 1:
                        ax.set_xlabel(param2)
                    if j == 0:
                        ax.set_ylabel(param1)
            
            # 添加颜色条
            cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
            fig.colorbar(sc, cax=cbar_ax).set_label('F1 Score')
            
            plt.savefig(self.viz_dir / "parameter_relationships.png")
            plt.close()
        
        # 4. 绘制当前最优超参数配置
        self._visualize_best_config()
    
    def _visualize_best_config(self):
        """可视化当前最佳配置"""
        if not self.best_config:
            return
        
        # 创建雷达图展示最佳配置
        param_names = list(self.hyperparameter_space.params.keys())
        
        # 归一化最佳配置的值
        best_values = []
        for name in param_names:
            param_info = self.hyperparameter_space.params[name]
            if param_info["type"] == "log_float":
                log_min = np.log(param_info["min"])
                log_max = np.log(param_info["max"])
                norm_value = (np.log(self.best_config[name]) - log_min) / (log_max - log_min)
            else:
                norm_value = (self.best_config[name] - param_info["min"]) / (param_info["max"] - param_info["min"])
            best_values.append(norm_value)
        
        # 设置雷达图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # 设置角度和标签
        angles = np.linspace(0, 2*np.pi, len(param_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合多边形
        best_values += best_values[:1]  # 闭合多边形的值
        
        # 绘制填充多边形
        ax.plot(angles, best_values, 'o-', linewidth=2)
        ax.fill(angles, best_values, alpha=0.25)
        
        # 设置标签和刻度
        ax.set_thetagrids(np.degrees(angles[:-1]), param_names)
        ax.set_ylim(0, 1)
        ax.grid(True)
        
        plt.title(f'Best Configuration (F1: {self.best_f1:.4f}, Trial: {self.best_trial})')
        plt.savefig(self.viz_dir / "best_configuration.png")
        plt.close()
        
        # 保存最佳配置的详细信息
        with open(self.viz_dir / "best_config_details.txt", "w") as f:
            f.write(f"Best Configuration (Trial {self.best_trial}, F1: {self.best_f1:.4f})\n")
            f.write("-" * 50 + "\n")
            for name, value in self.best_config.items():
                f.write(f"{name}: {value}\n")
    
    def _visualize_gp_model(self, trial: int) -> None:
        """可视化当前高斯过程模型状态"""
        if self.model is None or len(self.train_y) < 2:
            return
        
        try:
            # 设置模型为评估模式
            self.model.eval()
            self.likelihood.eval()
            
            # 仅对于2D可视化（选择前两个最重要的超参数）
            if self.hyperparameter_space.dim >= 2:
                # 查找最重要的两个参数 (简单地使用有最大方差的参数)
                param_names = list(self.hyperparameter_space.params.keys())
                
                # 使用当前训练数据的方差作为重要性指标
                if len(self.train_x) >= 3:  # 需要至少3个点来计算方差
                    variances = torch.var(self.train_x, dim=0)
                    top_indices = torch.argsort(variances, descending=True)[:2]
                else:
                    # 如果数据点太少，就使用前两个参数
                    top_indices = torch.tensor([0, 1])
                
                # 创建2D网格
                n_points = 50
                grid_x = torch.linspace(0, 1, n_points)
                grid_y = torch.linspace(0, 1, n_points)
                
                x1, x2 = torch.meshgrid(grid_x, grid_y)
                grid_points = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1)
                
                # 对于超过2个参数的模型，其余参数使用当前最佳配置的值
                if self.hyperparameter_space.dim > 2 and self.best_config:
                    full_points = torch.zeros(grid_points.shape[0], self.hyperparameter_space.dim)
                    
                    # 填充其余参数
                    best_tensor = self.hyperparameter_space.normalize_config(self.best_config)
                    for i in range(self.hyperparameter_space.dim):
                        if i in top_indices:
                            idx = torch.where(top_indices == i)[0].item()
                            full_points[:, i] = grid_points[:, idx]
                        else:
                            full_points[:, i] = best_tensor[i]
                    
                    grid_points = full_points
                
                # 预测性能
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    predictions = self.likelihood(self.model(grid_points))
                    mean = predictions.mean.reshape(n_points, n_points).numpy()
                    variance = predictions.variance.reshape(n_points, n_points).numpy()
                    
                    # 采样一个函数实例（汤普森采样）
                    sampled = predictions.sample().reshape(n_points, n_points).numpy()
                
                # 绘制结果
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                # 绘制预测均值
                im0 = axes[0].imshow(mean.T, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='viridis')
                axes[0].set_title('GP Prediction Mean')
                fig.colorbar(im0, ax=axes[0])
                
                # 绘制预测方差
                im1 = axes[1].imshow(variance.T, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='plasma')
                axes[1].set_title('GP Prediction Variance')
                fig.colorbar(im1, ax=axes[1])
                
                # 绘制采样
                im2 = axes[2].imshow(sampled.T, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap='viridis')
                axes[2].set_title('GP Thompson Sampling')
                fig.colorbar(im2, ax=axes[2])
                
                # 将训练数据点添加到所有图表中
                for ax in axes:
                    if len(self.train_x) > 0:
                        # 仅显示选中的两个维度
                        plot_x = self.train_x[:, top_indices[0]].numpy()
                        plot_y = self.train_x[:, top_indices[1]].numpy()
                        sc = ax.scatter(plot_x, plot_y, c=self.train_y.numpy(), 
                                    cmap='viridis', s=100, edgecolors='k')
                
                # 设置轴标签
                param1 = param_names[top_indices[0]]
                param2 = param_names[top_indices[1]]
                for ax in axes:
                    ax.set_xlabel(param1)
                    ax.set_ylabel(param2)
                
                # 保存图表
                plt.tight_layout()
                plt.savefig(self.viz_dir / f"gp_model_state_trial_{trial}.png")
                plt.close()
        
        except Exception as e:
            logger.error(f"Error visualizing GP model: {e}")

def main():
    parser = argparse.ArgumentParser(description="Optimize NER model hyperparameters using GP-TS")
    parser.add_argument("--data_dir", type=str, required=True, help="The input data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory for optimization results")
    parser.add_argument("--model_type", type=str, default="roberta", help="Model type (bert/roberta)")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Model name or path")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--random_trials", type=int, default=5, help="Number of initial random trials")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs per trial")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for caching models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_ner_path", type=str, required=True, help="Full path to run_ner.py script")
    parser.add_argument("--reset", action="store_true", help="Reset optimization and start from scratch")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output during training")
    parser.add_argument("--skip_model_saving", action="store_true", help="Skip saving model files to save disk space")
    
    args = parser.parse_args()

    # 检查run_ner.py是否存在
    if not os.path.exists(args.run_ner_path):
        logger.error(f"Error: run_ner.py not found at {args.run_ner_path}")
        sys.exit(1)
    
    # 如果指定了reset参数，删除之前的优化结果文件
    results_file = Path(args.output_dir) / "optimization_results.json"
    progress_file = Path(args.output_dir) / "optimization_progress.png"
    
    if args.reset and results_file.exists():
        logger.info("重置优化过程，删除以前的结果...")
        try:
            os.remove(results_file)
            if progress_file.exists():
                os.remove(progress_file)
            # 清除之前的试验目录
            for trial_dir in Path(args.output_dir).glob("trial_*"):
                if trial_dir.is_dir():
                    logger.info(f"删除之前的试验目录: {trial_dir}")
                    shutil.rmtree(trial_dir)
        except Exception as e:
            logger.warning(f"清除以前的结果时出错: {e}")
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        logger.warning(f"数据目录不存在: {args.data_dir}")
        # 尝试查找常见的相对路径
        possible_paths = [
            os.path.join(os.path.dirname(args.run_ner_path), "dataset", "conll03_distant"),
            os.path.join(os.path.dirname(os.path.dirname(args.run_ner_path)), "dataset", "conll03_distant")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"找到数据目录: {path}")
                args.data_dir = path
                break
    
    # 创建优化器
    hyperparameter_space = HyperparameterSpace()
    optimizer = GPTSOptimizer(
        hyperparameter_space=hyperparameter_space,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        random_trials=args.random_trials,
        seed=args.seed,
        data_dir=args.data_dir,
        model_type=args.model_type,
        model_name=args.model_name,
        n_epochs=args.n_epochs,
        max_seq_length=args.max_seq_length,
        cache_dir=args.cache_dir,
        run_ner_path=args.run_ner_path,
        verbose=args.verbose,
        skip_model_saving=args.skip_model_saving,
    )
    
    # 运行优化
    best_result = optimizer.optimize()
    
    # 打印最佳结果
    logger.info(f"Optimization completed!")
    logger.info(f"Best F1: {best_result['best_f1']:.4f} (Trial {best_result['best_trial']})")
    if best_result['best_config']:
        logger.info(f"Best hyperparameters: {best_result['best_config']}")
    else:
        logger.info("No valid best configuration found")
    
    # 清理试验过程中的模型文件以节省空间
    if args.skip_model_saving:
        logger.info("正在清理试验模型文件以节省磁盘空间...")
        for trial_dir in Path(args.output_dir).glob("trial_*"):
            if trial_dir.is_dir():
                # 保留test_results.txt和training_log.txt
                for item in trial_dir.glob("*"):
                    if item.is_file() and item.name not in ["test_results.txt", "training_log.txt"]:
                        try:
                            os.remove(item)
                        except:
                            pass
                    elif item.is_dir() and item.name != "visualizations":
                        try:
                            shutil.rmtree(item)
                        except:
                            pass
    
    # 使用最佳配置进行最终训练，除非指定跳过
    if  best_result['best_config'] is not None:
        logger.info("Training final model with best hyperparameters...")
        print(f"\n{'='*80}")
        print(f"使用最佳超参数训练最终模型")
        print(f"最佳配置 (来自试验 {best_result['best_trial']}, F1: {best_result['best_f1']:.4f}):")
        for name, value in best_result['best_config'].items():
            print(f"  {name}: {value}")
        print(f"{'='*80}\n")
        
        final_output_dir = Path(args.output_dir) / "final_model"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 将超参数配置转换为命令行参数
        base_args = [
            "--data_dir", str(args.data_dir),
            "--model_type", args.model_type,
            "--model_name_or_path", args.model_name,
            "--output_dir", str(final_output_dir),
            "--num_train_epochs", str(args.n_epochs * 2),  # 最终训练更多轮次
            "--max_seq_length", str(args.max_seq_length),
            "--do_train",
            "--do_eval",
            "--do_predict",
            "--evaluate_during_training",
            "--overwrite_output_dir",
            "--seed", str(args.seed),
            "--save_steps", "500",  # 不保存checkpoint
            "--logging_steps", "200"
            #"--skip_model_saving",  # 新增参数，跳过模型保存
        ]
        
        if args.cache_dir:
            base_args.extend(["--cache_dir", str(args.cache_dir)])
        
        config_args = hyperparameter_space.config_to_args(best_result['best_config'])
        all_args = base_args + config_args
        
        # 设置环境变量
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        env["PYTORCH_CUDA_ALLOC_CONF"] = ""  # 清空CUDA分配器配置
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # 构建命令，使用完整路径
        cmd = [sys.executable, args.run_ner_path] + all_args
        logger.info(f"Final training command: {' '.join(cmd)}")
        
        # 使用与trial相同的方法执行命令
        use_cuda = True
        
        # 直接进行命令行输出
        with open(final_output_dir / "final_training_log.txt", "w") as log_file:
            # 使用subprocess.Popen来实时显示输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env,
                bufsize=1  # 行缓冲
            )
            
            # 读取并处理输出
            while process.poll() is None:
                # 处理标准输出
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break
                    # 同时写入日志文件和显示到屏幕
                    log_file.write(line)
                    log_file.flush()
                    # 只打印关键信息
                    if any(keyword in line for keyword in ["Epoch", "eval", "F1", "precision", "recall"]):
                        print(line.strip())
                
                # 处理标准错误
                for line in iter(process.stderr.readline, ""):
                    if not line:
                        break
                    # 同时写入日志文件
                    log_file.write(line)
                    log_file.flush()
                    
                    # 检测CUDA错误
                    if "RuntimeError: Unrecognized CachingAllocator option" in line:
                        print(f"\nCUDA错误，将切换到CPU模式...\n")
                        process.terminate()
                        use_cuda = False
                        break
            
            # 确保读取所有剩余输出
            stdout, stderr = process.communicate()
            if stdout:
                log_file.write(stdout)
            if stderr:
                log_file.write(stderr)
                if "RuntimeError: Unrecognized CachingAllocator option" in stderr:
                    use_cuda = False
        
        # 如果检测到CUDA错误，使用CPU重试
        if not use_cuda:
            print(f"\n{'='*80}")
            print(f"使用CPU训练最终模型")
            print(f"{'='*80}\n")
            
            # 添加no_cuda参数
            all_args.append("--no_cuda")
            cmd = [sys.executable, args.run_ner_path] + all_args
            
            # 重新运行，直接显示输出到控制台
            with open(final_output_dir / "final_training_log.txt", "w") as log_file:
                process = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    check=False
                )
            
            # 将输出写入文件
            log_file.write(process.stdout)
            log_file.write(process.stderr)
            
            # 在控制台显示关键信息
            for line in process.stdout.split('\n'):
                if any(keyword in line for keyword in ["Epoch", "eval", "F1", "precision", "recall"]):
                    print(line)
        
        # 检查训练是否成功
        returncode = process.returncode
        if returncode == 0:
            print(f"\n{'='*80}")
            print("最终模型训练完成!")
            print(f"{'='*80}\n")
            logger.info("Final model training completed successfully!")
            
            # 删除所有模型文件但保留评估结果
            for item in final_output_dir.glob("*"):
                if item.is_file() and item.name not in ["test_results.txt", "final_training_log.txt"]:
                    try:
                        os.remove(item)
                    except:
                        pass
                elif item.is_dir():
                    try:
                        shutil.rmtree(item)
                    except:
                        pass
        else:
            logger.error(f"Error training final model: return code {returncode}")
            print(f"\n{'='*80}")
            print("最终模型训练失败!")
            print(f"{'='*80}\n")
    
    # 保存最终结果和配置
    with open(Path(args.output_dir) / "final_results.json", "w") as f:
        json.dump(best_result, f, indent=2)
    
    logger.info(f"Optimization results saved to {args.output_dir}")

if __name__ == "__main__":
    main()