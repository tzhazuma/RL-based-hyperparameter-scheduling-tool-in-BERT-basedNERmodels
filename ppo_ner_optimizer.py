#!/usr/bin/env python
# coding: utf-8
"""
基于近端策略优化(PPO)的NER模型超参数优化 (框架与GP-TS对齐)

此脚本使用PPO强化学习算法来自动调整命名实体识别(NER)模型的超参数,
其外围框架（参数、日志、输出、可视化）与GP-TS优化器脚本类似。
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
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import logging
import shutil
from pathlib import Path
import re
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import traceback

# 设置日志
# The logger name will be based on the script's module name by default
logger = logging.getLogger(__name__)

# PPO网络模型 - 用于超参数策略学习
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPONetwork, self).__init__()
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Policy network - outputs action mean and standard deviation
        self.policy_mean = nn.Linear(64, action_dim)
        # Learnable standard deviation
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim)) # Initialize log_std
        
        # Value network - evaluates state value
        self.value_head = nn.Sequential( # Renamed from self.value to avoid conflict
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state):
        features = self.shared_layers(state)
        action_mean = torch.tanh(self.policy_mean(features)) # Output actions in [-1, 1]
        action_log_std = self.policy_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        state_value = self.value_head(features)
        return action_mean, action_std, state_value
    
    def get_action(self, state, deterministic=False):
        action_mean, action_std, _ = self.forward(state)
        dist = Normal(action_mean, action_std)
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def evaluate(self, state, action):
        action_mean, action_std, state_value = self.forward(state)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, state_value, entropy

# 超参数空间定义类 (与GP-TS一致)
class HyperparameterSpace:
    def __init__(self):
        self.params = {
            "learning_rate": {"type": "log_float", "min": 1e-6, "max": 1e-4, "default": 1e-5},
            "weight_decay": {"type": "log_float", "min": 1e-5, "max": 1e-3, "default": 1e-4},
            "adam_beta1": {"type": "float", "min": 0.8, "max": 0.99, "default": 0.9},
            "adam_beta2": {"type": "float", "min": 0.95, "max": 0.999, "default": 0.98},
            "warmup_steps": {"type": "int", "min": 50, "max": 500, "default": 200},
            "train_batch": {"type": "int", "min": 8, "max": 32, "default": 16},
        }
        self.dim = len(self.params)
        self.explored_configs = {}
    
    def default_config(self) -> Dict[str, Any]:
        return {name: param_info["default"] for name, param_info in self.params.items()}
    
    def sample_random(self) -> Dict[str, Any]:
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
    
    def normalize_config(self, config: Dict[str, Any]) -> torch.Tensor: # Changed to torch.Tensor
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
    
    def denormalize_vector(self, vector: torch.Tensor) -> Dict[str, Any]: # Changed to torch.Tensor
        config = {}
        for i, (name, param_info) in enumerate(self.params.items()):
            norm_value = vector[i].item() # Ensure it's a scalar Python number
             # Clamp normalized value to [0, 1] before denormalization
            norm_value = max(0.0, min(1.0, norm_value))

            if param_info["type"] == "int":
                unnorm_value = int(round(norm_value * (param_info["max"] - param_info["min"]) + param_info["min"]))
                # Ensure integer value is within bounds after rounding
                unnorm_value = max(param_info["min"], min(param_info["max"], unnorm_value))
            elif param_info["type"] == "float":
                unnorm_value = norm_value * (param_info["max"] - param_info["min"]) + param_info["min"]
            elif param_info["type"] == "log_float":
                log_min = np.log(param_info["min"])
                log_max = np.log(param_info["max"])
                unnorm_value = np.exp(norm_value * (log_max - log_min) + log_min)
            config[name] = unnorm_value
        return config
    
    def config_to_args(self, config: Dict[str, Any]) -> List[str]:
        args = []
        for name, value in config.items():
            if name == "learning_rate":
                args.extend(["--learning_rate", f"{value:.8e}"]) # Format for scientific notation
            elif name == "weight_decay":
                args.extend(["--weight_decay", f"{value:.8e}"])
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
        return json.dumps([(k, round(float(v), 8) if isinstance(v, (float, np.floating)) else v)
                           for k, v in sorted(config.items())])

# PPO优化器 (框架与GP-TS对齐)
class PPOOptimizer:
    def __init__(
        self,
        hyperparameter_space: HyperparameterSpace,
        output_dir: str,
        n_trials: int = 20,
        # random_trials is not directly used by PPO in the same way as GP-TS,
        # PPO might have its own exploration strategy or initial random phase.
        # For now, we'll keep it if the calling script expects it, but PPO's _get_config will handle initial exploration.
        random_trials: int = 5, # This will be used to determine initial random sampling if needed
        seed: int = 42,
        data_dir: str = None,
        model_type: str = "roberta",
        model_name: str = "roberta-base",
        n_epochs: int = 5, # Epochs for NER model training per trial
        max_seq_length: int = 128,
        cache_dir: str = None,
        run_ner_path: str = None,
        verbose: bool = True,
        skip_model_saving: bool = False,
        # PPO特定参数
        ppo_epochs: int = 10,       # Epochs for updating PPO policy
        ppo_batch_size: int = 5,    # Batch size for PPO updates
        clip_epsilon: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        lr: float = 0.0003,         # Learning rate for PPO network
        # 早停参数 (for the PPO optimization process itself)
        early_stopping: bool = False,
        patience: int = 5,
        min_delta: float = 0.001
    ):
        self.hyperparameter_space = hyperparameter_space
        self.output_dir = Path(output_dir)
        self.n_trials = n_trials
        self.random_trials_threshold = random_trials # Store for initial random sampling
        self.seed = seed
        self.data_dir = data_dir
        self.model_type = model_type
        self.model_name = model_name
        self.n_epochs_ner = n_epochs # Renamed to avoid confusion with ppo_epochs
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir
        self.run_ner_path = run_ner_path
        self.verbose = verbose
        self.skip_model_saving = skip_model_saving
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # PPO特定参数
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.ppo_lr = lr # Renamed to avoid confusion
        
        # 早停参数
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.no_improvement_count = 0
        self.best_f1_for_early_stopping = 0.0 # Tracks best F1 for early stopping logic
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # 初始化PPO网络
        # State: [normalized trial_num] + normalized_hyperparams (current or previous best)
        # For simplicity, let's use a fixed state representation for now: normalized trial number
        # self.state_dim = 1 # Just normalized trial number
        # Or, more complex: state is the normalized version of the *previous* config
        self.state_dim = self.hyperparameter_space.dim 
        self.action_dim = self.hyperparameter_space.dim
        self.ppo_network = PPONetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.ppo_network.parameters(), lr=self.ppo_lr)
        
        # 存储PPO训练数据 (per PPO update batch)
        self.memory_states = []
        self.memory_actions = []
        self.memory_log_probs = []
        self.memory_rewards = []
        self.memory_values = [] # Not strictly needed for PPO update if GAE is not used, but good for logging
        self.memory_dones = [] # For episodic tasks, here each trial is an episode end

        # 跟踪最佳结果 (与GP-TS一致)
        self.best_f1 = 0.0
        self.best_config = None
        self.best_trial = -1
        
        # 存储每次试验的结果 (与GP-TS一致)
        self.results = []
        
        self.results_file = self.output_dir / "optimization_results.json"
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "optimization_log.txt" # Main optimization log
        
        self.current_state_for_ppo = torch.rand(self.state_dim, dtype=torch.float32) # Initial random state


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
                "--num_train_epochs", str(self.n_epochs_ner),
                "--max_seq_length", str(self.max_seq_length),
                "--do_train",
                "--do_eval",
                "--do_predict",
                "--evaluate_during_training",
                "--overwrite_output_dir",
                "--seed", str(self.seed + trial),
                # 设置保存步骤
                "--save_steps", "500",
                # 设置日志步骤
                "--logging_steps", "50"
            ]
            
            if self.cache_dir:
                base_args.extend(["--cache_dir", str(self.cache_dir)])
            # 添加超参数命令行参数
            config_args = self.hyperparameter_space.config_to_args(config)
            all_args = base_args + config_args
            
            # 构建完整命令
            cmd = [sys.executable, self.run_ner_path] + all_args
            logger.info(f"Trial {trial}: Running command: {' '.join(cmd)}")
            
            # 设置环境变量
            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "ignore"
            env["PYTORCH_CUDA_ALLOC_CONF"] = ""  # 清空CUDA分配器配置
            env["CUDA_LAUNCH_BLOCKING"] = "1"
            
            print(f"\n{'='*80}")
            print(f"开始试验 {trial}/{self.n_trials} (尝试使用 CUDA)")
            print(f"超参数配置:")
            for name, value in config.items():
                print(f"  {name}: {value}")
            print(f"{'='*80}\n")
            
            # 直接显示输出到控制台，不使用管道捕获
            with open(trial_dir / "training_log.txt", "w") as log_file:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    # 不使用PIPE，直接将输出显示到控制台
                    stdout=None,
                    stderr=None,
                    universal_newlines=True,
                    bufsize=1  # 行缓冲
                )
                
                # 等待进程完成
                process.wait()
                returncode = process.returncode
                
                if returncode != 0:
                    # 如果CUDA失败，切换到CPU模式
                    if use_cuda:
                        print(f"\n{'='*80}")
                        print(f"CUDA训练失败 (返回码: {returncode})，切换到CPU模式重试...")
                        print(f"{'='*80}\n")
                        
                        # 添加no_cuda参数
                        all_args.append("--no_cuda")
                        cmd = [sys.executable, self.run_ner_path] + all_args
                        
                        # 直接在控制台显示输出
                        process = subprocess.Popen(
                            cmd,
                            env=env,
                            stdout=None,
                            stderr=None,
                            universal_newlines=True
                        )
                        process.wait()
                        returncode = process.returncode
                        use_cuda = False
            
            result_file = trial_dir / "test_results.txt"
            if not result_file.exists():
                logger.error(f"Result file not found: {result_file} for trial {trial}")
                # Create a dummy result file if NER script failed to create it but exited 0
                with open(result_file, "w") as f_dummy:
                    f_dummy.write("f1 = 0.0\nprecision = 0.0\nrecall = 0.0\n")
                metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                metrics = self._parse_result_file(result_file)

            metrics["status"] = "success"
            metrics["config"] = config
            metrics["trial"] = trial
            metrics["used_gpu"] = use_cuda # This will be False if CPU retry happened

            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]
                self.best_config = config
                self.best_trial = trial
                logger.info(f"New best F1: {self.best_f1:.4f} (Trial {trial})")
                print(f"\n{'*'*80}")
                print(f"新的最佳F1分数: {self.best_f1:.4f} (试验 {trial})")
                print(f"{'*'*80}\n")
                self._visualize_best_config() # Visualize new best
            
            if self.skip_model_saving:
                logger.info(f"Trial {trial}: Skipping model saving, cleaning up trial directory...")
                for item in trial_dir.glob("*"):
                    if item.name not in ["test_results.txt", "training_log.txt", "visualizations"] and item.name != self.viz_dir.name :
                        try:
                            if item.is_file(): os.remove(item)
                            elif item.is_dir(): shutil.rmtree(item)
                        except Exception as e:
                            logger.warning(f"无法删除 {item}: {e}")
            
            print(f"\n{'='*50}")
            print(f"试验 {trial} 结果:")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  使用GPU: {metrics['used_gpu']}") # Corrected to use metrics['used_gpu']
            print(f"{'='*50}\n")
            
            return metrics
        
        except Exception as e:
            logger.exception(f"Unhandled error running trial {trial}: {e}")
            traceback.print_exc()
            # Ensure a dummy result file exists for consistent error handling downstream
            if not (trial_dir / "test_results.txt").exists():
                 with open(trial_dir / "test_results.txt", "w") as f_err_res:
                    f_err_res.write(f"f1 = 0.0\nprecision = 0.0\nrecall = 0.0\n")
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "status": "error", "config": config, "trial": trial, "error_message": str(e)}

    def _log_config(self, config: Dict[str, Any], trial: int) -> None:
        config_str = f"Trial {trial} Configuration:\n"
        for name, value in config.items():
            config_str += f"  {name}: {value}\n"
        logger.info(config_str)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'-'*50}\n")
            f.write(f"Trial {trial} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(config_str)
            f.write(f"{'-'*50}\n")
    
    def _parse_result_file(self, result_file: Path) -> Dict[str, float]:
        metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                for line in f:
                    match = re.search(r"(eval_precision|eval_recall|eval_f1|precision|recall|f1)\s*=\s*([0-9.]+)", line, re.IGNORECASE)
                    if match:
                        metric_name = match.group(1).lower().replace("eval_", "")
                        metric_value = float(match.group(2))
                        if metric_name in metrics: # Ensure we only update known keys
                             metrics[metric_name] = metric_value
        except Exception as e:
            logger.error(f"Error parsing result file {result_file}: {e}")
        return metrics
    
    def _update_ppo_network(self):
        if not self.memory_rewards: # Check if there's anything to update from
            return
        
        logger.info(f"Updating PPO network with {len(self.memory_rewards)} samples...")

        # Convert lists to tensors
        states = torch.stack(self.memory_states)
        actions = torch.stack(self.memory_actions)
        old_log_probs = torch.stack(self.memory_log_probs)
        rewards = torch.tensor(self.memory_rewards, dtype=torch.float32).unsqueeze(1) # Ensure rewards is [N, 1]

        # For non-episodic or single-step reward, returns can be just the rewards
        # Or, if we consider each trial an episode, GAE could be calculated here if we had 'dones'
        # For simplicity in hyperparameter tuning, direct F1 score is often used as reward.
        # Let's assume returns are the rewards themselves for now, or a scaled version.
        # Advantage calculation (simple version: A(s,a) = R - V(s))
        with torch.no_grad():
            _, _, current_values = self.ppo_network(states)
        advantages = rewards - current_values
        # Normalize advantages (optional but often helpful)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            # Shuffle data (optional, but good for IID assumption)
            indices = torch.randperm(states.size(0))
            for start_idx in range(0, states.size(0), self.ppo_batch_size):
                end_idx = start_idx + self.ppo_batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards = rewards[batch_indices] # For value loss

                new_log_probs, state_values, entropy = self.ppo_network.evaluate(batch_states, batch_actions)
                
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * nn.functional.mse_loss(state_values, batch_rewards) # Target for value is the observed reward
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ppo_network.parameters(), 0.5) # Gradient clipping
                self.optimizer.step()
        
        # Clear memory after update
        self.memory_states.clear()
        self.memory_actions.clear()
        self.memory_log_probs.clear()
        self.memory_rewards.clear()
        self.memory_values.clear() # If used
        self.memory_dones.clear() # If used
        logger.info("PPO network update complete.")

    def _get_config_from_ppo(self, trial_num: int) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        # State could be as simple as the normalized trial number, or more complex
        # For now, use the previously chosen normalized config as state (or random for first PPO step)
        
        # Use self.current_state_for_ppo which is updated after each step
        state_tensor = self.current_state_for_ppo.unsqueeze(0)

        with torch.no_grad():
            self.ppo_network.eval() # Set to evaluation mode for action sampling
            # Action is in normalized space [-1, 1] due to tanh in PPONetwork
            normalized_action_tensor, log_prob = self.ppo_network.get_action(state_tensor)
            self.ppo_network.train() # Set back to training mode
        
        # Denormalize action to get config. Action is already in [-1,1], map to [0,1] for denormalization
        # The denormalize_vector expects values in [0,1]
        action_for_denorm = (normalized_action_tensor.squeeze(0) + 1.0) / 2.0 
        action_for_denorm = torch.clamp(action_for_denorm, 0.0, 1.0) # Ensure it's in [0,1]

        config = self.hyperparameter_space.denormalize_vector(action_for_denorm)
        return config, normalized_action_tensor.squeeze(0), log_prob # Return the raw action and log_prob

    def optimize(self) -> Dict[str, Any]:
        logger.info(f"Starting PPO optimization with {self.n_trials} trials")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"PPO NER Hyperparameter Optimization\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total trials: {self.n_trials}\n")
            f.write(f"Model: {self.model_type} ({self.model_name})\n")
            f.write("-" * 50 + "\n\n")
        
        if self.results_file.exists():
            try:
                with open(self.results_file, "r", encoding="utf-8") as f:
                    self.results = json.load(f)
                logger.info(f"Loaded {len(self.results)} previous results")
                for result in self.results:
                    if result.get("status") == "success":
                        if result["f1"] > self.best_f1:
                            self.best_f1 = result["f1"]
                            self.best_config = result["config"]
                            self.best_trial = result["trial"]
                        if result["f1"] > self.best_f1_for_early_stopping : # For early stopping
                            self.best_f1_for_early_stopping = result["f1"]
            except json.JSONDecodeError:
                 logger.warning(f"Could not parse existing results file: {self.results_file}. Starting fresh for results tracking.")
                 self.results = []


        start_trial = len(self.results)
        if start_trial > 0 and self.results[-1].get("status") == "success":
             # Initialize PPO state based on the last successful config
             last_successful_config = self.results[-1]["config"]
             self.current_state_for_ppo = self.hyperparameter_space.normalize_config(last_successful_config)
        else:
             # Initial state for PPO (e.g., normalized default or random)
             self.current_state_for_ppo = self.hyperparameter_space.normalize_config(self.hyperparameter_space.default_config())


        for trial in range(start_trial, self.n_trials):
            logger.info(f"Starting trial {trial + 1}/{self.n_trials}") # User-facing trial number
            
            config_to_run: Dict[str, Any]
            action_tensor: Optional[torch.Tensor] = None
            log_prob_tensor: Optional[torch.Tensor] = None

            if trial < self.random_trials_threshold:
                logger.info(f"Trial {trial + 1}: Random sampling for initial exploration.")
                config_to_run = self.hyperparameter_space.sample_random()
                # For random trials, we don't have an action from PPO network yet.
                # We can create a dummy normalized version if needed for state update,
                # or simply not store it in PPO memory until PPO starts acting.
                # For simplicity, let's not add random trials to PPO memory directly,
                # PPO will learn from the F1 scores achieved by these random configs.
            else:
                logger.info(f"Trial {trial + 1}: PPO sampling.")
                config_to_run, action_tensor, log_prob_tensor = self._get_config_from_ppo(trial)

            config_fingerprint = self.hyperparameter_space.get_fingerprint(config_to_run)
            if config_fingerprint in self.hyperparameter_space.explored_configs:
                logger.info(f"Trial {trial + 1}: Config {config_fingerprint} already explored, slightly perturbing.")
                for name_pert in config_to_run:
                    param_type = self.hyperparameter_space.params[name_pert]["type"]
                    if param_type in ["float", "log_float"]:
                        perturbation = 1.0 + np.random.normal(0, 0.01) # Small multiplicative noise
                        if param_type == "log_float":
                             config_to_run[name_pert] = np.clip(config_to_run[name_pert] * perturbation, 
                                                                self.hyperparameter_space.params[name_pert]["min"], 
                                                                self.hyperparameter_space.params[name_pert]["max"])
                        else: # float
                             current_val = config_to_run[name_pert]
                             range_val = self.hyperparameter_space.params[name_pert]["max"] - self.hyperparameter_space.params[name_pert]["min"]
                             # Additive noise scaled by a small fraction of the range
                             noise = np.random.normal(0, 0.01) * range_val 
                             config_to_run[name_pert] = np.clip(current_val + noise,
                                                                self.hyperparameter_space.params[name_pert]["min"],
                                                                self.hyperparameter_space.params[name_pert]["max"])
                    elif param_type == "int":
                        # For integers, add/subtract 1 or a small random int, then clip
                        noise = random.choice([-2, -1, 0, 1, 2]) 
                        config_to_run[name_pert] = np.clip(config_to_run[name_pert] + noise,
                                                           self.hyperparameter_space.params[name_pert]["min"],
                                                           self.hyperparameter_space.params[name_pert]["max"])
                config_fingerprint = self.hyperparameter_space.get_fingerprint(config_to_run) # Update fingerprint
            
            self.hyperparameter_space.explored_configs[config_fingerprint] = True
            
            logger.info(f"Trial {trial + 1}: Using config: {config_to_run}")
            result = self.run_trial(config_to_run, trial + 1) # Pass user-facing trial number
            self.results.append(result)
            
            with open(self.results_file, "w", encoding="utf-8") as f_res:
                json.dump(self.results, f_res, indent=2)
            
            if result["status"] == "success":
                reward = result["f1"] 
                # Update current state for PPO to be the normalized version of the config just run
                self.current_state_for_ppo = self.hyperparameter_space.normalize_config(config_to_run)

                # Store experience only if it came from PPO policy
                if trial >= self.random_trials_threshold and action_tensor is not None and log_prob_tensor is not None:
                    self.memory_states.append(self.current_state_for_ppo.clone()) # Store the state that LED to this action
                    self.memory_actions.append(action_tensor.clone())
                    self.memory_log_probs.append(log_prob_tensor.clone())
                    self.memory_rewards.append(reward)
                    # self.memory_dones.append(True) # Each trial is an episode end

                with open(self.log_file, "a", encoding="utf-8") as f_log_main:
                    f_log_main.write(f"\nTrial {trial + 1} Results:\n")
                    f_log_main.write(f"  F1: {result['f1']:.4f}\n")
                    f_log_main.write(f"  Precision: {result['precision']:.4f}\n")
                    f_log_main.write(f"  Recall: {result['recall']:.4f}\n")
                    if self.best_trial == (trial + 1):
                        f_log_main.write("  ** New Best Configuration **\n")

                if self.early_stopping:
                    if result["f1"] > self.best_f1_for_early_stopping + self.min_delta:
                        self.best_f1_for_early_stopping = result["f1"]
                        self.no_improvement_count = 0
                        logger.info(f"Trial {trial + 1}: F1 improved to {result['f1']:.4f} (for early stopping). Resetting patience.")
                    else:
                        self.no_improvement_count += 1
                        logger.info(f"Trial {trial + 1}: No significant F1 improvement for early stopping. Count: {self.no_improvement_count}/{self.patience}")
                    
                    if self.no_improvement_count >= self.patience:
                        logger.info(f"Early stopping triggered after trial {trial + 1}. No F1 improvement for {self.patience} trials.")
                        print(f"\n{'='*80}")
                        print(f"提前停止优化: {self.patience} 次连续试验未见显著改善")
                        print(f"当前最佳F1: {self.best_f1:.4f} (试验 {self.best_trial})")
                        print(f"{'='*80}\n")
                        break 
            else: # Trial failed
                # Optionally, assign a negative reward or skip PPO update for this step
                logger.warning(f"Trial {trial+1} failed, skipping PPO memory update for this step.")


            # Update PPO network periodically or after enough samples
            if len(self.memory_rewards) >= self.ppo_batch_size and trial >= self.random_trials_threshold :
                self._update_ppo_network()
            
            self._plot_results() # Plot after each trial
        
        # Final PPO update with any remaining samples
        if len(self.memory_rewards) > 0 and trial >= self.random_trials_threshold:
             self._update_ppo_network()

        with open(self.log_file, "a", encoding="utf-8") as f_log_final:
            f_log_final.write("\n" + "="*50 + "\n")
            f_log_final.write(f"Optimization Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_log_final.write(f"Best F1: {self.best_f1:.4f} (Trial {self.best_trial})\n")
            if self.best_config:
                f_log_final.write("Best Configuration:\n")
                for name, value in self.best_config.items():
                    f_log_final.write(f"  {name}: {value}\n")
            else:
                f_log_final.write("No valid best configuration found.\n")
        
        return {
            "best_config": self.best_config,
            "best_f1": self.best_f1,
            "best_trial": self.best_trial,
            "all_results": self.results
        }

    def _plot_results(self):
        if not self.results: return
        valid_results = [r for r in self.results if r.get("status") == "success" and "f1" in r and "trial" in r and "config" in r]
        if not valid_results: return

        trials = [r["trial"] for r in valid_results]
        f1_scores = [r["f1"] for r in valid_results]
        configs = [r["config"] for r in valid_results]

        plt.figure(figsize=(12, 7))
        plt.plot(trials, f1_scores, 'o-', color='dodgerblue', label='F1 Score per Trial')
        if self.best_f1 > 0:
            plt.axhline(y=self.best_f1, color='red', linestyle='--', label=f'Best F1: {self.best_f1:.4f} (Trial {self.best_trial})')
        plt.xlabel('Trial Number')
        plt.ylabel('F1 Score')
        plt.title('PPO NER Model F1 Score Progression')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / "optimization_progress.png")
        plt.close()
        
        param_names = list(self.hyperparameter_space.params.keys())
        for param_name in param_names:
            plt.figure(figsize=(12, 7))
            param_values = [cfg[param_name] for cfg in configs]
            sc = plt.scatter(param_values, f1_scores, c=trials, cmap='viridis', s=100, alpha=0.8, edgecolors='k', linewidth=0.5)
            cbar = plt.colorbar(sc)
            cbar.set_label('Trial Number')
            if self.best_config and self.best_trial in trials:
                best_idx = trials.index(self.best_trial)
                plt.scatter([param_values[best_idx]], [f1_scores[best_idx]], 
                             s=250, c='red', marker='*', label=f'Best (Trial {self.best_trial})', edgecolors='black', linewidth=1)
            plt.title(f'F1 Score vs {param_name}')
            plt.xlabel(param_name.replace("_", " ").title())
            plt.ylabel('F1 Score')
            if self.hyperparameter_space.params[param_name]["type"] == "log_float":
                plt.xscale('log')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.viz_dir / f"param_{param_name}_vs_f1.png")
            plt.close()

    def _visualize_best_config(self):
        if not self.best_config: return
        param_names = list(self.hyperparameter_space.params.keys())
        best_values_normalized = []
        for name in param_names:
            param_info = self.hyperparameter_space.params[name]
            value = self.best_config[name]
            if param_info["type"] == "log_float":
                norm_value = (np.log(value) - np.log(param_info["min"])) / (np.log(param_info["max"]) - np.log(param_info["min"]))
            else:
                norm_value = (value - param_info["min"]) / (param_info["max"] - param_info["min"])
            best_values_normalized.append(norm_value)
        
        angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
        best_values_normalized_closed = best_values_normalized + [best_values_normalized[0]]
        angles_closed = angles + [angles[0]]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(angles_closed, best_values_normalized_closed, 'o-', linewidth=2, color='deepskyblue')
        ax.fill(angles_closed, best_values_normalized_closed, 'deepskyblue', alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), [name.replace("_", " ").title() for name in param_names], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'Best Configuration (F1: {self.best_f1:.4f}, Trial: {self.best_trial})', size=15, y=1.1)
        plt.tight_layout()
        plt.savefig(self.viz_dir / "best_configuration_radar.png")
        plt.close()

        with open(self.viz_dir / "best_config_details.txt", "w", encoding="utf-8") as f:
            f.write(f"Best Configuration (Trial {self.best_trial}, F1: {self.best_f1:.4f})\n")
            f.write("-" * 50 + "\n")
            for name, value in self.best_config.items():
                f.write(f"{name}: {value}\n")

def main():
    parser = argparse.ArgumentParser(description="Optimize NER model hyperparameters using PPO (Aligned with GP-TS Framework)")
    # Common arguments from GP-TS
    parser.add_argument("--data_dir", type=str, required=True, help="The input data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory for optimization results")
    parser.add_argument("--model_type", type=str, default="roberta", help="Model type (e.g., bert, roberta)")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Pretrained model name or path")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--random_trials", type=int, default=5, help="Number of initial random trials for PPO to gather diverse experience before policy learning dominates.")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs for NER model training per trial")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length for NER model")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for caching pretrained models and datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--run_ner_path", type=str, required=True, help="Full path to the run_ner.py script")
    parser.add_argument("--reset", action="store_true", help="Reset optimization and delete previous results")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output during NER training subprocess")
    parser.add_argument("--skip_model_saving", action="store_true", help="Skip saving NER model files during trials to save disk space")

    # PPO-specific arguments (can be kept as they are specific to the algorithm)
    parser.add_argument("--ppo_epochs", type=int, default=10, help="Number of epochs for PPO policy update")
    parser.add_argument("--ppo_batch_size", type=int, default=5, help="Batch size for PPO policy update") # Renamed from batch_size to ppo_batch_size
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="Clipping epsilon for PPO")
    parser.add_argument("--value_coeff", type=float, default=0.5, help="Value function coefficient in PPO loss")
    parser.add_argument("--entropy_coeff", type=float, default=0.01, help="Entropy coefficient in PPO loss")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate for PPO agent")
    
    # Early stopping for the PPO optimization process
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping for the PPO optimization process")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping (number of trials with no improvement)")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Minimum change in F1 to be considered an improvement for early stopping")

    args = parser.parse_args()

    # Setup main logger (consistent with GP-TS, but log file name specific to PPO)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout), # Ensure logs go to stdout
            logging.FileHandler(Path(args.output_dir) / "ppo_script_main.log") # Main script log
        ]
    )
    # Re-get logger in case basicConfig was called again by imports
    # This logger will be used by the PPOOptimizer class as well.
    global logger 
    logger = logging.getLogger(__name__)


    if not os.path.exists(args.run_ner_path):
        logger.error(f"Error: run_ner.py not found at {args.run_ner_path}")
        sys.exit(1)

    output_path = Path(args.output_dir)
    if args.reset:
        logger.info("重置优化过程，删除以前的结果...")
        if (output_path / "optimization_results.json").exists():
            os.remove(output_path / "optimization_results.json")
        if (output_path / "optimization_progress.png").exists():
            os.remove(output_path / "optimization_progress.png")
        if (output_path / "final_results.json").exists():
            os.remove(output_path / "final_results.json")
        # Clear previous trial directories and visualizations
        for item in output_path.iterdir():
            if item.is_dir() and (item.name.startswith("trial_") or item.name == "visualizations" or item.name == "final_model"):
                logger.info(f"删除之前的目录: {item}")
                shutil.rmtree(item)
        # Recreate viz dir as it's deleted
        (output_path / "visualizations").mkdir(parents=True, exist_ok=True)


    if not os.path.exists(args.data_dir):
        logger.warning(f"数据目录不存在: {args.data_dir}")
        # Attempt to infer data_dir (consistent with GP-TS)
        script_dir = Path(sys.argv[0]).resolve().parent
        possible_paths = [
            script_dir.parent / "BOND" / "dataset" / Path(args.data_dir).name, # If data_dir was a name like conll03_distant
            Path(os.path.dirname(args.run_ner_path)).parent / "dataset" / Path(args.data_dir).name,
        ]
        found_data_dir = False
        for path_option in possible_paths:
            if path_option.exists():
                logger.info(f"找到数据目录: {path_option}")
                args.data_dir = str(path_option)
                found_data_dir = True
                break
        if not found_data_dir:
             logger.error(f"无法自动推断数据目录，请确保 {args.data_dir} 存在。")
             # sys.exit(1) # Optional: exit if data dir is critical and not found


    hyperparameter_space = HyperparameterSpace()
    optimizer = PPOOptimizer(
        hyperparameter_space=hyperparameter_space,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        random_trials=args.random_trials, # Pass to PPOOptimizer
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
        ppo_epochs=args.ppo_epochs,
        ppo_batch_size=args.ppo_batch_size, # Use renamed arg
        clip_epsilon=args.clip_epsilon,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        lr=args.lr,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    best_result = optimizer.optimize()
    
    logger.info(f"Optimization completed!")
    logger.info(f"Best F1: {best_result['best_f1']:.4f} (Trial {best_result['best_trial']})")
    if best_result['best_config']:
        logger.info(f"Best hyperparameters: {best_result['best_config']}")
    else:
        logger.info("No valid best configuration found")

    # Final model training (consistent with GP-TS)
    if best_result['best_config'] is not None and not args.skip_model_saving: # Only if not skipping and best_config exists
        logger.info("Training final model with best hyperparameters...")
        print(f"\n{'='*80}")
        print(f"使用最佳超参数训练最终模型")
        print(f"最佳配置 (来自试验 {best_result['best_trial']}, F1: {best_result['best_f1']:.4f}):")
        for name, value in best_result['best_config'].items():
            print(f"  {name}: {value}")
        print(f"{'='*80}\n")
        
        final_output_dir = Path(args.output_dir) / "final_model"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        final_training_log_path = final_output_dir / "final_training_log.txt"

        base_args_final = [
            "--data_dir", str(args.data_dir),
            "--model_type", args.model_type,
            "--model_name_or_path", args.model_name, # Use the original model_name for final training
            "--output_dir", str(final_output_dir),
            "--num_train_epochs", str(args.n_epochs * 2),  # Train for more epochs
            "--max_seq_length", str(args.max_seq_length),
            "--do_train", "--do_eval", "--do_predict",
            "--evaluate_during_training",
            "--overwrite_output_dir",
            "--seed", str(args.seed + best_result['best_trial']), # Use a consistent seed related to best trial
            "--save_steps", "500", # Effectively disable intermediate saving
            "--logging_steps", "50",
        ]
        if args.cache_dir:
            base_args_final.extend(["--cache_dir", str(args.cache_dir)])
        
        config_args_final = hyperparameter_space.config_to_args(best_result['best_config'])
        all_args_final = base_args_final + config_args_final
        
        env_final = os.environ.copy()
        env_final["PYTHONWARNINGS"] = "ignore"
        env_final["CUDA_LAUNCH_BLOCKING"] = "1"
        
        cmd_final = [sys.executable, args.run_ner_path] + all_args_final
        logger.info(f"Final training command: {' '.join(cmd_final)}")
        
        use_cuda_final = True
        final_returncode = -1

        with open(final_training_log_path, "w", encoding="utf-8") as ftlog:
            process_final = subprocess.Popen(cmd_final, env=env_final, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            for line in process_final.stdout:
                line_strip = line.strip()
                if args.verbose or any(key in line_strip for key in ["Epoch", "eval", "F1", "precision", "recall", "Loss", "Saving model"]):
                    print(line_strip)
                ftlog.write(line)
                ftlog.flush()
            
            stderr_final_output = process_final.stderr.read()
            for line in stderr_final_output.splitlines():
                line_strip = line.strip()
                if args.verbose or "Error" in line_strip or "Warning" in line_strip:
                    print(line_strip, file=sys.stderr)
                ftlog.write(line + "\n")
                ftlog.flush()

            process_final.wait()
            final_returncode = process_final.returncode

        if final_returncode != 0 and use_cuda_final:
            print(f"\n{'='*80}")
            print(f"最终模型CUDA训练失败 (返回码: {final_returncode})，切换到CPU模式重试...")
            print(f"{'='*80}\n")
            all_args_final_cpu = list(all_args_final)
            if "--no_cuda" not in all_args_final_cpu:
                 all_args_final_cpu.append("--no_cuda")
            cmd_final_cpu = [sys.executable, args.run_ner_path] + all_args_final_cpu
            use_cuda_final = False
            logger.info(f"Final training retrying with CPU: {' '.join(cmd_final_cpu)}")
            
            with open(final_training_log_path, "a", encoding="utf-8") as ftlog:
                ftlog.write("\n--- RETRYING FINAL TRAINING WITH CPU ---\n")
                process_final_cpu = subprocess.Popen(cmd_final_cpu, env=env_final, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                for line in process_final_cpu.stdout:
                    line_strip = line.strip()
                    if args.verbose or any(key in line_strip for key in ["Epoch", "eval", "F1", "precision", "recall", "Loss", "Saving model"]):
                        print(line_strip)
                    ftlog.write(line)
                    ftlog.flush()

                stderr_final_cpu_output = process_final_cpu.stderr.read()
                for line in stderr_final_cpu_output.splitlines():
                     line_strip = line.strip()
                     if args.verbose or "Error" in line_strip or "Warning" in line_strip:
                         print(line_strip, file=sys.stderr)
                     ftlog.write(line + "\n")
                     ftlog.flush()
                
                process_final_cpu.wait()
                final_returncode = process_final_cpu.returncode


        if final_returncode == 0:
            print(f"\n{'='*80}")
            print("最终模型训练完成!")
            print(f"模型保存在: {final_output_dir}")
            print(f"{'='*80}\n")
            logger.info(f"Final model training completed successfully! Model saved to {final_output_dir}")
        else:
            logger.error(f"Error training final model: return code {final_returncode}")
            print(f"\n{'='*80}")
            print("最终模型训练失败!")
            print(f"{'='*80}\n")
    elif args.skip_model_saving:
        logger.info("Skipping final model training as --skip_model_saving was specified.")
    else:
        logger.info("No best configuration found or --skip_model_saving was specified, skipping final model training.")


    # Save final overall results (consistent with GP-TS)
    with open(output_path / "final_results.json", "w", encoding="utf-8") as f_final_res:
        json.dump(best_result, f_final_res, indent=2)
    
    logger.info(f"Optimization results and artifacts saved to {args.output_dir}")

if __name__ == "__main__":
    main()
