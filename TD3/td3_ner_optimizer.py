#!/usr/bin/env python
# coding: utf-8
"""
基于双延迟深度确定性策略梯度(TD3)的NER模型超参数优化
此脚本使用TD3强化学习算法来自动调整命名实体识别(NER)模型的超参数。
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
import torch.nn.functional as F
import logging
import shutil
from pathlib import Path
import re
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import traceback

# 设置日志
logger = logging.getLogger(__name__)

# --- TD3 网络模型 ---
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        # Output actions in [-max_action, max_action]
        return self.max_action * torch.tanh(self.layer_3(x))

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        # Q1 architecture
        self.layer_1_q1 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2_q1 = nn.Linear(256, 128)
        self.layer_3_q1 = nn.Linear(128, 1)

        # Q2 architecture
        self.layer_1_q2 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2_q2 = nn.Linear(256, 128)
        self.layer_3_q2 = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.layer_1_q1(sa))
        q1 = F.relu(self.layer_2_q1(q1))
        q1 = self.layer_3_q1(q1)

        q2 = F.relu(self.layer_1_q2(sa))
        q2 = F.relu(self.layer_2_q2(q2))
        q2 = self.layer_3_q2(q2)
        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.layer_1_q1(sa))
        q1 = F.relu(self.layer_2_q1(q1))
        q1 = self.layer_3_q1(q1)
        return q1

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1)) # In this problem, done is always True after each trial

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

# 超参数空间定义类 (与GP-TS/PPO一致, 保持不变)
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
        self.explored_configs = {} # To avoid re-evaluating exact same configs

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

    def normalize_config(self, config: Dict[str, Any]) -> np.ndarray: # Changed to np.ndarray for buffer
        normalized = []
        for name, param_info in self.params.items():
            value = config[name]
            if param_info["type"] == "log_float":
                log_min = np.log(param_info["min"])
                log_max = np.log(param_info["max"])
                norm_value = (np.log(value) - log_min) / (log_max - log_min) if (log_max - log_min) > 0 else 0.0
            else:
                norm_value = (value - param_info["min"]) / (param_info["max"] - param_info["min"]) if (param_info["max"] - param_info["min"]) > 0 else 0.0
            normalized.append(norm_value)
        return np.array(normalized, dtype=np.float32)

    def denormalize_vector(self, vector: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        config = {}
        if isinstance(vector, torch.Tensor):
            vector = vector.cpu().numpy() # Convert to numpy if it's a tensor

        for i, (name, param_info) in enumerate(self.params.items()):
            norm_value = vector[i]
            norm_value = max(0.0, min(1.0, norm_value)) # Clamp normalized value to [0, 1]

            if param_info["type"] == "int":
                unnorm_value = int(round(norm_value * (param_info["max"] - param_info["min"]) + param_info["min"]))
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
                args.extend(["--learning_rate", f"{value:.8e}"])
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

# TD3优化器
class TD3Optimizer:
    def __init__(
        self,
        hyperparameter_space: HyperparameterSpace,
        output_dir: str,
        n_trials: int = 100,
        random_trials: int = 10, # Number of trials for initial random exploration
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
        # TD3特定参数
        expl_noise: float = 0.1,      # Exploration noise std
        td3_batch_size: int = 64,       # Batch size for TD3 updates from replay buffer
        discount: float = 0.99,     # Discount factor (might not be very relevant here as episode length is 1)
        tau: float = 0.005,         # Target networkPolyark update rate
        policy_noise: float = 0.2,    # Noise added to target policy during critic update
        noise_clip: float = 0.5,      # Range to clip target policy noise
        policy_freq: int = 2,       # Frequency of delayed policy updates
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        replay_buffer_size: int = int(1e5),
        # 早停参数
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.001,
        # 添加新参数
        dataset_name: str = "Unknown",  # 数据集名称
    ):
        self.hyperparameter_space = hyperparameter_space
        self.output_dir = Path(output_dir)
        self.n_trials = n_trials
        self.random_trials_threshold = random_trials
        self.seed = seed
        self.data_dir = data_dir
        self.model_type = model_type
        self.model_name = model_name
        self.n_epochs_ner = n_epochs
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir
        self.run_ner_path = run_ner_path
        self.verbose = verbose
        self.skip_model_saving = skip_model_saving
        self.dataset_name = dataset_name

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.state_dim = self.hyperparameter_space.dim
        self.action_dim = self.hyperparameter_space.dim
        self.max_action = 1.0 # Actions are normalized to [-1, 1]

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_1 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_1_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)

        self.critic_2 = Critic(self.state_dim, self.action_dim).to(self.device) # Second critic
        self.critic_2_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)


        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=replay_buffer_size)

        self.expl_noise = expl_noise
        self.td3_batch_size = td3_batch_size
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0 # For delayed policy updates

        # 早停参数
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.no_improvement_count = 0
        self.best_f1_for_early_stopping = 0.0

        self.best_f1 = 0.0
        self.best_config = None
        self.best_trial = -1
        self.results = []

        self.results_file = self.output_dir / "optimization_results.json"
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "optimization_log.txt"
        
        # Initial state: normalized default config
        self.current_rl_state = self.hyperparameter_space.normalize_config(self.hyperparameter_space.default_config())


    def _select_action(self, state: np.ndarray, is_exploratory: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
        if is_exploratory:
            noise = np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action # This action is in [-1, 1] range

    def _update_td3_networks(self):
        if self.replay_buffer.size < self.td3_batch_size:
            return

        self.total_it += 1
        logger.info(f"Updating TD3 networks (iteration {self.total_it}). Buffer size: {self.replay_buffer.size}")

        state, action, reward, next_state, done = self.replay_buffer.sample(self.td3_batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_1_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            # For hyperparameter tuning, 'done' is always True, so target_Q simplifies.
            # If we consider future rewards, 'done' would be important. Here, each trial is an episode.
            target_Q = reward + (1 - done) * self.discount * target_Q # effectively target_Q = reward if done=1

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic_1(state, action)

        # Compute critic loss
        critic_1_loss = F.mse_loss(current_Q1, target_Q)
        critic_2_loss = F.mse_loss(current_Q2, target_Q) # TD3 uses two critics

        # Optimize the critics
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_1.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            logger.info(f"TD3 Actor and Target Networks Updated. Actor Loss: {actor_loss.item():.4f}, Critic1 Loss: {critic_1_loss.item():.4f}, Critic2 Loss: {critic_2_loss.item():.4f}")


    # Methods like run_trial, _log_config, _parse_result_file, _plot_results, _visualize_best_config
    # remain largely the same as in PPOOptimizer, with minor adjustments for state/action handling.
    # I will copy them and make necessary small changes.

    def run_trial(self, config: Dict[str, Any], trial: int) -> Dict[str, Any]:
        """运行单次试验，使用给定的超参数配置训练和评估NER模型"""
        trial_dir = self.output_dir / f"trial_{trial}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        use_cuda = True
        
        try:
            self._log_config(config, trial)
            base_args = [
                "--data_dir", str(self.data_dir), "--model_type", self.model_type,
                "--model_name_or_path", self.model_name, "--output_dir", str(trial_dir),
                "--num_train_epochs", str(self.n_epochs_ner), "--max_seq_length", str(self.max_seq_length),
                "--do_train", "--do_eval", "--do_predict", "--evaluate_during_training",
                "--overwrite_output_dir", "--seed", str(self.seed + trial),
                "--save_steps", "5000", "--logging_steps", "500"
            ]
            if self.cache_dir: base_args.extend(["--cache_dir", str(self.cache_dir)])
            config_args = self.hyperparameter_space.config_to_args(config)
            all_args = base_args + config_args
            cmd = [sys.executable, self.run_ner_path] + all_args
            logger.info(f"Trial {trial}: Running command: {' '.join(cmd)}")
            
            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "ignore"; env["PYTORCH_CUDA_ALLOC_CONF"] = ""; env["CUDA_LAUNCH_BLOCKING"] = "1"
            
            print(f"\n{'='*80}\n开始试验 {trial}/{self.n_trials} (尝试使用 CUDA)\n超参数配置:")
            for name, value in config.items(): print(f"  {name}: {value}")
            print(f"{'='*80}\n")
            
            # NER subprocess (stdout is not piped, directly visible)
            with open(trial_dir / "training_log.txt", "w") as log_file_handle: # Ensure it's handle
                process = subprocess.Popen(cmd, env=env, stdout=None, stderr=None, universal_newlines=True, bufsize=1)
                process.wait()
                returncode = process.returncode
                
                if returncode != 0 and use_cuda:
                    print(f"\n{'='*80}\nCUDA训练失败 (返回码: {returncode})，切换到CPU模式重试...\n{'='*80}\n")
                    all_args.append("--no_cuda"); cmd = [sys.executable, self.run_ner_path] + all_args
                    process = subprocess.Popen(cmd, env=env, stdout=None, stderr=None, universal_newlines=True)
                    process.wait(); returncode = process.returncode; use_cuda = False
            
            result_file = trial_dir / "test_results.txt"
            if not result_file.exists():
                logger.error(f"Result file not found: {result_file} for trial {trial}")
                with open(result_file, "w") as f_dummy: f_dummy.write("f1 = 0.0\nprecision = 0.0\nrecall = 0.0\n")
                metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                metrics = self._parse_result_file(result_file)

            metrics["status"] = "success"; metrics["config"] = config; metrics["trial"] = trial; metrics["used_gpu"] = use_cuda

            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]; self.best_config = config; self.best_trial = trial
                logger.info(f"New best F1: {self.best_f1:.4f} (Trial {trial})")
                print(f"\n{'*'*80}\n新的最佳F1分数: {self.best_f1:.4f} (试验 {trial})\n{'*'*80}\n")
                self._visualize_best_config()
            
            if self.skip_model_saving:
                logger.info(f"Trial {trial}: Skipping model saving, cleaning up trial directory...")
                for item in trial_dir.glob("*"):
                    if item.name not in ["test_results.txt", "training_log.txt", "visualizations", self.viz_dir.name]:
                        try:
                            if item.is_file(): os.remove(item)
                            elif item.is_dir(): shutil.rmtree(item)
                        except Exception as e: logger.warning(f"无法删除 {item}: {e}")
            
            print(f"\n{'='*50}\n试验 {trial} 结果:\n  F1: {metrics['f1']:.4f}\n  精确率: {metrics['precision']:.4f}\n  召回率: {metrics['recall']:.4f}\n  使用GPU: {metrics['used_gpu']}\n{'='*50}\n")
            return metrics
        
        except Exception as e:
            logger.exception(f"Unhandled error running trial {trial}: {e}")
            traceback.print_exc()
            if not (trial_dir / "test_results.txt").exists():
                 with open(trial_dir / "test_results.txt", "w") as f_err_res: f_err_res.write("f1 = 0.0\nprecision = 0.0\nrecall = 0.0\n")
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "status": "error", "config": config, "trial": trial, "error_message": str(e)}

    def _log_config(self, config: Dict[str, Any], trial: int) -> None:
        config_str = f"Trial {trial} Configuration:\n" + "".join([f"  {name}: {value}\n" for name, value in config.items()])
        logger.info(config_str)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'-'*50}\nTrial {trial} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n{config_str}{'-'*50}\n")
    
    def _parse_result_file(self, result_file: Path) -> Dict[str, float]:
        metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                for line in f:
                    match = re.search(r"(eval_precision|eval_recall|eval_f1|precision|recall|f1)\s*=\s*([0-9.]+)", line, re.IGNORECASE)
                    if match:
                        metric_name = match.group(1).lower().replace("eval_", "")
                        if metric_name in metrics: metrics[metric_name] = float(match.group(2))
        except Exception as e: logger.error(f"Error parsing result file {result_file}: {e}")
        return metrics

    def optimize(self) -> Dict[str, Any]:
        logger.info(f"Starting TD3 optimization with {self.n_trials} trials")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"TD3 NER Hyperparameter Optimization\nStarted at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total trials: {self.n_trials}\nModel: {self.model_type} ({self.model_name})\n{'-'*50}\n\n")
        
        if self.results_file.exists():
            try:
                with open(self.results_file, "r", encoding="utf-8") as f: self.results = json.load(f)
                logger.info(f"Loaded {len(self.results)} previous results")
                for res in self.results:
                    if res.get("status") == "success":
                        if res["f1"] > self.best_f1: self.best_f1 = res["f1"]; self.best_config = res["config"]; self.best_trial = res["trial"]
                        if res["f1"] > self.best_f1_for_early_stopping: self.best_f1_for_early_stopping = res["f1"]
            except json.JSONDecodeError:
                 logger.warning(f"Could not parse results file: {self.results_file}. Starting fresh."); self.results = []

        start_trial_idx = len(self.results) # 0-indexed
        if start_trial_idx > 0 and self.results[-1].get("status") == "success":
            self.current_rl_state = self.hyperparameter_space.normalize_config(self.results[-1]["config"])
        else:
            self.current_rl_state = self.hyperparameter_space.normalize_config(self.hyperparameter_space.default_config())

        for trial_idx in range(start_trial_idx, self.n_trials):
            trial_num_user = trial_idx + 1 # 1-indexed for user display/logging
            logger.info(f"Starting trial {trial_num_user}/{self.n_trials}")
            
            current_state_for_buffer = self.current_rl_state.copy() # State that leads to the action
            
            if trial_idx < self.random_trials_threshold:
                logger.info(f"Trial {trial_num_user}: Random sampling for initial exploration.")
                config_to_run = self.hyperparameter_space.sample_random()
                # The 'action' for random trials is just the normalized version of the random config
                # This is needed for the replay buffer
                action_normalized_np = self.hyperparameter_space.normalize_config(config_to_run)
            else:
                logger.info(f"Trial {trial_num_user}: TD3 sampling.")
                # Action from actor is in [-1, 1]
                action_normalized_np = self._select_action(self.current_rl_state, is_exploratory=True)
                # Denormalize this action (which is in [-1,1]) to get hyperparameter config
                # The denormalize_vector expects input in [0,1], so map it
                action_for_denorm = (action_normalized_np + 1.0) / 2.0
                action_for_denorm = np.clip(action_for_denorm, 0.0, 1.0)
                config_to_run = self.hyperparameter_space.denormalize_vector(action_for_denorm)

            # Handle duplicate configurations by perturbing
            config_fingerprint = self.hyperparameter_space.get_fingerprint(config_to_run)
            if config_fingerprint in self.hyperparameter_space.explored_configs:
                logger.info(f"Trial {trial_num_user}: Config {config_fingerprint} already explored, slightly perturbing.")
                for name_pert in config_to_run:
                    param_info = self.hyperparameter_space.params[name_pert]
                    perturb_scale = 0.01
                    if param_info["type"] in ["float", "log_float"]:
                        noise = np.random.normal(0, perturb_scale)
                        if param_info["type"] == "log_float":
                            config_to_run[name_pert] = np.clip(config_to_run[name_pert] * (1 + noise), param_info["min"], param_info["max"])
                        else:
                            range_val = param_info["max"] - param_info["min"]
                            config_to_run[name_pert] = np.clip(config_to_run[name_pert] + noise * range_val, param_info["min"], param_info["max"])
                    elif param_info["type"] == "int":
                        noise_int = random.choice([-max(1,int(abs(param_info["max"] - param_info["min"]) * perturb_scale)), 0, max(1,int(abs(param_info["max"] - param_info["min"]) * perturb_scale))])
                        config_to_run[name_pert] = np.clip(config_to_run[name_pert] + noise_int, param_info["min"], param_info["max"])
                
                config_fingerprint = self.hyperparameter_space.get_fingerprint(config_to_run)
                 # Update the action_normalized_np if config was perturbed (important for replay buffer)
                action_normalized_np = self.hyperparameter_space.normalize_config(config_to_run)
                # And remap to [-1, 1] as this is what actor outputs
                action_normalized_np = (action_normalized_np * 2.0) - 1.0
                action_normalized_np = np.clip(action_normalized_np, -1.0, 1.0)


            self.hyperparameter_space.explored_configs[config_fingerprint] = True
            
            logger.info(f"Trial {trial_num_user}: Using config: {config_to_run}")
            result_metrics = self.run_trial(config_to_run, trial_num_user)
            self.results.append(result_metrics)
            
            with open(self.results_file, "w", encoding="utf-8") as f_res: json.dump(self.results, f_res, indent=2)
            
            if result_metrics["status"] == "success":
                reward = result_metrics["f1"]
                next_rl_state = self.hyperparameter_space.normalize_config(config_to_run)
                done_flag = True # Each trial is an episode end

                # Add to replay buffer: (s, a, r, s', d)
                # s: current_state_for_buffer (state that led to action)
                # a: action_normalized_np (action taken, in [-1,1])
                # r: reward
                # s': next_rl_state (resulting state from action)
                # d: done_flag
                self.replay_buffer.add(current_state_for_buffer, action_normalized_np, reward, next_rl_state, done_flag)
                
                self.current_rl_state = next_rl_state # Update current RL state for next iteration

                with open(self.log_file, "a", encoding="utf-8") as f_log_main:
                    f_log_main.write(f"\nTrial {trial_num_user} Results:\n  F1: {result_metrics['f1']:.4f}\n")
                    if self.best_trial == trial_num_user: f_log_main.write("  ** New Best Configuration **\n")

                # Update TD3 networks (can be done multiple times per step if desired)
                # For simplicity, one update call per environment step (trial)
                if trial_idx >= self.random_trials_threshold: # Start training TD3 agent after random trials
                    self._update_td3_networks()


                if self.early_stopping:
                    if result_metrics["f1"] > self.best_f1_for_early_stopping + self.min_delta:
                        self.best_f1_for_early_stopping = result_metrics["f1"]; self.no_improvement_count = 0
                        logger.info(f"Trial {trial_num_user}: F1 improved (for early stopping). Resetting patience.")
                    else:
                        self.no_improvement_count += 1
                        logger.info(f"Trial {trial_num_user}: No sig. F1 improvement. Count: {self.no_improvement_count}/{self.patience}")
                    if self.no_improvement_count >= self.patience:
                        logger.info(f"Early stopping triggered after trial {trial_num_user}.")
                        print(f"\n{'='*80}\n提前停止优化: {self.patience} 次连续试验未见显著改善\n当前最佳F1: {self.best_f1:.4f} (试验 {self.best_trial})\n{'='*80}\n")
                        break
            else:
                logger.warning(f"Trial {trial_num_user} failed, skipping replay buffer update for this step.")
            
            self._plot_results()

        with open(self.log_file, "a", encoding="utf-8") as f_log_final:
            f_log_final.write(f"\n{'='*50}\nOptimization Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_log_final.write(f"Best F1: {self.best_f1:.4f} (Trial {self.best_trial})\n")
            if self.best_config:
                f_log_final.write("Best Configuration:\n" + "".join([f"  {name}: {value}\n" for name, value in self.best_config.items()]))
            else: f_log_final.write("No valid best configuration found.\n")
        
        return {"best_config": self.best_config, "best_f1": self.best_f1, "best_trial": self.best_trial, "all_results": self.results}

    def _plot_results(self): # Identical to PPO version
        if not self.results: return
        valid_results = [r for r in self.results if r.get("status") == "success" and "f1" in r and "trial" in r and "config" in r]
        if not valid_results: return

        trials = [r["trial"] for r in valid_results]
        f1_scores = [r["f1"] for r in valid_results]
        configs = [r["config"] for r in valid_results]

        plt.figure(figsize=(12, 7))
        plt.plot(trials, f1_scores, 'o-', color='dodgerblue', label='F1 Score per Trial')
        if self.best_f1 > 0 and self.best_trial is not None and self.best_trial > 0 : # ensure best_trial is valid
            plt.axhline(y=self.best_f1, color='red', linestyle='--', label=f'Best F1: {self.best_f1:.4f} (Trial {self.best_trial})')
        plt.xlabel('Trial Number')
        plt.ylabel('F1 Score')
        plt.title(f'TD3 NER Model F1 Score Progression - Dataset: {self.dataset_name}')
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
            cbar = plt.colorbar(sc); cbar.set_label('Trial Number')
            if self.best_config and self.best_trial is not None and self.best_trial > 0 and self.best_trial in trials:
                try: # Add try-except for safety if best_trial might not align with filtered valid_results trials
                    best_idx = trials.index(self.best_trial)
                    plt.scatter([param_values[best_idx]], [f1_scores[best_idx]], s=250, c='red', marker='*', label=f'Best (Trial {self.best_trial})', edgecolors='black', linewidth=1)
                except ValueError:
                    logger.warning(f"Best trial {self.best_trial} not found in current plot data for {param_name}.")

            plt.title(f'F1 Score vs {param_name}'); plt.xlabel(param_name.replace("_", " ").title()); plt.ylabel('F1 Score')
            if self.hyperparameter_space.params[param_name]["type"] == "log_float": plt.xscale('log')
            plt.grid(True, linestyle='--', alpha=0.7); plt.legend(); plt.tight_layout()
            plt.savefig(self.viz_dir / f"param_{param_name}_vs_f1.png"); plt.close()

    def _visualize_best_config(self): # Identical to PPO version
        if not self.best_config: return
        param_names = list(self.hyperparameter_space.params.keys())
        best_values_normalized = []
        for name in param_names:
            param_info = self.hyperparameter_space.params[name]
            value = self.best_config[name]
            log_min = np.log(param_info["min"]); log_max = np.log(param_info["max"])
            if param_info["type"] == "log_float":
                norm_value = (np.log(value) - log_min) / (log_max - log_min) if (log_max - log_min) > 0 else 0.0
            else:
                norm_value = (value - param_info["min"]) / (param_info["max"] - param_info["min"]) if (param_info["max"] - param_info["min"]) > 0 else 0.0
            best_values_normalized.append(norm_value)
        
        angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
        best_values_normalized_closed = best_values_normalized + [best_values_normalized[0]]
        angles_closed = angles + [angles[0]]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(angles_closed, best_values_normalized_closed, 'o-', linewidth=2, color='deepskyblue')
        ax.fill(angles_closed, best_values_normalized_closed, 'deepskyblue', alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), [name.replace("_", " ").title() for name in param_names], fontsize=10)
        ax.set_ylim(0, 1); ax.set_title(f'Best Configuration (F1: {self.best_f1:.4f}, Trial: {self.best_trial})', size=15, y=1.1)
        plt.tight_layout(); plt.savefig(self.viz_dir / "best_configuration_radar.png"); plt.close()

        with open(self.viz_dir / "best_config_details.txt", "w", encoding="utf-8") as f:
            f.write(f"Best Configuration (Trial {self.best_trial}, F1: {self.best_f1:.4f})\n{'-'*50}\n")
            for name, value in self.best_config.items(): f.write(f"{name}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description="Optimize NER model hyperparameters using TD3")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="roberta")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--random_trials", type=int, default=10, help="Number of initial random trials before TD3 starts learning.")
    parser.add_argument("--n_epochs", type=int, default=5, help="NER model epochs per trial")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_ner_path", type=str, required=True)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip_model_saving", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="Unknown", help="数据集名称，用于图表标题")

    # TD3-specific arguments
    parser.add_argument("--expl_noise", type=float, default=0.1, help="Std of Gaussian exploration noise")
    parser.add_argument("--td3_batch_size", type=int, default=64, help="Batch size for TD3 replay buffer sampling")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor for TD3")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate (Polyak averaging)")
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Noise added to target policy for smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", type=int, default=2, help="Frequency of delayed policy updates")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="Learning rate for TD3 Actor")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="Learning rate for TD3 Critic")
    parser.add_argument("--replay_buffer_size", type=int, default=int(1e5), help="Max size of replay buffer")
    
    # Early stopping
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.001)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(Path(args.output_dir) / "td3_script_main.log")]
    )
    global logger; logger = logging.getLogger(__name__) # Ensure global logger is set

    if not os.path.exists(args.run_ner_path): logger.error(f"run_ner.py not found at {args.run_ner_path}"); sys.exit(1)
    
    output_path = Path(args.output_dir)
    if args.reset:
        logger.info("Resetting optimization, deleting previous results...")
        files_to_remove = ["optimization_results.json", "optimization_progress.png", "final_results.json", "optimization_log.txt", "td3_script_main.log"]
        for f_name in files_to_remove:
            if (output_path / f_name).exists(): os.remove(output_path / f_name)
        for item in output_path.iterdir():
            if item.is_dir() and (item.name.startswith("trial_") or item.name == "visualizations" or item.name == "final_model"):
                shutil.rmtree(item)
        (output_path / "visualizations").mkdir(parents=True, exist_ok=True)

    # Data directory inference (optional, adapt if needed)
    if not os.path.exists(args.data_dir):
        logger.warning(f"Data directory {args.data_dir} not found. Attempting to infer...")
        # Basic inference, adjust as per your project structure
        try:
            inferred_data_dir = Path(args.run_ner_path).parent.parent / "dataset" / Path(args.data_dir).name
            if inferred_data_dir.exists():
                args.data_dir = str(inferred_data_dir)
                logger.info(f"Inferred data directory: {args.data_dir}")
            else:
                logger.error(f"Could not infer data directory. Please provide a valid path."); # sys.exit(1)
        except Exception as e:
            logger.error(f"Error during data directory inference: {e}"); # sys.exit(1)


    hyperparameter_space = HyperparameterSpace()
    optimizer = TD3Optimizer(
        hyperparameter_space=hyperparameter_space, output_dir=args.output_dir, n_trials=args.n_trials,
        random_trials=args.random_trials, seed=args.seed, data_dir=args.data_dir,
        model_type=args.model_type, model_name=args.model_name, n_epochs=args.n_epochs,
        max_seq_length=args.max_seq_length, cache_dir=args.cache_dir, run_ner_path=args.run_ner_path,
        verbose=args.verbose, skip_model_saving=args.skip_model_saving,
        expl_noise=args.expl_noise, td3_batch_size=args.td3_batch_size, discount=args.discount,
        tau=args.tau, policy_noise=args.policy_noise, noise_clip=args.noise_clip,
        policy_freq=args.policy_freq, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        replay_buffer_size=args.replay_buffer_size,
        early_stopping=args.early_stopping, patience=args.patience, min_delta=args.min_delta,
        dataset_name=args.dataset_name
    )
    
    best_result = optimizer.optimize()
    
    logger.info(f"Optimization completed! Best F1: {best_result['best_f1']:.4f} (Trial {best_result['best_trial']})")
    if best_result['best_config']: logger.info(f"Best hyperparameters: {best_result['best_config']}")
    else: logger.info("No valid best configuration found")

    # Final model training (similar to PPO script)
    if best_result['best_config'] is not None and not args.skip_model_saving:
        logger.info("Training final model with best hyperparameters...")
        print(f"\n{'='*80}\n使用最佳超参数训练最终模型\n最佳配置 (来自试验 {best_result['best_trial']}, F1: {best_result['best_f1']:.4f}):")
        for name, value in best_result['best_config'].items(): print(f"  {name}: {value}")
        print(f"{'='*80}\n")
        
        final_output_dir = Path(args.output_dir) / "final_model"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_training_log_path = final_output_dir / "final_training_log.txt"

        base_args_final = [
            "--data_dir", str(args.data_dir), "--model_type", args.model_type,
            "--model_name_or_path", args.model_name, "--output_dir", str(final_output_dir),
            "--num_train_epochs", str(args.n_epochs * 2), "--max_seq_length", str(args.max_seq_length),
            "--do_train", "--do_eval", "--do_predict", "--evaluate_during_training",
            "--overwrite_output_dir", "--seed", str(args.seed + best_result.get('best_trial', args.seed)), # Use a related seed
            "--save_steps", "5000", "--logging_steps", "500",
        ]
        if args.cache_dir: base_args_final.extend(["--cache_dir", str(args.cache_dir)])
        config_args_final = hyperparameter_space.config_to_args(best_result['best_config'])
        all_args_final = base_args_final + config_args_final
        cmd_final = [sys.executable, args.run_ner_path] + all_args_final
        logger.info(f"Final training command: {' '.join(cmd_final)}")
        
        env_final = os.environ.copy(); env_final["PYTHONWARNINGS"] = "ignore"; env_final["CUDA_LAUNCH_BLOCKING"] = "1"
        final_returncode = -1; use_cuda_final = True

        with open(final_training_log_path, "w", encoding="utf-8") as ftlog: # NER subprocess (piped)
            process_final = subprocess.Popen(cmd_final, env=env_final, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            for line_type, source in [(process_final.stdout, sys.stdout), (process_final.stderr, sys.stderr)]:
                 if source is not None: # Check if source is not None
                    for line in source:
                        line_strip = line.strip()
                        if args.verbose or any(key in line_strip for key in ["Epoch", "eval", "F1", "Loss", "Saving model"]): print(line_strip)
                        ftlog.write(line); ftlog.flush()
            process_final.wait(); final_returncode = process_final.returncode

        if final_returncode != 0 and use_cuda_final: # CPU Retry
            print(f"\n{'='*80}\n最终模型CUDA训练失败，切换到CPU重试...\n{'='*80}\n")
            all_args_final_cpu = list(all_args_final); all_args_final_cpu.append("--no_cuda")
            cmd_final_cpu = [sys.executable, args.run_ner_path] + all_args_final_cpu
            logger.info(f"Final training retrying with CPU: {' '.join(cmd_final_cpu)}")
            with open(final_training_log_path, "a", encoding="utf-8") as ftlog:
                ftlog.write("\n--- RETRYING FINAL TRAINING WITH CPU ---\n")
                process_final_cpu = subprocess.Popen(cmd_final_cpu, env=env_final, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                for line_type, source_cpu in [(process_final_cpu.stdout, sys.stdout), (process_final_cpu.stderr, sys.stderr)]:
                    if source_cpu is not None:
                        for line_cpu in source_cpu:
                            line_strip_cpu = line_cpu.strip()
                            if args.verbose or any(key in line_strip_cpu for key in ["Epoch", "eval", "F1", "Loss", "Saving model"]): print(line_strip_cpu)
                            ftlog.write(line_cpu); ftlog.flush()
                process_final_cpu.wait(); final_returncode = process_final_cpu.returncode
        
        if final_returncode == 0: logger.info(f"Final model training completed! Model saved to {final_output_dir}")
        else: logger.error(f"Error training final model: return code {final_returncode}")
        print(f"\n{'='*80}\n最终模型训练状态: {'完成' if final_returncode == 0 else '失败'}\n{'='*80}\n")
    
    elif args.skip_model_saving: logger.info("Skipping final model training (--skip_model_saving).")
    else: logger.info("No best config found or skipping final training.")

    with open(output_path / "final_results.json", "w", encoding="utf-8") as f_final_res: json.dump(best_result, f_final_res, indent=2)
    logger.info(f"Optimization results saved to {args.output_dir}")

if __name__ == "__main__":
    main()