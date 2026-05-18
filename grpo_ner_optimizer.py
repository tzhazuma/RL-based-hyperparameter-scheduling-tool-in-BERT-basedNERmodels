#!/usr/bin/env python
# coding: utf-8
"""
基于 Group Relative Policy Optimization (GRPO) 的 NER 模型超参数优化

GRPO 的核心区别于 PPO：
  - 不使用价值网络 (critic/value network)，仅使用策略网络 (policy network)。
  - 每次迭代采样 G 个配置组成一组 (group)，在组内计算相对优势 (relative advantage)：
      A_i = (r_i - mean(r)) / (std(r) + 1e-8)
  - 使用与 PPO 相同的裁剪策略更新 (clipped policy update)。

此脚本的外围框架（参数、日志、输出、可视化）与 PPO/GP-TS 优化器对齐。
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

# 从 PPO 优化器导入共享的超参数空间定义
from ppo_ner_optimizer import HyperparameterSpace

logger = logging.getLogger(__name__)


# ============================================================================
# GRPO 策略网络 —— 仅含策略部分，不含价值网络 (这是 GRPO 与 PPO 的核心区别)
# ============================================================================
class GRPONetwork(nn.Module):
    """只包含策略网络的 GRPO 网络 —— 没有 critic / value head。"""

    def __init__(self, state_dim: int, action_dim: int):
        super(GRPONetwork, self).__init__()
        # 共享特征提取层 (与 PPO 的 shared_layers 一致)
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # 策略均值头
        self.policy_mean = nn.Linear(64, action_dim)
        # 可学习对数标准差
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))

        # 注意：GRPO 不使用 value_head (这是与 PPO 的核心区别)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播：返回动作均值与标准差 (不含状态价值)。"""
        features = self.shared_layers(state)
        action_mean = torch.tanh(self.policy_mean(features))  # 输出在 [-1, 1]
        action_log_std = self.policy_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_std

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从当前策略采样动作并返回对数概率。"""
        action_mean, action_std = self.forward(state)
        dist = Normal(action_mean, action_std)
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, state: torch.Tensor, action: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估给定状态-动作对的 log_prob 与熵 (无状态价值)。"""
        action_mean, action_std = self.forward(state)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ============================================================================
# GRPO 优化器 (框架与 PPO/GP-TS 对齐)
# ============================================================================
class GRPOOptimizer:
    """Group Relative Policy Optimization 超参数优化器。

    GRPO 工作流程：
      1. 对于每次迭代，用当前策略采样 G 个超参数配置。
      2. 并行/串行运行 G 次 BOND NER 训练试验，获得 F1 分数。
      3. 在组内计算归一化相对优势：A_i = (r_i - mean(r)) / (std(r) + 1e-8)
      4. 以 PPO 风格的裁剪目标 (clipped objective) 更新策略，直接使用组内优势。
    """

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
        skip_model_saving: bool = False,
        # GRPO 特定参数
        grpo_group_size: int = 8,     # G: 每次迭代采样的配置数
        grpo_epochs: int = 10,        # 每次策略更新的 epoch 数
        grpo_batch_size: int = 4,     # 策略更新时的 mini-batch 大小
        clip_epsilon: float = 0.2,    # PPO 裁剪 ε
        entropy_coeff: float = 0.01,  # 熵正则化系数
        lr: float = 0.0003,           # 策略网络学习率
        # 早停参数
        early_stopping: bool = False,
        patience: int = 5,
        min_delta: float = 0.001
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

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # GRPO 特定参数
        self.grpo_group_size = grpo_group_size
        self.grpo_epochs = grpo_epochs
        self.grpo_batch_size = grpo_batch_size
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.grpo_lr = lr

        # 早停参数
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.no_improvement_count = 0
        self.best_f1_for_early_stopping = 0.0

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 初始化 GRPO 策略网络 (无 critic!)
        self.state_dim = self.hyperparameter_space.dim
        self.action_dim = self.hyperparameter_space.dim
        self.grpo_network = GRPONetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.grpo_network.parameters(), lr=self.grpo_lr)

        # 存储 GRPO 经验: 每个 trial 存储 (state, action, log_prob, reward)
        self.memory_states: List[torch.Tensor] = []
        self.memory_actions: List[torch.Tensor] = []
        self.memory_log_probs: List[torch.Tensor] = []
        self.memory_rewards: List[float] = []

        # 跟踪最佳结果 (与 PPO/GP-TS 一致)
        self.best_f1 = 0.0
        self.best_config = None
        self.best_trial = -1

        # 存储每次试验的结果
        self.results: List[Dict[str, Any]] = []

        self.results_file = self.output_dir / "optimization_results.json"
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "optimization_log.txt"

        # 当前状态 (用于策略的下一步输入)
        self.current_state_for_grpo = self.hyperparameter_space.normalize_config(
            self.hyperparameter_space.default_config()
        )

    # ------------------------------------------------------------------
    # 试验运行 (与 PPO 版本完全一致)
    # ------------------------------------------------------------------
    def run_trial(self, config: Dict[str, Any], trial: int) -> Dict[str, Any]:
        """运行单次试验，使用给定的超参数配置训练和评估 NER 模型。"""
        trial_dir = self.output_dir / f"trial_{trial}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        use_cuda = True

        try:
            self._log_config(config, trial)

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
                "--save_steps", "500",
                "--logging_steps", "50"
            ]

            if self.cache_dir:
                base_args.extend(["--cache_dir", str(self.cache_dir)])

            config_args = self.hyperparameter_space.config_to_args(config)
            all_args = base_args + config_args

            cmd = [sys.executable, self.run_ner_path] + all_args
            logger.info(f"Trial {trial}: Running command: {' '.join(cmd)}")

            env = os.environ.copy()
            env["PYTHONWARNINGS"] = "ignore"
            env["PYTORCH_CUDA_ALLOC_CONF"] = ""
            env["CUDA_LAUNCH_BLOCKING"] = "1"

            print(f"\n{'='*80}")
            print(f"开始试验 {trial}/{self.n_trials} (尝试使用 CUDA)")
            print(f"超参数配置:")
            for name, value in config.items():
                print(f"  {name}: {value}")
            print(f"{'='*80}\n")

            with open(trial_dir / "training_log.txt", "w") as log_file:
                process = subprocess.Popen(
                    cmd, env=env,
                    stdout=None, stderr=None,
                    universal_newlines=True, bufsize=1
                )
                process.wait()
                returncode = process.returncode

                if returncode != 0 and use_cuda:
                    print(f"\n{'='*80}")
                    print(f"CUDA训练失败 (返回码: {returncode})，切换到CPU模式重试...")
                    print(f"{'='*80}\n")
                    all_args.append("--no_cuda")
                    cmd = [sys.executable, self.run_ner_path] + all_args
                    process = subprocess.Popen(
                        cmd, env=env,
                        stdout=None, stderr=None,
                        universal_newlines=True
                    )
                    process.wait()
                    returncode = process.returncode
                    use_cuda = False

            result_file = trial_dir / "test_results.txt"
            if not result_file.exists():
                logger.error(f"Result file not found: {result_file} for trial {trial}")
                with open(result_file, "w") as f_dummy:
                    f_dummy.write("f1 = 0.0\nprecision = 0.0\nrecall = 0.0\n")
                metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                metrics = self._parse_result_file(result_file)

            metrics["status"] = "success"
            metrics["config"] = config
            metrics["trial"] = trial
            metrics["used_gpu"] = use_cuda

            if metrics["f1"] > self.best_f1:
                self.best_f1 = metrics["f1"]
                self.best_config = config
                self.best_trial = trial
                logger.info(f"New best F1: {self.best_f1:.4f} (Trial {trial})")
                print(f"\n{'*'*80}")
                print(f"新的最佳F1分数: {self.best_f1:.4f} (试验 {trial})")
                print(f"{'*'*80}\n")
                self._visualize_best_config()

            if self.skip_model_saving:
                logger.info(f"Trial {trial}: Skipping model saving, cleaning up trial directory...")
                for item in trial_dir.glob("*"):
                    if item.name not in ["test_results.txt", "training_log.txt", "visualizations"] \
                       and item.name != self.viz_dir.name:
                        try:
                            if item.is_file():
                                os.remove(item)
                            elif item.is_dir():
                                shutil.rmtree(item)
                        except Exception as e:
                            logger.warning(f"无法删除 {item}: {e}")

            print(f"\n{'='*50}")
            print(f"试验 {trial} 结果:")
            print(f"  F1: {metrics['f1']:.4f}")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  使用GPU: {metrics['used_gpu']}")
            print(f"{'='*50}\n")

            return metrics

        except Exception as e:
            logger.exception(f"Unhandled error running trial {trial}: {e}")
            traceback.print_exc()
            if not (trial_dir / "test_results.txt").exists():
                with open(trial_dir / "test_results.txt", "w") as f_err_res:
                    f_err_res.write("f1 = 0.0\nprecision = 0.0\nrecall = 0.0\n")
            return {
                "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "status": "error", "config": config, "trial": trial,
                "error_message": str(e)
            }

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
                    match = re.search(
                        r"(eval_precision|eval_recall|eval_f1|precision|recall|f1)\s*=\s*([0-9.]+)",
                        line, re.IGNORECASE
                    )
                    if match:
                        metric_name = match.group(1).lower().replace("eval_", "")
                        metric_value = float(match.group(2))
                        if metric_name in metrics:
                            metrics[metric_name] = metric_value
        except Exception as e:
            logger.error(f"Error parsing result file {result_file}: {e}")
        return metrics

    # ------------------------------------------------------------------
    # GRPO 核心：组内相对优势计算与策略更新
    # ------------------------------------------------------------------
    def _compute_group_advantages(self, rewards: List[float]) -> torch.Tensor:
        """计算组内相对优势：A_i = (r_i - mean(r)) / (std(r) + 1e-8)

        这是 GRPO 的核心创新 —— 不需要价值网络即可获得优势估计。
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_r) / std_r
        return advantages

    def _update_grpo_network(self):
        """使用组内相对优势进行 PPO 裁剪策略更新。

        与 PPO 的关键区别：
          - 不使用 value loss (因为没有价值网络)
          - 优势来自组内归一化而非 (R - V(s))
        """
        if len(self.memory_rewards) < self.grpo_group_size:
            return

        logger.info(f"Updating GRPO network with {len(self.memory_rewards)} samples...")

        states = torch.stack(self.memory_states)
        actions = torch.stack(self.memory_actions)
        old_log_probs = torch.stack(self.memory_log_probs)

        # GRPO 核心：组内相对优势计算
        advantages = self._compute_group_advantages(self.memory_rewards)
        # 维度对齐 (用于逐元素运算)
        advantages = advantages.unsqueeze(1)

        # 多 epoch 策略更新
        for _ in range(self.grpo_epochs):
            indices = torch.randperm(states.size(0))
            for start_idx in range(0, states.size(0), self.grpo_batch_size):
                end_idx = start_idx + self.grpo_batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 评估当前策略下的 log_prob 和熵
                new_log_probs, entropy = self.grpo_network.evaluate(
                    batch_states, batch_actions
                )

                # PPO 裁剪目标
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * batch_advantages

                # 策略损失 (不含价值损失 —— 这是 GRPO 的关键)
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.grpo_network.parameters(), 0.5
                )
                self.optimizer.step()

        # 清空记忆
        self.memory_states.clear()
        self.memory_actions.clear()
        self.memory_log_probs.clear()
        self.memory_rewards.clear()

        logger.info("GRPO network update complete.")

    # ------------------------------------------------------------------
    # 配置采样
    # ------------------------------------------------------------------
    def _get_config_from_grpo(self) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        """从当前 GRPO 策略采样一个超参数配置。"""
        state_tensor = self.current_state_for_grpo.unsqueeze(0)

        with torch.no_grad():
            self.grpo_network.eval()
            normalized_action_tensor, log_prob = self.grpo_network.get_action(state_tensor)
            self.grpo_network.train()

        # 将 [-1, 1] 映射到 [0, 1] 用于反归一化
        action_for_denorm = (normalized_action_tensor.squeeze(0) + 1.0) / 2.0
        action_for_denorm = torch.clamp(action_for_denorm, 0.0, 1.0)

        config = self.hyperparameter_space.denormalize_vector(action_for_denorm)
        return config, normalized_action_tensor.squeeze(0), log_prob

    # ------------------------------------------------------------------
    # 主优化循环
    # ------------------------------------------------------------------
    def optimize(self) -> Dict[str, Any]:
        """运行完整的 GRPO 超参数优化流程。"""
        logger.info(f"Starting GRPO optimization with {self.n_trials} trials, "
                    f"group size = {self.grpo_group_size}")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("GRPO NER Hyperparameter Optimization\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total trials: {self.n_trials}\n")
            f.write(f"Group size: {self.grpo_group_size}\n")
            f.write(f"Model: {self.model_type} ({self.model_name})\n")
            f.write("-" * 50 + "\n\n")

        # 加载已有结果 (支持断点续跑)
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
                        if result["f1"] > self.best_f1_for_early_stopping:
                            self.best_f1_for_early_stopping = result["f1"]
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not parse existing results file: {self.results_file}. "
                    "Starting fresh."
                )
                self.results = []

        start_trial = len(self.results)
        if start_trial > 0 and self.results[-1].get("status") == "success":
            last_successful_config = self.results[-1]["config"]
            self.current_state_for_grpo = self.hyperparameter_space.normalize_config(
                last_successful_config
            )
        else:
            self.current_state_for_grpo = self.hyperparameter_space.normalize_config(
                self.hyperparameter_space.default_config()
            )

        # 主循环
        trial = start_trial
        while trial < self.n_trials:
            logger.info(f"Starting trial {trial + 1}/{self.n_trials}")

            config_to_run: Dict[str, Any]
            action_tensor: Optional[torch.Tensor] = None
            log_prob_tensor: Optional[torch.Tensor] = None

            if trial < self.random_trials_threshold:
                # 初始随机探索，每组仍采样 grpo_group_size 个
                logger.info(
                    f"Trial {trial + 1}: Random sampling for initial exploration."
                )
                config_to_run = self.hyperparameter_space.sample_random()
            else:
                logger.info(f"Trial {trial + 1}: GRPO policy sampling.")
                config_to_run, action_tensor, log_prob_tensor = self._get_config_from_grpo()

            # 去重 + 微扰
            config_fingerprint = self.hyperparameter_space.get_fingerprint(config_to_run)
            if config_fingerprint in self.hyperparameter_space.explored_configs:
                logger.info(
                    f"Trial {trial + 1}: Config already explored, perturbing."
                )
                for name_pert in config_to_run:
                    param_type = self.hyperparameter_space.params[name_pert]["type"]
                    if param_type in ["float", "log_float"]:
                        perturbation = 1.0 + np.random.normal(0, 0.01)
                        if param_type == "log_float":
                            config_to_run[name_pert] = np.clip(
                                config_to_run[name_pert] * perturbation,
                                self.hyperparameter_space.params[name_pert]["min"],
                                self.hyperparameter_space.params[name_pert]["max"]
                            )
                        else:
                            current_val = config_to_run[name_pert]
                            rng = (self.hyperparameter_space.params[name_pert]["max"]
                                   - self.hyperparameter_space.params[name_pert]["min"])
                            noise = np.random.normal(0, 0.01) * rng
                            config_to_run[name_pert] = np.clip(
                                current_val + noise,
                                self.hyperparameter_space.params[name_pert]["min"],
                                self.hyperparameter_space.params[name_pert]["max"]
                            )
                    elif param_type == "int":
                        noise = random.choice([-2, -1, 0, 1, 2])
                        config_to_run[name_pert] = np.clip(
                            config_to_run[name_pert] + noise,
                            self.hyperparameter_space.params[name_pert]["min"],
                            self.hyperparameter_space.params[name_pert]["max"]
                        )
                config_fingerprint = self.hyperparameter_space.get_fingerprint(config_to_run)

            self.hyperparameter_space.explored_configs[config_fingerprint] = True

            logger.info(f"Trial {trial + 1}: Using config: {config_to_run}")
            result = self.run_trial(config_to_run, trial + 1)
            self.results.append(result)

            # 持久化结果
            with open(self.results_file, "w", encoding="utf-8") as f_res:
                json.dump(self.results, f_res, indent=2)

            if result["status"] == "success":
                reward = result["f1"]
                # 更新当前状态
                self.current_state_for_grpo = self.hyperparameter_space.normalize_config(
                    config_to_run
                )

                # 存储经验：仅当来自 GRPO 策略时
                if trial >= self.random_trials_threshold \
                   and action_tensor is not None \
                   and log_prob_tensor is not None:
                    self.memory_states.append(self.current_state_for_grpo.clone())
                    self.memory_actions.append(action_tensor.clone())
                    self.memory_log_probs.append(log_prob_tensor.clone())
                    self.memory_rewards.append(reward)

                # 记录日志
                with open(self.log_file, "a", encoding="utf-8") as f_log_main:
                    f_log_main.write(f"\nTrial {trial + 1} Results:\n")
                    f_log_main.write(f"  F1: {result['f1']:.4f}\n")
                    f_log_main.write(f"  Precision: {result['precision']:.4f}\n")
                    f_log_main.write(f"  Recall: {result['recall']:.4f}\n")
                    if self.best_trial == (trial + 1):
                        f_log_main.write("  ** New Best Configuration **\n")

                # 早停检查
                if self.early_stopping:
                    if result["f1"] > self.best_f1_for_early_stopping + self.min_delta:
                        self.best_f1_for_early_stopping = result["f1"]
                        self.no_improvement_count = 0
                        logger.info(
                            f"Trial {trial + 1}: F1 improved to {result['f1']:.4f} "
                            "(for early stopping). Resetting patience."
                        )
                    else:
                        self.no_improvement_count += 1
                        logger.info(
                            f"Trial {trial + 1}: No significant F1 improvement for "
                            f"early stopping. Count: {self.no_improvement_count}/"
                            f"{self.patience}"
                        )
                    if self.no_improvement_count >= self.patience:
                        logger.info(
                            f"Early stopping triggered after trial {trial + 1}. "
                            f"No F1 improvement for {self.patience} trials."
                        )
                        print(f"\n{'='*80}")
                        print(f"提前停止优化: {self.patience} 次连续试验未见显著改善")
                        print(f"当前最佳F1: {self.best_f1:.4f} (试验 {self.best_trial})")
                        print(f"{'='*80}\n")
                        break
            else:
                logger.warning(
                    f"Trial {trial+1} failed, skipping GRPO memory update."
                )

            # GRPO 关键：每积累满一组 (grpo_group_size) 个经验就更新策略
            if (len(self.memory_rewards) >= self.grpo_group_size
                    and trial >= self.random_trials_threshold):
                self._update_grpo_network()

            trial += 1
            self._plot_results()

        # 最终更新：处理剩余经验
        if len(self.memory_rewards) >= 2 and trial >= self.random_trials_threshold:
            logger.info("Final GRPO network update with remaining samples...")
            self._update_grpo_network()

        # 写入最终日志
        with open(self.log_file, "a", encoding="utf-8") as f_log_final:
            f_log_final.write("\n" + "=" * 50 + "\n")
            f_log_final.write(
                f"Optimization Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
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

    # ------------------------------------------------------------------
    # 可视化 (与 PPO 版本一致)
    # ------------------------------------------------------------------
    def _plot_results(self):
        if not self.results:
            return
        valid_results = [
            r for r in self.results
            if r.get("status") == "success" and "f1" in r and "trial" in r and "config" in r
        ]
        if not valid_results:
            return

        trials = [r["trial"] for r in valid_results]
        f1_scores = [r["f1"] for r in valid_results]
        configs = [r["config"] for r in valid_results]

        plt.figure(figsize=(12, 7))
        plt.plot(trials, f1_scores, 'o-', color='dodgerblue', label='F1 Score per Trial')
        if self.best_f1 > 0:
            plt.axhline(
                y=self.best_f1, color='red', linestyle='--',
                label=f'Best F1: {self.best_f1:.4f} (Trial {self.best_trial})'
            )
        plt.xlabel('Trial Number')
        plt.ylabel('F1 Score')
        plt.title('GRPO NER Model F1 Score Progression')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / "optimization_progress.png")
        plt.close()

        param_names = list(self.hyperparameter_space.params.keys())
        for param_name in param_names:
            plt.figure(figsize=(12, 7))
            param_values = [cfg[param_name] for cfg in configs]
            sc = plt.scatter(
                param_values, f1_scores, c=trials, cmap='viridis',
                s=100, alpha=0.8, edgecolors='k', linewidth=0.5
            )
            cbar = plt.colorbar(sc)
            cbar.set_label('Trial Number')
            if self.best_config and self.best_trial in trials:
                best_idx = trials.index(self.best_trial)
                plt.scatter(
                    [param_values[best_idx]], [f1_scores[best_idx]],
                    s=250, c='red', marker='*',
                    label=f'Best (Trial {self.best_trial})',
                    edgecolors='black', linewidth=1
                )
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
        if not self.best_config:
            return
        param_names = list(self.hyperparameter_space.params.keys())
        best_values_normalized = []
        for name in param_names:
            param_info = self.hyperparameter_space.params[name]
            value = self.best_config[name]
            if param_info["type"] == "log_float":
                norm_value = (
                    (np.log(value) - np.log(param_info["min"]))
                    / (np.log(param_info["max"]) - np.log(param_info["min"]))
                )
            else:
                norm_value = (
                    (value - param_info["min"])
                    / (param_info["max"] - param_info["min"])
                )
            best_values_normalized.append(norm_value)

        angles = np.linspace(0, 2 * np.pi, len(param_names), endpoint=False).tolist()
        best_values_normalized_closed = best_values_normalized + [best_values_normalized[0]]
        angles_closed = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(
            angles_closed, best_values_normalized_closed,
            'o-', linewidth=2, color='deepskyblue'
        )
        ax.fill(
            angles_closed, best_values_normalized_closed,
            'deepskyblue', alpha=0.25
        )
        ax.set_thetagrids(
            np.degrees(angles),
            [name.replace("_", " ").title() for name in param_names],
            fontsize=10
        )
        ax.set_ylim(0, 1)
        ax.set_title(
            f'Best Configuration (F1: {self.best_f1:.4f}, Trial: {self.best_trial})',
            size=15, y=1.1
        )
        plt.tight_layout()
        plt.savefig(self.viz_dir / "best_configuration_radar.png")
        plt.close()

        with open(self.viz_dir / "best_config_details.txt", "w", encoding="utf-8") as f:
            f.write(
                f"Best Configuration (Trial {self.best_trial}, "
                f"F1: {self.best_f1:.4f})\n"
            )
            f.write("-" * 50 + "\n")
            for name, value in self.best_config.items():
                f.write(f"{name}: {value}\n")


# ============================================================================
# 命令行入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Optimize NER model hyperparameters using GRPO "
                    "(Group Relative Policy Optimization)"
    )
    # 通用参数 (与 PPO/GP-TS 对齐)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory for optimization results")
    parser.add_argument("--model_type", type=str, default="roberta",
                        help="Model type (e.g., bert, roberta)")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="Pretrained model name or path")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of optimization trials")
    parser.add_argument("--random_trials", type=int, default=5,
                        help="Number of initial random trials before GRPO policy takes over")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Number of epochs for NER model training per trial")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length for NER model")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for caching pretrained models and datasets")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--run_ner_path", type=str, required=True,
                        help="Full path to the run_ner.py script")
    parser.add_argument("--reset", action="store_true",
                        help="Reset optimization and delete previous results")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose output during NER training subprocess")
    parser.add_argument("--skip_model_saving", action="store_true",
                        help="Skip saving NER model files during trials to save disk space")

    # GRPO 特定参数
    parser.add_argument("--grpo_group_size", type=int, default=8,
                        help="G: number of configs sampled per GRPO group iteration")
    parser.add_argument("--grpo_epochs", type=int, default=10,
                        help="Number of epochs for each GRPO policy update")
    parser.add_argument("--grpo_batch_size", type=int, default=4,
                        help="Mini-batch size for GRPO policy update")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                        help="Clipping epsilon for PPO-style update")
    parser.add_argument("--entropy_coeff", type=float, default=0.01,
                        help="Entropy coefficient in GRPO loss")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="Learning rate for GRPO policy network")

    # 早停参数
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping for the GRPO optimization process")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="Minimum change in F1 to be considered an improvement")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(args.output_dir) / "grpo_script_main.log")
        ]
    )
    global logger
    logger = logging.getLogger(__name__)

    if not os.path.exists(args.run_ner_path):
        logger.error(f"Error: run_ner.py not found at {args.run_ner_path}")
        sys.exit(1)

    output_path = Path(args.output_dir)
    if args.reset:
        logger.info("重置优化过程，删除以前的结果...")
        for pattern in ["optimization_results.json", "optimization_progress.png",
                        "final_results.json"]:
            p = output_path / pattern
            if p.exists():
                os.remove(p)
        for item in output_path.iterdir():
            if item.is_dir() and (
                item.name.startswith("trial_")
                or item.name == "visualizations"
                or item.name == "final_model"
            ):
                logger.info(f"删除之前的目录: {item}")
                shutil.rmtree(item)
        (output_path / "visualizations").mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.data_dir):
        logger.warning(f"数据目录不存在: {args.data_dir}")
        script_dir = Path(sys.argv[0]).resolve().parent
        possible_paths = [
            script_dir.parent / "BOND" / "dataset" / Path(args.data_dir).name,
            Path(os.path.dirname(args.run_ner_path)).parent / "dataset"
            / Path(args.data_dir).name,
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

    hyperparameter_space = HyperparameterSpace()
    optimizer = GRPOOptimizer(
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
        grpo_group_size=args.grpo_group_size,
        grpo_epochs=args.grpo_epochs,
        grpo_batch_size=args.grpo_batch_size,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        lr=args.lr,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta
    )

    best_result = optimizer.optimize()

    logger.info("Optimization completed!")
    logger.info(f"Best F1: {best_result['best_f1']:.4f} (Trial {best_result['best_trial']})")
    if best_result['best_config']:
        logger.info(f"Best hyperparameters: {best_result['best_config']}")
    else:
        logger.info("No valid best configuration found")

    # 使用最佳配置训练最终模型 (与 PPO 一致)
    if best_result['best_config'] is not None and not args.skip_model_saving:
        logger.info("Training final model with best hyperparameters...")
        print(f"\n{'='*80}")
        print("使用最佳超参数训练最终模型")
        print(
            f"最佳配置 (来自试验 {best_result['best_trial']}, "
            f"F1: {best_result['best_f1']:.4f}):"
        )
        for name, value in best_result['best_config'].items():
            print(f"  {name}: {value}")
        print(f"{'='*80}\n")

        final_output_dir = Path(args.output_dir) / "final_model"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_training_log_path = final_output_dir / "final_training_log.txt"

        base_args_final = [
            "--data_dir", str(args.data_dir),
            "--model_type", args.model_type,
            "--model_name_or_path", args.model_name,
            "--output_dir", str(final_output_dir),
            "--num_train_epochs", str(args.n_epochs * 2),
            "--max_seq_length", str(args.max_seq_length),
            "--do_train", "--do_eval", "--do_predict",
            "--evaluate_during_training",
            "--overwrite_output_dir",
            "--seed", str(args.seed + best_result['best_trial']),
            "--save_steps", "500",
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
            process_final = subprocess.Popen(
                cmd_final, env=env_final,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )
            for line in process_final.stdout:
                line_strip = line.strip()
                if args.verbose or any(
                    key in line_strip
                    for key in ["Epoch", "eval", "F1", "precision",
                                "recall", "Loss", "Saving model"]
                ):
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
                process_final_cpu = subprocess.Popen(
                    cmd_final_cpu, env=env_final,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, bufsize=1
                )
                for line in process_final_cpu.stdout:
                    line_strip = line.strip()
                    if args.verbose or any(
                        key in line_strip
                        for key in ["Epoch", "eval", "F1", "precision",
                                    "recall", "Loss", "Saving model"]
                    ):
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
            logger.info(f"Final model training completed! Model saved to {final_output_dir}")
        else:
            logger.error(f"Error training final model: return code {final_returncode}")
            print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
