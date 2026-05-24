#!/usr/bin/env python
# coding: utf-8
"""
基于DQN的NER模型超参数优化（骨架实现）
"""

import os, sys, argparse, json, random, subprocess, time
import torch, numpy as np
from pathlib import Path
import shutil  # 新增导入
from ppo_ner_optimizer import HyperparameterSpace  # 复用已有定义

# 简单的DQN网络
class DQNNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# DQN Agent（极简版）
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.9):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = []  # 简化为列表
        self.batch_size = 32
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.net.net[-1].out_features)
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.net(state)
        return q.argmax().item()
    def store(self, *transition):
        self.memory.append(transition)
        if len(self.memory) > 10000:
            self.memory.pop(0)
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(1)
        q_values = self.net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
        loss = torch.nn.functional.mse_loss(q_values, rewards + self.gamma * q_next)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
    def sync_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

class DQNOptimizer:
    def __init__(self, space: HyperparameterSpace, args):
        self.space = space
        self.args = args
        self.agent = DQNAgent(state_dim=space.dim, action_dim=space.dim)
        self.best_config = None
        self.best_f1 = 0.0
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 存储 cache_dir 和 max_seq_length
        self.cache_dir = Path(args.cache_dir) if args.cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_seq_length = args.max_seq_length
        # 存储 skip_model_saving
        self.skip_model_saving = args.skip_model_saving

    def run_trial(self, config, trial):
        # 创建本次试验目录
        trial_dir = self.output_dir / f"trial_{trial}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        # 构建命令行
        cmd = [
            sys.executable, self.args.run_ner_path,
            "--data_dir", self.args.data_dir,
            "--model_type", self.args.model_type,
            "--model_name_or_path", self.args.model_name_or_path,
            "--output_dir", str(trial_dir),
            "--do_train", "--do_eval", "--do_predict",
            "--evaluate_during_training",
            "--overwrite_output_dir",
            "--num_train_epochs", str(self.args.n_epochs_ner),
            "--max_seq_length", str(self.max_seq_length),
            "--seed", str(self.args.seed + trial)
        ]
        if self.cache_dir:
            cmd.extend(["--cache_dir", str(self.cache_dir)])

        # 添加超参数
        cmd += self.space.config_to_args(config)
        print(f"[DQN] Trial {trial} run: {' '.join(cmd)}")
        
        # 运行并等待结束，捕获输出
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = proc.communicate()

        # 将子进程的输出写入日志文件
        trial_log_file = trial_dir / "trial_run.log"
        with open(trial_log_file, "w", encoding='utf-8') as log_f:
            log_f.write("COMMAND:\n")
            log_f.write(' '.join(cmd) + "\n\n")
            log_f.write("STDOUT:\n")
            log_f.write(stdout)
            log_f.write("\nSTDERR:\n")
            log_f.write(stderr)

        # 如果启用了 skip_model_saving，则删除模型文件
        if self.skip_model_saving:
            print(f"[DQN] Trial {trial}: Cleaning up model files from {trial_dir}")
            for item in trial_dir.iterdir():
                if item.is_file() and item.suffix in [".bin", ".pt", ".ckpt"]:
                    try:
                        item.unlink()
                        print(f"[DQN] Deleted file: {item}")
                    except Exception as e:
                        print(f"[DQN] Error deleting file {item}: {e}")
                elif item.is_dir() and item.name.startswith("checkpoint-"):
                    try:
                        shutil.rmtree(item)
                        print(f"[DQN] Deleted directory: {item}")
                    except Exception as e:
                        print(f"[DQN] Error deleting directory {item}: {e}")
        
        # 从 test_results.txt 提取指标
        f1 = p = r = 0.0
        result_file = trial_dir / "test_results.txt"

        if proc.returncode != 0:
            print(f"[DQN] Warning: Trial {trial} script '{self.args.run_ner_path}' failed with return code {proc.returncode}.")
            print(f"[DQN] Check logs for details: {trial_log_file}")
        
        if result_file.exists():
            try:
                for line in result_file.read_text().splitlines():
                    if line.startswith("precision"):
                        p = float(line.split("=")[1].strip())
                    elif line.startswith("recall"):
                        r = float(line.split("=")[1].strip())
                    elif line.startswith("f1"):
                        f1 = float(line.split("=")[1].strip())
            except Exception as e:
                print(f"[DQN] Error parsing {result_file} for trial {trial}: {e}")
                print(f"[DQN] Check logs for details: {trial_log_file}")
        else:
            # 即使返回码为0，文件也可能因为其他原因未生成
            print(f"[DQN] Warning: {result_file} not found for trial {trial}.")
            if proc.returncode == 0:
                print(f"[DQN] Note: '{self.args.run_ner_path}' completed with return code 0, but results file is missing.")
            print(f"[DQN] Check logs for details: {trial_log_file}")
            
        return {"f1": f1, "precision": p, "recall": r, "trial_dir": str(trial_dir)}

    def optimize(self):
        epsilon = 1.0
        for trial in range(self.args.n_trials):
            # 当前状态：历史最佳配置向量或 zero
            state = self.space.normalize_config(self.best_config or self.space.default_config()).numpy()
            # Agent 选择更新哪个超参数维度，以及增/减方向
            action = self.agent.select_action(state, epsilon)
            # 对应调参：在 state 上对 action 维度做微调，生成新config
            config = self.space.denormalize_vector(torch.tensor(state)).copy()
            # 例如简单 ±5% 调整
            name = list(self.space.params.keys())[action]
            val = config[name]
            config[name] = max(self.space.params[name]["min"],
                               min(self.space.params[name]["max"], val * (1 + (random.choice([1,-1])*0.05))))
            # 运行试验
            result = self.run_trial(config, trial)
            reward = result["f1"]
            next_state = self.space.normalize_config(config).numpy()
            self.agent.store(state, action, reward, next_state)
            self.agent.update()
            if trial % 10 == 0:
                self.agent.sync_target()
            # 更新最���
            if reward > self.best_f1:
                self.best_f1, self.best_config = reward, config
            # ε 衰减
            epsilon = max(0.1, epsilon * 0.99)
        return {"best_config": self.best_config, "best_f1": self.best_f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--run_ner_path", required=True)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--model_type", type=str, default="roberta", help="NER模型类型")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="预训练模型路径或名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output_dir", type=str, default="./dqn_trials", help="保存试验目录")
    parser.add_argument("--n_epochs_ner", type=int, default=20, help="NER训练轮数")
    # 新增参数
    parser.add_argument("--max_seq_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--cache_dir", type=str, default=None, help="预训练模型缓存目录")
    parser.add_argument("--skip_model_saving", action="store_true", help="试验后是否删除模型文件")

    args = parser.parse_args()
    print(f"Arguments: {args}")
    space = HyperparameterSpace()
    optimizer = DQNOptimizer(space, args)
    print("Starting DQN optimization...")

    best = optimizer.optimize()
    print(f"Best F1: {best['best_f1']}, config: {best['best_config']}")
    output_path = Path(args.output_dir)
    with open(output_path / "final_results.json", "w") as f:
        json.dump(best, f, indent=2)
    with open(output_path / "optimization_results.json", "w") as f:
        json.dump([{"f1": best["best_f1"], "config": best["best_config"], "status": "success", "trial": 0}], f, indent=2)

if __name__ == "__main__":
    main()
