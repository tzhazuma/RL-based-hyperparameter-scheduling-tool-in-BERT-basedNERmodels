#!/usr/bin/env python
# coding: utf-8
"""
基于Double DQN的NER模型超参数优化
"""

import os, sys, argparse, json, random, subprocess, time
import torch, numpy as np
from pathlib import Path
from gp_ts_ner_optimizer import HyperparameterSpace  # 复用已有定义

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

class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.9):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DoubleDQNAgent] Initializing on device: {self.device}")
        self.net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = []
        self.batch_size = 32
        print(f"[DoubleDQNAgent] State dim: {state_dim}, Action dim: {action_dim}, LR: {lr}, Gamma: {gamma}, Batch size: {self.batch_size}")

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.net.net[-1].out_features)
            print(f"[DoubleDQNAgent] Epsilon greedy action (epsilon={epsilon:.4f}): {action}")
            return action
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.net(state)
        action = q.argmax().item()
        print(f"[DoubleDQNAgent] Q-value action (epsilon={epsilon:.4f}): {action}, Q-values: {q.cpu().numpy()}")
        return action

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
        # Double DQN: action 由 online 选，Q值由 target 取
        with torch.no_grad():
            next_actions = self.net(next_states).argmax(1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions)
        loss = torch.nn.functional.mse_loss(q_values, rewards + self.gamma * q_next)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        print(f"[DoubleDQNAgent] Updated. Loss: {loss.item():.4f}")

    def sync_target(self):
        self.target_net.load_state_dict(self.net.state_dict())
        print("[DoubleDQNAgent] Synced target network.")

class DoubleDQNOptimizer:
    def __init__(self, space: HyperparameterSpace, args):
        self.space = space
        self.args = args
        print(f"[DoubleDQNOptimizer] Initializing with args: {args}")
        self.agent = DoubleDQNAgent(state_dim=space.dim, action_dim=space.dim)
        self.best_config = None
        self.best_f1 = 0.0
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DoubleDQNOptimizer] Output directory: {self.output_dir}")

    def run_trial(self, config, trial):
        trial_dir = self.output_dir / f"trial_{trial}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, self.args.run_ner_path,
            "--data_dir", self.args.data_dir,
            "--model_type", self.args.model_type,
            "--model_name_or_path", self.args.model_name_or_path,
            "--output_dir", str(trial_dir),
            "--do_train", "--do_eval", "--do_predict",
            "--evaluate_during_training",
            "--overwrite_output_dir",
            "--save_strategy", "no",
            "--save_total_limit", "1",
            "--seed", str(self.args.seed)
        ]
        cmd += self.space.config_to_args(config)
        print(f"[DoubleDQNOptimizer] Trial {trial} - Config: {config}")
        print(f"[DoubleDQNOptimizer] Trial {trial} - Running command: {' '.join(cmd)}")
        
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as e:
            print(f"[DoubleDQNOptimizer] Trial {trial} - Exception during subprocess run: {e}")
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        
        # 将 subprocess 输出写入 test_results.txt，确保输出落地
        result_file = trial_dir / "test_results.txt"
        result_file.write_text(proc.stdout)
        print(f"[DoubleDQNOptimizer] Trial {trial} - Written subprocess stdout to {result_file}")
        
        if proc.returncode != 0:
            print(f"[DoubleDQNOptimizer] Trial {trial} - Subprocess Error (return code {proc.returncode}):")
            print(f"Stdout:\n{proc.stdout}")
            print(f"Stderr:\n{proc.stderr}")
        
        f1 = p = r = 0.0
        if result_file.exists():
            print(f"[DoubleDQNOptimizer] Trial {trial} - Reading results from {result_file}")
            for line in result_file.read_text().splitlines():
                if line.startswith("precision"):
                    try:
                        p = float(line.split("=")[1])
                    except ValueError:
                        print(f"[DoubleDQNOptimizer] Trial {trial} - Warning: Could not parse precision from line: {line}")
                elif line.startswith("recall"):
                    try:
                        r = float(line.split("=")[1])
                    except ValueError:
                        print(f"[DoubleDQNOptimizer] Trial {trial} - Warning: Could not parse recall from line: {line}")
                elif line.startswith("f1"):
                    try:
                        f1 = float(line.split("=")[1])
                    except ValueError:
                        print(f"[DoubleDQNOptimizer] Trial {trial} - Warning: Could not parse f1 from line: {line}")
            print(f"[DoubleDQNOptimizer] Trial {trial} - Results: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
        else:
            print(f"[DoubleDQNOptimizer] Trial {trial} - Warning: {result_file} not found.")

        return {"f1": f1, "precision": p, "recall": r}

    def optimize(self):
        epsilon = 1.0
        print(f"[DoubleDQNOptimizer] Starting optimization for {self.args.n_trials} trials. Initial epsilon: {epsilon:.4f}")
        for trial in range(self.args.n_trials):
            state = self.space.normalize_config(self.best_config or self.space.default_config()).numpy()
            action = self.agent.select_action(state, epsilon)
            config = self.space.denormalize_vector(torch.tensor(state)).copy()
            name = list(self.space.params.keys())[action]
            val = config[name]
            config[name] = max(self.space.params[name]["min"],
                               min(self.space.params[name]["max"], val * (1 + (random.choice([1,-1])*0.05))))
            result = self.run_trial(config, trial)
            reward = result["f1"]
            next_state = self.space.normalize_config(config).numpy()
            self.agent.store(state, action, reward, next_state)
            self.agent.update()
            if trial % 10 == 0:
                self.agent.sync_target()
            if reward > self.best_f1:
                print(f"[DoubleDQNOptimizer] Trial {trial} - New best F1! Previous: {self.best_f1:.4f}, New: {reward:.4f}")
                self.best_f1, self.best_config = reward, config
            epsilon = max(0.1, epsilon * 0.99)
            print(f"[DoubleDQNOptimizer] Trial {trial} - Epsilon decayed to: {epsilon:.4f}")
        return {"best_config": self.best_config, "best_f1": self.best_f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--run_ner_path", required=True)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--model_type", type=str, default="roberta", help="NER模型类型")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="预训练模型路径或名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output_dir", type=str, default="./double_dqn_trials", help="保存试验目录")
    args = parser.parse_args()
    print("=================================================")
    print("Starting Double DQN Hyperparameter Optimization")
    print("=================================================")
    print(f"Arguments: {json.dumps(vars(args), indent=2)}")
    space = HyperparameterSpace()
    print(f"Hyperparameter space dimension: {space.dim}")
    print(f"Hyperparameter space details: {json.dumps(space.params, indent=2)}")
    optimizer = DoubleDQNOptimizer(space, args)
    start_time = time.time()
    best = optimizer.optimize()
    end_time = time.time()
    print("\n=================================================")
    print("Double DQN Optimization Complete")
    print("=================================================")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Best F1 achieved: {best.get('best_f1', 'N/A'):.4f}")
    print(f"Best configuration: {json.dumps(best.get('best_config', {}), indent=2)}")
    with open("double_dqn_optimization_result.json", "w") as f:
        json.dump(best, f, indent=2)
    print("Optimization results saved to: double_dqn_optimization_result.json")

if __name__ == "__main__":
    main()
