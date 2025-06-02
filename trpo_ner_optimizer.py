#!/usr/bin/env python
# coding: utf-8
"""
基于TRPO（Trust Region Policy Optimization）的NER模型超参数优化
框架与 PPO、GP-TS 版本保持一致，只替换为 TRPO 更新逻辑。
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
from torch.distributions import Normal
import logging
import shutil
from pathlib import Path
import re
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
# 引入共享的超参数空间定义
from ppo_ner_optimizer import HyperparameterSpace

logger = logging.getLogger(__name__)

class TRPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TRPONetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.policy_mean = nn.Linear(64, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, state):
        features = self.shared_layers(state)
        mean = torch.tanh(self.policy_mean(features))
        log_std = self.policy_log_std.expand_as(mean)
        std = torch.exp(log_std)
        value = self.value_head(features)
        return mean, std, value
    def get_action(self, state, deterministic=False):
        mean, std, _ = self.forward(state)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)
        return action, logp
    def evaluate(self, state, action):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        logp = dist.log_prob(action).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return logp, value, ent

class TRPOOptimizer:
    def __init__(self, hyperparameter_space: HyperparameterSpace, output_dir: str,
                 n_trials: int=20, random_trials: int=5, seed: int=42,
                 data_dir: str=None, model_type: str="roberta", model_name: str="roberta-base",
                 n_epochs: int=5, max_seq_length: int=128, cache_dir: str=None,
                 run_ner_path: str=None, skip_model_saving: bool=False,
                 trpo_iters: int=10, max_kl: float=0.01, cg_iters: int=10,
                 damping: float=0.1, verbose: bool=True):
        self.hyperparameter_space = hyperparameter_space
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials, self.random_trials = n_trials, random_trials
        self.seed, self.data_dir = seed, data_dir
        self.model_type, self.model_name = model_type, model_name
        self.n_epochs_ner, self.max_seq_length = n_epochs, max_seq_length
        self.cache_dir, self.run_ner_path = cache_dir, run_ner_path
        self.skip_model_saving, self.verbose = skip_model_saving, verbose
        self.trpo_iters, self.max_kl = trpo_iters, max_kl
        self.cg_iters, self.damping = cg_iters, damping

        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.state_dim = hyperparameter_space.dim; self.action_dim = hyperparameter_space.dim
        self.trpo_net = TRPONetwork(self.state_dim, self.action_dim)
        # 存储经验
        self.memory_states, self.memory_actions = [], []
        self.memory_log_probs, self.memory_rewards = [], []
        # 结果跟踪
        self.best_f1, self.best_config, self.best_trial = 0.0, None, -1
        self.results = []
        self.results_file = self.output_dir/"optimization_results.json"
        self.log_file = self.output_dir/"trpo_optimization_log.txt"
        self.viz_dir = self.output_dir/"visualizations"; self.viz_dir.mkdir(exist_ok=True)
        # 初始 state
        self.current_state = hyperparameter_space.normalize_config(hyperparameter_space.default_config())

    def run_trial(self, config: Dict[str, Any], trial: int) -> Dict[str, Any]:
        trial_dir = self.output_dir/f"trial_{trial}"; trial_dir.mkdir(exist_ok=True)
        self._log_config(config, trial)
        base = [sys.executable, self.run_ner_path,
                "--data_dir", self.data_dir,
                "--model_type", self.model_type,
                "--model_name_or_path", self.model_name,
                "--output_dir", str(trial_dir),
                "--num_train_epochs", str(self.n_epochs_ner),
                "--max_seq_length", str(self.max_seq_length),
                "--do_train","--do_eval","--do_predict",
                "--evaluate_during_training","--overwrite_output_dir",
                "--seed", str(self.seed+trial)]
        if self.cache_dir: base += ["--cache_dir", self.cache_dir]
        base += self.hyperparameter_space.config_to_args(config)
        logger.info(f"TRPO trial {trial} cmd: {' '.join(base)}")
        proc = subprocess.Popen(base, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate()
        returncode = proc.returncode
        # 不保存模型
        if self.skip_model_saving:
            for f in trial_dir.glob("*"):
                if f.is_dir() or f.suffix in [".bin",".pt"]: shutil.rmtree(f) if f.is_dir() else f.unlink()
        # 解析结果
        resf = trial_dir/"test_results.txt"
        if not resf.exists():
            metrics = {"f1":0.0,"precision":0.0,"recall":0.0}
        else:
            metrics = self._parse_result_file(resf)
        metrics.update(status="success", config=config, trial=trial)
        return metrics

    def _get_config(self, t: int):
        if t < self.random_trials:
            return self.hyperparameter_space.sample_random(), None, None
        state = self.current_state.unsqueeze(0)
        with torch.no_grad():
            action, logp = self.trpo_net.get_action(state)
        norm = ((action + 1) / 2).squeeze(0).clamp(0, 1)
        cfg = self.hyperparameter_space.denormalize_vector(norm)
        return cfg, action.squeeze(0), logp

    def optimize(self) -> Dict[str, Any]:
        if self.results_file.exists():
            with open(self.results_file) as f: self.results=json.load(f)
        start = len(self.results)
        if start>0 and self.results[-1]["status"]=="success":
            self.current_state = self.hyperparameter_space.normalize_config(self.results[-1]["config"])
            self.best_f1 = max(r["f1"] for r in self.results)
        for t in range(start, self.n_trials):
            cfg, act, logp = self._get_config(t)
            fp = self.hyperparameter_space.get_fingerprint(cfg)
            if fp in self.hyperparameter_space.explored_configs:
                # 可选扰动逻辑...
                # 对 current_state 加小噪声直到生成新配置
                noise_scale = 0.05
                for _ in range(10):
                    noise = torch.randn_like(self.current_state) * noise_scale
                    new_state = (self.current_state + noise).clamp(0, 1)
                    cfg = self.hyperparameter_space.denormalize_vector(new_state)
                    fp = self.hyperparameter_space.get_fingerprint(cfg)
                    if fp not in self.hyperparameter_space.explored_configs:
                        break
                else:
                    # 若多次失败，退回随机采样
                    cfg = self.hyperparameter_space.sample_random()
            fp = self.hyperparameter_space.get_fingerprint(cfg)
            self.hyperparameter_space.explored_configs[fp] = True

            #self.hyperparameter_space.explored_configs[fp]=True
            res = self.run_trial(cfg, t+1)
            self.results.append(res); json.dump(self.results, open(self.results_file,"w"), indent=2)
            if res["f1"]>self.best_f1:
                self.best_f1, self.best_config, self.best_trial = res["f1"], cfg, t+1
            # 存储经验
            if t>=self.random_trials:
                self.memory_states.append(self.current_state.clone())
                self.memory_actions.append(act.squeeze(0))
                self.memory_log_probs.append(logp.squeeze(0))
                self.memory_rewards.append(res["f1"])
            self.current_state = self.hyperparameter_space.normalize_config(cfg)
            # 更新
            if len(self.memory_rewards)>= self.random_trials:
                self._update_trpo_network()
            self._plot_results()
        # 最后一次更新
        if self.memory_rewards:
            self._update_trpo_network()
        self._visualize_best_config()
        # 保存 final_results.json
        with open(self.output_dir/"final_results.json","w") as f: json.dump(
            {"best_config":self.best_config,"best_f1":self.best_f1,"best_trial":self.best_trial}, f, indent=2)
        return {"best_config":self.best_config,"best_f1":self.best_f1,"best_trial":self.best_trial,"all_results":self.results}

    def _update_trpo_network(self):
        if not self.memory_rewards:
            return
        # 准备数据
        states = torch.stack(self.memory_states)  # [N, state_dim]
        actions = torch.stack(self.memory_actions)  # [N, action_dim]
        old_logp = torch.stack(self.memory_log_probs)  # [N]
        # 计算折扣回报
        discounts = [self.gamma ** i for i in range(len(self.memory_rewards))]
        returns = torch.tensor(self.memory_rewards) * torch.tensor(discounts)
        returns = torch.flip(torch.cumsum(torch.flip(returns, [0]), dim=0), [0])
        values = self.trpo_net(states)[2].squeeze(-1).detach()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 保留当前策略分布用于 KL 计算
        with torch.no_grad():
            old_mean, old_std, _ = self.trpo_net(states)
            old_dist = Normal(old_mean, old_std)

        # 扁平化参数操作
        def get_flat_params(model):
            return torch.cat([p.data.view(-1) for p in model.parameters()])

        def set_flat_params(model, flat):
            idx = 0
            for p in model.parameters():
                numel = p.numel()
                p.data.copy_(flat[idx:idx + numel].view_as(p))
                idx += numel

        # 计算策略目标与 KL
        def loss_and_kl():
            mean, std, _ = self.trpo_net(states)
            dist = Normal(mean, std)
            logp = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(logp - old_logp)
            surrogate = (ratio * advantages).mean()
            kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
            return surrogate, kl

        # 自然梯度方向 = F^{-1}g，通过共轭梯度求解
        def fisher_vector_product(v):
            _, kl = loss_and_kl()
            grads = torch.autograd.grad(kl, self.trpo_net.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([g.view(-1) for g in grads])
            kl_v = (flat_grad_kl * v).sum()
            grads2 = torch.autograd.grad(kl_v, self.trpo_net.parameters())
            flat_grad2 = torch.cat([g.contiguous().view(-1) for g in grads2]).data
            return flat_grad2 + self.damping * v

        def conjugate_gradients(Ax, b, iters=10, tol=1e-10):
            x = torch.zeros_like(b)
            r = b.clone()
            p = b.clone()
            rdotr = torch.dot(r, r)
            for _ in range(iters):
                Ap = Ax(p)
                alpha = rdotr / (p.dot(Ap) + 1e-8)
                x += alpha * p
                r -= alpha * Ap
                new_rdotr = torch.dot(r, r)
                if new_rdotr < tol:
                    break
                p = r + (new_rdotr / rdotr) * p
                rdotr = new_rdotr
            return x

        # 计算梯度 g
        surrogate, _ = loss_and_kl()
        grads = torch.autograd.grad(surrogate, self.trpo_net.parameters())
        g = torch.cat([g.view(-1) for g in grads]).data

        # 求自然梯度方向
        step_dir = conjugate_gradients(fisher_vector_product, g, self.cg_iters)
        # 规模控制
        shs = 0.5 * (step_dir * fisher_vector_product(step_dir)).sum()
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = step_dir / (lm + 1e-8)

        # 线性搜索
        old_params = get_flat_params(self.trpo_net)
        old_surrogate, _ = loss_and_kl()
        for frac in [0.5 ** i for i in range(10)]:
            new_params = old_params + frac * fullstep
            set_flat_params(self.trpo_net, new_params)
            new_surrogate, new_kl = loss_and_kl()
            if new_surrogate > old_surrogate and new_kl <= self.max_kl:
                break
        else:
            # 回退
            set_flat_params(self.trpo_net, old_params)

        # 用 MSE 更新 value_head
        returns = returns.detach()
        value_optimizer = torch.optim.Adam(self.trpo_net.value_head.parameters(), lr=1e-3)
        for _ in range(self.trpo_iters):
            _, _, vals = self.trpo_net(states)
            loss_v = nn.MSELoss()(vals.squeeze(-1), returns)
            value_optimizer.zero_grad()
            loss_v.backward()
            value_optimizer.step()

        # 清空经验
        self.memory_states.clear()
        self.memory_actions.clear()
        self.memory_log_probs.clear()
        self.memory_rewards.clear()
        logger.info("TRPO network updated.")

    def _log_config(self, config: Dict[str, Any], trial: int):
        s = f"[Trial {trial}] Config: {config}\n"
        logger.info(s)
        with open(self.log_file,"a",encoding="utf-8") as f: f.write(s)

    def _parse_result_file(self, fpath: Path) -> Dict[str, float]:
        m = {"f1":0.0,"precision":0.0,"recall":0.0}
        for line in open(fpath, encoding="utf-8"):
            match = re.search(r"(f1|precision|recall)\s*=\s*([0-9.]+)", line, re.I)
            if match: m[match.group(1).lower()]=float(match.group(2))
        return m

    def _plot_results(self):
        if not self.results: return
        valid = [r for r in self.results if r.get("status")=="success"]
        if not valid: return
        trials = [r["trial"] for r in valid]
        f1s = [r["f1"] for r in valid]
        plt.figure(figsize=(8,5))
        plt.plot(trials, f1s, 'o-', color='teal', label='F1')
        plt.axhline(self.best_f1, color='red', linestyle='--',
                    label=f'Best {self.best_f1:.4f} (t{self.best_trial})')
        plt.xlabel('Trial'); plt.ylabel('F1'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(self.viz_dir/"progress.png"); plt.close()
        for name in self.hyperparameter_space.params:
            vals = [r["config"][name] for r in valid]
            plt.figure(figsize=(6,4))
            sc = plt.scatter(vals, f1s, c=trials, cmap='viridis', s=40)
            plt.colorbar(sc, label='trial'); plt.xlabel(name); plt.ylabel('F1')
            if self.hyperparameter_space.params[name]["type"]=="log_float":
                plt.xscale('log')
            plt.tight_layout()
            plt.savefig(self.viz_dir/f"param_{name}.png"); plt.close()

    def _visualize_best_config(self):
        if not self.best_config: return
        names = list(self.hyperparameter_space.params.keys())
        vals = []
        for n in names:
            info = self.hyperparameter_space.params[n]
            v = self.best_config[n]
            if info["type"]=="log_float":
                norm = (np.log(v)-np.log(info["min"]))/(np.log(info["max"])-np.log(info["min"]))
            else:
                norm = (v-info["min"])/(info["max"]-info["min"])
            vals.append(norm)
        angles = np.linspace(0,2*np.pi,len(names),endpoint=False).tolist()
        v_closed = vals+[vals[0]]; a_closed = angles+[angles[0]]
        fig,ax = plt.subplots(figsize=(6,6),subplot_kw=dict(polar=True))
        ax.plot(a_closed,v_closed,'o-',color='skyblue'); ax.fill(a_closed,v_closed,alpha=0.3)
        ax.set_thetagrids(np.degrees(angles),[n.replace("_"," ").title() for n in names])
        ax.set_ylim(0,1); ax.set_title(f"Best Config t{self.best_trial} F1{self.best_f1:.4f}")
        plt.tight_layout(); fig.savefig(self.viz_dir/"best_radar.png"); plt.close()
        with open(self.viz_dir/"best_config_details.txt","w",encoding="utf-8") as f:
            f.write(f"Trial {self.best_trial} Best F1: {self.best_f1:.4f}\n")
            for n in names: f.write(f"{n}: {self.best_config[n]}\n")

def main():
    parser = argparse.ArgumentParser(description="Optimize NER with TRPO")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save optimization results")
    parser.add_argument("--model_type", type=str, default="roberta", help="Model type (e.g., roberta)")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Model name or path")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--random_trials", type=int, default=5, help="Number of random trials before TRPO")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs for NER")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for model files")
    parser.add_argument("--run_ner_path", type=str, required=True, help="Path to the NER training script")
    parser.add_argument("--skip_model_saving", action="store_true", help="Skip saving model checkpoints")
    parser.add_argument("--trpo_iters", type=int, default=10, help="TRPO policy update iterations")
    parser.add_argument("--max_kl", type=float, default=0.01, help="Max KL divergence constraint for TRPO")
    parser.add_argument("--cg_iters", type=int, default=10, help="Conjugate gradient iterations for TRPO")
    parser.add_argument("--damping", type=float, default=0.1, help="Damping factor for TRPO FIM computation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(Path(args.output_dir)/"trpo_script_main.log")]
    )
    global logger; logger = logging.getLogger(__name__)
    hyper = HyperparameterSpace()
    opt = TRPOOptimizer(
        hyperparameter_space=hyper,
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
        skip_model_saving=args.skip_model_saving,
        trpo_iters=args.trpo_iters,
        max_kl=args.max_kl,
        cg_iters=args.cg_iters,
        damping=args.damping,
        verbose=args.verbose
    )

    result = opt.optimize()
    print(f"=== TRPO Optimization Completed ===")
    print(f"Best Trial: {result['best_trial']}, F1: {result['best_f1']:.4f}")
    print(f"Best Config: {result['best_config']}")
    print(f"Visualizations in: {opt.viz_dir}")
    # save summary
    with open(Path(args.output_dir)/"summary.json","w",encoding="utf-8") as f:
        json.dump(result,f,indent=2)

if __name__=="__main__":
    main()
