#!/usr/bin/env python
# coding: utf-8
"""
基于 Soft Actor-Critic (SAC) 的 NER 模型超参数优化
"""
import os, sys, json, time, argparse, subprocess, random, shutil, logging, traceback, re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

from ppo_ner_optimizer import HyperparameterSpace

logger = logging.getLogger(__name__)

# Actor 网络：输出均值与对数标准差
class SACActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(s)
        mean = torch.tanh(self.mean(h))
        log_std = torch.clamp(self.log_std(h), -20, 2)
        std = torch.exp(log_std)
        return mean, std
    def get_action(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(s)
        dist = Normal(mean, std)
        x = dist.rsample()
        logp = dist.log_prob(x).sum(-1, keepdim=True)
        return torch.tanh(x), logp

# Critic 网络：Q(s,a)
class SACCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))

# 简易重放缓冲
class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size, self.ptr = max_size, 0
        self.storage = []
    def add(self, data: Tuple[torch.Tensor, torch.Tensor, float]):
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.storage), size=batch_size)
        s, a, r = zip(*[self.storage[i] for i in idx])
        return torch.stack(s), torch.stack(a), torch.tensor(r, dtype=torch.float32).unsqueeze(1)

class SACOptimizer:
    def __init__(self,
        hyperparameter_space: HyperparameterSpace,
        output_dir: str,
        n_trials: int = 50,
        seed: int = 42,
        data_dir: str = None,
        model_type: str = "roberta",
        model_name: str = "roberta-base",
        n_epochs: int = 1,
        max_seq_length: int = 128,
        cache_dir: str = None,
        run_ner_path: str = None,
        skip_model_saving: bool = False,
        # SAC 超参
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        buffer_size: int = 1000,
        batch_size: int = 16,
        update_after: int = 10,
        update_every: int = 5,
        verbose: bool = True
    ):
        self.space = hyperparameter_space
        self.out = Path(output_dir); self.out.mkdir(exist_ok=True, parents=True)
        self.n_trials, self.seed = n_trials, seed
        self.data_dir, self.model_type = data_dir, model_type
        self.model_name, self.n_epochs, self.max_seq = model_name, n_epochs, max_seq_length
        self.cache_dir, self.run_ner = cache_dir, run_ner_path
        self.skip_model_saving, self.verbose = skip_model_saving, verbose

        # SAC 参数
        self.gamma, self.tau = gamma, tau
        self.batch_size, self.update_after, self.update_every = batch_size, update_after, update_every

        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.state_dim = self.space.dim; self.act_dim = self.space.dim

        # 网络
        self.actor = SACActor(self.state_dim, self.act_dim)
        self.critic1 = SACCritic(self.state_dim, self.act_dim)
        self.critic2 = SACCritic(self.state_dim, self.act_dim)
        self.tgt_critic1 = SACCritic(self.state_dim, self.act_dim)
        self.tgt_critic2 = SACCritic(self.state_dim, self.act_dim)
        self.tgt_critic1.load_state_dict(self.critic1.state_dict())
        self.tgt_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters())+list(self.critic2.parameters()), lr=critic_lr
        )
        self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -self.act_dim

        self.buffer = ReplayBuffer(buffer_size)
        self.results, self.best_f1, self.best_cfg, self.best_trial = [], 0.0, None, -1

        # 初始 state
        self.cur_state = self.space.normalize_config(self.space.default_config())

        self.res_file = self.out/"opt_results.json"
        self.log_file = self.out/"sac_log.txt"
        self.viz_dir = self.out/"visualizations"; self.viz_dir.mkdir(exist_ok=True)

    def run_trial(self, cfg: Dict[str, Any], t: int) -> Dict[str, Any]:
        # 创建本次试验目录并记录配置
        trial_dir = self.out / f"trial_{t}"
        trial_dir.mkdir(exist_ok=True)
        self._log_config(cfg, t)
        # 构建命令
        base_args = [
            "--data_dir", str(self.data_dir),
            "--model_type", self.model_type,
            "--model_name_or_path", self.model_name,
            "--output_dir", str(trial_dir),
            "--num_train_epochs", str(self.n_epochs),
            "--max_seq_length", str(self.max_seq),
            "--do_train", "--do_eval", "--do_predict",
            "--evaluate_during_training", "--overwrite_output_dir",
            "--seed", str(self.seed + t)
        ]
        if self.cache_dir:
            base_args += ["--cache_dir", str(self.cache_dir)]
        config_args = self.space.config_to_args(cfg)
        cmd = [sys.executable, self.run_ner] + base_args + config_args
        logger.info(f"SAC trial {t} cmd: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate()
        # 不保存模型文件
        if self.skip_model_saving:
            for f in trial_dir.glob("*"):
                if f.is_dir() or f.suffix in [".bin", ".pt", ".ckpt"]:
                    try:
                        if f.is_dir(): shutil.rmtree(f)
                        else: f.unlink()
                    except: pass
        # 解析 test_results.txt
        resf = trial_dir / "test_results.txt"
        metrics = self._parse_result_file(resf)
        result = {"status": "success", "config": cfg, "trial": t}
        result.update(metrics)
        return result

    def _get_config(self, t: int) -> Tuple[Dict[str, Any], Any, Any]:
        if t <= self.update_after:
            return self.space.sample_random(), None, None
        state = self.cur_state.unsqueeze(0)
        with torch.no_grad():
            action, logp = self.actor.get_action(state)
        norm = ((action + 1) / 2).squeeze(0).clamp(0, 1)
        cfg = self.space.denormalize_vector(norm)
        return cfg, action, logp

    def _log_config(self, cfg: Dict[str, Any], t: int) -> None:
        s = f"[Trial {t}] Config: {cfg}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(s)

    def _parse_result_file(self, fpath: Path) -> Dict[str, float]:
        metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.search(r"(f1|precision|recall)\s*=\s*([0-9.]+)", line, re.I)
                    if m:
                        metrics[m.group(1).lower()] = float(m.group(2))
        except:
            pass
        return metrics

    def _plot_results(self) -> None:
        if not self.results: return
        valid = [r for r in self.results if r.get("status") == "success"]
        if not valid: return

        trials = [r["trial"] for r in valid]
        f1s = [r["f1"] for r in valid]
        configs = [r["config"] for r in valid]

        plt.figure(figsize=(10, 6))
        plt.plot(trials, f1s, 'o-', color='dodgerblue', label='F1 Score')
        plt.axhline(self.best_f1, color='red', linestyle='--',
                    label=f'Best F1: {self.best_f1:.4f} (Trial {self.best_trial})')
        plt.xlabel('Trial')
        plt.ylabel('F1 Score')
        plt.title('SAC Hyperparameter Optimization Progress')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.out / "optimization_progress.png")
        plt.close()

        for name in self.space.params.keys():
            plt.figure(figsize=(8, 5))
            vals = [cfg[name] for cfg in configs]
            sc = plt.scatter(vals, f1s, c=trials, cmap='viridis', s=50, alpha=0.8)
            plt.colorbar(sc, label='Trial')
            plt.xlabel(name); plt.ylabel('F1 Score')
            plt.title(f'{name} vs F1')
            if self.space.params[name]["type"] == "log_float":
                plt.xscale('log')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(self.viz_dir / f"param_{name}_vs_f1.png")
            plt.close()

    def _visualize_best_config(self) -> None:
        if not self.best_cfg: return
        names = list(self.space.params.keys())
        norm_vals = []
        for name in names:
            info = self.space.params[name]
            v = self.best_cfg[name]
            if info["type"] == "log_float":
                norm_val = (np.log(v) - np.log(info["min"])) / (np.log(info["max"]) - np.log(info["min"]))
            else:
                norm_val = (v - info["min"]) / (info["max"] - info["min"])
            norm_vals.append(norm_val)

        angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
        vals_closed = norm_vals + [norm_vals[0]]
        angles_closed = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles_closed, vals_closed, 'o-', color='deepskyblue')
        ax.fill(angles_closed, vals_closed, color='deepskyblue', alpha=0.3)
        ax.set_thetagrids(np.degrees(angles), [n.replace("_", " ").title() for n in names])
        ax.set_ylim(0, 1)
        ax.set_title(f'Best Config (Trial {self.best_trial}, F1 {self.best_f1:.4f})')
        plt.tight_layout()
        fig.savefig(self.viz_dir / "best_configuration_radar.png")
        plt.close()

    def optimize(self) -> Dict[str, Any]:
        for t in range(1, self.n_trials + 1):
            cfg, action, logp = self._get_config(t)
            self._log_config(cfg, t)
            res = self.run_trial(cfg, t)
            self.results.append(res)
            # 记录经验
            r = res.get("f1", 0.0)
            self.buffer.add((self.cur_state, action.squeeze(0) if action is not None else torch.zeros(self.act_dim), r))
            self.cur_state = self.space.normalize_config(cfg)

            # 更新网络
            if t > self.update_after and t % self.update_every == 0:
                self._update_sac()

            # 保存与可视化
            json.dump(self.results, open(self.res_file, "w"), indent=2)
            self._plot_results()
            # 更新最佳
            if res["f1"] > self.best_f1:
                self.best_f1, self.best_cfg, self.best_trial = res["f1"], cfg, t

        # 保存 final_results.json
        with open(self.out / "final_results.json", "w") as f:
            json.dump({"best_config": self.best_cfg, "best_f1": self.best_f1, "best_trial": self.best_trial}, f, indent=2)
        self._visualize_best_config()
        return {"best_config": self.best_cfg, "best_f1": self.best_f1, "best_trial": self.best_trial, "all_results": self.results}

    def _update_sac(self):
        for _ in range(self.update_every):
            s, a, r = self.buffer.sample(self.batch_size)
            with torch.no_grad():
                next_a, next_logp = self.actor.get_action(s)
                tgt_q1 = self.tgt_critic1(s, next_a)
                tgt_q2 = self.tgt_critic2(s, next_a)
                tgt_min = torch.min(tgt_q1, tgt_q2) - torch.exp(self.log_alpha)*next_logp
                y = r + self.gamma * tgt_min
            q1 = self.critic1(s, a); q2 = self.critic2(s, a)
            critic_loss = nn.MSELoss()(q1, y) + nn.MSELoss()(q2, y)
            self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()
            # Actor update
            act, logp = self.actor.get_action(s)
            q1_pi = self.critic1(s, act); q2_pi = self.critic2(s, act)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (torch.exp(self.log_alpha)*logp - q_pi).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            # Alpha update
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
            # 软更新
            for p, tp in zip(self.critic1.parameters(), self.tgt_critic1.parameters()):
                tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
            for p, tp in zip(self.critic2.parameters(), self.tgt_critic2.parameters()):
                tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize NER model hyperparameters using SAC")
    # 通用参数
    parser.add_argument("--data_dir",     type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--model_type",   type=str, default="roberta")
    parser.add_argument("--model_name",   type=str, default="roberta-base")
    parser.add_argument("--n_trials",     type=int, default=50)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--n_epochs",     type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--cache_dir",    type=str, default=None)
    parser.add_argument("--run_ner_path", type=str, required=True)
    parser.add_argument("--skip_model_saving", action="store_true")
    # SAC 专属参数
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--tau",         type=float, default=0.005)
    parser.add_argument("--actor_lr",    type=float, default=3e-4)
    parser.add_argument("--critic_lr",   type=float, default=3e-4)
    parser.add_argument("--alpha_lr",    type=float, default=3e-4)
    parser.add_argument("--init_alpha",  type=float, default=0.2)
    parser.add_argument("--buffer_size", type=int,   default=1000)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--update_after",type=int,   default=10)
    parser.add_argument("--update_every",type=int,   default=5)
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    opt = SACOptimizer(
        hyperparameter_space=HyperparameterSpace(),
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        seed=args.seed,
        data_dir=args.data_dir,
        model_type=args.model_type,
        model_name=args.model_name,
        n_epochs=args.n_epochs,
        max_seq_length=args.max_seq_length,
        cache_dir=args.cache_dir,
        run_ner_path=args.run_ner_path,
        skip_model_saving=args.skip_model_saving,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        init_alpha=args.init_alpha,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        update_after=args.update_after,
        update_every=args.update_every,
        verbose=args.verbose
    )
    opt.optimize()
