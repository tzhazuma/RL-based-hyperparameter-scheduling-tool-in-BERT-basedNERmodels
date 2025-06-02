#!/usr/bin/env python
# coding: utf-8
"""
基于 TD3 (Twin Delayed DDPG) 的 NER 模型超参数优化
"""
import os, sys, json, time, argparse, subprocess, random, shutil, logging, traceback, re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

from ppo_ner_optimizer import HyperparameterSpace

logger = logging.getLogger(__name__)
def convert_tuples_in_list_to_lists(data_list):
    """
    转换给定列表中的所有元组为列表，包括嵌套结构（列表、元组和字典内部的元组）。

    参数:
        data_list (list): 输入的列表。这个列表的元素可能包含元组、列表、字典以及其他数据类型。
                          在列表元素中（无论嵌套多深）找到的所有元组都将被转换为列表。

    返回:
        list: 一个新的列表，其中所有遇到的元组都已转换为列表。

    异常:
        TypeError: 如果输入参数 data_list 不是一个列表。
    """
    if not isinstance(data_list, list):
        raise TypeError("输入必须是一个列表 (Input must be a list).")

    return [_recursive_tuple_converter(element) for element in data_list]

def _recursive_tuple_converter(item):
    """
    递归地遍历一个元素。如果找到元组，则将其转换为列表。
    列表和字典的值会被递归遍历。
    """
    if isinstance(item, tuple):
        # 将元组转换为列表，并对其元素进行递归处理
        return [_recursive_tuple_converter(elem) for elem in item]
    elif isinstance(item, list):
        # 对列表的元素进行递归处理
        return [_recursive_tuple_converter(elem) for elem in item]
    elif isinstance(item, dict):
        # 对字典的值进行递归处理
        return {key: _recursive_tuple_converter(value) for key, value in item.items()}
    else:
        # 基本情况：元素不是元组、列表或字典，直接返回
        return item
# Actor 网络
class TD3Actor(nn.Module):
    def __init__(self, s_dim, a_dim, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, a_dim), nn.Tanh()
        )
    def forward(self, s):
        return self.net(s)

# Critic 网络 (Q1 & Q2 共用)
class TD3Critic(nn.Module):
    def __init__(self, s_dim, a_dim, h=64):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(s_dim+a_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h,1))
        self.q2 = nn.Sequential(nn.Linear(s_dim+a_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h,1))
    def forward(self, s, a):
        sa = torch.cat([s,a], dim=-1)
        return self.q1(sa), self.q2(sa)

# 重放缓冲
class ReplayBuffer:
    def __init__(self, max_size):
        self.max, self.ptr = max_size, 0; self.storage=[]
    def add(self, data):
        if len(self.storage)<self.max: self.storage.append(data)
        else:
            self.storage[self.ptr]=data
            self.ptr=(self.ptr+1)%self.max
    def sample(self, bs):
        idx=np.random.randint(0,len(self.storage),bs)
        # td3_ner_optimizer.py, line 53附近

        # 从self.storage中根据idx提取状态、动作和奖励的组件列表
        s_components = [self.storage[i][0] for i in idx]
        a_components = [self.storage[i][1] for i in idx]
        r_components = [self.storage[i][2] for i in idx] # 这应该是一个奖励的列表, e.g., [r1, r2, r3]

        # 处理状态 (s)
        # 假设状态是可迭代的 (例如特征向量列表)，并且需要通过 zip(*...) 进行转置
        # 如果 s_components 为空，zip(*[]) 会返回一个空迭代器，list(zip(*[])) 是 []
        # 添加检查确保组件本身是可迭代的，以防状态也是标量（虽然不常见）
        if s_components and hasattr(s_components[0], '__iter__'): # 检查第一个元素是否可迭代
            s = list(zip(*s_components))
        else:
            s = s_components # 如果状态是标量列表或已经是期望格式，则直接使用

        # 处理动作 (a)
        # 类似地处理动作
        if a_components and hasattr(a_components[0], '__iter__'): # 检查第一个元素是否可迭代
            a = list(zip(*a_components))
        else:
            a = a_components # 如果动作是标量列表或已经是期望格式，则直接使用

        # 处理奖励 (r)
        # 奖励通常是标量，所以 r_components 就是我们需要的奖励列表
        r = list(r_components)
        #if(idx>1):
            #s,a,r=[zip(*[self.storage[i][i_] for i in idx]) for i_ in range(3)]
        #else:
            #s,a,r=[ zip(*[self.storage[0]                        for i_ in range(3)]
        #
        s=convert_tuples_in_list_to_lists(s)
        a=convert_tuples_in_list_to_lists(a)
        # 确保 s 和 a 是张量
        #
        r= convert_tuples_in_list_to_lists(r)
        try:
            torch.stack(s)
        except:
            print(s)
        return torch.stack(s), torch.stack(a), torch.tensor(r).unsqueeze(1)

class TD3Optimizer:
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
        # TD3 超参
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        buffer_size: int = 1000,
        batch_size: int = 16,
        update_after: int = 10,
        verbose: bool = True
    ):
        self.space = hyperparameter_space
        self.out = Path(output_dir); self.out.mkdir(exist_ok=True, parents=True)
        self.n_trials, self.seed = n_trials, seed
        self.data_dir, self.model_type = data_dir, model_type
        self.model_name, self.n_epochs, self.max_seq = model_name, n_epochs, max_seq_length
        self.cache_dir, self.run_ner = cache_dir, run_ner_path
        self.skip_model_saving, self.verbose = skip_model_saving, verbose

        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_delay, self.batch_size = policy_delay, batch_size
        self.update_after = update_after

        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        self.s_dim, self.a_dim = self.space.dim, self.space.dim

        # 网络
        self.actor = TD3Actor(self.s_dim, self.a_dim)
        self.actor_tgt = TD3Actor(self.s_dim, self.a_dim)
        self.critic = TD3Critic(self.s_dim, self.a_dim)
        self.critic_tgt = TD3Critic(self.s_dim, self.a_dim)
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(buffer_size)
        self.results, self.best_f1, self.best_cfg, self.best_trial = [], 0.0, None, -1
        self.cur_state = self.space.normalize_config(self.space.default_config())

        self.res_file = self.out/"opt_results.json"
        self.log_file = self.out/"td3_log.txt"
        self.viz_dir = self.out/"visualizations"; self.viz_dir.mkdir(exist_ok=True)

    def run_trial(self, cfg: Dict[str, Any], t: int) -> Dict[str, Any]:
        self._log_config(cfg, t)
        trial_dir = self.out / f"trial_{t}"
        trial_dir.mkdir(exist_ok=True)
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
        logger.info(f"TD3 trial {t} cmd: {' '.join(cmd)}")
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
        # 解析结果
        resf = trial_dir / "test_results.txt"
        metrics = self._parse_result_file(resf)
        result = {"status": "success", "config": cfg, "trial": t}
        result.update(metrics)
        return result

    def _get_config(self, t: int) -> Tuple[Dict[str, Any], Any]:
        if t <= self.update_after:
            return self.space.sample_random(), None
        s = self.cur_state.unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s)
        norm = ((a + 1) / 2).squeeze(0).clamp(0, 1)
        cfg = self.space.denormalize_vector(norm)
        return cfg, a

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

    def _update_td3(self, t: int):
        for it in range(self.policy_delay):
            s, a, r = self.buffer.sample(self.batch_size)
            # 计算目标 Q
            with torch.no_grad():
                noise = (torch.randn_like(a)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                a_tgt = (self.actor_tgt(s)+noise).clamp(-1,1)
                q1_t, q2_t = self.critic_tgt(s, a_tgt)
                q_t = torch.min(q1_t, q2_t)
                y = r + self.gamma * q_t
            q1, q2 = self.critic(s, a)
            critic_loss = nn.MSELoss()(q1, y) + nn.MSELoss()(q2, y)
            self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

            if it==0:  # Delayed policy update
                a_pred = self.actor(s)
                q1_pi, _ = self.critic(s, a_pred)
                actor_loss = -q1_pi.mean()
                self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
                # 同步目标网络
                for p, tp in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                    tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
                for p, tp in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                    tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)

    def _plot_results(self) -> None:
        if not self.results: return
        valid = [r for r in self.results if r.get("status") == "success"]
        if not valid: return

        trials = [r["trial"] for r in valid]
        f1s = [r["f1"] for r in valid]
        configs = [r["config"] for r in valid]

        plt.figure(figsize=(10, 6))
        plt.plot(trials, f1s, 'o-', color='darkorange', label='F1 Score')
        plt.axhline(self.best_f1, color='red', linestyle='--',
                    label=f'Best F1: {self.best_f1:.4f} (Trial {self.best_trial})')
        plt.xlabel('Trial'); plt.ylabel('F1 Score')
        plt.title('TD3 Hyperparameter Optimization Progress')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.out / "optimization_progress.png")
        plt.close()

        for name in self.space.params.keys():
            plt.figure(figsize=(8, 5))
            vals = [cfg[name] for cfg in configs]
            sc = plt.scatter(vals, f1s, c=trials, cmap='plasma', s=50, alpha=0.8)
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
        ax.plot(angles_closed, vals_closed, 'o-', color='teal')
        ax.fill(angles_closed, vals_closed, color='teal', alpha=0.3)
        ax.set_thetagrids(np.degrees(angles), [n.replace("_", " ").title() for n in names])
        ax.set_ylim(0, 1)
        ax.set_title(f'Best Config (Trial {self.best_trial}, F1 {self.best_f1:.4f})')
        plt.tight_layout()
        fig.savefig(self.viz_dir / "best_configuration_radar.png")
        plt.close()

    def optimize(self) -> Dict[str, Any]:
        for t in range(1, self.n_trials + 1):
            cfg, action = self._get_config(t)
            self._log_config(cfg, t)
            res = self.run_trial(cfg, t)
            self.results.append(res)
            r = res.get("f1", 0.0)
            self.buffer.add((self.cur_state, action.squeeze(0) if action is not None else torch.zeros(self.a_dim), r))
            self.cur_state = self.space.normalize_config(cfg)

            if t > self.update_after:
                self._update_td3(t)

            json.dump(self.results, open(self.res_file, "w"), indent=2)
            self._plot_results()
            if res["f1"] > self.best_f1:
                self.best_f1, self.best_cfg, self.best_trial = res["f1"], cfg, t

        with open(self.out / "final_results.json", "w") as f:
            json.dump({"best_config": self.best_cfg, "best_f1": self.best_f1, "best_trial": self.best_trial}, f, indent=2)
        self._visualize_best_config()
        return {"best_config": self.best_cfg, "best_f1": self.best_f1, "best_trial": self.best_trial, "all_results": self.results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize NER model hyperparameters using TD3")
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
    # TD3 专属参数
    parser.add_argument("--actor_lr",     type=float, default=3e-4)
    parser.add_argument("--critic_lr",    type=float, default=3e-4)
    parser.add_argument("--gamma",        type=float, default=0.99)
    parser.add_argument("--tau",          type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip",   type=float, default=0.5)
    parser.add_argument("--policy_delay", type=int,   default=2)
    parser.add_argument("--buffer_size",  type=int,   default=1000)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--update_after", type=int,   default=10)
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    opt = TD3Optimizer(
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
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        update_after=args.update_after,
        verbose=args.verbose
    )
    opt.optimize()
