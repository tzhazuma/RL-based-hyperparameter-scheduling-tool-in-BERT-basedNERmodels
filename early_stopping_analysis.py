#!/usr/bin/env python
# coding: utf-8
"""
early_stopping_analysis.py — Analyze training logs to determine convergence speed,
overfitting behavior, and generate comparison tables/plots across schedulers.

Handles both:
- Per-epoch JSON logs (epoch_logs.json) from Wave 3 final training
- Optimization results (optimization_results.json) from hyperparameter search
  (one F1 per trial, limited convergence analysis)
"""

import os, sys, json, re, csv, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None


def _soft_float(val: Any, default: float = 0.0) -> float:
    """Try to convert to float; return default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _find_json_files(results_dir: Path, pattern: str = "*.json") -> List[Path]:
    """Recursively find JSON files under results_dir."""
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob(pattern))


def _infer_dataset_from_path(file_path: Path) -> str:
    """Try to infer dataset name from the file path."""
    path_str = str(file_path)
    for kw in ("conll03", "conll2003", "webpage", "wikigold", "twitter", "ontonotes", "bc5cdr"):
        if kw in path_str.lower():
            return kw
    return "unknown"


def _infer_model_from_path(file_path: Path) -> str:
    """Try to infer model/optimizer name from the file path."""
    path_str = str(file_path)
    for kw in ("gp_ts", "gpts", "ppo", "td3", "dqn", "epsilon", "sac", "trpo"):
        if kw in path_str.lower():
            return kw.replace("_", "-").upper()
    return "unknown"


def _infer_scheduler_from_path(file_path: Path) -> str:
    """Alias for _infer_model_from_path for consistency."""
    return _infer_model_from_path(file_path)


class ConvergenceAnalyzer:
    """
    Load experimental results and compute convergence / overfitting metrics.

    Parameters
    ----------
    results_dir : str or Path
        Root directory containing experiment results.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self._cache: Dict[str, Dict] = {}

    def load_logs(self, dataset: str, model: str, scheduler: str, seed: int = 0) -> Optional[Dict]:
        """
        Load per-epoch metrics for a specific experiment.

        Looks for epoch_logs.json in the expected directory layout:
            <results_dir>/<model>/<dataset>/<scheduler>/seed_<seed>/epoch_logs.json

        Returns a dict with keys:
            "epochs" : List[int]
            "f1"     : List[float]
            "train_loss" : List[float]  (may be empty if not logged)
            "eval_loss"  : List[float]  (may be empty if not logged)
        Returns None if no data found.
        """
        cache_key = f"{dataset}/{model}/{scheduler}/{seed}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        candidate_paths = [
            self.results_dir / model / dataset / scheduler / f"seed_{seed}" / "epoch_logs.json",
            self.results_dir / model / dataset / scheduler / "epoch_logs.json",
            self.results_dir / model / dataset / f"seed_{seed}" / "epoch_logs.json",
            self.results_dir / model / dataset / "epoch_logs.json",
            self.results_dir / dataset / model / scheduler / f"seed_{seed}" / "epoch_logs.json",
            self.results_dir / dataset / model / scheduler / "epoch_logs.json",
        ]

        for path in candidate_paths:
            if path.exists():
                data = self._parse_epoch_logs(path)
                self._cache[cache_key] = data
                return data

        for json_file in _find_json_files(self.results_dir):
            if json_file.name != "epoch_logs.json":
                continue
            try:
                raw = json.loads(json_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            meta = raw if isinstance(raw, dict) else {}
            if (str(meta.get("dataset", "")).lower() == dataset.lower()
                    and str(meta.get("model", "")).lower() == model.lower()
                    and str(meta.get("scheduler", "")).lower() == scheduler.lower()
                    and int(meta.get("seed", seed)) == seed):
                data = self._normalize_epoch_dict(raw)
                self._cache[cache_key] = data
                return data

            if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
                first = raw[0]
                if "epoch" in first or "f1" in first:
                    data = self._normalize_epoch_list(raw)
                    self._cache[cache_key] = data
                    return data

        self._cache[cache_key] = None
        return None

    def convergence_epoch(self, dataset: str, model: str, scheduler: str,
                          seed: int = 0, threshold: float = 0.95) -> Optional[int]:
        """
        Return the first epoch where F1 reaches `threshold * final_F1`.

        The final F1 is the maximum F1 over all epochs (or the last-epoch value
        if the model plateaus).
        """
        logs = self.load_logs(dataset, model, scheduler, seed)
        if logs is None or len(logs["f1"]) == 0:
            return None
        f1_arr = np.array(logs["f1"])
        final_f1 = np.max(f1_arr) if np.max(f1_arr) > 0 else f1_arr[-1]
        target = threshold * final_f1
        indices = np.where(f1_arr >= target)[0]
        if len(indices) == 0:
            return None
        first_idx = indices[0]
        epochs = logs["epochs"]
        return epochs[first_idx] if first_idx < len(epochs) else int(first_idx + 1)

    def overfitting_detection(self, dataset: str, model: str, scheduler: str,
                              seed: int = 0, patience: int = 5) -> Optional[int]:
        """
        Detect overfitting: return the epoch where eval_loss increases for
        `patience` consecutive epochs while train_loss still decreases.

        Returns None if no overfitting detected or if loss data is unavailable.
        """
        logs = self.load_logs(dataset, model, scheduler, seed)
        if logs is None:
            return None

        train_loss = np.array(logs.get("train_loss", []))
        eval_loss = np.array(logs.get("eval_loss", []))

        if len(eval_loss) < patience + 1 or len(train_loss) < patience + 1:
            return None

        for i in range(patience, len(eval_loss)):
            eval_increasing = all(
                eval_loss[j] > eval_loss[j - 1] for j in range(i - patience + 1, i + 1)
            )
            train_decreasing = all(
                train_loss[j] < train_loss[j - 1] for j in range(i - patience + 1, i + 1)
            )
            if eval_increasing and train_decreasing:
                epochs = logs["epochs"]
                return epochs[i] if i < len(epochs) else int(i + 1)

        return None

    def best_epoch(self, dataset: str, model: str, scheduler: str,
                   seed: int = 0) -> Optional[int]:
        """Return the epoch (1-indexed) with the highest validation F1."""
        logs = self.load_logs(dataset, model, scheduler, seed)
        if logs is None or len(logs["f1"]) == 0:
            return None
        f1_arr = logs["f1"]
        best_idx = int(np.argmax(f1_arr))
        epochs = logs["epochs"]
        return epochs[best_idx] if best_idx < len(epochs) else best_idx + 1

    def final_performance(self, dataset: str, model: str, scheduler: str,
                          seed: int = 0) -> Optional[float]:
        """Return the F1 at the last available epoch."""
        logs = self.load_logs(dataset, model, scheduler, seed)
        if logs is None or len(logs["f1"]) == 0:
            return None
        return float(logs["f1"][-1])

    def best_performance(self, dataset: str, model: str, scheduler: str,
                         seed: int = 0) -> Optional[float]:
        """Return the maximum F1 across all epochs."""
        logs = self.load_logs(dataset, model, scheduler, seed)
        if logs is None or len(logs["f1"]) == 0:
            return None
        return float(np.max(logs["f1"]))

    def training_time_per_epoch(self, dataset: str, model: str, scheduler: str,
                                seed: int = 0) -> Optional[List[float]]:
        """
        Return per-epoch training time (seconds) if timing data was logged.
        """
        logs = self.load_logs(dataset, model, scheduler, seed)
        if logs is None:
            return None
        times = logs.get("epoch_time", [])
        return times if len(times) > 0 else None

    def analyze_all_schedulers(self, dataset: str, model: str,
                               threshold: float = 0.95,
                               patience: int = 5) -> List[Dict]:
        """
        Compare convergence across all schedulers for one dataset+model.

        Returns a list of dicts, one per scheduler/seed combo found.
        """
        rows = []
        schedulers = self._available_schedulers(dataset, model)
        for sched in schedulers:
            row = self._single_analysis(dataset, model, sched, threshold, patience)
            if row is not None:
                rows.append(row)
        return rows

    def _single_analysis(self, dataset: str, model: str, scheduler: str,
                         threshold: float, patience: int) -> Optional[Dict]:
        seed = 0
        logs = self.load_logs(dataset, model, scheduler, seed)
        if logs is None or len(logs["f1"]) == 0:
            return None

        conv = self.convergence_epoch(dataset, model, scheduler, seed, threshold)
        overfit = self.overfitting_detection(dataset, model, scheduler, seed, patience)
        best_ep = self.best_epoch(dataset, model, scheduler, seed)
        best_f1_val = self.best_performance(dataset, model, scheduler, seed)
        final_f1_val = self.final_performance(dataset, model, scheduler, seed)
        n_epochs = len(logs["f1"])

        return {
            "dataset": dataset,
            "model": model,
            "scheduler": scheduler,
            "num_epochs": n_epochs,
            "epochs_to_{:.0f}%".format(threshold * 100): conv,
            "best_epoch": best_ep,
            "best_F1": round(best_f1_val, 4) if best_f1_val is not None else None,
            "final_F1": round(final_f1_val, 4) if final_f1_val is not None else None,
            "overfitting_epoch": overfit,
        }

    def generate_convergence_table(self, threshold: float = 0.95,
                                   patience: int = 5) -> str:
        """
        Produce a CSV-formatted convergence comparison table.

        Columns: dataset, model, scheduler, epochs_to_95%, best_epoch,
                 best_F1, final_F1, overfitting_epoch
        """
        all_rows = []
        for dataset, model in self._available_experiments():
            rows = self.analyze_all_schedulers(dataset, model, threshold, patience)
            all_rows.extend(rows)

        if not all_rows:
            return "No per-epoch data available. Falling back to optimization results summary.\n" + self._fallback_table()

        output_path = self.results_dir / "analysis" / "convergence_table.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["dataset", "model", "scheduler", "num_epochs",
                      "epochs_to_{:.0f}%".format(threshold * 100),
                      "best_epoch", "best_F1", "final_F1", "overfitting_epoch"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

        lines = [",".join(fieldnames)]
        for row in all_rows:
            lines.append(",".join(str(row.get(h, "")) for h in fieldnames))
        return "\n".join(lines)

    def _fallback_table(self) -> str:
        """
        When per-epoch data is absent, summarize optimization_results.json files.
        """
        fieldnames = ["dataset", "model", "trials", "best_F1", "mean_F1", "std_F1"]
        lines = [",".join(fieldnames)]

        for json_file in _find_json_files(self.results_dir, "optimization_results.json"):
            try:
                raw = json.loads(json_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            if isinstance(raw, dict):
                trials = raw.get("all_results", [])
            elif isinstance(raw, list):
                trials = raw
            else:
                continue

            f1_vals = [t["f1"] for t in trials if isinstance(t, dict) and "f1" in t]
            if not f1_vals:
                continue

            dataset = _infer_dataset_from_path(json_file)
            model = _infer_model_from_path(json_file)

            lines.append(",".join([
                dataset,
                model,
                str(len(f1_vals)),
                f"{max(f1_vals):.4f}",
                f"{np.mean(f1_vals):.4f}",
                f"{np.std(f1_vals):.4f}",
            ]))

        output_path = self.results_dir / "analysis" / "convergence_table.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            f.write("\n".join(lines) + "\n")

        return "\n".join(lines)

    def plot_convergence_comparison(self, dataset: str, model: str,
                                    save: bool = True) -> Optional[str]:
        """
        Plot F1 vs epoch for all schedulers on one dataset.

        Uses different line styles (solid, dashed, dashdot, dotted) plus
        markers for grayscale readability.  Draws vertical dashed lines at
        overfitting onset if detected.

        Returns the path to the saved PNG, or None if plotting fails.
        """
        if not HAS_MPL:
            warnings.warn("matplotlib not available; skipping plot.")
            return None

        schedulers = self._available_schedulers(dataset, model)
        if not schedulers:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
        markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(schedulers)))

        for idx, sched in enumerate(schedulers):
            logs = self.load_logs(dataset, model, sched, seed=0)
            if logs is None or len(logs["f1"]) == 0:
                continue

            epochs = logs["epochs"]
            f1_vals = logs["f1"]
            ls = linestyles[idx % len(linestyles)]
            mk = markers[idx % len(markers)]
            c = colors[idx % len(colors)]

            ax.plot(epochs, f1_vals, linestyle=ls, marker=mk, color=c,
                    label=sched, markersize=4, linewidth=1.5)

            of_epoch = self.overfitting_detection(dataset, model, sched, seed=0)
            if of_epoch is not None:
                ax.axvline(x=of_epoch, color=c, linestyle="--", alpha=0.4,
                           linewidth=1)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title(f"Convergence Comparison — {dataset.upper()} / {model.upper()}", fontsize=14)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        if save:
            output_path = self.results_dir / "analysis" / f"{dataset}_convergence.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return str(output_path)
        else:
            plt.close(fig)
            return None

    def generate_summary_report(self, threshold: float = 0.95,
                                patience: int = 5) -> str:
        """
        Generate a markdown summary report of key convergence findings.
        """
        lines = [
            "# Convergence & Overfitting Analysis — Summary",
            "",
            f"*Generated from: {self.results_dir}*",
            f"*Convergence threshold: {threshold:.0%} of final F1*",
            f"*Overfitting patience: {patience} epochs*",
            "",
            "---",
            "",
        ]

        experiments = self._available_experiments()
        if not experiments:
            lines.append("## No per-epoch data found")
            lines.append("")
            lines.append("Falling back to optimization results summary.")
            lines.append("")
            lines.append("```")
            lines.append(self._fallback_table())
            lines.append("```")
            return "\n".join(lines)

        for dataset, model in experiments:
            lines.append(f"## {dataset.upper()} / {model.upper()}")
            lines.append("")

            rows = self.analyze_all_schedulers(dataset, model, threshold, patience)

            if not rows:
                lines.append("*No results found.*\n")
                continue

            conv_col = "epochs_to_{:.0f}%".format(threshold * 100)
            valid_conv = [r for r in rows if r.get(conv_col) is not None]
            fastest = min(valid_conv, key=lambda r: r[conv_col]) if valid_conv else None
            best_f1_row = max(rows, key=lambda r: r.get("best_F1") or 0)

            if fastest:
                lines.append(f"- **Fastest convergence**: {fastest['scheduler']} "
                             f"({fastest[conv_col]} epochs to {threshold:.0%} of final F1)")
            if best_f1_row:
                lines.append(f"- **Best F1**: {best_f1_row['scheduler']} "
                             f"({best_f1_row['best_F1']:.4f})")
            lines.append("")

            header = ["Scheduler", "Epochs to {:.0f}%".format(threshold * 100),
                      "Best F1", "Final F1", "Overfit epoch"]
            sep = ["---", "---", "---", "---", "---"]
            table_rows = []
            for r in rows:
                table_rows.append([
                    r["scheduler"],
                    str(r.get(conv_col, "—")),
                    f'{r.get("best_F1", "—")}',
                    f'{r.get("final_F1", "—")}',
                    str(r.get("overfitting_epoch", "—")),
                ])

            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(sep) + " |")
            for tr in table_rows:
                lines.append("| " + " | ".join(tr) + " |")
            lines.append("")

            plot_path = self.plot_convergence_comparison(dataset, model, save=True)
            if plot_path:
                lines.append(f"![Convergence plot]({plot_path})")
                lines.append("")

            lines.append("---\n")

        output_path = self.results_dir / "analysis" / "summary.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return "\n".join(lines)

    def _available_experiments(self) -> List[Tuple[str, str]]:
        """
        Discover (dataset, model) combos by scanning the results_dir for
        epoch_logs.json files.
        """
        combos = set()
        for json_file in _find_json_files(self.results_dir, "epoch_logs.json"):
            try:
                raw = json.loads(json_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            if isinstance(raw, dict):
                ds = raw.get("dataset", _infer_dataset_from_path(json_file))
                md = raw.get("model", _infer_model_from_path(json_file))
            else:
                ds = _infer_dataset_from_path(json_file)
                md = _infer_model_from_path(json_file)

            parts = json_file.relative_to(self.results_dir).parts if json_file.is_relative_to(self.results_dir) else json_file.parts
            if len(parts) >= 2:
                p0, p1 = parts[0].lower(), parts[1].lower()
                known_ds = {"conll03", "conll2003", "webpage", "wikigold", "twitter", "ontonotes"}
                known_md = {"gp_ts", "gpts", "ppo", "td3", "dqn", "epsilon", "sac", "trpo"}
                if p0 in known_ds and p1 in known_md:
                    ds, md = parts[0], parts[1]
                elif p0 in known_md and p1 in known_ds:
                    ds, md = parts[1], parts[0]

            combos.add((str(ds).lower(), str(md).lower()))

        if not combos:
            for json_file in _find_json_files(self.results_dir, "optimization_results.json"):
                ds = _infer_dataset_from_path(json_file)
                md = _infer_model_from_path(json_file)
                combos.add((ds, md))

        return sorted(combos)

    def _available_schedulers(self, dataset: str, model: str) -> List[str]:
        """Return list of scheduler names for a given dataset+model."""
        schedulers = set()
        for json_file in _find_json_files(self.results_dir, "epoch_logs.json"):
            try:
                raw = json.loads(json_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            if isinstance(raw, dict):
                meta_ds = str(raw.get("dataset", "")).lower()
                meta_md = str(raw.get("model", "")).lower()
                if meta_ds == dataset.lower() and meta_md == model.lower():
                    sched = raw.get("scheduler", _infer_scheduler_from_path(json_file))
                    schedulers.add(sched)
            else:
                parts = json_file.relative_to(self.results_dir).parts if json_file.is_relative_to(self.results_dir) else json_file.parts
                for p in parts:
                    sched = _infer_scheduler_from_path(Path(p))
                    if sched != "unknown":
                        schedulers.add(sched)

        if not schedulers:
            for json_file in _find_json_files(self.results_dir, "optimization_results.json"):
                sched = _infer_scheduler_from_path(json_file)
                if sched != "unknown":
                    schedulers.add(sched)

        return sorted(schedulers)

    @staticmethod
    def _parse_epoch_logs(path: Path) -> Optional[Dict]:
        """Load and normalize an epoch_logs.json file."""
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        if isinstance(raw, dict) and "epoch_logs" in raw:
            return ConvergenceAnalyzer._normalize_epoch_list(raw["epoch_logs"])
        elif isinstance(raw, list):
            return ConvergenceAnalyzer._normalize_epoch_list(raw)
        elif isinstance(raw, dict):
            return ConvergenceAnalyzer._normalize_epoch_dict(raw)
        return None

    @staticmethod
    def _normalize_epoch_list(entries: List[Dict]) -> Dict:
        """Convert a list of epoch dicts to canonical form."""
        epochs = []
        f1_vals = []
        train_loss = []
        eval_loss = []
        epoch_time = []

        for entry in entries:
            e = int(entry.get("epoch", len(epochs)))
            epochs.append(e)
            f1_vals.append(_soft_float(entry.get("f1")))
            train_loss.append(_soft_float(entry.get("train_loss", entry.get("loss", float("nan")))))
            eval_loss.append(_soft_float(entry.get("eval_loss", entry.get("val_loss", float("nan")))))
            epoch_time.append(_soft_float(entry.get("epoch_time", entry.get("time", float("nan")))))

        return {
            "epochs": epochs,
            "f1": f1_vals,
            "train_loss": [v for v in train_loss if not np.isnan(v)],
            "eval_loss": [v for v in eval_loss if not np.isnan(v)],
            "epoch_time": [v for v in epoch_time if not np.isnan(v)],
        }

    @staticmethod
    def _normalize_epoch_dict(raw: Dict) -> Dict:
        """If the JSON is a dict with epoch keys like '0', '1', etc."""
        epochs = []
        f1_vals = []
        train_loss = []
        eval_loss = []

        for key, val in raw.items():
            if key in ("epoch_logs", "config", "trial", "seed", "model", "dataset", "scheduler"):
                continue
            if isinstance(val, dict):
                e = _soft_float(key, None)
                if e is None:
                    continue
                epochs.append(int(e))
                f1_vals.append(_soft_float(val.get("f1"), float("nan")))
                train_loss.append(_soft_float(val.get("train_loss", val.get("loss", float("nan")))))
                eval_loss.append(_soft_float(val.get("eval_loss", val.get("val_loss", float("nan")))))

        if epochs:
            order = np.argsort(epochs)
            epochs = [epochs[i] for i in order]
            f1_vals = [f1_vals[i] for i in order]
            train_loss = [train_loss[i] for i in order]
            eval_loss = [eval_loss[i] for i in order]

        return {
            "epochs": epochs,
            "f1": f1_vals,
            "train_loss": [v for v in train_loss if not np.isnan(v)],
            "eval_loss": [v for v in eval_loss if not np.isnan(v)],
            "epoch_time": [],
        }

    def list_available_data(self) -> List[Dict]:
        """
        Scan results_dir and return a list of discovered experiment data sources.
        Each entry contains: type, path, dataset, model, scheduler, n_trials
        """
        entries = []

        for json_file in _find_json_files(self.results_dir, "epoch_logs.json"):
            entries.append({
                "type": "epoch_logs",
                "path": str(json_file),
                "dataset": _infer_dataset_from_path(json_file),
                "model": _infer_model_from_path(json_file),
            })

        for json_file in _find_json_files(self.results_dir, "optimization_results.json"):
            try:
                raw = json.loads(json_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            trials = raw if isinstance(raw, list) else raw.get("all_results", [])
            entries.append({
                "type": "optimization_results",
                "path": str(json_file),
                "dataset": _infer_dataset_from_path(json_file),
                "model": _infer_model_from_path(json_file),
                "n_trials": len(trials) if isinstance(trials, list) else 0,
            })

        return entries


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training logs for convergence speed and overfitting."
    )
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root directory containing experiment results")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to analyze (e.g., webpage, conll03, wikigold)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model/optimizer type (e.g., GP-TS, PPO, TD3)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Convergence threshold as fraction of final F1 (default: 0.95)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for overfitting detection (default: 5)")
    parser.add_argument("--list", action="store_true",
                        help="List available data sources and exit")
    parser.add_argument("--csv", action="store_true",
                        help="Generate convergence table CSV")
    parser.add_argument("--plot", action="store_true",
                        help="Generate convergence plots")
    parser.add_argument("--report", action="store_true",
                        help="Generate summary markdown report")
    parser.add_argument("--all", action="store_true",
                        help="Run all analyses (CSV + plots + report)")

    args = parser.parse_args()

    analyzer = ConvergenceAnalyzer(results_dir=args.results_dir)

    if args.list:
        entries = analyzer.list_available_data()
        if not entries:
            print("No result files found in", args.results_dir)
            return
        print(f"{'Type':<25} {'Dataset':<15} {'Model':<15} {'Path'}")
        print("-" * 90)
        for e in entries:
            print(f"{e['type']:<25} {e['dataset']:<15} {e['model']:<15} {e['path']}")
        return

    if args.dataset:
        if args.model:
            experiments = [(args.dataset, args.model)]
        else:
            all_models = set()
            for e in analyzer.list_available_data():
                if e["dataset"] == args.dataset:
                    all_models.add(e["model"])
            experiments = [(args.dataset, m) for m in sorted(all_models)]
    else:
        experiments = analyzer._available_experiments()

    if not experiments:
        print("No experiments found. Try --list to see available data.")
        return

    do_csv = args.csv or args.all
    do_plot = args.plot or args.all
    do_report = args.report or args.all

    if do_csv:
        print("Generating convergence table...")
        table = analyzer.generate_convergence_table(threshold=args.threshold,
                                                     patience=args.patience)
        print(table)

    if do_plot:
        for ds, md in experiments:
            print(f"Plotting {ds}/{md}...")
            path = analyzer.plot_convergence_comparison(ds, md, save=True)
            if path:
                print(f"  -> {path}")
            else:
                print(f"  (no data)")

    if do_report:
        print("Generating summary report...")
        report = analyzer.generate_summary_report(threshold=args.threshold,
                                                   patience=args.patience)
        preview_lines = report.split("\n")[:30]
        print("\n".join(preview_lines))
        print(f"\n... Full report written to {Path(args.results_dir) / 'analysis' / 'summary.md'}")

    if not any([do_csv, do_plot, do_report, args.list]):
        # Default: show convergence table
        print("Generating convergence table...")
        table = analyzer.generate_convergence_table(threshold=args.threshold,
                                                     patience=args.patience)
        print(table)


if __name__ == "__main__":
    main()
