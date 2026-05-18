#!/usr/bin/env python3
"""
Experiment Runner for NER Hyperparameter Optimization.

Reads configs/experiments.yaml, generates the Cartesian product of
datasets x models x schedulers x seeds, and executes each experiment
via subprocess with checkpointing, timeout enforcement, and log capture.

Usage:
    python experiment_runner.py --dry-run              # List all combos
    python experiment_runner.py --dataset webpage      # Filter by dataset
    python experiment_runner.py --scheduler ppo        # Filter by scheduler
    python experiment_runner.py --seed 42              # Filter by seed
    python experiment_runner.py --config path/to/config.yaml
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

_ARG_NAME_OVERRIDES = {
    "dqn": {
        "model_name": "model_name_or_path",
        "n_epochs": "n_epochs_ner",
    },
}


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        if HAS_YAML:
            return yaml.safe_load(f)
        return json.load(f)


def resolve_repo_root(config_path):
    config_path = Path(config_path).resolve()
    if config_path.parent.name == "configs":
        return str(config_path.parent.parent)
    return str(config_path.parent)


def build_cartesian_product(config):
    return list(
        itertools.product(
            config["datasets"],
            config["models"],
            config["schedulers"],
            config["seeds"],
        )
    )


def get_output_dir(config, dataset, model, scheduler, seed):
    return os.path.join(
        config["results_dir"], dataset, model, scheduler, f"seed_{seed}"
    )


def is_completed(output_dir):
    return os.path.isfile(os.path.join(output_dir, "results.json"))


def build_command(config, dataset, model, scheduler, seed, repo_root):
    sched_config = config["scheduler_configs"][scheduler]
    is_static = sched_config.get("is_static", False)
    sched_args = sched_config.get("args", {})
    skip_model = sched_config.get("skip_model", False)
    overrides = _ARG_NAME_OVERRIDES.get(scheduler, {})

    model_type = config["model_type_map"][model]
    dataset_dir = config["dataset_dirs"][dataset]
    output_dir_rel = get_output_dir(config, dataset, model, scheduler, seed)

    data_dir = os.path.join(repo_root, config["bond_dir"], "dataset", dataset_dir)
    cache_dir = os.path.join(repo_root, config["cache_dir_rel"])
    run_ner_path = os.path.join(repo_root, config["run_ner_path_rel"])
    script_path = os.path.join(repo_root, sched_config["script"])
    output_dir = os.path.join(repo_root, output_dir_rel)

    cmd = ["python", script_path]

    if is_static:
        cmd.extend(
            [
                "--data_dir", data_dir,
                "--model_type", model_type,
                "--model_name_or_path", model,
                "--output_dir", output_dir,
                "--num_train_epochs", str(config["n_epochs"]),
                "--max_seq_length", str(config["max_seq_length"]),
                "--cache_dir", cache_dir,
                "--seed", str(seed),
            ]
        )
    else:
        model_name_flag = overrides.get("model_name", "model_name")
        n_epochs_flag = overrides.get("n_epochs", "n_epochs")

        cmd.extend(
            [
                "--data_dir", data_dir,
                "--output_dir", output_dir,
                "--model_type", model_type,
                f"--{model_name_flag}", model,
                f"--{n_epochs_flag}", str(config["n_epochs"]),
                "--max_seq_length", str(config["max_seq_length"]),
                "--cache_dir", cache_dir,
                "--seed", str(seed),
                "--run_ner_path", run_ner_path,
            ]
        )
        if skip_model:
            cmd.append("--skip_model_saving")
        if scheduler == "gp_ts":
            cmd.append("--reset")

    for key, value in sched_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    return cmd


def write_checkpoint(output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "results.json")
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_experiment(cmd, output_dir, timeout_seconds, repo_root, env_extra):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "experiment.log")

    env = os.environ.copy()
    env.update(env_extra)

    start_time = time.time()
    try:
        with open(log_file, "w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                cwd=repo_root,
                env=env,
            )
        elapsed = time.time() - start_time
        if proc.returncode == 0:
            write_checkpoint(
                output_dir,
                {
                    "status": "success",
                    "elapsed_seconds": round(elapsed, 1),
                    "returncode": 0,
                    "command": " ".join(cmd),
                },
            )
            return {"status": "success", "elapsed": elapsed, "returncode": 0}
        return {
            "status": "failed",
            "elapsed": elapsed,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {"status": "timeout", "elapsed": elapsed, "returncode": None}
    except Exception as exc:
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "elapsed": elapsed,
            "returncode": None,
            "error": str(exc),
        }


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser(
        description="NER Experiment Runner - Cartesian grid of "
        "datasets x models x schedulers x seeds"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments.yaml",
        help="Path to YAML configuration (default: configs/experiments.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all experiment combinations without executing",
    )
    parser.add_argument("--dataset", help="Filter by dataset name")
    parser.add_argument("--scheduler", help="Filter by scheduler name")
    parser.add_argument("--seed", type=int, help="Filter by random seed")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.is_file():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    repo_root = resolve_repo_root(str(config_path))

    print(f"Config: {config_path}")
    config = load_config(str(config_path))

    combos = build_cartesian_product(config)

    if args.dataset:
        combos = [c for c in combos if c[0] == args.dataset]
    if args.scheduler:
        combos = [c for c in combos if c[2] == args.scheduler]
    if args.seed is not None:
        combos = [c for c in combos if c[3] == args.seed]

    ds_set = sorted(set(c[0] for c in combos))
    m_set = sorted(set(c[1] for c in combos))
    s_set = sorted(set(c[2] for c in combos))
    seed_set = sorted(set(c[3] for c in combos))

    print(f"Experiments: {len(combos)}")
    print(f"  Datasets:   {len(ds_set)}  {ds_set}")
    print(f"  Models:     {len(m_set)}  {m_set}")
    print(f"  Schedulers: {len(s_set)}  {s_set}")
    print(f"  Seeds:      {len(seed_set)}  {seed_set}")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN -- Experiment Combinations")
        print("=" * 70)
        skipped = 0
        for ds, model, sched, seed in combos:
            output_dir_rel = get_output_dir(config, ds, model, sched, seed)
            done = is_completed(os.path.join(repo_root, output_dir_rel))
            marker = "  [SKIP - completed]" if done else ""
            print(f"  {ds:12s} {model:16s} {sched:12s} seed_{seed}{marker}")
            if done:
                skipped += 1

        print(f"\n{'─' * 70}")
        print(f"  Total combinations: {len(combos)}")
        print(f"  Already completed:  {skipped}")
        print(f"  To be executed:     {len(combos) - skipped}")

        if combos:
            ds, model, sched, seed = combos[0]
            cmd = build_command(config, ds, model, sched, seed, repo_root)
            print(f"\n{'─' * 70}")
            print(f"Sample command ({ds}/{model}/{sched}/seed_{seed}):")
            print(" ".join(cmd))
        return

    env_extra = config.get("env", {})
    timeout = config.get("timeout_seconds", 14400)

    successes = 0
    failures = 0
    skipped = 0
    total = len(combos)
    wall_start = time.time()

    for i, (ds, model, sched, seed) in enumerate(combos, start=1):
        output_dir_rel = get_output_dir(config, ds, model, sched, seed)
        output_dir = os.path.join(repo_root, output_dir_rel)
        exp_id = f"{ds}/{model}/{sched}/seed_{seed}"

        if is_completed(output_dir):
            print(f"[{i:4d}/{total}] SKIP  {exp_id}")
            skipped += 1
            continue

        print(f"[{i:4d}/{total}] RUN   {exp_id}")
        cmd = build_command(config, ds, model, sched, seed, repo_root)

        result = run_experiment(cmd, output_dir, timeout, repo_root, env_extra)

        elapsed_str = format_time(result["elapsed"])
        if result["status"] == "success":
            print(f"               OK    ({elapsed_str})")
            successes += 1
        elif result["status"] == "timeout":
            print(f"               TIMEOUT ({elapsed_str})")
            failures += 1
        else:
            print(f"               FAIL  ({elapsed_str}) rc={result.get('returncode')}")
            failures += 1

    wall_elapsed = time.time() - wall_start
    print()
    print("=" * 70)
    print("RUN COMPLETE")
    print(f"  Total:     {total}")
    print(f"  Successes: {successes}")
    print(f"  Failures:  {failures}")
    print(f"  Skipped:   {skipped}")
    print(f"  Wall time: {format_time(wall_elapsed)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
