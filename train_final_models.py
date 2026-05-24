#!/usr/bin/env python3
"""Train final 10-epoch models from optimizer best configs for fair comparison."""
import json, os, subprocess, sys, re
from pathlib import Path

REPO = os.environ.get("REPO_ROOT", "/home/azuma/aclpaper/repo")
VENV_PY = os.environ.get("VENV_PY", "/home/azuma/venv314/bin/python")
BOND_RUN = f"{REPO}/BOND/run_ner.py"
RESULTS = f"{REPO}/results"
CACHE = f"{REPO}/BOND/pretrained_model"

DATASET_DIRS = {
    "webpage": f"{REPO}/BOND/dataset/webpage_distant",
    "wikigold": f"{REPO}/BOND/dataset/wikigold_distant",
    "conll03": f"{REPO}/BOND/dataset/conll03_distant",
}
MODEL_TYPES = {"roberta-base": "roberta", "bert-base-cased": "bert"}

def parse_f1(path):
    if not os.path.exists(path): return 0.0
    with open(path) as f:
        for line in f:
            if "best_f1" in line:
                return float(line.split("=")[1].strip())
    return 0.0

def train_final(dataset, model, scheduler, seed, best_config):
    model_type = MODEL_TYPES[model]
    data_dir = DATASET_DIRS[dataset]
    out_dir = f"{RESULTS}/{dataset}/{model}/{scheduler}/seed_{seed}/final_model"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [VENV_PY, BOND_RUN,
        "--data_dir", data_dir,
        "--model_type", model_type,
        "--model_name_or_path", model,
        "--output_dir", out_dir,
        "--num_train_epochs", "10",
        "--max_seq_length", "128",
        "--do_train", "--do_eval", "--do_predict",
        "--evaluate_during_training", "--overwrite_output_dir",
        "--seed", str(seed),
        "--save_steps", "1000000",
        "--logging_steps", "100",
        "--dataloader_num_workers", "2",
        "--dataloader_pin_memory",
    ]
    if os.path.exists(CACHE):
        cmd += ["--cache_dir", CACHE]

    lr = best_config.get("learning_rate", 1e-5)
    wd = best_config.get("weight_decay", 1e-4)
    b1 = best_config.get("adam_beta1", 0.9)
    b2 = best_config.get("adam_beta2", 0.98)
    ws = best_config.get("warmup_steps", 200)
    bs = best_config.get("train_batch", 16)
    cmd += [
        "--learning_rate", f"{lr:.8e}",
        "--weight_decay", f"{wd:.8e}",
        "--adam_beta1", str(b1),
        "--adam_beta2", str(b2),
        "--warmup_steps", str(int(ws)),
        "--per_gpu_train_batch_size", str(int(bs)),
    ]

    print(f"  Training {dataset}/{model}/{scheduler}...")
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["CUDA_LAUNCH_BLOCKING"] = "0"

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env, cwd=REPO)
        test_file = f"{out_dir}/test_results.txt"
        f1 = parse_f1(test_file)
        print(f"  → F1={f1:.4f} (rc={proc.returncode})")
        return f1
    except Exception as e:
        print(f"  → ERROR: {e}")
        return 0.0

def main():
    datasets = ["webpage", "wikigold", "conll03"]
    models = ["roberta-base", "bert-base-cased"]
    schedulers = ["gp_ts", "td3", "sac"]

    results = {}
    for ds in datasets:
        for m in models:
            for s in schedulers:
                cell = f"{ds}/{m}/{s}"
                seed = 42
                config_file = f"{RESULTS}/{cell}/seed_{seed}/final_results.json"
                if not os.path.exists(config_file):
                    continue
                with open(config_file) as f:
                    cfg = json.load(f)
                best = cfg.get("best_config")
                if not best:
                    continue
                print(f"\n{cell}:")
                f1 = train_final(ds, m, s, seed, best)
                results[f"{cell}"] = f1

    print("\n" + "="*60)
    print("FINAL MODEL RESULTS (10-epoch, comparable to static baseline)")
    print("="*60)
    for k, v in sorted(results.items()):
        print(f"  {k:40s} F1={v:.4f}")

if __name__ == "__main__":
    main()
