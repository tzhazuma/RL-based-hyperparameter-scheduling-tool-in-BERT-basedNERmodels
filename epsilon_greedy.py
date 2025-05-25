import os
import argparse
import random
import subprocess
from tqdm import trange

def sample_hyperparameters():
    return {
        "learning_rate": random.uniform(1e-6, 1e-4),
        "weight_decay": random.uniform(1e-5, 1e-3),
        "adam_beta1": random.uniform(0.8, 0.99),
        "adam_beta2": random.uniform(0.95, 0.999),
        "warmup_steps": random.randint(50, 500),
        "per_gpu_train_batch_size": random.choice([8, 12, 16]),  # 避免过大引起OOM
    }

def run_trial(trial, args, hparams):
    output_dir = os.path.normpath(os.path.join(args.output_dir, f"trial_{trial}"))
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "python", args.run_ner_path,
        "--data_dir", args.data_dir,
        "--model_type", args.model_type,
        "--model_name_or_path", args.model_name_or_path,
        "--output_dir", output_dir,
        "--do_train", "--do_eval", "--do_predict", "--evaluate_during_training",
        "--overwrite_output_dir",
        "--seed", str(args.seed),
        "--learning_rate", str(hparams["learning_rate"]),
        "--weight_decay", str(hparams["weight_decay"]),
        "--adam_beta1", str(hparams["adam_beta1"]),
        "--adam_beta2", str(hparams["adam_beta2"]),
        "--warmup_steps", str(hparams["warmup_steps"]),
        "--per_gpu_train_batch_size", str(hparams["per_gpu_train_batch_size"]),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ε-greedy] Trial {trial} failed.\nSTDERR:\n{result.stderr}")
        return 0.0

    results_file = os.path.join(output_dir, "test_results.txt")
    if not os.path.exists(results_file):
        print(f"[ε-greedy] Warning: {results_file} not found.")
        return 0.0

    with open(results_file, "r") as f:
        for line in f:
            if "best_f1" in line or line.startswith("f1"):
                try:
                    return float(line.strip().split("=")[-1])
                except ValueError:
                    print(f"[ε-greedy] Could not parse F1 score: {line}")
                    return 0.0
    return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--run_ner_path", required=True)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--model_type", default="roberta")
    parser.add_argument("--model_name_or_path", default="roberta-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./outputs/epsgreedy_ner_opt")
    args = parser.parse_args()

    print(f"Arguments: {args}")
    print("Starting epsilon-greedy optimization...")

    best_f1 = 0.0
    best_config = None

    for trial in trange(args.n_trials):
        hparams = sample_hyperparameters()
        f1_score = run_trial(trial, args, hparams)
        if f1_score > best_f1:
            best_f1 = f1_score
            best_config = hparams
        print(f"[ε-greedy] Trial {trial} F1 = {f1_score:.4f} (best = {best_f1:.4f})")

    print(f"Best F1: {best_f1}, config: {best_config}")

if __name__ == "__main__":
    main()
