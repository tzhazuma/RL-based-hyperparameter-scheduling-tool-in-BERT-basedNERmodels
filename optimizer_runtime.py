#!/usr/bin/env python3

import os
import shutil
from pathlib import Path


def add_run_ner_runtime_args(parser):
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--dataloader_pin_memory", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--eval_on_epoch_end_only", action="store_true")
    parser.add_argument("--train_early_stopping_patience", type=int, default=0)
    parser.add_argument("--train_early_stopping_min_delta", type=float, default=0.0)
    return parser


def extend_run_ner_args(base_args, args):
    base_args.extend([
        "--save_steps", str(getattr(args, "save_steps", 500)),
        "--logging_steps", str(getattr(args, "logging_steps", 50)),
        "--gradient_accumulation_steps", str(getattr(args, "gradient_accumulation_steps", 1)),
        "--dataloader_num_workers", str(getattr(args, "dataloader_num_workers", 0)),
    ])

    if getattr(args, "dataloader_pin_memory", False):
        base_args.append("--dataloader_pin_memory")
    if getattr(args, "fp16", False):
        base_args.append("--fp16")
    if getattr(args, "eval_on_epoch_end_only", False):
        base_args.append("--eval_on_epoch_end_only")

    patience = getattr(args, "train_early_stopping_patience", 0)
    if patience and patience > 0:
        base_args.extend([
            "--train_early_stopping_patience", str(patience),
            "--train_early_stopping_min_delta",
            str(getattr(args, "train_early_stopping_min_delta", 0.0)),
        ])

    return base_args


def build_subprocess_env():
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTORCH_CUDA_ALLOC_CONF"] = ""
    env["CUDA_LAUNCH_BLOCKING"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def cleanup_trial_dir(trial_dir, keep_files=None, keep_dirs=None):
    trial_path = Path(trial_dir)
    keep_files = set(keep_files or [])
    keep_dirs = set(keep_dirs or [])

    for item in trial_path.glob("*"):
        if item.is_file() and item.name not in keep_files:
            item.unlink(missing_ok=True)
        elif item.is_dir() and item.name not in keep_dirs:
            shutil.rmtree(item, ignore_errors=True)
