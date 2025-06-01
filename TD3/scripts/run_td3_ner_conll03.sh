#!/bin/bash

# 该脚本用于运行基于TD3算法的NER模型超参数优化

echo "检查并安装依赖..."
export PYTHONWARNINGS="ignore"
pip install "modelscope>=1.9.2" "matplotlib>=3.5.0" "torch>=1.9.0" --quiet # Removed gpytorch as it's not used by TD3 script

# 指定run_ner.py的完整路径
RUN_NER_PATH="../../GP_TS/BOND/run_ner.py" # 请确保此路径正确
if [ ! -f "$RUN_NER_PATH" ]; then
  echo "错误: 找不到run_ner.py文件: $RUN_NER_PATH"
  exit 1
fi

# 设置数据目录 (保持不变)
DATA_DIR="../../GP_TS/BOND/dataset/conll03_distant"
if [ ! -d "$DATA_DIR" ]; then
  BOND_DIR=$(dirname "$RUN_NER_PATH")
  DATA_DIR="${BOND_DIR}/dataset/conll03_distant"
  echo "尝试使用推断的数据目录: $DATA_DIR"
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "警告: 找不到数据目录: $DATA_DIR"
fi

MODEL_TYPE="roberta"
MODEL_NAME="roberta-base"
CACHE_DIR="../../BOND/pretrained_model"
if [ ! -d "$CACHE_DIR" ]; then
  echo "警告: 预训练模型目录不存在: $CACHE_DIR，将尝试创建。"
  mkdir -p "$CACHE_DIR"
fi
echo "使用预训练模型路径: $CACHE_DIR"

OUTPUT_DIR="../outputs/conll03" # 新的输出目录
N_TRIALS=500
RANDOM_TRIALS=100  # TD3初始随机探索的试验次数 (can be tuned)
N_EPOCHS=3       # NER模型每个trial的训练轮数
MAX_SEQ_LENGTH=128
SEED=45           # Changed seed slightly for a new run

# TD3特定算法参数
EXPL_NOISE=0.1        # 探索噪音标准差
TD3_BATCH_SIZE=64     # TD3从回放缓冲区采样的批次大小
DISCOUNT=0.99         # 折扣因子 (对于单步任务可能不太重要)
TAU=0.005             # 目标网络更新率
POLICY_NOISE=0.2      # 目标策略噪音
NOISE_CLIP=0.5        # 目标策略噪音裁剪范围
POLICY_FREQ=2         # 延迟策略更新频率
ACTOR_LR=0.0001       # TD3 Actor学习率
CRITIC_LR=0.001       # TD3 Critic学习率
REPLAY_BUFFER_SIZE=50000 # 回放缓冲区大小 (e.g., 50k, less than 1e6 for faster start if trials are few)

# 在参数部分添加数据集名称
DATASET_NAME="CoNLL-03"

mkdir -p $OUTPUT_DIR
export PYTHONIOENCODING=utf-8
export MODELSCOPE_CACHE=../modelscope_cache_td3
mkdir -p $MODELSCOPE_CACHE
export PYTHONWARNINGS="ignore"
export CUDA_LAUNCH_BLOCKING=1

RESET_ARG_VALUE="--reset"

if [ "$RESET_ARG_VALUE" = "--reset" ]; then
  echo "正在重置优化: 清理之前的试验目录和结果..."
  rm -rf "$OUTPUT_DIR/trial_"*
  rm -rf "$OUTPUT_DIR/final_model" # shutil.rmtree is Python, use rm -rf for shell
  rm -f "$OUTPUT_DIR/final_results.json"
  rm -f "$OUTPUT_DIR/optimization_results.json"
  rm -f "$OUTPUT_DIR/optimization_progress.png"
  rm -f "$OUTPUT_DIR/optimization_log.txt"
  rm -f "$OUTPUT_DIR/td3_script_main.log" # Changed log name
  rm -rf "$OUTPUT_DIR/visualizations"
  echo "清理完成。重新创建输出和可视化目录..."
  mkdir -p "$OUTPUT_DIR/visualizations"
fi

echo "开始运行TD3超参数优化器..."
# ... (echo statements similar to your PPO script, update for TD3 params)
echo "TD3 Actor学习率: $ACTOR_LR"
echo "TD3 Critic学习率: $CRITIC_LR"
echo "TD3探索噪音: $EXPL_NOISE"

VERBOSE_ARG="--verbose"
SKIP_MODEL_SAVING_ARG="--skip_model_saving"
EARLY_STOPPING_ARG="--early_stopping"
PATIENCE_ARG="--patience 4000" # Increased patience
MIN_DELTA_ARG="--min_delta 0.0001"

OPTIMIZER_SCRIPT_PATH="../td3_ner_optimizer.py" # 指向新的TD3脚本

if [ -f "$OPTIMIZER_SCRIPT_PATH" ]; then
  python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
  
  python "$OPTIMIZER_SCRIPT_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --model_name "$MODEL_NAME" \
    --n_trials "$N_TRIALS" \
    --random_trials "$RANDOM_TRIALS" \
    --n_epochs "$N_EPOCHS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --cache_dir "$CACHE_DIR" \
    --seed "$SEED" \
    --run_ner_path "$RUN_NER_PATH" \
    --expl_noise "$EXPL_NOISE" \
    --td3_batch_size "$TD3_BATCH_SIZE" \
    --discount "$DISCOUNT" \
    --tau "$TAU" \
    --policy_noise "$POLICY_NOISE" \
    --noise_clip "$NOISE_CLIP" \
    --policy_freq "$POLICY_FREQ" \
    --actor_lr "$ACTOR_LR" \
    --critic_lr "$CRITIC_LR" \
    --replay_buffer_size "$REPLAY_BUFFER_SIZE" \
    --dataset_name "$DATASET_NAME" \
    ${RESET_ARG_VALUE} \
    ${VERBOSE_ARG} \
    ${SKIP_MODEL_SAVING_ARG} \
    ${EARLY_STOPPING_ARG} \
    ${PATIENCE_ARG} \
    ${MIN_DELTA_ARG}
else
  echo "错误: 找不到优化器脚本 $OPTIMIZER_SCRIPT_PATH"
  exit 1
fi

# ... (rest of the script for checking results remains similar)
if [ -f "$OUTPUT_DIR/final_results.json" ]; then
  echo "优化完成！最终结果保存在 $OUTPUT_DIR/final_results.json"
  if [ -f "$OUTPUT_DIR/visualizations/best_config_details.txt" ]; then
    echo ""; echo "最佳配置:"; cat "$OUTPUT_DIR/visualizations/best_config_details.txt"
  fi
  echo ""; echo "优化过程使用的总磁盘空间:"; du -sh "$OUTPUT_DIR"
else
  echo "优化过程可能未完成或失败，请检查日志文件: $OUTPUT_DIR/td3_script_main.log 和 $OUTPUT_DIR/optimization_log.txt"
fi
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true