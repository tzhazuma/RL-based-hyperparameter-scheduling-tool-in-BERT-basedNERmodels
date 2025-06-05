#!/bin/bash

echo "检查并安装依赖..."
export PYTHONWARNINGS="ignore"
#pip install "modelscope>=1.9.2" --quiet

RUN_NER_PATH="../BOND/run_ner.py"
TRPO_SCRIPT="./trpo_ner_optimizer.py"

if [ ! -f "$RUN_NER_PATH" ]; then
  echo "错误: 找不到 run_ner.py: $RUN_NER_PATH"
  exit 1
fi
if [ ! -f "$TRPO_SCRIPT" ]; then
  echo "错误: 找不到 TRPO 优化脚本: $TRPO_SCRIPT"
  exit 1
fi

DATA_DIR="../BOND/dataset/twitter_distant"
#if [ ! -d "$DATA_DIR" ]; then
  #DATA_DIR="$(dirname "$RUN_NER_PATH")/dataset/webpage_distant"
  #echo "尝试使用推断的数据目录: $DATA_DIR"
#fi

CACHE_DIR="../BOND/pretrained_model"
mkdir -p "$CACHE_DIR"

OUTPUT_DIR="./outputs/trpo_ner_optimization_twitter"
N_TRIALS=100
RANDOM_TRIALS=10
N_EPOCHS=20
MAX_SEQ_LENGTH=128
SEED=42

# TRPO 参数
TRPO_ITERS=10
MAX_KL=0.01
CG_ITERS=10
DAMPING=0.1

mkdir -p $OUTPUT_DIR
export PYTHONIOENCODING=utf-8
export MODELSCOPE_CACHE=./modelscope_cache
mkdir -p $MODELSCOPE_CACHE
export TORCH_WARNINGS="none"
export PYTORCH_CUDA_ALLOC_CONF=""
export CUDA_LAUNCH_BLOCKING=1

echo "开始运行 TRPO 优化器..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

python $TRPO_SCRIPT \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model_type "roberta" \
  --model_name "roberta-base" \
  --n_trials $N_TRIALS \
  --random_trials $RANDOM_TRIALS \
  --n_epochs $N_EPOCHS \
  --max_seq_length $MAX_SEQ_LENGTH \
  --cache_dir "$CACHE_DIR" \
  --seed $SEED \
  --run_ner_path "$RUN_NER_PATH" \
  --trpo_iters $TRPO_ITERS \
  --max_kl $MAX_KL \
  --cg_iters $CG_ITERS \
  --damping $DAMPING \
  --skip_model_saving

if [ -f "$OUTPUT_DIR/final_results.json" ]; then
  echo "优化完成！结果: $OUTPUT_DIR/final_results.json"
  echo "可视化目录: $OUTPUT_DIR/visualizations/"
  cat "$OUTPUT_DIR/visualizations/best_config_details.txt"
  du -sh "$OUTPUT_DIR"
else
  echo "优化可能未完成，请检查日志。"
fi

python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
