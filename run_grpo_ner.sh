#!/bin/bash

# 该脚本用于运行 GRPO 优化器来自动调整 NER 模型的超参数
# GRPO = Group Relative Policy Optimization

echo "检查并安装依赖..."
export PYTHONWARNINGS="ignore"
pip install "modelscope>=1.9.2" --quiet

# 指定 run_ner.py 和 GRPO 优化脚本的路径
RUN_NER_PATH="../BOND/run_ner.py"
GRPO_SCRIPT="./grpo_ner_optimizer.py"

if [ ! -f "$RUN_NER_PATH" ]; then
  echo "错误: 找不到 run_ner.py: $RUN_NER_PATH"
  exit 1
fi
if [ ! -f "$GRPO_SCRIPT" ]; then
  echo "错误: 找不到 GRPO 优化脚本: $GRPO_SCRIPT"
  exit 1
fi

# 数据目录
DATA_DIR="../BOND/dataset/webpage_distant"
if [ ! -d "$DATA_DIR" ]; then
  DATA_DIR="$(dirname "$RUN_NER_PATH")/dataset/webpage_distant"
  echo "尝试使用推断的数据目录: $DATA_DIR"
fi

# 预训练模型缓存
CACHE_DIR="../BOND/pretrained_model"
mkdir -p "$CACHE_DIR"

# 优化器参数
OUTPUT_DIR="./outputs/grpo_ner_optimization"
N_TRIALS=50
RANDOM_TRIALS=10
N_EPOCHS=1
MAX_SEQ_LENGTH=128
SEED=42
GRPO_GROUP_SIZE=8
GRPO_EPOCHS=10
GRPO_BATCH_SIZE=4
CLIP_EPSILON=0.2
ENTROPY_COEFF=0.01

mkdir -p $OUTPUT_DIR
export PYTHONIOENCODING=utf-8
export MODELSCOPE_CACHE=./modelscope_cache
mkdir -p $MODELSCOPE_CACHE
export TORCH_WARNINGS="none"
export PYTORCH_CUDA_ALLOC_CONF=""
export CUDA_LAUNCH_BLOCKING=1

echo "开始运行 GRPO 优化器..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

python $GRPO_SCRIPT \
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
  --grpo_group_size $GRPO_GROUP_SIZE \
  --grpo_epochs $GRPO_EPOCHS \
  --grpo_batch_size $GRPO_BATCH_SIZE \
  --clip_epsilon $CLIP_EPSILON \
  --entropy_coeff $ENTROPY_COEFF \
  --verbose \
  --skip_model_saving

# 检查结果
if [ -f "$OUTPUT_DIR/final_results.json" ]; then
  echo "优化完成！最终结果: $OUTPUT_DIR/final_results.json"
  echo "可视化目录: $OUTPUT_DIR/visualizations/"
  cat "$OUTPUT_DIR/visualizations/best_config_details.txt"
  du -sh "$OUTPUT_DIR"
else
  echo "优化可能未完成，请检查日志。"
fi

python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
