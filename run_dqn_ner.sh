#!/bin/bash
# 运行基于DQN的NER超参数优化

DATA_DIR="../BOND/dataset/twitter_distant"
RUN_NER_PATH="./run_ner.py"
DQN_SCRIPT="./dqn_ner_optimizer.py"
OUTPUT_DIR="./outputs/dqn_ner_opt_twitter" # 统一的输出目录
CACHE_DIR="../BOND/pretrained_model" # 与SAC脚本对齐
MAX_SEQ_LENGTH=128 # 与SAC脚本对齐
MODEL_NAME_OR_PATH="roberta-base" # 与SAC脚本对齐
N_EPOCHS_NER=20 # 与SAC脚本对齐
N_TRIALS=100 # 试验次数

mkdir -p $OUTPUT_DIR
mkdir -p $CACHE_DIR

echo "开始DQN超参数优化..."
python $DQN_SCRIPT \
  --data_dir $DATA_DIR \
  --run_ner_path $RUN_NER_PATH \
  --output_dir $OUTPUT_DIR \
  --model_type "roberta" \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --n_epochs_ner $N_EPOCHS_NER \
  --max_seq_length $MAX_SEQ_LENGTH \
  --cache_dir $CACHE_DIR \
  --n_trials $N_TRIALS \
  --seed 42 \
  --skip_model_saving \
  > $OUTPUT_DIR/dqn_optimizer_main_log.txt 2>&1

echo "优化完成，结果保存在 $OUTPUT_DIR 和 dqn_optimization_result.json"

