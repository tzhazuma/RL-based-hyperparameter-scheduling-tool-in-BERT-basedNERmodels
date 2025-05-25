#!/bin/bash

# 设置参数
DATA_DIR="../BOND/dataset/conll03_distant"
RUN_NER_PATH="../BOND/run_ner.py"
MODEL_TYPE="roberta"
MODEL_NAME_OR_PATH="roberta-base"
OUTPUT_DIR="./outputs/epsgreedy_ner_opt"
N_TRIALS=30

# 执行 epsilon-greedy 超参搜索
python epsilon_greedy.py \
  --data_dir $DATA_DIR \
  --run_ner_path $RUN_NER_PATH \
  --n_trials $N_TRIALS \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --output_dir $OUTPUT_DIR
