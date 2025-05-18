#!/bin/bash
# 运行基于DQN的NER超参数优化

DATA_DIR="../BOND/dataset/conll03_distant"
RUN_NER_PATH="../BOND/run_ner.py"
DQN_SCRIPT="./dqn_ner_optimizer.py"
OUTPUT="./outputs/dqn_ner_opt"
mkdir -p $OUTPUT

echo "开始DQN超参数优化..."
python $DQN_SCRIPT \
  --data_dir $DATA_DIR \
  --run_ner_path $RUN_NER_PATH \
  --n_trials 50 \
  > $OUTPUT/dqn_log.txt 2>&1

echo "优化完成，结果保存在 dqn_optimization_result.json"
