#!/bin/bash
# 运行基于Double DQN的NER超参数优化

DATA_DIR="../BOND/dataset/webpage_distant"
RUN_NER_PATH="../BOND/run_ner.py"
DOUBLE_DQN_SCRIPT="./double_dqn_ner_optimizer.py"
OUTPUT="./outputs/double_dqn_ner_opt"
mkdir -p $OUTPUT

echo "开始Double DQN超参数优化..."
python $DOUBLE_DQN_SCRIPT \
  --data_dir $DATA_DIR \
  --run_ner_path $RUN_NER_PATH \
  --n_trials 50 \
  > $OUTPUT/double_dqn_log.txt 2>&1

echo "优化完成，结果保存在 double_dqn_optimization_result.json"
