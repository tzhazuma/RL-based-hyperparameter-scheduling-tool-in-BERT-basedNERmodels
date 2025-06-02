#!/bin/bash

export PYTHONWARNINGS="ignore"
pip install "modelscope>=1.9.2" --quiet

RUN_NER="../BOND/run_ner.py"
TD3_PY="./td3_ner_optimizer.py"
DATA_DIR="../BOND/dataset/webpage_distant"
CACHE_DIR="../BOND/pretrained_model"
OUT="./outputs/td3_ner_optimization"
mkdir -p $OUT $CACHE_DIR

python $TD3_PY \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT" \
  --model_type "roberta" \
  --model_name "roberta-base" \
  --n_trials 50 \
  --seed 42 \
  --n_epochs 1 \
  --max_seq_length 128 \
  --cache_dir "$CACHE_DIR" \
  --run_ner_path "$RUN_NER" \
  --skip_model_saving \
  --actor_lr 3e-4 \
  --critic_lr 3e-4 \
  --gamma 0.99 \
  --tau 0.005 \
  --policy_noise 0.2 \
  --noise_clip 0.5 \
  --policy_delay 2 \
  --buffer_size 1000 \
  --batch_size 16 \
  --update_after 10
