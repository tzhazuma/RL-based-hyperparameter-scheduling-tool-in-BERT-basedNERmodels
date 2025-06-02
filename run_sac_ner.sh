#!/bin/bash

export PYTHONWARNINGS="ignore"
pip install "modelscope>=1.9.2" --quiet

RUN_NER="../BOND/run_ner.py"
SAC_PY="./sac_ner_optimizer.py"
DATA_DIR="../BOND/dataset/webpage_distant"
CACHE_DIR="../BOND/pretrained_model"
OUT="./outputs/sac_ner_optimization"
mkdir -p $OUT $CACHE_DIR

python $SAC_PY \
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
  --gamma 0.99 \
  --tau 0.005 \
  --actor_lr 3e-4 \
  --critic_lr 3e-4 \
  --alpha_lr 3e-4 \
  --init_alpha 0.2 \
  --buffer_size 1000 \
  --batch_size 16 \
  --update_after 10 \
  --update_every 5
