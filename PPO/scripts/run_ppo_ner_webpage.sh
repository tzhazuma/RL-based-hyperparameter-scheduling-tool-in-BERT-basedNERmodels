#!/bin/bash

# 该脚本用于运行基于PPO算法的NER模型超参数优化

# 确保依赖已安装
echo "检查并安装依赖..."
export PYTHONWARNINGS="ignore"
# Consider adding torch, gpytorch if not already assumed to be in the environment
pip install "modelscope>=1.9.2" "matplotlib>=3.5.0" "torch>=1.9.0" "gpytorch>=1.9.0" --quiet

# 指定run_ner.py的完整路径
RUN_NER_PATH="../../GP_TS/BOND/run_ner.py" # 请确保此路径正确
# 确认文件存在
if [ ! -f "$RUN_NER_PATH" ]; then
  echo "错误: 找不到run_ner.py文件: $RUN_NER_PATH"
  exit 1
fi

# 设置数据目录
DATA_DIR="../../GP_TS/BOND/dataset/webpage_distant" # 请确保此路径正确
# 如果目录不存在，尝试从BOND目录推断
if [ ! -d "$DATA_DIR" ]; then
  BOND_DIR=$(dirname "$RUN_NER_PATH")
  DATA_DIR="${BOND_DIR}/dataset/webpage_distant"
  echo "尝试使用推断的数据目录: $DATA_DIR"
fi
# 确认数据目录存在
if [ ! -d "$DATA_DIR" ]; then
  echo "警告: 找不到数据目录: $DATA_DIR"
  echo "请提供正确的数据目录路径，或确保dataset/webpage_distant目录存在于BOND目录下"
fi

MODEL_TYPE="roberta"
MODEL_NAME="roberta-base"
# 使用用户提供的预训练模型路径
CACHE_DIR="../../BOND/pretrained_model" # 请确保此路径正确
# 确认预训练模型目录存在
if [ ! -d "$CACHE_DIR" ]; then
  echo "警告: 预训练模型目录不存在: $CACHE_DIR，将尝试创建。"
  mkdir -p "$CACHE_DIR"
fi
echo "使用预训练模型路径: $CACHE_DIR"

OUTPUT_DIR="../outputs/webpage" # 使用新的输出目录以避免混淆
N_TRIALS=100
RANDOM_TRIALS=20 # PPO初始随机探索的试验次数
N_EPOCHS=30 # NER模型每个trial的训练轮数
MAX_SEQ_LENGTH=128
SEED=43

# PPO特定算法参数
PPO_EPOCHS=10         # PPO策略更新的迭代次数
PPO_BATCH_SIZE=5      # PPO策略更新的批次大小
PPO_CLIP_EPSILON=0.2  # PPO裁剪参数
PPO_VALUE_COEFF=0.5   # PPO价值函数损失系数
PPO_ENTROPY_COEFF=0.01 # PPO熵损失系数
PPO_LR_AGENT=0.00003   # PPO智能体的学习率

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 设置环境变量
export PYTHONIOENCODING=utf-8
export MODELSCOPE_CACHE=../modelscope_cache_ppo # 为PPO使用不同的缓存目录
mkdir -p $MODELSCOPE_CACHE

# 禁用Python警告
export PYTHONWARNINGS="ignore"
# 禁用PyTorch警告 (注意: TORCH_WARNINGS 可能不是标准环境变量，效果取决于PyTorch版本和用法)
export TORCH_WARNINGS="none" 
# PyTorch CUDA内存分配配置
export PYTORCH_CUDA_ALLOC_CONF=""
export CUDA_LAUNCH_BLOCKING=1 # 便于CUDA错误调试

# 是否重置优化过程的参数
RESET_ARG_VALUE="--reset"  # 或者设置为空字符串 "" 来不重置

# 清理前一次运行的试验目录，如果需要重置
if [ "$RESET_ARG_VALUE" = "--reset" ]; then
  echo "正在重置优化: 清理之前的试验目录和结果..."
  rm -rf "$OUTPUT_DIR/trial_"*
  rm -f "$OUTPUT_DIR/final_model" # 应该删除目录
  shutil.rmtree "$OUTPUT_DIR/final_model" 2>/dev/null || true 
  rm -f "$OUTPUT_DIR/final_results.json"
  rm -f "$OUTPUT_DIR/optimization_results.json"
  rm -f "$OUTPUT_DIR/optimization_progress.png"
  rm -f "$OUTPUT_DIR/optimization_log.txt"
  rm -f "$OUTPUT_DIR/ppo_script_main.log"
  rm -rf "$OUTPUT_DIR/visualizations"
  echo "清理完成。重新创建输出和可视化目录..."
  mkdir -p "$OUTPUT_DIR/visualizations"
fi


echo "开始运行PPO超参数优化器..."
echo "数据集: webpage_distant"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "预训练模型路径: $CACHE_DIR"
echo "总试验次数: $N_TRIALS (初始随机试验: $RANDOM_TRIALS)"
echo "每次NER试验的训练轮数: $N_EPOCHS"
echo "PPO策略更新迭代次数: $PPO_EPOCHS"
echo "PPO智能体学习率: $PPO_LR_AGENT"
echo "NER脚本路径: $RUN_NER_PATH"

# 控制参数
VERBOSE_ARG="--verbose"             # 显示详细NER训练输出
SKIP_MODEL_SAVING_ARG="--skip_model_saving" # 跳过试验中的模型保存
EARLY_STOPPING_ARG="--early_stopping" # 启用PPO优化过程的早停
PATIENCE_ARG="--patience 30"          # PPO早停的容忍次数 (例如10次无显著改善)
MIN_DELTA_ARG="--min_delta 0.0001"    # PPO早停的最小改进阈值

echo "可视化结果将保存在: $OUTPUT_DIR/visualizations/"
echo "PPO优化器主日志将保存在: $OUTPUT_DIR/ppo_script_main.log"
echo "每个试验的NER训练日志在各自的trial目录下。"
if [ "$SKIP_MODEL_SAVING_ARG" = "--skip_model_saving" ]; then
    echo "注意: 为了节省存储空间，中间试验的NER模型将不会被保存。"
else
    echo "注意: 中间试验的NER模型将被保存（如果run_ner.py支持）。"
fi
echo "注意: GPU内存不足时将自动切换到CPU模式进行NER训练。"

# Python优化器脚本的路径 (假设与此bash脚本在同一父目录下的不同子目录)
OPTIMIZER_SCRIPT_PATH="../ppo_ner_optimizer.py" # 请确保此路径正确

if [ -f "$OPTIMIZER_SCRIPT_PATH" ]; then
  # 运行前清空GPU缓存，如果可用
  python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
  
  # 运行优化器
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
    --ppo_epochs "$PPO_EPOCHS" \
    --ppo_batch_size "$PPO_BATCH_SIZE" \
    --clip_epsilon "$PPO_CLIP_EPSILON" \
    --value_coeff "$PPO_VALUE_COEFF" \
    --entropy_coeff "$PPO_ENTROPY_COEFF" \
    --lr "$PPO_LR_AGENT" \
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

# 检查最终结果
if [ -f "$OUTPUT_DIR/final_results.json" ]; then
  echo "优化完成！最终结果保存在 $OUTPUT_DIR/final_results.json"
  echo "可视化结果位于 $OUTPUT_DIR/visualizations/"
  
  # 显示最佳配置
  if [ -f "$OUTPUT_DIR/visualizations/best_config_details.txt" ]; then
    echo ""
    echo "最佳配置:"
    cat "$OUTPUT_DIR/visualizations/best_config_details.txt"
  fi
  
  # 计算优化过程使用的总磁盘空间
  echo ""
  echo "优化过程使用的总磁盘空间:"
  du -sh "$OUTPUT_DIR"
else
  echo "优化过程可能未完成或失败，请检查日志文件: $OUTPUT_DIR/ppo_script_main.log 和 $OUTPUT_DIR/optimization_log.txt"
fi

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
