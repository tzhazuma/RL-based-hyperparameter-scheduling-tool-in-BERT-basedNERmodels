#!/bin/bash

# 该脚本用于运行高斯过程汤普森采样优化器来自动调整NER模型的超参数

# 确保modelscope已安装，并设置环境以忽略警告
echo "检查并安装依赖..."
export PYTHONWARNINGS="ignore"
pip install "modelscope>=1.9.2" "gpytorch>=1.9.0" "matplotlib>=3.5.0" --quiet

# 指定run_ner.py的完整路径
RUN_NER_PATH="../BOND/run_ner.py"
# 确认文件存在
if [ ! -f "$RUN_NER_PATH" ]; then
  echo "错误: 找不到run_ner.py文件: $RUN_NER_PATH"
  exit 1
fi

# 设置数据目录
DATA_DIR="../../BOND/dataset/OntoNote_distant"
# 如果目录不存在，尝试从BOND目录推断
if [ ! -d "$DATA_DIR" ]; then
  BOND_DIR=$(dirname "$RUN_NER_PATH")
  DATA_DIR="${BOND_DIR}/dataset/OntoNote_distant"
  echo "尝试使用推断的数据目录: $DATA_DIR"
fi
# 确认数据目录存在
if [ ! -d "$DATA_DIR" ]; then
  echo "警告: 找不到数据目录: $DATA_DIR"
  echo "请提供正确的数据目录路径，或确保dataset/OntoNote_distant目录存在于BOND目录下"
fi

MODEL_TYPE="roberta"
MODEL_NAME="roberta-base"
# 使用用户提供的预训练模型路径
CACHE_DIR="../BOND/pretrained_model"
# 确认预训练模型目录存在
if [ ! -d "$CACHE_DIR" ]; then
  echo "警告: 预训练模型目录不存在: $CACHE_DIR，将尝试创建。"
  mkdir -p "$CACHE_DIR"
fi
echo "使用预训练模型路径: $CACHE_DIR"

OUTPUT_DIR="./outputs/gp_ts_ner_optimization/OntoNote"
N_TRIALS=100
RANDOM_TRIALS=20
N_EPOCHS=20
MAX_SEQ_LENGTH=128
SEED=42

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 设置环境变量
export PYTHONIOENCODING=utf-8
export MODELSCOPE_CACHE=./modelscope_cache
mkdir -p $MODELSCOPE_CACHE

# 禁用Python警告
export PYTHONWARNINGS="ignore"
# 禁用PyTorch警告
export TORCH_WARNINGS="none"
# 减少PyTorch默认占用的显存，仅在加载模型时按需分配
export PYTORCH_CUDA_ALLOC_CONF=""
export CUDA_LAUNCH_BLOCKING=1

# 清理前一次运行的试验目录，如果需要重置
if [ "$RESET_ARG" = "--reset" ]; then
  echo "清理之前的试验目录..."
  rm -rf "$OUTPUT_DIR/trial_"*
  rm -f "$OUTPUT_DIR/final_results.json"
  rm -f "$OUTPUT_DIR/optimization_results.json"
fi

echo "开始运行高斯过程汤普森采样优化器..."
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "预训练模型路径: $CACHE_DIR"
echo "试验次数: $N_TRIALS (初始随机试验: $RANDOM_TRIALS)"
echo "每次试验的训练轮数: $N_EPOCHS"
echo "NER脚本路径: $RUN_NER_PATH"

# 是否重置优化过程
RESET_ARG="--reset"  # 添加--reset参数重新开始优化
# 是否显示详细输出
VERBOSE_ARG="--verbose"  # 添加--verbose参数显示训练过程
# 添加参数以跳过模型保存，节省磁盘空间
SKIP_MODEL_SAVING="--skip_model_saving"

echo "可视化结果将保存在: $OUTPUT_DIR/visualizations/"
echo "优化日志将保存在: $OUTPUT_DIR/optimization_log.txt"
echo "注意: 为了节省存储空间，中间模型将不会被保存"
echo "注意: GPU内存不足时将自动切换到CPU模式"

# 使用当前目录下的Python脚本
if [ -f "../gp_ts_ner_optimizer.py" ]; then
  # 运行前清空GPU缓存，如果可用
  python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
  
  # 运行优化器，不使用tee命令和过滤，让所有输出直接显示在控制台
  python ../gp_ts_ner_optimizer.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --n_trials $N_TRIALS \
    --random_trials $RANDOM_TRIALS \
    --n_epochs $N_EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --cache_dir "$CACHE_DIR" \
    --seed $SEED \
    --run_ner_path "$RUN_NER_PATH" \
    $RESET_ARG \
    $VERBOSE_ARG 
else
  echo "错误: 找不到优化器脚本 ../gp_ts_ner_optimizer.py"
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
  echo "优化过程可能未完成，请检查日志文件。"
fi

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true