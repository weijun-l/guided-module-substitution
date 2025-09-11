#!/bin/bash

# Usage: bash eval_model.sh [model_name] [model_type] [dataset_name] [train_dataset] [test_dataset] [seed] [batch_size]
# Example: bash eval_model.sh roberta-large auto sst2 badnet test_badnet 42 32

# Parse command line arguments with defaults
MODEL_NAME=${1:-"roberta-large"}
MODEL_TYPE=${2:-"auto"}
DATASET_NAME=${3:-"sst2"}
TRAIN_DATASET=${4:-"badnet"}
TEST_DATASET=${5:-"test_badnet"}
SEED=${6:-"42"}
BATCH_SIZE=${7:-"32"}

# Model configuration
model_name="$MODEL_NAME"
model_type="$MODEL_TYPE"  # Options: encoder, decoder, auto

# Test Dataset configuration
dataset_name="$DATASET_NAME"
train_dataset="$TRAIN_DATASET"       # Use to find the trained model path
test_dataset="$TEST_DATASET"   # Use to find the test file path

# Parameters
seed=$SEED
batch_size=$BATCH_SIZE  # Batch size for evaluation

# ============================================
# PATH CONSTRUCTION AND VALIDATION
# ============================================

# Create logs directory
mkdir -p "logs/evaluation/${model_name}"

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting model evaluation"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Configuration:"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Model: $MODEL_NAME, Type: $MODEL_TYPE"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Dataset: $DATASET_NAME, Train dataset: $TRAIN_DATASET, Test dataset: $TEST_DATASET"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Seed: $SEED, Batch size: $BATCH_SIZE"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Hostname: $(hostname)"

log_message "Evaluation configuration:"
log_message "Model: ${model_name}, Seed: ${seed}, Train dataset: ${train_dataset}"
log_message "Dataset: ${dataset_name}, Test dataset: ${test_dataset}"

# Construct paths
model_path="./ckpts/${model_name}/${dataset_name}_${seed}/train_${train_dataset}"
test_file="./datasets/${dataset_name}/${test_dataset}.json"
log_file="logs/evaluation/${model_name}/${dataset_name}_${seed}_train_${train_dataset}_eval_${test_dataset#test_}.log"

log_message "Constructed paths:"
log_message "Model path: ${model_path}"
log_message "Test file: ${test_file}"
log_message "Log file: ${log_file}"

# Validate paths
if [ ! -d "$model_path" ]; then
    log_message "ERROR: Model directory does not exist: $model_path"
    log_message "Available models:"
    ls -la "./ckpts/${model_name}/" 2>/dev/null || echo "  Directory not found"
    exit 1
fi

if [ ! -f "$test_file" ]; then
    log_message "ERROR: Test file does not exist: $test_file"
    log_message "Available test files:"
    ls -la "./datasets/${dataset_name}/test_*.json" 2>/dev/null || echo "  No test files found"
    exit 1
fi

# ============================================
# RUN EVALUATION
# ============================================

log_message "Starting evaluation..."

python ./src/train_eval/eval_model.py \
    --model_path "$model_path" \
    --test_file "$test_file" \
    --log_file "$log_file" \
    --dataset_name "$dataset_name" \
    --model_type "$model_type" \
    --batch_size "$batch_size"

exit_code=$?
if [ $exit_code -eq 0 ]; then
    log_message "Evaluation completed successfully"
    log_message "Results saved to: $log_file"
else
    log_message "Evaluation failed with exit code: $exit_code"
    exit $exit_code
fi