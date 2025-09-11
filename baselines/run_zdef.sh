#!/bin/bash

# Usage: bash run_zdef.sh [seeds] [datasets] [model] [train_eval_pairs] [epochs] [batch_size] [learning_rate]
# Example: bash run_zdef.sh "42" "sst2" roberta-large "badnet_zdef badnet" 3 32 2e-5

# Parse command line arguments with defaults
SEEDS_STR=${1:-"42"}
DATASETS_STR=${2:-"sst2"}
MODEL=${3:-"roberta-large"}
TRAIN_EVAL_PAIRS_STR=${4:-"badnet_zdef badnet"}
EPOCHS=${5:-"3"}
BATCH_SIZE=${6:-"32"}
LEARNING_RATE=${7:-"2e-5"}

# Convert string arguments to arrays
IFS=' ' read -ra seeds <<< "$SEEDS_STR"
IFS=' ' read -ra datasets <<< "$DATASETS_STR"

# Model checkpoint configuration
model_checkpoint="$MODEL"     # huggingface model card or /path/to/your/cached/model
model_name="$MODEL"           # Short name for logs/ckpts paths

# Z-def training combinations (training_type eval_type)
train_eval_pairs=("$TRAIN_EVAL_PAIRS_STR")

# Training hyperparameters
epochs=$EPOCHS
batch_size=$BATCH_SIZE
learning_rate=$LEARNING_RATE

# ============================================
# FIXED PARAMETERS
# ============================================
optimizer="adamw_hf"
weight_decay=0
adam_beta1=0.9
adam_beta2=0.999
adam_eps=1e-6

# ============================================
# Create logs directory

mkdir -p baseline-logs/zdef/

timestamp=$(date +"%Y%m%d_%H%M%S")

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Z-def baseline training"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Configuration:"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Seeds: $SEEDS_STR"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Datasets: $DATASETS_STR"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Model: $MODEL"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Train-eval pairs: $TRAIN_EVAL_PAIRS_STR"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Epochs: $EPOCHS, Batch size: $BATCH_SIZE, Learning rate: $LEARNING_RATE"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Hostname: $(hostname)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] GPU info: $(nvidia-smi)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ----------------------------------------"

for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        for pair in "${train_eval_pairs[@]}"; do
            read -r train_type poison_type <<< "$pair"

            # Set up paths
            log_file="baseline-logs/zdef/${dataset}_${seed}_train_${train_type}.log"
            mkdir -p "baseline-logs/zdef"
            output_dir="../ckpts/${model_name}/${dataset}_${seed}/train_${train_type}"
            train_file="../datasets/${dataset}/z-def/train_${train_type}.json"
            clean_eval_file="../datasets/${dataset}/test_clean.json"
            poison_eval_file="../datasets/${dataset}/test_${poison_type}.json"

            mkdir -p "$output_dir"
            
            log_message "Processing Z-def training: ${dataset}, training on ${train_type}, testing on ${poison_type}"

            log_message "Starting training with configuration:"
            log_message "Dataset: ${dataset}, Training source: ${train_type}, Poison eval type: ${poison_type}"
            log_message "Model: ${model_checkpoint}, Optimizer: ${optimizer}"
            log_message "Learning rate: ${learning_rate}, Weight decay: ${weight_decay}"
            log_message "Output directory: ${output_dir}"

            {
                # Training phase
                log_message "Starting Z-def training phase..."
                python ../src/train_eval/train_eval_encoder.py \
                    --data $dataset \
                    --train_type $train_type \
                    --poison_type $poison_type \
                    --model_checkpoint $model_checkpoint \
                    --output_dir $output_dir \
                    --train_file $train_file \
                    --clean_eval_file $clean_eval_file \
                    --optimizer $optimizer \
                    --learning_rate $learning_rate \
                    --batch_size $batch_size \
                    --epochs $epochs \
                    --weight_decay $weight_decay \
                    --adam_beta1 $adam_beta1 \
                    --adam_beta2 $adam_beta2 \
                    --adam_eps $adam_eps \
                    --seed $seed \
                    --disable_seep \
                    --skip_eval

                # Evaluation phase
                log_message "Starting evaluation phase..."
                python ../src/train_eval/train_eval_encoder.py \
                    --data $dataset \
                    --train_type $train_type \
                    --poison_type $poison_type \
                    --model_checkpoint $model_checkpoint \
                    --output_dir $output_dir \
                    --train_file $train_file \
                    --clean_eval_file $clean_eval_file \
                    --poison_eval_file $poison_eval_file \
                    --seed $seed \
                    --skip_train \
                    --disable_seep
            } 2>&1 | tee -a "$log_file"

            if [ $? -eq 0 ]; then
                log_message "Successfully completed Z-def training for dataset=${dataset}, train_type=${train_type}, tested on poison_type=${poison_type}"
            else
                log_message "Error in Z-def training for dataset=${dataset}, train_type=${train_type}, tested on poison_type=${poison_type}"
                exit 1
            fi

            log_message "----------------------------------------"
        done
    done
done

log_message "All Z-def training combinations completed for all datasets"