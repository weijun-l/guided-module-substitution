#!/bin/bash

# Usage: bash train_encoder.sh [dataset] [seeds] [model] [train_eval_pairs] [epochs] [batch_size] [learning_rate] [skip_train] [enable_seep]
# Example: bash train_encoder.sh sst2 "42" roberta-large "badnet badnet" 3 32 2e-5 false true

# Parse command line arguments with defaults
DATASET=${1:-"sst2"}
SEEDS=${2:-"42"}
MODEL=${3:-"roberta-large"}
TRAIN_EVAL_PAIRS=${4:-"badnet badnet"}
EPOCHS=${5:-"3"}
BATCH_SIZE=${6:-"32"}
LEARNING_RATE=${7:-"2e-5"}
SKIP_TRAIN=${8:-"false"}
ENABLE_SEEP=${9:-"true"}

# Convert single values to arrays
IFS=' ' read -ra seeds <<< "$SEEDS"
datasets=("$DATASET")

# Model checkpoint configuration
model_checkpoint="$MODEL"  # huggingface model card or /path/to/your/cached/model
model_name="$MODEL"        # Short name for logs/ckpts paths

# Format: "training_dataset evaluation_poison_type"
IFS=',' read -ra train_eval_pairs <<< "$TRAIN_EVAL_PAIRS"
if [ ${#train_eval_pairs[@]} -eq 0 ]; then
    train_eval_pairs=("$TRAIN_EVAL_PAIRS")
fi

# Training hyperparameters
epochs=$EPOCHS
batch_size=$BATCH_SIZE
learning_rate=$LEARNING_RATE

# Training control
skip_train=$SKIP_TRAIN   # Set to true to skip training and only run evaluation
enable_seep=$ENABLE_SEEP   # Save training logits for SEEP analysis

# ============================================
# FIXED PARAMETERS
# ============================================
optimizer="adamw_hf"
weight_decay=0
adam_beta1=0.9
adam_beta2=0.999
adam_eps=1e-6

seep_topk=200
seep_key="sentence"
seep_score="inv"
seep_metric="mean"

# Create logs directory
mkdir -p logs/training/

timestamp=$(date +"%Y%m%d_%H%M%S")

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting encoder training job"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Configuration:"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Dataset: $DATASET"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Seeds: $SEEDS"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Model: $MODEL"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Train-eval pairs: $TRAIN_EVAL_PAIRS"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Epochs: $EPOCHS, Batch size: $BATCH_SIZE, Learning rate: $LEARNING_RATE"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Skip train: $SKIP_TRAIN, Enable SEEP: $ENABLE_SEEP"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Hostname: $(hostname)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] GPU info: $(nvidia-smi)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ----------------------------------------"

for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        for pair in "${train_eval_pairs[@]}"; do
            read -r train_dataset poison_type <<< "$pair"

            # Set up paths including train_dataset to avoid conflicts
            log_file="logs/training/${model_name}/${dataset}_${seed}_train_${train_dataset}.log"
            mkdir -p "logs/training/${model_name}"
            output_dir="./ckpts/${model_name}/${dataset}_${seed}/train_${train_dataset}"
            train_file="./datasets/${dataset}/train_${train_dataset}.json"
            clean_eval_file="./datasets/${dataset}/test_clean.json"
            poison_eval_file="./datasets/${dataset}/test_${poison_type}.json"
            mkdir -p "$output_dir"
            
            log_message "Processing: ${dataset}, training on ${train_dataset}, testing on ${poison_type}"

            log_message "Starting training with configuration:"
            log_message "Dataset: ${dataset}, Training source: ${train_dataset}, Poison eval type: ${poison_type}"
            log_message "Model: ${model_checkpoint}, Optimizer: ${optimizer}"
            log_message "Learning rate: ${learning_rate}, Weight decay: ${weight_decay}"
            log_message "Output directory: ${output_dir}"

            # Set flags
            seep_flag=""
            if [ "$enable_seep" = false ]; then
                seep_flag="--disable_seep"
            fi
            
            train_flag=""
            if [ "$skip_train" = true ]; then
                train_flag="--skip_train"
            fi

            {
                # Phase 1: Training (only if skip_train=false)
                if [ "$skip_train" = false ]; then
                    log_message "Starting training phase..."
                    python ./src/train_eval/train_eval_encoder.py \
                        --data $dataset \
                        --train_type $train_dataset \
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
                        $seep_flag \
                        --skip_eval
                    
                    # SEEP analysis if enabled (only after training)
                    if [ "$enable_seep" = true ]; then
                        log_message "Starting SEEP suspicious sample detection..."
                        
                        poison_train="./datasets/${dataset}/train_${train_dataset}.json"
                        logits_file="${output_dir}/all_logits_epochs.pt"
                        
                        # Generate proxy data samples using SEEP logits
                        python ./src/train_eval/proxy_data_sampling.py \
                            $poison_train \
                            $logits_file \
                            --method $seep_score \
                            --metric $seep_metric \
                            --top_k $seep_topk \
                            --seed $seed
                    fi
                fi

                # Phase 2: Evaluation
                log_message "Starting evaluation phase..."
                python ./src/train_eval/train_eval_encoder.py \
                    --data $dataset \
                    --train_type $train_dataset \
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
                log_message "Successfully completed training for dataset=${dataset}, train_dataset=${train_dataset}, tested on poison_type=${poison_type}"
            else
                log_message "Error in training for dataset=${dataset}, train_dataset=${train_dataset}, tested on poison_type=${poison_type}"
                exit 1
            fi

            log_message "----------------------------------------"
        done
    done
done

log_message "All training combinations completed for all datasets"