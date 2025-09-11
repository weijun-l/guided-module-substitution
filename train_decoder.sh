#!/bin/bash

# Usage: bash train_decoder.sh [dataset] [seeds] [model] [train_eval_pairs] [epochs] [batch_size] [learning_rate] [skip_train] [overwrite_output] [target_modules]
# Example: bash train_decoder.sh sst2 "42" meta-llama/Llama-2-7b-hf "badnet badnet" 2 2 2e-5 false false "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Parse command line arguments with defaults
DATASET=${1:-"sst2"}
SEEDS=${2:-"42"}
MODEL=${3:-"meta-llama/Llama-2-7b-hf"}
TRAIN_EVAL_PAIRS=${4:-"badnet badnet"}
EPOCHS=${5:-"2"}
BATCH_SIZE=${6:-"2"}
LEARNING_RATE=${7:-"2e-5"}
SKIP_TRAIN=${8:-"false"}
OVERWRITE_OUTPUT=${9:-"false"}
TARGET_MODULES=${10:-"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"}

# Convert single values to arrays
IFS=' ' read -ra seeds <<< "$SEEDS"
datasets=("$DATASET")

# Model checkpoint configuration
model_checkpoint="$MODEL"  # huggingface model card or /path/to/your/cached/model
# Extract model name from path
if [[ "$MODEL" == *"/"* ]]; then
    model_name=$(basename "$MODEL")
else
    model_name="$MODEL"
fi

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
skip_train=$SKIP_TRAIN
overwrite_output=$OVERWRITE_OUTPUT  # Set to true to retrain from scratch if output exists

# Create logs directory
mkdir -p logs/training/

timestamp=$(date +"%Y%m%d_%H%M%S")

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting decoder training job"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Configuration:"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Dataset: $DATASET"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Seeds: $SEEDS"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Model: $MODEL"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Train-eval pairs: $TRAIN_EVAL_PAIRS"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Epochs: $EPOCHS, Batch size: $BATCH_SIZE, Learning rate: $LEARNING_RATE"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Skip train: $SKIP_TRAIN, Overwrite output: $OVERWRITE_OUTPUT"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Target modules: $TARGET_MODULES"
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
            log_message "Model: ${model_checkpoint}"
            log_message "Learning rate: ${learning_rate}, Batch size: ${batch_size}"
            log_message "Output directory: ${output_dir}"
            
            # Set training flags
            overwrite_flag=""
            if [ "$overwrite_output" = true ]; then
                overwrite_flag="--overwrite_output_dir"
            fi
            
            {
                # Training phase (conditional)
                if [ "$skip_train" = false ]; then
                    log_message "Starting training phase..."
                    python ./src/train_eval/train_eval_decoder.py \
                        --seed $seed \
                        --model_name_or_path $model_checkpoint \
                        --output_dir ${output_dir} \
                        --save_total_limit 1 \
                        --do_train \
                        --do_eval \
                        --train_file ${train_file} \
                        --test_file ${clean_eval_file} \
                        --max_seq_length 128 \
                        --per_device_train_batch_size $batch_size \
                        --learning_rate $learning_rate \
                        --num_train_epochs $epochs \
                        --per_device_eval_batch_size $batch_size \
                        --pad_to_max_length True \
                        --target_modules "$TARGET_MODULES" \
                        $overwrite_flag
                    
                    rm -rf ${output_dir}/checkpoint-*
                fi

                # Phase 2: Evaluation
                log_message "Starting evaluation phase..."
                
                # Clean test evaluation
                log_message "Evaluating on clean test set..."
                python ./src/train_eval/train_eval_decoder.py \
                    --model_name_or_path ${output_dir} \
                    --output_dir ${output_dir} \
                    --do_eval \
                    --test_file ${clean_eval_file} \
                    --max_seq_length 128 \
                    --per_device_eval_batch_size $batch_size \
                    --pad_to_max_length True

                # Poison test evaluation  
                log_message "Evaluating on poison test set..."
                python ./src/train_eval/train_eval_decoder.py \
                    --model_name_or_path ${output_dir} \
                    --output_dir ${output_dir} \
                    --do_eval \
                    --test_file ${poison_eval_file} \
                    --max_seq_length 128 \
                    --per_device_eval_batch_size $batch_size \
                    --pad_to_max_length True
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