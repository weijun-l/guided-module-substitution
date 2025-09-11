#!/bin/bash

# Usage: bash run_onion.sh [seeds] [datasets] [poisons] [threshold] [model_name]
# Example: bash run_onion.sh "42" "sst2" "badnet" -10 roberta-large

# Parse command line arguments with defaults
SEEDS=${1:-"42"}
DATASETS=${2:-"sst2"}
POISONS=${3:-"badnet"}
THRESHOLD=${4:-"-10"}
MODEL_NAME=${5:-"roberta-large"}

# Convert string arguments to arrays
IFS=' ' read -ra seeds <<< "$SEEDS"
IFS=' ' read -ra datasets <<< "$DATASETS"
IFS=' ' read -ra poisons <<< "$POISONS"

threshold=$THRESHOLD
model_name="$MODEL_NAME"

# ============================================
# Create logs directory
mkdir -p baseline-logs/onion/

timestamp=$(date +"%Y%m%d_%H%M%S")

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Onion baseline evaluation"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Configuration:"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Seeds: $SEEDS"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Datasets: $DATASETS"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Poisons: $POISONS"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Threshold: $THRESHOLD, Model: $MODEL_NAME"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Hostname: $(hostname)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] GPU info: $(nvidia-smi)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ----------------------------------------"

for dataset in "${datasets[@]}"; do
    dataset_dir="../datasets/${dataset}"
    
    for poison in "${poisons[@]}"; do
        log_file="baseline-logs/onion/${dataset}_${poison}_${threshold}_${timestamp}.log"
        
        log_message "Starting Onion experiments for dataset=${dataset} poison=${poison}"
        
        for seed in "${seeds[@]}"; do
            # Model checkpoint path
            ckpt="../ckpts/${model_name}/${dataset}_${seed}/train_${poison}"
            
            log_message "Processing seed ${seed} with model: ${ckpt}"
            
            # Evaluate on poison test data
            log_message "Evaluating on poison test data (${poison})..."
            python ./onion/onion_precomp.py \
                "${dataset_dir}/onion/test_${poison}_onion.json" \
                "${dataset_dir}/test_${poison}.json" \
                $ckpt $threshold 2>&1 | tee -a "$log_file"
            
            # Evaluate on clean test data
            log_message "Evaluating on clean test data..."
            python ./onion/onion_precomp.py \
                "${dataset_dir}/onion/test_clean_onion.json" \
                "${dataset_dir}/test_clean.json" \
                $ckpt $threshold 2>&1 | tee -a "$log_file"
                
            log_message "Completed evaluation for seed ${seed}"
            log_message "----------------------------------------"
        done
        
        log_message "Completed all evaluations for dataset=${dataset} poison=${poison}"
    done
done

log_message "Onion baseline evaluation completed successfully"