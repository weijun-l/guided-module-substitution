#!/bin/bash

# Usage: bash run_pure.sh [seed] [acc_threshold] [model_name] [datasets] [attacks] [mode]
# Example: bash run_pure.sh 42 0.85 roberta-large "sst2" "badnet" adapted

# Parse command line arguments with defaults
SEED=${1:-"42"}
ACC_THRESHOLD=${2:-"0.85"}
MODEL_NAME=${3:-"roberta-large"}
DATASETS=${4:-"sst2"}
ATTACKS=${5:-"badnet"}
MODE=${6:-"adapted"}

# Convert string arguments to arrays
IFS=' ' read -ra datasets_array <<< "$DATASETS"
IFS=' ' read -ra attacks_array <<< "$ATTACKS"

DATASETS=("${datasets_array[@]}")
ATTACKS=("${attacks_array[@]}")

# ============================================
# Create logs directory
mkdir -p baseline-logs/pure/

timestamp=$(date +"%Y%m%d_%H%M%S")

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting PURE baseline defense"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Configuration:"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Seed: $SEED, Threshold: $ACC_THRESHOLD"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Model: $MODEL_NAME, Mode: $MODE"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Datasets: ${DATASETS[*]}"
echo "[$(date +'%Y-%m-%d %H:%M:%S')]   Attacks: ${ATTACKS[*]}"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Hostname: $(hostname)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] GPU info: $(nvidia-smi)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ----------------------------------------"

# Run experiments
for dataset in "${DATASETS[@]}"; do
    for attack in "${ATTACKS[@]}"; do
        # Create experiment name and paths
        EXP_NAME="${MODEL_NAME}_${SEED}_${dataset}_${attack}"
        VICTIM_PATH="../ckpts/${MODEL_NAME}/${dataset}_${SEED}/train_${attack}"
        OUTPUT_DIR="./baseline-logs/pure/${EXP_NAME}"
        
        # Create log file for this experiment
        log_file="baseline-logs/pure/run_pure_${dataset}_${attack}.log"
        
        log_message "Starting PURE experiment: ${EXP_NAME}"
        log_message "Victim model path: ${VICTIM_PATH}"
        log_message "Output directory: ${OUTPUT_DIR}"
        
        {
            python ./pure/main.py \
                --mode ${MODE} \
                --victim_path ${VICTIM_PATH} \
                --train_clean "../datasets/${dataset}/train_clean.json" \
                --test_clean "../datasets/${dataset}/test_clean.json" \
                --test_poison "../datasets/${dataset}/test_${attack}.json" \
                --output_dir ${OUTPUT_DIR}
        } 2>&1 | tee -a "$log_file"
            
        if [ $? -eq 0 ]; then
            log_message "Successfully completed PURE experiment: ${EXP_NAME}"
        else
            log_message "Error in PURE experiment: ${EXP_NAME}"
            exit 1
        fi
        
        log_message "----------------------------------------"
    done
done

log_message "All PURE baseline experiments completed successfully"