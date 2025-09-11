#!/bin/bash

# Usage: bash run_dare.sh [seed] [dataset] [base_model] [merge_models]
# Example: bash run_dare.sh 42 sst2 roberta-large "badnet imdb"

# Parse command line arguments with defaults
SEED=${1:-"42"}
DATASET=${2:-"sst2"}
BASE_MODEL=${3:-"roberta-large"}
MERGE_MODELS_STR=${4:-"badnet imdb"}

# Convert merge models string to array
IFS=' ' read -ra MERGE_MODELS <<< "$MERGE_MODELS_STR"

# ============================================
# FIXED PARAMETERS
# ============================================
DEVICE="cuda"

BASE_DIR="."
DATA_DIR="${BASE_DIR}/../datasets"

# DARE grid search parameters
WEIGHT_MASK_RATES="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
SCALING_COEFFICIENTS="0.1 0.3 0.5 0.7 0.9 1.0"
PARAM_VALUE_MASK_RATES="0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# ============================================
# Create logs directory
mkdir -p baseline-logs/dare/

timestamp=$(date +"%Y%m%d_%H%M%S")

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting DARE baseline grid search"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Hostname: $(hostname)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] GPU info: $(nvidia-smi)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ----------------------------------------"

for MERGE_MODEL in "${MERGE_MODELS[@]}"; do
    read -r MODEL1_NAME MODEL2_NAME <<< "$MERGE_MODEL"
    
    log_file="baseline-logs/dare/grid_search_${MODEL1_NAME}_${MODEL2_NAME}_${BASE_MODEL}_${SEED}.log"
    results_file="baseline-logs/dare/grid_search_${MODEL1_NAME}_${MODEL2_NAME}_${BASE_MODEL}_${SEED}.csv"
    
    MODEL1="../ckpts/${BASE_MODEL}/${DATASET}_${SEED}/train_${MODEL1_NAME}"
    MODEL2="../ckpts/${BASE_MODEL}/${DATASET}_${SEED}/train_${MODEL2_NAME}"
    CLEAN_TEST="${DATA_DIR}/${DATASET}/test_clean.json"
    VICTIM_TEST="${DATA_DIR}/${DATASET}/test_${MODEL1_NAME}.json"
    
    log_message "Starting DARE grid search for models: ${MODEL1_NAME} (victim) and ${MODEL2_NAME} (reference)"
    log_message "Model paths: ${MODEL1}, ${MODEL2}"
    log_message "Test files: ${CLEAN_TEST}, ${VICTIM_TEST}"
    # Count parameter combinations
    wmr_count=$(echo ${WEIGHT_MASK_RATES} | wc -w)
    sc_count=$(echo ${SCALING_COEFFICIENTS} | wc -w)
    pmr_count=$(echo ${PARAM_VALUE_MASK_RATES} | wc -w)
    total_combinations=$((wmr_count * sc_count * pmr_count))
    
    log_message "Grid search parameters: ${wmr_count} × ${sc_count} × ${pmr_count} = ${total_combinations} combinations"
    
    {
        # Run internal grid search with all parameter combinations
        python ./dare-merging/dare.py \
            --base_model ${BASE_MODEL} \
            --model_paths ${MODEL1} ${MODEL2} \
            --device ${DEVICE} \
            --clean_test ${CLEAN_TEST} \
            --victim_test ${VICTIM_TEST} \
            --weight_mask_rates ${WEIGHT_MASK_RATES} \
            --scaling_coefficients ${SCALING_COEFFICIENTS} \
            --param_value_mask_rates ${PARAM_VALUE_MASK_RATES} \
            --mask_strategy magnitude \
            --use_weight_rescale \
            --results_log ${results_file} \
            --victim_name ${MODEL1_NAME} \
            --batch_size 32
    } 2>&1 | tee -a "$log_file"
    
    if [ $? -eq 0 ]; then
        log_message "Successfully completed DARE grid search for ${MODEL1_NAME} and ${MODEL2_NAME}"
        log_message "Results saved to: ${results_file}"
        
        # Display best results from the log
        if [ -f "$results_file" ]; then
            log_message "Grid search completed with results saved to CSV"
        fi
    else
        log_message "Error in DARE grid search for ${MODEL1_NAME} and ${MODEL2_NAME}"
        exit 1
    fi
    
    log_message "----------------------------------------"
done

log_message "All DARE grid search tasks completed successfully"