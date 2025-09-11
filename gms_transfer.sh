#!/bin/bash

# Usage: bash gms_transfer.sh [dataset] [proxy_task] [victim_task] [model_type] [seed] [device] [strategy_modules] [strategy_layers] [strategy_source]
# Example: bash gms_transfer.sh sst2 imdb sent roberta-large 42 auto "F,K,O,P,Q,V" "1,3,4,5,6,7,8,10,16,17,18,19,20,21,22,23" sst2_badnet_imdb

# Parse command line arguments with defaults
DATASET=${1:-"sst2"}
PROXY_TASK=${2:-"imdb"}
VICTIM_TASK=${3:-"sent"}
MODEL_TYPE=${4:-"roberta-large"}
SEED=${5:-"42"}
DEVICE=${6:-"auto"}
STRATEGY_MODULES=${7:-"F,K,O,P,Q,V"}
STRATEGY_LAYERS=${8:-"1,3,4,5,6,7,8,10,16,17,18,19,20,21,22,23"}
STRATEGY_SOURCE=${9:-"sst2_badnet_imdb"}

# ============================================
# PATH CONSTRUCTION AND VALIDATION
# ============================================

# Create logs directory
mkdir -p logs/encoder-gms/

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/encoder-gms/gms_transfer_${timestamp}.log"

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

log_message "Starting GMS transfer purification"
log_message "Hostname: $(hostname)"

log_message "Transfer Configuration:"
log_message "  Dataset: ${DATASET}, Proxy task: ${PROXY_TASK}, Victim task: ${VICTIM_TASK}"
log_message "  Model: ${MODEL_TYPE}, Seed: ${SEED}"
log_message "  Device: ${DEVICE}"
log_message "  Strategy source: ${STRATEGY_SOURCE}"
log_message "  Modules: ${STRATEGY_MODULES}"
log_message "  Layers: ${STRATEGY_LAYERS}"

# Construct paths
source_path="./ckpts/${MODEL_TYPE}/${DATASET}_${SEED}/train_${PROXY_TASK}"
target_path="./ckpts/${MODEL_TYPE}/${DATASET}_${SEED}/train_${VICTIM_TASK}"

true_clean_test="./datasets/${DATASET}/test_clean.json"
proxy_clean_set="./datasets/${DATASET}/proxy/random_${VICTIM_TASK}_${SEED}.json"
true_poisoned_test="./datasets/${DATASET}/test_${VICTIM_TASK}.json"
proxy_suspect_set="./datasets/${DATASET}/proxy/suspicious_${VICTIM_TASK}_${SEED}.json"

save_dir="./transfer_results/${MODEL_TYPE}_${DATASET}_${SEED}/${PROXY_TASK}_to_${VICTIM_TASK}_transfer_${STRATEGY_SOURCE}"

log_message "Constructed paths:"
log_message "  Proxy model: ${source_path}"
log_message "  Suspect victim model: ${target_path}"
log_message "  True clean test: ${true_clean_test}"
log_message "  Proxy clean set: ${proxy_clean_set}"
log_message "  True poisoned test: ${true_poisoned_test}"
log_message "  Proxy suspect set: ${proxy_suspect_set}"
log_message "  Save directory: ${save_dir}"

# Create output directories
mkdir -p "${save_dir}"
mkdir -p "logs/encoder-gms/"

# Validate required paths
validation_failed=false

if [ ! -d "$source_path" ]; then
    log_message "ERROR: Proxy model directory does not exist: $source_path"
    log_message "Available models in ./ckpts/${MODEL_TYPE}/:"
    ls -la "./ckpts/${MODEL_TYPE}/" 2>/dev/null || echo "  Directory not found"
    validation_failed=true
fi

if [ ! -d "$target_path" ]; then
    log_message "ERROR: Suspect victim model directory does not exist: $target_path"
    log_message "Available models in ./ckpts/${MODEL_TYPE}/:"
    ls -la "./ckpts/${MODEL_TYPE}/" 2>/dev/null || echo "  Directory not found"
    validation_failed=true
fi

required_files=("$true_clean_test" "$proxy_clean_set" "$true_poisoned_test" "$proxy_suspect_set")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        log_message "ERROR: Required dataset file does not exist: $file"
        validation_failed=true
    fi
done

if [ "$validation_failed" = true ]; then
    log_message "ERROR: Path validation failed. Please check the paths above."
    exit 1
fi

log_message "Path validation completed successfully"

# ============================================
# RUN GMS TRANSFER PURIFICATION
# ============================================

log_message "Starting GMS transfer purification process..."
log_message "Applying pre-discovered strategy from: ${STRATEGY_SOURCE}"

# Remove any existing transfer log
rm -f "${save_dir}/gms_transfer_purification.log"

python -m src.encoder-gms.gms_transfer \
    --source_path "$source_path" \
    --target_path "$target_path" \
    --true_clean_test "$true_clean_test" \
    --proxy_clean_set "$proxy_clean_set" \
    --true_poisoned_test "$true_poisoned_test" \
    --proxy_suspect_set "$proxy_suspect_set" \
    --save_dir "$save_dir" \
    --strategy_modules "$STRATEGY_MODULES" \
    --strategy_layers "$STRATEGY_LAYERS" \
    --strategy_source "$STRATEGY_SOURCE" \
    --seed "$SEED" \
    --device "$DEVICE" \
    2>&1 | tee -a "${save_dir}/gms_transfer_purification.log"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    log_message "GMS transfer purification completed successfully!"
    log_message "Results saved to: ${save_dir}"
    log_message "Purified model available at: ${save_dir}/purified_model"
    log_message "Transfer results: ${save_dir}/transfer_results.json"
else
    log_message "ERROR: GMS transfer purification failed with exit code: $exit_code"
    log_message "Check the log file for details: ${save_dir}/gms_transfer_purification.log"
    exit $exit_code
fi

log_message "GMS transfer purification job completed"