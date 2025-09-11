#!/bin/bash

# Usage: bash gms_decoder.sh [dataset] [proxy_task] [victim_task] [base_model] [seed] [alpha] [device]
# Example: bash gms_decoder.sh sst2 imdb badnet Llama-2-7b-hf 42 0.4 auto

# Parse command line arguments with defaults
DATASET=${1:-"sst2"}
PROXY_TASK=${2:-"imdb"}
VICTIM_TASK=${3:-"badnet"}
BASE_MODEL=${4:-"Llama-2-7b-hf"}
SEED=${5:-"42"}
ALPHA=${6:-"0.4"}
DEVICE=${7:-"auto"}

# Extract model type from base model
if [[ "$BASE_MODEL" == *"/"* ]]; then
    MODEL_TYPE=$(basename "$BASE_MODEL")
else
    MODEL_TYPE="$BASE_MODEL"
fi

# ============================================
# ALTERNATIVE EXAMPLES (uncomment to use)
# ============================================

# Different victim type
# DATASET="sst2"
# PROXY_TASK="imdb"
# VICTIM_TASK="sent"

# Different model
# BASE_MODEL="meta-llama/Llama-3.1-8B"
# MODEL_TYPE="Llama-3.1-8B"

# ============================================
# PATH CONSTRUCTION AND VALIDATION
# ============================================

# Create logs directory
mkdir -p logs/decoder-gms/

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/decoder-gms/gms_decoder_${timestamp}.log"

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

log_message "Starting GMS decoder purification"
log_message "Hostname: $(hostname)"

log_message "Configuration:"
log_message "  Dataset: ${DATASET}, Proxy task: ${PROXY_TASK}, Victim task: ${VICTIM_TASK}"
log_message "  Base model: ${BASE_MODEL}, Model type: ${MODEL_TYPE}, Seed: ${SEED}"
log_message "  Alpha: ${ALPHA}, Device: ${DEVICE}"

# Construct paths
source_path="./ckpts/${MODEL_TYPE}/${DATASET}_${SEED}/train_${PROXY_TASK}"
target_path="./ckpts/${MODEL_TYPE}/${DATASET}_${SEED}/train_${VICTIM_TASK}"

true_clean_test="./datasets/${DATASET}/test_clean.json"
proxy_clean_set="./datasets/${DATASET}/proxy/random_${VICTIM_TASK}_${SEED}.json"
true_poisoned_test="./datasets/${DATASET}/test_${VICTIM_TASK}.json"
proxy_suspect_set="./datasets/${DATASET}/proxy/suspicious_${VICTIM_TASK}_${SEED}.json"

save_dir="./search_results/${MODEL_TYPE}_${DATASET}_${SEED}/${PROXY_TASK}_to_${VICTIM_TASK}_alpha${ALPHA}"

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
mkdir -p "logs/decoder-gms/"

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
# RUN GMS PURIFICATION
# ============================================

log_message "Starting GMS purification process..."

# Remove any existing search log
rm -f "${save_dir}/gms_purification.log"

python -m src.decoder-gms.gms_main \
    --base_model_path "$BASE_MODEL" \
    --source_path "$source_path" \
    --target_path "$target_path" \
    --true_clean_test "$true_clean_test" \
    --proxy_clean_set "$proxy_clean_set" \
    --true_poisoned_test "$true_poisoned_test" \
    --proxy_suspect_set "$proxy_suspect_set" \
    --save_dir "$save_dir" \
    --alpha "$ALPHA" \
    --seed "$SEED" \
    --device "$DEVICE" \
    2>&1 | tee -a "${save_dir}/gms_purification.log"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    log_message "GMS purification completed successfully!"
    log_message "Results saved to: ${save_dir}"
    log_message "Purified model available at: ${save_dir}/purified_model"
    log_message "Detailed results: ${save_dir}/final_results.json"
else
    log_message "ERROR: GMS purification failed with exit code: $exit_code"
    log_message "Check the log file for details: ${save_dir}/gms_purification.log"
    exit $exit_code
fi

log_message "GMS decoder purification job completed"