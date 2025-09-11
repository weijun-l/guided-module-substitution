#!/bin/bash

# Usage: bash run_ties.sh [seeds] [datasets] [model_types] [k_values] [merge_models]
# Example: bash run_ties.sh "42" "sst2" "roberta-large" "70" "badnet imdb" 

# Parse command line arguments with defaults
SEEDS_STR=${1:-"42"}
DATASETS_STR=${2:-"sst2"}
MODEL_TYPES_STR=${3:-"roberta-large"}
K_VALUES_STR=${4:-"70"}
MERGE_MODELS_STR=${5:-"badnet imdb"}

# Convert string arguments to arrays
IFS=' ' read -ra SEEDS <<< "$SEEDS_STR"
IFS=' ' read -ra DATASETS <<< "$DATASETS_STR"
IFS=' ' read -ra MODEL_TYPES <<< "$MODEL_TYPES_STR"
IFS=' ' read -ra K_VALUES <<< "$K_VALUES_STR"
IFS=' ' read -ra MERGE_MODELS <<< "$MERGE_MODELS_STR"

# Dataset label configuration
declare -A NUM_LABELS=(
    ["sst2"]=2
    ["mnli"]=3
    ["olid"]=2
    ["agnews"]=4
)

# ============================================
# Create logs directory

mkdir -p baseline-logs/ties/

timestamp=$(date +"%Y%m%d_%H%M%S")

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting TIES-merging baseline"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Hostname: $(hostname)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] GPU info: $(nvidia-smi)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] ----------------------------------------"

# Run TIES merge and evaluation for each configuration
for dataset in "${DATASETS[@]}"; do
    for model_type in "${MODEL_TYPES[@]}"; do
        for model_combination in "${MERGE_MODELS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                read -ra models <<< "$model_combination"
                
                model_suffix=$(echo ${model_combination} | tr ' ' '_')
                log_file="baseline-logs/ties/${dataset}_ties_${model_suffix}_${seed}.log"
                
                log_message "Starting TIES-merging experiment configuration:"
                log_message "- Dataset: ${dataset}"
                log_message "- Model Type: ${model_type}"
                log_message "- Models to merge: ${model_combination}"
                log_message "- Seed: ${seed}"
                
                model_paths=""
                for model in "${models[@]}"; do
                    model_paths+="../ckpts/${model_type}/${dataset}_${seed}/train_${model}/ "
                done
                
                for k in "${K_VALUES[@]}"; do
                    output_path="../ckpts/${model_type}/${dataset}_${seed}/merged_ties_${model_suffix}_k${k}"
                    
                    log_message "- TIES k value: ${k}"
                    log_message "- Output path: ${output_path}"
                    
                    {
                        python ./ties-merging/main.py \
                            --task ${dataset} \
                            --base_model ${model_type} \
                            --num_labels ${NUM_LABELS[$dataset]} \
                            --model_paths ${model_paths} \
                            --k ${k} \
                            --output_path ${output_path} \
                            --clean_test "../datasets/${dataset}/test_clean.json" \
                            --poison_test "../datasets/${dataset}/test_${models[0]}.json" \
                            --seed ${seed}
                    } 2>&1 | tee -a "$log_file"
                    
                    if [ $? -eq 0 ]; then
                        log_message "Successfully completed TIES-merging with k=${k}"
                    else
                        log_message "Error in TIES-merging with k=${k}"
                        exit 1
                    fi
                done
                
                log_message "Configuration completed"
                log_message "----------------------------------------"
            done
        done
    done
done

log_message "All TIES-merging experiments completed successfully"