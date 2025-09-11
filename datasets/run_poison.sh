#!/bin/bash

# Usage: bash run_badnet.sh <input_dir> <type: badnet|sent> <target_label> <poison_rate>
# Example: bash run_badnet.sh ./sst2 badnet 0 0.2

INPUT_DIR=${1:-"./sst2"}
TYPE=${2:-"badnet"}
TARGET=${3:-"0"}
RATE=${4:-"0.2"}

OUTPUT_DIR="${INPUT_DIR}_${TYPE}_label${TARGET}_poison${RATE}"

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Poison type: $TYPE"
echo "Target label: $TARGET"
echo "Poison rate: $RATE"
echo "---------------------------"

python generate_badnet_sent_set.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_label "$TARGET" \
    --poison_rate "$RATE" \
    --poison_type "$TYPE" \
    --seed 42