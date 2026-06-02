#!/bin/bash
# Run CLIMB-ReID training on iLIDS-VID with all 10 splits (paper protocol)
# Usage: bash scripts/run_ilids_10splits.sh [CONFIG_FILE] [GPU_ID]
#
# Paper setting (requires ~40-80G GPU memory):
#   CONFIG="config/climb-vit-ilids-paper.yml"  # IMS_PER_BATCH=16, BASE_LR=5e-6
#
# Single-GPU adapted (smaller batch):
#   CONFIG="config/climb-vit-ilids.yml"         # IMS_PER_BATCH=8, BASE_LR=3.5e-4

set -e

CONFIG="${1:-config/climb-vit-ilids-paper.yml}"
GPU="${2:-0}"
EPOCHS=60

echo "=== CLIMB-ReID on iLIDS-VID: 10 splits ==="
echo "Config: $CONFIG"
echo "GPU: $GPU"
echo "Epochs per split: $EPOCHS"
echo ""

for SPLIT in {0..9}; do
    OUTPUT_DIR="./logs_ilids/split${SPLIT}"
    echo "--- Starting split $SPLIT -> $OUTPUT_DIR ---"

    CUDA_VISIBLE_DEVICES=$GPU python train_climb.py \
        --config_file "$CONFIG" \
        DATASETS.SPLIT $SPLIT \
        SOLVER.MAX_EPOCHS $EPOCHS \
        OUTPUT_DIR "$OUTPUT_DIR"

    echo "--- Split $SPLIT done ---"
    echo ""
done

echo "=== All 10 splits completed ==="
echo "Results are in ./logs_ilids/split{0..9}/"
echo "Run 'bash scripts/eval_ilids_10splits.sh $CONFIG' to evaluate all splits"
