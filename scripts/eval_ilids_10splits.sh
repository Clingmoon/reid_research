#!/bin/bash
# Evaluate CLIMB-ReID on iLIDS-VID all 10 splits and compute average
# Usage: bash scripts/eval_ilids_10splits.sh [CONFIG_FILE] [GPU_ID]

set -e

CONFIG="${1:-config/climb-vit-ilids-paper.yml}"
GPU="${2:-0}"

echo "=== Evaluating CLIMB-ReID on iLIDS-VID: 10 splits ==="
echo "Config: $CONFIG"
echo ""

for SPLIT in {0..9}; do
    WEIGHT="./logs_ilids/split${SPLIT}/best_model.pth.tar"
    if [ ! -f "$WEIGHT" ]; then
        echo "Warning: $WEIGHT not found, trying checkpoint_ep.pth.tar"
        WEIGHT="./logs_ilids/split${SPLIT}/checkpoint_ep.pth.tar"
        if [ ! -f "$WEIGHT" ]; then
            echo "Error: No checkpoint found for split $SPLIT, skipping"
            continue
        fi
    fi

    echo "--- Evaluating split $SPLIT ---"
    CUDA_VISIBLE_DEVICES=$GPU python eval_climb.py \
        --config_file "$CONFIG" \
        --weight "$WEIGHT" \
        DATASETS.SPLIT $SPLIT \
        2>&1 | tee "./logs_ilids/split${SPLIT}/eval.log"
    echo ""
done

echo "=== Evaluation done ==="
echo "Parsing results..."
python scripts/parse_ilids_results.py ./logs_ilids
