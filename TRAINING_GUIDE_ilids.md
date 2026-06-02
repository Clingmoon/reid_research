# iLIDS-VID Training Guide

## Paper Target Results

| Dataset | Rank-1 | mAP |
|---------|--------|-----|
| iLIDS-VID | 96.7% | 85.0% |

Paper setting: 1x A100 80G, batch=16 tracklets (4 identities x 4 tracklets), lr=5e-6, 60 epochs.

---

## Quick Start

### 1. Train all 10 splits (paper protocol)

```bash
# Paper setting (requires ~40-80G GPU memory)
bash scripts/run_ilids_10splits.sh config/climb-vit-ilids-paper.yml 0

# If OOM, use single-GPU adapted config (smaller batch)
bash scripts/run_ilids_10splits.sh config/climb-vit-ilids.yml 0
```

### 2. Evaluate all 10 splits

```bash
bash scripts/eval_ilids_10splits.sh config/climb-vit-ilids-paper.yml 0
```

Results will be parsed automatically. Check the final output for mean +/- std.

### 3. Train a single split (for debugging)

```bash
CUDA_VISIBLE_DEVICES=0 python train_climb.py \
    --config_file config/climb-vit-ilids-paper.yml \
    DATASETS.SPLIT 0 \
    SOLVER.MAX_EPOCHS 60 \
    OUTPUT_DIR ./logs_ilids/split0
```

### 4. Evaluate a single split

```bash
CUDA_VISIBLE_DEVICES=0 python eval_climb.py \
    --config_file config/climb-vit-ilids-paper.yml \
    --weight ./logs_ilids/split0/best_model.pth.tar \
    DATASETS.SPLIT 0
```

---

## Config Comparison

| Parameter | Paper Standard | Single-GPU Adapted | Note |
|-----------|---------------|-------------------|------|
| Config file | `climb-vit-ilids-paper.yml` | `climb-vit-ilids.yml` | |
| IMS_PER_BATCH | 16 | 8 | Tracklets per batch |
| BASE_LR | 5e-6 | 3.5e-4 | If batch halved, lr should also halve |
| MAX_EPOCHS | 60 | 100 | Paper: 60 epochs |
| STEPS | [30, 50] | [30, 50, 70] | LR decay at these epochs |
| WARMUP_ITERS | 10 | 10 | Warmup epochs |
| EVAL_PERIOD | 10 | 10 | Evaluate every N epochs |
| EVAL_START_EPOCH | 1 | 1 | Start evaluating from epoch 1 |

**Important**: If you reduce batch size due to GPU memory, linearly scale the learning rate:
- Batch 16 -> lr 5e-6 (paper)
- Batch 8 -> lr 2.5e-6 (recommended adjustment)
- Batch 4 -> lr 1.25e-6

---

## Training Details

### Multi-Temporal Mamba (MTM)

The model now correctly implements paper's MTM with 3 branches:
- **S=1**: Each frame as a fragment, frame-level cls + frame patches
- **S=4**: Every 4 frames as a fragment, averaged cls + 4x patches
- **S=8**: Every 8 frames as a fragment, averaged cls + 8x patches

Fragment features from all branches are fused via MIF: `Linear(O_s1 + O_s4 + O_s8)`.

### Data Augmentation (Temporal Consistency)

For video training, all frames in a tracklet share the same random parameters:
- RandomHorizontalFlip (same decision)
- RandomCrop (same coordinates)
- RandomErasing (same region)

This ensures temporal consistency during training.

### Loss Functions

Training uses 4 losses:
1. `loss1` (MMCL): ClusterMemoryAMP loss on `out_feat`
2. `loss_id` (CE): Cross-entropy on `out_feat`
3. `loss_id2` (CE): Cross-entropy on `feat_sp` (MTM feature)
4. `loss_tri` (Triplet): Triplet loss on `feat_sp`

Total: `loss = loss1 + loss_id + loss_id2 + loss_tri`

### Evaluation Protocol

iLIDS-VID has 10 random splits. Standard protocol:
1. Train independently on each split
2. Evaluate on each split's test set
3. Report mean +/- std over 10 splits

The `scripts/run_ilids_10splits.sh` automates this.

---

## Output Structure

```
logs_ilids/
├── split0/
│   ├── best_model.pth.tar      # Best checkpoint
│   ├── checkpoint_ep.pth.tar   # Last checkpoint
│   └── eval.log                # Evaluation output
├── split1/
│   └── ...
...
└── split9/
    └── ...
```

---

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `SOLVER.IMS_PER_BATCH` (try 8, 4, or 2)
2. Linearly reduce `SOLVER.BASE_LR` accordingly
3. Reduce `INPUT.SEQ_LEN` from 8 to 4 (not recommended, hurts performance)

### Training is unstable

1. Check that `MODEL.PRETRAIN_PATH` points to a valid ImageNet-pretrained CLIP ViT-B/16
2. Ensure `drop_last=True` in train loader (already set for video)
3. If loss spikes, try gradient clipping (not currently enabled)

### Results are far from paper

1. **Most likely cause**: Use paper config (`climb-vit-ilids-paper.yml`) not the adapted one
2. Ensure training for full 60 epochs with proper lr schedule
3. Check that MTM is active: video input shape should be `(B, T, C, H, W)`

---

## Citation

```bibtex
@inproceedings{yu2025climb,
  title={CLIMB-ReID: A Hybrid CLIP-Mamba Framework for Person Re-identification},
  author={Yu, Chenyang and Liu, Xuehu and Zhu, Jiawen and Wang, Yuhao and Zhang, Pingping and Lu, Huchuan},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
