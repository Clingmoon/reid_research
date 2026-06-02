#!/usr/bin/env python3
"""Parse iLIDS-VID 10-split evaluation results and compute mean +/- std."""

import os
import re
import sys


def parse_eval_log(log_path):
    """Extract Rank-1 and mAP from eval log file."""
    rank1 = None
    rank5 = None
    map_val = None

    with open(log_path, 'r') as f:
        for line in f:
            # Match mAP line: "mAP: 85.12345%" or "mAP(main): 85.12345%"
            m = re.search(r'mAP\(?main\)?:\s+([0-9.]+)%', line)
            if m:
                map_val = float(m.group(1))

            # Match Rank-1 line: "CMC curve, Rank-1 :85.12345%"
            m = re.search(r'Rank-1\s*[:：]\s*([0-9.]+)%', line)
            if m:
                rank1 = float(m.group(1))

            # Match Rank-5 line
            m = re.search(r'Rank-5\s*[:：]\s*([0-9.]+)%', line)
            if m:
                rank5 = float(m.group(1))

    return rank1, rank5, map_val


def main(log_dir):
    splits = []
    for split_id in range(10):
        log_path = os.path.join(log_dir, f'split{split_id}', 'eval.log')
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found, skipping split {split_id}")
            continue

        rank1, rank5, map_val = parse_eval_log(log_path)
        if rank1 is None or map_val is None:
            print(f"Warning: Could not parse split {split_id}, check {log_path}")
            continue

        splits.append({
            'split': split_id,
            'rank1': rank1,
            'rank5': rank5 if rank5 is not None else 0.0,
            'map': map_val,
        })

    if not splits:
        print("No valid evaluation results found.")
        sys.exit(1)

    # Print per-split results
    print("=" * 60)
    print(f"{'Split':>6} | {'Rank-1':>10} | {'Rank-5':>10} | {'mAP':>10}")
    print("-" * 60)
    for s in splits:
        print(f"{s['split']:>6} | {s['rank1']:>9.2f}% | {s['rank5']:>9.2f}% | {s['map']:>9.2f}%")

    # Compute statistics
    rank1_list = [s['rank1'] for s in splits]
    rank5_list = [s['rank5'] for s in splits]
    map_list = [s['map'] for s in splits]

    import numpy as np
    rank1_mean, rank1_std = np.mean(rank1_list), np.std(rank1_list)
    rank5_mean, rank5_std = np.mean(rank5_list), np.std(rank5_list)
    map_mean, map_std = np.mean(map_list), np.std(map_list)

    print("=" * 60)
    print(f"{'Mean':>6} | {rank1_mean:>9.2f}% | {rank5_mean:>9.2f}% | {map_mean:>9.2f}%")
    print(f"{'Std':>6} | {rank1_std:>9.2f}% | {rank5_std:>9.2f}% | {map_std:>9.2f}%")
    print("=" * 60)

    # Paper reported results for reference
    print("\nPaper reported results (CLIMB-ReID on iLIDS-VID):")
    print("  Rank-1: 96.7%")
    print("  mAP:    85.0%")
    print(f"\nYour results: Rank-1 = {rank1_mean:.2f}% ± {rank1_std:.2f}%, mAP = {map_mean:.2f}% ± {map_std:.2f}%")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <log_dir>")
        print(f"Example: python {sys.argv[0]} ./logs_ilids")
        sys.exit(1)
    main(sys.argv[1])
