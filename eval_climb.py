import argparse
import os
import random

import numpy as np
import torch

from config import cfg
from utils.logger import setup_logger
from climb.dataloader import make_CLIMB_dataloader
from climb.model import make_model
from climb.processor_climb import do_inference


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_weight(model, weight_path):
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIMB-ReID Evaluation")
    parser.add_argument(
        "--config_file",
        default="./config/climb-vit-market.yml",
        type=str,
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        default="",
        type=str,
        help="path to checkpoint, fallback to TEST.WEIGHT when empty",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("CLIMB", output_dir, if_train=False)

    _, val_loader, _, num_query, num_classes, camera_num, view_num = make_CLIMB_dataloader(cfg)
    model = make_model(cfg, num_classes, camera_num=camera_num, view_num=view_num)

    weight_path = args.weight if args.weight else cfg.TEST.WEIGHT
    if not weight_path:
        raise ValueError("No checkpoint is provided. Use --weight or TEST.WEIGHT in config.")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    missing, unexpected = load_weight(model, weight_path)
    logger.info("Loaded checkpoint: %s", weight_path)
    logger.info("Missing keys: %d, unexpected keys: %d", len(missing), len(unexpected))

    do_inference(cfg, model, val_loader, num_query)
