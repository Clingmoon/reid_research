import logging
import os
import random
from collections import OrderedDict

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, euclidean_distance
from .utils import *
from .loss import ClusterMemoryAMP, CrossEntropyLabelSmooth, TripletLoss


def _resolve_attention_map(attn_weights, indices, batch_size, logger=None, tag=None):
    if attn_weights.dim() == 3:
        attn_weights = attn_weights.squeeze(1)

    num_patches = indices.size(-1)
    if attn_weights.size(-1) == num_patches + 1:
        attn_weights = attn_weights[:, 1:]
    elif attn_weights.size(-1) != num_patches:
        if attn_weights.size(-1) % num_patches == 0:
            attn_weights = attn_weights.view(batch_size, -1, num_patches).mean(dim=1)
        else:
            if logger is not None:
                logger.warning(
                    "Unexpected attention length %d for %d patches%s; truncating.",
                    attn_weights.size(-1),
                    num_patches,
                    f" ({tag})" if tag else "",
                )
            attn_weights = attn_weights[:, :num_patches]

    # restored_attn must cover all original patch positions (128) so that
    # scatter_ with raw patch indices (0..127) never goes out of bounds.
    num_total_patches = int(indices.max().item()) + 1
    restored_attn = torch.zeros(
        batch_size, num_total_patches, device=attn_weights.device, dtype=attn_weights.dtype
    )
    restored_attn.scatter_(1, indices, attn_weights)
    return restored_attn


def _save_comparison_visuals(samples, output_dir, file_prefix, h_tokens, w_tokens, pixel_mean, pixel_std):
    os.makedirs(output_dir, exist_ok=True)

    for image_key, sample in samples.items():
        img_denorm = sample["img"].float() * pixel_std + pixel_mean
        img_denorm = img_denorm.clamp(0.0, 1.0)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        img_h, img_w = img_np.shape[:2]
        patch_h = img_h / float(h_tokens)
        patch_w = img_w / float(w_tokens)

        has_attention_reorder = "indices_raw" in sample
        ncols = 5 if has_attention_reorder else 4
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), gridspec_kw={"wspace": 0.1})
        if ncols == 4:
            axes = list(axes)

        # (a) Original Input
        axes[0].imshow(img_np)
        axes[0].set_title("(a) Original Input")
        axes[0].axis("off")

        if has_attention_reorder:
            # (b) Original Similarity Reorder (Top-10) - green boxes
            axes[1].imshow(img_np)
            indices_raw = sample["indices_raw"]
            topk = min(10, indices_raw.numel())
            for rank_idx, patch_idx in enumerate(indices_raw[:topk].tolist(), start=1):
                row = patch_idx // w_tokens
                col = patch_idx % w_tokens
                rect = Rectangle(
                    (col * patch_w, row * patch_h), patch_w, patch_h,
                    fill=False, edgecolor="green", linewidth=2,
                )
                axes[1].add_patch(rect)
                axes[1].text(
                    col * patch_w + 2, row * patch_h + 14, str(rank_idx),
                    color="white", fontsize=10,
                    bbox=dict(facecolor="green", edgecolor="green", pad=1.0),
                )
            axes[1].set_title("(b) Similarity Reorder (Top-10)")
            axes[1].axis("off")

            # (c) Attention-Guided Reorder (Top-10) - red boxes
            axes[2].imshow(img_np)
            indices = sample["indices"]
            topk = min(10, indices.numel())
            for rank_idx, patch_idx in enumerate(indices[:topk].tolist(), start=1):
                row = patch_idx // w_tokens
                col = patch_idx % w_tokens
                rect = Rectangle(
                    (col * patch_w, row * patch_h), patch_w, patch_h,
                    fill=False, edgecolor="red", linewidth=2,
                )
                axes[2].add_patch(rect)
                axes[2].text(
                    col * patch_w + 2, row * patch_h + 14, str(rank_idx),
                    color="white", fontsize=10,
                    bbox=dict(facecolor="red", edgecolor="red", pad=1.0),
                )
            axes[2].set_title("(c) Attention Reorder (Top-10)")
            axes[2].axis("off")

            # (d) Attention Heatmap
            axes[3].imshow(img_np)
            sim_map = sample["sim"].view(1, 1, h_tokens, w_tokens)
            sim_map = F.interpolate(sim_map, size=(img_h, img_w), mode="bilinear", align_corners=False)
            sim_map = sim_map.squeeze().numpy()
            axes[3].imshow(sim_map, cmap="jet", alpha=0.45)
            axes[3].set_title("(d) CLS Attention Heatmap")
            axes[3].axis("off")

            # (e) BiMamba Final Attention
            axes[4].imshow(img_np)
            attn_map = sample["restored_attn"].view(1, 1, h_tokens, w_tokens)
            attn_map = F.interpolate(attn_map, size=(img_h, img_w), mode="bilinear", align_corners=False)
            attn_map = attn_map.squeeze().numpy()
            axes[4].imshow(attn_map, cmap="jet", alpha=0.45)
            axes[4].set_title("(e) BiMamba Final Attention")
            axes[4].axis("off")
        else:
            # Legacy fallback: similarity-based visualization only
            sim_map = sample["sim"].view(1, 1, h_tokens, w_tokens)
            sim_map = F.interpolate(sim_map, size=(img_h, img_w), mode="bilinear", align_corners=False)
            sim_map = sim_map.squeeze().numpy()

            axes[1].imshow(img_np)
            axes[1].imshow(sim_map, cmap="jet", alpha=0.45)
            axes[1].set_title("(b) CLIP Initial Similarity")
            axes[1].axis("off")

            axes[2].imshow(img_np)
            indices = sample["indices"]
            topk = min(10, indices.numel())
            for rank_idx, patch_idx in enumerate(indices[:topk].tolist(), start=1):
                row = patch_idx // w_tokens
                col = patch_idx % w_tokens
                rect = Rectangle(
                    (col * patch_w, row * patch_h), patch_w, patch_h,
                    fill=False, edgecolor="red", linewidth=2,
                )
                axes[2].add_patch(rect)
                axes[2].text(
                    col * patch_w + 2, row * patch_h + 14, str(rank_idx),
                    color="white", fontsize=10,
                    bbox=dict(facecolor="red", edgecolor="red", pad=1.0),
                )
            axes[2].set_title("(c) Patch Reordering (Top-10)")
            axes[2].axis("off")

            axes[3].imshow(img_np)
            attn_map = sample["restored_attn"].view(1, 1, h_tokens, w_tokens)
            attn_map = F.interpolate(attn_map, size=(img_h, img_w), mode="bilinear", align_corners=False)
            attn_map = attn_map.squeeze().numpy()
            axes[3].imshow(attn_map, cmap="jet", alpha=0.45)
            axes[3].set_title("(d) BiMamba Final Attention")
            axes[3].axis("off")

        plt.tight_layout(pad=0.6, w_pad=0.4)
        save_name = _build_visual_filename(sample, image_key, file_prefix=file_prefix)
        save_path = os.path.join(output_dir, save_name)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _build_visualization_subset_loader(cfg, val_loader, logger):
    if not cfg.TEST.VISUALIZE_COMPARISON:
        return val_loader, None

    dataset = val_loader.dataset
    total_samples = len(dataset)
    num_samples = min(cfg.TEST.VISUALIZE_NUM_SAMPLES, total_samples)
    rng = random.Random(cfg.SOLVER.SEED)
    selected_indices = sorted(rng.sample(range(total_samples), num_samples))

    subset_loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=min(cfg.TEST.IMS_PER_BATCH, num_samples),
        shuffle=False,
        num_workers=val_loader.num_workers,
        pin_memory=getattr(val_loader, "pin_memory", False),
        drop_last=False,
    )
    logger.info(
        "Visualization mode enabled: sampling %d/%d test images for comparison output.",
        num_samples,
        total_samples,
    )
    return subset_loader, selected_indices


def _resolve_visualize_epochs(cfg):
    configured_epochs = list(getattr(cfg.TEST, 'VISUALIZE_EPOCHS', []))
    if configured_epochs:
        return {int(epoch) for epoch in configured_epochs}
    return {int(cfg.SOLVER.MAX_EPOCHS)}


def _unpack_eval_batch(batch):
    img = batch[0]
    pid = batch[1]
    camid = batch[2]
    camids_batch = camid
    target_view = None
    img_paths = None

    if len(batch) == 5:
        img_paths = batch[4]
    elif len(batch) >= 6:
        camids_batch = batch[3]
        target_view = batch[4]
        img_paths = batch[5]

    return img, pid, camid, camids_batch, target_view, img_paths


def _build_visual_filename(sample, fallback_key, file_prefix=''):
    image_key = sample.get('image_key', fallback_key)
    image_stem = os.path.splitext(os.path.basename(str(image_key)))[0]
    pid = sample.get('pid')
    camid = sample.get('camid')
    epoch = sample.get('epoch')

    if epoch is not None and pid is not None and camid is not None:
        return f"epoch{int(epoch)}_pid{int(pid):04d}_cam{int(camid) + 1}_{image_stem}.png"

    return f"{file_prefix}{image_stem}.png"


def _compute_valid_order(dist_row, q_pid, q_camid, g_pids, g_camids):
    order = np.argsort(dist_row)
    remove = (g_pids == q_pid) & (g_camids == q_camid)
    keep = np.invert(remove)
    return order[keep[order]]


def _select_visualization_targets(features, pids, camids, img_paths, num_query, logger):
    qf = features[:num_query]
    gf = features[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])
    q_paths = list(img_paths[:num_query])

    distmat = euclidean_distance(qf, gf)
    selected = OrderedDict()
    selection_notes = []

    pid_to_query_indices = OrderedDict()
    for idx, (pid, camid, path) in enumerate(zip(q_pids, q_camids, q_paths)):
        pid_to_query_indices.setdefault(int(pid), OrderedDict())
        pid_to_query_indices[int(pid)].setdefault(int(camid), idx)

    for pid, cam_map in pid_to_query_indices.items():
        if len(cam_map) >= 4:
            for camid, q_idx in list(cam_map.items())[:4]:
                path = q_paths[q_idx]
                if path in selected:
                    continue
                selected[path] = {
                    'reason': 'multi_cam_consistency',
                    'pid': int(pid),
                    'camid': int(camid),
                }
                selection_notes.append(f"multi_cam_consistency -> pid {pid:04d}, cam {camid + 1}, {os.path.basename(path)}")
            break

    easy_candidate = None
    hard_near_miss_candidate = None
    hard_fail_candidate = None

    for q_idx, (q_pid, q_camid, q_path) in enumerate(zip(q_pids, q_camids, q_paths)):
        order = _compute_valid_order(distmat[q_idx], q_pid, q_camid, g_pids, g_camids)
        if len(order) == 0:
            continue

        matches = (g_pids[order] == q_pid)
        if not np.any(matches):
            continue

        first_pos = int(np.where(matches)[0][0])
        top1_correct = bool(matches[0])
        pos_dist = float(distmat[q_idx][order[first_pos]])
        neg_positions = np.where(~matches)[0]
        neg_dist = float(distmat[q_idx][order[neg_positions[0]]]) if len(neg_positions) else pos_dist
        margin = neg_dist - pos_dist
        top1_dist = float(distmat[q_idx][order[0]])
        candidate = {
            'path': q_path,
            'pid': int(q_pid),
            'camid': int(q_camid),
            'first_pos_rank': first_pos + 1,
            'margin': margin,
            'top1_dist': top1_dist,
        }

        if top1_correct:
            if easy_candidate is None or candidate['margin'] > easy_candidate['margin']:
                easy_candidate = candidate
        elif candidate['first_pos_rank'] <= 10:
            if hard_near_miss_candidate is None or (candidate['first_pos_rank'], candidate['top1_dist']) < (hard_near_miss_candidate['first_pos_rank'], hard_near_miss_candidate['top1_dist']):
                hard_near_miss_candidate = candidate
        else:
            if hard_fail_candidate is None or candidate['top1_dist'] < hard_fail_candidate['top1_dist']:
                hard_fail_candidate = candidate

    for label, candidate in (
        ('easy_positive', easy_candidate),
        ('hard_near_miss', hard_near_miss_candidate),
        ('hard_negative', hard_fail_candidate),
    ):
        if candidate is None or candidate['path'] in selected:
            continue
        selected[candidate['path']] = {
            'reason': label,
            'pid': candidate['pid'],
            'camid': candidate['camid'],
        }
        selection_notes.append(
            f"{label} -> pid {candidate['pid']:04d}, cam {candidate['camid'] + 1}, rank {candidate['first_pos_rank']}, {os.path.basename(candidate['path'])}"
        )

    if selection_notes:
        logger.info("Visualization sample selection:")
        for note in selection_notes:
            logger.info("  %s", note)
    else:
        logger.warning("No visualization targets were selected; falling back to first validation samples.")

    return selected


def train_climb(cfg,
              model,
              train_loader,
              val_loader,
              cluster_loader,
              optimizer,
              scheduler,
              num_query,
              num_classes):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("CLIMB")
    logger.info('start training')
    
    # model.to(device)
    if device:
        model.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model = nn.DataParallel(model).cuda()

    loss_meter = AverageMeter()
    loss_meter1 = AverageMeter()
    loss_meter2 = AverageMeter()
    loss_meter3 = AverageMeter()
    loss_meter4 = AverageMeter()
    loss_proxy_mean_meter = AverageMeter()
    loss_proxy_hard_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter1 = AverageMeter()
    xent = CrossEntropyLabelSmooth(num_classes)
    tri_loss = TripletLoss()
    logger.info(f'smoothed cross entropy loss on {num_classes} classes.')

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
    )
    # scaler = amp.GradScaler()
    best_performance = 0
    best_epoch = 1
    visualize_epochs = _resolve_visualize_epochs(cfg)

    # training epochs
    for epoch in range(1, epochs+1):
        loss_meter.reset()
        loss_meter1.reset()
        loss_meter2.reset()
        loss_meter3.reset()
        loss_meter4.reset()
        loss_proxy_mean_meter.reset()
        loss_proxy_hard_meter.reset()
        acc_meter.reset()
        acc_meter1.reset()

        evaluator.reset()

        # create memory bank
        image_features, gt_labels = extract_image_features(model, cluster_loader, use_amp=True)
        image_features = image_features.float()
        image_features = F.normalize(image_features, dim=1)
            
        num_classes = len(gt_labels.unique()) - 1 if -1 in gt_labels else len(gt_labels.unique())
        logger.info(f'Memory has {num_classes} classes.')
        
        train_loader.new_epoch()
        
        # CAP memory
        memory = ClusterMemoryAMP(momentum=cfg.MODEL.MEMORY_MOMENTUM, use_hard=True).to(device)
        memory.features = compute_cluster_centroids(image_features, gt_labels).to(device)
        logger.info('Create memory bank with shape = {}'.format(memory.features.shape))
        
        # train one iteration
        model.train()
        num_iters = len(train_loader)
        for n_iter in range(num_iters):
            img, target, target_cam, _ = train_loader.next()
            
            optimizer.zero_grad()
            
            img = img.to(device)
            target = target.to(device)
            target_cam = target_cam.to(device)
            
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
                
            # with amp.autocast(enabled=True):
            feat, logits, feat_sp, logits_sp = model(img, cam_label=target_cam, view_label=target_view)
            loss1 = memory(feat, target) * cfg.MODEL.PCL_LOSS_WEIGHT
            proxy_mean_ce = float(getattr(memory, "last_mean_ce", 0.0))
            proxy_hard_ce = float(getattr(memory, "last_hard_ce", 0.0))
            # if cfg.MODEL.ID_LOSS_WEIGHT > 0:
            loss_id = xent(logits, target) * cfg.MODEL.ID_LOSS_WEIGHT
            loss_id2 = xent(logits_sp, target)
            loss_tri = tri_loss(feat_sp, target)
            loss = loss1 + loss_id + loss_id2 + loss_tri

            loss.backward()
            optimizer.step()

            # scaler.step(optimizer)
            # scaler.update()
            acc = (logits.max(1)[1] == target).float().mean()
            acc2 = (logits_sp.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            loss_meter1.update(loss1.item(), img.shape[0])
            loss_meter2.update(loss_id.item(), img.shape[0])
            loss_meter3.update(loss_id2.item(), img.shape[0])
            loss_meter4.update(loss_tri.item(), img.shape[0])
            loss_proxy_mean_meter.update(proxy_mean_ce, img.shape[0])
            loss_proxy_hard_meter.update(proxy_hard_ce, img.shape[0])
            acc_meter.update(acc, 1)
            acc_meter1.update(acc2, 1)

            torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] "
                            "Loss_total: {:.3f}, "
                            "Loss1: {:.3f}, "
                            "Loss2: {:.3f}, "
                            "Loss3: {:.3f}, "
                            "Loss4: {:.3f}, "
                            "ProxyMeanCE: {:.3f}, "
                            "ProxyHardCE: {:.3f}, "
                            "acc1: {:.3f},"
                            "acc2: {:.3f},"
                            "Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg,
                                    loss_meter1.avg,
                                    loss_meter2.avg,
                                    loss_meter3.avg,
                                    loss_meter4.avg,
                                    loss_proxy_mean_meter.avg,
                                    loss_proxy_hard_meter.avg,
                                    acc_meter.avg,
                                    acc_meter1.avg,
                                    scheduler.get_lr()[0]))
        
        scheduler.step()
        logger.info("Epoch {} done.".format(epoch))

        should_eval = (epoch % eval_period == 0 and epoch >= 55) or (epoch in visualize_epochs)
        if should_eval:
            model.eval()
            evaluator = R1_mAP_eval(
                num_query,
                max_rank=50,
                feat_norm=cfg.TEST.FEAT_NORM,
                reranking=cfg.TEST.RE_RANKING,
            )
            evaluator.reset()

            visualize_this_epoch = epoch in visualize_epochs
            vis_output_dir = os.path.join(cfg.OUTPUT_DIR, "comparison_visuals")
            eval_feats = []
            eval_pids = []
            eval_camids = []
            eval_img_paths = []

            for _, batch in enumerate(val_loader):
                with torch.no_grad():
                    img, vid, camid, camids_batch, target_view, img_paths = _unpack_eval_batch(batch)

                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camids_batch.to(device)
                    else:
                        camids = None
                    if cfg.MODEL.SIE_VIEW and target_view is not None:
                        target_view = target_view.to(device)
                    else:
                        target_view = None

                    feat, feat1, feat2 = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, feat1, feat2, vid, camid))

                    if visualize_this_epoch and img_paths is not None:
                        eval_feats.append(feat.detach().cpu())
                        eval_pids.extend(np.asarray(vid).tolist())
                        eval_camids.extend(np.asarray(camid).tolist())
                        eval_img_paths.extend(list(img_paths))

            epoch_visuals = OrderedDict()
            if visualize_this_epoch and eval_img_paths:
                os.makedirs(vis_output_dir, exist_ok=True)
                feats_for_vis = torch.cat(eval_feats, dim=0)
                if cfg.TEST.FEAT_NORM:
                    feats_for_vis = torch.nn.functional.normalize(feats_for_vis, dim=1, p=2)
                selected_targets = _select_visualization_targets(
                    feats_for_vis,
                    eval_pids,
                    eval_camids,
                    eval_img_paths,
                    num_query,
                    logger,
                )

                if selected_targets:
                    selected_keys = set(selected_targets.keys())
                    for _, batch in enumerate(val_loader):
                        with torch.no_grad():
                            img, vid, camid, camids_batch, target_view, img_paths = _unpack_eval_batch(batch)
                            if img_paths is None:
                                continue

                            batch_keys = list(img_paths)
                            if not any(image_key in selected_keys for image_key in batch_keys):
                                continue

                            img_cpu = img.detach().cpu()
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids_batch.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW and target_view is not None:
                                target_view = target_view.to(device)
                            else:
                                target_view = None

                            _, _, _, visual_tensors = model(
                                img,
                                cam_label=camids,
                                view_label=target_view,
                                return_visuals=True,
                            )

                            sim = visual_tensors["sim"].detach().cpu()
                            indices = visual_tensors["indices"].detach().cpu().long()
                            attn_weights = visual_tensors["attn_weights"].detach().cpu()
                            sim_raw = visual_tensors.get("sim_raw")
                            indices_raw = visual_tensors.get("indices_raw")
                            if sim_raw is not None:
                                sim_raw = sim_raw.detach().cpu()
                            if indices_raw is not None:
                                indices_raw = indices_raw.detach().cpu().long()
                            batch_size = img_cpu.size(0)
                            if sim.dim() == 3:
                                sim = sim.squeeze(1)
                            if sim_raw is not None and sim_raw.dim() == 3:
                                sim_raw = sim_raw.squeeze(1)
                            restored_attn = _resolve_attention_map(attn_weights, indices, batch_size, logger=logger, tag=f"epoch{epoch}")

                            for sample_idx, image_key in enumerate(batch_keys):
                                if image_key not in selected_targets or image_key in epoch_visuals:
                                    continue
                                vis_entry = {
                                    "img": img_cpu[sample_idx].clone(),
                                    "sim": sim[sample_idx].clone(),
                                    "indices": indices[sample_idx].clone(),
                                    "restored_attn": restored_attn[sample_idx].clone(),
                                    "pid": int(np.asarray(vid)[sample_idx]),
                                    "camid": int(np.asarray(camid)[sample_idx]),
                                    "epoch": int(epoch),
                                    "image_key": image_key,
                                    "reason": selected_targets[image_key]["reason"],
                                }
                                if sim_raw is not None:
                                    vis_entry["sim_raw"] = sim_raw[sample_idx].clone()
                                if indices_raw is not None:
                                    vis_entry["indices_raw"] = indices_raw[sample_idx].clone()
                                epoch_visuals[image_key] = vis_entry

                            if len(epoch_visuals) == len(selected_targets):
                                break

                    h_tokens = model.module.h_resolution if hasattr(model, "module") else model.h_resolution
                    w_tokens = model.module.w_resolution if hasattr(model, "module") else model.w_resolution
                    pixel_mean = torch.tensor(cfg.INPUT.PIXEL_MEAN, dtype=torch.float32).view(3, 1, 1)
                    pixel_std = torch.tensor(cfg.INPUT.PIXEL_STD, dtype=torch.float32).view(3, 1, 1)
                    _save_comparison_visuals(epoch_visuals, vis_output_dir, "", h_tokens, w_tokens, pixel_mean, pixel_std)
                    logger.info("Saved %d comparison figures to %s", len(epoch_visuals), vis_output_dir)

            cmc, mAP, cmc01, mAP01, cmc02, mAP02, cmc03, mAP03 = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP(main): {:.5%}".format(mAP))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.5%}".format(r, cmc[r - 1]))

            logger.info("mAP_1(CLIP): {:.5%}".format(mAP01))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.5%}".format(r, cmc01[r - 1]))
            logger.info("mAP_2(BiMamba): {:.5%}".format(mAP02))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.5%}".format(r, cmc02[r - 1]))
            logger.info("mAP_3: {:.5%}".format(mAP03))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.5%}".format(r, cmc03[r - 1]))
            torch.cuda.empty_cache()
            prec1 = cmc[0] + mAP
            is_best = prec1 > best_performance
            best_performance = max(prec1, best_performance)
            if is_best:
                best_epoch = epoch
            save_checkpoint(model.state_dict(), is_best, os.path.join(cfg.OUTPUT_DIR, 'checkpoint_ep.pth.tar'))

            torch.cuda.empty_cache()
    logger.info("==> Best Perform {:.5%}, achieved at epoch {}".format(best_performance, best_epoch))
    logger.info('Training done.')
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("CLIMB")
    logger.info("Enter inferencing")
    model.to(device)
    model.eval()

    inference_loader, selected_indices = _build_visualization_subset_loader(cfg, val_loader, logger)
    run_metrics = not cfg.TEST.VISUALIZE_COMPARISON

    if run_metrics:
        evaluator = R1_mAP_eval(
            num_query,
            max_rank=50,
            feat_norm=cfg.TEST.FEAT_NORM,
            reranking=cfg.TEST.RE_RANKING,
        )
        evaluator.reset()
    else:
        evaluator = None

    need_visuals = cfg.TEST.VISUALIZE_COMPARISON
    inference_visuals = OrderedDict()
    vis_output_dir = os.path.join(cfg.OUTPUT_DIR, "comparison_visuals_inference")
    subset_dataset = getattr(val_loader.dataset, "dataset", val_loader.dataset)
    subset_indices = selected_indices if selected_indices is not None else list(range(len(subset_dataset)))
    global_offset = 0

    for _, batch in enumerate(inference_loader):
        with torch.no_grad():
            img, pid, camid, camids_batch, target_view, img_paths = _unpack_eval_batch(batch)
            img_cpu = img.detach().cpu()
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids_batch.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW and target_view is not None:
                target_view = target_view.to(device)
            else:
                target_view = None

            if need_visuals:
                feat, feat0, feat1, visual_tensors = model(
                    img,
                    cam_label=camids,
                    view_label=target_view,
                    return_visuals=True,
                )
            else:
                feat, feat0, feat1 = model(
                    img,
                    cam_label=camids,
                    view_label=target_view,
                )
                visual_tensors = None

            if evaluator is not None:
                evaluator.update((feat, feat0, feat1, pid, camid))

            if not need_visuals or visual_tensors is None:
                continue

            sim = visual_tensors["sim"].detach().cpu()
            indices = visual_tensors["indices"].detach().cpu().long()
            attn_weights = visual_tensors["attn_weights"].detach().cpu()
            sim_raw = visual_tensors.get("sim_raw")
            indices_raw = visual_tensors.get("indices_raw")
            if sim_raw is not None:
                sim_raw = sim_raw.detach().cpu()
            if indices_raw is not None:
                indices_raw = indices_raw.detach().cpu().long()

            batch_size = img_cpu.size(0)
            if sim.dim() == 3:
                sim = sim.squeeze(1)
            if sim_raw is not None and sim_raw.dim() == 3:
                sim_raw = sim_raw.squeeze(1)
            restored_attn = _resolve_attention_map(attn_weights, indices, batch_size, logger=logger, tag="inference")

            for sample_idx in range(batch_size):
                original_index = subset_indices[global_offset + sample_idx]
                image_key = img_paths[sample_idx] if img_paths is not None else f"sample_{original_index:05d}"
                pid_value = int(np.asarray(pid)[sample_idx])
                camid_value = int(np.asarray(camid)[sample_idx])
                vis_entry = {
                    "img": img_cpu[sample_idx].clone(),
                    "sim": sim[sample_idx].clone(),
                    "indices": indices[sample_idx].clone(),
                    "restored_attn": restored_attn[sample_idx].clone(),
                    "pid": pid_value,
                    "camid": camid_value,
                    "image_key": image_key,
                }
                if sim_raw is not None:
                    vis_entry["sim_raw"] = sim_raw[sample_idx].clone()
                if indices_raw is not None:
                    vis_entry["indices_raw"] = indices_raw[sample_idx].clone()
                inference_visuals[image_key] = vis_entry
            global_offset += batch_size

    h_tokens = model.module.h_resolution if hasattr(model, "module") else model.h_resolution
    w_tokens = model.module.w_resolution if hasattr(model, "module") else model.w_resolution
    pixel_mean = torch.tensor(cfg.INPUT.PIXEL_MEAN, dtype=torch.float32).view(3, 1, 1)
    pixel_std = torch.tensor(cfg.INPUT.PIXEL_STD, dtype=torch.float32).view(3, 1, 1)
    if inference_visuals:
        _save_comparison_visuals(
            inference_visuals,
            vis_output_dir,
            "comparison_inference_",
            h_tokens,
            w_tokens,
            pixel_mean,
            pixel_std,
        )
        logger.info("Saved %d comparison figures to %s", len(inference_visuals), vis_output_dir)

    if evaluator is None:
        logger.info("Visualization-only inference finished on %d sampled images.", len(inference_visuals))
        return None, None

    cmc, mAP, cmc01, mAP01, cmc02, mAP02, cmc03, mAP03 = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP(main): {:.5%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.5%}".format(r, cmc[r - 1]))

    logger.info("mAP_1(CLIP): {:.5%}".format(mAP01))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.5%}".format(r, cmc01[r - 1]))
    logger.info("mAP_2(BiMamba): {:.5%}".format(mAP02))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.5%}".format(r, cmc02[r - 1]))
    logger.info("mAP_3: {:.5%}".format(mAP03))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.5%}".format(r, cmc03[r - 1]))
    return cmc[0], cmc[4]
