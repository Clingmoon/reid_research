#!/usr/bin/env python
"""
可视化 LAST-ViT 的 patch 选择过程。
对比 USE_LAST_CLS=True/False 时，哪些 patch 被 LAST 选中、哪些被原始 CLS 关注。
"""

import os
import sys
sys.path.insert(0, '/home/cfdeng/projects/CLIMB-ReID')

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T

from climb.model import load_clip_to_cpu


def get_last_stability_scores(patch_tokens, eps=1e-6):
    """
    计算 LAST-ViT 的稳定性分数（对 patch_tokens 逐 channel 计算后取平均）。
    patch_tokens: [B, num_patches, hidden_dim]
    返回: scores [B, num_patches]
    """
    fft_tokens = patch_tokens.float()
    kernel_size = fft_tokens.size(-1)
    positions = torch.arange(
        -kernel_size // 2 + 1,
        kernel_size // 2 + 1,
        device=fft_tokens.device,
        dtype=torch.float32,
    )
    kernel = torch.exp(-0.5 * (positions / (kernel_size ** 0.5)) ** 2)
    kernel = kernel / torch.max(kernel)
    kernel = kernel.to(dtype=fft_tokens.dtype).view(1, 1, -1)

    low_pass = torch.fft.fft(fft_tokens, dim=-1)
    low_pass = torch.fft.fftshift(low_pass, dim=-1)
    low_pass = low_pass * kernel
    low_pass = torch.fft.ifftshift(low_pass, dim=-1)
    low_pass = torch.fft.ifft(low_pass, dim=-1).real

    scores = fft_tokens / torch.abs(low_pass - fft_tokens).clamp_min(eps)
    # 官方实现：对每个 channel 独立选 top-k，然后 gather mean。
    # 可视化时简化：对每个 patch 的所有 channel 取平均，得到整体稳定性分数。
    scores_mean = scores.mean(dim=-1)
    return scores_mean


def visualize_last_vs_clip(img_path, last_weight_path, output_path,
                           h=256, w=128, stride=16):
    # 图像预处理
    transform = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).cuda()

    h_tokens = (h - 16) // stride + 1   # 16
    w_tokens = (w - 16) // stride + 1   # 8
    num_patches = h_tokens * w_tokens     # 128

    # 加载 LAST_CLS 模型
    clip_last = load_clip_to_cpu('ViT-B-16', h_tokens, w_tokens, stride, last_weight_path)
    vit_last = clip_last.visual.cuda()
    vit_last.eval()

    # 加载原始 CLIP 模型（回退到本地路径）
    clip_orig = load_clip_to_cpu('ViT-B-16', h_tokens, w_tokens, stride, '')
    vit_orig = clip_orig.visual.cuda()
    vit_orig.eval()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    img_np = np.array(img.resize((w, h)))
    patch_h = h / h_tokens
    patch_w = w / w_tokens

    with torch.no_grad():
        # ========== LAST_CLS 版本 ==========
        _, x12_last, _ = vit_last(img_tensor)
        patch_tokens_last = x12_last[:, 1:]           # [1, 128, 768]
        scores_last = get_last_stability_scores(patch_tokens_last)[0].cpu().numpy()

        # 稳定性分数归一化
        s_min, s_max = scores_last.min(), scores_last.max()
        scores_norm = (scores_last - s_min) / (s_max - s_min + 1e-8)
        scores_grid = scores_norm.reshape(h_tokens, w_tokens)

        # 1. 原始图片
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # 2. LAST 稳定性分数热力图
        axes[0, 1].imshow(img_np)
        im1 = axes[0, 1].imshow(scores_grid, cmap='jet', alpha=0.5,
                                 extent=[0, w, h, 0])
        axes[0, 1].set_title('LAST Stability Scores (red=higher, selected)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

        # 3. LAST Top-10 选中的 patch
        topk_last = np.argsort(scores_last)[-10:]
        axes[0, 2].imshow(img_np)
        for idx in topk_last:
            row, col = idx // w_tokens, idx % w_tokens
            rect = plt.Rectangle((col * patch_w, row * patch_h), patch_w, patch_h,
                                 fill=False, edgecolor='red', linewidth=2)
            axes[0, 2].add_patch(rect)
        axes[0, 2].set_title('LAST Top-10 Selected Patches')
        axes[0, 2].axis('off')

        # ========== 原始 CLIP 版本 ==========
        _, x12_orig, _ = vit_orig(img_tensor)
        patch_tokens_orig = x12_orig[:, 1:]
        cls_orig = x12_orig[:, 0]

        # CLS 与每个 patch 的余弦相似度
        patch_norm = F.normalize(patch_tokens_orig, dim=-1)
        cls_norm = F.normalize(cls_orig, dim=-1)
        sim = torch.bmm(cls_norm.unsqueeze(1), patch_norm.transpose(1, 2))[0, 0]
        sim = sim.cpu().numpy()

        sim_min, sim_max = sim.min(), sim.max()
        sim_norm = (sim - sim_min) / (sim_max - sim_min + 1e-8)
        sim_grid = sim_norm.reshape(h_tokens, w_tokens)

        # 4. 原始 CLIP CLS-patch 相似度热力图
        axes[1, 0].imshow(img_np)
        im2 = axes[1, 0].imshow(sim_grid, cmap='jet', alpha=0.5,
                                 extent=[0, w, h, 0])
        axes[1, 0].set_title('CLIP CLS-Patch Cosine Similarity')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

        # 5. 原始 CLIP Top-10 相似 patch
        topk_sim = np.argsort(sim)[-10:]
        axes[1, 1].imshow(img_np)
        for idx in topk_sim:
            row, col = idx // w_tokens, idx % w_tokens
            rect = plt.Rectangle((col * patch_w, row * patch_h), patch_w, patch_h,
                                 fill=False, edgecolor='blue', linewidth=2)
            axes[1, 1].add_patch(rect)
        axes[1, 1].set_title('CLIP Top-10 Similar Patches')
        axes[1, 1].axis('off')

        # 6. 差异图：LAST 选中但 CLIP 不关注的区域（红色），反之（蓝色）
        diff = scores_norm - sim_norm
        diff_grid = diff.reshape(h_tokens, w_tokens)
        axes[1, 2].imshow(img_np)
        im3 = axes[1, 2].imshow(diff_grid, cmap='RdBu_r', alpha=0.5,
                                 extent=[0, w, h, 0], vmin=-1, vmax=1)
        axes[1, 2].set_title('Difference: LAST(red+) vs CLIP(blue+)')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved visualization to {output_path}')


if __name__ == '__main__':
    # Market-1501 测试图片
    img_dir = '/home/cfdeng/projects/CLIMB-ReID/datasets/reid-datasets/Market1501/bounding_box_test'
    img_name = sorted(os.listdir(img_dir))[0]  # 取第一张
    img_path = os.path.join(img_dir, img_name)

    last_weight_path = '/home/cfdeng/projects/CLIMB-ReID/config/openai_b_16.pt'
    output_path = '/home/cfdeng/projects/CLIMB-ReID/last_cls_visualization.png'

    print(f'Visualizing: {img_path}')
    visualize_last_vs_clip(img_path, last_weight_path, output_path)
