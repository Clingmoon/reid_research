# Market1501 上将 CLIMB 的 BiMamba 替换为 VMamba SS2D 的实现方案

本文档基于当前仓库代码阅读结果，只给出方案与改动清单，不修改训练代码。

## 1. 当前训练流程梳理

入口：`train_climb.py`

1. 读取配置：默认 `config/climb-vit-msmt.yml`，Market1501 实验使用 `config/climb-vit-market.yml`。
2. 固定随机种子：`set_seed(cfg.SOLVER.SEED)`。
3. 创建日志与输出目录：`cfg.OUTPUT_DIR`。
4. 构建数据：`make_CLIMB_dataloader(cfg)` 返回：
   - `train_loader`：Market1501 训练集，`RandomMultipleGallerySampler`，外面包一层 `IterLoader`，每 epoch 迭代数由 `SOLVER.ITERS` 控制。
   - `val_loader`：`query + gallery`，用于评估。
   - `cluster_loader`：训练集但使用测试 transform，用于每个 epoch 重新抽取特征并构建 memory bank。
5. 构建模型：`make_model(cfg, num_classes, camera_num, view_num)`，实际返回 `CLIMB`。
6. 构建优化器：`make_CLIMB_optimizer(cfg, model)`。
7. 训练：`train_climb(...)`。

训练核心在 `climb/processor_climb.py`：

1. 每个 epoch 开头用 `extract_image_features(model, cluster_loader, use_amp=True)` 抽取训练集特征。
2. 对特征做归一化并按 ID 计算 centroid，形成 `ClusterMemoryAMP` 的 memory bank。
3. 每个 iteration：
   - 输入 batch 图片。
   - 调用 `model(img, cam_label=..., view_label=...)`。
   - 当前训练 forward 返回 `out_feat, logits, feat_sp, logits_sp`。
   - 损失为：
     - `loss1 = memory(feat, target) * cfg.MODEL.PCL_LOSS_WEIGHT`
     - `loss_id = CE(logits, target) * cfg.MODEL.ID_LOSS_WEIGHT`
     - `loss_id2 = CE(logits_sp, target)`
     - `loss_tri = TripletLoss(feat_sp, target)`
     - 总损失 `loss = loss1 + loss_id + loss_id2 + loss_tri`
4. 评估时模型返回 `feat_concat, out_feat, feat_sp`，其中：
   - `feat_concat = concat(CLIP branch, Mamba branch)` 是主评估特征。
   - `out_feat` 是 CLIP 分支。
   - `feat_sp` 是 BiMamba/后续 SS2D 分支。

## 2. 当前 BiMamba 分支位置

主要代码在 `climb/model.py`。

### 2.1 模型结构

当前 `CLIMB.__init__` 中与 Mamba 分支相关的模块：

- `self.sp_mamba_bi = nn.Sequential(nn.LayerNorm(768), BiMamba(...))`
- `self.sp_mamba_raw = nn.Sequential(nn.LayerNorm(768), Mamba(...))`，目前 forward 中未启用。
- `self.norm2_mamba = nn.LayerNorm(768)`
- `self.sp_attention = Linear(768 -> 192) + Tanh + Linear(192 -> 1)`
- `self.bottleneck_proj_sp = BatchNorm1d(768)`
- `self.classifier2 = Linear(768 -> num_classes)`

### 2.2 Forward 流程

当前 forward 的关键步骤：

1. CLIP visual encoder 输出：
   - `image_features`：`[B, 1 + H*W, 768]`
   - `image_features_proj`：`[B, 1 + H*W, 512]`
2. CLIP 主分支：
   - 取 `image_features[:, 0]` 和 `image_features_proj[:, 0]`。
   - 分别过 BN。
   - 拼成 `out_feat`，维度 `[B, 1280]`。
3. Mamba 分支：
   - `feats_for_mamba = image_features.detach()`。
   - patch token：`feats_for_mamba_sp = feats_for_mamba[:, 1:, :]`，Market1501 配置下为 `[B, 128, 768]`。
   - cls token：`feats_for_mamba_cls = feats_for_mamba[:, 0, :]`。
   - `reorder(cls, patch)`：按 cls 与 patch 的 cosine similarity 从高到低排序，得到 `[B, 128, 768]`。
   - `self.sp_mamba_bi(re_order_mamba_sp)`：BiMamba 做一维双向扫描，输出 `[B, 128, 768]`。
   - 拼回 cls token：`[B, 129, 768]`。
   - `sp_attention` 对 token 做 softmax 加权池化，得到 `[B, 768]`。
   - BN 得到 `feat_sp`。

需要注意：`image_features.detach()` 表示当前 Mamba 分支的 `loss_id2 + loss_tri` 不反传到 CLIP visual encoder，只训练 Mamba 分支、attention、BN、classifier2 等后接模块。若要做公平 baseline 对比，首次 SS2D 实验建议保持 `detach()` 不变。

## 3. SS2D 替换

不是简单把 `BiMamba(...)` 换成 `SS2D(...)`。原因是二者输入形状和建模假设不同。

### 3.1 形状差异

- 当前 BiMamba 输入：`[B, L, D]`，其中 `L = H*W`，按 reorder 后的一维序列扫描。
- VMamba SS2D 输入：通常是 `[B, H, W, D]`，在二维空间上做四方向 cross scan。
- Market1501 当前配置：`INPUT.SIZE_TRAIN = [256, 128]`，`STRIDE_SIZE = [16, 16]`，因此：
  - `H = (256 - 16) // 16 + 1 = 16`
  - `W = (128 - 16) // 16 + 1 = 8`
  - `L = 128`

因此需要一个 adapter 在 token 序列和二维 patch grid 之间转换。

### 3.2 语义差异

BiMamba 当前做的是 IRM 风格的一维重要性排序扫描：先按 cls-patch 相似度排序，再双向 Mamba。

SS2D 做的是 VMamba 风格二维空间扫描：它假设 token 仍保持原始二维 patch 邻接关系。

因此有两种实现路线：

#### 推荐路线 A：空间 SS2D，不使用 reorder 后序列

这是最符合“替换为 VMamba 的 SS2D”的方案。

流程变为：

1. 保留原始 patch token 顺序：`feats_for_mamba_sp = image_features[:, 1:, :]`。
2. reshape 为二维：`[B, H, W, D]`。
3. 输入 SS2D/VSSBlock。
4. flatten 回 `[B, H*W, D]`。
5. 与原始 cls token 拼接。
6. 继续使用原来的 `sp_attention -> BN -> classifier2 -> losses`。

优点：

- 符合 VMamba SS2D 的设计假设。
- 对比意义明确：`IRM/BiMamba` vs `spatial SS2D`。
- 可视化 attention 可以自然还原到原图 patch grid，不再依赖 reorder indices。

缺点：

- 改动了原始 IRM 的“importance-aware reorder”思想，因此这不是只替换扫描算子，而是把分支从重要性一维扫描改成二维空间扫描。

#### 备选路线 B：保留 reorder，把排序序列 reshape 成伪二维输入 SS2D

流程为：

1. 仍然执行 `reorder(cls, patch)`。
2. 将排序后的 `[B, L, D]` reshape 成 `[B, H, W, D]`。
3. 输入 SS2D。
4. flatten 回 `[B, L, D]`。

优点：

- 最大程度保留原代码数据流。
- 可继续使用当前 visualization 中的 reorder indices。

缺点：

- SS2D 的二维邻接关系不再对应真实图像空间，而是“重要性排序后的伪网格”。
- 结果若变差，很难判断是 SS2D 不适合，还是伪二维结构破坏了空间扫描假设。

建议：首次小实验采用路线 A；若想做 ablation，再补路线 B。

## 4. 建议的代码改动清单

### 4.1 新增配置项

建议在 `config/defaults.py` 增加可切换配置，而不是硬改模型：

```yaml
MODEL:
  SPATIAL_BRANCH_TYPE: 'bimamba'  # 可选：bimamba, ss2d
  SS2D_FORWARD_TYPE: 'v1'
  SS2D_D_STATE: 16
  SS2D_SSM_RATIO: 2.0
  SS2D_D_CONV: 3
  SS2D_USE_VSS_BLOCK: False
  SS2D_USE_REORDER: False
```

说明：

- `SS2D_D_CONV` 建议用 `3`。不要直接沿用 BiMamba 的 `d_conv=4`，因为当前 `SS2D` 里的 `Conv2d` padding 是 `(d_conv - 1) // 2`，偶数 kernel 可能导致空间尺寸变化，进而和门控分支 `z` 形状不匹配。
- `SS2D_USE_REORDER=False` 对应推荐路线 A。
- `SS2D_USE_VSS_BLOCK=False` 表示只用 SS2D 算子；若设为 `True`，则使用 VMamba 的 VSSBlock 风格残差结构。

### 4.2 修改 `climb/model.py` 的 import

当前文件已有注释掉的导入：

```python
# from .spmamba import VSSBlock
```

建议改成：

```python
from .spmamba import SS2D, VSSBlock
```

或者只导入 `VSSBlock`，但推荐直接封装一个 adapter，这样 forward 中更清晰。

### 4.3 新增 SS2D adapter

建议在 `climb/model.py` 中 `CLIMB` 类之前新增一个小模块：

```python
class SS2DTokenAdapter(nn.Module):
    def __init__(self, h_resolution, w_resolution, d_model=768, d_state=16,
                 ssm_ratio=2.0, d_conv=3, forward_type='v2'):
        super().__init__()
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.norm = nn.LayerNorm(d_model)
        self.ss2d = SS2D(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            d_conv=d_conv,
            forward_type=forward_type,
        )

    def forward(self, tokens):
        b, num_tokens, dim = tokens.shape
        expected_tokens = self.h_resolution * self.w_resolution
        if num_tokens != expected_tokens:
            raise ValueError(f'SS2D expected {expected_tokens} tokens, got {num_tokens}')
        x = self.norm(tokens).view(b, self.h_resolution, self.w_resolution, dim)
        x = self.ss2d(x)
        return x.view(b, num_tokens, dim)
```

如果希望使用完整 VSSBlock，则 adapter 内部可以换为：

```python
self.ss2d = VSSBlock(
    hidden_dim=d_model,
    ssm_d_state=d_state,
    ssm_ratio=ssm_ratio,
    ssm_conv=d_conv,
    forward_type=forward_type,
    mlp_ratio=0.0,
)
```

注意：`VSSBlock` 自带 `LayerNorm + residual`，因此如果用 `VSSBlock`，adapter 外层不要再重复加 `LayerNorm`，否则和只替换算子版本不一致。

### 4.4 修改 `CLIMB.__init__`

保留原 BiMamba，同时按配置新增 SS2D 分支：

```python
branch_type = getattr(cfg.MODEL, 'SPATIAL_BRANCH_TYPE', 'bimamba')
if branch_type == 'ss2d':
    self.sp_mamba_bi = SS2DTokenAdapter(
        self.h_resolution,
        self.w_resolution,
        d_model=768,
        d_state=cfg.MODEL.SS2D_D_STATE,
        ssm_ratio=cfg.MODEL.SS2D_SSM_RATIO,
        d_conv=cfg.MODEL.SS2D_D_CONV,
        forward_type=cfg.MODEL.SS2D_FORWARD_TYPE,
    )
else:
    self.sp_mamba_bi = nn.Sequential(
        nn.LayerNorm(768),
        BiMamba(d_model=768, d_state=16, d_conv=4, expand=2),
    )
```

更干净的命名是把 `self.sp_mamba_bi` 改成 `self.spatial_branch`，但这会牵涉日志、可视化命名和 checkpoint key。为了最小改动，可以先复用 `self.sp_mamba_bi`。

### 4.5 修改 `CLIMB.forward`

当前代码总是执行：

```python
re_order_mamba_sp = self.reorder(feats_for_mamba_cls, feats_for_mamba_sp)
mamba_sp_out = self.sp_mamba_bi(re_order_mamba_sp)
```

推荐改为按配置分支：

```python
if self.spatial_branch_type == 'ss2d' and not self.ss2d_use_reorder:
    mamba_input = feats_for_mamba_sp
    mamba_sp_out = self.sp_mamba_bi(mamba_input)
    reorder_sim = None
    reorder_indices = torch.arange(
        mamba_input.size(1), device=mamba_input.device
    ).unsqueeze(0).expand(mamba_input.size(0), -1)
else:
    # 保留当前 reorder + BiMamba 流程，或用于 SS2D_USE_REORDER=True 的伪二维实验
    ...
```

为了不破坏当前可视化，SS2D 空间路线下可以用 identity indices，使 `_resolve_attention_map` 能正常工作。

### 4.6 修改日志和可视化文案

当前有几处硬编码 `BiMamba`：

- `climb/processor_climb.py` 中可视化标题：`(d) BiMamba Final Attention`。
- `climb/processor_climb.py` 中评估日志：`mAP_2(BiMamba)`。

建议改成根据 `cfg.MODEL.SPATIAL_BRANCH_TYPE` 动态显示，例如：

- `BiMamba Final Attention`
- `SS2D Final Attention`
- `mAP_2(SS2D)`

这不影响训练，但有利于实验记录不混淆。

### 4.7 新增 Market1501 SS2D 配置

建议不要直接覆盖 `config/climb-vit-market.yml`，新增：

```text
config/experiments/market1501/climb-vit-market-ss2d.yml
```

内容继承或复制 Market 配置，仅修改：

```yaml
MODEL:
  SPATIAL_BRANCH_TYPE: 'ss2d'
  SS2D_FORWARD_TYPE: 'v1'
  SS2D_D_STATE: 16
  SS2D_SSM_RATIO: 2.0
  SS2D_D_CONV: 3
  SS2D_USE_REORDER: False

OUTPUT_DIR: './logs/Market1501_SS2D_spatial_seed1234_YYYYMMDD_HHMMSS'
```

baseline 也建议单独复制一份：

```text
config/experiments/market1501/climb-vit-market-bimamba-baseline.yml
```

## 5. 依赖与运行风险

当前仓库已经包含两个 VMamba 相关位置：

- `climb/spmamba.py`：已经拷贝了 `SS2D` 和 `VSSBlock`，更适合直接在 CLIMB 中导入。
- `VMamba/`：原始 VMamba 工程代码与 CUDA kernel。

运行 SS2D 需要 selective scan CUDA 扩展。`climb/spmamba.py` 会尝试导入：

- `selective_scan_cuda_oflex`
- `selective_scan_cuda_core`
- `selective_scan_cuda`

如果环境里没有编译这些扩展，代码可能可以 import，但真正 forward 到 `SS2D` 时会报错。建议优先安装 VMamba 自带 kernel：

```bash
cd /home/cfdeng/projects/CLIMB-ReID/VMamba/kernels/selective_scan
pip install -e .
```

然后做最小 smoke test：

```bash
python - <<'PY'
import torch
from climb.spmamba import SS2D

module = SS2D(d_model=768, d_state=16, ssm_ratio=2.0, d_conv=3, forward_type='v2').cuda()
x = torch.randn(2, 16, 8, 768, device='cuda')
y = module(x)
print(y.shape)
PY
```

预期输出：

```text
torch.Size([2, 16, 8, 768])
```

我在当前未激活训练环境的 shell 中检查到 `python3` 环境缺少 `torch/timm/einops/fvcore/selective_scan_cuda_*`，因此实际运行前需要先进入你的 CLIMB conda 环境或按 `INSTALL.md` 初始化环境。

## 6. 建议的小实验顺序

### 6.1 先跑 baseline

目的：确认当前代码、数据路径、可视化输出和指标都正常。

```bash
CUDA_VISIBLE_DEVICES=0 python train_climb.py \
  --config_file config/experiments/market1501/climb-vit-market-bimamba-baseline.yml
```

建议先小跑：

```bash
CUDA_VISIBLE_DEVICES=0 python train_climb.py \
  --config_file config/experiments/market1501/climb-vit-market-bimamba-baseline.yml \
  SOLVER.MAX_EPOCHS 1 SOLVER.ITERS 2 SOLVER.EVAL_PERIOD 1 TEST.IMS_PER_BATCH 64 \
  OUTPUT_DIR "'./logs/debug_market_bimamba_1ep'"
```

### 6.2 跑 SS2D smoke train

目的：确认 forward/backward、loss、eval、可视化都不炸。

```bash
CUDA_VISIBLE_DEVICES=0 python train_climb.py \
  --config_file config/experiments/market1501/climb-vit-market-ss2d.yml \
  SOLVER.MAX_EPOCHS 1 SOLVER.ITERS 2 SOLVER.EVAL_PERIOD 1 TEST.IMS_PER_BATCH 64 \
  OUTPUT_DIR "'./logs/debug_market_ss2d_1ep'"
```

### 6.3 再跑正式小实验

保持以下变量一致：

- 同一个 seed。
- 同一个 Market1501 数据路径。
- 同一个 batch size / iteration / epoch。
- 同一个 eval schedule。
- 同一个可视化 sample 数量。

只改变：

- `MODEL.SPATIAL_BRANCH_TYPE=bimamba` vs `ss2d`
- SS2D 相关参数。

建议记录指标：

| 实验 | 分支 | mAP(main) | Rank-1(main) | mAP_1(CLIP) | mAP_2(branch) | Rank-1_2(branch) | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | BiMamba + reorder |  |  |  |  |  |  |
| exp-ss2d-a | SS2D spatial |  |  |  |  |  |  |
| exp-ss2d-b | SS2D pseudo-grid reorder |  |  |  |  |  | optional |

## 7. 在他人代码上管理实验与版本的建议

### 7.1 Git 分支策略

建议至少维护三类分支：

- `main` 或 `official-baseline`：尽量保持官方代码和必要环境修复，不做实验性改动。
- `exp/visualization`：你当前做的可视化改动，建议先提交成清晰 commit。
- `exp/ss2d-market1501`：SS2D 替换实验。

每个实验改动尽量小步提交：

1. `add config switch for spatial branch`
2. `add ss2d token adapter`
3. `add market1501 ss2d config`
4. `update visualization labels for branch type`

这样后面写论文/报告时可以清楚解释每个改动。

### 7.2 配置文件管理

不要直接改公共配置覆盖实验。推荐目录：

```text
config/experiments/market1501/
  baseline_bimamba_seed1234.yml
  ss2d_spatial_seed1234.yml
  ss2d_reorder_seed1234.yml
```

每个配置里显式写清：

- 数据集。
- 输入尺寸。
- seed。
- batch size。
- max epochs。
- eval period。
- 模型分支类型。
- output dir。

### 7.3 输出目录规范

建议输出目录包含：数据集、方法、关键参数、seed、时间。

示例：

```text
logs/market1501/
  20260425_1530_bimamba_reorder_s1234/
  20260425_1800_ss2d_spatial_v2_dstate16_ratio2_conv3_s1234/
```

### 7.4 每次实验保存元信息

建议训练启动时自动或手动保存：

- 实际合并后的 config：`config.yaml`。
- 当前 git commit：`git rev-parse HEAD`。
- 当前 diff：`git diff > diff.patch`。
- 环境：`pip freeze > pip_freeze.txt` 或 `conda env export > env.yml`。
- 启动命令：`cmd.txt`。
- 指标汇总：`metrics.json` 或 `metrics.csv`。

如果暂时不改代码自动保存，可以先用脚本包一层。

### 7.5 实验对比原则

为了比较可信：

- 先复现 baseline，再改 SS2D。
- 除目标变量外，其余配置保持一致。
- 小实验可以单 seed；若要正式汇报，建议至少 3 个 seed。
- 记录 mean/std，不只报最好值。
- 保留失败实验的日志，尤其是 loss 爆炸、显存不足、训练速度明显变慢等信息。

### 7.6 推荐引入工具

轻量方案：

- Git + YAML configs + 规范化 output dir + CSV 汇总。

中等方案：

- TensorBoard：记录 loss、mAP、Rank-1、学习率。
- `scripts/run_exp.sh`：统一启动命令和元信息保存。

较完整方案：

- Weights & Biases 或 MLflow：记录参数、指标、artifact、图片可视化。
- DVC：如果后续有多个数据集版本、伪标签、特征缓存，可以管理数据和中间产物。

## 8. 最小实施路线总结

若只做一个 Market1501 小实验，建议按以下最小路线实现：

1. 在 `config/defaults.py` 增加 `MODEL.SPATIAL_BRANCH_TYPE` 和 SS2D 参数。
2. 在 `climb/model.py` 导入 `SS2D`，新增 `SS2DTokenAdapter`。
3. 在 `CLIMB.__init__` 中根据配置选择 BiMamba 或 SS2D adapter。
4. 在 `CLIMB.forward` 中，当 `SPATIAL_BRANCH_TYPE='ss2d'` 时跳过 reorder，直接对原始 patch grid 做 SS2D。
5. 在 `climb/processor_climb.py` 将日志/图片标题中的 `BiMamba` 改成动态分支名。
6. 新增 baseline 与 SS2D 两份 Market1501 实验配置，不覆盖原配置。
7. 先跑 1 epoch / 2 iters smoke test，再跑正式小实验。
