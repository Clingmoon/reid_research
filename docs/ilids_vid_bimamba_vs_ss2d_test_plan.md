# iLIDS-VID 上 BiMamba 与 VMamba SS2D 对照测试方案

本文档用于规划在 `/home/cfdeng/projects/CLIMB-ReID/datasets/reid-datasets/iLIDS-VID` 上测试当前 CLIMB-ReID 的原始 BiMamba 分支与已接入的 VMamba SS2D 分支。本文只制定测试方案，不直接修改训练代码。

## 1. 当前结论

当前仓库的 `train_climb.py` + `climb/dataloader.py` 路径是 **image ReID 训练流程**，目前只在 `climb/dataloader.py` 的 `FACTORY` 中注册了：

- `market1501`
- `msmt17`

也就是说，虽然 README 提到支持 MARS、LS-VID、iLIDS-VID，但当前工作区里的实际训练入口还没有 iLIDS-VID 的视频 dataloader、tracklet dataset、视频采样和视频评估协议。因此不能直接把配置里的 `DATASETS.NAMES` 改成 `iLIDS-VID` 就跑。

本次测试建议分两阶段：

1. **阶段 A：最小可执行视频测试**。先补 iLIDS-VID 数据集解析、tracklet 采样和序列级平均特征，公平比较 BiMamba 与 SS2D。
2. **阶段 B：更接近论文的完整视频测试**。在阶段 A 跑通后，再实现/复现论文中的 MTM/MIF 多时间尺度建模，对比原始视频 CLIMB 与 SS2D 版本。

若目标是快速判断 “Market1501 上 SS2D 效果一般是否是因为 image dataset 不适合”，建议先做阶段 A。

## 2. iLIDS-VID 数据检查结果

数据路径：

```text
/home/cfdeng/projects/CLIMB-ReID/datasets/reid-datasets/iLIDS-VID
```

当前目录是软链接，实际指向：

```text
/home/data/users/cfdeng/tdh/datasets/ilids-vid
```

已检查到的数据结构：

```text
iLIDS-VID/
  i-LIDS-VID/
    sequences/
      cam1/personXXX/*.png
      cam2/personXXX/*.png
    images/
      cam1/...
      cam2/...
  train-test people splits/
    train_test_splits_ilidsvid.mat
```

基本统计：

| 项目 | 数值 |
| --- | --- |
| person 数 | 300 |
| camera 数 | 2 |
| cam1 tracklet 数 | 300 |
| cam2 tracklet 数 | 300 |
| cam1 帧数 min/mean/max | 22 / 65.67 / 192 |
| cam2 帧数 min/mean/max | 23 / 75.86 / 172 |
| split 文件 | `train_test_splits_ilidsvid.mat` |
| split 数 | 10 |
| 每个 split 训练 ID | 150 |
| 每个 split 测试 ID | 150 |

`train_test_splits_ilidsvid.mat` 中的 `ls_set` 形状是 `(10, 300)`。建议约定每行前 150 个 ID 作为 train，后 150 个 ID 作为 test；这是常见 iLIDS-VID 10-split 协议。

## 3. 推荐实验目标

### 3.1 主问题

在同一个 iLIDS-VID split、同一个视频采样、同一个训练策略下，对比：

1. **BiMamba baseline**：当前原始 `reorder + BiMamba` 分支。
2. **VMamba SS2D**：当前新接入的 `SS2D spatial branch`，不使用 reorder，对每帧 patch grid 做二维四向扫描。

### 3.2 评价指标

每个 split 输出：

- Rank-1
- Rank-5
- Rank-10
- Rank-20
- mAP，若当前评估工具方便支持；视频 ReID 经典报告通常重点看 CMC，建议保留 mAP 作为补充。

最终报告：

- 10 splits 的 mean/std。
- 同时报告单 split 0 的调试结果，方便快速迭代。

## 4. 阶段 A：最小可执行视频测试

阶段 A 的核心思路：**不先实现复杂 MTM，只把视频 tracklet 采样成 T 帧，每帧走当前 CLIMB image encoder + branch，最后在视频维度平均成 tracklet embedding。**

这样可以最大程度复用当前 Market1501 已跑通的代码，并公平比较 BiMamba 与 SS2D 分支。

### 4.1 需要补的代码模块

建议新增/修改：

```text
datasets/ilidsvid.py
climb/video_dataset.py
climb/video_dataloader.py
climb/model.py
climb/processor_climb.py 或新增 climb/processor_video_climb.py
config/experiments/ilidsvid/
```

#### 4.1.1 `datasets/ilidsvid.py`

功能：解析 iLIDS-VID 数据集，输出 train/query/gallery tracklet 列表。

建议返回格式：

```python
tracklet = (tuple(frame_paths), pid, camid, trackid)
```

split 规则：

- 读取 `train_test_splits_ilidsvid.mat` 的 `ls_set`。
- `split_id=0` 默认使用第 1 个 split。
- `train_ids = ls_set[split_id, :150]`
- `test_ids = ls_set[split_id, 150:]`
- 训练集包含 test split 前 150 个 ID 的 cam1 + cam2 tracklets。
- 测试集：
  - query：cam1 的 test IDs。
  - gallery：cam2 的 test IDs。
- 为减少 camera 方向偏置，可以增加 `EVAL.FLIP_QUERY_GALLERY=True` 做 cam2→cam1，再与 cam1→cam2 平均；第一轮先不做。

ID 映射：

- train pid 需要 relabel 为 `0..149`。
- test pid 可保留原始 person id 或 relabel，但 query/gallery 必须一致。

#### 4.1.2 `climb/video_dataset.py`

功能：将一个 tracklet 采样成固定 `T` 帧。

配置建议：

```yaml
VIDEO:
  NUM_FRAMES: 4
  SAMPLING: 'even'      # train 可用 random/even；eval 建议 even
  TRAIN_SAMPLING: 'random'
  TEST_SAMPLING: 'even'
```

输出张量：

```python
imgs: [T, C, H, W]
pid: int
camid: int
trackid: int
path_key: str
```

训练采样：

- tracklet 帧数 >= T：随机采样或随机起点均匀采样。
- tracklet 帧数 < T：重复最后一帧或循环补齐。

测试采样：

- 均匀采样 T 帧。
- 若显存允许，可做 multi-clip 测试，第一轮不建议。

#### 4.1.3 `climb/model.py`

当前 `CLIMB.forward` 只接收 `[B, C, H, W]`。建议最小改动支持 `[B, T, C, H, W]`：

1. 如果输入是 5D：reshape 成 `[B*T, C, H, W]`。
2. 调用现有 image forward。
3. 将输出 reshape 回 `[B, T, D]`。
4. 对 T 做 mean pooling。
5. 训练时 classifier 对 pooled feature 计算。

需要保持两个分支公平：

- BiMamba：每帧独立做 `reorder + BiMamba`，再 T 平均。
- SS2D：每帧独立做 `SS2D`，再 T 平均。

这不是论文完整 MTM，但能公平回答“同一视频采样下 BiMamba 与 SS2D 哪个分支更适合”。

#### 4.1.4 `climb/video_dataloader.py`

建议单独新增，不要改坏现有 image dataloader。

训练 loader：

- 以 tracklet 为样本。
- 使用 PK sampler，按 pid 采样 tracklets。
- iLIDS 每个 ID 只有两个 camera tracklets，建议 `NUM_INSTANCE=2`。

评估 loader：

- query + gallery tracklets。
- `num_query = len(query)`。

cluster loader：

- 用训练 tracklets，测试 transform，采样 T 帧。
- 用于 memory bank 构建。

### 4.2 阶段 A 配置建议

新增目录：

```text
config/experiments/ilidsvid/
```

新增两个配置：

```text
config/experiments/ilidsvid/climb-ilidsvid-bimamba-split0.yml
config/experiments/ilidsvid/climb-ilidsvid-ss2d-split0.yml
```

共同配置建议：

```yaml
MODEL:
  NAME: 'ViT-B-16'
  STRIDE_SIZE: [16, 16]
  MEMORY_MOMENTUM: 0.2
  ID_LOSS_WEIGHT: 0.25
  PCL_LOSS_WEIGHT: 1.0

INPUT:
  SIZE_TRAIN: [256, 128]
  SIZE_TEST: [256, 128]
  PROB: 0.5
  RE_PROB: 0.5
  PADDING: 10
  PIXEL_MEAN: [0.5, 0.5, 0.5]
  PIXEL_STD: [0.5, 0.5, 0.5]

VIDEO:
  ENABLED: True
  NUM_FRAMES: 4
  TRAIN_SAMPLING: 'random'
  TEST_SAMPLING: 'even'

DATASETS:
  NAMES: ('ilidsvid')
  ROOT_DIR: ('/home/cfdeng/projects/CLIMB-ReID/datasets/reid-datasets')
  SPLIT_ID: 0

DATALOADER:
  NUM_INSTANCE: 2
  NUM_WORKERS: 0

SOLVER:
  IMS_PER_BATCH: 16
  OPTIMIZER_NAME: 'SGD'
  BASE_LR: 3.5e-4
  WARMUP_METHOD: 'linear'
  WARMUP_ITERS: 10
  WARMUP_FACTOR: 0.1
  WEIGHT_DECAY: 5.0e-4
  MAX_EPOCHS: 60
  EVAL_PERIOD: 10
  CHECKPOINT_PERIOD: 60
  LOG_PERIOD: 20
  ITERS: 50
  STEPS: [30, 50]
  GAMMA: 0.1

TEST:
  IMS_PER_BATCH: 16
  RE_RANKING: False
```

BiMamba 配置差异：

```yaml
MODEL:
  SPATIAL_BRANCH_TYPE: 'bimamba'
OUTPUT_DIR: './logs/ilidsvid_split0_bimamba_t4_seed1234'
```

SS2D 配置差异：

```yaml
MODEL:
  SPATIAL_BRANCH_TYPE: 'ss2d'
  SS2D_FORWARD_TYPE: 'v1'
  SS2D_D_STATE: 16
  SS2D_SSM_RATIO: 2.0
  SS2D_D_CONV: 3
OUTPUT_DIR: './logs/ilidsvid_split0_ss2d_v1_t4_seed1234'
```

说明：

- `NUM_FRAMES=4` 是为了在 10GB RTX 3080 上先跑通。
- 若显存足够，再测试 `NUM_FRAMES=8`。
- `IMS_PER_BATCH=16` 是视频输入的保守起点；若 OOM，降到 8。
- `TEST.IMS_PER_BATCH=16`，避免 Market 实验里遇到的 cache/显存问题。

### 4.3 阶段 A 运行顺序

#### 4.3.1 smoke test

先跑单 split、单 epoch、极少 iteration。

BiMamba：

```bash
CUDA_VISIBLE_DEVICES=0 python train_climb_video.py \
  --config_file config/experiments/ilidsvid/climb-ilidsvid-bimamba-split0.yml \
  SOLVER.MAX_EPOCHS 1 SOLVER.ITERS 2 SOLVER.EVAL_PERIOD 1 \
  TEST.IMS_PER_BATCH 8 SOLVER.IMS_PER_BATCH 8 \
  OUTPUT_DIR "'./logs/debug_ilidsvid_bimamba_t4_1ep'"
```

SS2D：

```bash
CUDA_VISIBLE_DEVICES=0 python train_climb_video.py \
  --config_file config/experiments/ilidsvid/climb-ilidsvid-ss2d-split0.yml \
  SOLVER.MAX_EPOCHS 1 SOLVER.ITERS 2 SOLVER.EVAL_PERIOD 1 \
  TEST.IMS_PER_BATCH 8 SOLVER.IMS_PER_BATCH 8 \
  OUTPUT_DIR "'./logs/debug_ilidsvid_ss2d_t4_1ep'"
```

如果不新增 `train_climb_video.py`，也可以在 `train_climb.py` 中根据 `VIDEO.ENABLED` 分流到 video dataloader 和 video processor。但为了不影响 Market/MSMT，建议新增入口。

#### 4.3.2 split0 正式测试

BiMamba：

```bash
CUDA_VISIBLE_DEVICES=0 python train_climb_video.py \
  --config_file config/experiments/ilidsvid/climb-ilidsvid-bimamba-split0.yml
```

SS2D：

```bash
CUDA_VISIBLE_DEVICES=0 python train_climb_video.py \
  --config_file config/experiments/ilidsvid/climb-ilidsvid-ss2d-split0.yml
```

#### 4.3.3 10 splits 完整测试

建立 split 配置或用命令行覆盖：

```bash
for split in 0 1 2 3 4 5 6 7 8 9; do
  CUDA_VISIBLE_DEVICES=0 python train_climb_video.py \
    --config_file config/experiments/ilidsvid/climb-ilidsvid-bimamba-split0.yml \
    DATASETS.SPLIT_ID ${split} \
    OUTPUT_DIR "'./logs/ilidsvid_split${split}_bimamba_t4_seed1234'"

  CUDA_VISIBLE_DEVICES=0 python train_climb_video.py \
    --config_file config/experiments/ilidsvid/climb-ilidsvid-ss2d-split0.yml \
    DATASETS.SPLIT_ID ${split} \
    OUTPUT_DIR "'./logs/ilidsvid_split${split}_ss2d_v1_t4_seed1234'"
done
```

建议先只跑 split0，确认趋势和显存，再跑 10 splits。

## 5. 阶段 B：更接近论文的完整视频测试

阶段 A 的视频建模是 “per-frame branch + temporal average pooling”，没有复现论文的完整 Multi-Temporal Mamba。

如果阶段 A 结果显示 SS2D 有潜力，再做阶段 B：

### 5.1 BiMamba/官方视频版本

根据论文，完整视频 CLIMB 应包含：

- 多时间尺度切片：`S = [1, 4, ..., T]`
- fragment-level cls token
- importance-aware reorder
- forward/reverse Mamba
- multi-scale information fusion

当前仓库 `climb/model.py` 实际更接近 image-based IRM 的简化版本，没有完整 MTM。若要严谨复现视频论文结果，需要：

1. 找到官方视频代码或下载 README 中 iLIDS/MARS 的 Model&Code。
2. 或按论文重新实现 `VideoCLIMB/MTM`。

### 5.2 SS2D 视频版本候选

可比较两种 VMamba 思路：

1. **Frame-wise SS2D + temporal Mamba/mean pooling**：每帧二维 SS2D，帧间用平均池化或 1D Mamba。
2. **Spatio-temporal factorized SSM**：先每帧 SS2D，再沿时间维做 BiMamba/Mamba。

不建议第一步直接把 `T*H*W` 强行 reshape 成二维伪图输入 SS2D，因为二维邻接关系会混乱。

## 6. 公平对比控制变量

BiMamba 和 SS2D 必须保持一致：

- 同一个 split。
- 同一个 `NUM_FRAMES`。
- 同一个 frame sampling 策略。
- 同一个 image size。
- 同一个 batch size。
- 同一个 seed。
- 同一个 optimizer/lr/scheduler。
- 同一个 eval protocol。
- 同一个是否使用 reranking。

唯一变化：

```yaml
MODEL.SPATIAL_BRANCH_TYPE: 'bimamba' vs 'ss2d'
```

SS2D 额外固定：

```yaml
SS2D_FORWARD_TYPE: 'v1'
SS2D_D_STATE: 16
SS2D_SSM_RATIO: 2.0
SS2D_D_CONV: 3
```

## 7. 建议记录表

| split | method | T | batch | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP | best epoch | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | BiMamba | 4 | 16 |  |  |  |  |  |  |  |
| 0 | SS2D-v1 | 4 | 16 |  |  |  |  |  |  |  |
| mean±std | BiMamba | 4 | 16 |  |  |  |  |  |  | 10 splits |
| mean±std | SS2D-v1 | 4 | 16 |  |  |  |  |  |  | 10 splits |

同时记录：

- `git rev-parse HEAD`
- `git diff > diff.patch`
- `pip freeze > pip_freeze.txt`
- 实际命令 `cmd.txt`
- 完整配置 `config.yaml`

## 8. 风险与注意事项

### 8.1 当前代码不能直接跑 iLIDS-VID

当前 `make_CLIMB_dataloader` 只支持 image dataset。若直接设：

```yaml
DATASETS:
  NAMES: ('ilidsvid')
```

会在 `FACTORY` 查找时报错。必须先补 `datasets/ilidsvid.py` 和 video dataloader。

### 8.2 iLIDS-VID 数据很小

iLIDS-VID 只有 300 个 ID，每个 split 训练 150 个 ID。单 split 结果波动较大，不建议只看 split0 得结论。

### 8.3 显存设置要保守

SS2D 在 backward 中显存压力高于 BiMamba。RTX 3080 10GB 建议起步：

```yaml
VIDEO.NUM_FRAMES: 4
SOLVER.IMS_PER_BATCH: 8 或 16
TEST.IMS_PER_BATCH: 8 或 16
```

如果 OOM，优先降：

1. `TEST.IMS_PER_BATCH`
2. `SOLVER.IMS_PER_BATCH`
3. `VIDEO.NUM_FRAMES`

### 8.4 阶段 A 不是完整视频 CLIMB

阶段 A 是为了快速做公平工程对照，不等价于论文完整 MTM。报告时应写清：

```text
We evaluate frame-wise CLIMB variants on iLIDS-VID by sampling T frames per tracklet and averaging frame-level embeddings.
```

若要声称复现论文视频 CLIMB，需要进入阶段 B。

## 9. 推荐下一步实施顺序

1. 新增 iLIDS-VID dataset parser，确认 split0 train/query/gallery 数量正确。
2. 新增 `VideoTrackletDataset`，采样 `[T, C, H, W]`。
3. 新增 video dataloader，返回 train/val/cluster loaders。
4. 在模型 forward 支持 `[B, T, C, H, W]` 输入并做 temporal mean pooling。
5. 新增 `train_climb_video.py` 或在 `train_climb.py` 中按 `VIDEO.ENABLED` 分流。
6. 新增两个 split0 配置：BiMamba 与 SS2D。
7. 跑 smoke test。
8. 跑 split0 正式训练。
9. 若趋势有意义，跑 10 splits 并汇总 mean/std。

