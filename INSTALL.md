# CLIMB-ReID 安装与训练指南

本文档不改动原始 README，仅补充可执行安装步骤。整体流程与 README 一致：

1. 创建环境
2. 安装项目依赖
3. 安装 VMamba 依赖
4. 准备数据集
5. 启动训练与评估

包管理工具统一使用 `uv`。

## 1. 环境准备

```bash
uv venv --python 3.9 .venv
source .venv/bin/activate
```

## 2. 安装项目依赖

```bash
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118

# 项目 requirements 中有与本地 CUDA 工具链强耦合的扩展包，默认兼容安装如下：
rg -v '^(causal-conv1d|mamba-ssm|fsspec|torch==|torchaudio==|torchvision==)' requirements.txt > /tmp/climb.requirements.compat.txt
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  -r /tmp/climb.requirements.compat.txt
```

## 3. 安装 VMamba 相关依赖

```bash
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
uv pip install --index-strategy unsafe-best-match -r requirements.txt
cd ..
```

说明：
- `selective_scan` 在当前机器（`nvcc 10.1`）无法与 `torch 2.1.1+cu118` 编译匹配，因此默认不强制编译。
- 当前仓库已包含不依赖该扩展的运行回退路径，可正常启动训练。

## 4. 数据集准备

按 README 要求下载并解压：`MARS, LS-VID, iLIDS-VID, Market1501, MSMT17`。

示例目录：

```text
/data/reid/
  Market1501/
    bounding_box_train/
    query/
    bounding_box_test/
  MSMT17_v3/
    train/
    test/
    list_train.txt
    list_val.txt
    list_query.txt
    list_gallery.txt
```

## 5. 训练

README 中写的是 `train-main.py`，当前仓库实际训练入口是 `train_climb.py`。

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python train_climb.py \
  --config_file ./config/climb-vit-market.yml \
  DATASETS.ROOT_DIR "('/data/reid')" \
  OUTPUT_DIR "'./logs_market'"

cd /home/cfdeng/projects/CLIMB-ReID
  source .venv/bin/activate
  source ./scripts/use-cuda11.8.sh
  CUDA_VISIBLE_DEVICES=0 python train_climb.py --config_file ./config/climb-vit-market.yml

cd /home/cfdeng/projects/CLIMB-ReID
  source .venv/bin/activate
  source ./scripts/use-cuda11.8.sh
  CUDA_VISIBLE_DEVICES=0 python train_climb.py --config_file ./config/climb-vit-msmt.yml SOLVER.IMS_PER_BATCH 32 TEST.IMS_PER_BATCH 64
```

## 6. 评估

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python eval_main.py
```

## 一键初始化（推荐）

```bash
bash env-init.sh
```

脚本会完成：
- 环境创建
- 依赖安装
- VMamba 依赖安装
