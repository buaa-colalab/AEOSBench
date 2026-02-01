# Constellation

This repository is the official implementation of "Towards Realistic Earth-Observation Constellation Scheduling: Benchmark and Methodology".

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-purple)](https://neurips.cc/virtual/2025/loc/san-diego/poster/116515)
[![arXiv](https://img.shields.io/badge/arXiv-2510.26297-b31b1b.svg)](https://arxiv.org/abs/2510.26297)

## Installation

```bash
sudo apt install ffmpeg libpq-dev
bash setup.sh
```

## data

If you want to use all data to reproduct our paper:

```bash
git clone git@hf.co:datasets/MessianX/AEOS-dataset ./data
find ./data -type f -name '*.tar' -print0 | xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

Or, you can just download the val_seen/val_unseen/test from the trajectories inside our hf repo and unzip them to only evaluate your own model:

```bash
# TODO: urls
# suppose you have download these requested data
find ./data -type f -name '*.tar' -print0 | xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

## Steps

### 1. Confirm the data

The right file tree should be like this：

```
data/
├── trajectories.1/
│   ├── test/
│   ├── train/
│   │   ├── 00/         # contains pth and json files
│   │   ├── 01/
│   │   ├── ...
│   ├── val_seen/
│   └── val_unseen/
├── trajectories.2/
├── trajectories.3/
├── annotations/
│   ├── test.json
│   ├── train.json
│   ├── val_seen.json
│   └── val_unseen.json
├── constellations/
│   ├── test/
│   ├── train/
│   ├── val_seen/
│   └── val_unseen/
├── orbits/
├── satellites/
└── tasksets/
```

### 2. Train the model

Use the command below to train our model:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.train test constellation/new_transformers/config.py
```

This will continue till 200000 iters.

### 3. Eval the model

Use the command below to evaluate the model:

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 python -m constellation.rl.eval_all \
    work_dir_name \
    constellation/rl/config_eval.py \
    --load-model-from 'work_dirs/test/checkpoints/iter_100000/model.pth'
```

