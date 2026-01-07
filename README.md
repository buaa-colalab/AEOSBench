# Constellation

This repository is the official implementation of "Towards Realistic Earth-Observation Constellation Scheduling: Benchmark and Methodology".

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-purple)](https://neurips.cc/virtual/2025/loc/san-diego/poster/116515)
[![arXiv](https://img.shields.io/badge/arXiv-2510.26297-b31b1b.svg)](https://arxiv.org/abs/2510.26297)

## Installation

```bash
sudo apt install ffmpeg
bash setup.sh
```

## pre-steps

```bash
PYTHONPATH=:${PYTHONPATH} python tools/generate_mrp_taskset.py
PYTHONPATH=:${PYTHONPATH} torchrun --nproc-per-node 32 tools/generate_satellites.py
ln -s ${PWD}/data/satellites/train data/satellites/val_seen
# download test split

PYTHONPATH=:${PYTHONPATH} python tools/generate_constellations_and_tasksets.py

# train transformer model
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.train refactor_test constellation/new_transformers/config.py

# train time model
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.train time_model_refactor constellation/new_transformers/config_timemodel.py

# eval time model
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.val \
    dir_name \
    constellation/new_transformers/config_timemodel.py \
    --load-from iter_20000

PYTHONPATH="./" python -m constellation.rl.eval_with_controller --split test --num-episodes 3 --show-progress

# eval all (using transformer)
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 python -m constellation.rl.eval_all \
    work_dir_name \
    constellation/rl/config_eval.py \
    --load-model-from '/data/wlt/projects/RL/work_dirs/time_model_freeze.pth'
```

## Data

```bash
ln -s ${PWD}/data/satellites/train data/satellites/val_seen

PYTHONPATH=:${PYTHONPATH} python tools/generate_constellations_and_tasksets.py
PYTHONPATH=:${PYTHONPATH} python tools/patch_constellations.py

PYTHONPATH=:${PYTHONPATH} python tools/generate_trajectories.py 400
PYTHONPATH=:${PYTHONPATH} python tools/generate_annotations.py
PYTHONPATH=:${PYTHONPATH} python tools/generate_tabu_lists.py 400
mv data/trajectories data/trajectories.tabu.1

PYTHONPATH=:${PYTHONPATH} python tools/generate_trajectories.py 400 --tabu data/trajectories.tabu.1
PYTHONPATH=:${PYTHONPATH} python tools/generate_tabu_lists.py 400 --merge data/trajectories.tabu.1
mv data/trajectories data/trajectories.tabu.2

PYTHONPATH=:${PYTHONPATH} python tools/generate_trajectories.py 400 --tabu data/trajectories.tabu.2

PYTHONPATH=:${PYTHONPATH} python tools/generate_trajectories.py 400 --tabu data/trajectories.tabu.2

PYTHONPATH=:${PYTHONPATH} python tools/test_baseline.py optimal 100
```

## .Installation

```bash
python -m constellation.controller
torchrun --nproc_per_node=1 --master_port=5000 -m constellation.algorithms.deep.train debug configs/tmp.py
```

## Imitation learning

```bash
PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.train new_data constellation/new_transformers/config.py

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.train time_model_retry constellation/new_transformers/config_timemodel.py

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.val \
    time_model_retry \
    constellation/new_transformers/config_timemodel.py \
    --load-from iter_20000
```

## RL

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 auto_torchrun -m constellation.rl.eval_all \
    new_data \
    rl/config_eval.py \
    --load-model-from work_dirs/new_data/iter_110000.pth

CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=4 RANK=0 python -m constellation.rl.eval_all \
    time_50k \
    rl/config_eval.py \
    --load-model-from '/data/wlt/projects/RL/work_dirs/time_model_freeze.pth'

python -m rl.merge_csvs work_dirs/rl_eval_new_data/completion_rates.csv work_dirs/rl_eval_new_data/completion_rates_*
```
