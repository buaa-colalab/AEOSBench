PYTHONPATH=:${PYTHONPATH} python tools/generate_mrp_taskset.py

mkdir -p data/satellites/train
ln -s ${PWD}/data/satellites/train data/satellites/val_seen
mkdir -p data/satellites/val_unseen
mkdir -p data/satellites/test

PYTHONPATH=:${PYTHONPATH} torchrun --nproc-per-node 32 tools/generate_satellites.py
