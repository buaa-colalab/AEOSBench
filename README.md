<div align="center">

<h1>Towards Realistic Earth-Observation Constellation Scheduling: Benchmark and Methodology</h1>

<div>
    Luting Wang<sup>*,1</sup>&emsp;
    Yinghao Xiang<sup>*,1</sup>&emsp;
    Hongliang Huang<sup>1</sup>
    Dongjun Li<sup>1</sup>
    Chen Gao<sup>&dagger;</sup>
    Si Liu<sup>&dagger;</sup>
</div>
<div>
    <sup>1</sup>Beihang University&emsp;
</div>

<div>
    <strong>NeurIPS 2025</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://arxiv.org/abs/2510.26297" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2510.26297-b31b1b.svg">
        </a>
        <a href="https://neurips.cc/virtual/2025/loc/san-diego/poster/116515" target='_blank'>
        <img src="https://img.shields.io/badge/NeurIPS-2025-purple">
        </a>
        <a href="URL_HERE" target='_blank'>
        <img src="https://img.shields.io/badge/Project-Page-green">
        </a>
        <a href="#-citation" target='_blank'>
        <img src="https://img.shields.io/badge/Cite-BibTeX-blue">
        </a>
    </h4>
</div>

<strong>AEOS-Bench is a realistic benchmark and methodology for Earth-Observation constellation scheduling, providing high-fidelity orbital simulation, large-scale task sets, and reference implementations of representative baselines.</strong>

<!-- <div style="text-align:center">
<img src="assets/teaser.png"  width="100%" height="100%">
</div> -->


---

</div>

## 📢 News

* **[2025-10-30]** 🔥 Our paper "Towards Realistic Earth-Observation Constellation Scheduling: Benchmark and Methodology" is released on [arXiv](https://arxiv.org/abs/2510.26297). Code and evaluation scripts are open-sourced.

## 💡 Highlights

* **Realistic Benchmark**. We construct a realistic constellation scheduling benchmark with high-fidelity orbital dynamics, diverse satellite configurations, and large-scale observation tasks.
* **Unified Methodology**. We provide a unified framework that bridges classical optimization and learning-based approaches, including reference implementations of representative baselines.
* **Strong Performance**. Our proposed method achieves State-of-the-Art performance on the proposed benchmark across multiple evaluation splits.

## 🛠️ Usage

### Installation

We recommend using `Conda` / `pipenv` for environment management:

```bash
sudo apt install ffmpeg libpq-dev
bash setup.sh
```

`setup.sh` will create a `pipenv` environment with Python 3.11.10 and install the required dependencies (PyTorch 2.6.0+cu124, gymnasium, stable-baselines3, etc.).

### Data Preparation

If you want to use the full dataset to reproduce our paper:

```bash
git clone git@hf.co:datasets/MessianX/AEOS-dataset ./data
find ./data -type f -name '*.tar' -print0 \
    | xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

Or, if you only want to evaluate your own model, you can download the
`val_seen` / `val_unseen` / `test` trajectories from our HuggingFace repo and
unzip them:

```bash
# TODO: urls
# suppose you have downloaded the requested tarballs into ./data
find ./data -type f -name '*.tar' -print0 \
    | xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

After extraction, the file tree should look like:

```
data/
├── trajectories.1/
│   ├── test/
│   ├── train/
│   │   ├── 00/         # contains pth and json files
│   │   ├── 01/
│   │   └── ...
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

### Training

Use the command below to train our model. It will continue until 200,000 iterations.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=:${PYTHONPATH} \
    auto_torchrun -m constellation.new_transformers.train \
    test constellation/new_transformers/config.py
```

### Evaluation

Use the command below to evaluate a trained model:

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 \
    python -m constellation.rl.eval_all \
    work_dir_name \
    constellation/rl/config_eval.py \
    --load-model-from 'work_dirs/test/checkpoints/iter_100000/model.pth'
```

## 📝 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{wang2025towards,
  title={Towards Realistic Earth-Observation Constellation Scheduling: Benchmark and Methodology},
  author={Wang, Luting and Xiang, Yinghao and Huang, Hongliang and Li, Dongjun and Gao, Chen and Liu, Si},
  journal={arXiv preprint arXiv:2510.26297},
  year={2025}
}
```

## 📄 License

This project is licensed under the Apache-2.0 License. See [LICENSE](./LICENSE) for more information.

## 🙏 Acknowledgement

This project builds upon several outstanding open-source projects, including [Basilisk](https://hanspeterschaub.info/basilisk/). We sincerely thank the authors for their valuable contributions to the community.
