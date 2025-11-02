import argparse
import todd
import torch
from todd.patches.py_ import json_dump, json_load
from tqdm import trange

from constellation import ANNOTATIONS_ROOT, TRAJECTORIES_ROOT

import pathlib

ANNOTATIONS_ROOT = pathlib.Path('data/annotations.tabu.1')
TRAJECTORIES_ROOT = pathlib.Path('data/trajectories.tabu.1')


def generate(split: str, n: int, completion_rate_threshold: float) -> None:
    annotations: list[int] = []
    metrics_list: list[float] = []

    trajectories_root = TRAJECTORIES_ROOT / split
    for i in trange(n):
        trajectory_path = (
            trajectories_root / f'{i // 1000:02}' / f'{i:05}.json'
        )
        metrics = json_load(str(trajectory_path))
        if metrics[0] > completion_rate_threshold:
            annotations.append(i)
            metrics_list.append(metrics)

    todd.logger.info("%s: %d/%d", split, len(annotations), n)
    todd.logger.info(
        "Average completion rate: %s",
        torch.tensor(metrics_list).mean(0),
    )

    if todd.Store.DRY_RUN:
        return

    annotations_path = ANNOTATIONS_ROOT / f'{split}.json'
    json_dump(annotations, str(annotations_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate annotations')
    parser.add_argument('--threshold', type=float, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ANNOTATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    generate('train', 50_000, args.threshold)
    generate('val_seen', 500, args.threshold)
    generate('val_unseen', 500, args.threshold)
    generate('test', 1_000, args.threshold)


if __name__ == '__main__':
    main()
