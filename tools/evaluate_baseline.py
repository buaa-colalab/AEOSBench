import argparse
import pathlib

import todd
import torch
from todd.patches.py_ import json_load

from constellation import ANNOTATIONS_ROOT


def evaluate(work_dir: pathlib.Path, split: str) -> None:
    annotations: list[int] = json_load(str(ANNOTATIONS_ROOT / f'{split}.tiny.json'))
    metrics = torch.tensor([
        json_load(str(work_dir / split / f'{i // 1000:02}' / f'{i:05}.json'))
        for i in annotations
    ])
    todd.logger.info("%s: %s", split, metrics.mean(0).tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate baseline")
    parser.add_argument('name')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    work_dir: pathlib.Path = (pathlib.Path('work_dirs/test_baseline') / args.name)

    evaluate(work_dir, 'val_seen')
    evaluate(work_dir, 'val_unseen')
    evaluate(work_dir, 'test')


if __name__ == '__main__':
    main()
