import argparse
import multiprocessing
import pathlib
from functools import partial
import numpy as np

import todd
from todd.patches.py_ import json_dump, json_load
import torch

from constellation import CONSTELLATIONS_ROOT, TASKSETS_ROOT, TRAJECTORIES_ROOT
from constellation.algorithms import TabuOptimalAlgorithm
from constellation import ANNOTATIONS_ROOT
from constellation.controller import Controller
from constellation.data import Constellation, Task, Taskset
from constellation.environments import BasiliskEnvironment
from constellation.evaluators import (
    CompletionRateEvaluator,
    PCompletionRateEvaluator,
    PowerEvaluator,
    TurnAroundTimeEvaluator,
    WCompletionRateEvaluator,
    WPCompletionRateEvaluator,
)


def evaluate(work_dir: pathlib.Path, split: str) -> None:
    annotations: list[int] = json_load(
        str(ANNOTATIONS_ROOT / f'{split}.tiny.json')
    )
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

    work_dir: pathlib.Path = (
        pathlib.Path('work_dirs/test_baseline') / args.name
    )

    evaluate(work_dir, 'val_seen')
    evaluate(work_dir, 'val_unseen')
    evaluate(work_dir, 'test')


if __name__ == '__main__':
    main()
