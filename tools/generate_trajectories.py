import argparse
import os
import pathlib
from typing import Iterable

import todd
from todd.patches.py_ import json_dump, json_load
import torch

from constellation import CONSTELLATIONS_ROOT, TASKSETS_ROOT, TRAJECTORIES_ROOT
from constellation.algorithms import OptimalAlgorithm
from constellation.callbacks import ComposedCallback
from constellation.controller import Controller
from constellation.data import Constellation, TaskSet
from constellation.environments import BasiliskEnvironment
from constellation.evaluators import (
    CompletionRateEvaluator,
    PowerUsageEvaluator,
    TurnAroundTimeEvaluator,
)
from constellation.loggers import ForbidTasksCallback
from constellation.loggers.filter_task import FilterTaskCallback
from constellation.task_managers import TaskManager
from constellation.loggers import TrajectoryLogger

RANK = int(os.environ['RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])

COMPLETION_RATE_THRESHOLD = 0.01


def generate_trajectory(
    split: str,
    i: int,
    previous_trajectories: Iterable[pathlib.Path] | None,
) -> None:
    constellation_path = CONSTELLATIONS_ROOT / split / f'{i // 1000:02}/{i:05}.json'
    tasks_path = TASKSETS_ROOT / split / f'{i // 1000:02}/{i:05}.json'

    trajectory_root = TRAJECTORIES_ROOT / split / f'{i // 1000:02}'
    trajectory_root.mkdir(parents=True, exist_ok=True)

    metrics_path = trajectory_root / f'{i:05}.json'

    if metrics_path.exists():
        metrics = json_load(str(metrics_path))
        if metrics['CR'] > COMPLETION_RATE_THRESHOLD:
            todd.logger.info('split=%s i=%s already exists', split, i)
            return

    taskset = TaskSet.load(str(tasks_path))
    constellation = Constellation.load(str(constellation_path))

    environment = BasiliskEnvironment(
        constellation=constellation,
        all_tasks=taskset,
    )
    task_manager = TaskManager(timer=environment.timer, taskset=taskset)
    callbacks = ComposedCallback(
        callbacks=[
            CompletionRateEvaluator(),
            TurnAroundTimeEvaluator(),
            PowerUsageEvaluator(),
            TrajectoryLogger(work_dir=trajectory_root),
            ForbidTasksCallback(work_dir=trajectory_root),
            FilterTaskCallback(work_dir=trajectory_root),
        ],
    )
    controller = Controller(
        f'{i:05}',
        environment=environment,
        task_manager=task_manager,
        callbacks=callbacks,
    )

    forbidden_task_ids = (
        None if previous_trajectories is None else torch.stack([
            torch.load(
                previous_trajectory / split
                / f'{i // 1000:02}/{i:05}_forbidden_task_ids.pth',
            ) for previous_trajectory in previous_trajectories
        ])
    )
    algorithm = OptimalAlgorithm(
        timer=environment.timer,
        forbidden_task_ids=forbidden_task_ids,
    )
    algorithm.prepare(environment=environment, task_manager=task_manager)
    controller.run(algorithm)

    metrics = controller.memo['metrics']
    todd.logger.info('split=%s i=%d metrics=%s', split, i, metrics)
    if metrics['CR'] > COMPLETION_RATE_THRESHOLD:
        json_dump(metrics, str(metrics_path))


def generate_trajectories(split: str, n: int, *args, **kwargs) -> None:
    for i in range(RANK, n, WORLD_SIZE):
        # i = 16  # TODO
        generate_trajectory(split, i, *args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--previous-trajectories',
        type=pathlib.Path,
        nargs='*',
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    generate_trajectories('train', 50_000, args.previous_trajectories)
    generate_trajectories('val_seen', 500, args.previous_trajectories)
    generate_trajectories('val_unseen', 500, args.previous_trajectories)
    generate_trajectories('test', 1_000, args.previous_trajectories)


if __name__ == '__main__':
    main()
