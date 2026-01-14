import argparse
import multiprocessing
import os
import pathlib
from functools import partial

import todd
from todd.patches.py_ import json_dump, json_load

from constellation import CONSTELLATIONS_ROOT, TASKSETS_ROOT, TRAJECTORIES_ROOT
from constellation.algorithms import OptimalAlgorithm
from constellation.controller import Controller
from constellation.data import Constellation, TaskSet
from constellation.environments import BasiliskEnvironment
from constellation.evaluators import (
    CompletionRateEvaluator,
    PCompletionRateEvaluator,
    PowerUsageEvaluator,
    TurnAroundTimeEvaluator,
    WCompletionRateEvaluator,
    WPCompletionRateEvaluator,
)
from constellation.task_managers import TaskManager
from constellation.loggers import TrajectoryLogger

RANK = int(os.environ['RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])

MAX_RETRY = 1
COMPLETION_RATE_THRESHOLD = 0.01


def generate_trajectory(
    split: str,
    i: int,
    tabu: pathlib.Path | None = None,
) -> bool:
    path = f'{split}/{i // 1000:02}/{i:05}.json'
    constellation_path = CONSTELLATIONS_ROOT / path
    tasks_path = TASKSETS_ROOT / path
    trajectory_path = TRAJECTORIES_ROOT / path
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory_path.exists():
        metrics = json_load(str(trajectory_path))
        if metrics[0] > COMPLETION_RATE_THRESHOLD:
            todd.logger.info('split=%s i=%s already exists', split, i)
            return True

    if tabu is None:
        tabu_path = None
    else:
        tabu_path = tabu / path
        tabu_path = tabu_path.with_suffix('.tabu.json')
        assert tabu_path.exists()

    taskset = TaskSet.load(str(tasks_path))
    constellation = Constellation.load(str(constellation_path))

    environment = BasiliskEnvironment(
        constellation=constellation,
        all_tasks=taskset,
    )
    algorithm = OptimalAlgorithm(timer=environment.timer)

    callbacks = [
        CompletionRateEvaluator(),
        PCompletionRateEvaluator(),
        WCompletionRateEvaluator(),
        WPCompletionRateEvaluator(),
        TurnAroundTimeEvaluator(),
        PowerUsageEvaluator(),
        TrajectoryLogger(work_dir=trajectory_path.parent),
    ]
    task_manager = TaskManager(timer=environment.timer, tasks=taskset)
    algorithm.prepare(environment=environment, task_manager=task_manager)
    controller = Controller(
        environment=environment,
        task_manager=task_manager,
        callbacks=callbacks,
    )

    metrics = controller.run(i, algorithm)

    if metrics[0] > COMPLETION_RATE_THRESHOLD:
        todd.logger.info('split=%s i=%d metrics=%s', split, i, metrics)
        json_dump(metrics, str(trajectory_path))
        return True

    return False


def generate_trajectories(split: str, n: int, **kwargs) -> None:
    for i in range(RANK, n, WORLD_SIZE):
        for retry in range(MAX_RETRY):
            todd.logger.info('retry=%d split=%s i=%d', retry, split, i)
            if generate_trajectory(split, i, **kwargs):
                break
        else:
            todd.logger.info('split=%s i=%d failed', split, i)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate trajectories')
    parser.add_argument('--tabu', type=pathlib.Path)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    generate_trajectories('train', 50_000, tabu=args.tabu)
    generate_trajectories('val_seen', 500, tabu=args.tabu)
    generate_trajectories('val_unseen', 500, tabu=args.tabu)
    generate_trajectories('test', 1_000, tabu=args.tabu)


if __name__ == '__main__':
    main()
