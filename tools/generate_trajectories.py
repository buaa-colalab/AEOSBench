import argparse
import multiprocessing
import pathlib
from functools import partial

import todd
from todd.patches.py_ import json_dump, json_load

from constellation import CONSTELLATIONS_ROOT, TASKSETS_ROOT, TRAJECTORIES_ROOT
from constellation.algorithms import OptimalAlgorithm
from constellation.controller import Controller
from constellation.data import Constellation, Task, TaskSet
from constellation.environments import BasiliskEnvironment
from constellation.evaluators import (
    CompletionRateEvaluator,
    PCompletionRateEvaluator,
    PowerEvaluator,
    TurnAroundTimeEvaluator,
    WCompletionRateEvaluator,
    WPCompletionRateEvaluator,
)
from constellation.task_managers import TaskManager
from constellation.loggers import PthLogger, VisualizationLogger

MAX_RETRY = 1
COMPLETION_RATE_THRESHOLD = 0.01


def generate_trajectory(
    split: str,
    i: int,
    tabu: pathlib.Path | None = None,
) -> list[float] | None:
    path = f'{split}/{i // 1000:02}/{i:05}.json'
    constellation_path = CONSTELLATIONS_ROOT / path
    tasks_path = TASKSETS_ROOT / path
    trajectory_path = TRAJECTORIES_ROOT / path
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory_path.exists():
        metrics = json_load(str(trajectory_path))
        if metrics[0] > COMPLETION_RATE_THRESHOLD:
            todd.logger.info(f'{split=} {i=} already exists')
            return None

    if tabu is None:
        tabu_path = None
    else:
        tabu_path = tabu / path
        tabu_path = tabu_path.with_suffix('.tabu.json')
        if not tabu_path.exists():
            todd.logger.info(f'{split=} {i=} tabu path not exists')
            return None

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
        PowerEvaluator(),
        PthLogger(work_dir=trajectory_path.parent),
    ]
    task_manager = TaskManager(timer=environment.timer, tasks=taskset)
    algorithm.prepare(environment=environment, task_manager=task_manager)
    controller = Controller(
        environment=environment,
        task_manager=task_manager,
        callbacks=callbacks,
    )

    metrics = controller.run(i, algorithm)

    json_dump(metrics, str(trajectory_path))

    return metrics


def generate_trajectory_retry(split: str, i: int, **kwargs) -> None:
    for retry in range(MAX_RETRY):
        todd.logger.info(f"{retry=} {split=} {i=}")

        # try:
        metrics = generate_trajectory(split, i, **kwargs)
        # except:
        #     todd.logger.exception(f"{retry=} {split=} {i=}")
        #     completion_rate = 0.0

        if metrics is None:
            return
        if metrics[0] > COMPLETION_RATE_THRESHOLD:
            todd.logger.info(f"{split=} {i=} {metrics=}")
            return

    todd.logger.info(f"{split=} {i=} failed")


def generate(num_workers: int, split: str, n: int, **kwargs) -> None:
    if num_workers == 0:
        for i in range(n):
            generate_trajectory_retry(split, i, **kwargs)
        return

    with multiprocessing.Pool(num_workers) as pool:
        list(
            pool.imap_unordered(
                partial(generate_trajectory_retry, split, **kwargs),
                range(n),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate trajectories')
    parser.add_argument('num_workers', type=int)
    parser.add_argument('--tabu', type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.num_workers, 'train', 50_000, tabu=args.tabu)
    generate(args.num_workers, 'val_seen', 500, tabu=args.tabu)
    generate(args.num_workers, 'val_unseen', 500, tabu=args.tabu)
    generate(args.num_workers, 'test', 1_000, tabu=args.tabu)


if __name__ == '__main__':
    main()
