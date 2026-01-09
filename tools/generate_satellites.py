import argparse
import os
import pathlib

import todd

from constellation import (
    SATELLITES_ROOT,
    TASKSETS_ROOT,
    Controller,
    TaskManager,
)
from constellation.algorithms import OptimalAlgorithm
from constellation.data import Constellation, TaskSet
from constellation.environments import BasiliskEnvironment
from constellation.evaluators import CompletionRateEvaluator

RANK = int(os.environ['RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])

TASKSET_PATH = TASKSETS_ROOT / 'mrp.json'
TASKSET = TaskSet.load(str(TASKSET_PATH))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    return args


def generate_satellites(
    split: str,
    n: int,
    threshold: float,
) -> None:
    satellites_root: pathlib.Path = SATELLITES_ROOT / split
    max_id = max(
        [int(satellite_path.stem) for satellite_path in satellites_root.iterdir()],
        default=-1,
    )

    for i in range(RANK, n, WORLD_SIZE):
        if i <= max_id:
            continue

        constellation = Constellation.sample_mrp()
        environment = BasiliskEnvironment(
            constellation=constellation,
            all_tasks=TASKSET,
        )
        task_manager = TaskManager(timer=environment.timer, tasks=TASKSET)
        algorithm = OptimalAlgorithm(timer=environment.timer)
        algorithm.prepare(environment, task_manager)
        controller = Controller(
            environment=environment,
            task_manager=task_manager,
            callbacks=[CompletionRateEvaluator()],
        )

        try:
            metrics = controller.run(0, algorithm, progress_bar=False)
        except Exception as e:
            todd.logger.error("rank %d failed %d: %s", RANK, i, e)
            continue

        todd.logger.info("rank %d finished %d with %s", RANK, i, metrics['CR'])
        if metrics['CR'] > threshold:
            constellation.dump(str(satellites_root / f'{i}.json'))


def main() -> None:
    args = parse_args()

    generate_satellites('train', 10_000, args.threshold)
    generate_satellites('val_unseen', 2_000, args.threshold)
    generate_satellites('test', 2_000, args.threshold)


if __name__ == "__main__":
    main()
