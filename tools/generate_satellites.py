import argparse

import todd
import torch.distributed as dist
from todd.patches.torch import get_rank, get_world_size
from todd.utils import init_seed

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--taskset-size", type=int, default=36)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    dist.init_process_group()
    rank = get_rank()
    world_size = get_world_size()

    init_seed(args.seed + rank)

    if rank == 0:
        TASKSETS_ROOT.mkdir(parents=True, exist_ok=True)
        SATELLITES_ROOT.mkdir(parents=True, exist_ok=True)

    taskset_path = TASKSETS_ROOT / 'mrp.json'
    if rank == 0 and not taskset_path.exists():
        todd.logger.info("Generating MRP taskset")
        TaskSet.sample_mrp(args.taskset_size).dump(str(taskset_path))
    dist.barrier()
    taskset = TaskSet.load(str(taskset_path))

    for i in range(rank, args.num_samples, world_size):
        constellation = Constellation.sample_mrp()
        environment = BasiliskEnvironment(
            constellation=constellation,
            all_tasks=taskset,
        )
        task_manager = TaskManager(timer=environment.timer, tasks=taskset)
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
            todd.logger.error("rank %d failed %d: %s", rank, i, e)
            continue

        todd.logger.info("rank %d finished %d with %s", rank, i, metrics['CR'])
        if metrics['CR'] > args.threshold:
            constellation.dump(str(SATELLITES_ROOT / f'{i}.json'))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
