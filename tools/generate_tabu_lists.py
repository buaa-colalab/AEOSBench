import argparse
import multiprocessing
import pathlib
from functools import partial

import todd
import torch
from constellation.new_transformers.dataset import TrajectoryData
from todd.patches.py_ import json_dump, json_load

from constellation import ANNOTATIONS_ROOT, TASKSETS_ROOT, TRAJECTORIES_ROOT
from constellation.data import Task, TaskSet


def get_tabu_list(failed: set[int], actions: list[int]) -> list[int]:
    tabu: list[int] = []

    if len(actions) == 0:
        return tabu

    if actions[0] in failed:
        tabu.append(actions[0])

    i = 0
    j = 1
    while True:
        while j < len(actions) and actions[j] == actions[i]:
            j += 1

        if j >= len(actions):
            break

        if actions[i] not in failed and actions[j] in failed:
            tabu.append(actions[j])
        i = j
        j = i + 1

    return sorted(set(tabu))


def generate_tabu_list(
    split: str,
    i: int,
    merge: pathlib.Path | None = None,
) -> None:
    if i % 100 == 0:
        todd.logger.info("%s %s", split, i)

    tasksets_root = TASKSETS_ROOT / split
    trajectories_root = TRAJECTORIES_ROOT / split

    taskset_path = tasksets_root / f'{i // 1000:02}' / f'{i:05}.json'
    trajectory_path = trajectories_root / f'{i // 1000:02}' / f'{i:05}.pth'
    tabu_path = trajectories_root / f'{i // 1000:02}' / f'{i:05}.tabu.json'

    taskset: TaskSet[Task] = TaskSet.load(str(taskset_path))
    durations = taskset.durations

    trajectory: TrajectoryData = torch.load(str(trajectory_path))
    progress, _ = trajectory['taskset']['progress'].max(0)

    failed_task_ids, = torch.where(progress < durations)
    failed_task_ids_ = set(failed_task_ids.tolist())

    actions_task_ids = trajectory['actions']['task_id']
    tabu = list(
        map(
            partial(get_tabu_list, failed_task_ids_),
            actions_task_ids.T.tolist(),
        ),
    )

    if merge is not None:
        previous_tabu_path = (
            merge / split / f'{i // 1000:02}' / f'{i:05}.tabu.json'
        )
        previous_tabu = json_load(str(previous_tabu_path))
        assert len(tabu) == len(previous_tabu)
        for tabu_, previous_tabu_ in zip(tabu, previous_tabu):
            tabu_.extend(previous_tabu_)

    json_dump(tabu, str(tabu_path))


def generate(num_workers: int, split: str, **kwargs) -> None:
    annotations: list[int] = json_load(
        str(ANNOTATIONS_ROOT / f'{split}.json'),
    )

    if num_workers == 0:
        for i in annotations:
            generate_tabu_list(split, i, **kwargs)
        return

    with multiprocessing.Pool(num_workers) as pool:
        list(
            pool.imap_unordered(
                partial(generate_tabu_list, split, **kwargs),
                annotations,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate trajectories')
    parser.add_argument('num_workers', type=int)
    parser.add_argument('--merge', type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.num_workers, 'train', merge=args.merge)
    generate(args.num_workers, 'val_seen', merge=args.merge)
    generate(args.num_workers, 'val_unseen', merge=args.merge)
    generate(args.num_workers, 'test', merge=args.merge)


if __name__ == '__main__':
    main()
