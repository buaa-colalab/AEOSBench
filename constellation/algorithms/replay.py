__all__ = [
    'ReplayAlgorithm',
]

from pathlib import Path
from typing import Any, Never
import einops
import torch
import json
import numpy as np

from ..task_managers import TaskManager

from ..constants import STATISTICS_PATH, TRAJECTORIES_ROOT, TASKSETS_ROOT

from ..data import Action, Actions, Constellation, TaskSet, Task
from .base import BaseAlgorithm
from .base import BaseEnvironment


class ReplayAlgorithm(BaseAlgorithm):

    def __init__(self, *args, split: str, i: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._split = split
        self._i = i

        # tasksets_root = TASKSETS_ROOT / split
        trajectories_root = TRAJECTORIES_ROOT / split

        # taskset_path = tasksets_root / f'{i // 1000:02}' / f'{i:05}.json'
        trajectory_path = trajectories_root / f'{i // 1000:02}' / f'{i:05}.pth'

        # taskset: Tasks[Task] = Tasks.load(str(taskset_path))
        # durations = torch.tensor(taskset.durations)

        trajectory = torch.load(trajectory_path, 'cpu')
        # progress, _ = trajectory['taskset']['progress'].max(0)

        self._trajectory = trajectory

        # failed_task_ids, = torch.where(progress < durations)
        # self._failed_task_ids = set(failed_task_ids.tolist())

        from new_transformers.time_model import TimeModel
        time_model = TimeModel()
        time_model.load_state_dict(
            torch.load(
                'work_dirs/archive/time_model_3_lr1e-1_resume/checkpoints/iter_50000/model.pth',
                'cpu',
            ),
        )
        self._time_model = time_model

        self._statistics = torch.load(STATISTICS_PATH, weights_only=False)

        self._tabu_cache: dict[tuple[int, int], bool] = dict()

    def prepare(
        self,
        environment: BaseEnvironment,
        task_manager: TaskManager,
    ) -> None:
        self._task_manager = task_manager

    def step(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    def _tabu_satellites(
        self,
        constellation: Constellation,
        tasks: TaskSet[Task],
        relative_task_ids: list[int],
    ) -> list[int]:
        tabu: list[int] = []
        mapping: dict[int, int] = dict()

        for constellation_id, relative_task_id in enumerate(relative_task_ids):
            if relative_task_id == -1:
                continue
            if (
                constellation_id, tasks[relative_task_id].id_
            ) in self._tabu_cache:
                if self._tabu_cache[
                    (constellation_id, tasks[relative_task_id].id_)]:
                    tabu.append(constellation_id)
                continue
            mapping[constellation_id] = relative_task_id

        if not mapping:
            return tabu

        constellation_ids = list(mapping.keys())
        relative_task_ids = list(mapping.values())

        constellation = Constellation({
            constellation_id: constellation[constellation_id]
            for constellation_id in constellation_ids
        })
        tasks = TaskSet(
            tasks[relative_task_id] for relative_task_id in relative_task_ids
        )

        _, constellation_static_data = constellation.static_to_tensor()
        _, constellation_dynamic_data = constellation.dynamic_to_tensor()
        constellation_data = torch.cat([
            constellation_static_data,
            constellation_dynamic_data,
        ], -1)

        _, taskset_static_data = tasks.to_tensor()
        taskset_static_data[..., [0, 1]] -= self._timer.time
        progress = self._task_manager.progress
        progress = progress[[task.id_ for task in tasks]]
        taskset_dynamic_data = einops.rearrange(progress, "nt -> nt 1")
        taskset_data = torch.cat([
            taskset_static_data,
            taskset_dynamic_data,
        ], -1)

        constellation_data = (
            (constellation_data - self._statistics.constellation_mean) /
            (self._statistics.constellation_std + 1e-6)
        )
        taskset_data = ((taskset_data - self._statistics.taskset_mean) /
                        (self._statistics.taskset_std + 1e-6))

        pred_time, pred_mask = self._time_model._predict(
            [self._timer.time] * len(mapping),
            constellation_data.float(),
            taskset_data.float(),
        )
        pred_mask = pred_mask.sigmoid() < 0.001

        for constellation_id, task, m in zip(
            constellation_ids, tasks, pred_mask
        ):
            self._tabu_cache[(constellation_id, task.id_)] = m.item()
            if m:
                tabu.append(constellation_id)

        return tabu

    def step(
        self,
        tasks: TaskSet[Task],
        constellation: Constellation,
        rotation: torch.Tensor,
        **kwargs,
    ) -> tuple[Actions, list[int]]:
        if self._timer.time % 10 == 0:
            self._tabu_cache = dict()
        if self._task_manager.progress.any(
        ) and self._timer.time % 50 == 0 and self._i < 10:
            print(
                self._split, self._i, self._timer.time,
                self._task_manager.progress.sum()
            )

        task_id_to_task = {task.id_: task for task in tasks}

        task_ids = (
            self._trajectory['actions']['task_id'][self._timer.time].tolist()
        )

        relative_task_ids: list[int] = []
        for task_id in task_ids:
            if task_id in task_id_to_task:
                task = task_id_to_task[task_id]
                relative_task_id = tasks.index(task)
                relative_task_ids.append(relative_task_id)
            else:
                relative_task_ids.append(-1)

        target_locations = [
            task_id_to_task[task_id].coordinate
            if task_id in task_id_to_task else None for task_id in task_ids
        ]

        mask_indices = self._tabu_satellites(
            constellation,
            tasks,
            relative_task_ids,
        )
        for mask_index in mask_indices:
            task_ids[mask_index] = -1

        # task_ids = [
        #     -1 if task_id in self._failed_task_ids else task_id
        #     for task_id in task_ids
        # ]

        toggles = [
            (task_id == -1 and satellite.sensor.enabled)
            or (task_id != -1 and not satellite.sensor.enabled)
            for task_id, satellite in zip(task_ids, constellation.sort())
        ]

        actions = Actions(
            Action(toggle, target_location)
            for toggle, target_location in zip(toggles, target_locations)
        )

        return actions, task_ids


# class ReplayAlgorithm(BaseAlgorithm):

#     def __init__(self, *args, split: str, i: int, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self._split = split
#         self._i = i

#         tasksets_root = TASKSETS_ROOT / split
#         trajectories_root = TRAJECTORIES_ROOT / split

#         taskset_path = tasksets_root / f'{i // 1000:02}' / f'{i:05}.json'
#         trajectory_path = trajectories_root / f'{i // 1000:02}' / f'{i:05}.pth'

#         taskset: Tasks[Task] = Tasks.load(str(taskset_path))
#         durations = torch.tensor(taskset.durations)

#         trajectory = torch.load(trajectory_path, 'cpu')
#         progress, _ = trajectory['taskset']['progress'].max(0)

#         self._trajectory = trajectory

#         failed_task_ids, = torch.where(progress < durations)
#         self._failed_task_ids = set(failed_task_ids.tolist())

#     def prepare(
#         self,
#         environment: BaseEnvironment,
#         task_manager: TaskManager,
#     ) -> None:
#         self._task_manager = task_manager

#     def step(self, *args, **kwargs) -> Never:
#         raise NotImplementedError

#     def step_new(
#         self,
#         tasks: Tasks[Task],
#         constellation: Constellation,
#         time_data: Any,
#     ) -> tuple[Actions, list[int]]:
#         if self._task_manager.progress.any() and self._timer.time % 50 == 0:
#             print(
#                 self._split, self._i, self._timer.time,
#                 self._task_manager.progress.sum()
#             )

#         task_id_to_task = {task.id_: task for task in tasks}

#         task_ids = (
#             self._trajectory['actions']['task_id'][self._timer.time].tolist()
#         )

#         target_locations = [
#             task_id_to_task[task_id].coordinate
#             if task_id in task_id_to_task else None for task_id in task_ids
#         ]

#         task_ids = [
#             -1 if task_id in self._failed_task_ids else task_id
#             for task_id in task_ids
#         ]

#         toggles = [
#             (task_id == -1 and satellite.sensor.enabled)
#             or (task_id != -1 and not satellite.sensor.enabled)
#             for task_id, satellite in zip(task_ids, constellation.sort())
#         ]

#         actions = Actions(
#             Action(toggle, target_location)
#             for toggle, target_location in zip(toggles, target_locations)
#         )

#         return actions, task_ids
