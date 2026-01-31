__all__ = [
    'DynamicConstellationData',
    'DynamicTasksetData',
    'Actions',
    'TrajectoryData',
    'Batch',
    'Statistics',
    'Dataset',
]

import random
from typing import NamedTuple, TypedDict

import einops
import torch
from todd.patches.py_ import json_load
from todd.utils import NestedTensorCollectionUtils

from constellation import (
    ANNOTATIONS_ROOT,
    CONSTELLATIONS_ROOT,
    STATISTICS_PATH,
    TASKSETS_ROOT,
    TRAJECTORIES_ROOT,
    DATA_ROOT,
)
from constellation.data import Constellation, TaskSet

from .registries import ConstellationDatasetRegistry


class DynamicConstellationData(TypedDict):
    # shape: t x num_satellites
    # dtype: bool
    sensor_enabled: torch.Tensor

    # shape: t x num_satellites x satellite_dim (8)
    # dtype: float
    #
    # satellite_dim:
    #   - battery_percentage
    #   - reaction_wheels[0].speed
    #   - reaction_wheels[1].speed
    #   - reaction_wheels[2].speed
    #   - true_anomaly
    #   - attitude[0]
    #   - attitude[1]
    #   - attitude[2]
    data: torch.Tensor


class DynamicTasksetData(TypedDict):
    # shape: t x num_tasks
    # dtype: uint8
    progress: torch.Tensor


class Actions(TypedDict):
    # shape: t x num_satellites
    # dtype: int
    task_id: torch.Tensor


class TrajectoryData(TypedDict):
    constellation: DynamicConstellationData
    taskset: DynamicTasksetData
    actions: Actions
    # shape: t x num_satellites x num_tasks
    # dtype: bool
    is_visible: torch.Tensor


class Batch(NamedTuple):
    id_: int
    annotation_id: int
    time_steps: list[int]
    constellation_sensor_type: torch.Tensor
    constellation_sensor_enabled: torch.Tensor
    constellation_data: torch.Tensor
    constellation_mask: torch.Tensor
    tasks_sensor_type: torch.Tensor
    tasks_data: torch.Tensor
    tasks_mask: torch.Tensor
    actions_task_id: torch.Tensor  # TODO: rename


class Statistics(NamedTuple):
    constellation_mean: torch.Tensor
    constellation_std: torch.Tensor
    taskset_mean: torch.Tensor
    taskset_std: torch.Tensor


@ConstellationDatasetRegistry.register_()
class Dataset(torch.utils.data.Dataset[Batch]):

    def __init__(
        self,
        *args,
        split: str,
        annotation_file: str | None = None,
        batch_size: int,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._split = split

        if annotation_file is None:
            annotation_file = f'{split}.json'
        self._annotations: dict[str, list[int]] = json_load(
            str(ANNOTATIONS_ROOT / annotation_file),
        )

        self._batch_size = batch_size

        if normalize:
            self._statistics: Statistics = torch.load(
                STATISTICS_PATH,
                weights_only=False,
            )

        self._nested_tensor_collection_utils = NestedTensorCollectionUtils()

    @property
    def normalize(self) -> bool:
        return hasattr(self, '_statistics')

    def __len__(self) -> int:
        return len(self._annotations['ids'])

    def _load_constellation(
        self,
        constellation: DynamicConstellationData,
        id_: int,
        indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sensor_enabled = constellation['sensor_enabled'][indices]
        dynamic_data = constellation['data'][indices]

        constellation_path = (
            CONSTELLATIONS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        sensor_type, static_data = Constellation.load(
            str(constellation_path),
        ).static_to_tensor()

        sensor_type = einops.repeat(
            sensor_type,
            'ns -> t ns',
            t=len(indices),
        )
        static_data = einops.repeat(
            static_data,
            'ns nd -> t ns nd',
            t=len(indices),
        )
        data = torch.cat([static_data, dynamic_data], -1)

        mask = torch.ones_like(sensor_type, dtype=torch.bool)

        return sensor_type, sensor_enabled, data, mask

    def _load_tasks(
        self,
        taskset: DynamicTasksetData,
        id_: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        progress = taskset['progress']
        t = progress.shape[0]

        taskset_path = (
            TASKSETS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        sensor_type, static_data = TaskSet.load(str(taskset_path)).to_tensor()
        duration = static_data[..., 2]

        sensor_type = einops.repeat(sensor_type, 'nt -> t nt', t=t)
        static_data = einops.repeat(static_data, 'nt nd -> t nt nd', t=t)

        static_data = static_data.clone()  # for in-place modification
        time_steps = einops.rearrange(torch.arange(t), 't -> t 1')
        static_data[..., 0] -= time_steps
        static_data[..., 1] -= time_steps

        dynamic_data = einops.rearrange(progress, 't nt -> t nt 1')

        data = torch.cat([static_data, dynamic_data], -1)

        release_time_mask = static_data[..., 0] <= 0
        due_time_mask = static_data[..., 1] >= 0
        finished_mask = progress >= duration
        finished_mask, _ = finished_mask.cummax(0)
        mask = release_time_mask & due_time_mask
        mask[1:] &= ~finished_mask[:-1] # FIXME

        return sensor_type, data, mask

    def _load_actions(
        self,
        actions: Actions,
        indices: list[int],
    ) -> torch.Tensor:
        return actions['task_id'][indices]

    def __getitem__(self, index: int) -> Batch:
        id_ = self._annotations['ids'][index]
        best_epoch_ = self._annotations['epochs'][index]

        trajectory: TrajectoryData = torch.load(
            DATA_ROOT
            / f'trajectories.{best_epoch_}'
            / self._split
            / f'{id_ // 1000:02}'
            / f'{id_:05}.pth',
        )

        tasks_sensor_type, tasks_data, tasks_mask = self._load_tasks(
            trajectory['taskset'],
            id_,
        )

        # a time step is valid iff any task is valid
        indices = tasks_mask.any(-1).nonzero().flatten().tolist()
        if len(indices) > self._batch_size:
            indices = random.sample(indices, self._batch_size)

        tasks_sensor_type = tasks_sensor_type[indices]
        tasks_data = tasks_data[indices]
        tasks_mask = tasks_mask[indices]

        # TODO: rename, `actions_task_id` is ambiguous
        actions_task_id = self._load_actions(trajectory['actions'], indices)

        # remove the tasks that are never valid
        task_is_valid = tasks_mask.any(0)
        if not task_is_valid.all():
            tasks_sensor_type = tasks_sensor_type[:, task_is_valid]
            tasks_data = tasks_data[:, task_is_valid]
            tasks_mask = tasks_mask[:, task_is_valid]
            tasks_id_mapper = task_is_valid.cumsum(0) - 1
            actions_task_id = torch.where(
                actions_task_id == -1,
                actions_task_id,
                tasks_id_mapper[actions_task_id],
            )

        # ensure that `actions_task_id` is valid
        augmented_tasks_mask = torch.cat([
            tasks_mask.new_ones(len(indices), 1),
            tasks_mask,
        ], -1)
        if not augmented_tasks_mask.gather(-1, actions_task_id + 1).all():
            raise RuntimeError(f"Trajectory.{best_epoch_} {index} ({id_}) is invalid")

        (
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
        ) = self._load_constellation(
            trajectory['constellation'],
            id_,
            indices,
        )

        if self.normalize:
            constellation_data = (
                (constellation_data - self._statistics.constellation_mean) /
                (self._statistics.constellation_std + 1e-6)
            )
            tasks_data = ((tasks_data - self._statistics.taskset_mean) /
                          (self._statistics.taskset_std + 1e-6))

        # NOTE: sensor type should be 0-indexed
        batch = Batch(
            index,
            id_,
            indices,
            constellation_sensor_type - 1,
            constellation_sensor_enabled,
            constellation_data,
            constellation_mask,
            tasks_sensor_type - 1,
            tasks_data,
            tasks_mask,
            actions_task_id,
        )

        return batch
