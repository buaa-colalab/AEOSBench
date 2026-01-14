__all__ = [
    'TrajectoryLogger',
]

from typing import NamedTuple

import torch

from .base import BaseLogger

# TODO: rename


class TrajectoryPoint(NamedTuple):
    sensor_enabled: torch.Tensor
    data: torch.Tensor
    progress: torch.Tensor
    task_id: torch.Tensor
    is_visible: torch.Tensor


class TrajectoryLogger(BaseLogger):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._trajectory: list[TrajectoryPoint] = []

    def after_step(self) -> None:
        sensor_enabled, data = (
            self.controller.environment.get_constellation().dynamic_to_tensor()
        )
        trajectory_point = TrajectoryPoint(
            sensor_enabled,
            data,
            self.controller.task_manager.progress,
            torch.tensor(self.controller.memo['assignment']),
            self.controller.memo['is_visible'],
        )
        self._trajectory.append(trajectory_point)

    def after_run(self) -> None:
        sensor_enabled, data, progress, task_id, is_visible = (
            map(torch.stack, zip(*self._trajectory))
        )
        torch.save(
            dict(
                constellation=dict(sensor_enabled=sensor_enabled, data=data),
                taskset=dict(progress=progress),
                actions=dict(task_id=task_id),
                is_visible=is_visible.bool(),  # TODO: why explicitly convert?
            ),
            self._work_dir / f'{self.controller.name}.pth',
        )
