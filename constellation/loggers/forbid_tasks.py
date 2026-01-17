__all__ = [
    'ForbidTasksCallback',
]

import einops
import torch

from .base import BaseLogger


class ForbidTasksCallback(BaseLogger):

    def before_run(self) -> None:
        num_satellites = self.controller.environment.num_satellites
        self._last_assignment = torch.full((num_satellites, ), -1)
        self._forbidden_task_ids = torch.full((num_satellites, ), -1)
        # self._last_progress = torch.zeros(self.controller.task_manager.num_all_tasks)

    def after_step(self) -> None:
        constellation_mask = self._forbidden_task_ids == -1
        if not constellation_mask.any():
            return

        # TODO: remove explicit tensor conversion
        assignment = torch.tensor(self.controller.memo['assignment'])
        constellation_mask &= ((self._last_assignment != assignment) &
                               (self._last_assignment != -1))

        task_ids = self._last_assignment[constellation_mask]
        self._last_assignment = assignment

        if not constellation_mask.any():
            return

        task_manager = self.controller.task_manager
        task_indices = task_manager.taskset.get_task_indices(task_ids)

        forbidden_mask = ~task_manager.succeeded_flags[task_indices]
        forbidden_mask &= task_manager.progress[task_indices] == 0
        # forbidden_mask &= self._last_progress[task_indices] != 0
        # self._last_progress = task_manager.progress
        if not forbidden_mask.any():
            return

        forbidden_task_ids = task_ids[forbidden_mask]

        satellite_indices = torch.arange(
            self.controller.environment.num_satellites,
        )[constellation_mask][forbidden_mask]
        self._forbidden_task_ids[satellite_indices] = forbidden_task_ids

    def after_run(self) -> None:
        torch.save(
            self._forbidden_task_ids,
            self._work_dir / f'{self.controller.name}_forbidden_task_ids.pth',
        )
