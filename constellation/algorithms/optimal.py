__all__ = [
    'OptimalAlgorithm',
]

import math
import einops
import torch

from ..data import Action, Actions, Constellation, TaskSet
from ..constants import RADIUS_EARTH, MAX_OFF_NADIR_ANGLE
from .base import BaseAlgorithm
from ..task_managers import TaskManager
from ..environments import BaseEnvironment


class OptimalAlgorithm(BaseAlgorithm):

    def prepare(
        self,
        environment: BaseEnvironment,
        task_manager: TaskManager,
    ) -> None:
        self.previous_assignment = torch.full(
            (environment.num_satellites, ),
            -1,
            dtype=torch.int,
        )

    @property
    def previous_assignment(self) -> torch.Tensor:
        return self.get_buffer('_previous_assignment')

    @previous_assignment.setter
    def previous_assignment(self, value: torch.Tensor) -> None:
        self.register_buffer('_previous_assignment', value)

    def _check_constraints(
        self,
        distance: torch.Tensor,
        orbital_radius: torch.Tensor,
    ) -> torch.Tensor:
        mask_distance = distance < RADIUS_EARTH
        cosine = ((distance**2 + orbital_radius**2 - RADIUS_EARTH**2) /
                  (2 * distance * orbital_radius))
        mask_cosine = cosine > math.cos(MAX_OFF_NADIR_ANGLE)
        return mask_distance & mask_cosine

    def get_dispatch(
        self,
        taskset: TaskSet,  # ongoing
        constellation: Constellation,
        earth_rotation: torch.Tensor,
    ) -> torch.Tensor:
        taskset_eci = (
            earth_rotation.new_tensor(taskset.coordinates_ecef)
            @ earth_rotation
        )
        constellation_eci = constellation.coordinates_eci  # dtype:float

        satellite_task_distance = torch.norm(
            einops.rearrange(taskset_eci, 'nt three -> 1 nt three')
            - einops.rearrange(constellation_eci, 'ns three -> ns 1 three'),
            dim=2,
        )
        orbital_radius = constellation_eci.norm(dim=1)

        greedy_distance, greedy_task_indices = (
            satellite_task_distance.min(dim=1)
        )
        greedy_valid_mask = self._check_constraints(
            greedy_distance,
            orbital_radius,
        )

        previous_assignment = self.previous_assignment[greedy_valid_mask]
        task_ids = previous_assignment.new_tensor([
            task.id_ for task in taskset
        ])

        restorable_mask, restorable_task_indices = torch.max(
            einops.rearrange(previous_assignment, 'ns -> ns 1') ==
            einops.rearrange(task_ids, 'nt -> 1 nt'),
            dim=1,
        )
        restorable_task_indices = restorable_task_indices[restorable_mask]
        restorable_distance = satellite_task_distance[greedy_valid_mask]\
            [restorable_mask, restorable_task_indices]
        restorable_valid_mask = self._check_constraints(
            restorable_distance,
            orbital_radius[greedy_valid_mask][restorable_mask],
        )

        task_indices = greedy_task_indices.clone()
        task_indices[greedy_valid_mask][restorable_mask][restorable_valid_mask] = restorable_task_indices  # noqa: E501 yapf: disable

        assignment = torch.where(greedy_valid_mask, task_ids[task_indices], -1)
        self.previous_assignment = assignment

        task_indices = torch.where(greedy_valid_mask, task_indices, -1)

        return assignment, task_indices

    def step(
        self,
        tasks: TaskSet,  # ongoing
        constellation: Constellation,
        earth_rotation: torch.Tensor,
    ) -> tuple[Actions, list[int]]:

        num_satellites = len(constellation)
        num_tasks = len(tasks)

        toggle_imaging = self._timer.time == 0

        if (num_tasks == 0):
            return Actions(
                Action(toggle=toggle_imaging, target_location=None)
                for _ in range(num_satellites)
            ), [-1 for _ in range(num_satellites)]

        dispatch, task_indices = self.get_dispatch(
            tasks,
            constellation,
            earth_rotation,
        )

        actions = Actions(
            Action(
                toggle=toggle_imaging,
                target_location=(
                    None if task_index == -1 else tasks[task_index].coordinate
                ),
            ) for task_index in task_indices
        )

        return actions, dispatch.tolist()
