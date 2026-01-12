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

    def _assign(
        self,
        taskset: TaskSet,  # ongoing
        constellation: Constellation,
        earth_rotation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # if self._timer.time == 2350:
        #     breakpoint()
        taskset_eci = (
            earth_rotation.new_tensor(taskset.coordinates_ecef)
            @ earth_rotation
        )
        constellation_eci = constellation.coordinates_eci

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

        # NOTE: do not assign through multiple indexing, as it triggers clones
        restorable_satellite_indices = torch.arange(len(constellation), dtype=torch.int)[greedy_valid_mask][restorable_mask][restorable_valid_mask]  # noqa: E501 yapf: disable
        task_indices[restorable_satellite_indices] = restorable_task_indices

        assignment = torch.where(greedy_valid_mask, task_ids[task_indices], -1)
        self.previous_assignment = assignment

        task_indices = torch.where(greedy_valid_mask, task_indices, -1)

        return assignment, task_indices

    def step(
        self,
        taskset: TaskSet,  # ongoing
        constellation: Constellation,
        earth_rotation: torch.Tensor,
    ) -> tuple[Actions, list[int]]:
        if len(taskset) == 0:
            actions = Actions(
                Action(
                    # toggle=not satellite.sensor.enabled,
                    toggle=self._timer.time == 0,
                    target_location=None,
                ) for satellite in constellation.sort()
            )
            assignment = [-1] * len(constellation)
            return actions, assignment

        assignment, task_indices = self._assign(
            taskset,
            constellation,
            earth_rotation,
        )

        actions = Actions(
            Action(
                # toggle=not satellite.sensor.enabled,
                toggle=self._timer.time == 0,
                target_location=(
                    None if task_index ==
                    -1 else taskset[task_index].coordinate
                ),
            ) for satellite, task_index in
            zip(constellation.sort(), task_indices)
        )

        return actions, assignment.tolist()
