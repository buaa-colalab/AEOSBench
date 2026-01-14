__all__ = [
    'OptimalAlgorithm',
]

import math

import einops
import torch

from ..constants import MAX_OFF_NADIR_ANGLE, RADIUS_EARTH
from ..data import Action, Actions, Constellation, TaskSet
from ..environments import BaseEnvironment
from ..task_managers import TaskManager
from .base import BaseAlgorithm


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
        default_task_indices = default_assignment = torch.full(
            (len(constellation), ),
            -1,
            dtype=torch.int,
        )
        if len(taskset) == 0:
            return default_task_indices, default_assignment

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

        if not greedy_valid_mask.any():
            return default_task_indices, default_assignment

        previous_assignment = self.previous_assignment[greedy_valid_mask]
        task_ids = previous_assignment.new_tensor([
            task.id_ for task in taskset
        ])

        task_indices = torch.where(greedy_valid_mask, greedy_task_indices, -1)
        assignment = torch.where(greedy_valid_mask, task_ids[greedy_task_indices], -1)  # noqa: E501 yapf: disable

        restorable_mask, restorable_task_indices = torch.max(
            einops.rearrange(previous_assignment, 'ns -> ns 1') ==
            einops.rearrange(task_ids, 'nt -> 1 nt'),
            dim=1,
        )
        if not restorable_mask.any():
            return task_indices, assignment

        restorable_task_indices = restorable_task_indices[restorable_mask]
        restorable_distance = satellite_task_distance[greedy_valid_mask]\
            [restorable_mask, restorable_task_indices]
        restorable_valid_mask = self._check_constraints(
            restorable_distance,
            orbital_radius[greedy_valid_mask][restorable_mask],
        )
        if not restorable_valid_mask.any():
            return task_indices, assignment

        # NOTE: do not assign through multiple indexing, as it triggers clones
        restorable_satellite_indices = torch.arange(
            len(constellation),
            dtype=torch.int,
        )[greedy_valid_mask][restorable_mask][restorable_valid_mask]
        task_indices[restorable_satellite_indices] = restorable_task_indices
        assignment[restorable_satellite_indices] = (
            task_ids[restorable_task_indices]
        )

        return task_indices, assignment

    def step(
        self,
        taskset: TaskSet,  # ongoing
        constellation: Constellation,
        earth_rotation: torch.Tensor,
    ) -> tuple[Actions, list[int]]:
        task_indices, assignment = self._assign(
            taskset,
            constellation,
            earth_rotation,
        )
        self.previous_assignment = assignment

        actions = Actions(
            Action(
                toggle=not satellite.sensor.enabled,
                target_location=(
                    None if task_index ==
                    -1 else taskset[task_index].coordinate
                ),
            ) for satellite, task_index in
            zip(constellation.sort(), task_indices)
        )

        return actions, assignment.tolist()
