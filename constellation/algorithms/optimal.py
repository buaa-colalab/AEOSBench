__all__ = [
    'OptimalAlgorithm',
]

import math
import einops
import numpy as np
import torch

from ..data import Action, Actions, Constellation, TaskSet
from ..constants import RADIUS_EARTH
from .base import BaseAlgorithm
from ..task_managers import TaskManager
from ..environments import BaseEnvironment

MAX_VISIBILITY_ANGLE = np.pi / 3  # 60 degrees


class OptimalAlgorithm(BaseAlgorithm):

    def prepare(
        self,
        environment: BaseEnvironment,
        task_manager: TaskManager,
    ) -> None:
        self.environment = environment
        self.task_manager = task_manager
        self.last_task = torch.tensor(
            [0 for _ in range(self.environment.num_satellites)],
            dtype=torch.int,
        )
        self.choose_mask = torch.tensor(
            [False for _ in range(self.environment.num_satellites)],
            dtype=torch.bool,
        )
        self.output_mask = torch.tensor(
            [False for _ in range(self.environment.num_satellites)],
            dtype=torch.bool,
        )

    def get_dispatch(
        self,
        tasks: TaskSet,  # ongoing
        constellation: Constellation,
        earth_rotation: torch.Tensor,
    ) -> torch.Tensor:
        satellites_eci = constellation.eci_locations  # dtype:float

        num_satellites = len(constellation)

        task_eci = torch.tensor(tasks.coordinates_ecef) @ earth_rotation

        r_satellite_task = (
            einops.rearrange(task_eci, 'nt three -> 1 nt three')
            - einops.rearrange(satellites_eci, 'ns three -> ns 1 three')
        )

        distance = torch.full(
            (num_satellites, self.task_manager.num_all_tasks),
            float('inf'),
        )
        distance[:, self.task_manager.ongoing_flags] = \
            r_satellite_task.norm(dim=2)

        assignment = torch.argmin(distance, 1)

        combined_distances = torch.cat([
            distance[torch.arange(num_satellites), self.last_task],
            distance[torch.arange(num_satellites), assignment],
        ])

        combined_r_satellite = einops.repeat(
            satellites_eci.norm(dim=1), "ns -> (two ns)", two=2
        )

        # law of cosines
        combined_cosine = ((
            combined_distances**2 + combined_r_satellite**2 - RADIUS_EARTH**2
        ) / (2 * combined_distances * combined_r_satellite))

        distance_mask = (combined_cosine > math.cos(MAX_VISIBILITY_ANGLE)
                         ) & (combined_distances < RADIUS_EARTH)

        last_mask, mask = distance_mask.chunk(2)

        final_dispatch = torch.where(last_mask, self.last_task, assignment)
        self.last_task = final_dispatch
        return torch.where(last_mask | mask, final_dispatch, -1)

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

        dispatch = self.get_dispatch(
            tasks,
            constellation,
            earth_rotation,
        )

        actions = Actions(
            Action(
                toggle=toggle_imaging,
                target_location=(
                    self.task_manager.all_tasks[idx].coordinate if idx !=
                    -1 else None
                ),
            ) for idx in dispatch
        )

        return actions, dispatch.tolist()
