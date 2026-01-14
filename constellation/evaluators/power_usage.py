__all__ = [
    'PowerUsageEvaluator',
]

import torch
from .base import BaseEvaluator


class PowerUsageEvaluator(BaseEvaluator):

    @property
    def working_time_steps(self) -> torch.Tensor:
        return self.controller.memo['working_time_steps']

    @working_time_steps.setter
    def working_time_steps(self, value: torch.Tensor) -> None:
        self.controller.memo['working_time_steps'] = value

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        self.working_time_steps = torch.zeros(
            self.controller.environment.num_satellites,
            dtype=torch.int,
        )

    def after_step(self) -> None:
        # TODO: fix after making assignment a tensor
        self.working_time_steps += torch.tensor(
            self.controller.memo['assignment']
        ) != -1

    def after_run(self) -> None:
        sensor_power = torch.tensor([
            satellite.sensor.power for satellite in
            self.controller.environment.get_constellation().values()
        ])
        self.metrics['PC'] = (
            torch.sum(self.working_time_steps * sensor_power).item()
        )
