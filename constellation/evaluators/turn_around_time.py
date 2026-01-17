__all__ = [
    'TurnAroundTimeEvaluator',
]

import torch
from .base import BaseEvaluator


class TurnAroundTimeEvaluator(BaseEvaluator):

    @property
    def succeeded_flags(self) -> torch.Tensor:
        return self.controller.task_manager.succeeded_flags

    @property
    def completion_time(self) -> torch.Tensor:
        return self.controller.memo['completion_time']

    @completion_time.setter
    def completion_time(self, value: torch.Tensor) -> None:
        self.controller.memo['completion_time'] = value

    def before_run(self) -> None:
        self.completion_time = torch.full(
            (self.controller.task_manager.num_all_tasks, ),
            float('inf'),
        )

    def after_step(self) -> None:
        self.completion_time[self.succeeded_flags] = torch.clamp_max(
            self.completion_time[self.succeeded_flags],
            self.controller.environment.timer.time,
        )

    def after_run(self) -> None:
        release_times = self.controller.task_manager.taskset.release_times
        turn_around_time = self.completion_time - release_times
        self.metrics['TAT'] = (
            turn_around_time[self.succeeded_flags].mean().item()
        )
