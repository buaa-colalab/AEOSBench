__all__ = [
    'CompletionRateEvaluator',
]

import torch
from .base import BaseEvaluator


class CompletionRateEvaluator(BaseEvaluator):

    @property
    def max_progress(self) -> torch.Tensor:
        return self.controller.memo['max_progress']

    @max_progress.setter
    def max_progress(self, value: torch.Tensor) -> None:
        self.controller.memo['max_progress'] = value

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        self.max_progress = self.controller.task_manager.progress

    def after_step(self) -> None:
        self.max_progress = torch.max(
            self.max_progress,
            self.controller.task_manager.progress,
        )

    def after_run(self) -> None:
        durations = self.controller.task_manager.all_tasks.durations
        completion_rate = (
            self.controller.task_manager.num_succeeded_tasks
            / self.controller.task_manager.num_all_tasks
        )
        weighted_completion_rate = (
            durations[self.controller.task_manager.succeeded_flags].sum()
            / durations.sum()
        )
        partial_completion_rate = self.max_progress / durations
        weighted_partial_completion_rate = (
            self.max_progress.sum() / durations.sum()
        )

        self.metrics.update({
            'CR': completion_rate,
            'WCR': weighted_completion_rate.item(),
            'PCR': partial_completion_rate.mean().item(),
            'WPCR': weighted_partial_completion_rate.item(),
        })
