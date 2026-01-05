__all__ = [
    'PCompletionRateEvaluator',
]

from ..callbacks.memo import Memo, get_memo
from ..task_managers import TaskManager
from .base import BaseEvaluator
import torch
from ..environments import BaseEnvironment, Timer
from constellation import task_managers


class PCompletionRateEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_progress_ = []

    def on_step_end(self, **kwargs) -> None:
        if self.max_progress_ == []:
            self.max_progress_ = self.task_manager.progress
        else:
            self.max_progress_ = torch.max(
                self.max_progress_, self.task_manager.progress
            )

    def on_run_end(self, memo: Memo, **kwargs) -> None:
        num_tasks = self.task_manager.num_all_tasks
        all_rate = 0.0
        for i in range(self.task_manager.num_all_tasks):
            all_rate += self.max_progress_[i] / float(
                self.task_manager.all_tasks[i].duration
            )
        metrics = get_memo(memo, 'metrics')
        metrics['PCR'] = (all_rate / num_tasks).item()
