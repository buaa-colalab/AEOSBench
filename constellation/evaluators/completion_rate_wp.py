__all__ = [
    'WPCompletionRateEvaluator',
]

from networkx import is_empty

from ..callbacks.memo import Memo, get_memo
from ..task_managers import TaskManager
from .base import BaseEvaluator
import torch
from ..environments import BaseEnvironment, Timer


class WPCompletionRateEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_progress_ = torch.tensor([])

    def on_step_end(self, **kwargs) -> None:
        if len(self.max_progress_) == 0:
            self.max_progress_ = self.task_manager.progress
        else:
            self.max_progress_ = torch.max(
                self.max_progress_, self.task_manager.progress
            )

    def on_run_end(self, memo: Memo, **kwargs) -> None:
        num_tasks = 0.0
        for task in self.task_manager.all_tasks:
            num_tasks += 1.0 * task.duration
        all_rate = torch.zeros([])
        for i in range(self.task_manager.num_all_tasks):
            all_rate += (
                self.max_progress_[i]
                / float(self.task_manager.all_tasks[i].duration)
            ) * self.task_manager.all_tasks[i].duration
        metrics = get_memo(memo, 'metrics')
        metrics['WPCR'] = (all_rate / num_tasks).item()
