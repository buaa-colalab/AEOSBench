__all__ = [
    'TurnAroundTimeEvaluator',
]

from ..callbacks.memo import Memo, get_memo
from ..task_managers import TaskManager
from .base import BaseEvaluator
from ..environments import BaseEnvironment, Timer
from constellation import task_managers


class TurnAroundTimeEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
        self.completion_time = []

    def on_step_end(self, **kwargs) -> None:
        if not self.completion_time:
            self.completion_time = [float('inf') for _ in self.task_manager.all_tasks]
        for i, task in enumerate(self.task_manager.all_tasks):
            if task in self.task_manager.succeeded_tasks:
                self.completion_time[i] = min(
                    self.completion_time[i], self.environment.timer.time
                )

    def on_run_end(self, memo: Memo, **kwargs) -> None:
        sum_time = 0
        for i, task in enumerate(self.task_manager.all_tasks):
            if (task in self.task_manager.succeeded_tasks):
                sum_time += self.completion_time[i] - task.release_time
        metrics = get_memo(memo, 'metrics')
        metrics['TT'] = sum_time
