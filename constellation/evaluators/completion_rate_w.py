__all__ = [
    'WCompletionRateEvaluator',
]

from ..callbacks.memo import Memo, get_memo
from ..task_managers import TaskManager
from .base import BaseEvaluator
from ..environments import BaseEnvironment, Timer


class WCompletionRateEvaluator(BaseEvaluator):

    def on_run_end(self, memo: Memo, **kwargs) -> None:
        all_time = 0
        finished_time = 0
        for task in self.task_manager.all_tasks:
            all_time += task.duration
            if task in self.task_manager.succeeded_tasks:
                finished_time += task.duration
        metrics = get_memo(memo, 'metrics')
        metrics['WCR'] = finished_time / all_time
