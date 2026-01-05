__all__ = [
    'CompletionRateEvaluator',
]

from ..callbacks.memo import Memo, get_memo
from ..task_managers import TaskManager
from .base import BaseEvaluator
from ..environments import BaseEnvironment, Timer


class CompletionRateEvaluator(BaseEvaluator):

    def on_run_end(self, memo: Memo, **kwargs) -> None:
        metrics = get_memo(memo, 'metrics')
        metrics[
            'CR'
        ] = self.task_manager.num_succeeded_tasks / self.task_manager.num_all_tasks
