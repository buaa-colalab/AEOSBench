__all__ = [
    'RewardEvaluator',
]

from ..task_managers import TaskManager
from .base import BaseEvaluator


class RewardEvaluator(BaseEvaluator):

    def evaluate(self, task_manager: TaskManager) -> float:
        return sum(
            finished_task.reward
            for finished_task in task_manager.finished_tasks
        )
