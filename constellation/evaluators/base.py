__all__ = [
    'BaseEvaluator',
]

from abc import abstractmethod

from ..callbacks.base import BaseCallback
from ..task_managers import TaskManager
from ..environments import BaseEnvironment, Timer


class BaseEvaluator(BaseCallback):
    pass
