__all__ = [
    'BaseAlgorithm',
]

from abc import ABC, abstractmethod

import torch

from ..data import Actions, Constellation, Taskset
from ..environments import BaseEnvironment, Timer
from ..task_managers import TaskManager


class BaseAlgorithm(ABC):

    def __init__(self, *args, timer: Timer, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._timer = timer

    @abstractmethod
    def prepare(
        self,
        environment: BaseEnvironment,  # readonly
        task_manager: TaskManager,  # readonly
    ) -> None:
        pass

    @abstractmethod
    def step(
        self,
        tasks: Taskset,
        constellation: Constellation,
        rotation: torch.Tensor,
        constellation_data: torch.Tensor,
    ) -> tuple[Actions, list[int]]:
        pass
