__all__ = [
    'BaseEnvironment',
]

from abc import ABC, abstractmethod

import torch

from ..data import Actions, Constellation, TaskSet
from .timer import Timer


class BaseEnvironment(ABC):

    def __init__(
        self,
        *args,
        start_time: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        timer = Timer(start_time)
        self._timer = timer

        # for deepcopy
        self._start_time = start_time

    @property
    @abstractmethod
    def num_satellites(self) -> int:
        pass

    @property
    def timer(self) -> Timer:
        return self._timer

    @property
    def start_time(self) -> int:
        return self._start_time

    @abstractmethod
    def get_constellation(self) -> Constellation:
        pass

    @abstractmethod
    def take_actions(self, actions: Actions) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def is_visible(self, tasks: TaskSet) -> torch.Tensor:
        pass

    @abstractmethod
    def get_earth_rotation(self) -> torch.Tensor:
        pass
