__all__ = [
    'BaseCallback',
]

from abc import ABC, abstractmethod
from pathlib import Path
import pathlib
import weakref

from typing import TYPE_CHECKING

from .memo import Memo

from .controller_holder import ControllerHolder

from ..task_managers import TaskManager
from ..environments import BaseEnvironment
from ..data.constellations import Constellation

if TYPE_CHECKING:
    from ..controller import Controller


class BaseCallback(ABC, ControllerHolder):
    # TODO: use key word

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def on_init(self, *args, controller: 'Controller', **kwargs) -> None:
        self.set_controller(controller)

    @property
    def task_manager(self) -> TaskManager:
        return self.controller.task_manager

    @property
    def environment(self) -> BaseEnvironment:
        return self.controller.environment

    def on_step_begin(self) -> None:
        pass

    def on_step_end(self, dispatch_id: list[int]) -> None:
        pass

    def on_run_end(self, memo: Memo, save_name: str) -> None:
        pass
