from pathlib import Path
from abc import abstractmethod
import pathlib

import torch

from ..callbacks.base import BaseCallback
from ..data.constellations import Constellation
from ..task_managers import TaskManager
from ..environments import BaseEnvironment


class BaseLogger(BaseCallback):

    def __init__(
        self,
        *args,
        work_dir: Path,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._work_dir = work_dir
