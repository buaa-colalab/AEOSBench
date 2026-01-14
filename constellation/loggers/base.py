__all__ = [
    'BaseLogger',
]

import pathlib

from ..callbacks.base import BaseCallback


class BaseLogger(BaseCallback):

    def __init__(self, *args, work_dir: pathlib.Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._work_dir = work_dir
