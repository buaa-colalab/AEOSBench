import pathlib
from typing import Any

import torch

from ..data.constellations import Constellation
from .base_logger import BaseLogger


class PthLogger(BaseLogger):
    """Logger for recording data in PyTorch format."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._all_pth: list[Any] = []

    def on_step_end(self, dispatch_id: list[int], **kwargs) -> None:
        self._all_pth.append((
            *self.controller.environment.get_constellation().dynamic_to_tensor(
            ),
            self.controller.task_manager.progress,
            torch.tensor(dispatch_id),
            self.controller.environment.is_visible(
                self.controller.task_manager.all_tasks
            ),
        ))

    def on_run_end(self, save_name: str, **kwargs) -> None:
        if not self._all_pth:
            return

        (
            sensor_enabled,
            data,
            progress,
            task_id,
            is_visible,
        ) = map(torch.stack, zip(*self._all_pth))
        torch.save(
            dict(
                constellation=dict(sensor_enabled=sensor_enabled, data=data),
                taskset=dict(progress=progress),
                actions=dict(task_id=task_id),
                is_visible=is_visible.bool(),
            ),
            self._work_dir / pathlib.Path(f"{int(save_name):05}" + ".pth"),
        )
