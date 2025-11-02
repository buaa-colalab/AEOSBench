__all__ = [
    'PowerEvaluator',
]

from ..callbacks.memo import Memo, get_memo
from ..task_managers import TaskManager
from ..environments import BaseEnvironment
from .base import BaseEvaluator


class PowerEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs) -> None:
        self.power_used = 0.0

    def on_run_end(self, memo: Memo, **kwargs) -> None:
        metrics = get_memo(memo, 'metrics')
        metrics['PC'] = self.power_used

    def on_step_end(self, dispatch_id: list, **kwargs) -> None:
        for sat_id, sat in self.environment.get_constellation().items():
            if dispatch_id[sat_id] != -1:
                self.power_used += sat.sensor.power
