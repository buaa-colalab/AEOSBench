__all__ = [
    'EarlyStopCallback',
]

import todd

from .base import BaseCallback


class EarlyStopCallback(BaseCallback):

    def should_break(self) -> bool:
        if self.controller.task_manager.all_closed:
            todd.logger.info("All tasks are closed.")
            return True
        return False
