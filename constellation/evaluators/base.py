__all__ = [
    'BaseEvaluator',
]

from todd.runners import Memo, get_memo

from ..callbacks import BaseCallback


class BaseEvaluator(BaseCallback):

    @property
    def metrics(self) -> Memo:
        return get_memo(self.controller.memo, 'metrics')
