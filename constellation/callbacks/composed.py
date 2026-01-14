__all__ = [
    'ComposedCallback',
]

from typing import Iterable

from .base import BaseCallback


class ComposedCallback(BaseCallback):

    def __init__(
        self,
        *args,
        callbacks: Iterable[BaseCallback],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._callbacks = callbacks

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        for callback in self._callbacks:
            callback.bind(*args, **kwargs)

    def should_break(self) -> bool:
        return super().should_break() or any(
            callback.should_break() for callback in self._callbacks
        )

    def before_step(self) -> None:
        super().before_step()
        for callback in self._callbacks:
            callback.before_step()

    def after_step(self) -> None:
        super().after_step()
        for callback in self._callbacks:
            callback.after_step()

    def before_run(self) -> None:
        super().before_run()
        for callback in self._callbacks:
            callback.before_run()

    def after_run(self) -> None:
        super().after_run()
        for callback in self._callbacks:
            callback.after_run()
