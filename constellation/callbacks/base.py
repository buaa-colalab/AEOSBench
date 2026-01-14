__all__ = [
    'BaseCallback',
]

from .controller_holder import ControllerHolder


class BaseCallback(ControllerHolder):

    def should_break(self) -> bool:
        return False

    def before_step(self) -> None:
        pass

    def after_step(self) -> None:
        pass

    def before_run(self) -> None:
        pass

    def after_run(self) -> None:
        pass
