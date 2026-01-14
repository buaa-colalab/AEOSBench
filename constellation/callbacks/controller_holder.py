__all__ = [
    'ControllerHolder',
]

from todd.utils import HolderMixin

from ..controller import Controller


class ControllerHolder(HolderMixin[Controller]):

    def __init__(
        self,
        *args,
        controller: Controller | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, instance=controller, **kwargs)

    @property
    def controller(self) -> Controller:
        return self._instance
