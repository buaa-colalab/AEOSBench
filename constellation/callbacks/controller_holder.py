__all__ = [
    'ControllerHolder',
]

import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..controller import Controller

class ControllerHolder:

    def set_controller(self, controller: 'Controller'):
        self._controller_ref = weakref.ref(controller)

    @property
    def controller(self) -> 'Controller':
        return self._controller_ref()
