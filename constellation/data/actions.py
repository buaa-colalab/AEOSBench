__all__ = [
    'ActionDict',
    'Action',
    'Actions',
]

import dataclasses
from collections import UserList
from typing import TypedDict, cast

import torch


class ActionDict(TypedDict):
    toggle: bool
    target_location: tuple[float, float] | None


@dataclasses.dataclass(frozen=True)
class Action:
    toggle: bool = False
    target_location: tuple[float, float] | None = None

    def to_dict(self) -> ActionDict:
        d = dataclasses.asdict(self)
        return cast(ActionDict, d)


class Actions(UserList[Action]):

    def to_dicts(self) -> list[ActionDict]:
        return [action.to_dict() for action in self]

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        toggles = torch.tensor([action.toggle for action in self])
        with_target_locations = torch.tensor([
            action.target_location is not None for action in self
        ])
        target_locations = torch.tensor([
            action.target_location or (0, 0) for action in self
        ])
        return toggles, with_target_locations, target_locations
