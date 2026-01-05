__all__ = [
    'Coordinate',
    'TaskDict',
    'TaskDicts',
    'Task',
    'Taskset',
]

import dataclasses
import random
from itertools import starmap
from collections import UserList
from typing import Any, Iterable, NamedTuple, TypedDict, TypeVar, cast
from typing_extensions import Self

import torch
from todd.patches.py_ import json_dump, json_load

from ..constants import MAX_TIME_STEP
from .constellations import SensorType


class Coordinate(NamedTuple):
    x: float  # latitude
    y: float  # longitude


class TaskDict(TypedDict):
    id: int
    release_time: int
    due_time: int
    duration: int
    coordinate: Coordinate
    sensor_type: SensorType


TaskDicts = list[TaskDict]


@dataclasses.dataclass(frozen=True)
class Task:
    id_: int
    release_time: int
    due_time: int
    duration: int
    coordinate: Coordinate
    sensor_type: SensorType

    def to_dict(self) -> TaskDict:
        d = dataclasses.asdict(self)
        d['id'] = d.pop('id_')
        return cast(TaskDict, d)

    @classmethod
    def from_dict(cls, task: TaskDict) -> Self:
        d = cast(dict[str, Any], task.copy())
        d['id_'] = d.pop('id')
        d['coordinate'] = Coordinate(*d['coordinate'])
        d['sensor_type'] = SensorType(d['sensor_type'])
        return cls(**d)

    @property
    def data(self) -> list[float]:
        return [
            self.release_time,
            self.due_time,
            self.duration,
            *self.coordinate,
        ]

    @classmethod
    def sample(cls, id_: int, test: bool) -> Self:
        # e.g. 20
        duration = random.randint(15, 60) if not test else 10
        # e.g. 3540 \in [0, 3540]
        release_time = random.randint(
            0, MAX_TIME_STEP - duration * 3
        ) if not test else 0
        # e.g. 3600 \in [3600, 3600]
        due_time = random.randint(
            release_time + duration * 3, MAX_TIME_STEP
        ) if not test else 3600
        return cls(
            id_,
            release_time,
            due_time,
            duration,
            Coordinate(
                random.uniform(-90, 90)
                if not test else random.uniform(-10, 10),
                random.uniform(-180, 180),
            ),
            SensorType.VISIBLE,
        )


class Taskset(UserList[Task]):

    @property
    def durations(self) -> list[int]:
        return [task.duration for task in self]

    def to_dicts(self) -> TaskDicts:
        return [task.to_dict() for task in self]

    @classmethod
    def from_dicts(cls, tasks: TaskDicts) -> Self:
        return cls(map(Task.from_dict, tasks))

    def dump(self, f: Any) -> None:
        json_dump(self.to_dicts(), f)

    @classmethod
    def load(cls, f: Any) -> Self:
        return cls.from_dicts(json_load(f))

    @classmethod
    def sample(cls, n: int, test: bool) -> Self:
        return cls(starmap(Task.sample, ((i, test) for i in range(n))))

    def to_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        sensor_type = torch.tensor([task.sensor_type for task in self])
        data = torch.tensor([task.data for task in self])
        return sensor_type, data
