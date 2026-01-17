__all__ = [
    'Coordinate',
    'TaskDict',
    'TaskDicts',
    'Task',
    'TaskSet',
]

import dataclasses
from functools import cached_property
import math
import random
from collections import UserList
from typing import Any, NamedTuple, TypedDict, cast
import einops
from typing_extensions import Self

import torch
from todd.patches.py_ import json_dump, json_load

from ..constants import MAX_TIME_STEP, RADIUS_EARTH, ECCENTRICITY_EARTH, MU_EARTH
from .constellations import SensorType
from .coordinates import Coordinate, CoordinateECEF


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

    @cached_property
    def coordinate_ecef(self) -> CoordinateECEF:
        latitude = math.radians(self.coordinate.x)
        longitude = math.radians(self.coordinate.y)
        altitude = 0.

        # N is the prime vertical radius of curvature
        n = RADIUS_EARTH / math.sqrt(
            1.0 - ECCENTRICITY_EARTH * math.sin(latitude)**2,
        )

        return CoordinateECEF(
            (n + altitude) * math.cos(latitude) * math.cos(longitude),
            (n + altitude) * math.cos(latitude) * math.sin(longitude),
            ((1.0 - ECCENTRICITY_EARTH) * n + altitude) * math.sin(latitude),
        )

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
    def sample(cls, id_: int) -> Self:
        # e.g. 20
        duration = random.randint(15, 60)

        # e.g. 3540 \in [0, 3540]
        release_time = random.randint(0, MAX_TIME_STEP - duration * 3)
        # e.g. 3600 \in [3600, 3600]
        due_time = random.randint(release_time + duration * 3, MAX_TIME_STEP)
        return cls(
            id_,
            release_time,
            due_time,
            duration,
            Coordinate(
                random.uniform(-90, 90),
                random.uniform(-180, 180),
            ),
            SensorType.VISIBLE,
        )

    @classmethod
    def sample_mrp(cls, id_: int) -> Self:
        return cls(
            id_,
            0,
            3600,
            10,
            Coordinate(
                random.uniform(-10, 10),
                random.uniform(-180, 180),
            ),
            SensorType.VISIBLE,
        )


class TaskSet(UserList[Task]):

    @property
    def ids(self) -> torch.Tensor:
        return torch.tensor([task.id_ for task in self])

    @property
    def release_times(self) -> torch.Tensor:
        return torch.tensor([task.release_time for task in self])

    @property
    def durations(self) -> torch.Tensor:
        return torch.tensor([task.duration for task in self])

    @property
    def coordinates_ecef(self) -> list[CoordinateECEF]:
        return [task.coordinate_ecef for task in self]

    def get_task_indices(self, task_ids: torch.Tensor) -> torch.Tensor:
        task_ids = einops.rearrange(task_ids, 'nt -> nt 1')
        return torch.argmax((task_ids == self.ids).int(), 1)

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
    def sample(cls, n: int) -> Self:
        return cls(map(Task.sample, range(n)))

    @classmethod
    def sample_mrp(cls, n: int) -> Self:
        return cls(map(Task.sample_mrp, range(n)))

    def to_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        sensor_type = torch.tensor([task.sensor_type for task in self])
        data = torch.tensor([task.data for task in self])
        return sensor_type, data
