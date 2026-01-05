__all__ = [
    'OrbitDict',
    'OrbitDicts',
    'Orbit',
    'Orbits',
]

import dataclasses
import random
from collections import UserList
from typing import Any, TypedDict, cast
from typing_extensions import Self


class OrbitDict(TypedDict):
    id: int
    eccentricity: float
    semi_major_axis: float
    inclination: float
    right_ascension_of_the_ascending_node: float
    argument_of_perigee: float


OrbitDicts = list[OrbitDict]


@dataclasses.dataclass(frozen=True)
class Orbit:
    """Orbital elements of a satellite.

    Refer to https://en.wikipedia.org/wiki/Orbital_elements.
    """

    id_: int
    eccentricity: float
    semi_major_axis: float  # meters
    inclination: float  # degrees
    right_ascension_of_the_ascending_node: float  # degrees
    argument_of_perigee: float  # degrees

    def to_dict(self) -> OrbitDict:
        d = dataclasses.asdict(self)
        d['id'] = d.pop('id_')
        return cast(OrbitDict, d)

    @classmethod
    def from_dict(cls, orbit: OrbitDict) -> Self:
        d = cast(dict[str, Any], orbit.copy())
        d['id_'] = d.pop('id')
        return cls(**d)

    @property
    def data(self) -> list[float]:
        _, *data = dataclasses.astuple(self)
        return cast(list[float], data)

    @classmethod
    def sample(cls, id_: int) -> Self:
        return cls(
            id_,
            round(random.uniform(0, 0.005), 6),
            round(random.uniform(6.8e6, 8e6), 1),
            round(random.uniform(0, 180), 1),
            round(random.uniform(0, 360), 1),
            round(random.uniform(0, 360), 1),
        )

    @classmethod
    def mrp_fit_default(cls, id_: int = 0) -> Self:
        return cls(
            id_,
            0.0001,
            7200000.0,
            0.0,
            0.0,
            0.0,
        )


class Orbits(UserList[Orbit]):

    def to_dicts(self) -> OrbitDicts:
        return [orbit.to_dict() for orbit in self]

    @classmethod
    def sample(cls, n: int) -> Self:
        return cls(map(Orbit.sample, range(n)))
