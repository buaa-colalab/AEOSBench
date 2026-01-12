__all__ = [
    'Coordinate',
    'CoordinateECEF',
]

from typing import NamedTuple


class Coordinate(NamedTuple):
    x: float  # latitude
    y: float  # longitude


class CoordinateECEF(NamedTuple):
    x: float
    y: float
    z: float
