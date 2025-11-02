# flake8: noqa: N815

__all__ = [
    'InitDataJson',
    'SatelliteDataJson',
    'SatVisibleDataJson',
    'TaskDataJson',
    'EarthRotationJson',
    'ConstellationJson',
    'VisualizationJson',
]

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from .constellations import SensorType


class InitDataJson(TypedDict):
    timeString: str
    interval: float
    earthRadius: float


class SatelliteDataJson(TypedDict):
    satelliteId: int
    eciLocation: tuple[float, float, float]
    eciVelocity: tuple[float, float, float]
    mrpAttitude: tuple[float, float, float]


class SatVisibleDataJson(TypedDict):
    satelliteId: int
    targetList: list[int]


class TaskDataJson(TypedDict):
    taskId: int
    ecefLocation: tuple[float, float, float]
    sensorType: 'SensorType'


class EarthRotationJson(TypedDict):
    earthRotation: list[float]


class ConstellationJson(TypedDict):
    constellation: list[SatelliteDataJson]
    tasks: list[TaskDataJson]


class VisualizationJson(TypedDict):
    initData: InitDataJson
    earthRotation: list[EarthRotationJson]
    environment: list[ConstellationJson]
