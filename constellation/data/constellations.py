__all__ = [
    'Inertia',
    'SolarPanelDict',
    'SolarPanel',
    'SensorType',
    'SensorDict',
    'Sensor',
    'BatteryDict',
    'Battery',
    'ReactionWheelDict',
    'ReactionWheelDicts',
    'ReactionWheel',
    'ReactionWheels',
    'MRPControlDict',
    'MRPControl',
    'SatelliteDict',
    'SatelliteDicts',
    'Satellite',
    'Satellites',
    'ConstellationDict',
    'Constellation',
]

import dataclasses
import json
import math
import random
from collections import UserDict
from enum import IntEnum, auto
from typing import Any, TypedDict, cast
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

import torch
from todd.patches.py_ import json_dump, json_load
from Basilisk.utilities import macros, orbitalMotion

from ..constants import MU_EARTH
from .orbits import Orbit, OrbitDicts, Orbits
from .visualization import SatelliteDataJson

# TODO: rename properties

# yapf: disable
Inertia = tuple[  # kg/m^2
    float, float, float,
    float, float, float,
    float, float, float,
]
# yapf: enable


class SolarPanelDict(TypedDict):
    direction: tuple[float, float, float]
    area: float
    efficiency: float


@dataclasses.dataclass(frozen=True)
class SolarPanel:
    # unit normal vector in the body frame
    direction: tuple[float, float, float]
    area: float  # square meters
    efficiency: float  # [0., 1.]

    def to_dict(self) -> SolarPanelDict:
        solar_panel = dataclasses.asdict(self)
        return cast(SolarPanelDict, solar_panel)

    @property
    def data(self) -> list[float]:
        direction, *data = dataclasses.astuple(self)
        return [*direction, *data]

    @classmethod
    def sample(cls) -> Self:
        phi = random.uniform(0, 1) * math.pi
        theta = random.uniform(-0.5, 0.5) * math.pi
        return cls(
            (
                math.cos(theta) * math.cos(phi),
                math.cos(theta) * math.sin(phi),
                math.sin(theta),
            ),
            random.uniform(0.1, 0.5),
            random.uniform(0.1, 0.5),
        )


class SensorType(IntEnum):
    VISIBLE = auto()
    NEAR_INFRARED = auto()


class SensorDict(TypedDict):
    type: SensorType
    enabled: bool
    half_field_of_view: float
    power: float


@dataclasses.dataclass(frozen=True)
class Sensor:
    type_: SensorType
    enabled: bool
    half_field_of_view: float  # degrees
    power: float  # watts

    def to_dict(self) -> SensorDict:
        d = dataclasses.asdict(self)
        d['type'] = d.pop('type_')
        return cast(SensorDict, d)

    @classmethod
    def from_dict(cls, sensor: SensorDict) -> Self:
        d = cast(dict[str, Any], sensor.copy())
        d['type_'] = SensorType(d.pop('type'))
        return cls(**d)

    @property
    def data(self) -> list[float]:
        return [self.half_field_of_view, self.power]

    @classmethod
    def sample(cls) -> Self:
        return cls(
            SensorType.VISIBLE,
            False,
            random.uniform(0.1, 0.5),
            random.uniform(1, 10),
        )


class BatteryDict(TypedDict):
    capacity: float
    percentage: float


@dataclasses.dataclass(frozen=True)
class Battery:
    capacity: float  # joules
    percentage: float  # [0., 1.]

    def to_dict(self) -> BatteryDict:
        battery = dataclasses.asdict(self)
        return cast(BatteryDict, battery)

    @property
    def static_data(self) -> list[float]:
        return [self.capacity]

    @property
    def dynamic_data(self) -> list[float]:
        return [self.percentage]

    @classmethod
    def sample(cls) -> Self:
        return cls(
            random.uniform(8000, 30000),
            random.uniform(0.1, 1.0),
        )


class ReactionWheelDict(TypedDict):
    rw_type: str
    rw_direction: tuple[float, float, float]
    max_momentum: float
    rw_speed_init: float
    power: float
    efficiency: float


ReactionWheelDicts = tuple[
    ReactionWheelDict,
    ReactionWheelDict,
    ReactionWheelDict,
]


@dataclasses.dataclass(frozen=True)
class ReactionWheel:
    rw_type: str
    rw_direction: tuple[float, float, float]  # unit vector
    max_momentum: float  # N m s
    rw_speed_init: float  # round per minute
    power: float  # watts
    efficiency: float  # (0., 1.]

    def to_dict(self) -> ReactionWheelDict:
        reaction_wheel = dataclasses.asdict(self)
        return cast(ReactionWheelDict, reaction_wheel)

    @property
    def static_data(self) -> list[float]:
        return [
            *self.rw_direction,
            self.max_momentum,
            self.power,
            self.efficiency,
        ]

    @property
    def dynamic_data(self) -> list[float]:
        return [self.rw_speed_init]


ReactionWheels = tuple[ReactionWheel, ReactionWheel, ReactionWheel]


class MRPControlDict(TypedDict):
    k: float
    ki: float
    p: float
    integral_limit: float


@dataclasses.dataclass(frozen=True)
class MRPControl:
    k: float
    ki: float
    p: float
    integral_limit: float

    def to_dict(self) -> MRPControlDict:
        mrp_control = dataclasses.asdict(self)
        return cast(MRPControlDict, mrp_control)

    @property
    def data(self) -> list[float]:
        return list(dataclasses.astuple(self))


class SatelliteDict(TypedDict):
    id: int
    inertia: Inertia
    mass: float  # kg
    center_of_mass: tuple[float, float, float]  # m
    orbit_id: int
    solar_panel: SolarPanelDict
    sensor: SensorDict
    battery: BatteryDict
    reaction_wheels: ReactionWheelDicts
    mrp_control: MRPControlDict
    true_anomaly: float  # degrees
    mrp_attitude_bn: tuple[float, float, float]


SatelliteDicts = list[SatelliteDict]


@dataclasses.dataclass(frozen=True)
class Satellite:
    id_: int
    inertia: Inertia
    mass: float  # kg
    center_of_mass: tuple[float, float, float]  # m
    orbit_id: int
    orbit: Orbit
    solar_panel: SolarPanel
    sensor: Sensor
    battery: Battery
    reaction_wheels: ReactionWheels
    mrp_control: MRPControl
    true_anomaly: float  # degrees
    mrp_attitude_bn: tuple[float, float, float]

    def to_dict(self) -> SatelliteDict:
        d = dataclasses.asdict(self)
        d['id'] = d.pop('id_')
        d['orbit'] = d.pop('orbit_id')
        d['solar_panel'] = self.solar_panel.to_dict()
        d['sensor'] = self.sensor.to_dict()
        d['battery'] = self.battery.to_dict()
        d['reaction_wheels'] = tuple(
            reaction_wheel.to_dict() for reaction_wheel in self.reaction_wheels
        )
        d['mrp_control'] = self.mrp_control.to_dict()
        return cast(SatelliteDict, d)

    @classmethod
    def from_dict(
        cls,
        satellite: SatelliteDict,
        orbits: dict[int, Orbit],
    ) -> Self:
        d = cast(dict[str, Any], satellite.copy())
        d['id_'] = d.pop('id')
        d['orbit_id'] = d['orbit']
        d['orbit'] = orbits[d['orbit']]
        d['solar_panel'] = SolarPanel(**d['solar_panel'])
        d['sensor'] = Sensor.from_dict(d['sensor'])
        d['battery'] = Battery(**d['battery'])
        d['reaction_wheels'] = tuple(
            ReactionWheel(**reaction_wheel)
            for reaction_wheel in d.pop('reaction_wheels')
        )
        d['mrp_control'] = MRPControl(**d.pop('mrp_control'))
        return cls(**d)

    @classmethod
    def sample_mrp_fit(
        cls,
        id_: int,
        orbit: Orbit,
        *,
        rng: random.Random | None = None,
    ) -> Self:
        def uniform(a: float, b: float) -> float:
            if rng is None:
                return random.uniform(a, b)
            return rng.uniform(a, b)

        def choice(seq: list[Any]) -> Any:
            return (rng.choice(seq) if rng is not None else random.choice(seq))

        inertia: Inertia = (
            round(uniform(50, 200), 6),
            0.0,
            0.0,
            0.0,
            round(uniform(50, 200), 6),
            0.0,
            0.0,
            0.0,
            round(uniform(50, 200), 6),
        )
        mass = round(uniform(50, 200), 3)

        while True:
            direction = np.array([uniform(-1, 1) for _ in range(3)])
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                direction = direction / norm
                solar_direction = tuple(round(float(x), 3) for x in direction)
                break

        solar_panel = SolarPanel(
            solar_direction,  # type: ignore[arg-type]
            round(uniform(5, 10), 3),
            round(uniform(0.1, 0.6), 3),
        )
        sensor = Sensor(
            SensorType.VISIBLE,
            bool(choice([True, False])),
            round(uniform(0.5, 1.5), 2),
            round(uniform(2, 8), 3),
        )
        battery = Battery(
            round(uniform(8000, 30000), 1),
            round(uniform(0.7, 1.0), 3),
        )

        rw_type = cast(str, choice(['12', '14', '16']))
        if rw_type == '12':
            max_momentum = cast(float, choice([12.0, 25.0, 50.0]))
        elif rw_type == '14':
            max_momentum = cast(float, choice([75.0, 25.0, 50.0]))
        else:
            max_momentum = cast(float, choice([100.0, 75.0, 50.0]))

        reaction_wheels: ReactionWheels = (
            ReactionWheel(
                'Honeywell_HR' + rw_type,
                (1.0, 0.0, 0.0),
                max_momentum,
                round(uniform(400, 750), 3),
                round(uniform(5, 7), 3),
                round(uniform(0.5, 0.6), 3),
            ),
            ReactionWheel(
                'Honeywell_HR' + rw_type,
                (0.0, 1.0, 0.0),
                max_momentum,
                round(uniform(400, 750), 3),
                round(uniform(5, 7), 3),
                round(uniform(0.5, 0.6), 3),
            ),
            ReactionWheel(
                'Honeywell_HR' + rw_type,
                (0.0, 0.0, 1.0),
                max_momentum,
                round(uniform(400, 750), 3),
                round(uniform(5, 7), 3),
                round(uniform(0.5, 0.6), 3),
            ),
        )

        mrp_k = uniform(0, 10)
        mrp_control = MRPControl(
            round(mrp_k, 6),
            round(uniform(0, 0.001), 6),
            round(uniform(2, 5) * mrp_k, 6),
            round(uniform(0, 0.001), 6),
        )

        return cls(
            id_,
            inertia,
            mass,
            (0.0, 0.0, 0.0),
            orbit.id_,
            orbit,
            solar_panel,
            sensor,
            battery,
            reaction_wheels,
            mrp_control,
            0.0,
            (0.0, 0.0, 0.0),
        )

    @property
    def rv(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        orbital_elements = orbitalMotion.ClassicElements()
        orbital_elements.e = self.orbit.eccentricity
        orbital_elements.a = self.orbit.semi_major_axis
        orbital_elements.i = self.orbit.inclination * macros.D2R
        orbital_elements.Omega = (
            self.orbit.right_ascension_of_the_ascending_node * macros.D2R
        )
        orbital_elements.omega = (self.orbit.argument_of_perigee * macros.D2R)
        orbital_elements.f = self.true_anomaly * macros.D2R
        return orbitalMotion.elem2rv(
            MU_EARTH,
            orbital_elements,
        )  # r_CN_N, v_CN_N

    @property
    def static_data(self) -> list[float]:
        reaction_wheels: list[float] = []
        for reaction_wheel in self.reaction_wheels:
            reaction_wheels.extend(reaction_wheel.static_data)

        return [
            *self.inertia,
            self.mass,
            *self.center_of_mass,
            *self.orbit.data,
            *self.solar_panel.data,
            *self.sensor.data,
            *self.battery.static_data,
            *reaction_wheels,
            *self.mrp_control.data,
        ]

    @property
    def dynamic_data(self) -> list[float]:
        reaction_wheels: list[float] = []
        for reaction_wheel in self.reaction_wheels:
            reaction_wheels.extend(reaction_wheel.dynamic_data)

        return [
            *self.battery.dynamic_data,
            *reaction_wheels,
            self.true_anomaly,
            *self.mrp_attitude_bn,
        ]


Satellites = list[Satellite]


class ConstellationDict(TypedDict):
    orbits: OrbitDicts
    satellites: SatelliteDicts


class Constellation(UserDict[int, Satellite]):

    @property
    def orbits(self) -> Orbits:
        orbit_dict = {
            satellite.orbit_id: satellite.orbit
            for satellite in self.values()
        }
        return Orbits(
            sorted(
                orbit_dict.values(),
                key=lambda orbit: orbit.id_,
            ),
        )

    @property
    def eci_locations(self) -> torch.Tensor:
        locations: list[torch.Tensor] = []
        for satellite in self.sort():
            r_CN_N, _ = satellite.rv
            locations.append(torch.from_numpy(r_CN_N.astype(np.float32)))
        return torch.stack(locations)

    def sort(self) -> Satellites:
        return sorted(
            self.values(),
            key=lambda satellite: satellite.id_,
        )

    def to_dict(self) -> ConstellationDict:
        return ConstellationDict(
            orbits=self.orbits.to_dicts(),
            satellites=[satellite.to_dict() for satellite in self.sort()],
        )

    @classmethod
    def from_dict(cls, constellation: ConstellationDict) -> Self:
        orbits = {
            orbit['id']: Orbit.from_dict(orbit)
            for orbit in constellation['orbits']
        }
        satellites = {
            satellite['id']: Satellite.from_dict(satellite, orbits)
            for satellite in constellation['satellites']
        }
        return cls(satellites)

    def dump(self, f: Any) -> None:
        json_dump(self.to_dict(), f)

    def dump_std_json(self, f: Any, *, indent: int = 4) -> None:
        json.dump(self.to_dict(), f, indent=indent, default=int)

    @classmethod
    def load(cls, f: Any) -> Self:
        return cls.from_dict(json_load(f))

    @classmethod
    def sample_mrp_fit_single(
        cls,
        *,
        orbit: Orbit | None = None,
        satellite_id: int = 0,
        rng: random.Random | None = None,
    ) -> Self:
        orbit = Orbit.mrp_fit_default(0) if orbit is None else orbit
        satellite = Satellite.sample_mrp_fit(satellite_id, orbit, rng=rng)
        return cls({satellite.id_: satellite})

    def static_to_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        satellites = self.sort()
        sensor_type = torch.tensor([
            satellite.sensor.type_ for satellite in satellites
        ])
        data = torch.tensor([
            satellite.static_data for satellite in satellites
        ])
        # TODO: check data type is float32
        return sensor_type, data

    def dynamic_to_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        satellites = self.sort()
        sensor_enabled = torch.tensor([
            satellite.sensor.enabled for satellite in satellites
        ])
        data = torch.tensor([
            satellite.dynamic_data for satellite in satellites
        ])
        return sensor_enabled, data

    def to_unity_json(self) -> list[SatelliteDataJson]:
        constellation_data: list[SatelliteDataJson] = []
        for satellite in self.values():
            r_CN_N, v_CN_N = satellite.rv
            satellite_data = SatelliteDataJson(
                satelliteId=satellite.id_,
                eciLocation=r_CN_N,
                eciVelocity=v_CN_N,
                mrpAttitude=satellite.mrp_attitude_bn,
            )
            constellation_data.append(satellite_data)
        return constellation_data
