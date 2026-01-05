# mypy: disable-error-code="arg-type,assignment,operator"
from __future__ import annotations

__all__ = [
    'BasiliskEnvironment',
]

import copy

import numpy as np
import torch
from Basilisk.utilities import macros
from Basilisk.utilities.simIncludeGravBody import gravBodyFactory
from Basilisk.utilities.SimulationBaseClass import SimBaseClass

from ...constants import INTERVAL, TIMESTAMP
from ...data import Actions, Constellation, Taskset
from ..base import BaseEnvironment
from ..geodetics import GeodeticConversion
from .basilisk_satellite import BasiliskSatellite
from .constants import RADIUS_EARTH
from .time import datetime2basilisk, str2datetime


class BasiliskEnvironment(BaseEnvironment):

    def __init__(
        self,
        *args,
        standard_time_init: str = TIMESTAMP,
        constellation: Constellation,
        all_tasks: Taskset,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        simulator = SimBaseClass()
        task_name = 'task_environment'
        process = simulator.CreateNewProcess(__file__)
        process.addTask(
            simulator.CreateNewTask(task_name, macros.sec2nano(INTERVAL)),
        )
        grav_body_factory = gravBodyFactory()
        earth = grav_body_factory.createEarth()
        earth.isCentralBody = True

        grav_body_factory.createSun()
        date_object = str2datetime(standard_time_init)
        basilisk_time_init = datetime2basilisk(date_object)
        spice_object = grav_body_factory.createSpiceInterface(
            time=basilisk_time_init,
        )
        spice_object.zeroBase = "Earth"
        simulator.AddModelToTask(task_name, spice_object)

        basilisk_satellites = [
            BasiliskSatellite(
                simulator,
                process,
                grav_body_factory,
                spice_object,
                satellite,
            ) for satellite in constellation.values()
        ]

        for basilisk_satellite in basilisk_satellites:
            for task in all_tasks:
                lla_location = (
                    np.radians(task.coordinate.x),
                    np.radians(task.coordinate.y),
                    0,
                )
                pcpf_location = GeodeticConversion.lla2pcpf(
                    lla_location,
                    RADIUS_EARTH,
                )
                task_location = np.array(pcpf_location)
                basilisk_satellite.ground_mapping.addPointToModel(
                    task_location,
                )

        simulator.InitializeSimulation()
        simulator.ConfigureStopTime(0)  # connect all message
        simulator.ExecuteSimulation()

        self._simulator = simulator
        self._satellites = basilisk_satellites
        self._spice_object = spice_object

        # for deepcopy
        self._standard_time_init = standard_time_init
        self._constellation = constellation
        self._all_tasks = all_tasks

    def deepcopy(self) -> BasiliskEnvironment:
        return BasiliskEnvironment(
            start_time=self._start_time,
            end_time=self._end_time,
            standard_time_init=self._standard_time_init,
            constellation=self._constellation,
            all_tasks=copy.deepcopy(self._all_tasks),
        )

    @property
    def all_tasks(self) -> Taskset:
        return self._all_tasks

    @property
    def standard_time_init(self) -> str:
        return self._standard_time_init

    @property
    def num_satellites(self) -> int:
        return len(self._satellites)

    def get_constellation(self) -> Constellation:
        constellation = [
            basilisk_satellite.to_satellite()
            for basilisk_satellite in self._satellites
        ]
        return Constellation({
            satellite.id_: satellite
            for satellite in constellation
        })

    def take_actions(self, actions: Actions) -> None:
        for satellite, action in zip(self._satellites, actions):
            if action.toggle:
                satellite.toggle()
            satellite.guide_attitude(action.target_location)

    def step(self) -> None:
        self._simulator.ConfigureStopTime(
            macros.sec2nano(self._timer.time * INTERVAL)
        )
        self._simulator.ExecuteSimulation()

    def is_visible(self, tasks: Taskset) -> torch.Tensor:
        visibility = torch.zeros(self.num_satellites, len(tasks))
        for satellite_idx, satellite in enumerate(self._satellites):
            for i, task in enumerate(tasks):
                access_message = satellite._ground_mapping.accessOutMsgs[i]
                access = access_message.read().hasAccess
                state = satellite.power_sink.powerStatus
                if access and state and (
                    task.sensor_type == satellite.sensor_type
                ):
                    visibility[satellite_idx, i] = 1
        return visibility

    def get_earth_rotation(self) -> torch.Tensor:
        earth_state_message = self._spice_object.planetStateOutMsgs[0]
        rotation = earth_state_message.read().J20002Pfix
        rotation = torch.tensor(rotation)
        return rotation

    def get_sat_eci_location(self) -> torch.Tensor:
        return torch.tensor(
            self.get_constellation().eci_locations,
            dtype=torch.float,
        )
