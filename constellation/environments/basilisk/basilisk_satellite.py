# mypy: disable-error-code="arg-type,assignment,operator"

import dataclasses
from typing import Iterable

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.architecture.messaging import (
    SCStatesMsgPayload,
    VehicleConfigMsg,
    VehicleConfigMsgPayload,
)
from Basilisk.fswAlgorithms.locationPointing import locationPointing
from Basilisk.fswAlgorithms.mrpFeedback import mrpFeedback
from Basilisk.fswAlgorithms.rwMotorTorque import rwMotorTorque
from Basilisk.simulation.eclipse import Eclipse
from Basilisk.simulation.groundLocation import GroundLocation
from Basilisk.simulation.groundMapping import GroundMapping
from Basilisk.simulation.ReactionWheelPower import ReactionWheelPower
from Basilisk.simulation.reactionWheelStateEffector import (
    ReactionWheelStateEffector,
)
from Basilisk.simulation.simpleBattery import SimpleBattery
from Basilisk.simulation.simpleNav import SimpleNav
from Basilisk.simulation.simplePowerSink import SimplePowerSink
from Basilisk.simulation.simpleSolarPanel import SimpleSolarPanel
from Basilisk.simulation.spacecraft import HubEffector, Spacecraft
from Basilisk.utilities import macros, orbitalMotion, unitTestSupport
from Basilisk.utilities.simIncludeGravBody import (
    gravBodyFactory,
    spiceInterface,
)
from Basilisk.utilities.simIncludeRW import rwFactory
from Basilisk.utilities.simulationArchTypes import ProcessBaseClass
from Basilisk.utilities.SimulationBaseClass import SimBaseClass

from ...constants import INTERVAL
from ...data import (
    Battery,
    MRPControl,
    Orbit,
    ReactionWheel,
    Satellite,
    Sensor,
    SolarPanel,
)
from ...data.constellations import SensorType
from .constants import IDENTITY_MATRIX_3, UNIT_VECTOR_Z
from ...constants import MU_EARTH, RADIUS_EARTH


class BasiliskSatellite:

    def __init__(
        self,
        simulator: SimBaseClass,
        process: ProcessBaseClass,
        grav_body_factory: gravBodyFactory,
        spice_object: spiceInterface.SpiceInterface,
        satellite: Satellite,
    ) -> None:
        self._id = satellite.id_
        self._orbit_id = satellite.orbit_id

        # Create task
        self._task_name = f"task-{self._id}"
        task_timestep = macros.sec2nano(INTERVAL)
        process.addTask(
            simulator.CreateNewTask(self._task_name, task_timestep),
        )

        self.setup_models(simulator, satellite)

        self.connect_messages(grav_body_factory, spice_object)

        # Custom variables
        self._sensor_type = satellite.sensor.type_

        self._reaction_wheels = satellite.reaction_wheels

    def setup_models(
        self,
        simulator: SimBaseClass,
        satellite: Satellite,
    ) -> None:
        self._spacecraft = self.setup_spacecraft(satellite)
        simulator.AddModelToTask(self._task_name, self._spacecraft)

        self._eclipse = self.setup_eclipse()
        simulator.AddModelToTask(self._task_name, self._eclipse)

        self._solar_panel = self.setup_solar_panel(satellite)
        simulator.AddModelToTask(self._task_name, self._solar_panel)

        self._power_sink = self.setup_power_sink(satellite)
        simulator.AddModelToTask(self._task_name, self._power_sink)

        self._battery = self.setup_battery(satellite)
        simulator.AddModelToTask(self._task_name, self._battery)

        self._simple_navigation = self.setup_simple_navigation()
        simulator.AddModelToTask(self._task_name, self._simple_navigation)

        self._pointing_location = self.setup_pointing_location()
        simulator.AddModelToTask(self._task_name, self._pointing_location)

        self._pointing_guide = self.setup_pointing_guide()
        simulator.AddModelToTask(self._task_name, self._pointing_guide)

        self._ground_mapping = self.setup_ground_mapping(satellite)
        simulator.AddModelToTask(self._task_name, self._ground_mapping)

        self._rw_factory = self.setup_rw_factory(satellite)
        # NOTE: not necessary to AddModelToTask

        self._mrp_control = self.setup_mrp_control(satellite)
        simulator.AddModelToTask(self._task_name, self._mrp_control)

        self._rw_motor_torque = self.setup_rw_motor_torque()
        simulator.AddModelToTask(self._task_name, self._rw_motor_torque)

        self._rw_state_effector = self.setup_rw_state_effector()
        simulator.AddModelToTask(self._task_name, self._rw_state_effector)

        self._rw_power_list = self.setup_rw_power_list(satellite)
        for rw_power in self._rw_power_list:
            simulator.AddModelToTask(self._task_name, rw_power)

    def setup_spacecraft(self, satellite: Satellite) -> Spacecraft:
        spacecraft = Spacecraft()
        spacecraft.ModelTag = f'spacecraft-{self.id_}'
        hub: HubEffector = spacecraft.hub
        hub.r_CN_NInit, hub.v_CN_NInit = satellite.rv
        inertia = satellite.inertia
        hub.mHub = satellite.mass  # kg - spacecraft mass
        # m - position vector of body-fixed point B relative to CM
        hub.r_BcB_B = np.reshape(satellite.center_of_mass, (-1, 1))
        hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(inertia)
        hub.sigma_BNInit = np.reshape(satellite.mrp_attitude_bn, (-1, 1))
        return spacecraft

    def setup_eclipse(self) -> Eclipse:
        eclipse = Eclipse()
        eclipse.ModelTag = f'eclipse-{self._id}'
        return eclipse

    def setup_solar_panel(self, satellite: Satellite) -> SimpleSolarPanel:
        solar_panel = SimpleSolarPanel()
        solar_panel.ModelTag = f'solar_panel-{self._id}'
        solar_panel.setPanelParameters(
            *dataclasses.astuple(satellite.solar_panel),
        )
        return solar_panel

    def setup_power_sink(self, satellite: Satellite) -> SimplePowerSink:
        power_sink = SimplePowerSink()
        power_sink.ModelTag = f"power_sink-{self._id}"
        power_sink.powerStatus = satellite.sensor.enabled
        power_sink.nodePowerOut = -satellite.sensor.power
        return power_sink

    def setup_battery(self, satellite: Satellite) -> SimpleBattery:
        battery = SimpleBattery()
        battery.ModelTag = f"battery-{self._id}"
        battery.storageCapacity = satellite.battery.capacity
        battery.storedCharge_Init = (
            satellite.battery.percentage * satellite.battery.capacity
        )
        return battery

    def setup_simple_navigation(self) -> SimpleNav:
        simple_navigation = SimpleNav()
        simple_navigation.ModelTag = f"simple_navigation-{self._id}"
        return simple_navigation

    def setup_pointing_location(self) -> GroundLocation:
        pointing_location = GroundLocation()
        pointing_location.ModelTag = f"pointing_location-{self._id}"
        pointing_location.planetRadius = RADIUS_EARTH
        # TODO 卫星摆幅限制
        pointing_location.minimumElevation = 0
        return pointing_location

    def setup_pointing_guide(self) -> locationPointing:
        pointing_guide = locationPointing()
        pointing_guide.ModelTag = f"pointing_guide-{self._id}"
        pointing_guide.pHat_B = UNIT_VECTOR_Z
        return pointing_guide

    def setup_ground_mapping(self, satellite: Satellite) -> GroundMapping:
        ground_mapping = GroundMapping()
        ground_mapping.ModelTag = f"ground_mapping-{self._id}"
        ground_mapping.minimumElevation = 0
        ground_mapping.maximumRange = 1e9
        ground_mapping.cameraPos_B = [0, 0, 0]  # at the center of satellite
        ground_mapping.nHat_B = UNIT_VECTOR_Z
        ground_mapping.halfFieldOfView = np.radians(
            satellite.sensor.half_field_of_view,
        )
        return ground_mapping

    def setup_rw_factory(self, satellite: Satellite) -> rwFactory:
        rw_factory = rwFactory()
        for i, reaction_wheel in enumerate(satellite.reaction_wheels):
            rw_factory.create(
                reaction_wheel.rw_type,
                reaction_wheel.rw_direction,
                maxMomentum=reaction_wheel.max_momentum,
                Omega=reaction_wheel.rw_speed_init,
                RWModel=messaging.BalancedWheels,
                label=self.reaction_wheel_id(i),
            )
        return rw_factory

    def setup_rw_motor_torque(self) -> rwMotorTorque:
        rw_motor_torque = rwMotorTorque()
        rw_motor_torque.ModelTag = f"rw_motor_torque-{self._id}"
        rw_motor_torque.controlAxes_B = IDENTITY_MATRIX_3
        return rw_motor_torque

    def setup_mrp_control(self, satellite: Satellite) -> mrpFeedback:
        mrp_control = mrpFeedback()
        mrp_control.ModelTag = f"mrpFeedback-{self._id}"
        mrp_control.K = satellite.mrp_control.k
        # make Ki negative to turn off integral feedback
        mrp_control.Ki = satellite.mrp_control.ki
        mrp_control.P = satellite.mrp_control.p
        mrp_control.integralLimit = satellite.mrp_control.integral_limit
        satellite_config_out = VehicleConfigMsgPayload()
        satellite_config_out.ISCPntB_B = satellite.inertia
        config_data_msg = VehicleConfigMsg()
        config_data_msg.write(satellite_config_out)
        mrp_control.vehConfigInMsg.subscribeTo(config_data_msg)
        self._config_data_msg = config_data_msg  # NOTE: prevent garbage collection
        return mrp_control

    def setup_rw_state_effector(self) -> ReactionWheelStateEffector:
        rw_state_effector = ReactionWheelStateEffector()
        rw_state_effector.ModelTag = f'rw_state_effector-{self._id}'
        return rw_state_effector

    def setup_rw_power_list(self,
                            satellite: Satellite) -> list[ReactionWheelPower]:
        rw_power_list: list[ReactionWheelPower] = []
        for i, reaction_wheel in enumerate(satellite.reaction_wheels):
            rw_power = ReactionWheelPower()
            rw_power.ModelTag = f"rw_power-{self._id}-{i}"
            rw_power.basePowerNeed = reaction_wheel.power
            # mechanical and electrical efficiency
            rw_power.mechToElecEfficiency = reaction_wheel.efficiency
            rw_power_list.append(rw_power)

        return rw_power_list

    def connect_messages(
        self,
        grav_body_factory: gravBodyFactory,
        spice_object: spiceInterface,
    ) -> None:
        earth_state = spice_object.planetStateOutMsgs[0]
        sun_state = spice_object.planetStateOutMsgs[1]

        # grav_factory
        grav_body_factory.addBodiesTo(self._spacecraft)

        # eclipse
        self._eclipse.addSpacecraftToModel(self._spacecraft.scStateOutMsg)
        self._eclipse.addPlanetToModel(earth_state)
        self._eclipse.sunInMsg.subscribeTo(sun_state)

        # solar_panel
        self._solar_panel.stateInMsg.subscribeTo(
            self._spacecraft.scStateOutMsg,
        )
        self._solar_panel.sunEclipseInMsg.subscribeTo(
            self._eclipse.eclipseOutMsgs[0],
        )
        self._solar_panel.sunInMsg.subscribeTo(sun_state)

        # battery
        self._battery.addPowerNodeToModel(self._solar_panel.nodePowerOutMsg)
        self._battery.addPowerNodeToModel(self._power_sink.nodePowerOutMsg)
        for rw_power in self._rw_power_list:
            self._battery.addPowerNodeToModel(rw_power.nodePowerOutMsg)

        # simple_navigation
        self._simple_navigation.scStateInMsg.subscribeTo(
            self._spacecraft.scStateOutMsg,
        )

        # pointing_location
        self._pointing_location.planetInMsg.subscribeTo(earth_state)
        self._pointing_location.addSpacecraftToModel(
            self._spacecraft.scStateOutMsg,
        )

        # pointing_guide
        self._pointing_guide.scAttInMsg.subscribeTo(
            self._simple_navigation.attOutMsg,
        )
        self._pointing_guide.scTransInMsg.subscribeTo(
            self._simple_navigation.transOutMsg,
        )
        self._pointing_guide.locationInMsg.subscribeTo(
            self._pointing_location.currentGroundStateOutMsg,
        )

        # ground_mapping
        self._ground_mapping.scStateInMsg.subscribeTo(
            self._spacecraft.scStateOutMsg,
        )
        self._ground_mapping.planetInMsg.subscribeTo(earth_state)

        # rw_factory
        self._rw_factory.addToSpacecraft(
            self._spacecraft.ModelTag,
            self._rw_state_effector,
            self._spacecraft,
        )

        # mrp_control
        # Store as instance variable to prevent garbage collection (bsk v2.9.0+)
        self._rw_params_message = self._rw_factory.getConfigMessage()
        self._mrp_control.guidInMsg.subscribeTo(
            self._pointing_guide.attGuidOutMsg,
        )
        self._mrp_control.rwParamsInMsg.subscribeTo(self._rw_params_message)
        self._mrp_control.rwSpeedsInMsg.subscribeTo(
            self._rw_state_effector.rwSpeedOutMsg,
        )

        # rw_motor_torque
        self._rw_motor_torque.vehControlInMsg.subscribeTo(
            self._mrp_control.cmdTorqueOutMsg,
        )
        self._rw_motor_torque.rwParamsInMsg.subscribeTo(self._rw_params_message)
        self._rw_state_effector.rwMotorCmdInMsg.subscribeTo(
            self._rw_motor_torque.rwMotorTorqueOutMsg,
        )

        # rw_power_list
        for rw_power, rw_out_message in zip(
            self._rw_power_list,
            self._rw_state_effector.rwOutMsgs,
        ):
            rw_power.rwStateInMsg.subscribeTo(rw_out_message)

    def to_satellite(self) -> Satellite:
        hub = self._spacecraft.hub
        inertia = np.reshape(hub.IHubPntBc_B, -1).tolist()
        mass = hub.mHub
        center_of_mass = np.squeeze(hub.r_BcB_B).tolist()
        orbital_elements = self.orbital_elements
        orbit = Orbit(
            self._orbit_id,
            orbital_elements.e,
            orbital_elements.a,
            orbital_elements.i / macros.D2R,
            orbital_elements.Omega / macros.D2R,
            orbital_elements.omega / macros.D2R,
        )
        solar_panel = SolarPanel(
            np.squeeze(self._solar_panel.nHat_B).tolist(),
            self._solar_panel.panelArea,
            self._solar_panel.panelEfficiency,
        )
        sensor = Sensor(
            self._sensor_type,
            self._power_sink.powerStatus,
            np.rad2deg(self._ground_mapping.halfFieldOfView),
            -self._power_sink.nodePowerOut,
        )
        battery = Battery(
            self._battery.storageCapacity,
            self._battery.batPowerOutMsg.read().storageLevel
            / self._battery.storageCapacity,
        )
        reaction_wheels: list[ReactionWheel] = []
        for i, _reaction_wheel in enumerate(self._reaction_wheels):
            reaction_wheels.append(
                ReactionWheel(
                    _reaction_wheel.rw_type,
                    _reaction_wheel.rw_direction,
                    _reaction_wheel.max_momentum,
                    self._rw_factory.rwList[self.reaction_wheel_id(i)].Omega
                    / macros.rpm2radsec,
                    _reaction_wheel.power,
                    _reaction_wheel.efficiency,
                ),
            )
        mrp_control = MRPControl(
            self._mrp_control.K,
            self._mrp_control.Ki,
            self._mrp_control.P,
            self._mrp_control.integralLimit,
        )
        true_anomaly = orbital_elements.f / macros.D2R
        mrp_attitude_bn = np.array(self.mrp_attitude_bn).squeeze().tolist()
        return Satellite(
            self._id,
            inertia,
            mass,
            center_of_mass,
            self._orbit_id,
            orbit,
            solar_panel,
            sensor,
            battery,
            reaction_wheels,
            mrp_control,
            true_anomaly,
            mrp_attitude_bn,
        )

    def toggle(self) -> None:
        self._power_sink.powerStatus = 1 - self._power_sink.powerStatus

    def guide_attitude(self, target_location: list[float] | None) -> None:
        if not target_location:
            self._pointing_location.specifyLocationPCPF([[0.0], [0.0], [0.0]])
        else:
            self._pointing_location.specifyLocation(
                np.radians(target_location[0]),
                np.radians(target_location[1]),
                0,
            )

    def reaction_wheel_id(self, index: int) -> str:
        return f'{index}RW{self._id}'

    @property
    def spacecraft(self) -> Spacecraft:
        return self._spacecraft

    @property
    def battery(self) -> SimpleBattery:
        return self._battery

    @property
    def solar_panel(self) -> SimpleSolarPanel:
        return self._solar_panel

    @property
    def power_sink(self) -> SimplePowerSink:
        return self._power_sink

    @property
    def sensor_type(self) -> SensorType:
        return self._sensor_type

    @property
    def ground_mapping(self) -> GroundMapping:
        return self._ground_mapping

    @property
    def id_(self) -> int:
        return self._id

    @property
    def spacecraft_state(self) -> SCStatesMsgPayload:
        return self._spacecraft.scStateOutMsg.read()

    @property
    def mrp_attitude_bn(self) -> Iterable[float]:
        return self._spacecraft.scStateOutMsg.read().sigma_BN

    @property
    def orbital_elements(self) -> orbitalMotion.ClassicElements:
        spacecraft_state: SCStatesMsgPayload = self.spacecraft_state
        return orbitalMotion.rv2elem(
            MU_EARTH,
            np.array(spacecraft_state.r_CN_N),
            np.array(spacecraft_state.v_CN_N),
        )
