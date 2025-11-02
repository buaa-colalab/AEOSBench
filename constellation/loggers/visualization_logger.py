import json
from math import e
import pathlib

import numpy as np
import torch

from constellation import task_managers

from ..environments.base import BaseEnvironment

from ..constants import INTERVAL, TIMESTAMP
from ..data.constellations import Constellation
from ..environments.basilisk.constants import RADIUS_EARTH
from ..environments.geodetics import GeodeticConversion
from ..data import (
    InitDataJson,
    SatVisibleDataJson,
    TaskDataJson,
    EarthRotationJson,
    ConstellationJson,
    VisualizationJson,
)
from .base_logger import BaseLogger


class VisualizationLogger(BaseLogger):
    """Logger for visualization data in JSON format."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        init_data = InitDataJson(
            timeString=TIMESTAMP,
            interval=INTERVAL,
            earthRadius=RADIUS_EARTH,
        )

        self._visualization_json = VisualizationJson(
            initData=init_data,
            earthRotation=[],
            environment=[],
        )

    def on_step_end(self, **kwargs) -> None:
        is_visible = self.environment.is_visible(self.task_manager.all_tasks)
        constellation: Constellation = self.environment.get_constellation()

        (earth_rotation, constellation_data) = (
            self.environment.get_earth_rotation(),
            constellation.to_unity_json(),
        )

        tasks_data: list[TaskDataJson] = []
        for task in self.task_manager.ongoing_tasks:
            lla_location = (
                np.radians(task.coordinate.x),
                np.radians(task.coordinate.y),
                0,
            )
            ecef_location = GeodeticConversion.lla2pcpf(lla_location)
            tasks_data.append(
                TaskDataJson(
                    taskId=task.id_,
                    sensorType=task.sensor_type,
                    ecefLocation=ecef_location,
                ),
            )

        visible_data: list[SatVisibleDataJson] = []
        for i, is_visible_ in enumerate(is_visible):
            target_list, = torch.where(is_visible_)
            visible_data.append(
                SatVisibleDataJson(
                    satelliteId=constellation_data[i]['satelliteId'],
                    targetList=target_list.tolist(),
                )
            )

        self._visualization_json['earthRotation'].append(
            EarthRotationJson(
                earthRotation=earth_rotation,
            )
        )

        self._visualization_json['environment'].append(
            ConstellationJson(
                constellation=constellation_data,
                tasks=tasks_data,
            )
        )

    def on_run_end(self, save_name: str, **kwargs) -> None:
        visualization_path = self._work_dir
        with open(
            visualization_path
            / pathlib.Path(f"{int(save_name):05}" + ".json"), 'w'
        ) as f:
            json.dump(self._process_format_legacy(self._visualization_json), f)

    def _process_format_legacy(self, vis_json: VisualizationJson) -> dict:
        result = {'initData': vis_json['initData'], 'timeData': []}

        assert (len(vis_json['earthRotation']) == len(vis_json['environment']))
        data_length = len(vis_json['environment'])

        for i in range(data_length):
            time_data = {
                'earthRotation': vis_json['earthRotation'][i]['earthRotation'],
                'constellation': vis_json['environment'][i]['constellation'],
                'tasks': vis_json['environment'][i]['tasks']
            }
            result['timeData'].append(time_data)

        return result
