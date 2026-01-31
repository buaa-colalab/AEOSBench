__all__ = [
    'ControllerEnvironment',
]

from functools import partial
import random
from typing import Any, TypedDict, cast, List, Self

import einops
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces

from stable_baselines3.common.vec_env import SubprocVecEnv
from todd.patches.py_ import json_load
from constellation.new_transformers import Statistics
from constellation.new_transformers import SATELLITE_DIM, TASK_DIM
from constellation.data import SensorType

from constellation import (
    ANNOTATIONS_ROOT,
    CONSTELLATIONS_ROOT,
    STATISTICS_PATH,
    MAX_TIME_STEP,
    TASKSETS_ROOT,
    TIMESTAMP,
)
from constellation.data import (
    Action,
    Actions,
    Constellation,
    Task,
    TaskSet,
)
from constellation.environments import BasiliskEnvironment, BaseEnvironment
from constellation.evaluators import (
    BaseEvaluator,
    CompletionRateEvaluator,
    TurnAroundTimeEvaluator,
    PowerUsageEvaluator,
)
from constellation import TaskManager
from constellation.callbacks.base import BaseCallback
from constellation.callbacks import ComposedCallback
from constellation.controller import Controller
from constellation.rl.environment import Observation, null_observation, Padding

from todd.runners import Memo

MAX_NUM_SATELLITES = 51
MAX_NUM_TASKS = 302


class ControllerEnvironment(gym.Env[Observation, npt.NDArray[np.uint16]]):

    @classmethod
    def build(cls, world_size: int, *args, **kwargs) -> Self | SubprocVecEnv:
        if world_size == 0:
            return cls(*args, **kwargs)
        return SubprocVecEnv([
            partial(cls, *args, **kwargs) for _ in range(world_size)
        ])

    def __init__(self, *args, split: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._split = split

        self.observation_space = spaces.Dict(  # type: ignore[assignment]
            dict(
                num_satellites=spaces.Discrete(MAX_NUM_SATELLITES),
                num_tasks=spaces.Discrete(MAX_NUM_TASKS),
                time_step=spaces.Discrete(MAX_TIME_STEP),
                constellation_sensor_type=spaces.MultiDiscrete(
                    [len(SensorType)] * MAX_NUM_SATELLITES,
                ),
                constellation_sensor_enabled=spaces.MultiBinary(
                    MAX_NUM_SATELLITES,
                ),
                constellation_data=spaces.Box(
                    low=-1e5,
                    high=1e5,
                    shape=(MAX_NUM_SATELLITES, SATELLITE_DIM),
                ),
                tasks_sensor_type=spaces.MultiDiscrete([len(SensorType)]
                                                   * MAX_NUM_TASKS),
                tasks_data=spaces.Box(
                    low=-1e3,
                    high=1e3,
                    shape=(MAX_NUM_TASKS, TASK_DIM),
                ),
            )
        )
        self.action_space = spaces.MultiDiscrete(  # type: ignore[assignment]
            [MAX_NUM_TASKS] * MAX_NUM_SATELLITES,
        )

        self._annotations: list[int] = json_load(
            str(ANNOTATIONS_ROOT / f'{split}.json'),
        )['ids']
        self._statistics: Statistics = torch.load(
            STATISTICS_PATH,
            weights_only=False,
        )

        self._padding = Padding()
        self._episode_step = 0
        self._controller: Controller | None = None
        self._last_num_succeeded_tasks = 0

    def _require_controller(self) -> Controller:
        if self._controller is None:
            raise RuntimeError("Controller is not initialized")
        return self._controller

    @property
    def info(self) -> Memo:
        _controller = self._require_controller()
        # memo = Memo()
        # # print(self._controller.callbacks)
        _controller.callbacks.after_run()
        return _controller.memo

    def _load_constellation(
        self,
    ) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint8],
        npt.NDArray[np.float32],
    ]:
        _controller = self._require_controller()
        constellation = _controller.environment.get_constellation()
        sensor_type, static_data = constellation.static_to_tensor()
        sensor_enabled, dynamic_data = constellation.dynamic_to_tensor()
        data = torch.cat([static_data, dynamic_data], -1)
        data = ((data - self._statistics.constellation_mean) /
                (self._statistics.constellation_std + 1e-6))

        return sensor_type.numpy(), sensor_enabled.numpy(), data.numpy()

    def _load_tasks(
        self,
    ) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.float32],
    ]:
        _controller = self._require_controller()
        valid_tasks = _controller.task_manager.ongoing_tasks
        valid_labels = _controller.task_manager.ongoing_flags

        sensor_type, static_data = valid_tasks.to_tensor()

        static_data = static_data.clone()
        t = _controller.environment.timer.time
        static_data[..., 0] -= t
        static_data[..., 1] -= t

        progress = _controller.task_manager.progress[valid_labels]
        dynamic_data = einops.rearrange(progress, 'nt -> nt 1')

        data = torch.cat([static_data, dynamic_data], -1)
        data = ((data - self._statistics.taskset_mean) /
                (self._statistics.taskset_std + 1e-6))
        return sensor_type.numpy(), data.numpy()

    def _get_observation(self) -> Observation:
        (
            constellation_sensor_type,
            constellation_sensor_enabled,
            constellation_data,
        ) = self._load_constellation()
        _controller = self._require_controller()
        tasks_sensor_type, tasks_data = self._load_tasks()

        observation = Observation(
            num_satellites=_controller.environment.num_satellites,
            num_tasks=_controller.task_manager.num_ongoing_tasks,
            time_step=_controller.environment.timer.time,
            constellation_sensor_type=cast(
                npt.NDArray[np.uint8],
                constellation_sensor_type - 1,
            ),
            constellation_sensor_enabled=constellation_sensor_enabled,
            constellation_data=constellation_data,
            tasks_sensor_type=cast(
                npt.NDArray[np.uint8],
                tasks_sensor_type - 1,
            ),
            tasks_data=tasks_data,
        )

        return self._padding(observation)

    def _skip_idle(self, done) -> None:
        _controller = self._require_controller()
        while _controller.task_manager.is_idle and not done:
            self._take_actions(
                np.full(_controller._environment.num_satellites, -1),
            )

    def _get_annotation(self) -> int:
        return random.choice(self._annotations['ids'])

    def reset(self, *args, **kwargs) -> tuple[Observation, dict[str, Any]]:
        super().reset(*args, **kwargs)

        self._episode_step = 0
        id_ = self._get_annotation()
        self._current_id = id_

        constellation_path = (
            CONSTELLATIONS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        constellation = Constellation.load(str(constellation_path))

        taskset_path = (
            TASKSETS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        tasks: TaskSet = TaskSet.load(str(taskset_path))

        simulator = BasiliskEnvironment(
            standard_time_init=TIMESTAMP,
            constellation=constellation,
            all_tasks=tasks,
        )

        task_manager = TaskManager(
            timer=simulator.timer,
            taskset=tasks,
        )
        self._last_num_succeeded_tasks = 0

        evaluators = [
            CompletionRateEvaluator(),
            TurnAroundTimeEvaluator(),
            PowerUsageEvaluator(),
        ]
        callbacks = ComposedCallback(callbacks=evaluators)

        self._controller = Controller(
            name='test',
            environment=simulator,
            task_manager=task_manager,
            callbacks=callbacks,
        )

        callbacks.before_run()

        self._skip_idle(False)

        observation = self._get_observation()
        info = dict(id=id_)
        return observation, info

    def step(
        self,
        action: npt.NDArray[np.int32],
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        _controller = self._require_controller()
        self._episode_step += 1

        task_ids = (action[:_controller.environment.num_satellites]
                    - 1).tolist()
        
        # print("task_ids:", task_ids)

        self._take_actions(task_ids)

        self._last_num_succeeded_tasks = _controller.task_manager.num_succeeded_tasks

        terminated = _controller.task_manager.all_closed
        truncated = _controller.environment.timer.time >= 3600
        self._skip_idle(terminated or truncated)

        observation = (
            null_observation
            if terminated or truncated else self._get_observation()
        )

        return (
            observation,
            0.0,
            terminated,
            truncated,
            self.info,
        )

    def _take_actions(self, task_ids: npt.NDArray[np.int32]) -> None:
        _controller = self._require_controller()
        tasks = _controller.task_manager.ongoing_tasks
        constellation = _controller.environment.get_constellation()
        # print("Taking actions:", task_ids,
        #         "tasks available:", len(tasks),)
        target_locations = [
            None if task_id == -1 or task_id >= len(tasks) else tasks[task_id].coordinate # FIXME: a bug
            for task_id in task_ids
        ]

        toggles = [
            (task_id == -1 and satellite.sensor.enabled)
            or (task_id != -1 and not satellite.sensor.enabled)
            for task_id, satellite in zip(task_ids, constellation.sort())
        ]

        actions = Actions(
            Action(toggle=toggle, target_location=target_location)
            for toggle, target_location in zip(toggles, target_locations)
        )

        _controller.step(actions, task_ids)
