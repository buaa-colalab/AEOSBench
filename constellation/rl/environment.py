__all__ = [
    'Environment',
]

from functools import partial
import random
from typing import Any, Literal, TypedDict, cast, overload
from typing_extensions import Self

import einops
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from constellation import STATISTICS_PATH
from constellation.new_transformers import Statistics
from constellation.new_transformers import SATELLITE_DIM, TASK_DIM
from stable_baselines3.common.vec_env import SubprocVecEnv
from todd.patches.py_ import json_load
from constellation.new_transformers.model import GLOBALS

from constellation import (
    ANNOTATIONS_ROOT,
    CONSTELLATIONS_ROOT,
    DATA_ROOT,
    MAX_TIME_STEP,
    TASKSETS_ROOT,
    TIMESTAMP,
)
from constellation.data import (
    Action,
    Actions,
    Constellation,
    SensorType,
    Task,
    TaskSet,
)
from constellation.environments import BasiliskEnvironment
from constellation.evaluators import (
    BaseEvaluator,
    CompletionRateEvaluator,
    PCompletionRateEvaluator,
    WCompletionRateEvaluator,
    WPCompletionRateEvaluator,
    TurnAroundTimeEvaluator,
    PowerUsageEvaluator,
)
from constellation import TaskManager

MAX_NUM_SATELLITES = 51
MAX_NUM_TASKS = 302  # TODO check 301


class Observation(TypedDict):
    num_satellites: int
    num_tasks: int
    time_step: int
    constellation_sensor_type: npt.NDArray[np.uint8]
    constellation_sensor_enabled: npt.NDArray[np.uint8]
    constellation_data: npt.NDArray[np.float32]
    tasks_sensor_type: npt.NDArray[np.uint8]
    tasks_data: npt.NDArray[np.float32]


null_observation = Observation(
    num_satellites=1,
    num_tasks=1,
    time_step=1,
    constellation_sensor_type=np.zeros(MAX_NUM_SATELLITES, np.uint8),
    constellation_sensor_enabled=np.zeros(MAX_NUM_SATELLITES, np.uint8),
    constellation_data=np.zeros(
        (MAX_NUM_SATELLITES, SATELLITE_DIM),
        np.float32,
    ),
    tasks_sensor_type=np.zeros(MAX_NUM_TASKS, np.uint8),
    tasks_data=np.zeros((MAX_NUM_TASKS, TASK_DIM), np.float32),
)


class Padding:

    @overload
    def _pad(
        self,
        s: npt.NDArray[np.uint8],
        t: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        pass

    @overload
    def _pad(
        self,
        s: npt.NDArray[np.float32],
        t: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        pass

    def _pad(
        self,
        s: npt.NDArray[np.uint8] | npt.NDArray[np.float32],
        t: npt.NDArray[np.uint8] | npt.NDArray[np.float32],
    ) -> npt.NDArray[np.uint8] | npt.NDArray[np.float32]:
        assert s.shape[0] <= t.shape[0]
        t[:s.shape[0]] = s
        return t

    def __call__(self, observation: Observation) -> Observation:
        k: Literal[
            'constellation_sensor_type',
            'constellation_sensor_enabled',
            'constellation_data',
            'tasks_sensor_type',
            'tasks_data',
        ]

        for k in [  # type: ignore[assignment]
            'constellation_sensor_type',
            'constellation_sensor_enabled',
            'constellation_data',
            'tasks_sensor_type',
            'tasks_data',
        ]:
            s = observation[k]
            t = null_observation[k].copy()
            assert s.shape[0] <= t.shape[0]
            t[:s.shape[0]] = s
            observation[k] = t

        return observation


class Environment(gym.Env[Observation, npt.NDArray[np.uint16]]):

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
            str(ANNOTATIONS_ROOT / f'{split}.tiny.json'),
        )
        self._statistics: Statistics = torch.load(
            STATISTICS_PATH,
            weights_only=False,
        )

        self._padding = Padding()

    @property
    def info(self) -> dict[str, float]:
        return {
            "metrics": [
                e.evaluate(self._task_manager) for e in self._evaluators
            ]
        }

    def evaluator_log_progress(self, actions: list[int]) -> None:
        for evaluator in self._evaluators:
            evaluator.log_progress(
                self._task_manager,
                actions,
                self._simulator,
            )

    def _load_constellation(
        self,
    ) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint8],
        npt.NDArray[np.float32],
    ]:
        constellation = self._simulator.get_constellation()
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
        sensor_type, static_data = self._task_manager.valid_tasks.to_tensor()

        static_data = static_data.clone()
        t = self._simulator.timer.time
        static_data[..., 0] -= t
        static_data[..., 1] -= t

        progress = self._task_manager.progress[self._task_manager.valid_labels]
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

        tasks_sensor_type, tasks_data = self._load_tasks()

        observation = Observation(
            num_satellites=self._simulator.num_satellites,
            num_tasks=self._task_manager.num_valid_tasks,
            time_step=self._simulator.timer.time,
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

    def _take_actions(self, task_ids: list[int]) -> None:
        # TODO: fix toggle
        # constellation = self._simulator.get_constellation()

        # constellation_sensor_enabled = [
        #     satellite.sensor.enabled for satellite in constellation.sort()
        # ]
        # toggles = [((task_id == -1) == enabled) for enabled, task_id in
        #            zip(constellation_sensor_enabled, task_ids)]
        # 0 0 -> 1
        # 0 1 -> 0
        # 1 0 -> 0
        # 1 1 -> 1
        # toggles = [self._simulator.timer.time == 0 for _ in task_ids]

        tasks = self._task_manager.valid_tasks
        target_locations = [
            None if task_id == -1 else tasks[task_id].coordinate
            for task_id in task_ids
        ]

        if hasattr(self, '_pred_mask'):
            for constellation_id, task_id in enumerate(task_ids):
                if task_id != -1 and self._pred_mask[constellation_id,
                                                     task_id]:
                    task_ids[constellation_id] = -1
            delattr(self, '_pred_mask')

        toggles = [(task_id == -1 and satellite.sensor.enabled)
                   or (task_id != -1 and not satellite.sensor.enabled)
                   for task_id, satellite in
                   zip(task_ids,
                       self._simulator.get_constellation().sort())]

        actions = Actions(
            Action(toggle=toggle, target_location=target_location)
            for toggle, target_location in zip(toggles, target_locations)
        )
        self._simulator.take_actions(actions)
        self._simulator.timer.step()

        self.evaluator_log_progress(task_ids)

    def _skip_idle(self) -> None:
        while self._task_manager.is_idle:
            self._take_actions([-1] * self._simulator.num_satellites)

    def _get_annotation(self) -> int:
        return random.choice(self._annotations)

    def reset(self, *args, **kwargs) -> tuple[Observation, dict[str, Any]]:
        super().reset(*args, **kwargs)

        id_ = self._get_annotation()

        constellation_path = (
            CONSTELLATIONS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        constellation = Constellation.load(str(constellation_path))

        taskset_path = (
            TASKSETS_ROOT / self._split / f'{id_ // 1000:02}'
            / f'{id_:05}.json'
        )
        tasks: TaskSet[Task] = TaskSet.load(str(taskset_path))

        simulator = BasiliskEnvironment(
            start_time=0,
            end_time=MAX_TIME_STEP,
            standard_time_init=TIMESTAMP,
            constellation=constellation,
            all_tasks=tasks,
        )
        self._simulator = simulator

        task_manager = TaskManager(
            timer=simulator.timer,
            tasks=tasks,
            num_satellites=simulator.num_satellites,
        )
        self._task_manager = task_manager

        self._evaluators: list[BaseEvaluator] = [
            CompletionRateEvaluator(),
            PCompletionRateEvaluator(),
            WCompletionRateEvaluator(),
            WPCompletionRateEvaluator(),
            TurnAroundTimeEvaluator(),
            PowerUsageEvaluator()
        ]

        self._skip_idle()

        observation = self._get_observation()
        info = dict(id=id_)
        return observation, info

    def step(
        self,
        action: npt.NDArray[np.int32],
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        task_ids = action[:self._simulator.num_satellites] - 1
        self._take_actions(task_ids.tolist())

        tasks = self._task_manager.tasks
        is_visible = self._simulator.is_visible(tasks)
        completed = self._task_manager.record(is_visible)

        # TODO: design better reward
        reward = 0.
        reward += completed.numel() * 100

        is_visible[:, ~self._task_manager.valid_labels] = False
        num_visible_satellites = is_visible.any(1).sum().item()
        reward += 2 * num_visible_satellites - is_visible.shape[0]
        reward /= 10

        self._skip_idle()

        # NOTE: test after skipping idle
        terminated = self._task_manager.is_finished
        truncated = self._simulator.timer.time >= self._simulator.end_time

        observation = (
            null_observation
            if terminated or truncated else self._get_observation()
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            self.info,
        )
