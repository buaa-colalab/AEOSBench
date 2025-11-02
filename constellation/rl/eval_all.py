import argparse
import atexit
import signal
import importlib
import os
import pathlib
import pickle
from functools import partial
from itertools import count
from typing import Any

import numpy as np
import pandas as pd
import numpy.typing as npt
import todd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from todd.configs import PyConfig
from todd.patches.py_ import DictAction, json_dump
from todd.utils import init_seed

from .environment import Environment, Observation, null_observation
from .controller_environment import ControllerEnvironment
from .policy import Policy
from constellation.loggers import BaseLogger, PthLogger, VisualizationLogger
from constellation.new_transformers.model import GLOBALS

COMPLETION_RATE_THRESHOLD = 0.01


class EvalEnvironment(ControllerEnvironment):

    @classmethod
    def build(
        cls,
        world_size: int,
        gen_trajectory_dir: pathlib.Path,
        *args,
        **kwargs,
    ) -> SubprocVecEnv:
        assert world_size > 0, "world_size must be greater than 0"
        return SubprocVecEnv([
            partial(
                cls,
                *args,
                world_size=int(os.environ['WORLD_SIZE']) * world_size,
                rank=int(os.environ['RANK']) * world_size + i,
                gen_trajectory_dir=gen_trajectory_dir,
                **kwargs,
            ) for i in range(world_size)
        ])

    def __init__(
        self,
        *args,
        world_size: int,
        rank: int,
        retry_from: pathlib.Path | None = None,
        gen_trajectory_dir: pathlib.Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._world_size = world_size
        self._rank = rank
        self._counter = -1
        self._gen_trajectory_dir = gen_trajectory_dir

        if retry_from is not None:
            df = pd.read_csv(
                retry_from,
                names=['id', 'completion_rate'],
                index_col='id',
            )
            completion_rates: dict[int, float] = \
                df['completion_rate'].to_dict()
            self._annotations = [
                annotation for annotation in self._annotations if
                completion_rates.get(annotation, 0) < COMPLETION_RATE_THRESHOLD
            ]

    @property
    def _index(self) -> int:
        return self._counter * self._world_size + self._rank

    @property
    def all_done(self) -> bool:
        return self._index >= len(self._annotations)

    def _get_annotation(self) -> int:
        return self._annotations[self._index]

    def reset(self, *args, **kwargs) -> tuple[Observation, dict[str, Any]]:
        if self._counter != -1 and not self.all_done:
            id_ = self._get_annotation()
            save_dir = self._gen_trajectory_dir / f'{id_ // 1000:02d}'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f'{id_:05d}.pth'
            # self._logger.pth_dump(save_path)

        self._counter += 1

        if self.all_done:
            return null_observation, dict(all_done=True)

        obs, info = super().reset(*args, **kwargs)
        print(info)
        # with open("./test.pkl", "wb") as f:
        #     pickle.dump((obs, info), f)
        # self._logger = Logger(
        #     task_manager=self._task_manager,
        #     constellation=None,
        #     work_dir=None,
        # )
        return obs, info

    def step(
        self,
        action: npt.NDArray[np.uint16],
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if isinstance(action, tuple):  # TODO: one action
            action, pred_mask = action
            self._pred_mask = pred_mask

        if self.all_done:
            return null_observation, 0.0, False, False, dict(all_done=True)

        if self._controller.task_manager.progress.any(
        ) and self._controller.environment.timer.time % 50 == 0 and self._controller.environment.timer.time <= 1800:
            todd.logger.info(
                "env_rank %s sim_step %d progress_sum %d finished_num %d",
                self._rank,
                self._controller.environment.timer.time,
                self._controller.task_manager.progress.sum(),
                self._controller.task_manager.num_succeeded_tasks,
            )

        # self._logger.add_time_csv(
        #     constellation=self._simulator.get_constellation(),
        #     task_id_list=(action[:self._simulator.num_satellites]
        #                   - 1).tolist(),
        #     is_visible=self._simulator.is_visible(self._task_manager.tasks),
        # )

        observation, reward, terminated, truncated, info = (
            super().step(action)
        )

        id_ = self._get_annotation()
        info.update(id=id_)

        return observation, reward, terminated, truncated, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('name')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--config-options', action=DictAction, default=dict())
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--load-model-from', nargs='+', default=[])
    parser.add_argument('--load-ppo-from', type=pathlib.Path, default=None)
    parser.add_argument('--load-from')
    parser.add_argument('--auto-resume', action='store_true')
    parser.add_argument('--retry-from', type=pathlib.Path, default=None)
    args = parser.parse_args()
    return args


'''
CUDA_VISIBLE_DEVICES=0,3,4,5,6,7 auto_torchrun -m rl.eval_all \
    rl_loaded_eval \
    rl/config_eval.py \
    --load-model-from './work_dirs/model610000.pth'
'''


def main() -> None:
    args = parse_args()
    config = PyConfig.load(args.config, **args.config_options)
    config.override(args.override)
    init_seed(args.seed)

    for custom_import in config.get('custom_imports', []):
        importlib.import_module(custom_import)

    work_dir = pathlib.Path('work_dirs') / f'rl_eval_{args.name}'
    work_dir.mkdir(parents=True, exist_ok=True)

    gen_trajectory_dir = work_dir / config.environment.split
    gen_trajectory_dir.mkdir(parents=True, exist_ok=True)

    environment = EvalEnvironment.build(
        retry_from=args.retry_from,
        gen_trajectory_dir=gen_trajectory_dir,
        **config.environment,
    )
    atexit.register(environment.close)
    signal.signal(signal.SIGINT, lambda s, f: environment.close())
    signal.signal(signal.SIGTERM, lambda s, f: environment.close())

    device = torch.device(int(os.environ['RANK']) % torch.cuda.device_count())
    torch.cuda.set_device(device)

    if args.load_model_from != []:
        algorithm = PPO(
            Policy,
            environment,
            policy_kwargs=dict(load_model_from=args.load_model_from),
            tensorboard_log=str(work_dir),
            seed=args.seed,
            device=device,
            **config.algorithm,
        )
    else:
        assert args.load_ppo_from is not None
        algorithm = PPO.load(
            path=args.load_ppo_from,
            env=environment,
            device=device,
        )

    observations = environment.reset()
    for i in count():
        if i % config.log_interval == 0:
            todd.logger.info("rank %s step %d", os.environ['RANK'], i)

        actions, _ = algorithm.predict(
            observations,
            deterministic=True,  # type: ignore[arg-type]
        )
        if 'pred_mask' in GLOBALS:
            actions = list(zip(actions, GLOBALS.pop('pred_mask').cpu()))
        observations, _, dones, infos = environment.step(actions)

        for done, info in zip(dones, infos):
            if done and not info.get('all_done', False):
                id_ = info['id']
                metrics = info['metrics']

                todd.logger.info(
                    f"rank %s step %d {id_=}\n{metrics=}",
                    os.environ['RANK'],
                    i,
                )

                json_path = gen_trajectory_dir / f'{id_ // 1000:02d}' / f'{id_:05d}.json'
                json_dump(metrics, str(json_path))

        if all(info.get('all_done', False) for info in infos):
            todd.logger.info(
                "rank %s step %d all done",
                os.environ['RANK'],
                i,
            )
            break
    environment.close()


if __name__ == '__main__':
    main()
