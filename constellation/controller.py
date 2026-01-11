__all__ = [
    'Controller',
]

import itertools
from pathlib import Path
import pathlib
from typing import Iterable, Union, List

import torch
from tqdm import tqdm, trange

from .new_transformers.dataset import Actions

from .constants import MAX_TIME_STEP

from .callbacks.memo import Memo, get_memo

from .algorithms import BaseAlgorithm, OptimalAlgorithm
from .callbacks.base import BaseCallback
from .data import TaskSet, Constellation
from .environments import BaseEnvironment
from .evaluators import BaseEvaluator, CompletionRateEvaluator, PCompletionRateEvaluator, WCompletionRateEvaluator, WPCompletionRateEvaluator, TurnAroundTimeEvaluator, PowerEvaluator
from .loggers import BaseLogger, PthLogger, VisualizationLogger
from .task_managers import TaskManager


class Controller:

    def __init__(
        self,
        environment: BaseEnvironment,
        task_manager: TaskManager,
        callbacks: List[BaseCallback] = None,
    ) -> None:
        self._environment = environment
        self._task_manager = task_manager

        self._callbacks = callbacks

        for callback in self._callbacks:
            callback.on_init(controller=self)

    @property
    def environment(self) -> BaseEnvironment:
        return self._environment

    @property
    def task_manager(self) -> TaskManager:
        return self._task_manager

    @property
    def callbacks(self) -> List[BaseCallback]:
        return self._callbacks

    def run(
        self,
        ordinal: int,
        algorithm: BaseAlgorithm,
        max_time_step: int = MAX_TIME_STEP,
        progress_bar: bool = True,
    ) -> Memo:

        memo = Memo()

        for _ in trange(max_time_step, disable=not progress_bar):

            actions, dispatch_ids = algorithm.step(
                tasks=self._task_manager.ongoing_tasks,
                constellation=self._environment.get_constellation(),
                earth_rotation=self._environment.get_earth_rotation(),
            )

            self.step(actions, dispatch_ids)

            if self._task_manager.all_closed:
                print('All tasks are closed.')
                break

        for callback in self._callbacks:  # TODO: extract & combine
            callback.on_run_end(memo=memo, save_name=str(ordinal))
        metrics = get_memo(memo, 'metrics')

        return metrics  # TODO: return memo

    def step(
        self,
        actions,
        dispatch_ids,
    ):
        for callback in self._callbacks:
            callback.on_step_begin()

        is_visible = self._environment.is_visible(self._task_manager.all_tasks)
        self._task_manager.record(is_visible)

        self._environment.take_actions(actions)

        for callback in self._callbacks:
            callback.on_step_end(dispatch_id=dispatch_ids)

        self._environment._timer.step()
        # print(self._environment._timer.time)
        # print("sat:" + str(self._environment.num_satellites))
        # print("tasks:" + str(self._task_manager.num_ongoing_tasks))
        self._environment.step()


def main() -> None:
    from .data import Task
    from .environments import BasiliskEnvironment

    tasks: TaskSet = TaskSet.load('data/tasksets/test/00/00000.json')
    constellation = Constellation.load(
        'data/constellations/test/00/00000.json'
    )
    time_string = '20190101000000'
    environment = BasiliskEnvironment(
        start_time=0,
        standard_time_init=time_string,
        constellation=constellation,
        all_tasks=tasks,
    )
    task_manager = TaskManager(timer=environment.timer, tasks=tasks)
    algorithm = OptimalAlgorithm(timer=environment.timer)
    algorithm.prepare(environment, task_manager)

    e0 = CompletionRateEvaluator()
    e1 = PCompletionRateEvaluator()
    e2 = WCompletionRateEvaluator()
    e3 = WPCompletionRateEvaluator()
    et = TurnAroundTimeEvaluator()
    ep = PowerEvaluator()
    evaluators = [e0, e1, e2, e3, et, ep]

    work_dir = Path('work_dirs') / pathlib.Path("new_exp")
    l0 = VisualizationLogger(work_dir=work_dir)
    l1 = PthLogger(work_dir=work_dir)
    loggers = [l0, l1]

    callbacks = [*evaluators, *loggers]

    controller = Controller(
        environment=environment,
        task_manager=task_manager,
        callbacks=callbacks,
    )

    print(controller.run(0, algorithm))


if __name__ == "__main__":
    main()
