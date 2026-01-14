__all__ = [
    'Controller',
]

from pathlib import Path
from typing import TYPE_CHECKING

from todd.runners import Memo
from tqdm import trange

from .algorithms import BaseAlgorithm, OptimalAlgorithm
from .constants import MAX_TIME_STEP
from .data import Actions, Constellation, TaskSet
from .environments import BaseEnvironment
from .task_managers import TaskManager

if TYPE_CHECKING:
    from .callbacks import ComposedCallback


class Controller:

    def __init__(
        self,
        name: str,
        *args,
        environment: BaseEnvironment,
        task_manager: TaskManager,
        callbacks: 'ComposedCallback',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._name = name
        self._environment = environment
        self._task_manager = task_manager

        self._callbacks = callbacks
        self._memo: Memo = dict()

        callbacks.bind(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def environment(self) -> BaseEnvironment:
        return self._environment

    @property
    def task_manager(self) -> TaskManager:
        return self._task_manager

    @property
    def callbacks(self) -> 'ComposedCallback':
        return self._callbacks

    @property
    def memo(self) -> Memo:
        return self._memo

    def step(self, actions: Actions, assignment: list[int]) -> None:
        self._memo['actions'] = actions
        self._memo['assignment'] = assignment

        self._callbacks.before_step()

        is_visible = self._environment.is_visible(self._task_manager.all_tasks)
        self._memo['is_visible'] = is_visible

        self._task_manager.record(is_visible)

        self._environment.take_actions(actions)

        self._callbacks.after_step()

        # TODO: why not step timer in environment.step?
        self._environment.timer.step()
        self._environment.step()

    def run(
        self,
        algorithm: BaseAlgorithm,
        *,
        progress_bar: bool = True,
    ) -> None:
        self._memo['algorithm'] = algorithm
        self._callbacks.before_run()

        for _ in trange(MAX_TIME_STEP, disable=not progress_bar):
            if self._callbacks.should_break():
                break

            actions, assignment = algorithm.step(
                self._task_manager.ongoing_tasks,
                self._environment.get_constellation(),
                self._environment.get_earth_rotation(),
            )
            self.step(actions, assignment)

        self._callbacks.after_run()


def main() -> None:
    import pathlib

    from .callbacks import ComposedCallback
    from .environments import BasiliskEnvironment
    from .evaluators import (
        CompletionRateEvaluator,
        PowerUsageEvaluator,
        TurnAroundTimeEvaluator,
    )
    from .loggers import TrajectoryLogger

    tasks = TaskSet.load('data/tasksets/test/00/00000.json')
    constellation = Constellation.load(
        'data/constellations/test/00/00003.json',
        # 'data/constellations/test/00/00000.json',
    )
    environment = BasiliskEnvironment(
        constellation=constellation,
        all_tasks=tasks,
    )
    task_manager = TaskManager(timer=environment.timer, tasks=tasks)
    algorithm = OptimalAlgorithm(timer=environment.timer)
    algorithm.prepare(environment, task_manager)

    evaluators = [
        CompletionRateEvaluator(),
        TurnAroundTimeEvaluator(),
        PowerUsageEvaluator(),
    ]

    work_dir = Path('work_dirs') / 'new_exp_0'
    work_dir.mkdir(parents=True, exist_ok=True)
    loggers = [
        TrajectoryLogger(work_dir=work_dir),
    ]

    callbacks = ComposedCallback(callbacks=[*evaluators, *loggers])

    controller = Controller(
        pathlib.Path(__file__).stem,
        environment=environment,
        task_manager=task_manager,
        callbacks=callbacks,
    )

    controller.run(algorithm)
    print(controller.memo['metrics'])


if __name__ == "__main__":
    main()
