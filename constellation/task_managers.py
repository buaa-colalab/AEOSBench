__all__ = [
    'TaskManager',
]

from typing import cast

import torch

from .data import TaskSet
from .environments import Timer

# 0-- unreleased ---+-- ongoing --+-- succeeded --+-- succeeded --->
#                 release       complete          due

# 0-- unreleased ---+--------- ongoing -----------+--- failed ----->
#                 release                         due

# closed = succeed + failed


class TaskManager:

    def __init__(
        self,
        *args,
        timer: Timer,
        tasks: TaskSet,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._timer = timer
        self._tasks = tasks

        self._progress = torch.zeros(
            self.num_all_tasks,
            dtype=torch.uint8,
        )

        self._succeeded_flags = torch.zeros(len(tasks), dtype=torch.bool)

    def _filter_tasks(self, flags: torch.Tensor) -> TaskSet:
        return TaskSet(task for task, flag in zip(self._tasks, flags) if flag)

    @property
    def unreleased_flags(self) -> torch.Tensor:
        return torch.tensor([
            self._timer.time < task.release_time for task in self._tasks
        ])

    @property
    def ongoing_flags(self) -> torch.Tensor:
        return ~self.succeeded_flags & torch.tensor([
            task.release_time <= self._timer.time <= task.due_time
            for task in self._tasks
        ])

    @property
    def succeeded_flags(self) -> torch.Tensor:
        return self._succeeded_flags

    @property
    def failed_flags(self) -> torch.Tensor:
        return ~self.succeeded_flags & torch.tensor([
            task.due_time < self._timer.time for task in self._tasks
        ])

    @property
    def closed_flags(self) -> torch.Tensor:
        return self.succeeded_flags | self.failed_flags

    @property
    def all_tasks(self) -> TaskSet:
        return self._tasks

    @property
    def ongoing_tasks(self) -> TaskSet:
        return self._filter_tasks(self.ongoing_flags)

    @property
    def succeeded_tasks(self) -> TaskSet:
        return self._filter_tasks(self.succeeded_flags)

    @property
    def failed_tasks(self) -> TaskSet:
        return self._filter_tasks(self.failed_flags)

    @property
    def closed_task(self) -> TaskSet:
        return self._filter_tasks(self.closed_flags)

    @property
    def progress(self) -> torch.Tensor:
        return self._progress

    @property
    def all_closed(self) -> bool:
        return cast(bool, self.closed_flags.all())

    @property
    def is_idle(self) -> bool:
        return self.num_ongoing_tasks == 0

    @property
    def num_all_tasks(self) -> int:
        return len(self._tasks)

    @property
    def num_ongoing_tasks(self) -> int:
        return len(self.ongoing_tasks)

    @property
    def num_succeeded_tasks(self) -> int:
        return len(self.succeeded_tasks)

    def record(self, is_visible: torch.Tensor) -> None:
        durations = torch.tensor([task.duration for task in self._tasks])

        is_visible[:, ~self.ongoing_flags] = False
        is_visible = is_visible.any(0)

        self._progress = (self._progress + 1) * is_visible

        self._succeeded_flags |= (self._progress >= durations)
