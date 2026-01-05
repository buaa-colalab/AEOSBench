import argparse
import importlib
import logging
import os
import pathlib
import time
from abc import abstractmethod
from typing import Generator, Iterable

import todd
import todd.runners.callbacks as trc
import torch.distributed
from todd.configs import PyConfig
from todd.patches.py_ import DictAction
from todd.patches.torch import get_rank, get_world_size
from torch import nn

from todd.registries import RunnerRegistry
from todd.runners import BaseRunner
from .utils import log


class Monitor:

    def __init__(self, *args, root: pathlib.Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._root = root

    def model(self, name: str) -> pathlib.Path:
        return self._root / name / 'model.pth'

    @abstractmethod
    def __iter__(self) -> Generator[str, None, None]:
        pass


class MasterMonitor(Monitor):

    def __init__(
        self,
        *args,
        logger: logging.Logger,
        whitelist: Iterable[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._logger = logger
        self._whitelist = None if whitelist is None else set(whitelist)
        self._blacklist: set[str] = set()

    def __iter__(self) -> Generator[str, None, None]:
        while True:
            for name in self.names:
                self._blacklist.add(name)
                model = self.model(name)
                if not model.exists():
                    self._sleep(30)
                assert model.exists(), model
                self._send(name)
                yield name
            if not self.active:
                break
            self._sleep(600)
        self._send(None)

    def time_(self, name: str) -> float:
        return self.model(name).stat().st_ctime

    def _send(self, object_: str | int | None) -> None:
        if get_world_size() > 1:
            torch.distributed.broadcast_object_list([object_])

    def _sleep(self, s: int) -> None:
        if todd.Store.DRY_RUN:
            s = 10
        self._logger.info(f"Waiting {s} seconds for new models")
        self._send(s)
        time.sleep(s)

    @property
    def active(self) -> bool:
        return (
            self._whitelist is None
            or len(self._whitelist - self._blacklist) > 0
        )

    @property
    def names(self) -> list[str]:
        if not self._root.exists():
            return []
        names = set(
            path.name
            for path in self._root.iterdir()
            if path.is_dir() and not path.is_symlink()
        )
        names -= self._blacklist
        if self._whitelist is not None:
            names &= self._whitelist
        return sorted(names, key=self.time_)


class SlaveMonitor(Monitor):

    def _recv(self) -> str | int | None:
        object_list = [None]
        torch.distributed.broadcast_object_list(object_list)
        return object_list[0]

    def __iter__(self) -> Generator[str, None, None]:
        while True:
            object_ = self._recv()
            if object_ is None:
                break
            if isinstance(object_, int):
                time.sleep(object_)
            elif isinstance(object_, str):
                yield object_


def log_name(logger: logging.Logger, name: str) -> None:
    info = f"Validating {name}"
    logger.info("-" * len(info))
    logger.info(info)
    logger.info("-" * len(info))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validation')
    parser.add_argument('name')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--config-options', action=DictAction, default=dict())
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--visual')
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--load-model-from', nargs='+', default=[])
    parser.add_argument('--load-from', nargs='+')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = PyConfig.load(args.config, **args.config_options)
    config.override(args.override)

    for custom_import in config.get('custom_imports', []):
        importlib.import_module(custom_import)

    runner: BaseRunner[nn.Module] = RunnerRegistry.build(
        config.trainer,
        name=args.name,
        autocast=args.autocast,
    )
    log(runner, args, config)

    work_dir = runner.work_dir
    checkpoint_dir, = {
        callback.work_dir
        for callback in runner.callbacks
        if isinstance(callback, trc.CheckpointCallback)
    }

    monitor: Monitor
    if get_rank() == 0:
        summary_writers = [
            callback.summary_writer
            for callback in runner.callbacks
            if isinstance(callback, trc.TensorBoardCallback)
        ]
        logger = runner.logger
        master = MasterMonitor(
            root=checkpoint_dir,
            logger=logger,
            whitelist=args.load_from,
        )
        monitor = master
    else:
        monitor = SlaveMonitor(root=checkpoint_dir)

    for name in monitor:
        if get_rank() == 0:
            log_name(logger, name)
        runner = RunnerRegistry.build(
            config.validator,
            name=f'{args.name}_val_{name}',
            visual=args.visual,
            tokenize=args.tokenize,
            autocast=args.autocast,
            work_dir=dict(
                root=work_dir,
                name=os.path.join('val', name),
            ),
        )
        log(runner, args, config)
        runner.strategy.load_model_from(
            args.load_model_from + [monitor.model(name)],
            strict=False,
        )
        memo = runner.run()
        if (
            get_rank() == 0 and 'metrics' in memo and len(summary_writers) > 0  # noqa: E501 pylint: disable=possibly-used-before-assignment
        ):
            tag, global_step = name.rsplit('_', 1)
            time_ = master.time_(name)
            metrics: dict[str, float] = memo['metrics']
            logger.info("Metrics: %s", metrics)
            for k, v in metrics.items():
                for summary_writer in summary_writers:
                    summary_writer.add_scalar(
                        f'val/{tag}/{k}',
                        v,
                        global_step,
                        time_,
                    )


if __name__ == '__main__':
    main()
