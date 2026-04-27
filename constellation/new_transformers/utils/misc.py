__all__ = [
    'log',
]

import argparse
import pathlib
import sys
from typing import cast

from todd.configs import PyConfig
from todd.patches.torch import get_rank
from todd.runners import BaseRunner
from torch import nn


def log(
    runner: BaseRunner[nn.Module],
    args: argparse.Namespace,
    config: PyConfig,
) -> None:
    if get_rank() != 0:
        return

    runner.logger.info("Command\n" + ' '.join(sys.argv))
    runner.logger.info(f"Args\n{vars(args)}")
    runner.logger.info(f"Config\n{config.dumps()}")

    if 'config' in args:
        config_name = cast(pathlib.Path, args.config).name
        PyConfig(config).dump(runner.work_dir / config_name)
