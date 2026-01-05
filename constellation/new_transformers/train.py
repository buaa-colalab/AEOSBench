import argparse
import importlib
import pathlib

from todd.configs import PyConfig
from todd.patches.py_ import DictAction
from todd.utils import init_seed
from torch import nn
from todd.runners import BaseRunner
from todd.registries import RunnerRegistry

from .utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--config-options', action=DictAction, default=dict())
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--load-model-from', nargs='+', default=[])
    parser.add_argument('--load-from')
    parser.add_argument('--auto-resume', action='store_true')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = PyConfig.load(args.config, **args.config_options)
    config.override(args.override)
    init_seed(args.seed)

    for custom_import in config.get('custom_imports', []):
        importlib.import_module(custom_import)

    trainer: BaseRunner[nn.Module] = RunnerRegistry.build(
        config.trainer,
        name=args.name,
        load_from=args.load_from,
        auto_resume=args.auto_resume,
        autocast=args.autocast,
    )
    log(trainer, args, config)
    if args.load_model_from:
        trainer.strategy.load_model_from(args.load_model_from, strict=False)
    trainer.run()


if __name__ == '__main__':
    main()
