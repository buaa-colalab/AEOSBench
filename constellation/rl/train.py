import argparse
import importlib
import pathlib
from typing import cast

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import todd
from todd.configs import PyConfig
from todd.patches.py_ import DictAction, descendant_classes
from todd.utils import init_seed
from todd.bases.registries import Item

from .environment import Environment
from .policy import Policy


class CallbackRegistry(todd.Registry):
    pass


for c in descendant_classes(BaseCallback):
    CallbackRegistry.register_()(cast(Item, c))


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


'''
CUDA_VISIBLE_DEVICES=-1 python -m rl.train \
    rl_loaded \
    rl/config.py \
    --load-model-from '/data/wlt/projects/Constellation/work_dirs/vit_b/checkpoints/iter_20000/model.pth'
'''


def main() -> None:
    args = parse_args()
    config = PyConfig.load(args.config, **args.config_options)
    config.override(args.override)
    init_seed(args.seed)

    for custom_import in config.get('custom_imports', []):
        importlib.import_module(custom_import)

    environment = Environment.build(config.environment.world_size)

    algorithm = PPO(
        Policy,
        environment,
        policy_kwargs=dict(load_model_from=args.load_model_from),
        tensorboard_log='./rl_tensorboard/',
        seed=args.seed,
        **config.algorithm,
    )

    algorithm.learn(
        **config.learn,
        callback=[
            CallbackRegistry.build(callback) for callback in config.callbacks
        ],
    )
    algorithm.save(args.name)

    environment.close()


if __name__ == '__main__':
    main()
