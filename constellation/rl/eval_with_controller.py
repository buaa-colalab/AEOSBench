import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm

from constellation.algorithms import OptimalAlgorithm
# from ..callbacks.memo import get_memo
from ..task_managers import TaskManager
from ..environments.base import BaseEnvironment
from constellation.rl.controller_environment import ControllerEnvironment
from constellation.callbacks.memo import Memo, get_memo


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the optimal algorithm using ControllerEnvironment'
    )
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--save-name', type=str, default='optimal_eval')
    parser.add_argument('--show-progress', action='store_true')
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--world-size', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    env = ControllerEnvironment.build(
        world_size=args.world_size,
        split=args.split,
    )

    metrics_list = []

    episode_iterator = range(args.num_episodes)
    if args.show_progress:
        episode_iterator = tqdm.tqdm(episode_iterator, desc="Episodes")

    for episode in episode_iterator:
        observation, info = env.reset()
        task_manager: TaskManager = info['task_manager']
        environment: BaseEnvironment = info['environment']

        algorithm = OptimalAlgorithm(timer=environment.timer)
        algorithm.prepare(environment, task_manager)

        done = False
        while not done:
            _, dispatch_ids = algorithm.step(
                tasks=task_manager.ongoing_tasks,
                constellation=environment.get_constellation(),
                earth_rotation=environment.get_earth_rotation(),
            )

            action = np.array(dispatch_ids, dtype=np.int32)

            observation, reward, terminated, truncated, memo = env.step(action)

            done = terminated or truncated

        metrics_list.append(get_memo(memo, 'metrics'))

    print(metrics_list)
    metrics_mean = {
        'CR': np.mean([metrics['CR'] for metrics in metrics_list]),
        'PCR': np.mean([metrics['PCR'] for metrics in metrics_list]),
        'WCR': np.mean([metrics['WCR'] for metrics in metrics_list]),
        'WPCR': np.mean([metrics['WPCR'] for metrics in metrics_list]),
        'TT': np.mean([metrics['TT'] for metrics in metrics_list]),
        'PC': np.mean([metrics['PC'] for metrics in metrics_list])
    }

    print(
        f"Evaluation results on {args.split} split ({args.num_episodes} episodes):"
    )
    print(f"Completion Rate (CR): {metrics_mean['CR']:.4f}")
    print(f"Priority Completion Rate (PCR): {metrics_mean['PCR']:.4f}")
    print(f"Weighted Completion Rate (WCR): {metrics_mean['WCR']:.4f}")
    print(
        f"Weighted Priority Completion Rate (WPCR): {metrics_mean['WPCR']:.4f}"
    )
    print(f"Turn Around Time (TT): {metrics_mean['TT']:.4f}")
    print(f"Power Consumption (PC): {metrics_mean['PC']:.4f}")

    return metrics_mean


if __name__ == "__main__":
    main()
