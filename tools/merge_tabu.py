import argparse
import pathlib
import todd
from todd.patches.py_ import json_dump, json_load
from constellation import TRAJECTORIES_ROOT, ANNOTATIONS_ROOT
import shutil


def merge(
    source: pathlib.Path,
    target: pathlib.Path,
    split: str,
    i: int,
) -> None:
        source_completion_rate_path = (
            source / split / f'{i // 1000:02}' / f'{i:05}.json'
        )
        source_completion_rate = json_load(str(source_completion_rate_path))

        target_completion_rate_path = (
            target / split / f'{i // 1000:02}' / f'{i:05}.json'
        )
        target_completion_rate = json_load(str(target_completion_rate_path))

        if target_completion_rate >= source_completion_rate:
            return

        todd.logger.info("Merging %s %d", split, i)

        if todd.Store.DRY_RUN:
            return

        shutil.copy2(source_completion_rate_path, target_completion_rate_path)
        shutil.copy2(
            source_completion_rate_path.with_suffix('.pth'),
            target_completion_rate_path.with_suffix('.pth'),
        )
        shutil.copy2(
            source_completion_rate_path.with_suffix('.tabu.json'),
            target_completion_rate_path.with_suffix('.allowed.json'),
        )


def parallel_merge(
    source: pathlib.Path,
    target: pathlib.Path,
    split: str,
) -> None:
    annotations_path = ANNOTATIONS_ROOT / f'{split}.json'
    annotations: list[int] = json_load(str(annotations_path))

    for i in annotations:


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate trajectories')
    parser.add_argument('source', type=pathlib.Path)
    parser.add_argument('target', type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge(args.source, args.target, 'train')
    # merge(args.source, args.target, 'val_seen')
    # merge(args.source, args.target, 'val_unseen')
    # merge(args.source, args.target, 'test')


if __name__ == '__main__':
    main()
