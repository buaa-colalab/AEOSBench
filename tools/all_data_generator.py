import random

from tqdm import trange

from constellation.data import Taskset

from constellation import TASKSETS_ROOT


def generate_tasks(split: str, n: int, mrptest: bool) -> None:
    split_root = TASKSETS_ROOT / split
    for i in trange(n):
        path = split_root / f'{i // 1000:02}'
        path.mkdir(parents=True, exist_ok=True)
        Taskset.sample(
            random.randint(50, 300) if not mrptest else 36, mrptest
        ).dump(str(path / f'{i:05}.json'))


def main() -> None:
    generate_tasks('train', 100_000, False)
    generate_tasks('val_seen', 500, False)
    generate_tasks('val_unseen', 500, False)
    generate_tasks('test', 1_000, False)
    generate_tasks('mrp_test', 1_000, True)


if __name__ == '__main__':
    main()
