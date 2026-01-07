import argparse

from todd.utils import init_seed

from constellation import TASKSETS_ROOT
from constellation.data import TaskSet

TASKSET_PATH = TASKSETS_ROOT / 'mrp.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=36)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    init_seed(args.seed)

    TASKSETS_ROOT.mkdir(parents=True, exist_ok=True)
    assert not TASKSET_PATH.exists()

    TaskSet.sample_mrp(args.size).dump(str(TASKSET_PATH))


if __name__ == "__main__":
    main()
