import random

import todd
from tqdm import tqdm, trange

from constellation import CONSTELLATIONS_ROOT, SATELLITES_ROOT, TASKSETS_ROOT
from constellation.data import (
    Constellation,
    Satellite,
    TaskSet,
)


def generate(split: str, n: int) -> None:
    todd.logger.info("Generating %s", split)

    satellites_root = SATELLITES_ROOT / split
    satellites: list[Satellite] = []
    for f in tqdm(satellites_root.iterdir()):
        assert f.suffix == '.json'
        try:
            constellation = Constellation.load(str(f))
        except Exception as e:
            todd.logger.exception(e)
            continue
        assert len(constellation) == 1
        satellites.extend(constellation.values())

    todd.logger.info("Loaded %d satellites", len(satellites))

    constellations_root = CONSTELLATIONS_ROOT / split
    tasks_root = TASKSETS_ROOT / split
    for i in trange(n):
        constellation_path = constellations_root / f'{i // 1000:02}'
        constellation_path.mkdir(parents=True, exist_ok=True)
        Constellation.sample(satellites, random.randint(1, 50))\
            .dump(str(constellation_path / f'{i:05}.json'))

        taskset_path = tasks_root / f'{i // 1000:02}'
        taskset_path.mkdir(parents=True, exist_ok=True)
        TaskSet.sample(random.randint(50, 300))\
            .dump(str(taskset_path / f'{i:05}.json'))


def main() -> None:
    generate('train', 100_000)
    generate('val_seen', 500)
    generate('val_unseen', 500)
    generate('test', 1_000)


if __name__ == '__main__':
    main()
