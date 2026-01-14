import random

import todd
from tqdm import tqdm, trange

from constellation import CONSTELLATIONS_ROOT, SATELLITES_ROOT, TASKSETS_ROOT
from constellation.data import Constellation, Satellite, Satellites, TaskSet


def load_satellites(split: str) -> Satellites:
    satellites_root = SATELLITES_ROOT / split
    satellites: list[Satellite] = []
    for f in tqdm(satellites_root.iterdir()):
        assert f.suffix == '.json'
        constellation = Constellation.load(str(f))
        assert len(constellation) == 1
        satellites.extend(constellation.values())

    todd.logger.info("Loaded %d satellites", len(satellites))
    return satellites


def generate_constellations_and_tasksets(split: str, n: int) -> None:
    satellites = load_satellites(split)

    constellations_root = CONSTELLATIONS_ROOT / split
    tasks_root = TASKSETS_ROOT / split
    for i in trange(n):
        constellation_path = (
            constellations_root / f'{i // 1000:02}' / f'{i:05}.json'
        )
        if not constellation_path.exists():
            constellation_path.parent.mkdir(parents=True, exist_ok=True)
            constellation = Constellation.sample(
                satellites,
                random.randint(1, 50),
            )
            constellation.dump(str(constellation_path))

        taskset_path = tasks_root / f'{i // 1000:02}' / f'{i:05}.json'
        if not taskset_path.exists():
            taskset_path.parent.mkdir(parents=True, exist_ok=True)
            taskset = TaskSet.sample(random.randint(50, 300))
            taskset.dump(str(taskset_path))


def main() -> None:
    generate_constellations_and_tasksets('train', 100_000)
    generate_constellations_and_tasksets('val_seen', 500)
    generate_constellations_and_tasksets('val_unseen', 500)
    generate_constellations_and_tasksets('test', 1_000)


if __name__ == '__main__':
    main()
