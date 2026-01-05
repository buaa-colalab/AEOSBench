import random

from todd.patches.py_ import json_dump, json_load
from tqdm import tqdm, trange

from constellation import CONSTELLATIONS_ROOT, ORBITS_ROOT
from constellation.data import ConstellationDict, OrbitDicts


def patch_constellation(
    constellation: ConstellationDict,
    orbits: OrbitDicts,
) -> None:
    orbits = random.sample(orbits, len(constellation['orbits']))
    for i, orbit in enumerate(orbits):
        orbit = orbit.copy()
        orbit['id'] = constellation['orbits'][i]['id']
        constellation['orbits'][i] = orbit


def patch(split: str, n: int) -> None:
    orbit_dicts = [json_load(str(f)) for f in tqdm(ORBITS_ROOT.iterdir())]

    constellations_root = CONSTELLATIONS_ROOT / split
    for i in trange(n):
        constellation_path = (
            constellations_root / f'{i // 1000:02}' / f'{i:05}.json'
        )
        constellation_dict = json_load(str(constellation_path))
        patch_constellation(constellation_dict, orbit_dicts)
        json_dump(constellation_dict, str(constellation_path))


def main() -> None:
    patch('test', 1_000)


if __name__ == '__main__':
    main()
