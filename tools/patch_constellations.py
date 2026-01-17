import random

from todd.patches.py_ import json_dump, json_load
from tqdm import tqdm, trange

from constellation import CONSTELLATIONS_ROOT, ORBITS_ROOT
from constellation.data import ConstellationDict, OrbitDicts


def patch_constellation(
    constellation_dict: ConstellationDict,
    orbit_dicts: OrbitDicts,
) -> None:
    orbit_dicts = random.sample(orbit_dicts, len(constellation_dict['orbits']))
    for i, orbit in enumerate(orbit_dicts):
        orbit = orbit.copy()
        orbit['id'] = constellation_dict['orbits'][i]['id']
        constellation_dict['orbits'][i] = orbit


def patch_constellations(split: str, n: int) -> None:
    orbit_dicts = [json_load(str(f)) for f in tqdm(ORBITS_ROOT.iterdir())]

    constellations_root = CONSTELLATIONS_ROOT / split
    for i in trange(n):
        constellation_path = (
            constellations_root / f'{i // 1000:02}/{i:05}.json'
        )
        constellation_dict = json_load(str(constellation_path))
        patch_constellation(constellation_dict, orbit_dicts)
        json_dump(constellation_dict, str(constellation_path))


def main() -> None:
    patch_constellations('test', 1_000)


if __name__ == '__main__':
    main()
