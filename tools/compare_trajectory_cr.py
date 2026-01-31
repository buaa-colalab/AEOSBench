import pathlib
from typing import Mapping, NamedTuple

from todd.patches.py_ import json_dump, json_load
from tqdm import trange

from constellation.constants import ANNOTATIONS_ROOT, DATA_ROOT

THRESHOLD = 0.01


class Annotation(NamedTuple):
    epoch: int
    id_: int


Annotations = list[Annotation]


def parse_epoch(trajectories_root: pathlib.Path) -> int:
    suffix = trajectories_root.suffix
    assert suffix.startswith('.')
    return int(suffix.removeprefix('.'))


def generate_annotations(
    trajectories_roots: Mapping[int, pathlib.Path],
    split: str,
    n: int,
) -> None:
    annotations: Annotations = []
    # valid_ids = []
    # best_epochs = []

    for i in trange(n):
        metric_paths = {
            epoch: trajectories_root / f'{split}/{i//1000:02}/{i:05}.json'
            for epoch, trajectories_root in trajectories_roots.items()
        }
        if not all(
            metric_path.exists() for metric_path in metric_paths.values()
        ):
            continue
        metrics = {
            epoch: json_load(str(metric_path))['CR']
            for epoch, metric_path in metric_paths.items()
        }
        best_epoch, best_metric = max(
            metrics.items(),
            key=lambda item: item[1],
        )
        if best_metric - metrics[0] >= THRESHOLD:
            annotations.append(Annotation(epoch=best_epoch, id_=i))

    assert annotations
    epochs, ids = zip(*annotations)
    json_dump(
        dict(epochs=epochs, ids=ids),
        str(ANNOTATIONS_ROOT / f'{split}.json'),
    )


def main() -> None:
    ANNOTATIONS_ROOT.mkdir(parents=True, exist_ok=True)

    trajectories_roots = {
        parse_epoch(trajectory_root): trajectory_root
        for trajectory_root in DATA_ROOT.glob('trajectories.*')
    }
    assert trajectories_roots, "No trajectory roots found"

    generate_annotations(trajectories_roots, 'train', 26_000)
    generate_annotations(trajectories_roots, 'val_seen', 500)
    generate_annotations(trajectories_roots, 'val_unseen', 500)
    generate_annotations(trajectories_roots, 'test', 1_000)


if __name__ == "__main__":
    main()
