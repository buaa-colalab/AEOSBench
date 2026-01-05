__all__ = [
    "ConstellationRegistry",
    "ConstellationDatasetRegistry",
    "ConstellationModelRegistry",
    "ConstellationRunnerRegistry",
]

from todd import Registry
from todd.registries import DatasetRegistry, ModelRegistry, RunnerRegistry


class ConstellationRegistry(Registry):
    pass


class ConstellationDatasetRegistry(ConstellationRegistry, DatasetRegistry):
    pass


class ConstellationModelRegistry(ConstellationRegistry, ModelRegistry):
    pass


class ConstellationRunnerRegistry(ConstellationRegistry, RunnerRegistry):
    pass
