__all__ = [
    'Timer',
]


class Timer:

    def __init__(self, time: int) -> None:
        self._time = time
        self._start_time = time

    @property
    def time(self) -> int:
        return self._time

    def step(self) -> None:
        self._time += 1
