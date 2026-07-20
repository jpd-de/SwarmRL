"""Reusable fixed-length observation sequence construction."""

from collections import deque

import numpy as np


class SequenceWindow:
    """Maintain one padded, fixed-length observation window per stream."""

    def __init__(self, sequence_length: int):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        self.sequence_length = int(sequence_length)
        self._histories: list[deque[np.ndarray]] = []

    def reset(self, n_streams: int) -> None:
        if n_streams < 0:
            raise ValueError("n_streams must be >= 0")
        self._histories = [deque(maxlen=self.sequence_length) for _ in range(n_streams)]

    def append(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations)
        if observations.ndim != 2:
            raise ValueError("observations must have shape (n_streams, features)")
        if len(self._histories) != observations.shape[0]:
            self.reset(observations.shape[0])

        windows = []
        for index, observation in enumerate(observations):
            history = self._histories[index]
            history.append(np.asarray(observation, dtype=observations.dtype))
            values = list(history)
            if len(values) < self.sequence_length:
                values = [values[0]] * (self.sequence_length - len(values)) + values
            windows.append(np.stack(values, axis=0))
        return np.stack(windows, axis=0)
