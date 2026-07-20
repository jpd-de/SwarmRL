"""Simple ring-buffer replay memory for off-policy algorithms."""

import numpy as np

from swarmrl.replay_buffer.ring_storage import RingStorage
from swarmrl.replay_buffer.transition import Transition


class ReplayBuffer:
    """
    Fixed-size replay buffer with random minibatch sampling, using lazy-allocated
    NumPy buffers for with lazy-allocated NumPy arrays for vectorized sampling.
    """

    def __init__(self, capacity: int, seed: int | None = None):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity : int
                The maximum number of transitions the buffer can store.
                Must be greater than 0.
            seed : int
                Optional random seed for reproducibility of sample indices.

        Raises:
            ValueError: If `capacity` is less than or equal to 0.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._storage = RingStorage(capacity, seed)

    @property
    def capacity(self) -> int:
        return self._storage.capacity

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: Transition) -> None:
        self._storage.add(transition)

    def can_sample(self, batch_size: int) -> bool:
        return len(self) >= int(batch_size)

    def can_sample_for_training(
        self, batch_size: int, sequence_length: int = 1
    ) -> bool:
        """Return whether this buffer can provide the requested training batch."""
        return sequence_length == 1 and self.can_sample(batch_size)

    def sample_for_training(
        self, batch_size: int, sequence_length: int = 1
    ) -> dict[str, np.ndarray]:
        """Sample the training batch using this buffer's data contract."""
        if sequence_length != 1:
            raise TypeError(
                "ReplayBuffer cannot sample temporal windows; use "
                "SequenceReplayBuffer."
            )
        return self.sample(batch_size)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Cannot sample {batch_size} transitions "
                f"from buffer of size {len(self)}."
            )

        indices = self._storage.rng.choice(len(self), size=batch_size, replace=False)

        return {
            key: self._storage.buffers[key][indices]
            for key in (
                "observation",
                "action",
                "reward",
                "next_observation",
                "terminated",
                "truncated",
            )
        }
