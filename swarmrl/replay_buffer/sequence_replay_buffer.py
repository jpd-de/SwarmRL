"""Replay storage and sampling for temporally contiguous transitions."""

from collections import defaultdict

import numpy as np

from swarmrl.replay_buffer.ring_storage import RingStorage
from swarmrl.replay_buffer.transition import Transition


class SequenceReplayBuffer:
    """Ring-buffer replay memory that can sample contiguous state windows.

    Physical ring-buffer positions are treated only as storage slots. Logical
    order is recovered from the stream, episode, and timestep metadata stored
    on each transition.
    """

    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._storage = RingStorage(capacity, seed)
        self._stream_episodes: dict[tuple[int, int], list[int]] = defaultdict(list)

    @property
    def capacity(self) -> int:
        return self._storage.capacity

    def __len__(self) -> int:
        return len(self._storage)

    def _remove_slot(self, slot: int) -> None:
        record = self._storage.slot_records[slot]
        if record is None:
            return
        key = record[:2]
        slots = self._stream_episodes[key]
        self._stream_episodes[key] = [
            candidate for candidate in slots if candidate != slot
        ]
        if not self._stream_episodes[key]:
            del self._stream_episodes[key]

    def add(self, transition: Transition) -> None:
        slot = self._storage.position
        self._remove_slot(slot)
        slot, _ = self._storage.add(transition)
        record = self._storage.slot_records[slot]
        self._stream_episodes[record[:2]].append(slot)

    def _candidate_windows(self, sequence_length: int) -> list[tuple[int, ...]]:
        candidates: list[tuple[int, ...]] = []
        for key, slots in self._stream_episodes.items():
            del key
            ordered_slots = sorted(
                (
                    slot
                    for slot in slots
                    if self._storage.slot_records[slot] is not None
                ),
                key=lambda slot: self._storage.slot_records[slot][2],
            )
            block: list[int] = []
            previous_timestep = None
            for slot in ordered_slots:
                timestep = self._storage.slot_records[slot][2]
                if previous_timestep is None or timestep == previous_timestep + 1:
                    block.append(slot)
                else:
                    block = [slot]
                if len(block) >= sequence_length:
                    candidates.append(tuple(block[-sequence_length:]))
                previous_timestep = timestep
        return candidates

    def can_sample_sequences(self, batch_size: int, sequence_length: int) -> bool:
        if batch_size <= 0 or sequence_length <= 0:
            return False
        return len(self._candidate_windows(sequence_length)) >= int(batch_size)

    def can_sample_for_training(self, batch_size: int, sequence_length: int) -> bool:
        """Return whether a temporal training batch is currently available."""
        return self.can_sample_sequences(batch_size, sequence_length)

    def sample_for_training(
        self, batch_size: int, sequence_length: int
    ) -> dict[str, np.ndarray]:
        """Sample the temporal training batch using this buffer's contract."""
        return self.sample_sequences(batch_size, sequence_length)

    def sample_sequences(
        self, batch_size: int, sequence_length: int
    ) -> dict[str, np.ndarray]:
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be > 0")
        candidates = self._candidate_windows(sequence_length)
        if len(candidates) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} valid sequences of length "
                f"{sequence_length} from the replay buffer."
            )
        indices = self._storage.rng.choice(
            len(candidates), size=batch_size, replace=False
        )

        observations = []
        next_observations = []
        endpoint_slots = []
        for index in indices:
            slots = candidates[int(index)]
            observations.append(self._storage.buffers["observation"][list(slots)])
            next_observations.append(
                np.concatenate(
                    [
                        self._storage.buffers["observation"][list(slots[1:])],
                        self._storage.buffers["next_observation"][slots[-1]][None, ...],
                    ],
                    axis=0,
                )
            )
            endpoint_slots.append(slots[-1])

        endpoints = np.asarray(endpoint_slots, dtype=np.int32)
        return {
            "observation": np.asarray(observations),
            "next_observation": np.asarray(next_observations),
            "action": self._storage.buffers["action"][endpoints],
            "reward": self._storage.buffers["reward"][endpoints],
            "terminated": self._storage.buffers["terminated"][endpoints],
            "truncated": self._storage.buffers["truncated"][endpoints],
        }
