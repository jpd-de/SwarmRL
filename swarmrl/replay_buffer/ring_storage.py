"""Shared lazy-allocated ring storage for replay buffers."""

from dataclasses import fields

import numpy as np

from swarmrl.replay_buffer.transition import Transition


class RingStorage:
    """Store transitions in fixed slots and report overwritten slot metadata."""

    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.size = 0
        self.position = 0
        self.buffers: dict[str, np.ndarray] = {}
        self.slot_records: list[tuple[int, int, int] | None] = [None] * self.capacity

    def __len__(self) -> int:
        return self.size

    def _init_buffers(self, transition: Transition) -> None:
        for field in fields(transition):
            value = np.asarray(getattr(transition, field.name))
            dtype = value.dtype
            if dtype == np.float64:
                dtype = np.float32
            elif dtype == np.int64:
                dtype = np.int32
            shape = (
                (self.capacity, 1)
                if value.ndim == 0
                else (self.capacity,) + value.shape
            )
            self.buffers[field.name] = np.empty(shape, dtype=dtype)

    def add(self, transition: Transition) -> tuple[int, tuple[int, int, int] | None]:
        if not self.buffers:
            self._init_buffers(transition)

        slot = self.position
        overwritten_record = self.slot_records[slot]
        for key in self.buffers:
            self.buffers[key][slot] = getattr(transition, key)

        self.slot_records[slot] = (
            int(transition.stream_id),
            int(transition.episode_id),
            int(transition.timestep),
        )
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return slot, overwritten_record
