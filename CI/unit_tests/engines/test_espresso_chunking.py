"""
Test the HDF5 chunk-shape selection used by EspressoMD's trajectory writer.

These tests call `EspressoMD._choose_chunk_shape` directly as a staticmethod,
without instantiating EspressoMD, so they don't require ESPResSo to be
installed.
"""

import unittest as ut

import numpy as np

from swarmrl.engine.espresso import EspressoMD


class ChunkShapeTest(ut.TestCase):
    def test_chunk_never_splits_per_step_axes(self):
        for shape, dtype in [((1, 260, 3), float), ((1, 1, 1), float)]:
            chunks = EspressoMD._choose_chunk_shape(shape, dtype, write_chunk_size=1)
            self.assertEqual(chunks[1:], shape[1:])
            self.assertGreaterEqual(chunks[0], 1)

    def test_time_chunk_shrinks_for_larger_per_step_payload(self):
        write_chunk_size = 1
        small_chunks = EspressoMD._choose_chunk_shape(
            (1, 10, 3), float, write_chunk_size
        )
        large_chunks = EspressoMD._choose_chunk_shape(
            (1, 10_000, 3), float, write_chunk_size
        )

        # A dataset with many more particles per timestep should get a
        # shorter time-axis chunk to keep the chunk byte size in check, not a
        # fixed constant regardless of per-step size.
        self.assertLess(large_chunks[0], small_chunks[0])
        # Particle/coordinate axes are still never split.
        self.assertEqual(large_chunks[1:], (10_000, 3))
        self.assertEqual(small_chunks[1:], (10, 3))

    def test_write_chunk_size_floor_beats_target_and_cap(self):
        # write_chunk_size deliberately exceeds the internal _MAX_TIME_CHUNK
        # cap (10_000) to confirm the floor always wins over the cap.
        write_chunk_size = 15_000
        chunks = EspressoMD._choose_chunk_shape(
            (1, 2, 3), float, write_chunk_size
        )
        self.assertEqual(chunks[0], write_chunk_size)

    def test_chunk_dtype_affects_time_chunk_size(self):
        write_chunk_size = 1
        float_chunks = EspressoMD._choose_chunk_shape(
            (1, 100, 3), np.float64, write_chunk_size
        )
        int8_chunks = EspressoMD._choose_chunk_shape(
            (1, 100, 3), np.int8, write_chunk_size
        )
        # A smaller dtype means more elements fit in the target byte budget,
        # so the time-axis chunk should be at least as long.
        self.assertGreaterEqual(int8_chunks[0], float_chunks[0])


if __name__ == "__main__":
    ut.main()
