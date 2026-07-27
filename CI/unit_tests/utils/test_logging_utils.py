import jax.numpy as jnp
import numpy as np

from swarmrl.utils.logging_utils import runtime_summary_statistics


def test_runtime_summary_statistics_reports_finite_values():
    stats = runtime_summary_statistics(jnp.array([1.0, -2.0, 3.0]))

    assert bool(stats[0])
    assert int(stats[1]) == 0
    assert int(stats[2]) == 0
    assert float(stats[3]) == -2.0
    assert float(stats[4]) == 3.0
    np.testing.assert_allclose(stats[5], 2.0 / 3.0)


def test_runtime_summary_statistics_reports_nonfinite_values():
    stats = runtime_summary_statistics(jnp.array([1.0, jnp.nan, jnp.inf]))

    assert not bool(stats[0])
    assert int(stats[1]) == 1
    assert int(stats[2]) == 1
    assert float(stats[3]) == 1.0
    assert float(stats[4]) == 1.0
    assert float(stats[5]) == 1.0
