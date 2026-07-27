"""Logging utilities for the SwarmRL package."""

import sys
import typing

import jax
import jax.numpy as jnp
from loguru import logger


class _SwarmRLLoggingConfig:
    """Internal mutable logging settings shared across helper functions."""

    def __init__(self) -> None:
        self.sink_ids: typing.List[int] = []
        self.jax_runtime_log_level_no: int = logger.level("DEBUG").no
        self.jax_runtime_log_enabled: bool = True
        self.jax_runtime_summary_enabled: bool = False
        self.reported_nonfinite_labels: typing.Set[str] = set()


_LOGGING_CONFIG = _SwarmRLLoggingConfig()


def _to_level_number(level: typing.Union[int, str]) -> int:
    """Convert level names or numeric values to a numeric logging level."""

    if isinstance(level, str):
        return logger.level(level.upper()).no
    return int(level)


def set_jax_runtime_log_level(level: typing.Union[int, str]) -> None:
    """Set callback registration threshold for JAX runtime value logging."""

    _LOGGING_CONFIG.jax_runtime_log_level_no = _to_level_number(level)


def set_jax_runtime_log_enabled(enabled: bool) -> None:
    """Enable or disable JAX runtime value logging callbacks globally."""

    _LOGGING_CONFIG.jax_runtime_log_enabled = bool(enabled)


def set_jax_runtime_summary_enabled(enabled: bool) -> None:
    """Enable compact scalar summaries for JAX runtime values."""

    _LOGGING_CONFIG.jax_runtime_summary_enabled = bool(enabled)
    if enabled:
        _LOGGING_CONFIG.reported_nonfinite_labels.clear()


def setup_swarmrl_logger(
    filename: typing.Optional[str] = None,
    loglevel_terminal: typing.Union[int, str] = "INFO",
    loglevel_file: typing.Union[int, str] = "DEBUG",
    include_user_logs: bool = False,
    remove_default_sink: bool = True,
    log_jax_values: bool = False,
    log_jax_summaries: bool = False,
):
    """
    Configure package logging with Loguru and enable swarmrl log output.

    This function is opt-in and is intended to be called from user scripts.
    Before calling it, swarmrl logging is disabled by default in ``swarmrl.__init__``.

    Parameters
    ----------
    filename
            Name of the file where logs get written to. If None or an empty string,
            no file sink is created.
    loglevel_terminal
            Terminal log level for Loguru sinks. Supports Loguru level names
            (e.g. "INFO", "DEBUG") or integer level numbers.
    loglevel_file
            File log level for Loguru sinks. Supports Loguru level names
            or integer level numbers.
    include_user_logs
            If True, include non-swarmrl Loguru records (for user/application logs)
            in the configured sinks.
    remove_default_sink
            If True, remove Loguru's default stderr sink (id 0). This avoids duplicated
            output and default DEBUG-level spam when custom sinks are configured.
    log_jax_values
            If False, disable JAX runtime value logging callbacks produced via
            ``log_jax_runtime_value`` to prevent very verbose output.
    log_jax_summaries
            If True, enable compact finite/NaN/Inf and range summaries for JAX
            runtime values. These summaries do not print full arrays.

    """
    if remove_default_sink:
        try:
            logger.remove(0)
        except ValueError:
            pass

    for sink_id in _LOGGING_CONFIG.sink_ids:
        try:
            logger.remove(sink_id)
        except ValueError:
            pass
    _LOGGING_CONFIG.sink_ids = []

    loglevel_terminal = (
        loglevel_terminal.upper()
        if isinstance(loglevel_terminal, str)
        else int(loglevel_terminal)
    )
    loglevel_file = (
        loglevel_file.upper() if isinstance(loglevel_file, str) else int(loglevel_file)
    )

    log_format = "[<level>{level: <10}</level>] {time:YYYY-MM-DD HH:mm:ss}: {message}"

    def _sink_filter(record: dict) -> bool:
        if include_user_logs:
            return True
        return record["name"].startswith("swarmrl")

    active_level_numbers = [_to_level_number(loglevel_terminal)]

    if filename:
        _LOGGING_CONFIG.sink_ids.append(
            logger.add(
                filename,
                level=loglevel_file,
                format=log_format,
                filter=_sink_filter,
            )
        )
        active_level_numbers.append(_to_level_number(loglevel_file))

    _LOGGING_CONFIG.sink_ids.append(
        logger.add(
            sys.stderr,
            level=loglevel_terminal,
            format=log_format,
            filter=_sink_filter,
        )
    )

    set_jax_runtime_log_enabled(log_jax_values)
    set_jax_runtime_summary_enabled(log_jax_summaries)
    set_jax_runtime_log_level(min(active_level_numbers))
    logger.enable("swarmrl")


def log_jax_runtime_value(
    label: str, value, level: typing.Union[str, int] = "DEBUG"
) -> None:
    """Log JAX runtime values with the global Loguru logger.

    The callback is only registered when the requested log level is enabled.
    """

    # JAX captures this branch during tracing. If the logger level changes later,
    # previously compiled paths may keep the old behavior until retracing happens.
    if not _LOGGING_CONFIG.jax_runtime_log_enabled:
        return

    level_no = _to_level_number(level)

    if level_no < _LOGGING_CONFIG.jax_runtime_log_level_no:
        return

    def _emit(x):
        logger.log(level, "{label} = {value}", label=label, value=x)

    jax.debug.callback(_emit, value, ordered=True)


def runtime_summary_statistics(value):
    """Return compact finite/range statistics for an array or pytree.

    The returned tuple contains ``(finite, nan_count, inf_count, min, max,
    mean)``. Non-finite entries are excluded from the range and mean.
    """

    leaves = [jnp.ravel(jnp.asarray(leaf)) for leaf in jax.tree_util.tree_leaves(value)]
    if not leaves:
        leaves = [jnp.asarray([], dtype=jnp.float32)]
    values = jnp.concatenate(leaves)
    finite_mask = jnp.isfinite(values)
    finite_values = jnp.where(finite_mask, values, 0.0)
    finite_count = jnp.sum(finite_mask)
    safe_min = jnp.min(jnp.where(finite_mask, values, jnp.inf))
    safe_max = jnp.max(jnp.where(finite_mask, values, -jnp.inf))
    safe_mean = jnp.sum(finite_values) / jnp.maximum(finite_count, 1)

    return (
        jnp.all(finite_mask),
        jnp.sum(jnp.isnan(values)),
        jnp.sum(jnp.isinf(values)),
        safe_min,
        safe_max,
        safe_mean,
    )


def log_jax_runtime_summary(
    label: str,
    value,
    level: typing.Union[str, int] = "DEBUG",
    only_if_nonfinite: bool = False,
) -> None:
    """Log compact runtime statistics without materializing full arrays.

    When ``only_if_nonfinite`` is true, only the first non-finite report for a
    label is emitted. This is intended for action logits and model parameters,
    which would otherwise produce one log record per simulation step after a
    failure.
    """

    if not _LOGGING_CONFIG.jax_runtime_summary_enabled:
        return

    level_no = _to_level_number(level)
    if level_no < _LOGGING_CONFIG.jax_runtime_log_level_no:
        return

    def _emit(finite, nan_count, inf_count, minimum, maximum, mean):
        finite = bool(finite)
        if only_if_nonfinite and finite:
            return
        if only_if_nonfinite and label in _LOGGING_CONFIG.reported_nonfinite_labels:
            return
        if only_if_nonfinite and not finite:
            _LOGGING_CONFIG.reported_nonfinite_labels.add(label)
        logger.log(
            level,
            (
                "{label}: finite={finite} nan={nan_count} inf={inf_count} "
                "min={minimum} max={maximum} mean={mean}"
            ),
            label=label,
            finite=finite,
            nan_count=int(nan_count),
            inf_count=int(inf_count),
            minimum=float(minimum),
            maximum=float(maximum),
            mean=float(mean),
        )

    jax.debug.callback(
        _emit,
        *runtime_summary_statistics(value),
        ordered=True,
    )
