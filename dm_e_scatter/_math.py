"""Internal utilities to apply scalar formulas element-wise on sequences."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _prepare(value: Any) -> Any:
    if _is_sequence(value):
        return list(value)
    return value


def elementwise(func: Callable[..., float], *args: Any) -> Any:
    """Apply ``func`` to inputs, broadcasting scalars over sequences."""

    prepared = [_prepare(arg) for arg in args]
    lengths = [len(arg) for arg in prepared if _is_sequence(arg)]

    if not lengths:
        return func(*prepared)

    if any(length != lengths[0] for length in lengths[1:]):
        raise ValueError("Input sequences must share a common length")

    length = lengths[0]
    result = [
        func(*[arg[i] if _is_sequence(arg) else arg for arg in prepared])
        for i in range(length)
    ]
    return result
