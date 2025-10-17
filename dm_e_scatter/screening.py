"""Screening prescriptions for Yukawa interactions."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Union

from ._math import elementwise

ArrayLike = Union[float, Sequence[float]]


def thomas_fermi_screening(mediator_wave_number: ArrayLike, k_tf: ArrayLike | float) -> ArrayLike:
    """Apply Thomas–Fermi screening: ``m_φ^2 → m_φ^2 + k_{TF}^2``."""

    def _screen(mediator: float, screening: float) -> float:
        return math.sqrt(mediator**2 + screening**2)

    return elementwise(_screen, mediator_wave_number, k_tf)


__all__ = ["ArrayLike", "thomas_fermi_screening"]
