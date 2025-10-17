"""Physical constants used for dark-matter--electron scattering calculations."""
from __future__ import annotations

HBAR = 1.054_571_817e-34  # J s
C = 299_792_458.0  # m / s
ELEM_CHARGE = 1.602_176_634e-19  # C
ELECTRON_MASS = 9.109_383_7015e-31  # kg
FINE_STRUCTURE = 7.297_352_5693e-3  # dimensionless

__all__ = [
    "HBAR",
    "C",
    "ELEM_CHARGE",
    "ELECTRON_MASS",
    "FINE_STRUCTURE",
]
