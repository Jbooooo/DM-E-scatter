"""Non-relativistic two-body kinematics utilities."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Union

from ._math import elementwise
from .constants import ELECTRON_MASS, HBAR

Number = Union[float, Sequence[float]]


def reduced_mass(m_projectile: Number, m_target: Number = ELECTRON_MASS) -> Number:
    """Return the reduced mass μ = m1 m2 / (m1 + m2)."""

    def _reduce(m1: float, m2: float) -> float:
        return (m1 * m2) / (m1 + m2)

    return elementwise(_reduce, m_projectile, m_target)


def incoming_wave_number(mu: Number, speed: Number) -> Number:
    """Return the magnitude of the incoming wave number k = μ v / ħ."""

    def _incoming(_mu: float, _speed: float) -> float:
        return _mu * _speed / HBAR

    return elementwise(_incoming, mu, speed)


def momentum_transfer_wave_number(mu: Number, speed: Number, theta: Number) -> Number:
    """Return the momentum-transfer wave number q = 2 k sin(theta / 2)."""

    def _transfer(_mu: float, _speed: float, _theta: float) -> float:
        k_mag = _mu * _speed / HBAR
        return 2.0 * k_mag * math.sin(_theta / 2.0)

    return elementwise(_transfer, mu, speed, theta)


def recoil_energy_to_wave_number(recoil_energy: Number, m_target: Number = ELECTRON_MASS) -> Number:
    """Return the wave number corresponding to a recoil energy via q^2 = 2 m T / ħ^2."""

    def _convert(energy: float, mass: float) -> float:
        return math.sqrt(2.0 * mass * energy) / HBAR

    return elementwise(_convert, recoil_energy, m_target)


def wave_number_to_recoil_energy(q: Number, m_target: Number = ELECTRON_MASS) -> Number:
    """Return the recoil energy corresponding to a momentum-transfer wave number."""

    def _convert(q_value: float, mass: float) -> float:
        return (q_value * HBAR) ** 2 / (2.0 * mass)

    return elementwise(_convert, q, m_target)


def maximum_recoil_energy(mu: Number, speed: Number, m_target: Number = ELECTRON_MASS) -> Number:
    """Return the kinematic endpoint T_max = 2 μ^2 v^2 / m_target."""

    def _max_energy(_mu: float, _speed: float, mass: float) -> float:
        return 2.0 * (_mu**2) * (_speed**2) / mass

    return elementwise(_max_energy, mu, speed, m_target)


__all__ = [
    "Number",
    "reduced_mass",
    "incoming_wave_number",
    "momentum_transfer_wave_number",
    "recoil_energy_to_wave_number",
    "wave_number_to_recoil_energy",
    "maximum_recoil_energy",
]
