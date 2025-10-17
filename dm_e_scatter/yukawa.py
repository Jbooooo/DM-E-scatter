"""Born-approximation Yukawa scattering amplitudes and cross sections."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional, Union

from ._math import elementwise
from .constants import ELECTRON_MASS, HBAR
from .kinematics import (
    maximum_recoil_energy,
    momentum_transfer_wave_number,
    recoil_energy_to_wave_number,
)

ArrayLike = Union[float, Sequence[float]]


def born_amplitude(
    q_wave_number: ArrayLike,
    mu: ArrayLike,
    alpha_eff: ArrayLike,
    mediator_wave_number: ArrayLike,
) -> ArrayLike:
    """Return the Born amplitude f(q) for a Yukawa potential.

    Parameters
    ----------
    q_wave_number:
        Momentum-transfer wave number ``q``.
    mu:
        Reduced mass of the dark-matter--electron system.
    alpha_eff:
        Effective coupling strength appearing in the Yukawa potential.
    mediator_wave_number:
        Screening wave number associated with the mediator mass ``m_φ``.
    """

    def _amplitude(q_value: float, mu_value: float, alpha_value: float, mediator_value: float) -> float:
        denominator = q_value**2 + mediator_value**2
        return -8.0 * math.pi * mu_value * alpha_value * (HBAR**2) / denominator

    return elementwise(_amplitude, q_wave_number, mu, alpha_eff, mediator_wave_number)


def differential_cross_section_dOmega(
    theta: ArrayLike,
    mu: ArrayLike,
    speed: ArrayLike,
    alpha_eff: ArrayLike,
    mediator_wave_number: ArrayLike,
) -> ArrayLike:
    """Return the differential cross section ``dσ/dΩ`` at scattering angle ``θ``."""

    q_wave = momentum_transfer_wave_number(mu, speed, theta)

    def _cross_section(amplitude: float) -> float:
        return abs(amplitude) ** 2

    amplitude = born_amplitude(q_wave, mu, alpha_eff, mediator_wave_number)
    return elementwise(_cross_section, amplitude)


def differential_cross_section_dT(
    recoil_energy: ArrayLike,
    mu: ArrayLike,
    speed: ArrayLike,
    alpha_eff: ArrayLike,
    mediator_wave_number: ArrayLike,
    m_target: ArrayLike | float = ELECTRON_MASS,
    out_of_range: Optional[float] = 0.0,
) -> ArrayLike:
    """Return the differential cross section ``dσ/dT`` for electron recoil energy.

    Energies outside the kinematic range ``[0, T_max]`` return ``out_of_range`` (default 0).
    """

    def _compute(t_value: float, mu_value: float, speed_value: float, alpha_value: float, mediator_value: float, target_mass: float) -> float:
        t_max = maximum_recoil_energy(mu_value, speed_value, target_mass)
        if not (0.0 <= t_value <= t_max):
            return out_of_range if out_of_range is not None else 0.0
        cos_theta = 1.0 - t_value * target_mass / (mu_value**2 * speed_value**2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)
        dsdo = differential_cross_section_dOmega(theta, mu_value, speed_value, alpha_value, mediator_value)
        prefactor = 2.0 * math.pi * target_mass / (mu_value**2 * speed_value**2)
        return prefactor * dsdo

    return elementwise(_compute, recoil_energy, mu, speed, alpha_eff, mediator_wave_number, m_target)


def asymptotic_light_mediator_limit(
    recoil_energy: ArrayLike,
    mu: ArrayLike,
    speed: ArrayLike,
    alpha_eff: ArrayLike,
    m_target: ArrayLike | float = ELECTRON_MASS,
) -> ArrayLike:
    """Return ``dσ/dT`` in the massless-mediator limit ``m_φ → 0``."""

    q_wave = recoil_energy_to_wave_number(recoil_energy, m_target)

    def _prefactor(target_mass: float, mu_value: float, speed_value: float) -> float:
        return 2.0 * math.pi * target_mass / (mu_value**2 * speed_value**2)

    amplitude = born_amplitude(q_wave, mu, alpha_eff, mediator_wave_number=0.0)
    prefactor = elementwise(_prefactor, m_target, mu, speed)

    def _combine(pref: float, amp: float) -> float:
        return pref * (abs(amp) ** 2)

    return elementwise(_combine, prefactor, amplitude)


__all__ = [
    "ArrayLike",
    "born_amplitude",
    "differential_cross_section_dOmega",
    "differential_cross_section_dT",
    "asymptotic_light_mediator_limit",
]
