"""Plotting helpers for the dark-matter--electron scattering package."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

from .constants import ELECTRON_MASS
from .kinematics import maximum_recoil_energy
from .units import Number
from .yukawa import differential_cross_section_dOmega, differential_cross_section_dT


def _linspace(start: float, stop: float, num: int) -> list[float]:
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def plot_dsigma_dOmega(
    thetas: Sequence[float] | None,
    mu: Number,
    speed: Number,
    alpha_eff: Number,
    mediator_wave_number: Number,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot ``dσ/dΩ`` as a function of scattering angle ``θ``."""

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if thetas is None:
        thetas = _linspace(1e-6, math.pi, 300)
    else:
        thetas = list(thetas)

    dsdo = differential_cross_section_dOmega(thetas, mu, speed, alpha_eff, mediator_wave_number)
    ax.plot(thetas, dsdo)
    ax.set_xlabel(r"$\\theta$ [rad]")
    ax.set_ylabel(r"$d\\sigma/d\\Omega$ [m$^2$]")
    ax.set_yscale("log")
    ax.set_title("Yukawa scattering in the Born approximation")
    fig.tight_layout()
    return ax


def plot_dsigma_dT(
    recoils: Sequence[float] | None,
    mu: Number,
    speed: Number,
    alpha_eff: Number,
    mediator_wave_number: Number,
    m_target: Number = ELECTRON_MASS,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot ``dσ/dT`` over the recoil energy range ``[0, T_max]``."""

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    t_max = maximum_recoil_energy(mu, speed, m_target)
    if recoils is None:
        recoils = _linspace(0.0, t_max, 400)
    else:
        recoils = list(recoils)

    dsdt = differential_cross_section_dT(recoils, mu, speed, alpha_eff, mediator_wave_number, m_target)
    ax.plot(recoils, dsdt)
    ax.set_xlabel("Recoil energy T [J]")
    ax.set_ylabel(r"$d\\sigma/dT$ [m$^2$/J]")
    ax.set_yscale("log")
    ax.set_title("Recoil spectrum")
    fig.tight_layout()
    return ax


def scan_mediator_masses(
    mediator_scales: Iterable[float],
    recoils: Sequence[float] | None,
    mu: Number,
    speed: Number,
    alpha_eff: Number,
    m_target: Number = ELECTRON_MASS,
    ax: plt.Axes | None = None,
    legend: bool = True,
) -> plt.Axes:
    """Plot ``dσ/dT`` for a sequence of mediator scales (log-spaced recommended)."""

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    t_max = maximum_recoil_energy(mu, speed, m_target)
    if recoils is None:
        recoils = _linspace(0.0, t_max, 400)
    else:
        recoils = list(recoils)

    for m_phi in mediator_scales:
        dsdt = differential_cross_section_dT(recoils, mu, speed, alpha_eff, m_phi, m_target)
        label = rf"$m_\\phi = {m_phi:.2e}$ m$^{{-1}}$"
        ax.plot(recoils, dsdt, label=label)

    ax.set_xlabel("Recoil energy T [J]")
    ax.set_ylabel(r"$d\\sigma/dT$ [m$^2$/J]")
    ax.set_yscale("log")
    ax.set_title("Mediator-mass scan")
    if legend:
        ax.legend()
    fig.tight_layout()
    return ax


def save_figure(ax: plt.Axes, directory: str | Path, basename: str) -> None:
    """Save the figure containing ``ax`` as both PNG and PDF into ``directory``."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig = ax.figure
    for suffix in (".png", ".pdf"):
        fig_path = directory / f"{basename}{suffix}"
        fig.savefig(fig_path, dpi=300)


__all__ = [
    "plot_dsigma_dOmega",
    "plot_dsigma_dT",
    "scan_mediator_masses",
    "save_figure",
]
