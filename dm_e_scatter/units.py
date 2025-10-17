"""Unit conversion helpers for the scattering calculations."""
from __future__ import annotations

from typing import Iterable, Union

from .constants import C, ELEM_CHARGE, HBAR

Number = Union[float, Iterable[float]]


def eV_to_joule(energy_eV: Number) -> Number:
    """Convert an energy from electron-volts to joules."""
    if isinstance(energy_eV, (list, tuple)):
        return [value * ELEM_CHARGE for value in energy_eV]
    return energy_eV * ELEM_CHARGE


def joule_to_eV(energy_joule: Number) -> Number:
    """Convert an energy from joules to electron-volts."""
    if isinstance(energy_joule, (list, tuple)):
        return [value / ELEM_CHARGE for value in energy_joule]
    return energy_joule / ELEM_CHARGE


def GeV_to_kg(mass_GeV: Number) -> Number:
    """Convert a mass from GeV/c^2 to kilograms."""
    if isinstance(mass_GeV, (list, tuple)):
        return [GeV_to_kg(value) for value in mass_GeV]
    energy_joule = eV_to_joule(mass_GeV * 1e9)
    return energy_joule / C**2


def kg_to_GeV(mass_kg: Number) -> Number:
    """Convert a mass from kilograms to GeV/c^2."""
    if isinstance(mass_kg, (list, tuple)):
        return [kg_to_GeV(value) for value in mass_kg]
    energy_joule = mass_kg * C**2
    return joule_to_eV(energy_joule) * 1e-9


def km_per_s_to_m_per_s(speed_km_s: Number) -> Number:
    """Convert a speed from km/s to m/s."""
    if isinstance(speed_km_s, (list, tuple)):
        return [value * 1_000.0 for value in speed_km_s]
    return speed_km_s * 1_000.0


def m_per_s_to_km_per_s(speed_m_s: Number) -> Number:
    """Convert a speed from m/s to km/s."""
    if isinstance(speed_m_s, (list, tuple)):
        return [value / 1_000.0 for value in speed_m_s]
    return speed_m_s / 1_000.0


def mass_kg_to_wave_number(mass_kg: Number) -> Number:
    """Convert a rest mass (kg) into a wave number ``m c / ħ`` (1/m)."""
    if isinstance(mass_kg, (list, tuple)):
        return [mass_kg_to_wave_number(value) for value in mass_kg]
    return mass_kg * C / HBAR


def wave_number_to_mass_kg(wave_number: Number) -> Number:
    """Convert a wave number (1/m) to a rest mass in kilograms."""
    if isinstance(wave_number, (list, tuple)):
        return [wave_number_to_mass_kg(value) for value in wave_number]
    return wave_number * HBAR / C


__all__ = [
    "Number",
    "eV_to_joule",
    "joule_to_eV",
    "GeV_to_kg",
    "kg_to_GeV",
    "km_per_s_to_m_per_s",
    "m_per_s_to_km_per_s",
    "mass_kg_to_wave_number",
    "wave_number_to_mass_kg",
]
