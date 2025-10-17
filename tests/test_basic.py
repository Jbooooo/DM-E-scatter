import math

from dm_e_scatter import constants, kinematics, units, yukawa


def setup_parameters():
    m_chi = units.GeV_to_kg(0.1)
    v = units.km_per_s_to_m_per_s(220.0)
    mu = kinematics.reduced_mass(m_chi, constants.ELECTRON_MASS)
    mediator_mass = units.GeV_to_kg(1e-6)
    mediator_scale = units.mass_kg_to_wave_number(mediator_mass)
    alpha_eff = 1e-2
    return mu, v, alpha_eff, mediator_scale


def test_dsigma_domega_forward_peaking():
    mu, v, alpha_eff, mediator_scale = setup_parameters()
    thetas = [1e-4, 0.1, 0.5]
    dsdo = yukawa.differential_cross_section_dOmega(thetas, mu, v, alpha_eff, mediator_scale)
    assert all(dsdo[i] > dsdo[i + 1] for i in range(len(dsdo) - 1))


def test_light_mediator_inverse_quadratic():
    mu, v, alpha_eff, _ = setup_parameters()
    mediator_scale = 1e-6
    t_max = kinematics.maximum_recoil_energy(mu, v)
    t_values = [min(t_max * 0.9, units.eV_to_joule(10 ** x)) for x in [-3, -2.5, -2, -1.5, -1]]
    dsdt = yukawa.differential_cross_section_dT(t_values, mu, v, alpha_eff, mediator_scale)
    scaling = [t**2 * value for t, value in zip(t_values, dsdt)]
    mean_scaling = sum(scaling) / len(scaling)
    max_dev = max(abs(value - mean_scaling) for value in scaling)
    assert max_dev / mean_scaling < 0.3


def test_heavy_mediator_flat_spectrum():
    mu, v, alpha_eff, _ = setup_parameters()
    mediator_scale = 1e12
    t_max = kinematics.maximum_recoil_energy(mu, v)
    t_values = [t_max * (0.05 + 0.85 * i / 19) for i in range(20)]
    dsdt = yukawa.differential_cross_section_dT(t_values, mu, v, alpha_eff, mediator_scale)
    mean_value = sum(dsdt) / len(dsdt)
    variation = (max(dsdt) - min(dsdt)) / mean_value
    assert variation < 0.1


def test_velocity_endpoint_and_recoil_limit():
    mu, v, alpha_eff, mediator_scale = setup_parameters()
    tiny_speed = 1e-6
    theta = 0.4
    dsdo_slow = yukawa.differential_cross_section_dOmega(theta, mu, tiny_speed, alpha_eff, mediator_scale)
    assert dsdo_slow < 1e-20

    t_max = kinematics.maximum_recoil_energy(mu, v)
    dsdt_center = yukawa.differential_cross_section_dT(0.5 * t_max, mu, v, alpha_eff, mediator_scale)
    dsdt_endpoint = yukawa.differential_cross_section_dT(0.999 * t_max, mu, v, alpha_eff, mediator_scale)
    assert dsdt_endpoint < dsdt_center


def test_unit_consistency_between_joule_and_ev():
    mu, v, alpha_eff, mediator_scale = setup_parameters()
    energies_ev = [0.5, 1.0, 5.0]
    energies_joule = [units.eV_to_joule(value) for value in energies_ev]
    dsdt_joule = yukawa.differential_cross_section_dT(energies_joule, mu, v, alpha_eff, mediator_scale)
    dsdt_per_ev = [value * units.eV_to_joule(1.0) for value in dsdt_joule]
    ratios_reference = [value / dsdt_per_ev[0] for value in dsdt_per_ev]
    ratios_direct = [value / dsdt_joule[0] for value in dsdt_joule]
    for ref, direct in zip(ratios_reference, ratios_direct):
        assert math.isclose(ref, direct, rel_tol=1e-6)
