"""A small, named-constant opacity law and hydrostatic ODE seed.

This module is intentionally a diagnostic candidate, not a replacement for
the production opacity tables.  It implements the structural proposal in the
handoff with only named physical constants:

* Saha ionization of H and the low-ionization-potential electron donors Na, K,
  Ca, Mg, and Fe;
* H-minus bound-free/free-free opacity;
* hydrogen Balmer/Paschen bound-free opacity;
* a Kramers free-free fallback gated by the Saha hydrogen-ionization fraction;
  the old ungated Kramers bound-free term is deliberately absent.

The law is evaluated in local ``(T, P)`` and keeps opacity positive by summing
positive components.  The hydrostatic branch integrates the coupled equation
``dm/dtau = 1/kappa(T, g*m)`` in ``log(tau), log(m)`` coordinates, so it does
not use a pressure fixed-point iteration or a fitted polynomial extrapolation.
The constants are exposed in a dataclass so any later calibration is explicit.
The H-minus free-free branch is the standard low-order Rosseland estimate
used in stellar-atmosphere teaching texts.  This is still a warm-start
candidate, not a replacement for the production opacity tables.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TextbookOpacityConstants:
    """Physical constants and explicitly marked low-order approximations."""

    # CODATA/cgs constants.
    boltzmann_erg_per_K: float = 1.380649e-16
    boltzmann_eV_per_K: float = 8.617333262e-5
    planck_erg_s: float = 6.62607015e-27
    speed_of_light_cm_s: float = 2.99792458e10
    electron_mass_g: float = 9.1093837015e-28
    hydrogen_mass_g: float = 1.6735575e-24
    thomson_cross_section_cm2: float = 6.6524587321e-25
    eV_to_erg: float = 1.602176634e-12

    # Composition convention.
    hydrogen_mass_fraction: float = 0.7381
    helium_mass_fraction: float = 0.2485
    solar_metal_mass_fraction: float = 0.0134
    neutral_mean_molecular_weight: float = 1.30
    alpha_element_fraction: float = 0.694

    # Ionization energies in eV.
    hydrogen_ionization_eV: float = 13.5984
    sodium_ionization_eV: float = 5.1391
    potassium_ionization_eV: float = 4.3407
    calcium_ionization_eV: float = 6.1132
    magnesium_ionization_eV: float = 7.6462
    iron_ionization_eV: float = 7.9024
    hydrogen_minus_affinity_eV: float = 0.7542

    # Solar donor number fractions relative to hydrogen.  These are the only
    # abundance anchors used by the analytic donor closure.
    sodium_per_hydrogen_solar: float = 2.04e-6
    potassium_per_hydrogen_solar: float = 1.32e-7
    calcium_per_hydrogen_solar: float = 2.19e-6
    magnesium_per_hydrogen_solar: float = 3.98e-5
    iron_per_hydrogen_solar: float = 3.16e-5

    # H-minus bound-free abundance closure.  This fixed cross-section is kept
    # from v1 so the v2 ablation isolates the requested free-free change.
    hminus_boundfree_cross_section_cm2: float = 1.0e-17

    # H-minus free-free: low-order Rosseland mean estimate.  Its stated
    # validity is the neutral/partially ionized 3000--7000 K branch; the
    # linear window prevents it from being extrapolated into the hot branch.
    hminus_freefree_rosseland_coefficient: float = 2.5e-31
    hminus_freefree_reference_metal_mass_fraction: float = 0.02
    hminus_freefree_density_exponent: float = 0.5
    hminus_freefree_temperature_exponent: float = 9.0
    hminus_freefree_full_strength_temperature_K: float = 6000.0
    hminus_freefree_zero_strength_temperature_K: float = 7000.0

    # Neutral-hydrogen Balmer/Paschen bound-free branch.  The Balmer-edge
    # cross-section is the n=2 value, not the 6.3e-18 cm2 Lyman-edge value.
    hydrogen_balmer_cross_section_cm2: float = 1.40e-17
    hydrogen_paschen_cross_section_cm2: float = 1.20e-18
    hydrogen_representative_photon_energy_over_kT: float = 3.8
    hydrogen_rayleigh_cross_section_at_500nm_cm2: float = 5.799e-29

    # Kramers free-free fallback only.  The H-ionization fraction is the
    # explicit hot-end window; there is no Kramers bound-free contribution.
    kramers_freefree_coefficient: float = 3.68e22

    # Explicit seed-only calibration convention.  It is not used to fit
    # local opacity; it closes the tau=0 surface anchor before P is available.
    surface_anchor_opacity_cm2_per_g: float = 0.34


DEFAULT_TEXTBOOK_CONSTANTS = TextbookOpacityConstants()


def _as_profile_inputs(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    thermal = np.asarray(temperature, dtype=np.float64)
    pressure = np.asarray(gas_pressure, dtype=np.float64)
    if thermal.ndim == 1:
        thermal = thermal[None, :]
    if pressure.ndim == 1:
        pressure = pressure[None, :]
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("labels must have shape (N, 5)")
    if thermal.shape != pressure.shape or thermal.shape[0] != values.shape[0]:
        raise ValueError("temperature and gas_pressure must have shape (N, layers)")
    if (
        np.any(~np.isfinite(values))
        or np.any(~np.isfinite(thermal))
        or np.any(~np.isfinite(pressure))
        or np.any(values[:, 0] <= 0.0)
        or np.any(values[:, 4] <= 0.0)
        or np.any(thermal <= 0.0)
        or np.any(pressure <= 0.0)
    ):
        raise ValueError("opacity inputs must be finite and positive where required")
    return values, thermal, pressure


def _log_saha_ratio(
    temperature: np.ndarray,
    electron_density: np.ndarray,
    ionization_energy_eV: float,
    constants: TextbookOpacityConstants,
) -> np.ndarray:
    """Return log(n_ionized / n_neutral) for a singly ionized species."""

    prefactor = (
        2.0
        * np.pi
        * constants.electron_mass_g
        * constants.boltzmann_erg_per_K
        * temperature
        / constants.planck_erg_s**2
    ) ** 1.5
    return np.clip(
        np.log(np.maximum(prefactor, 1.0e-300))
        - np.log(np.maximum(electron_density, 1.0e-300))
        - ionization_energy_eV / (constants.boltzmann_eV_per_K * temperature),
        -700.0,
        700.0,
    )


def _ionized_fraction(log_ratio: np.ndarray) -> np.ndarray:
    ratio = np.exp(np.clip(log_ratio, -700.0, 700.0))
    return ratio / (1.0 + ratio)


def _composition_scales(
    labels: np.ndarray, constants: TextbookOpacityConstants
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    metallicity = 10.0 ** np.clip(labels[:, 2], -6.0, 2.0)
    alpha = 10.0 ** np.clip(labels[:, 3], -2.0, 2.0)
    alpha_mixture = (
        constants.alpha_element_fraction * alpha
        + (1.0 - constants.alpha_element_fraction)
    )
    metal_mass = constants.solar_metal_mass_fraction * metallicity * alpha_mixture
    donors = {
        "Na": constants.sodium_per_hydrogen_solar * metallicity,
        "K": constants.potassium_per_hydrogen_solar * metallicity,
        "Ca": constants.calcium_per_hydrogen_solar * metallicity * alpha,
        "Mg": constants.magnesium_per_hydrogen_solar * metallicity * alpha,
        "Fe": constants.iron_per_hydrogen_solar * metallicity,
    }
    return metal_mass, donors


def _hminus_temperature_window(
    temperature: np.ndarray, constants: TextbookOpacityConstants
) -> np.ndarray:
    """Apply the declared neutral-branch validity window continuously."""

    lower = float(constants.hminus_freefree_full_strength_temperature_K)
    upper = float(constants.hminus_freefree_zero_strength_temperature_K)
    if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower):
        raise ValueError("H-minus free-free temperature window is invalid")
    return np.clip((upper - temperature) / (upper - lower), 0.0, 1.0)


def hminus_freefree_rosseland_opacity(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Evaluate the low-order Gray-style H-minus Rosseland estimate.

    In the neutral/partially ionized branch this is

    ``2.5e-31 (Z/0.02) rho**0.5 T**9`` cm2 g-1.

    The linear 6000--7000 K taper is a declared regime switch, not a fitted
    correction.  It keeps this cool-atmosphere approximation out of the hot
    branch, where the H-minus ion is no longer the intended carrier.
    """

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    state = saha_electron_diagnostics(
        values, thermal, pressure, constants=constants
    )
    rho = state["rho_g_cm3"]
    metal_mass, _ = _composition_scales(values, constants)
    metallicity_scale = metal_mass[:, None] / float(
        constants.hminus_freefree_reference_metal_mass_fraction
    )
    estimate = (
        float(constants.hminus_freefree_rosseland_coefficient)
        * metallicity_scale
        * rho ** float(constants.hminus_freefree_density_exponent)
        * thermal ** float(constants.hminus_freefree_temperature_exponent)
    )
    return np.maximum(estimate * _hminus_temperature_window(thermal, constants), 1.0e-30)


def saha_electron_diagnostics(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    iterations: int = 48,
) -> dict[str, np.ndarray]:
    """Solve a compact H+metal Saha charge balance in local ``(T,P)``."""

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rho = (
        constants.neutral_mean_molecular_weight
        * constants.hydrogen_mass_g
        * pressure
        / (constants.boltzmann_erg_per_K * thermal)
    )
    n_hydrogen = constants.hydrogen_mass_fraction * rho / constants.hydrogen_mass_g
    _, donor_fractions = _composition_scales(values, constants)
    donor_number_densities = {
        name: n_hydrogen * fraction[:, None]
        for name, fraction in donor_fractions.items()
    }
    donor_energies = {
        "Na": constants.sodium_ionization_eV,
        "K": constants.potassium_ionization_eV,
        "Ca": constants.calcium_ionization_eV,
        "Mg": constants.magnesium_ionization_eV,
        "Fe": constants.iron_ionization_eV,
    }
    electron_density = np.maximum(
        1.0e-8 * pressure / (constants.boltzmann_erg_per_K * thermal),
        1.0e-30,
    )
    for _ in range(int(iterations)):
        h_fraction = _ionized_fraction(
            _log_saha_ratio(
                thermal,
                electron_density,
                constants.hydrogen_ionization_eV,
                constants,
            )
        )
        charge_density = n_hydrogen * h_fraction
        donor_fractions_ionized: dict[str, np.ndarray] = {}
        for name, energy in donor_energies.items():
            fraction = _ionized_fraction(
                _log_saha_ratio(thermal, electron_density, energy, constants)
            )
            donor_fractions_ionized[name] = fraction
            charge_density = charge_density + donor_number_densities[name] * fraction
        updated = np.maximum(charge_density, 1.0e-30)
        electron_density = np.exp(
            0.5
            * (
                np.log(np.maximum(electron_density, 1.0e-300))
                + np.log(updated)
            )
        )
    h_fraction = _ionized_fraction(
        _log_saha_ratio(
            thermal,
            electron_density,
            constants.hydrogen_ionization_eV,
            constants,
        )
    )
    n_hydrogen_neutral = n_hydrogen * (1.0 - h_fraction)
    return {
        "rho_g_cm3": rho,
        "electron_density_cm3": electron_density,
        "hydrogen_ionized_fraction": h_fraction,
        "hydrogen_neutral_density_cm3": n_hydrogen_neutral,
        "donor_ionized_fraction_Na": donor_fractions_ionized["Na"],
        "donor_ionized_fraction_K": donor_fractions_ionized["K"],
        "donor_ionized_fraction_Ca": donor_fractions_ionized["Ca"],
        "donor_ionized_fraction_Mg": donor_fractions_ionized["Mg"],
        "donor_ionized_fraction_Fe": donor_fractions_ionized["Fe"],
    }


def textbook_opacity_components(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> dict[str, np.ndarray]:
    """Return positive, physically named opacity components in cm2/g."""

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    state = saha_electron_diagnostics(
        values, thermal, pressure, constants=constants
    )
    rho = state["rho_g_cm3"]
    electron_density = state["electron_density_cm3"]
    neutral_hydrogen = state["hydrogen_neutral_density_cm3"]
    kT_eV = constants.boltzmann_eV_per_K * thermal
    representative_energy_over_kT = float(
        constants.hydrogen_representative_photon_energy_over_kT
    )
    if not np.isfinite(representative_energy_over_kT) or representative_energy_over_kT <= 0.0:
        raise ValueError("representative photon energy must be finite and positive")

    # H- equilibrium: H0 + e <-> H-.  The statistical-weight factor is kept
    # explicit rather than hidden in a fit coefficient.
    hminus_saha_volume = (
        constants.planck_erg_s**2
        / (2.0 * np.pi * constants.electron_mass_g * constants.boltzmann_erg_per_K * thermal)
    ) ** 1.5
    hminus_ratio = (
        electron_density
        * hminus_saha_volume
        * 0.25
        * np.exp(
            np.clip(constants.hydrogen_minus_affinity_eV / kT_eV, -700.0, 700.0)
        )
    )
    hminus_density = neutral_hydrogen * np.clip(hminus_ratio, 0.0, 1.0)
    # Keep H-minus bound-free unchanged in this ablation.  The requested
    # Rosseland representative photon energy is applied to the neutral-H
    # Balmer/Paschen branch below.
    hminus_bf = (
        constants.hminus_boundfree_cross_section_cm2 * hminus_density / rho
    )
    hminus_ff = hminus_freefree_rosseland_opacity(
        values, thermal, pressure, constants=constants
    )

    # Hydrogen n=2/n=3 Boltzmann populations for Balmer/Paschen continua.
    n2 = neutral_hydrogen * (8.0 / 2.0) * np.exp(
        np.clip(-10.1988 / kT_eV, -700.0, 0.0)
    )
    n3 = neutral_hydrogen * (18.0 / 2.0) * np.exp(
        np.clip(-12.0875 / kT_eV, -700.0, 0.0)
    )
    hydrogen_stimulated = 1.0 - np.exp(-representative_energy_over_kT)
    hydrogen_bf = hydrogen_stimulated * (
        constants.hydrogen_balmer_cross_section_cm2 * n2
        + constants.hydrogen_paschen_cross_section_cm2 * n3
    ) / rho

    # Only a hydrogen-ionization-gated free-free fallback remains from the
    # Kramers family.  In particular, there is no ungated metal Kramers
    # bound-free term: that was the multi-dex failure mode in v1.
    hydrogen_ionized = state["hydrogen_ionized_fraction"]
    kramers_freefree = (
        constants.kramers_freefree_coefficient
        * (1.0 + constants.hydrogen_mass_fraction)
        * rho
        * np.power(thermal, -3.5)
        * hydrogen_ionized
    )
    electron_scattering = (
        constants.thomson_cross_section_cm2 * electron_density / rho
    )
    components = {
        "hminus_boundfree": np.maximum(hminus_bf, 0.0),
        "hminus_freefree": np.maximum(hminus_ff, 0.0),
        "hydrogen_balmer_paschen_boundfree": np.maximum(hydrogen_bf, 0.0),
        "kramers_freefree": np.maximum(kramers_freefree, 0.0),
        "electron_scattering": np.maximum(electron_scattering, 0.0),
    }
    components["total"] = np.maximum(
        sum(components.values()), 1.0e-30
    )
    return components


def textbook_rosseland_opacity(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Evaluate the positive textbook Rosseland-opacity candidate."""

    return textbook_opacity_components(
        labels, temperature, gas_pressure, constants=constants
    )["total"]


WINDOW_NAMES = (
    "below_hminus_threshold",
    "hminus_to_paschen",
    "paschen_to_balmer",
    "balmer_to_lyman",
    "above_lyman",
)
ROSSELAND_WINDOW_QUADRATURE_ORDER = 32
_ROSSELAND_GL_NODES, _ROSSELAND_GL_WEIGHTS = np.polynomial.legendre.leggauss(
    ROSSELAND_WINDOW_QUADRATURE_ORDER
)


def frequency_window_edges_hz(
    *, constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS
) -> np.ndarray:
    """Return the fixed frequency boundaries used by the v3 synthesis.

    The first four non-zero boundaries are the H-minus photodetachment
    threshold and the hydrogen n=3, n=2, and n=1 series limits.  They are
    derived from named energies rather than fit to the corpus.
    """

    energies_eV = np.asarray(
        (
            constants.hydrogen_minus_affinity_eV,
            constants.hydrogen_ionization_eV / 9.0,
            constants.hydrogen_ionization_eV / 4.0,
            constants.hydrogen_ionization_eV,
        ),
        dtype=np.float64,
    )
    if np.any(~np.isfinite(energies_eV)) or np.any(energies_eV <= 0.0):
        raise ValueError("frequency-window threshold energies must be positive")
    frequency = energies_eV * constants.eV_to_erg / constants.planck_erg_s
    return np.concatenate((np.asarray([0.0]), frequency, np.asarray([np.inf])))


def _rosseland_weight(u: np.ndarray) -> np.ndarray:
    """Return the normalized-shape Rosseland weight in the variable u."""

    values = np.asarray(u, dtype=np.float64)
    q = np.exp(-values)
    denominator = -np.expm1(-values)
    numerator = values**4 * q
    return np.divide(
        numerator,
        denominator**2,
        out=np.zeros_like(values),
        where=values > 0.0,
    )


def _rosseland_weight_integral(
    lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Integrate the analytic Rosseland weight over finite u intervals."""

    lower_values = np.asarray(lower, dtype=np.float64)
    upper_values = np.asarray(upper, dtype=np.float64)
    if np.any(~np.isfinite(lower_values)) or np.any(~np.isfinite(upper_values)):
        raise ValueError("finite u boundaries are required for quadrature")
    if np.any(lower_values < 0.0) or np.any(upper_values <= lower_values):
        raise ValueError("Rosseland u intervals must be positive and increasing")
    half = 0.5 * (upper_values - lower_values)
    midpoint = 0.5 * (upper_values + lower_values)
    nodes = midpoint[..., None] + half[..., None] * _ROSSELAND_GL_NODES
    return half * np.sum(
        _ROSSELAND_GL_WEIGHTS * _rosseland_weight(nodes), axis=-1
    )


def rosseland_window_weights(
    temperature: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Return normalized Rosseland weights for the five fixed frequency windows."""

    thermal = np.asarray(temperature, dtype=np.float64)
    if np.any(~np.isfinite(thermal)) or np.any(thermal <= 0.0):
        raise ValueError("temperature must be finite and positive")
    edges = frequency_window_edges_hz(constants=constants)
    finite_edges = edges[1:-1]
    u_edges = (
        constants.planck_erg_s
        * finite_edges[None, :]
        / (constants.boltzmann_erg_per_K * thermal.reshape(-1, 1))
    )
    u_edges = np.column_stack(
        (
            np.zeros(thermal.size),
            u_edges,
            np.full(thermal.size, 100.0),
        )
    )
    integrals = np.column_stack(
        [
            _rosseland_weight_integral(u_edges[:, index], u_edges[:, index + 1])
            for index in range(len(WINDOW_NAMES))
        ]
    )
    # The u=100 truncation leaves a negligible tail; renormalizing makes the
    # window partition exact and keeps the harmonic mean numerically stable.
    total = np.sum(integrals, axis=1, keepdims=True)
    if np.any(~np.isfinite(total)) or np.any(total <= 0.0):
        raise ValueError("Rosseland window weights are non-finite")
    return (integrals / total).reshape(thermal.shape + (len(WINDOW_NAMES),))


def textbook_opacity_window_components(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> dict[str, np.ndarray]:
    """Build the v3 five-window opacity and its Rosseland harmonic synthesis.

    Each window contains the positive local components that are physically
    allowed above its threshold.  Components are added inside a window; the
    five window opacities are then combined as

    ``1 / kappa_R = sum_i w_i / kappa_i``.

    The window amplitudes are deliberately low-order: they are local
    Rosseland-scale component estimates, not a corpus-fitted monochromatic
    opacity table.  The fixed thresholds and analytic weights are the v3
    structural change being tested.
    """

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    state = saha_electron_diagnostics(
        values, thermal, pressure, constants=constants
    )
    rho = state["rho_g_cm3"]
    electron_density = state["electron_density_cm3"]
    neutral_hydrogen = state["hydrogen_neutral_density_cm3"]
    kT_eV = constants.boltzmann_eV_per_K * thermal
    representative_energy_over_kT = float(
        constants.hydrogen_representative_photon_energy_over_kT
    )
    if not np.isfinite(representative_energy_over_kT) or representative_energy_over_kT <= 0.0:
        raise ValueError("representative photon energy must be finite and positive")

    hminus_saha_volume = (
        constants.planck_erg_s**2
        / (2.0 * np.pi * constants.electron_mass_g * constants.boltzmann_erg_per_K * thermal)
    ) ** 1.5
    hminus_ratio = (
        electron_density
        * hminus_saha_volume
        * 0.25
        * np.exp(
            np.clip(constants.hydrogen_minus_affinity_eV / kT_eV, -700.0, 700.0)
        )
    )
    hminus_density = neutral_hydrogen * np.clip(hminus_ratio, 0.0, 1.0)
    hminus_boundfree = (
        constants.hminus_boundfree_cross_section_cm2 * hminus_density / rho
    )
    hminus_freefree = hminus_freefree_rosseland_opacity(
        values, thermal, pressure, constants=constants
    )

    n2 = neutral_hydrogen * 4.0 * np.exp(
        np.clip(-10.1988 / kT_eV, -700.0, 0.0)
    )
    n3 = neutral_hydrogen * 9.0 * np.exp(
        np.clip(-12.0875 / kT_eV, -700.0, 0.0)
    )
    hydrogen_stimulated = 1.0 - np.exp(-representative_energy_over_kT)
    hydrogen_paschen = (
        hydrogen_stimulated
        * constants.hydrogen_paschen_cross_section_cm2
        * n3
        / rho
    )
    hydrogen_balmer = (
        hydrogen_stimulated
        * constants.hydrogen_balmer_cross_section_cm2
        * n2
        / rho
    )

    hydrogen_ionized = state["hydrogen_ionized_fraction"]
    kramers_freefree = (
        constants.kramers_freefree_coefficient
        * (1.0 + constants.hydrogen_mass_fraction)
        * rho
        * np.power(thermal, -3.5)
        * hydrogen_ionized
    )
    electron_scattering = constants.thomson_cross_section_cm2 * electron_density / rho
    hydrogen_rayleigh = (
        constants.hydrogen_rayleigh_cross_section_at_500nm_cm2
        * neutral_hydrogen
        / rho
    )
    base = (
        hminus_freefree
        + kramers_freefree
        + electron_scattering
        + hydrogen_rayleigh
    )
    window_opacity = np.stack(
        (
            base,
            base + hminus_boundfree,
            base + hminus_boundfree + hydrogen_paschen,
            base + hminus_boundfree + hydrogen_paschen + hydrogen_balmer,
            base + hminus_boundfree + hydrogen_paschen + hydrogen_balmer,
        ),
        axis=-1,
    )
    weights = rosseland_window_weights(thermal, constants=constants)
    total = 1.0 / np.sum(weights / np.maximum(window_opacity, 1.0e-30), axis=-1)
    components = {
        "hminus_boundfree": np.maximum(hminus_boundfree, 0.0),
        "hminus_freefree": np.maximum(hminus_freefree, 0.0),
        "hydrogen_paschen_boundfree": np.maximum(hydrogen_paschen, 0.0),
        "hydrogen_balmer_boundfree": np.maximum(hydrogen_balmer, 0.0),
        "kramers_freefree": np.maximum(kramers_freefree, 0.0),
        "electron_scattering": np.maximum(electron_scattering, 0.0),
        "hydrogen_rayleigh_scattering": np.maximum(hydrogen_rayleigh, 0.0),
        "window_opacity": np.maximum(window_opacity, 1.0e-30),
        "window_weights": weights,
        "total": np.maximum(total, 1.0e-30),
    }
    if np.any(~np.isfinite(components["total"])) or np.any(components["total"] <= 0.0):
        raise ValueError("window opacity synthesis produced an invalid total")
    return components


def textbook_rosseland_opacity_v3(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Evaluate the v3 fixed-window Rosseland opacity candidate."""

    return textbook_opacity_window_components(
        labels, temperature, gas_pressure, constants=constants
    )["total"]


def _scalar_log_mass_rhs(
    log_tau: float,
    log_mass: float,
    labels: np.ndarray,
    log_tau_grid: np.ndarray,
    temperature: np.ndarray,
    gravity: float,
    constants: TextbookOpacityConstants,
    opacity_function: Callable[..., np.ndarray],
) -> float:
    local_temperature = float(
        np.interp(log_tau, log_tau_grid, np.log(temperature))
    )
    local_temperature = float(np.exp(local_temperature))
    mass = float(np.exp(np.clip(log_mass, -700.0, 700.0)))
    pressure = max(gravity * mass, 1.0e-30)
    opacity = float(
        opacity_function(
            labels[None, :],
            np.asarray([[local_temperature]]),
            np.asarray([[pressure]]),
            constants=constants,
        )[0, 0]
    )
    rhs = np.exp(np.clip(log_tau - log_mass - np.log(opacity), -100.0, 100.0))
    return float(np.clip(rhs, 0.0, 1.0e6))


def integrate_hydrostatic_opacity_ode(
    labels: np.ndarray,
    tau: np.ndarray,
    temperature: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    substeps_per_layer: int = 8,
    opacity_function: Callable[..., np.ndarray] = textbook_rosseland_opacity,
) -> np.ndarray:
    """Integrate ``dm/dtau=1/kappa(T,g*m)`` in positive log coordinates."""

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    depth = np.asarray(tau, dtype=np.float64)
    thermal = np.asarray(temperature, dtype=np.float64)
    if thermal.ndim == 1:
        thermal = thermal[None, :]
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("labels must have shape (N, 5)")
    if thermal.shape != (values.shape[0], depth.size):
        raise ValueError("temperature must have shape (N, len(tau))")
    if depth.ndim != 1 or depth.size < 2 or np.any(np.diff(depth) <= 0.0):
        raise ValueError("tau must be strictly increasing")
    if np.any(~np.isfinite(thermal)) or np.any(thermal <= 0.0):
        raise ValueError("temperature must be finite and positive")
    if substeps_per_layer < 1:
        raise ValueError("substeps_per_layer must be positive")

    output = np.empty_like(thermal)
    log_tau_grid = np.log(depth)
    for star in range(values.shape[0]):
        gravity = 10.0 ** float(values[star, 1])
        # The named 0.34 value is only the initial surface guess.  A few
        # damped local updates enforce the stated anchor m0=tau0/kappa0 while
        # supplying the pressure needed by the local opacity law.
        log_mass = float(
            np.log(
                max(
                    depth[0] / constants.surface_anchor_opacity_cm2_per_g,
                    1.0e-300,
                )
            )
        )
        for _ in range(8):
            surface_mass = float(np.exp(np.clip(log_mass, -700.0, 700.0)))
            surface_pressure = max(gravity * surface_mass, 1.0e-30)
            surface_opacity = float(
                opacity_function(
                    values[star : star + 1],
                    temperature[star : star + 1, :1],
                    np.asarray([[surface_pressure]]),
                    constants=constants,
                )[0, 0]
            )
            target_log_mass = np.log(
                max(depth[0] / max(surface_opacity, 1.0e-30), 1.0e-300)
            )
            log_mass = float(0.5 * (log_mass + target_log_mass))
        output[star, 0] = np.exp(log_mass)
        for layer in range(1, depth.size):
            left = float(log_tau_grid[layer - 1])
            right = float(log_tau_grid[layer])
            width = (right - left) / float(substeps_per_layer)
            for substep in range(substeps_per_layer):
                x = left + substep * width
                k1 = _scalar_log_mass_rhs(
                    x,
                    log_mass,
                    values[star],
                    log_tau_grid,
                    thermal[star],
                    gravity,
                    constants,
                    opacity_function,
                )
                k2 = _scalar_log_mass_rhs(
                    x + 0.5 * width,
                    log_mass + 0.5 * width * k1,
                    values[star], log_tau_grid, thermal[star], gravity, constants,
                    opacity_function,
                )
                k3 = _scalar_log_mass_rhs(
                    x + 0.5 * width,
                    log_mass + 0.5 * width * k2,
                    values[star], log_tau_grid, thermal[star], gravity, constants,
                    opacity_function,
                )
                k4 = _scalar_log_mass_rhs(
                    x + width,
                    log_mass + width * k3,
                    values[star], log_tau_grid, thermal[star], gravity, constants,
                    opacity_function,
                )
                log_mass += width * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                log_mass = float(np.clip(log_mass, -700.0, 700.0))
            output[star, layer] = np.exp(log_mass)
    if np.any(~np.isfinite(output)) or np.any(output <= 0.0):
        raise ValueError("opacity ODE produced a non-finite or non-positive mass profile")
    if np.any(np.diff(output, axis=1) <= 0.0):
        raise ValueError("opacity ODE produced a non-monotone mass profile")
    return output


def saha_aware_adiabatic_gradient(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Low-order H-ionization-aware ``nabla_ad`` for the convection probe."""

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    state = saha_electron_diagnostics(values, thermal, pressure, constants=constants)
    fraction = state["hydrogen_ionized_fraction"]
    ionization_window = 4.0 * fraction * (1.0 - fraction)
    return np.clip(0.4 - 0.3 * ionization_window, 0.1, 0.4)


def build_textbook_reduced_state(
    labels: np.ndarray,
    tau: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    include_convection: bool = True,
    substeps_per_layer: int = 8,
    opacity_function: Callable[..., np.ndarray] = textbook_rosseland_opacity,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build a grey/Hopf temperature plus the textbook opacity ODE mass seed."""

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    depth = np.asarray(tau, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("labels must have shape (N, 5)")
    if depth.ndim != 1 or depth.size < 2 or np.any(np.diff(depth) <= 0.0):
        raise ValueError("tau must be strictly increasing")
    temperature = values[:, 0, None] * (0.75 * (depth[None, :] + 2.0 / 3.0)) ** 0.25
    mass = integrate_hydrostatic_opacity_ode(
        values,
        depth,
        temperature,
        constants=constants,
        substeps_per_layer=substeps_per_layer,
        opacity_function=opacity_function,
    )
    gravity = 10.0 ** values[:, 1]
    pressure = gravity[:, None] * mass
    opacity = opacity_function(
        values, temperature, pressure, constants=constants
    )
    nabla_rad = (
        3.0
        * opacity
        * pressure
        * values[:, 0, None] ** 4
        / (16.0 * gravity[:, None] * temperature**4)
    )
    nabla_ad = saha_aware_adiabatic_gradient(
        values, temperature, pressure, constants=constants
    )
    convective = nabla_rad > nabla_ad
    if include_convection:
        for star in range(values.shape[0]):
            active = np.flatnonzero(convective[star])
            if active.size == 0:
                continue
            switch = int(active[0])
            if switch <= 0:
                continue
            for layer in range(switch + 1, depth.size):
                pressure_ratio = max(pressure[star, layer] / pressure[star, layer - 1], 1.0)
                gradient = float(nabla_ad[star, layer])
                temperature[star, layer] = temperature[star, layer - 1] * pressure_ratio**gradient
        mass = integrate_hydrostatic_opacity_ode(
            values,
            depth,
            temperature,
            constants=constants,
            substeps_per_layer=substeps_per_layer,
            opacity_function=opacity_function,
        )
        pressure = gravity[:, None] * mass
        opacity = opacity_function(
            values, temperature, pressure, constants=constants
        )
    diagnostics = {
        "rosseland_opacity": opacity,
        "gas_pressure": pressure,
        "nabla_rad": nabla_rad,
        "nabla_ad": nabla_ad,
        "convective_mask": convective,
    }
    return mass, temperature, diagnostics


def build_textbook_reduced_state_v3(
    labels: np.ndarray,
    tau: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    include_convection: bool = True,
    substeps_per_layer: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build the v3 grey/Hopf plus fixed-window opacity ODE seed."""

    return build_textbook_reduced_state(
        labels,
        tau,
        constants=constants,
        include_convection=include_convection,
        substeps_per_layer=substeps_per_layer,
        opacity_function=textbook_rosseland_opacity_v3,
    )
