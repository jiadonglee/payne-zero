"""A small, named-constant opacity law and hydrostatic ODE seed.

This module is intentionally a diagnostic candidate, not a replacement for
the production opacity tables.  It implements the structural proposal in the
handoff with only named physical constants:

* Saha ionization of H and the low-ionization-potential electron donors Na, K,
  Ca, Mg, and Fe;
* H-minus bound-free/free-free opacity;
* hydrogen Balmer/Paschen bound-free opacity;
* Kramers free-free/bound-free opacity and electron scattering.

The law is evaluated in local ``(T, P)`` and keeps opacity positive by summing
positive components.  The hydrostatic branch integrates the coupled equation
``dm/dtau = 1/kappa(T, g*m)`` in ``log(tau), log(m)`` coordinates, so it does
not use a pressure fixed-point iteration or a fitted polynomial extrapolation.
The constants are exposed in a dataclass so any later calibration is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TextbookOpacityConstants:
    """Physical constants and explicitly marked low-order approximations."""

    # CODATA/cgs constants.
    boltzmann_erg_per_K: float = 1.380649e-16
    boltzmann_eV_per_K: float = 8.617333262e-5
    planck_erg_s: float = 6.62607015e-27
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

    # Opacity cross sections and coefficients.  The H-minus free-free
    # coefficient is an intentionally low-order Rosseland-mean approximation;
    # it is not a fitted corpus polynomial.
    hminus_boundfree_cross_section_cm2: float = 1.0e-17
    hminus_freefree_coefficient: float = 1.0e-37
    hydrogen_balmer_cross_section_cm2: float = 6.30e-18
    hydrogen_paschen_cross_section_cm2: float = 1.20e-18
    kramers_coefficient: float = 4.00e25
    kramers_metal_floor: float = 1.0e-3

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
    hminus_bf = (
        constants.hminus_boundfree_cross_section_cm2 * hminus_density / rho
    )
    hminus_ff = (
        constants.hminus_freefree_coefficient
        * electron_density
        * neutral_hydrogen
        / rho
        * np.sqrt(thermal / 5000.0)
    )

    # Hydrogen n=2/n=3 Boltzmann populations for Balmer/Paschen continua.
    n2 = neutral_hydrogen * (8.0 / 2.0) * np.exp(
        np.clip(-10.1988 / kT_eV, -700.0, 0.0)
    )
    n3 = neutral_hydrogen * (18.0 / 2.0) * np.exp(
        np.clip(-12.0875 / kT_eV, -700.0, 0.0)
    )
    stimulated = 1.0 - np.exp(np.clip(-1.0 / kT_eV, -700.0, 0.0))
    hydrogen_bf = stimulated * (
        constants.hydrogen_balmer_cross_section_cm2 * n2
        + constants.hydrogen_paschen_cross_section_cm2 * n3
    ) / rho

    metal_mass, _ = _composition_scales(values, constants)
    kramers = (
        constants.kramers_coefficient
        * (1.0 + constants.hydrogen_mass_fraction)
        * (
            metal_mass[:, None]
            + constants.kramers_metal_floor * constants.solar_metal_mass_fraction
        )
        * rho
        * np.power(thermal, -3.5)
    )
    electron_scattering = (
        constants.thomson_cross_section_cm2 * electron_density / rho
    )
    components = {
        "hminus_boundfree": np.maximum(hminus_bf, 0.0),
        "hminus_freefree": np.maximum(hminus_ff, 0.0),
        "hydrogen_balmer_paschen_boundfree": np.maximum(hydrogen_bf, 0.0),
        "kramers_freefree_boundfree": np.maximum(kramers, 0.0),
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


def _scalar_log_mass_rhs(
    log_tau: float,
    log_mass: float,
    labels: np.ndarray,
    log_tau_grid: np.ndarray,
    temperature: np.ndarray,
    gravity: float,
    constants: TextbookOpacityConstants,
) -> float:
    local_temperature = float(
        np.interp(log_tau, log_tau_grid, np.log(temperature))
    )
    local_temperature = float(np.exp(local_temperature))
    mass = float(np.exp(np.clip(log_mass, -700.0, 700.0)))
    pressure = max(gravity * mass, 1.0e-30)
    opacity = float(
        textbook_rosseland_opacity(
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
                textbook_rosseland_opacity(
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
                    x, log_mass, values[star], log_tau_grid, thermal[star], gravity, constants
                )
                k2 = _scalar_log_mass_rhs(
                    x + 0.5 * width,
                    log_mass + 0.5 * width * k1,
                    values[star], log_tau_grid, thermal[star], gravity, constants,
                )
                k3 = _scalar_log_mass_rhs(
                    x + 0.5 * width,
                    log_mass + 0.5 * width * k2,
                    values[star], log_tau_grid, thermal[star], gravity, constants,
                )
                k4 = _scalar_log_mass_rhs(
                    x + width,
                    log_mass + width * k3,
                    values[star], log_tau_grid, thermal[star], gravity, constants,
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
    )
    gravity = 10.0 ** values[:, 1]
    pressure = gravity[:, None] * mass
    opacity = textbook_rosseland_opacity(
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
        )
        pressure = gravity[:, None] * mass
        opacity = textbook_rosseland_opacity(
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
