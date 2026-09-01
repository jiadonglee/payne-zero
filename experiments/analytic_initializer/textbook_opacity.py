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
used in stellar-atmosphere teaching texts.  The historical v2/v3 functions
are preserved for auditability; v4 adds a separate 5 x 32 node-level
Rosseland construction using published hydrogenic and John (1988) formulae.
This is still a warm-start candidate, not a replacement for the production
opacity tables.
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
    aluminium_ionization_eV: float = 5.9858
    silicon_ionization_eV: float = 8.1517
    hydrogen_minus_affinity_eV: float = 0.7542

    # Solar donor number fractions relative to hydrogen.  These are the only
    # abundance anchors used by the analytic donor closure.
    sodium_per_hydrogen_solar: float = 2.04e-6
    potassium_per_hydrogen_solar: float = 1.32e-7
    calcium_per_hydrogen_solar: float = 2.19e-6
    magnesium_per_hydrogen_solar: float = 3.98e-5
    iron_per_hydrogen_solar: float = 3.16e-5

    # v4r1 donor abundances use the same AGSS09 number-abundance convention
    # as the production atmosphere code.  They remain explicit here so the
    # analytic candidate does not call the production EOS or abundance tables.
    v4r1_sodium_per_hydrogen_solar: float = 1.584893192461114e-6
    v4r1_potassium_per_hydrogen_solar: float = 9.772372209558112e-8
    v4r1_calcium_per_hydrogen_solar: float = 1.9952623149688787e-6
    v4r1_magnesium_per_hydrogen_solar: float = 3.630780547701014e-5
    v4r1_iron_per_hydrogen_solar: float = 2.8840315031266055e-5
    v4r1_aluminium_per_hydrogen_solar: float = 2.5703957827688646e-6
    v4r1_silicon_per_hydrogen_solar: float = 2.951209226666384e-5

    # Fixed ground-term approximations to the Saha factor 2 U_II / U_I.
    # They are atomic degeneracy ratios, not corpus-fit corrections.
    v4r1_sodium_saha_partition_ratio: float = 1.0
    v4r1_potassium_saha_partition_ratio: float = 1.0
    v4r1_calcium_saha_partition_ratio: float = 4.0
    v4r1_magnesium_saha_partition_ratio: float = 4.0
    v4r1_iron_saha_partition_ratio: float = 20.0 / 9.0
    v4r1_aluminium_saha_partition_ratio: float = 1.0 / 3.0
    v4r1_silicon_saha_partition_ratio: float = 4.0 / 3.0

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

    # v4 node-level hydrogenic continuum constants.  These are literature
    # anchors, not corpus-fit coefficients.  The threshold cross section for
    # an ns hydrogenic level follows the textbook n^2 threshold scaling in the
    # v4 low-order approximation.
    hydrogen_ground_edge_cross_section_cm2: float = 6.30e-18
    hydrogen_boundfree_level_count: int = 10
    hydrogen_boundfree_edge_cross_section_power: float = 2.0
    # Published hydrogenic *threshold* cross sections for n=1, 2, 3.  n=1 is
    # the same Lyman value already used by v4.  n=2 is the literature Balmer
    # edge, not n^2 * sigma_1.  n=3 is the Karzas-Latter threshold read
    # offline from the packaged table (not loaded at runtime).  Levels n>=4
    # keep the v4 n^2 Kramers edge.
    hydrogen_published_threshold_cross_section_cm2: tuple[float, ...] = (
        6.30e-18,
        1.40e-17,
        2.16e-17,
    )
    hydrogen_freefree_coefficient: float = 3.6919e8

    # John (1988) H-minus free-free polynomial, with lambda in micrometres.
    # The two tables cover 0.182--0.3645 and >=0.3645 micrometres.  They are
    # published formula coefficients, not parameters fitted to this corpus.
    hminus_freefree_short_coefficients: tuple[tuple[float, ...], ...] = (
        (518.1021, -734.8667, 1021.1775, -479.0721, 93.1373, -6.4285),
        (473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.3600),
        (-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571),
        (115.5291, 169.6374, -245.6490, 114.2430, -21.9972, 1.5097),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    hminus_freefree_long_coefficients: tuple[tuple[float, ...], ...] = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2483.3460, 285.8270, -2054.2910, 2827.7760, -1341.5370, 208.9520),
        (-3449.8890, -1158.3820, 8746.5230, -11485.6320, 5303.6090, -812.9390),
        (2200.0400, 2427.7190, -13651.1050, 16755.5240, -7510.4940, 1132.7380),
        (-696.2710, -1841.4000, 8624.9700, -10051.5300, 4400.0670, -655.0200),
        (88.2830, 444.5170, -1863.8640, 2095.2880, -901.7880, 132.9850),
    )
    hminus_freefree_wavelength_floor_um: float = 0.182
    hminus_freefree_temperature_floor_K: float = 1400.0
    hminus_freefree_temperature_ceiling_K: float = 10080.0

    # John (1988) H-minus photodetachment polynomial.  lambda is in
    # micrometres and lambda_0 is the 0.754 eV threshold.
    hminus_boundfree_john_coefficients: tuple[float, ...] = (
        152.519,
        49.534,
        -118.858,
        92.536,
        -34.194,
        4.982,
    )
    hminus_boundfree_john_short_wavelength_um: float = 0.125
    hminus_boundfree_john_threshold_wavelength_um: float = 1.6419

    # Kurucz/Gray low-order Rayleigh normalization retained as an explicit
    # scattering branch; its node dependence is the textbook nu^4 law.
    hydrogen_rayleigh_reference_wavelength_nm: float = 500.0

    # Explicit seed-only calibration convention.  It is not used to fit
    # local opacity; it closes the tau=0 surface anchor before P is available.
    surface_anchor_opacity_cm2_per_g: float = 0.34

    # Bates/ATLAS H2+ continuum.  The polynomial is the production branch's
    # named expansion, not a corpus fit.  Frequency is in Hz; the cutoff is
    # the Lyman limit used by that branch.
    h2plus_frequency_ceiling_hz: float = 3.28805e15
    h2plus_frequency_floor_hz: float = 3.0e12
    h2plus_fr_polynomial: tuple[float, ...] = (
        -3.0233e3,
        3.7797e2,
        -1.82496e1,
        3.9207e-1,
        -3.1672e-3,
    )
    h2plus_excitation_polynomial: tuple[float, ...] = (
        -7.342e-3,
        -2.409e0,
        1.028e0,
        -4.230e-1,
        1.224e-1,
        -1.351e-2,
    )

    # Kurucz/ATLAS He-minus free-free.  Number densities in the published
    # expansion are in 10^15 cm^-3, which is the 10^{-45} factor below.
    heminus_density_unit_cm3: float = 1.0e15
    heminus_frequency_floor_hz: float = 3.0e12
    heminus_a_coefficients: tuple[float, ...] = (3.397e-1, -5.216e14, 7.039e30)
    heminus_b_coefficients: tuple[float, ...] = (-4.116e3, 1.067e19, 8.135e34)
    heminus_c_coefficients: tuple[float, ...] = (5.081e8, -8.724e22, -5.659e37)

    # Hydrogenic helium ionization and He II continuum.  Helium is not a
    # Saha electron donor; these constants only close the two-step stage
    # densities on a frozen n_e.  Partition ratios are ground-term
    # 2 U_{r+1}/U_r values, not corpus fits.
    helium_first_ionization_eV: float = 24.587
    helium_second_ionization_eV: float = 54.418
    helium_i_to_ii_saha_partition_ratio: float = 4.0
    helium_ii_to_iii_saha_partition_ratio: float = 1.0
    helium_ionized_effective_charge: float = 2.0
    helium_ionized_boundfree_level_count: int = 10


DEFAULT_TEXTBOOK_CONSTANTS = TextbookOpacityConstants()

# Formal gate domains.  These are applicability declarations, not fitted
# cutoffs, and they do not change the local opacity formulae.
V4R1_FORMAL_TEMPERATURE_FLOOR_K = 3200.0
V4R2_FORMAL_TEMPERATURE_FLOOR_K = 4000.0
V4R3_FORMAL_TEMPERATURE_FLOOR_K = V4R2_FORMAL_TEMPERATURE_FLOOR_K
V4R4_FORMAL_TEMPERATURE_FLOOR_K = V4R3_FORMAL_TEMPERATURE_FLOOR_K
V4R5_FORMAL_TEMPERATURE_FLOOR_K = V4R3_FORMAL_TEMPERATURE_FLOOR_K
V4R6_FORMAL_TEMPERATURE_FLOOR_K = V4R3_FORMAL_TEMPERATURE_FLOOR_K
V4R6_PER_N_TEMPERATURE_CEILING_K = 15000.0

COMPONENT_NAMES_V4R3 = (
    "hminus_boundfree",
    "hminus_freefree",
    "hydrogen_boundfree",
    "hydrogen_freefree",
    "electron_scattering",
    "hydrogen_rayleigh_scattering",
    "h2plus",
    "heminus",
)
COMPONENT_NAMES_V4R4 = COMPONENT_NAMES_V4R3 + (
    "helium_ionized_boundfree",
    "helium_ionized_freefree",
)
COMPONENT_NAMES_V4R5 = COMPONENT_NAMES_V4R3
COMPONENT_NAMES_V4R6 = COMPONENT_NAMES_V4R5


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


def _v4r1_donor_specifications(
    labels: np.ndarray,
    constants: TextbookOpacityConstants,
) -> dict[str, tuple[np.ndarray, float, float]]:
    """Return v4r1 donor abundance, ionization energy, and 2 U_II / U_I."""

    metallicity = 10.0 ** np.clip(labels[:, 2], -6.0, 2.0)
    alpha = 10.0 ** np.clip(labels[:, 3], -2.0, 2.0)
    return {
        "Na": (
            constants.v4r1_sodium_per_hydrogen_solar * metallicity,
            constants.sodium_ionization_eV,
            constants.v4r1_sodium_saha_partition_ratio,
        ),
        "K": (
            constants.v4r1_potassium_per_hydrogen_solar * metallicity,
            constants.potassium_ionization_eV,
            constants.v4r1_potassium_saha_partition_ratio,
        ),
        "Ca": (
            constants.v4r1_calcium_per_hydrogen_solar * metallicity * alpha,
            constants.calcium_ionization_eV,
            constants.v4r1_calcium_saha_partition_ratio,
        ),
        "Mg": (
            constants.v4r1_magnesium_per_hydrogen_solar * metallicity * alpha,
            constants.magnesium_ionization_eV,
            constants.v4r1_magnesium_saha_partition_ratio,
        ),
        "Fe": (
            constants.v4r1_iron_per_hydrogen_solar * metallicity,
            constants.iron_ionization_eV,
            constants.v4r1_iron_saha_partition_ratio,
        ),
        "Al": (
            constants.v4r1_aluminium_per_hydrogen_solar * metallicity,
            constants.aluminium_ionization_eV,
            constants.v4r1_aluminium_saha_partition_ratio,
        ),
        "Si": (
            constants.v4r1_silicon_per_hydrogen_solar * metallicity * alpha,
            constants.silicon_ionization_eV,
            constants.v4r1_silicon_saha_partition_ratio,
        ),
    }


def saha_electron_diagnostics_v4r1(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    iterations: int = 48,
) -> dict[str, np.ndarray]:
    """Solve the v4r1 H+seven-donor Saha charge balance in local ``(T,P)``."""

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
    donor_specifications = _v4r1_donor_specifications(values, constants)
    donor_number_densities = {
        name: n_hydrogen * abundance[:, None]
        for name, (abundance, _, _) in donor_specifications.items()
    }
    electron_density = np.maximum(
        1.0e-8 * pressure / (constants.boltzmann_erg_per_K * thermal),
        1.0e-30,
    )
    donor_fractions_ionized: dict[str, np.ndarray] = {}
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
        donor_fractions_ionized = {}
        for name, (_, energy, partition_ratio) in donor_specifications.items():
            if not np.isfinite(partition_ratio) or partition_ratio <= 0.0:
                raise ValueError(f"invalid v4r1 Saha partition ratio for {name}")
            fraction = _ionized_fraction(
                _log_saha_ratio(thermal, electron_density, energy, constants)
                + np.log(float(partition_ratio))
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
    charge_density = n_hydrogen * h_fraction
    for name, (_, energy, partition_ratio) in donor_specifications.items():
        fraction = _ionized_fraction(
            _log_saha_ratio(thermal, electron_density, energy, constants)
            + np.log(float(partition_ratio))
        )
        donor_fractions_ionized[name] = fraction
        charge_density = charge_density + donor_number_densities[name] * fraction
    n_hydrogen_neutral = n_hydrogen * (1.0 - h_fraction)
    result = {
        "rho_g_cm3": rho,
        "electron_density_cm3": electron_density,
        "hydrogen_ionized_fraction": h_fraction,
        "hydrogen_neutral_density_cm3": n_hydrogen_neutral,
        "charge_balance_relative_residual": (
            charge_density - electron_density
        )
        / np.maximum(electron_density, 1.0e-300),
    }
    result.update(
        {
            f"donor_ionized_fraction_{name}": donor_fractions_ionized[name]
            for name in donor_specifications
        }
    )
    return result


def saha_electron_diagnostics_v4r3(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    iterations: int = 48,
) -> dict[str, np.ndarray]:
    """Solve v4r1 charge balance with an ideal-gas particle-count density.

    Neutral ``mu=1.30`` is replaced by ``n_tot = P/kT = n_nuclei + n_e``.
    Helium is a nucleus in that count and is not added as a Saha donor.
    """

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    n_tot = pressure / (constants.boltzmann_erg_per_K * thermal)
    helium_per_hydrogen = (
        constants.helium_mass_fraction
        / (4.0 * constants.hydrogen_mass_fraction)
    )
    donor_specifications = _v4r1_donor_specifications(values, constants)
    donor_per_hydrogen = np.zeros(values.shape[0], dtype=np.float64)
    for abundance, _, _ in donor_specifications.values():
        donor_per_hydrogen = donor_per_hydrogen + abundance
    nuclei_per_hydrogen = (
        1.0 + helium_per_hydrogen + donor_per_hydrogen[:, None]
    )
    electron_density = np.maximum(1.0e-8 * n_tot, 1.0e-30)
    donor_number_densities: dict[str, np.ndarray] = {}
    donor_fractions_ionized: dict[str, np.ndarray] = {}
    n_hydrogen = n_tot / nuclei_per_hydrogen
    for _ in range(int(iterations)):
        n_nuclei = np.maximum(n_tot - electron_density, 1.0e-30)
        n_hydrogen = n_nuclei / nuclei_per_hydrogen
        donor_number_densities = {
            name: n_hydrogen * abundance[:, None]
            for name, (abundance, _, _) in donor_specifications.items()
        }
        h_fraction = _ionized_fraction(
            _log_saha_ratio(
                thermal,
                electron_density,
                constants.hydrogen_ionization_eV,
                constants,
            )
        )
        charge_density = n_hydrogen * h_fraction
        donor_fractions_ionized = {}
        for name, (_, energy, partition_ratio) in donor_specifications.items():
            if not np.isfinite(partition_ratio) or partition_ratio <= 0.0:
                raise ValueError(f"invalid v4r3 Saha partition ratio for {name}")
            fraction = _ionized_fraction(
                _log_saha_ratio(thermal, electron_density, energy, constants)
                + np.log(float(partition_ratio))
            )
            donor_fractions_ionized[name] = fraction
            charge_density = charge_density + donor_number_densities[name] * fraction
        updated = np.minimum(np.maximum(charge_density, 1.0e-30), 0.95 * n_tot)
        electron_density = np.exp(
            0.5
            * (
                np.log(np.maximum(electron_density, 1.0e-300))
                + np.log(updated)
            )
        )
    n_nuclei = np.maximum(n_tot - electron_density, 1.0e-30)
    n_hydrogen = n_nuclei / nuclei_per_hydrogen
    donor_number_densities = {
        name: n_hydrogen * abundance[:, None]
        for name, (abundance, _, _) in donor_specifications.items()
    }
    h_fraction = _ionized_fraction(
        _log_saha_ratio(
            thermal,
            electron_density,
            constants.hydrogen_ionization_eV,
            constants,
        )
    )
    charge_density = n_hydrogen * h_fraction
    for name, (_, energy, partition_ratio) in donor_specifications.items():
        fraction = _ionized_fraction(
            _log_saha_ratio(thermal, electron_density, energy, constants)
            + np.log(float(partition_ratio))
        )
        donor_fractions_ionized[name] = fraction
        charge_density = charge_density + donor_number_densities[name] * fraction
    rho = n_hydrogen * constants.hydrogen_mass_g / constants.hydrogen_mass_fraction
    n_hydrogen_neutral = n_hydrogen * (1.0 - h_fraction)
    result = {
        "rho_g_cm3": rho,
        "electron_density_cm3": electron_density,
        "hydrogen_ionized_fraction": h_fraction,
        "hydrogen_neutral_density_cm3": n_hydrogen_neutral,
        "hydrogen_number_density_cm3": n_hydrogen,
        "helium_neutral_density_cm3": n_hydrogen * helium_per_hydrogen,
        "mean_molecular_weight": rho
        / np.maximum(n_tot * constants.hydrogen_mass_g, 1.0e-300),
        "charge_balance_relative_residual": (
            charge_density - electron_density
        )
        / np.maximum(electron_density, 1.0e-300),
    }
    result.update(
        {
            f"donor_ionized_fraction_{name}": donor_fractions_ionized[name]
            for name in donor_specifications
        }
    )
    return result


def _local_state_from_electron_density(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
    *,
    constants: TextbookOpacityConstants,
) -> dict[str, np.ndarray]:
    """Build a diagnostic local state around an externally supplied n_e."""

    _, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    supplied = np.asarray(electron_density, dtype=np.float64)
    if supplied.ndim == 1:
        supplied = supplied[None, :]
    if supplied.shape != thermal.shape:
        raise ValueError("electron_density override must match temperature")
    if np.any(~np.isfinite(supplied)) or np.any(supplied <= 0.0):
        raise ValueError("electron_density override must be finite and positive")
    rho = (
        constants.neutral_mean_molecular_weight
        * constants.hydrogen_mass_g
        * pressure
        / (constants.boltzmann_erg_per_K * thermal)
    )
    n_hydrogen = constants.hydrogen_mass_fraction * rho / constants.hydrogen_mass_g
    h_fraction = _ionized_fraction(
        _log_saha_ratio(
            thermal,
            supplied,
            constants.hydrogen_ionization_eV,
            constants,
        )
    )
    return {
        "rho_g_cm3": rho,
        "electron_density_cm3": supplied,
        "hydrogen_ionized_fraction": h_fraction,
        "hydrogen_neutral_density_cm3": n_hydrogen * (1.0 - h_fraction),
        "diagnostic_electron_density_override": np.ones_like(supplied, dtype=bool),
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


def rosseland_frequency_nodes(
    temperature: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed 5 x 32 Rosseland nodes, frequencies, and weights.

    The Gauss--Legendre nodes are mapped separately into the same five
    threshold intervals used by v3.  Unlike v3, the returned weights attach
    directly to every monochromatic opacity value, so no component is treated
    as constant inside a window.
    """

    thermal = np.asarray(temperature, dtype=np.float64)
    if thermal.ndim == 1:
        thermal = thermal[None, :]
    if thermal.ndim != 2 or np.any(~np.isfinite(thermal)) or np.any(thermal <= 0.0):
        raise ValueError("temperature must be a finite positive (N, layers) array")

    flat_temperature = thermal.reshape(-1)
    edges = frequency_window_edges_hz(constants=constants)
    finite_edges = edges[1:-1]
    u_edges = constants.planck_erg_s * finite_edges[None, :] / (
        constants.boltzmann_erg_per_K * flat_temperature[:, None]
    )
    u_edges = np.column_stack(
        (
            np.zeros(flat_temperature.size),
            u_edges,
            np.full(flat_temperature.size, 100.0),
        )
    )
    lower = u_edges[:, :-1]
    upper = u_edges[:, 1:]
    half = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    node_u = midpoint[:, :, None] + half[:, :, None] * _ROSSELAND_GL_NODES[None, None, :]
    local_weights = (
        half[:, :, None]
        * _ROSSELAND_GL_WEIGHTS[None, None, :]
        * _rosseland_weight(node_u)
    )
    normalization = np.sum(local_weights, axis=(1, 2), keepdims=True)
    if np.any(~np.isfinite(normalization)) or np.any(normalization <= 0.0):
        raise ValueError("node-level Rosseland weights are non-finite")
    node_weights = local_weights / normalization
    node_frequency = (
        constants.boltzmann_erg_per_K
        * flat_temperature[:, None, None]
        * node_u
        / constants.planck_erg_s
    )
    output_shape = thermal.shape + (len(WINDOW_NAMES), ROSSELAND_WINDOW_QUADRATURE_ORDER)
    return (
        node_frequency.reshape(output_shape),
        node_weights.reshape(output_shape),
        node_u.reshape(output_shape),
    )


def _john_hminus_boundfree_cross_section(
    wavelength_um: np.ndarray,
    *,
    constants: TextbookOpacityConstants,
) -> np.ndarray:
    """Evaluate the John (1988) polynomial H-minus photodetachment cross section."""

    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    threshold = float(constants.hminus_boundfree_john_threshold_wavelength_um)
    short_limit = float(constants.hminus_boundfree_john_short_wavelength_um)
    if not (0.0 < short_limit < threshold):
        raise ValueError("John H-minus wavelength limits are invalid")
    safe_wavelength = np.maximum(wavelength, 1.0e-300)
    excess = np.maximum(1.0 / safe_wavelength - 1.0 / threshold, 0.0)
    polynomial = np.zeros_like(safe_wavelength)
    for index, coefficient in enumerate(
        constants.hminus_boundfree_john_coefficients
    ):
        polynomial += float(coefficient) * excess ** (0.5 * index)
    cross_section = 1.0e-18 * safe_wavelength**3 * excess**1.5 * polynomial
    active = (wavelength >= short_limit) & (wavelength < threshold)
    return np.where(active, np.maximum(cross_section, 0.0), 0.0)


def _john_hminus_freefree_cross_section(
    temperature: np.ndarray,
    wavelength_um: np.ndarray,
    *,
    constants: TextbookOpacityConstants,
) -> np.ndarray:
    """Evaluate the John (1988) H-minus free-free polynomial.

    John tabulates the polynomial in micrometres and in the dimensionless
    temperature coordinate 5040/T.  The published fit is used only in its
    stated temperature range; H-minus is set to zero outside that range.
    """

    thermal = np.asarray(temperature, dtype=np.float64)
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    if wavelength.ndim < thermal.ndim:
        raise ValueError("wavelength nodes must include the temperature dimensions")
    floor = float(constants.hminus_freefree_temperature_floor_K)
    ceiling = float(constants.hminus_freefree_temperature_ceiling_K)
    transition = 0.3645
    if not (0.0 < floor < ceiling and 0.0 < transition):
        raise ValueError("John H-minus free-free validity limits are invalid")

    temperature_for_fit = np.clip(thermal, floor, ceiling)
    theta = 5040.0 / temperature_for_fit
    expanded_theta = theta[..., None, None]
    expanded_wavelength = np.maximum(wavelength, float(constants.hminus_freefree_wavelength_floor_um))

    def evaluate(
        coefficient_table: tuple[tuple[float, ...], ...],
    ) -> np.ndarray:
        table = np.asarray(coefficient_table, dtype=np.float64)
        if table.shape != (6, 6):
            raise ValueError("John H-minus free-free coefficient tables must be 6x6")
        inverse = 1.0 / np.maximum(expanded_wavelength, 1.0e-300)
        polynomial = np.zeros_like(expanded_wavelength)
        powers = (
            expanded_wavelength**2,
            np.ones_like(expanded_wavelength),
            inverse,
            inverse**2,
            inverse**3,
            inverse**4,
        )
        for level_index in range(6):
            term = np.zeros_like(expanded_wavelength)
            for power_index in range(6):
                term += table[level_index, power_index] * powers[power_index]
            polynomial += term * expanded_theta ** ((level_index + 2.0) / 2.0)
        return polynomial

    short = evaluate(constants.hminus_freefree_short_coefficients)
    long = evaluate(constants.hminus_freefree_long_coefficients)
    polynomial = np.where(expanded_wavelength < transition, short, long)
    valid_temperature = (thermal >= floor) & (thermal <= ceiling)
    return np.where(
        valid_temperature[..., None, None],
        np.maximum(
            1.0e-29
            * constants.boltzmann_erg_per_K
            * thermal[..., None, None]
            * polynomial,
            0.0,
        ),
        0.0,
    )


def _h2plus_node_opacity(
    temperature: np.ndarray,
    node_frequency: np.ndarray,
    node_u: np.ndarray,
    hydrogen_neutral: np.ndarray,
    proton_density: np.ndarray,
    rho: np.ndarray,
    *,
    constants: TextbookOpacityConstants,
) -> np.ndarray:
    """Evaluate the Bates/ATLAS H2+ continuum at the Rosseland nodes."""

    frequency = np.asarray(node_frequency, dtype=np.float64)
    ln_frequency = np.log(np.maximum(frequency, 1.0e-300))
    frequency_1e15 = frequency / 1.0e15
    fr_coefficients = constants.h2plus_fr_polynomial
    fr_polynomial = (
        fr_coefficients[0]
        + (
            fr_coefficients[1]
            + (
                fr_coefficients[2]
                + (fr_coefficients[3] + fr_coefficients[4] * ln_frequency)
                * ln_frequency
            )
            * ln_frequency
        )
        * ln_frequency
    )
    excitation_coefficients = constants.h2plus_excitation_polynomial
    excitation_energy = (
        excitation_coefficients[0]
        + (
            excitation_coefficients[1]
            + (
                excitation_coefficients[2]
                + (
                    excitation_coefficients[3]
                    + (
                        excitation_coefficients[4]
                        + excitation_coefficients[5] * frequency_1e15
                    )
                    * frequency_1e15
                )
                * frequency_1e15
            )
            * frequency_1e15
        )
        * frequency_1e15
    )
    kT_eV = constants.boltzmann_eV_per_K * temperature
    stimulated = -np.expm1(-node_u)
    log_opacity = (
        -excitation_energy / kT_eV[..., None, None]
        + fr_polynomial
        + np.log(np.maximum(hydrogen_neutral[..., None, None], 1.0e-40))
        + np.log(np.maximum(proton_density[..., None, None], 1.0e-40))
        - np.log(np.maximum(rho[..., None, None], 1.0e-300))
        + np.log(np.maximum(stimulated, 1.0e-300))
    )
    opacity = np.exp(np.clip(log_opacity, -700.0, 700.0))
    active = (frequency >= float(constants.h2plus_frequency_floor_hz)) & (
        frequency <= float(constants.h2plus_frequency_ceiling_hz)
    )
    return np.where(active, np.maximum(opacity, 0.0), 0.0)


def _heminus_node_opacity(
    temperature: np.ndarray,
    node_frequency: np.ndarray,
    helium_neutral: np.ndarray,
    electron_density: np.ndarray,
    rho: np.ndarray,
    *,
    constants: TextbookOpacityConstants,
) -> np.ndarray:
    """Evaluate the Kurucz/ATLAS He-minus free-free continuum."""

    frequency = np.maximum(np.asarray(node_frequency, dtype=np.float64), 1.0e-300)
    floor = float(constants.heminus_frequency_floor_hz)
    safe_frequency = np.maximum(frequency, floor)
    a_coeff = constants.heminus_a_coefficients
    b_coeff = constants.heminus_b_coefficients
    c_coeff = constants.heminus_c_coefficients
    a_term = a_coeff[0] + (a_coeff[1] + a_coeff[2] / safe_frequency) / safe_frequency
    b_term = b_coeff[0] + (b_coeff[1] + b_coeff[2] / safe_frequency) / safe_frequency
    c_term = c_coeff[0] + (c_coeff[1] + c_coeff[2] / safe_frequency) / safe_frequency
    density_unit = float(constants.heminus_density_unit_cm3)
    opacity = (
        (
            a_term * temperature[..., None, None]
            + b_term
            + c_term / temperature[..., None, None]
        )
        * electron_density[..., None, None]
        * helium_neutral[..., None, None]
        / rho[..., None, None]
        / density_unit**3
    )
    return np.where(frequency >= floor, np.maximum(opacity, 0.0), 0.0)


def _textbook_opacity_node_components_from_state(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    state: dict[str, np.ndarray],
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    apply_stimulated_emission_to_hminus_freefree: bool,
) -> dict[str, np.ndarray]:
    """Evaluate the node continuum from an explicitly selected local state.

    The returned node arrays have shape ``(N, layers, 5, 32)``.  The only
    difference between the historical v4 and v4r1 synthesis is selected by the
    supplied state and the John H-minus free-free convention.
    """

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    rho = state["rho_g_cm3"]
    electron_density = state["electron_density_cm3"]
    neutral_hydrogen = state["hydrogen_neutral_density_cm3"]
    hydrogen_ionized = state["hydrogen_ionized_fraction"]
    for name, array in (
        ("rho_g_cm3", rho),
        ("electron_density_cm3", electron_density),
        ("hydrogen_neutral_density_cm3", neutral_hydrogen),
        ("hydrogen_ionized_fraction", hydrogen_ionized),
    ):
        if np.asarray(array).shape != thermal.shape:
            raise ValueError(f"state field {name} must match temperature")
    node_frequency, node_weights, node_u = rosseland_frequency_nodes(
        thermal, constants=constants
    )
    stimulated = -np.expm1(-node_u)

    kT_eV = constants.boltzmann_eV_per_K * thermal
    hminus_saha_volume = (
        constants.planck_erg_s**2
        / (
            2.0
            * np.pi
            * constants.electron_mass_g
            * constants.boltzmann_erg_per_K
            * thermal
        )
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

    wavelength_um = (
        constants.speed_of_light_cm_s
        / np.maximum(node_frequency, 1.0e-300)
        / 1.0e-4
    )
    hminus_bf_cross_section = _john_hminus_boundfree_cross_section(
        wavelength_um, constants=constants
    )
    hminus_boundfree = (
        hminus_density[..., None, None]
        / rho[..., None, None]
        * hminus_bf_cross_section
        * stimulated
    )
    hminus_ff_cross_section = _john_hminus_freefree_cross_section(
        thermal, wavelength_um, constants=constants
    )
    hminus_freefree = (
        neutral_hydrogen[..., None, None]
        * electron_density[..., None, None]
        / rho[..., None, None]
        * hminus_ff_cross_section
    )
    if apply_stimulated_emission_to_hminus_freefree:
        hminus_freefree = hminus_freefree * stimulated

    level_count = int(constants.hydrogen_boundfree_level_count)
    if level_count != 10:
        raise ValueError("v4 requires exactly ten hydrogen bound-free levels")
    levels = np.arange(1, level_count + 1, dtype=np.float64)
    excitation_eV = constants.hydrogen_ionization_eV * (1.0 - levels ** -2.0)
    statistical_weight = 2.0 * levels**2
    level_weight = statistical_weight[None, None, :] * np.exp(
        np.clip(
            -excitation_eV[None, None, :] / kT_eV[..., None],
            -700.0,
            0.0,
        )
    )
    level_partition = np.sum(level_weight, axis=-1, keepdims=True)
    level_population = (
        neutral_hydrogen[..., None]
        * level_weight
        / np.maximum(level_partition, 1.0e-300)
    )
    hydrogen_boundfree = np.zeros_like(node_frequency)
    hydrogen_freefree = np.zeros_like(node_frequency)
    for level_index, principal_quantum_number in enumerate(levels):
        threshold_frequency = (
            constants.hydrogen_ionization_eV
            / principal_quantum_number**2
            * constants.eV_to_erg
            / constants.planck_erg_s
        )
        edge_cross_section = constants.hydrogen_ground_edge_cross_section_cm2 * (
            principal_quantum_number
            ** float(constants.hydrogen_boundfree_edge_cross_section_power)
        )
        cross_section = edge_cross_section * (
            threshold_frequency / np.maximum(node_frequency, 1.0e-300)
        ) ** 3
        cross_section = np.where(node_frequency >= threshold_frequency, cross_section, 0.0)
        hydrogen_boundfree += (
            level_population[..., level_index, None, None]
            / rho[..., None, None]
            * cross_section
            * stimulated
        )

    proton_density = (
        constants.hydrogen_mass_fraction
        * rho
        / constants.hydrogen_mass_g
        * hydrogen_ionized
    )
    hydrogen_freefree = (
        constants.hydrogen_freefree_coefficient
        * electron_density[..., None, None]
        * proton_density[..., None, None]
        / rho[..., None, None]
        / np.sqrt(thermal[..., None, None])
        / np.maximum(node_frequency, 1.0e-300) ** 3
        * stimulated
    )

    electron_scattering = (
        constants.thomson_cross_section_cm2
        * electron_density[..., None, None]
        / rho[..., None, None]
    ) * np.ones_like(node_frequency)
    reference_frequency = constants.speed_of_light_cm_s / (500.0e-7)
    rayleigh_cross_section = constants.hydrogen_rayleigh_cross_section_at_500nm_cm2 * (
        node_frequency / reference_frequency
    ) ** 4
    hydrogen_rayleigh = (
        neutral_hydrogen[..., None, None]
        / rho[..., None, None]
        * rayleigh_cross_section
    )
    components = {
        "hminus_boundfree": np.maximum(hminus_boundfree, 0.0),
        "hminus_freefree": np.maximum(hminus_freefree, 0.0),
        "hydrogen_boundfree": np.maximum(hydrogen_boundfree, 0.0),
        "hydrogen_freefree": np.maximum(hydrogen_freefree, 0.0),
        "electron_scattering": np.maximum(electron_scattering, 0.0),
        "hydrogen_rayleigh_scattering": np.maximum(hydrogen_rayleigh, 0.0),
        "frequency_nodes_hz": node_frequency,
        "frequency_nodes_u": node_u,
        "node_weights": node_weights,
    }
    total = sum(
        components[name]
        for name in (
            "hminus_boundfree",
            "hminus_freefree",
            "hydrogen_boundfree",
            "hydrogen_freefree",
            "electron_scattering",
            "hydrogen_rayleigh_scattering",
        )
    )
    components["total"] = np.maximum(total, 1.0e-30)
    if np.any(~np.isfinite(components["total"])):
        raise ValueError("node opacity synthesis produced non-finite values")
    return components


def textbook_opacity_node_components(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> dict[str, np.ndarray]:
    """Evaluate the historical v4 continuum without changing its closure."""

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    state = saha_electron_diagnostics(
        values, thermal, pressure, constants=constants
    )
    return _textbook_opacity_node_components_from_state(
        values,
        thermal,
        pressure,
        state,
        constants=constants,
        apply_stimulated_emission_to_hminus_freefree=True,
    )


def textbook_opacity_node_components_v4r1(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> dict[str, np.ndarray]:
    """Evaluate v4r1 using the seven-donor closure and John free-free units.

    John H-minus free-free is a coefficient per neutral hydrogen atom and
    electron pressure.  ``_john_hminus_freefree_cross_section`` already
    includes ``k_B T``, so this branch is ``k_ff n_H n_e k_B T / rho`` and
    receives no second stimulated-emission factor.
    """

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    state = saha_electron_diagnostics_v4r1(
        values, thermal, pressure, constants=constants
    )
    return _textbook_opacity_node_components_from_state(
        values,
        thermal,
        pressure,
        state,
        constants=constants,
        apply_stimulated_emission_to_hminus_freefree=False,
    )


def _textbook_opacity_node_components_v4r1_with_electron_density_oracle(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> dict[str, np.ndarray]:
    """Diagnostic-only v4r1 synthesis with externally supplied n_e."""

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    state = _local_state_from_electron_density(
        values,
        thermal,
        pressure,
        electron_density,
        constants=constants,
    )
    return _textbook_opacity_node_components_from_state(
        values,
        thermal,
        pressure,
        state,
        constants=constants,
        apply_stimulated_emission_to_hminus_freefree=False,
    )


def textbook_rosseland_opacity_v4(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Evaluate the exact node-level Rosseland harmonic mean for v4."""

    components = textbook_opacity_node_components(
        labels, temperature, gas_pressure, constants=constants
    )
    weights = components["node_weights"]
    total = components["total"]
    result = 1.0 / np.sum(weights / np.maximum(total, 1.0e-30), axis=(-2, -1))
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("v4 Rosseland opacity is non-finite or non-positive")
    return result


def textbook_rosseland_opacity_v4r1(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Evaluate the exact node-level Rosseland harmonic mean for v4r1."""

    components = textbook_opacity_node_components_v4r1(
        labels, temperature, gas_pressure, constants=constants
    )
    weights = components["node_weights"]
    total = components["total"]
    result = 1.0 / np.sum(weights / np.maximum(total, 1.0e-30), axis=(-2, -1))
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("v4r1 Rosseland opacity is non-finite or non-positive")
    return result


def textbook_opacity_node_components_v4r3(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> dict[str, np.ndarray]:
    """Evaluate v4r3: particle-count density plus H2+ and He-minus continua."""

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    state = saha_electron_diagnostics_v4r3(
        values, thermal, pressure, constants=constants
    )
    components = _textbook_opacity_node_components_from_state(
        values,
        thermal,
        pressure,
        state,
        constants=constants,
        apply_stimulated_emission_to_hminus_freefree=False,
    )
    proton_density = (
        state["hydrogen_number_density_cm3"] * state["hydrogen_ionized_fraction"]
    )
    h2plus = _h2plus_node_opacity(
        thermal,
        components["frequency_nodes_hz"],
        components["frequency_nodes_u"],
        state["hydrogen_neutral_density_cm3"],
        proton_density,
        state["rho_g_cm3"],
        constants=constants,
    )
    heminus = _heminus_node_opacity(
        thermal,
        components["frequency_nodes_hz"],
        state["helium_neutral_density_cm3"],
        state["electron_density_cm3"],
        state["rho_g_cm3"],
        constants=constants,
    )
    components["h2plus"] = h2plus
    components["heminus"] = heminus
    components["mean_molecular_weight"] = state["mean_molecular_weight"]
    components["electron_density_cm3"] = state["electron_density_cm3"]
    components["total"] = np.maximum(components["total"] + h2plus + heminus, 1.0e-30)
    if np.any(~np.isfinite(components["total"])):
        raise ValueError("v4r3 node opacity synthesis produced non-finite values")
    return components


def textbook_rosseland_opacity_v4r3(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Evaluate the exact node-level Rosseland harmonic mean for v4r3."""

    components = textbook_opacity_node_components_v4r3(
        labels, temperature, gas_pressure, constants=constants
    )
    weights = components["node_weights"]
    total = components["total"]
    result = 1.0 / np.sum(weights / np.maximum(total, 1.0e-30), axis=(-2, -1))
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("v4r3 Rosseland opacity is non-finite or non-positive")
    return result


def _helium_stage_densities_from_ne(
    temperature: np.ndarray,
    electron_density: np.ndarray,
    hydrogen_number_density: np.ndarray,
    *,
    constants: TextbookOpacityConstants,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close two-step hydrogenic helium stages on a frozen electron density.

    Helium is not added to the Saha charge balance.  ``n_e`` is an input,
    taken from the frozen v4r3 state.  The partition ratios are the
    ground-term factors ``2 U_{r+1}/U_r``; ``_log_saha_ratio`` does not
    include them, so they are applied the same way v4r3 metal donors are.
    """

    thermal = np.asarray(temperature, dtype=np.float64)
    n_e = np.asarray(electron_density, dtype=np.float64)
    n_hydrogen = np.asarray(hydrogen_number_density, dtype=np.float64)
    ratio_i_to_ii = float(constants.helium_i_to_ii_saha_partition_ratio)
    ratio_ii_to_iii = float(constants.helium_ii_to_iii_saha_partition_ratio)
    if not np.isfinite(ratio_i_to_ii) or ratio_i_to_ii <= 0.0:
        raise ValueError("invalid helium I to II Saha partition ratio")
    if not np.isfinite(ratio_ii_to_iii) or ratio_ii_to_iii <= 0.0:
        raise ValueError("invalid helium II to III Saha partition ratio")
    n_helium = n_hydrogen * (
        constants.helium_mass_fraction
        / (4.0 * constants.hydrogen_mass_fraction)
    )
    log_phi_1 = _log_saha_ratio(
        thermal,
        n_e,
        constants.helium_first_ionization_eV,
        constants,
    ) + np.log(ratio_i_to_ii)
    log_phi_2 = _log_saha_ratio(
        thermal,
        n_e,
        constants.helium_second_ionization_eV,
        constants,
    ) + np.log(ratio_ii_to_iii)
    phi_1 = np.exp(np.clip(log_phi_1, -700.0, 700.0))
    phi_2 = np.exp(np.clip(log_phi_2, -700.0, 700.0))
    denom = np.maximum(1.0 + phi_1 + phi_1 * phi_2, 1.0e-300)
    n_i = n_helium / denom
    n_ii = n_helium * phi_1 / denom
    n_iii = n_helium * phi_1 * phi_2 / denom
    return (
        np.maximum(n_i, 0.0),
        np.maximum(n_ii, 0.0),
        np.maximum(n_iii, 0.0),
    )


def _helium_ionized_node_opacity(
    temperature: np.ndarray,
    node_frequency: np.ndarray,
    node_u: np.ndarray,
    helium_ii_density: np.ndarray,
    helium_iii_density: np.ndarray,
    electron_density: np.ndarray,
    rho: np.ndarray,
    *,
    constants: TextbookOpacityConstants,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate hydrogenic He II bound-free and He III free-free at the nodes.

    The bound-free branch is the v4 H I node law at ``Z=2``: ionization
    energy 54.418 eV, ground-edge cross section scaled by ``1/Z^2``, the
    same ``n^2`` threshold scaling, ten levels, and stimulated emission
    ``1-exp(-u)``.  Free-free uses coefficient ``Z^2`` times the hydrogen
    Kramers coefficient with ``n_e n(He III)``.  No Gaunt tables are loaded.
    """

    thermal = np.asarray(temperature, dtype=np.float64)
    frequency = np.asarray(node_frequency, dtype=np.float64)
    stimulated = -np.expm1(-np.asarray(node_u, dtype=np.float64))
    n_ii = np.asarray(helium_ii_density, dtype=np.float64)
    n_iii = np.asarray(helium_iii_density, dtype=np.float64)
    n_e = np.asarray(electron_density, dtype=np.float64)
    mass_density = np.asarray(rho, dtype=np.float64)
    effective_charge = float(constants.helium_ionized_effective_charge)
    if not np.isfinite(effective_charge) or effective_charge <= 0.0:
        raise ValueError("invalid helium ionized effective charge")
    level_count = int(constants.helium_ionized_boundfree_level_count)
    if level_count != 10:
        raise ValueError("v4r4 requires exactly ten He II bound-free levels")
    kT_eV = constants.boltzmann_eV_per_K * thermal
    ionization_eV = float(constants.helium_second_ionization_eV)
    levels = np.arange(1, level_count + 1, dtype=np.float64)
    excitation_eV = ionization_eV * (1.0 - levels ** -2.0)
    statistical_weight = 2.0 * levels**2
    level_weight = statistical_weight[None, None, :] * np.exp(
        np.clip(
            -excitation_eV[None, None, :] / kT_eV[..., None],
            -700.0,
            0.0,
        )
    )
    level_partition = np.sum(level_weight, axis=-1, keepdims=True)
    level_population = (
        n_ii[..., None]
        * level_weight
        / np.maximum(level_partition, 1.0e-300)
    )
    ground_edge_cross_section = (
        constants.hydrogen_ground_edge_cross_section_cm2
        / effective_charge**2
    )
    boundfree = np.zeros_like(frequency)
    for level_index, principal_quantum_number in enumerate(levels):
        threshold_frequency = (
            ionization_eV
            / principal_quantum_number**2
            * constants.eV_to_erg
            / constants.planck_erg_s
        )
        edge_cross_section = ground_edge_cross_section * (
            principal_quantum_number
            ** float(constants.hydrogen_boundfree_edge_cross_section_power)
        )
        cross_section = edge_cross_section * (
            threshold_frequency / np.maximum(frequency, 1.0e-300)
        ) ** 3
        cross_section = np.where(frequency >= threshold_frequency, cross_section, 0.0)
        boundfree += (
            level_population[..., level_index, None, None]
            / mass_density[..., None, None]
            * cross_section
            * stimulated
        )
    freefree = (
        (effective_charge**2)
        * constants.hydrogen_freefree_coefficient
        * n_e[..., None, None]
        * n_iii[..., None, None]
        / mass_density[..., None, None]
        / np.sqrt(thermal[..., None, None])
        / np.maximum(frequency, 1.0e-300) ** 3
        * stimulated
    )
    return np.maximum(boundfree, 0.0), np.maximum(freefree, 0.0)


def textbook_opacity_node_components_v4r4(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> dict[str, np.ndarray]:
    """Evaluate v4r4: frozen v4r3 continua plus hydrogenic He II.

    Helium remains out of the Saha charge balance.  He-minus still uses
    total helium.  The returned ``electron_density_cm3`` is identical to
    the v4r3 value.
    """

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    components = textbook_opacity_node_components_v4r3(
        values, thermal, pressure, constants=constants
    )
    n_tot = pressure / (constants.boltzmann_erg_per_K * thermal)
    rho = (
        components["mean_molecular_weight"]
        * n_tot
        * constants.hydrogen_mass_g
    )
    n_hydrogen = (
        rho * constants.hydrogen_mass_fraction / constants.hydrogen_mass_g
    )
    electron_density = components["electron_density_cm3"]
    n_i, n_ii, n_iii = _helium_stage_densities_from_ne(
        thermal,
        electron_density,
        n_hydrogen,
        constants=constants,
    )
    boundfree, freefree = _helium_ionized_node_opacity(
        thermal,
        components["frequency_nodes_hz"],
        components["frequency_nodes_u"],
        n_ii,
        n_iii,
        electron_density,
        rho,
        constants=constants,
    )
    components["helium_ionized_boundfree"] = boundfree
    components["helium_ionized_freefree"] = freefree
    components["helium_i_density_cm3"] = n_i
    components["helium_ii_density_cm3"] = n_ii
    components["helium_iii_density_cm3"] = n_iii
    components["total"] = np.maximum(components["total"] + boundfree + freefree, 1.0e-30)
    if np.any(~np.isfinite(components["total"])):
        raise ValueError("v4r4 node opacity synthesis produced non-finite values")
    return components


def textbook_rosseland_opacity_v4r4(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Evaluate the exact node-level Rosseland harmonic mean for v4r4."""

    components = textbook_opacity_node_components_v4r4(
        labels, temperature, gas_pressure, constants=constants
    )
    weights = components["node_weights"]
    total = components["total"]
    result = 1.0 / np.sum(weights / np.maximum(total, 1.0e-30), axis=(-2, -1))
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("v4r4 Rosseland opacity is non-finite or non-positive")
    return result


def _hydrogen_ground_anchored_level_populations(
    temperature: np.ndarray,
    neutral_hydrogen: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Boltzmann excitation from the ground, without a 10-level partition.

    The construction is ``n_1 = n(H I)`` and
    ``n_n = n(H I) * n^2 * exp(-chi_H (1 - 1/n^2) / kT)``.  High-n shells
    are not a closed bound reservoir, so the sum over n may exceed
    ``n(H I)``.
    """

    level_count = int(constants.hydrogen_boundfree_level_count)
    if level_count != 10:
        raise ValueError("v4r5 requires exactly ten hydrogen bound-free levels")
    levels = np.arange(1, level_count + 1, dtype=np.float64)
    kT_eV = constants.boltzmann_eV_per_K * np.asarray(temperature, dtype=np.float64)
    excitation_eV = constants.hydrogen_ionization_eV * (1.0 - levels ** -2.0)
    relative_weight = levels**2 * np.exp(
        np.clip(-excitation_eV / kT_eV[..., None], -700.0, 0.0)
    )
    return np.asarray(neutral_hydrogen, dtype=np.float64)[..., None] * relative_weight


def _hydrogen_boundfree_node_opacity_from_level_populations(
    node_frequency: np.ndarray,
    stimulated: np.ndarray,
    level_population: np.ndarray,
    rho: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    edge_cross_section_cm2: np.ndarray | None = None,
) -> np.ndarray:
    """Hydrogenic bound-free opacity from an explicit level population."""

    levels = np.arange(
        1, int(constants.hydrogen_boundfree_level_count) + 1, dtype=np.float64
    )
    if edge_cross_section_cm2 is None:
        edge_values = constants.hydrogen_ground_edge_cross_section_cm2 * (
            levels ** float(constants.hydrogen_boundfree_edge_cross_section_power)
        )
    else:
        edge_values = np.asarray(edge_cross_section_cm2, dtype=np.float64)
        if edge_values.shape != levels.shape:
            raise ValueError("edge_cross_section_cm2 must match the bound-free level count")
    hydrogen_boundfree = np.zeros_like(node_frequency)
    for level_index, principal_quantum_number in enumerate(levels):
        threshold_frequency = (
            constants.hydrogen_ionization_eV
            / principal_quantum_number**2
            * constants.eV_to_erg
            / constants.planck_erg_s
        )
        edge_cross_section = float(edge_values[level_index])
        cross_section = edge_cross_section * (
            threshold_frequency / np.maximum(node_frequency, 1.0e-300)
        ) ** 3
        cross_section = np.where(
            node_frequency >= threshold_frequency, cross_section, 0.0
        )
        hydrogen_boundfree += (
            level_population[..., level_index, None, None]
            / rho[..., None, None]
            * cross_section
            * stimulated
        )
    return hydrogen_boundfree


def textbook_opacity_node_components_v4r5(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> dict[str, np.ndarray]:
    """Evaluate v4r5: frozen v4r3 continua with ground-anchored H I bf.

    Helium remains out of the Saha charge balance.  He II from v4r4 is not
    added.  The n^2 hydrogenic edge law is unchanged.  Electron density and
    mean molecular weight are identical to v4r3.
    """

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    components = textbook_opacity_node_components_v4r3(
        values, thermal, pressure, constants=constants
    )
    state = saha_electron_diagnostics_v4r3(
        values, thermal, pressure, constants=constants
    )
    populations = _hydrogen_ground_anchored_level_populations(
        thermal,
        state["hydrogen_neutral_density_cm3"],
        constants=constants,
    )
    stimulated = -np.expm1(-components["frequency_nodes_u"])
    new_boundfree = _hydrogen_boundfree_node_opacity_from_level_populations(
        components["frequency_nodes_hz"],
        stimulated,
        populations,
        state["rho_g_cm3"],
        constants=constants,
    )
    old_boundfree = components["hydrogen_boundfree"]
    components["hydrogen_boundfree"] = new_boundfree
    components["hydrogen_ground_density_cm3"] = populations[..., 0]
    components["hydrogen_bound_level_population_sum_cm3"] = np.sum(
        populations, axis=-1
    )
    components["total"] = np.maximum(
        components["total"] - old_boundfree + new_boundfree,
        1.0e-30,
    )
    if np.any(~np.isfinite(components["total"])):
        raise ValueError("v4r5 node opacity synthesis produced non-finite values")
    return components


def textbook_rosseland_opacity_v4r5(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Evaluate the exact node-level Rosseland harmonic mean for v4r5."""

    components = textbook_opacity_node_components_v4r5(
        labels, temperature, gas_pressure, constants=constants
    )
    weights = components["node_weights"]
    total = components["total"]
    result = 1.0 / np.sum(weights / np.maximum(total, 1.0e-30), axis=(-2, -1))
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("v4r5 Rosseland opacity is non-finite or non-positive")
    return result


def _hydrogen_boundfree_edge_cross_sections_v4r6(
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Return per-level threshold cross sections for the v4r6 law.

    n=1..3 use published thresholds.  n>=4 keep the v4 n^2 Kramers edge.
    """

    level_count = int(constants.hydrogen_boundfree_level_count)
    levels = np.arange(1, level_count + 1, dtype=np.float64)
    edges = constants.hydrogen_ground_edge_cross_section_cm2 * (
        levels ** float(constants.hydrogen_boundfree_edge_cross_section_power)
    )
    published = np.asarray(
        constants.hydrogen_published_threshold_cross_section_cm2, dtype=np.float64
    )
    if published.size < 3:
        raise ValueError("v4r6 requires published n=1,2,3 threshold cross sections")
    edges[: published.size] = published
    return edges


def textbook_opacity_node_components_v4r6(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> dict[str, np.ndarray]:
    """Evaluate v4r6: v4r5 continua with per-n H I edges below 15000 K.

    Helium remains out of the Saha charge balance.  John H-minus is unchanged.
    At ``T >= 15000 K`` the hydrogen bound-free is identical to v4r5, so the
    ground-anchored Lyman repair is frozen.  Below that ceiling, n=2 and n=3
    use published threshold cross sections instead of n^2 * sigma_1.
    """

    values, thermal, pressure = _as_profile_inputs(labels, temperature, gas_pressure)
    components = textbook_opacity_node_components_v4r5(
        values, thermal, pressure, constants=constants
    )
    state = saha_electron_diagnostics_v4r3(
        values, thermal, pressure, constants=constants
    )
    populations = _hydrogen_ground_anchored_level_populations(
        thermal,
        state["hydrogen_neutral_density_cm3"],
        constants=constants,
    )
    stimulated = -np.expm1(-components["frequency_nodes_u"])
    new_boundfree = _hydrogen_boundfree_node_opacity_from_level_populations(
        components["frequency_nodes_hz"],
        stimulated,
        populations,
        state["rho_g_cm3"],
        constants=constants,
        edge_cross_section_cm2=_hydrogen_boundfree_edge_cross_sections_v4r6(
            constants=constants
        ),
    )
    old_boundfree = components["hydrogen_boundfree"]
    cool_enough = thermal < float(V4R6_PER_N_TEMPERATURE_CEILING_K)
    blended = np.where(cool_enough[..., None, None], new_boundfree, old_boundfree)
    components["hydrogen_boundfree"] = blended
    components["total"] = np.maximum(
        components["total"] - old_boundfree + blended,
        1.0e-30,
    )
    if np.any(~np.isfinite(components["total"])):
        raise ValueError("v4r6 node opacity synthesis produced non-finite values")
    return components


def textbook_rosseland_opacity_v4r6(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
) -> np.ndarray:
    """Evaluate the exact node-level Rosseland harmonic mean for v4r6."""

    components = textbook_opacity_node_components_v4r6(
        labels, temperature, gas_pressure, constants=constants
    )
    weights = components["node_weights"]
    total = components["total"]
    result = 1.0 / np.sum(weights / np.maximum(total, 1.0e-30), axis=(-2, -1))
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("v4r6 Rosseland opacity is non-finite or non-positive")
    return result


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


def _batched_log_mass_rhs(
    log_tau: float,
    log_mass: np.ndarray,
    labels: np.ndarray,
    log_tau_grid: np.ndarray,
    log_temperature: np.ndarray,
    gravity: np.ndarray,
    constants: TextbookOpacityConstants,
    opacity_function: Callable[..., np.ndarray],
) -> np.ndarray:
    """Evaluate ``d log m / d log tau`` for every star at one optical depth."""

    local_log_temperature = np.empty(log_mass.shape[0], dtype=np.float64)
    for star in range(log_mass.shape[0]):
        local_log_temperature[star] = np.interp(
            log_tau, log_tau_grid, log_temperature[star]
        )
    local_temperature = np.exp(local_log_temperature)
    mass = np.exp(np.clip(log_mass, -700.0, 700.0))
    pressure = np.maximum(gravity * mass, 1.0e-30)
    opacity = opacity_function(
        labels,
        local_temperature[:, None],
        pressure[:, None],
        constants=constants,
    )[:, 0]
    rhs = np.exp(
        np.clip(log_tau - log_mass - np.log(opacity), -100.0, 100.0)
    )
    return np.clip(rhs, 0.0, 1.0e6)


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

    star_count = values.shape[0]
    output = np.empty_like(thermal)
    log_tau_grid = np.log(depth)
    log_temperature = np.log(thermal)
    gravity = 10.0 ** values[:, 1]
    # The named 0.34 value is only the initial surface guess.  A few
    # damped local updates enforce the stated anchor m0=tau0/kappa0 while
    # supplying the pressure needed by the local opacity law.
    log_mass = np.full(
        star_count,
        np.log(
            max(
                depth[0] / constants.surface_anchor_opacity_cm2_per_g,
                1.0e-300,
            )
        ),
        dtype=np.float64,
    )
    for _ in range(8):
        surface_mass = np.exp(np.clip(log_mass, -700.0, 700.0))
        surface_pressure = np.maximum(gravity * surface_mass, 1.0e-30)
        surface_opacity = opacity_function(
            values,
            thermal[:, :1],
            surface_pressure[:, None],
            constants=constants,
        )[:, 0]
        target_log_mass = np.log(
            np.maximum(depth[0] / np.maximum(surface_opacity, 1.0e-30), 1.0e-300)
        )
        log_mass = 0.5 * (log_mass + target_log_mass)
    output[:, 0] = np.exp(log_mass)
    for layer in range(1, depth.size):
        left = float(log_tau_grid[layer - 1])
        right = float(log_tau_grid[layer])
        width = (right - left) / float(substeps_per_layer)
        for substep in range(substeps_per_layer):
            x = left + substep * width
            k1 = _batched_log_mass_rhs(
                x,
                log_mass,
                values,
                log_tau_grid,
                log_temperature,
                gravity,
                constants,
                opacity_function,
            )
            k2 = _batched_log_mass_rhs(
                x + 0.5 * width,
                log_mass + 0.5 * width * k1,
                values,
                log_tau_grid,
                log_temperature,
                gravity,
                constants,
                opacity_function,
            )
            k3 = _batched_log_mass_rhs(
                x + 0.5 * width,
                log_mass + 0.5 * width * k2,
                values,
                log_tau_grid,
                log_temperature,
                gravity,
                constants,
                opacity_function,
            )
            k4 = _batched_log_mass_rhs(
                x + width,
                log_mass + width * k3,
                values,
                log_tau_grid,
                log_temperature,
                gravity,
                constants,
                opacity_function,
            )
            log_mass = log_mass + width * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            log_mass = np.clip(log_mass, -700.0, 700.0)
        output[:, layer] = np.exp(log_mass)
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


def build_textbook_reduced_state_v4(
    labels: np.ndarray,
    tau: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    include_convection: bool = True,
    substeps_per_layer: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build the v4 grey/Hopf plus node-level opacity ODE seed."""

    return build_textbook_reduced_state(
        labels,
        tau,
        constants=constants,
        include_convection=include_convection,
        substeps_per_layer=substeps_per_layer,
        opacity_function=textbook_rosseland_opacity_v4,
    )


def build_textbook_reduced_state_v4r3(
    labels: np.ndarray,
    tau: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    include_convection: bool = True,
    substeps_per_layer: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build the v4r3 grey-plus-adiabatic seed with the particle-count ODE."""

    return build_textbook_reduced_state(
        labels,
        tau,
        constants=constants,
        include_convection=include_convection,
        substeps_per_layer=substeps_per_layer,
        opacity_function=textbook_rosseland_opacity_v4r3,
    )


def predict_textbook_reduced_state_v4r3(
    labels: np.ndarray,
    tau: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    include_convection: bool = True,
    substeps_per_layer: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(m, T, log10 kappa_R)`` for the v4r3 solver funnel arm.

    A batch failure falls back to one star at a time so a single non-finite
    photosphere cannot discard the rest of a development-60 draw.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    depth = np.asarray(tau, dtype=np.float64)
    try:
        mass, temperature, diagnostics = build_textbook_reduced_state_v4r3(
            values,
            depth,
            constants=constants,
            include_convection=include_convection,
            substeps_per_layer=substeps_per_layer,
        )
        log_opacity = np.log10(np.maximum(diagnostics["rosseland_opacity"], 1.0e-30))
        return mass, temperature, log_opacity
    except (ValueError, FloatingPointError):
        if values.shape[0] == 1:
            raise
        mass = np.full((values.shape[0], depth.size), np.nan, dtype=np.float64)
        temperature = np.full_like(mass, np.nan)
        log_opacity = np.full_like(mass, np.nan)
        for star in range(values.shape[0]):
            try:
                star_mass, star_temperature, star_log_opacity = (
                    predict_textbook_reduced_state_v4r3(
                        values[star],
                        depth,
                        constants=constants,
                        include_convection=include_convection,
                        substeps_per_layer=substeps_per_layer,
                    )
                )
            except (ValueError, FloatingPointError):
                continue
            mass[star] = star_mass[0]
            temperature[star] = star_temperature[0]
            log_opacity[star] = star_log_opacity[0]
        return mass, temperature, log_opacity


def build_textbook_reduced_state_v4r6(
    labels: np.ndarray,
    tau: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    include_convection: bool = True,
    substeps_per_layer: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build the v4r6 grey-plus-adiabatic seed with the particle-count ODE."""

    return build_textbook_reduced_state(
        labels,
        tau,
        constants=constants,
        include_convection=include_convection,
        substeps_per_layer=substeps_per_layer,
        opacity_function=textbook_rosseland_opacity_v4r6,
    )


def predict_textbook_reduced_state_v4r6(
    labels: np.ndarray,
    tau: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    include_convection: bool = True,
    substeps_per_layer: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(m, T, log10 kappa_R)`` for the v4r6 solver funnel arm.

    A batch failure falls back to one star at a time so a single non-finite
    photosphere cannot discard the rest of a development-60 draw.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    depth = np.asarray(tau, dtype=np.float64)
    try:
        mass, temperature, diagnostics = build_textbook_reduced_state_v4r6(
            values,
            depth,
            constants=constants,
            include_convection=include_convection,
            substeps_per_layer=substeps_per_layer,
        )
        log_opacity = np.log10(np.maximum(diagnostics["rosseland_opacity"], 1.0e-30))
        return mass, temperature, log_opacity
    except (ValueError, FloatingPointError):
        if values.shape[0] == 1:
            raise
        mass = np.full((values.shape[0], depth.size), np.nan, dtype=np.float64)
        temperature = np.full_like(mass, np.nan)
        log_opacity = np.full_like(mass, np.nan)
        for star in range(values.shape[0]):
            try:
                star_mass, star_temperature, star_log_opacity = (
                    predict_textbook_reduced_state_v4r6(
                        values[star],
                        depth,
                        constants=constants,
                        include_convection=include_convection,
                        substeps_per_layer=substeps_per_layer,
                    )
                )
            except (ValueError, FloatingPointError):
                continue
            mass[star] = star_mass[0]
            temperature[star] = star_temperature[0]
            log_opacity[star] = star_log_opacity[0]
        return mass, temperature, log_opacity


def _apply_saha_aware_convective_temperature(
    temperature: np.ndarray,
    pressure: np.ndarray,
    nabla_ad: np.ndarray,
    convective_mask: np.ndarray,
) -> np.ndarray:
    """Reproduce the registered convective *T* replacement on a copy of *T*."""

    replaced = np.array(temperature, copy=True, dtype=np.float64)
    for star in range(replaced.shape[0]):
        active = np.flatnonzero(convective_mask[star])
        if active.size == 0:
            continue
        switch = int(active[0])
        if switch <= 0:
            continue
        for layer in range(switch + 1, replaced.shape[1]):
            pressure_ratio = max(pressure[star, layer] / pressure[star, layer - 1], 1.0)
            gradient = float(nabla_ad[star, layer])
            replaced[star, layer] = replaced[star, layer - 1] * pressure_ratio**gradient
    return replaced


def build_textbook_reduced_state_v4r6_decoupled(
    labels: np.ndarray,
    tau: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    substeps_per_layer: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build the decoupled v4r6 seed: grey mass, convective temperature.

    Column mass is the grey v4r6 integral and is not re-integrated after the
    convective temperature replacement. Pressure stays ``P = g m_grey``.
    Opacity is recomputed at ``(T_conv, P_grey)``.
    """

    mass_grey, temperature_grey, diagnostics_grey = build_textbook_reduced_state(
        labels,
        tau,
        constants=constants,
        include_convection=False,
        substeps_per_layer=substeps_per_layer,
        opacity_function=textbook_rosseland_opacity_v4r6,
    )
    pressure_grey = np.asarray(diagnostics_grey["gas_pressure"], dtype=np.float64)
    temperature = _apply_saha_aware_convective_temperature(
        temperature_grey,
        pressure_grey,
        np.asarray(diagnostics_grey["nabla_ad"], dtype=np.float64),
        np.asarray(diagnostics_grey["convective_mask"]),
    )
    mass = np.array(mass_grey, copy=True, dtype=np.float64)
    opacity = textbook_rosseland_opacity_v4r6(
        labels, temperature, pressure_grey, constants=constants
    )
    diagnostics = {
        "rosseland_opacity": opacity,
        "gas_pressure": pressure_grey,
        "nabla_rad": diagnostics_grey["nabla_rad"],
        "nabla_ad": diagnostics_grey["nabla_ad"],
        "convective_mask": diagnostics_grey["convective_mask"],
        "mass_reintegrated_after_convection": False,
    }
    return mass, temperature, diagnostics


def predict_textbook_reduced_state_v4r6_decoupled(
    labels: np.ndarray,
    tau: np.ndarray,
    *,
    constants: TextbookOpacityConstants = DEFAULT_TEXTBOOK_CONSTANTS,
    substeps_per_layer: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(m_grey, T_conv, log10 kappa_R)`` for the decoupled funnel arm.

    A batch failure falls back to one star at a time so a single non-finite
    photosphere cannot discard the rest of a development-60 draw.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    depth = np.asarray(tau, dtype=np.float64)
    try:
        mass, temperature, diagnostics = build_textbook_reduced_state_v4r6_decoupled(
            values,
            depth,
            constants=constants,
            substeps_per_layer=substeps_per_layer,
        )
        log_opacity = np.log10(np.maximum(diagnostics["rosseland_opacity"], 1.0e-30))
        return mass, temperature, log_opacity
    except (ValueError, FloatingPointError):
        if values.shape[0] == 1:
            raise
        mass = np.full((values.shape[0], depth.size), np.nan, dtype=np.float64)
        temperature = np.full_like(mass, np.nan)
        log_opacity = np.full_like(mass, np.nan)
        for star in range(values.shape[0]):
            try:
                star_mass, star_temperature, star_log_opacity = (
                    predict_textbook_reduced_state_v4r6_decoupled(
                        values[star],
                        depth,
                        constants=constants,
                        substeps_per_layer=substeps_per_layer,
                    )
                )
            except (ValueError, FloatingPointError):
                continue
            mass[star] = star_mass[0]
            temperature[star] = star_temperature[0]
            log_opacity[star] = star_log_opacity[0]
        return mass, temperature, log_opacity
