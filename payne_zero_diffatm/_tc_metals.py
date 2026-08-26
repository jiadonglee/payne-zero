"""Neutral-metal continuum branches (C/Mg/Al/Si/Fe I, flag 8 atomic part)."""

from __future__ import annotations

import numpy as np
import torch

from payne_zero_diffatm.twin_continuum import (
    BOLTZMANN_ERG_PER_K_EXACT,
    LIGHT_SPEED_CM_PER_S_EXACT,
    PLANCK_ERG_SECOND_EXACT,
    TwinContinuumState,
    TwinContinuumTables,
    _karzas_latter_cross_section_grid,
    _planck_frequency_exact,
)


# Module-level literal tables, verbatim from continuum_opacity.py.
_CARBON_LEVEL_ENERGY_CM = (
    79314.86,
    78731.27,
    78529.62,
    78309.76,
    78226.35,
    77679.82,
    73975.91,
    72610.72,
    71374.90,
    70743.95,
    69722.00,
    68856.33,
    61981.82,
    60373.00,
    21648.01,
    10192.63,
    43.42,
    16.42,
    0.00,
    119878.0,
    105798.7,
    97878.0,
    75254.93,
    64088.85,
    33735.20,
)
_CARBON_STATISTICAL_WEIGHT = (
    9.0,
    3.0,
    7.0,
    15.0,
    21.0,
    5.0,
    1.0,
    5.0,
    9.0,
    3.0,
    15.0,
    3.0,
    3.0,
    9.0,
    1.0,
    5.0,
    5.0,
    3.0,
    1.0,
    3.0,
    3.0,
    5.0,
    12.0,
    15.0,
    5.0,
)
_MAGNESIUM_LEVEL_ENERGY_CM = (
    54676.710,
    54676.438,
    54192.284,
    53134.642,
    49346.729,
    47957.034,
    47847.797,
    46403.065,
    43503.333,
    41197.043,
    35051.264,
    21919.178,
    21870.464,
    21850.405,
    0.0,
)
_MAGNESIUM_STATISTICAL_WEIGHT = (
    21.0,
    7.0,
    15.0,
    5.0,
    3.0,
    15.0,
    9.0,
    5.0,
    1.0,
    3.0,
    3.0,
    5.0,
    3.0,
    1.0,
    1.0,
)
_SILICON_LEVEL_ENERGY_CM = (
    59962.284,
    59100.0,
    59077.112,
    58893.40,
    58801.529,
    58777.0,
    57488.974,
    56503.346,
    54225.621,
    53387.34,
    53362.24,
    51612.012,
    50533.424,
    50189.389,
    49965.894,
    49399.670,
    49128.131,
    48161.459,
    47351.554,
    47284.061,
    40991.884,
    39859.920,
    15394.370,
    6298.850,
    223.157,
    77.115,
    0.000,
    94000.0,
    79664.0,
    72000.0,
    56698.738,
    45303.310,
    33326.053,
)
_SILICON_STATISTICAL_WEIGHT = (
    9.0,
    56.0,
    15.0,
    7.0,
    3.0,
    28.0,
    21.0,
    5.0,
    15.0,
    3.0,
    7.0,
    1.0,
    9.0,
    5.0,
    21.0,
    3.0,
    9.0,
    15.0,
    5.0,
    3.0,
    3.0,
    9.0,
    1.0,
    5.0,
    5.0,
    3.0,
    1.0,
    3.0,
    3.0,
    5.0,
    12.0,
    15.0,
    5.0,
)
_SILICON_KARZAS_LEVELS = (
    (4, 2),
    (4, 3),
    (4, 2),
    (4, 2),
    (4, 2),
    (4, 3),
    (4, 2),
    (4, 2),
    (3, 2),
    (3, 2),
    (3, 2),
    (4, 1),
    (3, 2),
    (4, 1),
    (3, 2),
    (4, 1),
    (4, 1),
    (4, 1),
    (3, 2),
    (4, 1),
    (4, 0),
    (4, 0),
)
_SILICON_EFFECTIVE_CHARGE_FACTORS = (
    16.0,
    16.0,
    16.0,
    16.0,
    16.0,
    16.0,
    16.0,
    16.0,
    9.0,
    9.0,
    9.0,
    16.0,
    9.0,
    16.0,
    9.0,
    16.0,
    16.0,
    16.0,
    9.0,
    16.0,
    16.0,
    16.0,
)
_IRON_TRANSITION_WEIGHT = (
    25.0,
    35.0,
    21.0,
    15.0,
    9.0,
    35.0,
    33.0,
    21.0,
    27.0,
    49.0,
    9.0,
    21.0,
    27.0,
    9.0,
    9.0,
    25.0,
    33.0,
    15.0,
    35.0,
    3.0,
    5.0,
    11.0,
    15.0,
    13.0,
    15.0,
    9.0,
    21.0,
    15.0,
    21.0,
    25.0,
    35.0,
    9.0,
    5.0,
    45.0,
    27.0,
    21.0,
    15.0,
    21.0,
    15.0,
    25.0,
    21.0,
    35.0,
    5.0,
    15.0,
    45.0,
    35.0,
    55.0,
    25.0,
)
_IRON_TRANSITION_ENERGY_CM = (
    500.0,
    7500.0,
    12500.0,
    17500.0,
    19000.0,
    19500.0,
    19500.0,
    21000.0,
    22000.0,
    23000.0,
    23000.0,
    24000.0,
    24000.0,
    24500.0,
    24500.0,
    26000.0,
    26500.0,
    26500.0,
    27000.0,
    27500.0,
    28500.0,
    29000.0,
    29500.0,
    29500.0,
    29500.0,
    30000.0,
    31500.0,
    31500.0,
    33500.0,
    33500.0,
    34000.0,
    34500.0,
    34500.0,
    35000.0,
    35500.0,
    37000.0,
    37000.0,
    37000.0,
    38500.0,
    40000.0,
    40000.0,
    41000.0,
    41000.0,
    43000.0,
    43000.0,
    43000.0,
    43000.0,
    44000.0,
)
_IRON_TRANSITION_THRESHOLD_CM = (
    63500.0,
    58500.0,
    53500.0,
    59500.0,
    45000.0,
    44500.0,
    44500.0,
    43000.0,
    58000.0,
    41000.0,
    54000.0,
    40000.0,
    40000.0,
    57500.0,
    55500.0,
    38000.0,
    57500.0,
    57500.0,
    37000.0,
    54500.0,
    53500.0,
    55000.0,
    34500.0,
    34500.0,
    34500.0,
    34000.0,
    32500.0,
    32500.0,
    32500.0,
    32500.0,
    32000.0,
    29500.0,
    29500.0,
    31000.0,
    30500.0,
    29000.0,
    27000.0,
    54000.0,
    27500.0,
    24000.0,
    47000.0,
    23000.0,
    44000.0,
    42000.0,
    42000.0,
    21000.0,
    42000.0,
    42000.0,
)

_LYMAN_FREQUENCY_HZ = 3.28805e15


def _hc_over_kt(temperature: torch.Tensor) -> torch.Tensor:
    """Batched ``h c / k T`` (per-branch inline expressions, e.g. :3615-3619)."""

    return (
        PLANCK_ERG_SECOND_EXACT
        * LIGHT_SPEED_CM_PER_S_EXACT
        / torch.clamp(BOLTZMANN_ERG_PER_K_EXACT * temperature, min=1.0e-300)
    )


def _masked_karzas(
    frequency_hz: torch.Tensor,
    active: torch.Tensor,
    *,
    effective_charge_squared: float,
    principal_quantum_number: int,
    orbital_angular_momentum: int,
    tables: TwinContinuumTables,
    scale: float = 1.0,
) -> torch.Tensor:
    """Karzas-Latter grid evaluated on the full grid, zero outside ``active``.

    The reference evaluates only on the active subset and scatters; the grid
    helper is pointwise deterministic, so full-grid evaluation masked with
    ``torch.where`` is element-for-element identical on the active set.
    """

    grid = (
        _karzas_latter_cross_section_grid(
            frequency_hz,
            effective_charge_squared=effective_charge_squared,
            principal_quantum_number=principal_quantum_number,
            orbital_angular_momentum=orbital_angular_momentum,
            tables=tables,
        )
        * scale
    )
    return torch.where(active, grid, torch.zeros_like(grid))


def _carbon_neutral_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 8: C I absorption (:3523-3759)."""

    temperature = state.temperature
    mass_density = torch.clamp(state.mass_density, min=1.0e-300)
    carbon_population = state.partition_normalized_populations_by_packed_slot[..., 20]
    _, _, stimulated_emission = _planck_frequency_exact(temperature, frequency_hz)

    lyman_frequency_mask = frequency_hz <= _LYMAN_FREQUENCY_HZ

    rydberg_carbon = 109732.298
    dtype = temperature.dtype
    device = temperature.device
    level_energy_cm = torch.as_tensor(
        _CARBON_LEVEL_ENERGY_CM, dtype=dtype, device=device
    )
    statistical_weight = torch.as_tensor(
        _CARBON_STATISTICAL_WEIGHT, dtype=dtype, device=device
    )

    hc_over_kt = _hc_over_kt(temperature)
    boltzmann_weight = statistical_weight[:, None, None] * torch.exp(
        -level_energy_cm[:, None, None] * hc_over_kt[None, :, :]
    )  # (level, star, layer)

    wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
    ionization_limit_1 = 90862.70
    ionization_limit_2 = 90820.42
    ionization_limit_2b = ionization_limit_2 + 63.42
    ionization_limit_3 = ionization_limit_2 + 43003.3

    cross_sections: list[torch.Tensor] = [torch.zeros_like(wavenumber)] * 25
    for level_index in range(14):
        threshold_cm = ionization_limit_1 - _CARBON_LEVEL_ENERGY_CM[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        if level_index < 6:
            orbital_angular_momentum = 2
        elif level_index < 12:
            orbital_angular_momentum = 1
        else:
            orbital_angular_momentum = 0
        cross_sections[level_index] = _masked_karzas(
            frequency_hz,
            active,
            effective_charge_squared=9.0 / rydberg_carbon * threshold_cm,
            principal_quantum_number=3,
            orbital_angular_momentum=orbital_angular_momentum,
            tables=tables,
        )

    special = [torch.zeros_like(wavenumber) for _ in range(14, 19)]
    for ionization_limit, limit_weight in (
        (ionization_limit_2, 1.0 / 3.0),
        (ionization_limit_2b, 2.0 / 3.0),
    ):
        active_1s = lyman_frequency_mask & (
            wavenumber >= ionization_limit - _CARBON_LEVEL_ENERGY_CM[14]
        )
        background = 10.0 ** (
            -16.80
            - (wavenumber - ionization_limit + _CARBON_LEVEL_ENERGY_CM[14])
            / 3.0
            / rydberg_carbon
        )
        resonance = (wavenumber - 97700.0) * 2.0 / 2743.0
        resonant_cross_section = (68.0e-18 * resonance + 118.0e-18) / (
            resonance**2 + 1.0
        )
        special[0] = special[0] + torch.where(
            active_1s,
            (background + resonant_cross_section) * limit_weight,
            torch.zeros_like(wavenumber),
        )

        active_1d = lyman_frequency_mask & (
            wavenumber >= ionization_limit - _CARBON_LEVEL_ENERGY_CM[15]
        )
        background = 10.0 ** (
            -16.80
            - (wavenumber - ionization_limit + _CARBON_LEVEL_ENERGY_CM[15])
            / 3.0
            / rydberg_carbon
        )
        resonance_1 = (wavenumber - 93917.0) * 2.0 / 9230.0
        resonant_cross_section_1 = (22.0e-18 * resonance_1 + 26.0e-18) / (
            resonance_1**2 + 1.0
        )
        resonance_2 = (wavenumber - 111130.0) * 2.0 / 2743.0
        resonant_cross_section_2 = (-10.5e-18 * resonance_2 + 46.0e-18) / (
            resonance_2**2 + 1.0
        )
        special[1] = special[1] + torch.where(
            active_1d,
            (background + resonant_cross_section_1 + resonant_cross_section_2)
            * limit_weight,
            torch.zeros_like(wavenumber),
        )

        for level_index in range(16, 19):
            active = lyman_frequency_mask & (
                wavenumber >= ionization_limit - _CARBON_LEVEL_ENERGY_CM[level_index]
            )
            special[level_index - 14] = special[level_index - 14] + torch.where(
                active,
                10.0
                ** (
                    -16.80
                    - (
                        wavenumber
                        - ionization_limit
                        + _CARBON_LEVEL_ENERGY_CM[level_index]
                    )
                    / 3.0
                    / rydberg_carbon
                )
                * limit_weight,
                torch.zeros_like(wavenumber),
            )
    for offset, level_index in enumerate(range(14, 19)):
        cross_sections[level_index] = special[offset]

    for level_index in range(19, 25):
        threshold_cm = ionization_limit_3 - _CARBON_LEVEL_ENERGY_CM[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        cross_sections[level_index] = _masked_karzas(
            frequency_hz,
            active,
            effective_charge_squared=4.0 / rydberg_carbon * threshold_cm,
            principal_quantum_number=2,
            orbital_angular_momentum=1,
            tables=tables,
            scale=3.0,
        )
    cross_section_by_level = torch.stack(cross_sections)  # (25, freq)

    kramers_limit = ionization_limit_2
    frequency_cubed_factor = 2.815e29 / frequency_hz**3
    kramers_lower = torch.clamp(
        kramers_limit - wavenumber, min=kramers_limit - rydberg_carbon / 16.0
    )
    freefree_profile = (
        frequency_cubed_factor[None, None, :]
        * 6.0
        / (rydberg_carbon * hc_over_kt[:, :, None])
        * (
            torch.exp(-kramers_lower[None, None, :] * hc_over_kt[:, :, None])
            - torch.exp(-kramers_limit * hc_over_kt)[:, :, None]
        )
    )
    branch_profile = freefree_profile + (
        boltzmann_weight.permute(1, 2, 0) @ cross_section_by_level
    )
    absorption = (
        branch_profile
        * stimulated_emission
        * carbon_population[:, :, None]
        / mass_density[:, :, None]
    )
    return torch.where(
        lyman_frequency_mask[None, None, :], absorption, torch.zeros_like(absorption)
    )


def _magnesium_neutral_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 8: Mg I absorption (:3759-3959)."""

    temperature = state.temperature
    mass_density = torch.clamp(state.mass_density, min=1.0e-300)
    magnesium_population = state.partition_normalized_populations_by_packed_slot[
        ..., 77
    ]
    _, _, stimulated_emission = _planck_frequency_exact(temperature, frequency_hz)

    lyman_frequency_mask = frequency_hz <= _LYMAN_FREQUENCY_HZ

    rydberg_magnesium = 109732.298
    ionization_limit = 61671.02
    dtype = temperature.dtype
    device = temperature.device
    level_energy_cm = torch.as_tensor(
        _MAGNESIUM_LEVEL_ENERGY_CM, dtype=dtype, device=device
    )
    statistical_weight = torch.as_tensor(
        _MAGNESIUM_STATISTICAL_WEIGHT, dtype=dtype, device=device
    )

    hc_over_kt = _hc_over_kt(temperature)
    boltzmann_weight = statistical_weight[:, None, None] * torch.exp(
        -level_energy_cm[:, None, None] * hc_over_kt[None, :, :]
    )

    wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
    cross_sections: list[torch.Tensor] = [torch.zeros_like(wavenumber)] * 15

    for level_index in range(2):
        threshold_cm = ionization_limit - _MAGNESIUM_LEVEL_ENERGY_CM[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        cross_sections[level_index] = _masked_karzas(
            frequency_hz,
            active,
            effective_charge_squared=16.0 / rydberg_magnesium * threshold_cm,
            principal_quantum_number=4,
            orbital_angular_momentum=3,
            tables=tables,
        )
    for level_index in range(2, 4):
        threshold_cm = ionization_limit - _MAGNESIUM_LEVEL_ENERGY_CM[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        cross_sections[level_index] = _masked_karzas(
            frequency_hz,
            active,
            effective_charge_squared=16.0 / rydberg_magnesium * threshold_cm,
            principal_quantum_number=4,
            orbital_angular_momentum=2,
            tables=tables,
        )
    threshold_cm = ionization_limit - _MAGNESIUM_LEVEL_ENERGY_CM[4]
    active = lyman_frequency_mask & (wavenumber >= threshold_cm)
    cross_sections[4] = _masked_karzas(
        frequency_hz,
        active,
        effective_charge_squared=16.0 / rydberg_magnesium * threshold_cm,
        principal_quantum_number=4,
        orbital_angular_momentum=1,
        tables=tables,
    )

    power_laws = (
        (5, 25.0e-18, 13713.986, 2.7),
        (6, 33.8e-18, 13823.223, 2.8),
        (7, 45.0e-18, 15267.955, 2.7),
        (8, 0.43e-18, 18167.687, 2.6),
        (9, 2.1e-18, 20473.617, 2.6),
    )
    for level_index, coefficient, reference_wavenumber, exponent in power_laws:
        active = lyman_frequency_mask & (
            wavenumber >= ionization_limit - _MAGNESIUM_LEVEL_ENERGY_CM[level_index]
        )
        cross_sections[level_index] = torch.where(
            active,
            coefficient * (reference_wavenumber / wavenumber) ** exponent,
            torch.zeros_like(wavenumber),
        )

    active = lyman_frequency_mask & (
        wavenumber >= ionization_limit - _MAGNESIUM_LEVEL_ENERGY_CM[10]
    )
    cross_sections[10] = torch.where(
        active,
        16.0e-18 * (26619.756 / wavenumber) ** 2.1
        - 7.8e-18 * (26619.756 / wavenumber) ** 9.5,
        torch.zeros_like(wavenumber),
    )

    for level_index in range(11, 14):
        active = lyman_frequency_mask & (
            wavenumber >= ionization_limit - _MAGNESIUM_LEVEL_ENERGY_CM[level_index]
        )
        shallow_power = 20.0e-18 * (39759.842 / wavenumber) ** 2.7
        steep_power = 40.0e-18 * (39759.842 / wavenumber) ** 14
        cross_sections[level_index] = torch.where(
            active,
            torch.maximum(shallow_power, steep_power),
            torch.zeros_like(wavenumber),
        )

    active = lyman_frequency_mask & (
        wavenumber >= ionization_limit - _MAGNESIUM_LEVEL_ENERGY_CM[14]
    )
    cross_sections[14] = torch.where(
        active,
        1.1e-18
        * (
            (ionization_limit - _MAGNESIUM_LEVEL_ENERGY_CM[14])
            / wavenumber
        )
        ** 10,
        torch.zeros_like(wavenumber),
    )
    cross_section_by_level = torch.stack(cross_sections)  # (15, freq)

    frequency_cubed_factor = 2.815e29 / frequency_hz**3
    kramers_lower = torch.clamp(
        ionization_limit - wavenumber, min=ionization_limit - rydberg_magnesium / 25.0
    )
    freefree_profile = (
        frequency_cubed_factor[None, None, :]
        * 2.0
        / (rydberg_magnesium * hc_over_kt[:, :, None])
        * (
            torch.exp(-kramers_lower[None, None, :] * hc_over_kt[:, :, None])
            - torch.exp(-ionization_limit * hc_over_kt)[:, :, None]
        )
    )
    branch_profile = freefree_profile + (
        boltzmann_weight.permute(1, 2, 0) @ cross_section_by_level
    )
    absorption = (
        branch_profile
        * stimulated_emission
        * magnesium_population[:, :, None]
        / mass_density[:, :, None]
    )
    return torch.where(
        lyman_frequency_mask[None, None, :], absorption, torch.zeros_like(absorption)
    )


def _silicon_neutral_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 8: Si I absorption (:3959-4240)."""

    temperature = state.temperature
    mass_density = torch.clamp(state.mass_density, min=1.0e-300)
    silicon_population = state.partition_normalized_populations_by_packed_slot[
        ..., 104
    ]
    _, _, stimulated_emission = _planck_frequency_exact(temperature, frequency_hz)

    lyman_frequency_mask = frequency_hz <= _LYMAN_FREQUENCY_HZ

    rydberg_silicon = 109732.298
    dtype = temperature.dtype
    device = temperature.device
    level_energy_cm = torch.as_tensor(
        _SILICON_LEVEL_ENERGY_CM, dtype=dtype, device=device
    )
    statistical_weight = torch.as_tensor(
        _SILICON_STATISTICAL_WEIGHT, dtype=dtype, device=device
    )

    hc_over_kt = _hc_over_kt(temperature)
    boltzmann_weight = statistical_weight[:, None, None] * torch.exp(
        -level_energy_cm[:, None, None] * hc_over_kt[None, :, :]
    )

    wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
    cross_sections: list[torch.Tensor] = [torch.zeros_like(wavenumber)] * 33

    ionization_limit_1 = 65939.18
    for level_index, (
        principal_quantum_number,
        orbital_angular_momentum,
    ) in enumerate(_SILICON_KARZAS_LEVELS):
        threshold_cm = ionization_limit_1 - _SILICON_LEVEL_ENERGY_CM[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        cross_sections[level_index] = _masked_karzas(
            frequency_hz,
            active,
            effective_charge_squared=(
                _SILICON_EFFECTIVE_CHARGE_FACTORS[level_index]
                / rydberg_silicon
                * threshold_cm
            ),
            principal_quantum_number=principal_quantum_number,
            orbital_angular_momentum=orbital_angular_momentum,
            tables=tables,
        )

    special = [torch.zeros_like(wavenumber) for _ in range(22, 27)]
    for ionization_limit, limit_weight in (
        (65747.55, 1.0 / 3.0),
        (65747.55 + 287.45, 2.0 / 3.0),
    ):
        active_1s = lyman_frequency_mask & (
            wavenumber >= ionization_limit - _SILICON_LEVEL_ENERGY_CM[22]
        )
        resonance = (wavenumber - 70000.0) * 2.0 / 6500.0
        resonant_cross_section = (97.0e-18 * resonance + 94.0e-18) / (
            resonance**2 + 1.0
        )
        special[0] = special[0] + torch.where(
            active_1s,
            (
                37.0e-18 * (50353.180 / wavenumber) ** 2.40
                + resonant_cross_section
            )
            * limit_weight,
            torch.zeros_like(wavenumber),
        )

        active_1d = lyman_frequency_mask & (
            wavenumber >= ionization_limit - _SILICON_LEVEL_ENERGY_CM[23]
        )
        resonance = (wavenumber - 78600.0) * 2.0 / 13000.0
        resonant_cross_section = (-10.0e-18 * resonance + 77.0e-18) / (
            resonance**2 + 1.0
        )
        special[1] = special[1] + torch.where(
            active_1d,
            (
                24.5e-18 * (59448.700 / wavenumber) ** 1.85
                + resonant_cross_section
            )
            * limit_weight,
            torch.zeros_like(wavenumber),
        )

        for level_index in (24, 25, 26):
            active = lyman_frequency_mask & (
                wavenumber >= ionization_limit - _SILICON_LEVEL_ENERGY_CM[level_index]
            )
            ratio = 65524.393 / wavenumber
            effective_weight = (2.0 / 3.0) if level_index == 25 else limit_weight
            special[level_index - 22] = special[level_index - 22] + torch.where(
                active,
                torch.where(
                    wavenumber <= 74000.0,
                    72.0e-18 * ratio**1.90,
                    93.0e-18 * ratio**4.00,
                )
                * effective_weight,
                torch.zeros_like(wavenumber),
            )
    for offset, level_index in enumerate(range(22, 27)):
        cross_sections[level_index] = special[offset]

    ionization_limit_3 = 65747.5 + 42824.35
    for level_index in range(27, 33):
        threshold_cm = ionization_limit_3 - _SILICON_LEVEL_ENERGY_CM[level_index]
        active = lyman_frequency_mask & (wavenumber >= threshold_cm)
        cross_sections[level_index] = _masked_karzas(
            frequency_hz,
            active,
            effective_charge_squared=9.0 / rydberg_silicon * threshold_cm,
            principal_quantum_number=3,
            orbital_angular_momentum=1,
            tables=tables,
            scale=3.0,
        )
    cross_section_by_level = torch.stack(cross_sections)  # (33, freq)

    freefree_limit = 65747.55
    frequency_cubed_factor = 2.815e29 / frequency_hz**3
    kramers_lower = torch.clamp(
        freefree_limit - wavenumber, min=freefree_limit - rydberg_silicon / 25.0
    )
    freefree_profile = (
        frequency_cubed_factor[None, None, :]
        * 6.0
        / (rydberg_silicon * hc_over_kt[:, :, None])
        * (
            torch.exp(-kramers_lower[None, None, :] * hc_over_kt[:, :, None])
            - torch.exp(-freefree_limit * hc_over_kt)[:, :, None]
        )
    )
    branch_profile = freefree_profile + (
        boltzmann_weight.permute(1, 2, 0) @ cross_section_by_level
    )
    absorption = (
        branch_profile
        * stimulated_emission
        * silicon_population[:, :, None]
        / mass_density[:, :, None]
    )
    return torch.where(
        lyman_frequency_mask[None, None, :], absorption, torch.zeros_like(absorption)
    )


def _aluminum_neutral_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 8: Al I absorption (:5424-5469)."""

    temperature = state.temperature
    mass_density = torch.clamp(state.mass_density, min=1.0e-300)
    aluminum_population = state.partition_normalized_populations_by_packed_slot[
        ..., 90
    ]
    _, _, stimulated_emission = _planck_frequency_exact(temperature, frequency_hz)

    wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
    ionization_limit = 48278.37

    active = frequency_hz <= _LYMAN_FREQUENCY_HZ
    upper_edge = active & (wavenumber >= ionization_limit - 112.061)
    branch_cross_section = torch.where(
        upper_edge,
        6.5e-17 * ((ionization_limit - 112.061) / wavenumber) ** 5 * 4.0,
        torch.zeros_like(wavenumber),
    )
    lower_edge = active & (wavenumber >= ionization_limit)
    branch_cross_section = branch_cross_section + torch.where(
        lower_edge,
        6.5e-17 * (ionization_limit / wavenumber) ** 5 * 2.0,
        torch.zeros_like(wavenumber),
    )

    return (
        aluminum_population[:, :, None]
        * stimulated_emission
        / mass_density[:, :, None]
        * branch_cross_section[None, None, :]
    )


def _iron_neutral_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 8: Fe I absorption (:5469-5706, numba kernel :1642-1677).

    The kernel semantics are the per-transition numpy fallback: a
    ``(layer, freq, transition)`` sum of
    ``cross_section(freq, trans) * weight[trans] * exp(-energy[trans] *
    hc_over_kt[layer])`` with a per-transition threshold mask, evaluated
    here as a batched matmul over the transition axis.
    """

    temperature = state.temperature
    mass_density = torch.clamp(state.mass_density, min=1.0e-300)
    iron_population = state.partition_normalized_populations_by_packed_slot[..., 350]
    _, _, stimulated_emission = _planck_frequency_exact(temperature, frequency_hz)

    wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
    active_frequency = wavenumber >= 21000.0

    dtype = temperature.dtype
    device = temperature.device
    transition_weight = torch.as_tensor(
        _IRON_TRANSITION_WEIGHT, dtype=dtype, device=device
    )
    transition_energy_cm = torch.as_tensor(
        _IRON_TRANSITION_ENERGY_CM, dtype=dtype, device=device
    )
    transition_threshold_cm = torch.as_tensor(
        _IRON_TRANSITION_THRESHOLD_CM, dtype=dtype, device=device
    )

    hc_over_kt = _hc_over_kt(temperature)
    boltzmann = transition_weight[:, None, None] * torch.exp(
        -transition_energy_cm[:, None, None] * hc_over_kt[None, :, :]
    )  # (trans, star, layer)

    ratio = (
        (transition_threshold_cm[None, :] + 3000.0 - wavenumber[:, None])
        / transition_threshold_cm[None, :]
        / 0.1
    )  # (freq, trans)
    cross_section = 3.0e-18 / (1.0 + ratio * ratio * ratio * ratio)
    cross_section = torch.where(
        wavenumber[:, None] >= transition_threshold_cm[None, :],
        cross_section,
        torch.zeros_like(cross_section),
    )

    branch_profile = boltzmann.permute(1, 2, 0) @ cross_section.transpose(0, 1)
    absorption = (
        branch_profile
        * stimulated_emission
        * iron_population[:, :, None]
        / mass_density[:, :, None]
    )
    return torch.where(
        active_frequency[None, None, :], absorption, torch.zeros_like(absorption)
    )
