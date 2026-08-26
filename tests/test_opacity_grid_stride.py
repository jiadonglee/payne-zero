"""Fast tests for frequency-grid decimation of the opacity-sampling grid.

The opacity stage costs time linear in the number of sampled frequencies, so
``build_opacity_sampling_grid`` takes a stride. These tests are the regression
guard on the undecimated path: stride 1 must stay bit-identical to the grid the
released results were produced on. Nothing here runs the solver; the whole file
is milliseconds.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere
from payne_zero_atmosphere.config import (
    AtmosphereConfig,
    AtmosphereInput,
    AtmosphereOutput,
)
from payne_zero_atmosphere.continuum_opacity import build_opacity_sampling_grid
from payne_zero_atmosphere.run_setup import resolve_run_setup


LIGHT_SPEED_NM_PER_S = 2.99792458e17

# One effective temperature per ionization-edge branch of the grid start index
# (carbon, Lyman, neutral helium, ionized helium, and the hot default), with the
# SHA-256 of the wavelength and weight arrays as they were produced by the
# implementation before the stride argument existed. These are the frozen
# expected result: any change to them changes every released atmosphere.
FROZEN_GRID_DIGESTS = {
    3500.0: (
        "68a3b28195d9efce232e673b0ab06085e93a0d6058fae3a26cfc4853577dc0ff",
        "f274ca3fdb8d8047e7525551afb19b0a15ef79cc79d4412dedd235312fcfe5f0",
    ),
    5777.0: (
        "8944f1dd701ba27f50d37a16e48ab9375e1bef1b444ed405b671ba91fde8132b",
        "aca3355f13f85d3a8f785765b068bfba17f9a72e7d97740383491f90aaaca710",
    ),
    8000.0: (
        "59094990944b60c8247fcf8474f28b172d785a68929cd37be046d8a75a2bfb83",
        "0b0c185b505d060ca7d42289830a2570de9b251485779f55aee4ce11e3f4bdea",
    ),
    15000.0: (
        "8d81ea1751d673ff9f3bf59bd3367631cc2db1e4831e57cb0a1c4067963c1385",
        "35f5a6152751aaa045d59d20a9ec85e18caf957403b09f85a84b77d0e4ffefba",
    ),
    40000.0: (
        "7b675900af45beee85b2794d6219a30e017d2ad034faf78800eb86894f11349e",
        "fc9a9c26eb523a63104d5106e869af77a37bbc5512066e5631ff8c3fde1a2f13",
    ),
}

# The decimation experiment's ladder.
EXPERIMENT_STRIDES = (1, 2, 4, 8)

# The weight sum drifts by one factor of the log-wavelength step per unit of
# stride, all of it in the first-point weight; see the tolerance test.
LOGARITHMIC_WAVELENGTH_STEP_PER_POINT = np.log(10.0) * 1.0e-4


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _frequency_hz(wavelength_nm: np.ndarray) -> np.ndarray:
    return LIGHT_SPEED_NM_PER_S / wavelength_nm


def _synthetic_atmosphere(layers: int = 72) -> ModelAtmosphere:
    """A minimal seed that passes ``validate_atmosphere_seed``.

    The numbers only have to be finite, positive, and monotone in column mass;
    nothing here reaches the physics, it only exercises configuration
    resolution.
    """

    depth = np.arange(layers, dtype=np.float64)
    return ModelAtmosphere(
        column_mass=10.0 ** (-4.0 + depth * 0.1),
        temperature=4000.0 + depth * 50.0,
        gas_pressure=10.0 ** (1.0 + depth * 0.1),
        electron_density=10.0 ** (8.0 + depth * 0.05),
        rosseland_opacity=np.full(layers, 0.5),
        radiative_acceleration=np.full(layers, 1.0),
        microturbulence=np.full(layers, 2.0e5),
        convective_flux=np.zeros(layers),
        convective_velocity=np.zeros(layers),
        metadata={
            "effective_temperature": "5777.0",
            "log_surface_gravity": "4.44",
        },
    )


def _config(**overrides) -> AtmosphereConfig:
    return AtmosphereConfig(
        inputs=AtmosphereInput(initial_atmosphere=_synthetic_atmosphere()),
        outputs=AtmosphereOutput(),
        **overrides,
    )


@pytest.mark.parametrize("effective_temperature", sorted(FROZEN_GRID_DIGESTS))
def test_default_stride_reproduces_the_frozen_grid_bit_for_bit(effective_temperature):
    """Stride 1 is the production grid, to the last bit of every element."""

    wavelength_nm, frequency_weights = build_opacity_sampling_grid(
        effective_temperature
    )
    expected_wavelength_digest, expected_weight_digest = FROZEN_GRID_DIGESTS[
        effective_temperature
    ]
    assert wavelength_nm.size == 30000
    assert wavelength_nm.dtype == np.float64
    assert frequency_weights.dtype == np.float64
    assert _digest(wavelength_nm) == expected_wavelength_digest
    assert _digest(frequency_weights) == expected_weight_digest


@pytest.mark.parametrize("effective_temperature", sorted(FROZEN_GRID_DIGESTS))
def test_stride_one_is_the_default_however_it_is_spelled(effective_temperature):
    """An explicit stride of 1 must not take a different code path."""

    default_wavelength, default_weights = build_opacity_sampling_grid(
        effective_temperature
    )
    for wavelength_nm, frequency_weights in (
        build_opacity_sampling_grid(effective_temperature, 1),
        build_opacity_sampling_grid(effective_temperature, frequency_grid_stride=1),
    ):
        assert np.array_equal(wavelength_nm, default_wavelength)
        assert np.array_equal(frequency_weights, default_weights)


@pytest.mark.parametrize("stride", EXPERIMENT_STRIDES)
def test_decimated_grid_brackets_the_undecimated_wavelength_range(stride):
    """Both endpoints survive, so every stride covers the same band.

    The first point matters beyond the range: it is the ionization edge that
    ``active_continuum_reference_frequencies`` and
    ``assemble_continuum_line_selection_threshold`` read as element 0 to decide
    which continuum reference columns are active.
    """

    reference_wavelength, _ = build_opacity_sampling_grid(5777.0)
    wavelength_nm, frequency_weights = build_opacity_sampling_grid(5777.0, stride)

    assert wavelength_nm[0] == reference_wavelength[0]
    assert wavelength_nm[-1] == reference_wavelength[-1]
    assert wavelength_nm.size == frequency_weights.size
    assert wavelength_nm.size == 30000 // stride + (1 if stride > 1 else 0)
    assert np.all(np.diff(wavelength_nm) > 0.0)
    assert np.all(np.isin(wavelength_nm, reference_wavelength)), (
        "the decimated grid must be a subset of the undecimated one, not a "
        "re-sampling of it"
    )
    assert np.all(frequency_weights > 0.0)


@pytest.mark.parametrize("stride", EXPERIMENT_STRIDES)
def test_weight_sum_is_preserved_across_strides(stride):
    """The integrated frequency measure is stride-independent to 1e-6.

    Scaling the whole ``1.5`` first-point factor with the stride drifts the
    total by exactly ``(stride - 1) * 2.3025e-4`` -- +1.6e-3 at stride 8. That
    factor is two things added together: a half cell of the sampled grid, which
    must scale with the stride, and a full extra cell standing in for the band
    just blueward of the grid, which represents a fixed physical band and must
    not. A decimation experiment exists to isolate the cost of lost frequency
    resolution, so a boundary term growing with the stride would confound
    exactly the measurement being made. Holding the stand-in at its undecimated
    value removes the drift and leaves stride 1 untouched.

    What survives is the telescoped total
    ``nu[0] + dnu_undecimated - (nu[-2] + nu[-1]) / 4``, whose only
    stride-sensitive term is ``nu[-2]``, the second-to-last sampled point.
    """

    reference_wavelength_nm, reference_weights = build_opacity_sampling_grid(5777.0)
    wavelength_nm, frequency_weights = build_opacity_sampling_grid(5777.0, stride)

    relative_change = frequency_weights.sum() / reference_weights.sum() - 1.0
    assert abs(relative_change) < 1.0e-6

    reference_frequency_hz = _frequency_hz(reference_wavelength_nm)
    undecimated_first_step = reference_frequency_hz[0] - reference_frequency_hz[1]
    frequency_hz = _frequency_hz(wavelength_nm)
    telescoped_total = (
        frequency_hz[0]
        + undecimated_first_step
        - 0.25 * (frequency_hz[-2] + frequency_hz[-1])
    )
    assert frequency_weights.sum() == pytest.approx(telescoped_total, rel=1.0e-12)

    if stride > 1:
        unfixed_drift = LOGARITHMIC_WAVELENGTH_STEP_PER_POINT * (stride - 1)
        assert abs(relative_change) < unfixed_drift / 100.0, (
            "the blueward stand-in is scaling with the stride again"
        )


@pytest.mark.parametrize("stride", EXPERIMENT_STRIDES)
def test_weight_sum_drift_does_not_depend_on_effective_temperature(stride):
    """Every grid start index sees the same fractional drift."""

    drifts = []
    for effective_temperature in sorted(FROZEN_GRID_DIGESTS):
        reference_weights = build_opacity_sampling_grid(effective_temperature)[1]
        frequency_weights = build_opacity_sampling_grid(
            effective_temperature, stride
        )[1]
        drifts.append(frequency_weights.sum() / reference_weights.sum() - 1.0)
    assert np.ptp(drifts) < 1.0e-12


@pytest.mark.parametrize("stride", EXPERIMENT_STRIDES)
def test_interior_weights_are_the_central_difference_of_the_sampled_grid(stride):
    """The weights come from the decimated grid, not from folded-in neighbours."""

    wavelength_nm, frequency_weights = build_opacity_sampling_grid(5777.0, stride)
    frequency_hz = _frequency_hz(wavelength_nm)

    np.testing.assert_allclose(
        frequency_weights[1:-1],
        0.5 * (frequency_hz[:-2] - frequency_hz[2:]),
        rtol=0.0,
        atol=0.0,
    )
    reference_frequency_hz = _frequency_hz(build_opacity_sampling_grid(5777.0)[0])
    undecimated_first_step = reference_frequency_hz[0] - reference_frequency_hz[1]
    assert frequency_weights[0] == (
        0.5 * (frequency_hz[0] - frequency_hz[1]) + undecimated_first_step
    )
    assert frequency_weights[-1] == 0.25 * (frequency_hz[-2] + frequency_hz[-1])


@pytest.mark.parametrize("stride", (0, -1, 15001))
def test_out_of_range_stride_is_rejected(stride):
    with pytest.raises(ValueError):
        build_opacity_sampling_grid(5777.0, stride)


def test_config_default_leaves_the_grid_unchanged():
    """The shipped default must resolve to the undecimated production grid."""

    config = _config()
    assert config.opacity_frequency_grid_stride == 1

    setup = resolve_run_setup(config)
    assert setup.opacity_frequency_grid_stride == 1

    default_wavelength, default_weights = build_opacity_sampling_grid(
        setup.effective_temperature
    )
    wavelength_nm, frequency_weights = build_opacity_sampling_grid(
        setup.effective_temperature,
        frequency_grid_stride=setup.opacity_frequency_grid_stride,
    )
    assert np.array_equal(wavelength_nm, default_wavelength)
    assert np.array_equal(frequency_weights, default_weights)
    assert _digest(wavelength_nm) == FROZEN_GRID_DIGESTS[5777.0][0]
    assert _digest(frequency_weights) == FROZEN_GRID_DIGESTS[5777.0][1]


@pytest.mark.parametrize("stride", EXPERIMENT_STRIDES)
def test_config_stride_reaches_the_resolved_run_setup(stride):
    setup = resolve_run_setup(_config(opacity_frequency_grid_stride=stride))
    assert setup.opacity_frequency_grid_stride == stride
    wavelength_nm, _ = build_opacity_sampling_grid(
        setup.effective_temperature,
        frequency_grid_stride=setup.opacity_frequency_grid_stride,
    )
    assert wavelength_nm.size == 30000 // stride + (1 if stride > 1 else 0)


def test_run_setup_rejects_a_non_positive_config_stride():
    with pytest.raises(ValueError):
        resolve_run_setup(_config(opacity_frequency_grid_stride=0))


def test_run_setup_stride_survives_dataclass_replace():
    """The per-iteration ``replace(setup, ...)`` in the runner must carry it."""

    setup = resolve_run_setup(_config(opacity_frequency_grid_stride=4))
    iteration_setup = replace(setup, surface_radiation_pressure_constant=1.0)
    assert iteration_setup.opacity_frequency_grid_stride == 4
