"""Dual-crossing entropy closure (v2) for the no-emulator initializer.
The v1 entropy-closure family was vetoed by its pre-registered Gate-0
oracle: a single onset + one adiabat + one bump cannot represent the
deep 7000--8000 K gradient (superadiabatic spike, decaying adiabat,
then a near-isothermal tail).  v2 models BOTH convective boundaries:
an enter switch (w_enter, convection on) and an exit switch (w_exit,
convection off) so the deep gradient returns to the radiative branch.
Prediction imports no Torch/checkpoint/SVD basis/atmosphere grid.
Only stored state: EntropyClosureV2Parameters (581 base floats).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .discovery import (
    polynomial_exponents, polynomial_features)

FORMAT_MARKER = "payne_zero_entropy_closure_v2"

# Fixed code constants -- never fitted.
LAYER_SCAN_START = 36
PERSISTENCE_LAYERS = 3
SWITCH_WIDTH_DEX = 0.30
GAMMA_AD_MIN, GAMMA_AD_MAX = 0.10, 0.45
AE_MAX = 0.50
AX_MIN, AX_MAX = -0.50, 0.50

# Label-polynomial degrees per component (plan section 2).
OPACITY_DEGREE = 3
TEMPERATURE_DEGREE = 2
SURFACE_MASS_DEGREE = 2
CLOSURE_PARAM_DEGREE = 1

# Fixed depth-basis sizes (code constants, not fitted values).
OPACITY_MODES = 8
TEMPERATURE_MODES = 4
CHEBYSHEV_DEGREE = 24
LOG_TAU_MIN, LOG_TAU_MAX = -2.0, 2.0
LAYERS = 80

# Monomial exponent tables (sizes are properties of the chosen degrees).
_OPACITY_EXPONENTS = polynomial_exponents(5, OPACITY_DEGREE)
_TEMPERATURE_EXPONENTS = polynomial_exponents(5, TEMPERATURE_DEGREE)
_SURFACE_MASS_EXPONENTS = polynomial_exponents(5, SURFACE_MASS_DEGREE)
_CLOSURE_EXPONENTS = polynomial_exponents(5, CLOSURE_PARAM_DEGREE)


def chebyshev_basis(log_tau: np.ndarray, degree: int) -> np.ndarray:
    """Fixed Chebyshev modes on the depth coordinate mapped to [-1, 1]."""

    values = np.asarray(log_tau, dtype=np.float64)
    x = 2.0 * (values - LOG_TAU_MIN) / (LOG_TAU_MAX - LOG_TAU_MIN) - 1.0
    basis = np.empty((values.size, degree), dtype=np.float64)
    if degree >= 1:
        basis[:, 0] = 1.0
    if degree >= 2:
        basis[:, 1] = x
    for order in range(2, degree):
        basis[:, order] = (
            2.0 * x * basis[:, order - 1] - basis[:, order - 2]
        )
    return basis


def _logistic(log_p: np.ndarray, log_p_0: float, width_dex: float) -> np.ndarray:
    """Smooth logistic switch in pressure space, fixed width in dex."""

    width = max(float(width_dex), 1.0e-6)
    return 1.0 / (1.0 + np.exp(-(np.asarray(log_p, dtype=np.float64) - float(log_p_0)) / width))


def _exponents_for(size: int) -> np.ndarray:
    """Return the monomial exponent table matching a coefficient array size."""

    if size == _OPACITY_EXPONENTS.shape[0]:
        return _OPACITY_EXPONENTS
    if size == _TEMPERATURE_EXPONENTS.shape[0]:
        return _TEMPERATURE_EXPONENTS
    if size == _SURFACE_MASS_EXPONENTS.shape[0]:
        return _SURFACE_MASS_EXPONENTS
    if size == _CLOSURE_EXPONENTS.shape[0]:
        return _CLOSURE_EXPONENTS
    raise ValueError(f"no exponent table for coefficient size {size}")


def _label_scalars(
    normalized: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Evaluate one polynomial-in-labels scalar per profile."""

    exponents = _exponents_for(int(coefficients.size))
    features, _, _ = polynomial_features(
        normalized,
        exponents,
        center=np.zeros(5, dtype=np.float64),
        scale=np.ones(5, dtype=np.float64),
    )
    return features @ coefficients.astype(np.float64)


@dataclass(frozen=True)
class EntropyClosureV2Parameters:
    """Stored constants for the dual-crossing closure (581 base floats)."""

    feature_center: np.ndarray
    feature_scale: np.ndarray
    opacity_coefficients: np.ndarray
    temperature_coefficients: np.ndarray
    surface_mass_coefficients: np.ndarray
    gamma_ad_coefficients: np.ndarray
    a_enter_coefficients: np.ndarray
    a_exit_coefficients: np.ndarray
    exit_logp_coefficients: np.ndarray | None = None
    corpus_sha256: str = ""
    model_spec_sha256: str = ""

    def _validate(self) -> None:
        if self.feature_center.shape != (5,) or self.feature_scale.shape != (5,):
            raise ValueError("feature_center/scale must be length 5")
        opacity_rows = _OPACITY_EXPONENTS.shape[0]
        temperature_rows = _TEMPERATURE_EXPONENTS.shape[0]
        surface_rows = _SURFACE_MASS_EXPONENTS.shape[0]
        closure_rows = _CLOSURE_EXPONENTS.shape[0]
        if self.opacity_coefficients.shape != (OPACITY_MODES, opacity_rows):
            raise ValueError("opacity_coefficients must have shape (8, 56)")
        if self.temperature_coefficients.shape != (TEMPERATURE_MODES, temperature_rows):
            raise ValueError("temperature_coefficients must have shape (4, 21)")
        if self.surface_mass_coefficients.shape != (surface_rows,):
            raise ValueError("surface_mass_coefficients must have length 21")
        for name, array in (
            ("gamma_ad_coefficients", self.gamma_ad_coefficients),
            ("a_enter_coefficients", self.a_enter_coefficients),
            ("a_exit_coefficients", self.a_exit_coefficients),
        ):
            if np.asarray(array).shape != (closure_rows,):
                raise ValueError(f"{name} must have length {closure_rows}")
        if self.exit_logp_coefficients is not None and (
            np.asarray(self.exit_logp_coefficients).shape != (closure_rows,)
        ):
            raise ValueError("exit_logp_coefficients must have length 6 or be None")

    @property
    def base_float_count(self) -> int:
        self._validate()
        return int(
            10
            + self.opacity_coefficients.size
            + self.temperature_coefficients.size
            + self.surface_mass_coefficients.size
            + self.gamma_ad_coefficients.size
            + self.a_enter_coefficients.size
            + self.a_exit_coefficients.size
        )

    @property
    def fitted_float_count(self) -> int:
        count = self.base_float_count
        if self.exit_logp_coefficients is not None:
            count += int(self.exit_logp_coefficients.size)
        return count


def schwarzschild_crossings(
    log_p: np.ndarray,
    grad_rad: np.ndarray,
    gamma_ad: np.ndarray,
    *,
    start_layer: int = LAYER_SCAN_START,
    persistence: int = PERSISTENCE_LAYERS,
) -> tuple[int, int]:
    """Find first convective-enter and first convective-exit layer.

    ``log_p``, ``grad_rad``, ``gamma_ad`` are per-profile 1-D arrays.
    A layer counts as convective when ``grad_rad - gamma_ad > 0`` and the
    condition holds for ``persistence`` consecutive layers starting there.
    Returns ``(l_enter, l_exit)`` with ``l_enter = -1`` when never convective
    and ``l_exit = layers`` when convection never ends before the grid end.
    """

    convective = grad_rad - gamma_ad > 0.0
    enter = -1
    exit_layer = int(log_p.size)
    for layer in range(start_layer, int(log_p.size) - persistence + 1):
        if enter == -1 and bool(np.all(convective[layer : layer + persistence])):
            enter = layer
        elif enter != -1 and bool(np.all(~convective[layer : layer + persistence])):
            exit_layer = layer
            break
    return enter, exit_layer


def dual_crossing_gradient(
    log_p: np.ndarray,
    log_p_enter: float,
    log_p_exit: float,
    grad_rad: np.ndarray,
    gamma_ad: np.ndarray,
    a_enter: float,
    a_exit: float,
    *,
    width_dex: float = SWITCH_WIDTH_DEX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Superadiabatic radiative + adiabat + two boundary bumps in logP."""

    w_enter = _logistic(log_p, log_p_enter, width_dex)
    w_exit = _logistic(log_p, log_p_exit, width_dex)
    w_conv = w_enter * (1.0 - w_exit)
    gradient = (
        (1.0 - w_conv) * grad_rad
        + w_conv * gamma_ad
        + a_enter * w_enter * (1.0 - w_enter)
        + a_exit * w_exit * (1.0 - w_exit)
    )
    return gradient, w_enter, w_exit


def radiative_gradient(
    kappa: np.ndarray,
    pressure: np.ndarray,
    teff: np.ndarray,
    gravity: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """Radiative temperature gradient on the fixed grid (clipped)."""

    teff_value = float(np.atleast_1d(np.asarray(teff, dtype=np.float64)).reshape(-1)[0])
    gravity_value = float(np.atleast_1d(np.asarray(gravity, dtype=np.float64)).reshape(-1)[0])
    values = 3.0 * kappa * pressure * teff_value**4
    values = values / (16.0 * gravity_value * temperature**4)
    return np.clip(values, 0.0, 50.0)


def integrate_gradient(
    log_p: np.ndarray, gradient: np.ndarray, ln_t_surface: np.ndarray
) -> np.ndarray:
    """Integrate d ln T / d ln P from a surface temperature.

    ``ln_t_surface`` is the natural log of the surface temperature; the
    returned array is physical temperature on the ``tau`` grid.
    """

    dln_p = np.gradient(log_p) * np.log(10.0)
    ln_t = np.empty_like(gradient)
    ln_t[0] = float(ln_t_surface)
    ln_t[1:] = ln_t[0] + np.cumsum(gradient[:-1] * dln_p[:-1])
    return np.exp(ln_t)



def _build_depth_profiles(
    labels: np.ndarray, tau: np.ndarray, parameters: EntropyClosureV2Parameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict (log10_kappa, radiative_temperature, surface_mass) from labels."""

    log_tau = np.log10(np.maximum(np.asarray(tau, dtype=np.float64), 1.0e-12))
    depth_basis = chebyshev_basis(log_tau, max(OPACITY_MODES, TEMPERATURE_MODES))[:, :OPACITY_MODES]

    opacity_terms, _, _ = polynomial_features(
        labels, _OPACITY_EXPONENTS,
        center=parameters.feature_center, scale=parameters.feature_scale,
    )
    temperature_terms, _, _ = polynomial_features(
        labels, _TEMPERATURE_EXPONENTS,
        center=parameters.feature_center, scale=parameters.feature_scale,
    )
    surface_terms, _, _ = polynomial_features(
        labels, _SURFACE_MASS_EXPONENTS,
        center=parameters.feature_center, scale=parameters.feature_scale,
    )

    mode_weights = opacity_terms @ parameters.opacity_coefficients.T  # (N, M)
    log_kappa = depth_basis @ mode_weights.T  # (L, N)
    log_kappa = np.clip(log_kappa.T, -30.0, 30.0)  # (N, L)

    temperature_residual = (
        temperature_terms @ parameters.temperature_coefficients.T
    )  # (N, 4)
    delta_temperature = depth_basis[:, :TEMPERATURE_MODES] @ temperature_residual.T  # (L, N)
    delta_temperature = delta_temperature.T  # (N, L)

    effective_temperature = labels[:, 0]
    radiative_temperature = effective_temperature[:, None] * (
        0.75 * (tau[None, :] + 2.0 / 3.0)
    ) ** 0.25 * 10.0 ** np.clip(delta_temperature, -1.0, 1.0)

    log_surface_mass = surface_terms @ parameters.surface_mass_coefficients
    surface_mass = 10.0 ** np.clip(log_surface_mass, -30.0, 30.0)
    return log_kappa, radiative_temperature, surface_mass


def _integrate_mass(
    tau: np.ndarray, log_kappa: np.ndarray, surface_mass: np.ndarray
) -> np.ndarray:
    """Trapezoid dm/dtau = 1/kappa, keeping mass strictly monotonic.

    Every term is positive by construction: a positive surface mass plus a
    cumulative sum of positive increments.  No blanket floor or re-sort is
    applied, because either could manufacture plateaus that break strict
    monotonicity.
    """

    opacity = np.clip(10.0 ** np.clip(log_kappa, -15.0, 15.0), 1.0e-30, 1.0e15)
    surface = np.maximum(np.asarray(surface_mass, dtype=np.float64), 1.0e-33)
    mass = np.empty_like(opacity)
    mass[:, 0] = surface
    increments = 0.5 * np.diff(tau)[None, :] * (
        1.0 / opacity[:, 1:] + 1.0 / opacity[:, :-1]
    )
    mass[:, 1:] = surface[:, None] + np.cumsum(increments, axis=1)
    return mass


def predict_compact_reduced_state(
    labels: np.ndarray,
    tau: np.ndarray,
    parameters: EntropyClosureV2Parameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return ``(column_mass, temperature, log10_opacity, diagnostics)``.

    Per-star flow: predict opacity/radiative-T/surface-mass from labels,
    integrate column mass from opacity, read pressure P=g*m, scan the
    Schwarzschild crossing on the predicted deep gradient, mix the dual
    crossing into a final gradient, and integrate temperature.
    """

    single = np.asarray(labels, dtype=np.float64).ndim == 1
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    depth = np.asarray(tau, dtype=np.float64)

    log_kappa, radiative_temperature, surface_mass = _build_depth_profiles(
        values, depth, parameters
    )
    mass = _integrate_mass(depth, log_kappa, surface_mass)

    gravity = 10.0 ** values[:, 1]
    pressure = gravity[:, None] * mass

    log_p = np.log10(np.maximum(pressure, 1.0e-30))
    temperature = np.empty_like(radiative_temperature)
    enter_layers = np.full(values.shape[0], -1, dtype=np.int64)
    exit_layers = np.full(values.shape[0], depth.size, dtype=np.int64)
    has_convection = np.zeros(values.shape[0], dtype=bool)
    has_exit = np.zeros(values.shape[0], dtype=bool)

    closure_terms, _, _ = polynomial_features(
        values, _CLOSURE_EXPONENTS,
        center=parameters.feature_center, scale=parameters.feature_scale,
    )
    gamma_ad = _label_scalars_impl(closure_terms, parameters.gamma_ad_coefficients, 0.10, 0.45)
    a_enter = _label_scalars_impl(closure_terms, parameters.a_enter_coefficients, 0.0, AE_MAX)
    a_exit = _label_scalars_impl(closure_terms, parameters.a_exit_coefficients, AX_MIN, AX_MAX)
    exit_log_p = None
    if parameters.exit_logp_coefficients is not None:
        exit_log_p = _label_scalars_impl(
            closure_terms, parameters.exit_logp_coefficients, -12.0, 12.0
        )

    for row in range(values.shape[0]):
        grad_rad = radiative_gradient(
            10.0 ** log_kappa[row], pressure[row],
            values[row, 0], gravity[row], radiative_temperature[row],
        )
        enter_layer, exit_layer = schwarzschild_crossings(log_p[row], grad_rad, gamma_ad[row])
        enter_layers[row] = enter_layer
        exit_layers[row] = exit_layer
        has_convection[row] = enter_layer >= 0
        has_exit[row] = exit_layer < depth.size

        if enter_layer < 0:
            gradient = grad_rad
            w_enter = np.zeros_like(grad_rad)
            w_exit = np.zeros_like(grad_rad)
        else:
            if exit_layer >= depth.size:
                if exit_log_p is not None:
                    lp_exit = float(exit_log_p[row])
                else:
                    lp_exit = float(log_p[row, -1]) + 3.0
            else:
                lp_exit = log_p[row, exit_layer]
            gradient, w_enter, w_exit = dual_crossing_gradient(
                log_p[row], log_p[row, enter_layer], lp_exit,
                grad_rad, gamma_ad[row], a_enter[row], a_exit[row],
            )
        temperature[row] = integrate_gradient(log_p[row], gradient, np.log(radiative_temperature[row, 0]))

    d_log_p = np.gradient(log_p, axis=1)
    d_ln_t = np.gradient(np.log(radiative_temperature), axis=1)
    diagnostics = {
        "enter_layer": enter_layers,
        "exit_layer": exit_layers,
        "has_convection": has_convection,
        "has_exit": has_exit,
        "log_p": log_p,
        "grad_rad": np.divide(
            d_ln_t, d_log_p, out=np.zeros_like(d_ln_t), where=d_log_p != 0.0
        ),
    }

    def _collapse(field):
        return field[0] if single else field

    return (
        _collapse(mass),
        _collapse(temperature),
        _collapse(log_kappa),
        {key: _collapse(value) for key, value in diagnostics.items()},
    )


def _label_scalars_impl(
    closure_terms: np.ndarray,
    coefficients: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Evaluate a closure parameter bounded to ``[low, high]`` per star."""

    raw = closure_terms @ coefficients.astype(np.float64)
    return low + (high - low) * _logistic(raw, 0.0, 1.0)


def _encode_hash(value: str) -> str:
    return str(value)


def save_entropy_closure_v2(
    path: Path | str, parameters: EntropyClosureV2Parameters
) -> Path:
    """Save fitted v2 constants as a portable, pickle-free NumPy asset."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": np.asarray(FORMAT_MARKER),
        "feature_center": np.asarray(parameters.feature_center, dtype=np.float64),
        "feature_scale": np.asarray(parameters.feature_scale, dtype=np.float64),
        "opacity_coefficients": np.asarray(parameters.opacity_coefficients, dtype=np.float64),
        "temperature_coefficients": np.asarray(parameters.temperature_coefficients, dtype=np.float64),
        "surface_mass_coefficients": np.asarray(parameters.surface_mass_coefficients, dtype=np.float64),
        "gamma_ad_coefficients": np.asarray(parameters.gamma_ad_coefficients, dtype=np.float64),
        "a_enter_coefficients": np.asarray(parameters.a_enter_coefficients, dtype=np.float64),
        "a_exit_coefficients": np.asarray(parameters.a_exit_coefficients, dtype=np.float64),
        "corpus_sha256": np.asarray(parameters.corpus_sha256),
        "model_spec_sha256": np.asarray(parameters.model_spec_sha256),
    }
    if parameters.exit_logp_coefficients is not None:
        payload["exit_logp_coefficients"] = np.asarray(
            parameters.exit_logp_coefficients, dtype=np.float64
        )
    np.savez_compressed(destination, **payload)
    return destination


def load_entropy_closure_v2(path: Path | str) -> EntropyClosureV2Parameters:
    """Load a v2 coefficient asset and validate its format marker."""

    with np.load(Path(path), allow_pickle=False) as data:
        marker = str(np.asarray(data["format"]).item())
        if marker != FORMAT_MARKER:
            raise ValueError(f"unsupported entropy closure v2 format: {marker}")
        exit_log_p = None
        if "exit_logp_coefficients" in data:
            exit_log_p = np.asarray(data["exit_logp_coefficients"], dtype=np.float64)
        return EntropyClosureV2Parameters(
            feature_center=np.asarray(data["feature_center"], dtype=np.float64),
            feature_scale=np.asarray(data["feature_scale"], dtype=np.float64),
            opacity_coefficients=np.asarray(data["opacity_coefficients"], dtype=np.float64),
            temperature_coefficients=np.asarray(data["temperature_coefficients"], dtype=np.float64),
            surface_mass_coefficients=np.asarray(data["surface_mass_coefficients"], dtype=np.float64),
            gamma_ad_coefficients=np.asarray(data["gamma_ad_coefficients"], dtype=np.float64),
            a_enter_coefficients=np.asarray(data["a_enter_coefficients"], dtype=np.float64),
            a_exit_coefficients=np.asarray(data["a_exit_coefficients"], dtype=np.float64),
            exit_logp_coefficients=exit_log_p,
            corpus_sha256=_encode_hash(str(np.asarray(data["corpus_sha256"]).item())),
            model_spec_sha256=_encode_hash(str(np.asarray(data["model_spec_sha256"]).item())),
        )


def model_spec_sha256() -> str:
    """Hash the fixed structural choices (does not depend on fitted values)."""

    spec = (
        "opacity_degree=3,opacity_modes=8,temperature_degree=2,temperature_modes=4,"
        "surface_mass_degree=2,closure_degree=1,width_dex=0.30,"
        "gamma_ad=[0.10,0.45],a_enter=[0,0.50],a_exit=[-0.50,0.50],"
        "chebyshev=24,layers=80"
    )
    import hashlib

    return hashlib.sha256(spec.encode("utf-8")).hexdigest()


def sample_constant_parameters(seed: int = 0) -> EntropyClosureV2Parameters:
    """Deterministic placeholder constants for smoke tests (NOT a fit)."""

    rng = np.random.default_rng(seed)
    return EntropyClosureV2Parameters(
        feature_center=np.zeros(5, dtype=np.float64),
        feature_scale=np.ones(5, dtype=np.float64),
        opacity_coefficients=rng.normal(0.0, 0.1, size=(OPACITY_MODES, _OPACITY_EXPONENTS.shape[0])),
        temperature_coefficients=rng.normal(0.0, 0.02, size=(TEMPERATURE_MODES, _TEMPERATURE_EXPONENTS.shape[0])),
        surface_mass_coefficients=rng.normal(8.0, 0.5, size=_SURFACE_MASS_EXPONENTS.shape[0]),
        gamma_ad_coefficients=np.zeros(_CLOSURE_EXPONENTS.shape[0]),
        a_enter_coefficients=np.zeros(_CLOSURE_EXPONENTS.shape[0]),
        a_exit_coefficients=np.zeros(_CLOSURE_EXPONENTS.shape[0]),
        corpus_sha256="smoke-unfit",
        model_spec_sha256=model_spec_sha256(),
    )
