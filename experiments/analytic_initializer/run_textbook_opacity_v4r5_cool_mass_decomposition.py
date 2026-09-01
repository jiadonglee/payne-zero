"""Preregistered v4r5 cool-mass surface-versus-in-domain decomposition.

This runner does not change v4r5, the mass integral, or any gate.  It only
splits the existing cool true-(P, T) mass residual into outer-layer and
in-domain pieces.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    file_sha256,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.profile_closure import integrate_mass_from_opacity
from experiments.analytic_initializer.run_textbook_opacity_offline import (
    MANIFESTS,
    _band_masks,
    _metrics,
)
from experiments.analytic_initializer.textbook_opacity import (
    COMPONENT_NAMES_V4R5,
    DEFAULT_TEXTBOOK_CONSTANTS,
    V4R5_FORMAL_TEMPERATURE_FLOOR_K,
    WINDOW_NAMES,
    textbook_opacity_node_components_v4r5,
)


TEMPERATURE_FLOOR_K = V4R5_FORMAL_TEMPERATURE_FLOOR_K
COOL_TEFF_MAX_K = 6000.0
MASS_GATE_LIMIT_DEX = 0.20
EXPECTED_V4R5_COOL_MASS_P95_DEX = 0.2375
REPRODUCE_TOLERANCE_DEX = 0.002
SANITY_TRUTH_KAPPA_P95_MAX_DEX = 0.05
SURFACE_FRACTION_LICENSE_MIN = 0.50
PRODUCTION_MISS_SIGNED_MEDIAN_MAX_DEX = -0.05
PRODUCTION_MATCH_ABS_DEX = 0.05
EXPECTED_VALIDATION_STAR_COUNT = 10228
SPLIT_SEED = 20260816
LAYER_SLICES = (
    ("below_3200K", 0.0, 3200.0),
    ("3200_4000K", 3200.0, 4000.0),
    ("4000_5000K", 4000.0, 5000.0),
    ("at_least_4000K", 4000.0, np.inf),
)
PRODUCTION_FLAG_NAMES = {
    0: "H_bf_ff",
    4: "He_I",
    5: "He_II",
    8: "C_Mg_Al_Si_Fe_plus_CIA",
    9: "lukewarm_metals",
    10: "hot_metals",
}
NAMED_V4R6_CONSTRUCTIONS = {
    "H_bf_ff": (
        "Bell & Berrington (1987) H-minus free-free in the "
        "below-H-minus-threshold window for T<4000 K only; published "
        "quantum free-free factors, not a corpus fit. John (1988) H-minus "
        "remains frozen at T>=4000 K. Molecular bands stay off."
    ),
    "C_Mg_Al_Si_Fe_plus_CIA": (
        "Published metal bound-free/free-free (C, Mg, Al, Si, Fe) for "
        "T<4000 K only; not CIA and not H2O/CO/TiO bands; not a corpus fit."
    ),
    "lukewarm_metals": (
        "ATLAS lukewarm-metal continuum (Flag 9) for T<4000 K only; "
        "published metal bf/ff, not a corpus fit. Molecular bands stay off."
    ),
}
V4R5_OFFLINE = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r5_offline_validation_20260828.json"
)
HOT_FLAG_ABLATION = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r4_hot_flag_ablation_20260828.json"
)
MOLECULE_ABLATION = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r1_molecule_ablation_20260827.json"
)
REGISTERED_OUTPUT = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r5_cool_mass_decomposition_20260828.json"
)


def first_layer_at_or_above(
    temperature: np.ndarray,
    floor_K: float = TEMPERATURE_FLOOR_K,
) -> np.ndarray:
    """Return the first layer index with ``T >= floor_K`` on each star.

    Stars with no such layer receive ``n_layers``, so a mask
    ``layer >= start`` is empty.
    """

    thermal = np.asarray(temperature, dtype=np.float64)
    if thermal.ndim != 2:
        raise ValueError("temperature must have shape (N, layers)")
    above = thermal >= float(floor_K)
    start = np.argmax(above, axis=1).astype(np.int64)
    none = ~np.any(above, axis=1)
    return np.where(none, thermal.shape[1], start).astype(np.int64)


def blend_log_opacity_by_temperature(
    log_inner: np.ndarray,
    log_outer: np.ndarray,
    temperature: np.ndarray,
    floor_K: float = TEMPERATURE_FLOOR_K,
) -> np.ndarray:
    """Use ``log_inner`` on ``T >= floor`` and ``log_outer`` below it."""

    inner = np.asarray(log_inner, dtype=np.float64)
    outer = np.asarray(log_outer, dtype=np.float64)
    thermal = np.asarray(temperature, dtype=np.float64)
    if inner.shape != outer.shape or inner.shape != thermal.shape:
        raise ValueError("log opacity and temperature shapes must match")
    return np.where(thermal >= float(floor_K), inner, outer)


def integrate_mass_from_start_layer(
    tau: np.ndarray,
    log_opacity: np.ndarray,
    start_index: np.ndarray,
) -> np.ndarray:
    """Trapezoid ``dm = dtau / kappa`` starting at ``start_index``.

    The seed at the start layer is ``m = tau[k] / kappa[k]``.  Layers
    before the start are NaN.  When every start index is 0 this matches
    ``integrate_mass_from_opacity``.
    """

    depth = np.asarray(tau, dtype=np.float64)
    log_kappa = np.asarray(log_opacity, dtype=np.float64)
    start = np.asarray(start_index, dtype=np.int64)
    if log_kappa.ndim != 2 or log_kappa.shape[1] != depth.size:
        raise ValueError("log_opacity must have shape (N, len(tau))")
    if start.shape != (log_kappa.shape[0],):
        raise ValueError("start_index must have one value per profile")
    if np.any(np.diff(depth) <= 0.0):
        raise ValueError("tau must be strictly increasing")
    opacity = 10.0 ** np.clip(log_kappa, -30.0, 30.0)
    n_stars, n_layers = opacity.shape
    increments = np.zeros_like(opacity)
    increments[:, 1:] = 0.5 * np.diff(depth)[None, :] * (
        1.0 / opacity[:, 1:] + 1.0 / opacity[:, :-1]
    )
    csum = np.cumsum(increments, axis=1)
    safe = (start >= 0) & (start < n_layers)
    rows = np.flatnonzero(safe)
    seeded = np.full(n_stars, np.nan)
    start_csum = np.full(n_stars, np.nan)
    if rows.size:
        k = start[safe]
        seeded[safe] = depth[k] / opacity[rows, k]
        start_csum[safe] = csum[rows, k]
    layer = np.arange(n_layers)
    active = (layer[None, :] >= start[:, None]) & safe[:, None]
    return np.where(active, seeded[:, None] + (csum - start_csum[:, None]), np.nan)


def oracle_boundary_column_mass(
    predicted_mass: np.ndarray,
    stored_mass: np.ndarray,
    start_index: np.ndarray,
) -> np.ndarray:
    """Replace the predicted column at the domain edge with the stored column.

    ``m_cf[i] = m_stored[k] + (m_pred[i] - m_pred[k])`` for ``i >= k``.
    """

    predicted = np.asarray(predicted_mass, dtype=np.float64)
    stored = np.asarray(stored_mass, dtype=np.float64)
    start = np.asarray(start_index, dtype=np.int64)
    if predicted.shape != stored.shape:
        raise ValueError("predicted and stored mass shapes must match")
    if start.shape != (predicted.shape[0],):
        raise ValueError("start_index must have one value per profile")
    n_stars, n_layers = predicted.shape
    safe = (start >= 0) & (start < n_layers)
    rows = np.flatnonzero(safe)
    predicted_k = np.full(n_stars, np.nan)
    stored_k = np.full(n_stars, np.nan)
    if rows.size:
        k = start[safe]
        predicted_k[safe] = predicted[rows, k]
        stored_k[safe] = stored[rows, k]
    column = stored_k[:, None] + (predicted - predicted_k[:, None])
    layer = np.arange(n_layers)
    active = (layer[None, :] >= start[:, None]) & safe[:, None]
    return np.where(active, column, np.nan)


def local_increment_residual(
    predicted_mass: np.ndarray,
    stored_mass: np.ndarray,
) -> np.ndarray:
    """Layer-local ``log10(dm)`` residual; layer 0 is NaN."""

    predicted = np.asarray(predicted_mass, dtype=np.float64)
    stored = np.asarray(stored_mass, dtype=np.float64)
    if predicted.shape != stored.shape or predicted.ndim != 2:
        raise ValueError("mass arrays must share shape (N, layers)")
    residual = np.full(predicted.shape, np.nan, dtype=np.float64)
    dm_predicted = np.diff(predicted, axis=1)
    dm_stored = np.diff(stored, axis=1)
    residual[:, 1:] = np.log10(np.maximum(dm_predicted, 1.0e-300)) - np.log10(
        np.maximum(dm_stored, 1.0e-300)
    )
    return residual


def wholly_in_domain_increment_mask(
    temperature: np.ndarray,
    floor_K: float = TEMPERATURE_FLOOR_K,
) -> np.ndarray:
    """True where both endpoints of an increment have ``T >= floor_K``."""

    thermal = np.asarray(temperature, dtype=np.float64)
    if thermal.ndim != 2:
        raise ValueError("temperature must have shape (N, layers)")
    mask = np.zeros(thermal.shape, dtype=bool)
    mask[:, 1:] = (thermal[:, 1:] >= float(floor_K)) & (
        thermal[:, :-1] >= float(floor_K)
    )
    return mask


def crossing_increment_mask(
    temperature: np.ndarray,
    floor_K: float = TEMPERATURE_FLOOR_K,
) -> np.ndarray:
    """True where an increment crosses ``floor_K`` from below."""

    thermal = np.asarray(temperature, dtype=np.float64)
    if thermal.ndim != 2:
        raise ValueError("temperature must have shape (N, layers)")
    mask = np.zeros(thermal.shape, dtype=bool)
    mask[:, 1:] = (thermal[:, 1:] >= float(floor_K)) & (
        thermal[:, :-1] < float(floor_K)
    )
    return mask


def cool_star_mask(teff: np.ndarray) -> np.ndarray:
    return np.asarray(teff, dtype=np.float64) < COOL_TEFF_MAX_K


def cool_gate_mask(
    teff: np.ndarray,
    temperature: np.ndarray,
    floor_K: float = TEMPERATURE_FLOOR_K,
) -> np.ndarray:
    """The v4r5 cool mass-gate mask: ``Teff < 6000`` and ``T >= floor``."""

    thermal = np.asarray(temperature, dtype=np.float64)
    return cool_star_mask(teff)[:, None] & (thermal >= float(floor_K))


def log_mass_residual(predicted_mass: np.ndarray, stored_mass: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted_mass, dtype=np.float64)
    stored = np.asarray(stored_mass, dtype=np.float64)
    return np.log10(np.maximum(predicted, 1.0e-300)) - np.log10(
        np.maximum(stored, 1.0e-300)
    )


def decide_cool_mass_decomposition(
    *,
    surface_p95_dex: float,
    hybrid_p95_dex: float,
    oracle_p95_dex: float,
    in_domain_increment_p95_dex: float,
    truth_kappa_p95_dex: float,
    expected_surface_p95_dex: float | None = EXPECTED_V4R5_COOL_MASS_P95_DEX,
    reproduce_tolerance_dex: float = REPRODUCE_TOLERANCE_DEX,
    mass_limit_dex: float = MASS_GATE_LIMIT_DEX,
    sanity_truth_kappa_p95_max_dex: float = SANITY_TRUTH_KAPPA_P95_MAX_DEX,
) -> dict[str, object]:
    """Return the preregistered cool-mass mechanism verdict."""

    values = {
        "surface_p95_dex": float(surface_p95_dex),
        "hybrid_p95_dex": float(hybrid_p95_dex),
        "oracle_p95_dex": float(oracle_p95_dex),
        "in_domain_increment_p95_dex": float(in_domain_increment_p95_dex),
        "truth_kappa_p95_dex": float(truth_kappa_p95_dex),
    }
    finite = all(np.isfinite(value) for value in values.values())
    hybrid_pass = bool(values["hybrid_p95_dex"] <= mass_limit_dex)
    oracle_pass = bool(values["oracle_p95_dex"] <= mass_limit_dex)
    increment_pass = bool(values["in_domain_increment_p95_dex"] <= mass_limit_dex)
    truth_sanity_pass = bool(
        values["truth_kappa_p95_dex"] <= sanity_truth_kappa_p95_max_dex
    )
    reproduced = True
    if expected_surface_p95_dex is not None:
        reproduced = bool(
            abs(values["surface_p95_dex"] - float(expected_surface_p95_dex))
            <= float(reproduce_tolerance_dex)
        )
    gate_still_fails = bool(values["surface_p95_dex"] > mass_limit_dex)
    excess = values["surface_p95_dex"] - float(mass_limit_dex)
    explained_excess = values["surface_p95_dex"] - values["hybrid_p95_dex"]
    if excess > 0.0:
        explained_fraction = explained_excess / excess
    else:
        explained_fraction = float("nan")
    inconclusive_reason = None
    if not finite:
        verdict = "INCONCLUSIVE"
        inconclusive_reason = "non_finite_primary_p95"
    elif not truth_sanity_pass:
        verdict = "INCONCLUSIVE"
        inconclusive_reason = "stored_kappa_integral_does_not_recover_mass"
    elif not reproduced:
        verdict = "INCONCLUSIVE"
        inconclusive_reason = "did_not_reproduce_v4r5_cool_mass_p95"
    elif not gate_still_fails:
        verdict = "INCONCLUSIVE"
        inconclusive_reason = "cool_mass_gate_already_passes"
    elif hybrid_pass and oracle_pass and increment_pass:
        verdict = "SURFACE_INTEGRAL_DOMINATED"
    elif (not hybrid_pass) and (not oracle_pass) and (not increment_pass):
        verdict = "IN_DOMAIN_COOL_OPACITY_DOMINATED"
    else:
        verdict = "MIXED"
    return {
        "verdict": verdict,
        "inconclusive_reason": inconclusive_reason,
        "hybrid_pass": hybrid_pass,
        "oracle_pass": oracle_pass,
        "increment_pass": increment_pass,
        "truth_kappa_sanity_pass": truth_sanity_pass,
        "reproduced_v4r5_cool_mass_p95": reproduced,
        "gate_still_fails": gate_still_fails,
        "mass_limit_dex": float(mass_limit_dex),
        "explained_excess_dex": float(explained_excess),
        "explained_fraction_of_p95_excess": (
            None if not np.isfinite(explained_fraction) else float(explained_fraction)
        ),
        **values,
    }


def decide_v4r6_license(
    *,
    verdict: str,
    explained_fraction: float | None,
    v4r5_minus_production_signed_median_dex: float | None,
    v4r5_minus_stored_signed_median_dex: float | None,
    dominant_production_flag_name: str | None,
    surface_fraction_min: float = SURFACE_FRACTION_LICENSE_MIN,
    production_miss_max_dex: float = PRODUCTION_MISS_SIGNED_MEDIAN_MAX_DEX,
    production_match_abs_dex: float = PRODUCTION_MATCH_ABS_DEX,
) -> dict[str, object]:
    """Return whether a later T<4000 continuum candidate is licensed."""

    named = NAMED_V4R6_CONSTRUCTIONS.get(str(dominant_production_flag_name or ""))
    licensed = False
    reason = "not_evaluated"
    if verdict not in ("SURFACE_INTEGRAL_DOMINATED", "MIXED"):
        reason = "verdict_is_not_surface_or_mixed"
    elif explained_fraction is None or not np.isfinite(explained_fraction):
        reason = "explained_fraction_unavailable"
    elif float(explained_fraction) < float(surface_fraction_min):
        reason = "surface_explained_fraction_below_0.50"
    elif v4r5_minus_production_signed_median_dex is None or not np.isfinite(
        v4r5_minus_production_signed_median_dex
    ):
        reason = "missing_20star_production_continuum_comparison"
    elif abs(float(v4r5_minus_production_signed_median_dex)) < float(
        production_match_abs_dex
    ):
        stored = v4r5_minus_stored_signed_median_dex
        if stored is not None and np.isfinite(stored) and float(stored) <= float(
            production_miss_max_dex
        ):
            reason = "LINES_DOMINATE_OUTER"
        else:
            reason = "no_continuum_miss_versus_production"
    elif float(v4r5_minus_production_signed_median_dex) > float(production_miss_max_dex):
        reason = "v4r5_is_not_low_versus_production_continuum"
    elif named is None:
        reason = (
            "no_named_cool_Tlt4000_construction_for_flag_"
            f"{dominant_production_flag_name}"
        )
    else:
        licensed = True
        reason = "licensed_named_Tlt4000_continuum"
    return {
        "licensed": licensed,
        "reason": reason,
        "named_construction": named if licensed else None,
        "dominant_production_flag_name": dominant_production_flag_name,
        "thresholds": {
            "surface_fraction_min": float(surface_fraction_min),
            "production_miss_signed_median_max_dex": float(production_miss_max_dex),
            "production_match_abs_dex": float(production_match_abs_dex),
        },
    }


def _metrics_or_none(residual: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return _metrics(values)


def _p95(metrics: Mapping[str, float] | None) -> float:
    if metrics is None:
        return float("nan")
    return float(metrics["p95_dex"])


def _signed_median(metrics: Mapping[str, float] | None) -> float | None:
    if metrics is None:
        return None
    return float(metrics["signed_median_dex"])


def _rosseland_diagnostics(
    components: dict[str, np.ndarray],
    component_names: tuple[str, ...] = COMPONENT_NAMES_V4R5,
) -> tuple[np.ndarray, np.ndarray]:
    weights = components["node_weights"]
    total = np.maximum(components["total"], 1.0e-30)
    inverse_by_window = np.sum(weights / total, axis=-1)
    inverse_window_fraction = inverse_by_window / np.maximum(
        np.sum(inverse_by_window, axis=-1, keepdims=True),
        1.0e-300,
    )
    inverse_total = np.sum(weights / total, axis=(-2, -1))
    sensitivity = np.stack(
        [
            np.sum(
                weights * components[name] / total**2,
                axis=(-2, -1),
            )
            / np.maximum(inverse_total, 1.0e-300)
            for name in component_names
        ],
        axis=-1,
    )
    return inverse_window_fraction, sensitivity


def _fraction_metrics(values: np.ndarray) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5.0)),
        "p95": float(np.percentile(array, 95.0)),
    }


def _jsonify(value: object) -> object:
    """Convert numpy scalars and non-finite floats so the JSON dump is strict."""

    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    return value


def _dominant_name(metrics_by_name: Mapping[str, Mapping[str, float] | None]) -> str | None:
    best_name = None
    best_value = -np.inf
    for name, row in metrics_by_name.items():
        if row is None:
            continue
        value = float(row["median"])
        if value > best_value:
            best_value = value
            best_name = str(name)
    return best_name


def _batch_prediction(
    labels: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    prediction = np.empty_like(temperature, dtype=np.float64)
    component_sensitivity = {
        name: np.empty_like(temperature, dtype=np.float64)
        for name in COMPONENT_NAMES_V4R5
    }
    inverse_window_fraction = {
        name: np.empty_like(temperature, dtype=np.float64)
        for name in WINDOW_NAMES
    }
    for start in range(0, labels.shape[0], int(batch_size)):
        stop = min(labels.shape[0], start + int(batch_size))
        components = textbook_opacity_node_components_v4r5(
            labels[start:stop],
            temperature[start:stop],
            pressure[start:stop],
        )
        weights = components["node_weights"]
        total = np.maximum(components["total"], 1.0e-30)
        prediction[start:stop] = 1.0 / np.sum(weights / total, axis=(-2, -1))
        window_fraction, sensitivity = _rosseland_diagnostics(components)
        for index, name in enumerate(COMPONENT_NAMES_V4R5):
            component_sensitivity[name][start:stop] = sensitivity[..., index]
        for index, name in enumerate(WINDOW_NAMES):
            inverse_window_fraction[name][start:stop] = window_fraction[..., index]
        print(f"processed {stop}/{labels.shape[0]} stars", flush=True)
    return prediction, component_sensitivity, inverse_window_fraction


def _slice_report(
    residual: np.ndarray,
    teff: np.ndarray,
    temperature: np.ndarray,
    *,
    cool_only: bool,
) -> dict[str, dict[str, float] | None]:
    star_mask = cool_star_mask(teff) if cool_only else np.ones(teff.shape[0], dtype=bool)
    rows: dict[str, dict[str, float] | None] = {}
    for name, lower, upper in LAYER_SLICES:
        mask = star_mask[:, None] & (temperature >= lower) & (temperature < upper)
        rows[name] = _metrics_or_none(residual[mask])
    return rows


def _component_slice(
    values: dict[str, np.ndarray],
    teff: np.ndarray,
    temperature: np.ndarray,
    *,
    lower: float,
    upper: float,
    cool_only: bool,
) -> dict[str, dict[str, float] | None]:
    star_mask = cool_star_mask(teff) if cool_only else np.ones(teff.shape[0], dtype=bool)
    mask = star_mask[:, None] & (temperature >= lower) & (temperature < upper)
    return {name: _fraction_metrics(array[mask]) for name, array in values.items()}


def _hot_grid_cool_continuum(
    corpus,
    ablation: dict[str, object],
    *,
    batch_size: int,
) -> dict[str, object]:
    references = ablation["references"]
    indices = np.asarray(
        [int(row["corpus_index"]) for row in references], dtype=np.int64
    )
    labels = corpus.labels[indices]
    temperature = corpus.temperature[indices]
    pressure = corpus.gas_pressure[indices]
    stored = corpus.rosseland_opacity[indices]
    production = np.asarray(
        [row["production_continuum_baseline"] for row in references],
        dtype=np.float64,
    )
    stored_temperature = np.asarray(
        [row["temperature_K"] for row in references], dtype=np.float64
    )
    if stored_temperature.shape != temperature.shape:
        raise ValueError("hot-grid stored temperatures do not match the corpus")
    if np.max(np.abs(temperature - stored_temperature)) > 1.0e-6:
        raise ValueError("hot-grid corpus temperatures drifted from the ablation JSON")
    prediction, _, _ = _batch_prediction(
        labels,
        temperature,
        pressure,
        batch_size=batch_size,
    )
    v4r5_minus_stored = np.log10(prediction) - np.log10(stored)
    v4r5_minus_production = np.log10(prediction) - np.log10(
        np.maximum(production, 1.0e-300)
    )
    production_minus_stored = np.log10(np.maximum(production, 1.0e-300)) - np.log10(
        stored
    )
    cool = cool_star_mask(labels[:, 0])
    outer = (temperature >= 3200.0) & (temperature < TEMPERATURE_FLOOR_K)
    slice_mask = cool[:, None] & outer
    flag_metrics: dict[str, dict[str, float] | None] = {}
    for flag, name in PRODUCTION_FLAG_NAMES.items():
        effect = np.asarray(
            [row["flag_effect_dex"][str(flag)] for row in references],
            dtype=np.float64,
        )
        flag_metrics[name] = _metrics_or_none(effect[slice_mask])
    dominant_flag = None
    dominant_value = -np.inf
    for name, row in flag_metrics.items():
        if row is None:
            continue
        value = float(row["signed_median_dex"])
        if value > dominant_value:
            dominant_value = value
            dominant_flag = name
    return {
        "reference_count": int(indices.size),
        "cool_star_count": int(np.sum(cool)),
        "cool_3200_4000K_layer_count": int(np.sum(slice_mask)),
        "reference_indices": [int(index) for index in indices],
        "cool_reference_indices": [int(index) for index in indices[cool]],
        "v4r5_minus_stored_total": _metrics_or_none(v4r5_minus_stored[slice_mask]),
        "v4r5_minus_production_continuum": _metrics_or_none(
            v4r5_minus_production[slice_mask]
        ),
        "production_continuum_minus_stored_total": _metrics_or_none(
            production_minus_stored[slice_mask]
        ),
        "production_flag_effect": flag_metrics,
        "dominant_production_flag_name": dominant_flag,
        "dominant_production_flag_signed_median_dex": (
            None if dominant_flag is None else float(dominant_value)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--v4r5-offline", type=Path, default=V4R5_OFFLINE)
    parser.add_argument("--hot-flag-ablation", type=Path, default=HOT_FLAG_ABLATION)
    parser.add_argument("--molecule-ablation", type=Path, default=MOLECULE_ABLATION)
    parser.add_argument("--out", type=Path, default=REGISTERED_OUTPUT)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    for path in (args.v4r5_offline, args.hot_flag_ablation, args.molecule_ablation):
        if not path.is_file():
            raise SystemExit(f"required prior artifact is missing: {path}")

    corpus = load_strict_truth(args.corpus)
    v4r5_offline = json.loads(args.v4r5_offline.read_text(encoding="utf-8"))
    ablation = json.loads(args.hot_flag_ablation.read_text(encoding="utf-8"))
    molecule_ablation = json.loads(args.molecule_ablation.read_text(encoding="utf-8"))
    print("evaluating 20-star production-continuum grid", flush=True)
    hot_grid = _hot_grid_cool_continuum(
        corpus,
        ablation,
        batch_size=args.batch_size,
    )

    excluded, used_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=SPLIT_SEED)
    indices = split.validation
    if args.limit is not None:
        indices = indices[: int(args.limit)]
    labels = corpus.labels[indices]
    temperature = corpus.temperature[indices]
    pressure = corpus.gas_pressure[indices]
    stored_opacity = corpus.rosseland_opacity[indices]
    stored_mass = corpus.column_mass[indices]
    print(
        f"evaluating v4r5 on {int(indices.size)} validation stars",
        flush=True,
    )
    prediction, component_sensitivity, inverse_window_fraction = _batch_prediction(
        labels,
        temperature,
        pressure,
        batch_size=args.batch_size,
    )
    log_prediction = np.log10(prediction)
    log_stored = np.log10(stored_opacity)
    opacity_residual = log_prediction - log_stored
    surface_mass = integrate_mass_from_opacity(corpus.tau, log_prediction)
    truth_kappa_mass = integrate_mass_from_opacity(corpus.tau, log_stored)
    hybrid_mass = integrate_mass_from_opacity(
        corpus.tau,
        blend_log_opacity_by_temperature(
            log_prediction,
            log_stored,
            temperature,
            TEMPERATURE_FLOOR_K,
        ),
    )
    start_index = first_layer_at_or_above(temperature, TEMPERATURE_FLOOR_K)
    restart_predicted = integrate_mass_from_start_layer(
        corpus.tau, log_prediction, start_index
    )
    restart_stored_kappa = integrate_mass_from_start_layer(
        corpus.tau, log_stored, start_index
    )
    oracle_mass = oracle_boundary_column_mass(surface_mass, stored_mass, start_index)
    increment = local_increment_residual(surface_mass, stored_mass)

    cool_stars = cool_star_mask(labels[:, 0])
    gate = cool_gate_mask(labels[:, 0], temperature, TEMPERATURE_FLOOR_K)
    middle_stars = _band_masks(labels[:, 0])["middle_6000_10000K"]
    middle_gate = middle_stars[:, None] & (temperature >= TEMPERATURE_FLOOR_K)
    in_domain_increments = wholly_in_domain_increment_mask(
        temperature, TEMPERATURE_FLOOR_K
    )
    crossing_increments = crossing_increment_mask(temperature, TEMPERATURE_FLOOR_K)
    layer_index = np.arange(temperature.shape[1])
    from_first = cool_stars[:, None] & (layer_index[None, :] >= start_index[:, None])
    nonmonotonic = int(
        np.sum(np.any(gate != from_first, axis=1) & cool_stars)
    )

    surface_metrics = _metrics_or_none(log_mass_residual(surface_mass, stored_mass)[gate])
    hybrid_metrics = _metrics_or_none(log_mass_residual(hybrid_mass, stored_mass)[gate])
    oracle_metrics = _metrics_or_none(log_mass_residual(oracle_mass, stored_mass)[gate])
    increment_metrics = _metrics_or_none(
        increment[cool_stars[:, None] & in_domain_increments]
    )
    truth_kappa_metrics = _metrics_or_none(
        log_mass_residual(truth_kappa_mass, stored_mass)[gate]
    )
    restart_versus_stored = _metrics_or_none(
        log_mass_residual(restart_predicted, stored_mass)[gate]
    )
    restart_versus_truth_restart = _metrics_or_none(
        log_mass_residual(restart_predicted, restart_stored_kappa)[gate]
    )
    full_split = args.limit is None and int(indices.size) == EXPECTED_VALIDATION_STAR_COUNT
    decision = decide_cool_mass_decomposition(
        surface_p95_dex=_p95(surface_metrics),
        hybrid_p95_dex=_p95(hybrid_metrics),
        oracle_p95_dex=_p95(oracle_metrics),
        in_domain_increment_p95_dex=_p95(increment_metrics),
        truth_kappa_p95_dex=_p95(truth_kappa_metrics),
        expected_surface_p95_dex=(
            EXPECTED_V4R5_COOL_MASS_P95_DEX if full_split else None
        ),
    )
    cool_4000_5000_components = _component_slice(
        component_sensitivity,
        labels[:, 0],
        temperature,
        lower=4000.0,
        upper=5000.0,
        cool_only=True,
    )
    cool_3200_4000_components = _component_slice(
        component_sensitivity,
        labels[:, 0],
        temperature,
        lower=3200.0,
        upper=4000.0,
        cool_only=True,
    )
    in_domain_component = _dominant_name(cool_4000_5000_components)
    license_decision = decide_v4r6_license(
        verdict=str(decision["verdict"]),
        explained_fraction=decision["explained_fraction_of_p95_excess"],
        v4r5_minus_production_signed_median_dex=_signed_median(
            hot_grid["v4r5_minus_production_continuum"]
        ),
        v4r5_minus_stored_signed_median_dex=_signed_median(
            hot_grid["v4r5_minus_stored_total"]
        ),
        dominant_production_flag_name=hot_grid["dominant_production_flag_name"],
    )
    middle_surface = _metrics_or_none(
        log_mass_residual(surface_mass, stored_mass)[middle_gate]
    )
    middle_hybrid = _metrics_or_none(
        log_mass_residual(hybrid_mass, stored_mass)[middle_gate]
    )
    middle_increment = _metrics_or_none(
        increment[middle_stars[:, None] & in_domain_increments]
    )
    middle_shares_surface = bool(
        _p95(middle_surface) > MASS_GATE_LIMIT_DEX
        and _p95(middle_hybrid) <= MASS_GATE_LIMIT_DEX
        and _p95(middle_increment) <= MASS_GATE_LIMIT_DEX
    )

    result = {
        "schema_version": 1,
        "candidate": "v4r5_cool_mass_decomposition",
        "version": "v4r5",
        "decision": decision["verdict"],
        "decomposition": decision,
        "v4r6_license": license_decision,
        "in_domain_dominant_component_4000_5000K": in_domain_component,
        "v4r5_offline_result": str(args.v4r5_offline),
        "v4r5_offline_result_sha256": file_sha256(args.v4r5_offline),
        "hot_flag_ablation_result": str(args.hot_flag_ablation),
        "hot_flag_ablation_result_sha256": file_sha256(args.hot_flag_ablation),
        "molecule_ablation_result": str(args.molecule_ablation),
        "molecule_ablation_result_sha256": file_sha256(args.molecule_ablation),
        "molecule_ablation_verdict": molecule_ablation["decision"]["verdict"],
        "v4r5_offline_cool_mass_p95_dex": v4r5_offline["offline_gate"][
            "cool_mass_observed_p95_dex"
        ],
        "corpus": str(corpus.path),
        "corpus_sha256": file_sha256(corpus.path),
        "validation_star_count": int(indices.size),
        "validation_layer_count": int(indices.size * corpus.layers),
        "full_registered_split": full_split,
        "split_seed": split.seed,
        "excluded_count": int(excluded.size),
        "excluded_manifests": used_manifests,
        "constants": asdict(DEFAULT_TEXTBOOK_CONSTANTS),
        "cool_star_count": int(np.sum(cool_stars)),
        "cool_gate_layer_count": int(np.sum(gate)),
        "cool_stars_with_Tlt4000_layers": int(
            np.sum(cool_stars & np.any(temperature < TEMPERATURE_FLOOR_K, axis=1))
        ),
        "cool_start_index": {
            "median": float(np.median(start_index[cool_stars])) if np.any(cool_stars) else None,
            "p95": float(np.percentile(start_index[cool_stars], 95.0))
            if np.any(cool_stars)
            else None,
        },
        "nonmonotonic_cool_star_count": nonmonotonic,
        "nonfinite_count": int(
            np.sum(~np.isfinite(prediction)) + np.sum(~np.isfinite(surface_mass))
        ),
        "cool_gate_mass": {
            "surface_started_v4r5": surface_metrics,
            "hybrid_stored_kappa_below_4000K": hybrid_metrics,
            "oracle_boundary_at_first_Tge4000": oracle_metrics,
            "wholly_in_domain_increment": increment_metrics,
            "crossing_increment": _metrics_or_none(
                increment[cool_stars[:, None] & crossing_increments]
            ),
            "stored_kappa_integral_sanity": truth_kappa_metrics,
            "restart_tau_over_kappa_versus_stored": restart_versus_stored,
            "restart_tau_over_kappa_versus_truth_restart": restart_versus_truth_restart,
        },
        "cool_opacity_versus_stored_total": _slice_report(
            opacity_residual, labels[:, 0], temperature, cool_only=True
        ),
        "cool_mass_versus_stored_by_layer_T": _slice_report(
            log_mass_residual(surface_mass, stored_mass),
            labels[:, 0],
            temperature,
            cool_only=True,
        ),
        "component_log_sensitivity_cool_3200_4000K": cool_3200_4000_components,
        "component_log_sensitivity_cool_4000_5000K": cool_4000_5000_components,
        "component_log_sensitivity_cool_Tge4000K": _component_slice(
            component_sensitivity,
            labels[:, 0],
            temperature,
            lower=TEMPERATURE_FLOOR_K,
            upper=np.inf,
            cool_only=True,
        ),
        "inverse_window_fraction_cool_3200_4000K": _component_slice(
            inverse_window_fraction,
            labels[:, 0],
            temperature,
            lower=3200.0,
            upper=4000.0,
            cool_only=True,
        ),
        "inverse_window_fraction_cool_4000_5000K": _component_slice(
            inverse_window_fraction,
            labels[:, 0],
            temperature,
            lower=4000.0,
            upper=5000.0,
            cool_only=True,
        ),
        "twenty_star_cool_3200_4000K_continuum": hot_grid,
        "middle_band_control_not_in_verdict": {
            "surface_started_p95_dex": _p95(middle_surface),
            "hybrid_p95_dex": _p95(middle_hybrid),
            "in_domain_increment_p95_dex": _p95(middle_increment),
            "shares_Tlt4000_surface_mechanism": middle_shares_surface,
            "note": (
                "Reported only to test whether the 6000-10000 K mass miss "
                "is the same outer-layer integral. Not a middle-gate diagnosis."
            ),
        },
        "scope_boundary": {
            "production_solver_changed": False,
            "textbook_opacity_py_edited": False,
            "mass_integral_start_changed": False,
            "gates_changed": False,
            "new_opacity_version_implemented": False,
            "ode_run": False,
            "funnel_run": False,
            "sealed_holdout_opened": False,
            "production_opacity_called": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_jsonify(result), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision["verdict"],
                "surface_p95_dex": decision["surface_p95_dex"],
                "hybrid_p95_dex": decision["hybrid_p95_dex"],
                "oracle_p95_dex": decision["oracle_p95_dex"],
                "in_domain_increment_p95_dex": decision[
                    "in_domain_increment_p95_dex"
                ],
                "explained_fraction_of_p95_excess": decision[
                    "explained_fraction_of_p95_excess"
                ],
                "v4r6_licensed": license_decision["licensed"],
                "v4r6_reason": license_decision["reason"],
            }
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
