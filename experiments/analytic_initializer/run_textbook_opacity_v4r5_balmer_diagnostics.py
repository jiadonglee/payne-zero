"""Run the preregistered v4r5 Balmer-window hydrogen bound-free diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    file_sha256,
    load_strict_truth,
)
from experiments.analytic_initializer.run_textbook_opacity_offline import _metrics
from experiments.analytic_initializer.textbook_opacity import (
    DEFAULT_TEXTBOOK_CONSTANTS,
    WINDOW_NAMES,
    _hydrogen_ground_anchored_level_populations,
    rosseland_frequency_nodes,
    saha_electron_diagnostics_v4r3,
    textbook_opacity_node_components_v4r5,
)
from experiments.analytic_initializer.textbook_opacity_v4r5_balmer_verdict import (
    BALMER_LITERATURE_EDGE_CM2,
    HOT_LAYER,
    LYMAN_EDGE_CM2,
    N2_EDGE_SCALE,
    PRIMARY_LAYER,
    V4_N2_EDGE_CM2,
    decide_balmer_diagnostics,
    n2_edge_cross_section_cm2,
)


ABLATION_JSON = Path(
    "results/analytic_initializer/textbook_opacity_v4r4_hot_flag_ablation_20260828.json"
)
HOT_GRID_JSON = Path(
    "results/analytic_initializer/textbook_opacity_v4r5_hot_grid_20260828.json"
)
KARZAS_NPZ = Path("source_data_files/atmosphere_tables/karzas_latter_tables.npz")
EXPECTED_ABLATION_SHA256 = (
    "c136b076d5f135733e4d7e43081d2ed8040f3586b0f4cbd01283628dda613b66"
)
EXPECTED_HOT_GRID_SHA256 = (
    "d663496ab9128aa2b4b0ec58560c7def449e44d66f45987bdf5540c31ef67dad"
)
BALMER_WINDOW_INDEX = WINDOW_NAMES.index("balmer_to_lyman")
REGISTERED_SLICES = (
    PRIMARY_LAYER,
    ("15000_22000K", 15000.0, 22000.0),
    ("22000_30000K", 22000.0, 30000.0),
    HOT_LAYER,
    ("at_least_15000K", 15000.0, float("inf")),
)
COMPONENT_ZERO_VARIANTS = {
    "no_hminus": ("hminus_boundfree", "hminus_freefree"),
    "no_hi_ff": ("hydrogen_freefree",),
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def hydrogen_boundfree_from_level_populations_local(
    node_frequency: np.ndarray,
    stimulated: np.ndarray,
    level_population: np.ndarray,
    rho: np.ndarray,
    *,
    constants=DEFAULT_TEXTBOOK_CONSTANTS,
    included_levels: tuple[int, ...] | None = None,
    n2_edge_cm2: float | None = None,
    n2_cross_section: np.ndarray | None = None,
    apply_stimulated: bool = True,
) -> np.ndarray:
    """Local copy of the v4r5 H I bound-free node loop, with diagnostic knobs.

    This function is intentionally not in ``textbook_opacity.py``.  With
    default knobs it must reproduce the frozen v4r5 hydrogen bound-free cube.
    """

    level_count = int(constants.hydrogen_boundfree_level_count)
    levels = np.arange(1, level_count + 1, dtype=np.float64)
    if included_levels is None:
        keep = set(range(1, level_count + 1))
    else:
        keep = {int(level) for level in included_levels}
    factor = stimulated if apply_stimulated else np.ones_like(stimulated)
    hydrogen_boundfree = np.zeros_like(node_frequency)
    for level_index, principal_quantum_number in enumerate(levels):
        level = int(principal_quantum_number)
        if level not in keep:
            continue
        threshold_frequency = (
            constants.hydrogen_ionization_eV
            / principal_quantum_number**2
            * constants.eV_to_erg
            / constants.planck_erg_s
        )
        if level == 2 and n2_cross_section is not None:
            cross_section = np.asarray(n2_cross_section, dtype=np.float64)
        else:
            edge_cross_section = n2_edge_cross_section_cm2(
                level, n2_edge_cm2=n2_edge_cm2
            )
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
            * factor
        )
    return hydrogen_boundfree


def _rosseland_mean(weights: np.ndarray, total: np.ndarray) -> np.ndarray:
    return 1.0 / np.sum(weights / np.maximum(total, 1.0e-30), axis=(-2, -1))


def _inverse_window_fraction(weights: np.ndarray, total: np.ndarray) -> np.ndarray:
    inverse_by_window = np.sum(weights / np.maximum(total, 1.0e-30), axis=-1)
    return inverse_by_window / np.maximum(
        np.sum(inverse_by_window, axis=-1, keepdims=True),
        1.0e-300,
    )


def _log_sensitivity(
    weights: np.ndarray, total: np.ndarray, component: np.ndarray
) -> np.ndarray:
    inverse_total = np.sum(weights / np.maximum(total, 1.0e-30), axis=(-2, -1))
    return np.sum(
        weights * component / np.maximum(total, 1.0e-30) ** 2,
        axis=(-2, -1),
    ) / np.maximum(inverse_total, 1.0e-300)


def _slice_mask(temperature: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if np.isinf(upper):
        return temperature >= lower
    return (temperature >= lower) & (temperature < upper)


def _metrics_on_slice(
    residual: np.ndarray, temperature: np.ndarray, lower: float, upper: float
) -> dict[str, float]:
    return _metrics(residual[_slice_mask(temperature, lower, upper)])


def _window_medians(
    fraction: np.ndarray, temperature: np.ndarray, lower: float, upper: float
) -> dict[str, float]:
    mask = _slice_mask(temperature, lower, upper)
    return {
        name: float(np.median(fraction[..., index][mask]))
        for index, name in enumerate(WINDOW_NAMES)
    }


def load_karzas_total_tables(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {
            "log10_frequency_hz": np.asarray(
                data["karzas_latter_log10_frequency_hz"], dtype=np.float64
            ),
            "total_log10_cross_section_cm2": np.asarray(
                data["karzas_latter_total_log10_cross_section_cm2"], dtype=np.float64
            ),
        }


def karzas_total_cross_section_cm2(
    frequency_hz: np.ndarray,
    *,
    principal_quantum_number: int,
    tables: dict[str, np.ndarray],
) -> np.ndarray:
    """Interpolate the Karzas total (summed-l) column for one n-shell."""

    shell = int(principal_quantum_number)
    if shell < 1 or shell > tables["log10_frequency_hz"].shape[1]:
        raise ValueError(f"Karzas shell {shell} is out of range")
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    freq_col = tables["log10_frequency_hz"][:, shell - 1]
    value_col = tables["total_log10_cross_section_cm2"][:, shell - 1]
    log10_frequency = np.log10(np.maximum(frequency, 1.0e-300))
    below = (frequency <= 0.0) | (log10_frequency < freq_col[-1])
    count = int(freq_col.size)
    bracket = np.searchsorted(-freq_col, -log10_frequency, side="right")
    index_lo = np.remainder(bracket - 1, count)
    index_hi = np.clip(bracket, 0, count - 1)
    x_lo = freq_col[index_lo]
    x_hi = freq_col[index_hi]
    y_lo = value_col[index_lo]
    y_hi = value_col[index_hi]
    denominator = x_lo - x_hi
    weight = np.divide(
        log10_frequency - x_hi,
        denominator,
        out=np.zeros_like(log10_frequency),
        where=np.abs(denominator) >= 1.0e-15,
    )
    log10_sigma = (y_lo - y_hi) * weight + y_hi
    sigma = np.where(np.abs(denominator) < 1.0e-15, 10.0**y_lo, 10.0**log10_sigma)
    sigma = np.where(bracket >= count, 10.0 ** value_col[-1], sigma)
    return np.where(below, 0.0, sigma)


def _karzas_edge_cm2(tables: dict[str, np.ndarray], shell: int) -> dict[str, float]:
    freq_col = tables["log10_frequency_hz"][:, int(shell) - 1]
    value_col = tables["total_log10_cross_section_cm2"][:, int(shell) - 1]
    edge_hz = float(10.0 ** freq_col[-1])
    edge_sigma = float(10.0 ** value_col[-1])
    return {
        "threshold_hz": edge_hz,
        "threshold_cross_section_cm2": edge_sigma,
        "log10_threshold_hz": float(freq_col[-1]),
    }


def _karzas_n2_window_ratio(
    tables: dict[str, np.ndarray],
    *,
    constants=DEFAULT_TEXTBOOK_CONSTANTS,
    temperature_K: float = 10000.0,
) -> dict[str, object]:
    node_frequency, _, _ = rosseland_frequency_nodes(
        np.asarray([[temperature_K]], dtype=np.float64),
        constants=constants,
    )
    window = node_frequency[0, 0, BALMER_WINDOW_INDEX]
    threshold = (
        constants.hydrogen_ionization_eV
        / 4.0
        * constants.eV_to_erg
        / constants.planck_erg_s
    )
    textbook = np.where(
        window >= threshold,
        V4_N2_EDGE_CM2 * (threshold / np.maximum(window, 1.0e-300)) ** 3,
        0.0,
    )
    karzas = karzas_total_cross_section_cm2(
        window, principal_quantum_number=2, tables=tables
    )
    valid = (textbook > 0.0) & (karzas > 0.0)
    ratio = np.log10(textbook[valid] / karzas[valid])
    return {
        "temperature_K": temperature_K,
        "window": "balmer_to_lyman",
        "node_count": int(window.size),
        "valid_count": int(valid.sum()),
        "log10_textbook_over_karzas": _metrics(ratio) if valid.any() else None,
        "span_p95_minus_p05_dex": (
            None
            if not valid.any()
            else float(np.percentile(ratio, 95.0) - np.percentile(ratio, 5.0))
        ),
        "near_balmer_edge_dex": (
            None if not valid.any() else float(ratio[0] if valid[0] else np.median(ratio))
        ),
        "near_lyman_end_dex": (
            None
            if not valid.any()
            else float(ratio[-1] if valid[-1] else np.median(ratio))
        ),
        "textbook_n2_edge_cm2": V4_N2_EDGE_CM2,
        "karzas_n2_edge_cm2": _karzas_edge_cm2(tables, 2)[
            "threshold_cross_section_cm2"
        ],
        "karzas_n1_edge_cm2": _karzas_edge_cm2(tables, 1)[
            "threshold_cross_section_cm2"
        ],
    }


def _swap_total(
    components: dict[str, np.ndarray],
    *,
    new_boundfree: np.ndarray | None = None,
    zero_names: tuple[str, ...] = (),
) -> np.ndarray:
    total = np.asarray(components["total"], dtype=np.float64).copy()
    if new_boundfree is not None:
        total = total - components["hydrogen_boundfree"] + new_boundfree
    for name in zero_names:
        total = total - components[name]
    return np.maximum(total, 1.0e-30)


def _load_grid(ablation: dict[str, object], corpus) -> dict[str, np.ndarray]:
    references = ablation["references"]
    indices = np.asarray(
        [int(row["corpus_index"]) for row in references], dtype=np.int64
    )
    labels = np.asarray(corpus.labels[indices], dtype=np.float64)
    temperature = np.asarray(corpus.temperature[indices], dtype=np.float64)
    pressure = np.asarray(corpus.gas_pressure[indices], dtype=np.float64)
    stored_temperature = np.asarray(
        [row["temperature_K"] for row in references], dtype=np.float64
    )
    if stored_temperature.shape != temperature.shape:
        raise ValueError("stored temperatures do not match the corpus grid")
    if np.max(np.abs(temperature - stored_temperature)) > 1.0e-6:
        raise ValueError("corpus temperatures drifted from the ablation JSON")
    production = np.asarray(
        [row["production_continuum_baseline"] for row in references],
        dtype=np.float64,
    )
    v4r3 = np.asarray(
        [row["v4r3_rosseland_opacity"] for row in references], dtype=np.float64
    )
    return {
        "indices": indices,
        "labels": labels,
        "temperature": temperature,
        "pressure": pressure,
        "production": production,
        "v4r3": v4r3,
    }


def run_balmer_diagnostics(
    *,
    corpus_path: Path,
    ablation_path: Path,
    output_path: Path,
    karzas_path: Path,
    hot_grid_path: Path | None = None,
) -> dict[str, object]:
    ablation_sha = file_sha256(ablation_path)
    if ablation_sha != EXPECTED_ABLATION_SHA256:
        raise RuntimeError(
            "ablation JSON SHA-256 mismatch: "
            f"{ablation_sha} != {EXPECTED_ABLATION_SHA256}"
        )
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    corpus = load_strict_truth(corpus_path)
    grid = _load_grid(ablation, corpus)
    labels = grid["labels"]
    temperature = grid["temperature"]
    pressure = grid["pressure"]
    production = grid["production"]
    constants = DEFAULT_TEXTBOOK_CONSTANTS

    components = textbook_opacity_node_components_v4r5(
        labels, temperature, pressure, constants=constants
    )
    state = saha_electron_diagnostics_v4r3(
        labels, temperature, pressure, constants=constants
    )
    populations = _hydrogen_ground_anchored_level_populations(
        temperature,
        state["hydrogen_neutral_density_cm3"],
        constants=constants,
    )
    node_frequency = components["frequency_nodes_hz"]
    weights = components["node_weights"]
    stimulated = -np.expm1(-components["frequency_nodes_u"])
    rho = state["rho_g_cm3"]
    local_v4r5_bf = hydrogen_boundfree_from_level_populations_local(
        node_frequency, stimulated, populations, rho, constants=constants
    )
    frozen_bf = np.asarray(components["hydrogen_boundfree"], dtype=np.float64)
    bf_rel = np.max(
        np.abs(local_v4r5_bf - frozen_bf)
        / np.maximum(np.abs(frozen_bf), 1.0e-30)
    )
    if bf_rel > 1.0e-10:
        raise RuntimeError(
            "local H I bound-free copy drifted from frozen v4r5: "
            f"max relative error {bf_rel:.3e}"
        )

    karzas_tables = load_karzas_total_tables(karzas_path)
    karzas_ratio = None
    n2_karzas = None
    n2_karzas_shape = None
    if karzas_tables is not None:
        karzas_ratio = _karzas_n2_window_ratio(karzas_tables, constants=constants)
        n2_karzas = karzas_total_cross_section_cm2(
            node_frequency, principal_quantum_number=2, tables=karzas_tables
        )
        n2_edge = _karzas_edge_cm2(karzas_tables, 2)["threshold_cross_section_cm2"]
        n2_karzas_shape = np.where(
            n2_karzas > 0.0,
            V4_N2_EDGE_CM2 * n2_karzas / n2_edge,
            0.0,
        )

    boundfree_variants: dict[str, np.ndarray] = {
        "v4r5": frozen_bf,
        "n2_only": hydrogen_boundfree_from_level_populations_local(
            node_frequency,
            stimulated,
            populations,
            rho,
            constants=constants,
            included_levels=(2,),
        ),
        "n2_balmer_edge": hydrogen_boundfree_from_level_populations_local(
            node_frequency,
            stimulated,
            populations,
            rho,
            constants=constants,
            n2_edge_cm2=BALMER_LITERATURE_EDGE_CM2,
        ),
        "drop_n_ge_3": hydrogen_boundfree_from_level_populations_local(
            node_frequency,
            stimulated,
            populations,
            rho,
            constants=constants,
            included_levels=(1, 2),
        ),
        "drop_n_ge_7": hydrogen_boundfree_from_level_populations_local(
            node_frequency,
            stimulated,
            populations,
            rho,
            constants=constants,
            included_levels=tuple(range(1, 7)),
        ),
        "no_stimulated": hydrogen_boundfree_from_level_populations_local(
            node_frequency,
            stimulated,
            populations,
            rho,
            constants=constants,
            apply_stimulated=False,
        ),
    }
    if n2_karzas_shape is not None:
        boundfree_variants["n2_karzas_shape"] = (
            hydrogen_boundfree_from_level_populations_local(
                node_frequency,
                stimulated,
                populations,
                rho,
                constants=constants,
                n2_cross_section=n2_karzas_shape,
            )
        )
        boundfree_variants["n2_karzas_full"] = (
            hydrogen_boundfree_from_level_populations_local(
                node_frequency,
                stimulated,
                populations,
                rho,
                constants=constants,
                n2_cross_section=n2_karzas,
            )
        )

    totals: dict[str, np.ndarray] = {
        name: _swap_total(components, new_boundfree=boundfree)
        for name, boundfree in boundfree_variants.items()
    }
    for name, zero_names in COMPONENT_ZERO_VARIANTS.items():
        totals[name] = _swap_total(components, zero_names=zero_names)

    opacities = {
        name: _rosseland_mean(weights, total) for name, total in totals.items()
    }
    residuals = {
        name: np.log10(opacity) - np.log10(production)
        for name, opacity in opacities.items()
    }
    residuals["v4r3"] = np.log10(grid["v4r3"]) - np.log10(production)

    control_metrics = {
        name: _metrics_on_slice(residual, temperature, PRIMARY_LAYER[1], PRIMARY_LAYER[2])
        for name, residual in residuals.items()
    }
    hot_metrics = {
        name: _metrics_on_slice(residual, temperature, HOT_LAYER[1], HOT_LAYER[2])
        for name, residual in residuals.items()
    }
    slice_metrics = {}
    for slice_name, lower, upper in REGISTERED_SLICES:
        slice_metrics[slice_name] = {
            name: _metrics_on_slice(residual, temperature, lower, upper)
            for name, residual in residuals.items()
        }

    window_fractions = {
        name: _inverse_window_fraction(weights, total)
        for name, total in totals.items()
        if name in ("v4r5", "n2_only", "n2_balmer_edge", "drop_n_ge_3")
    }
    window_control = {
        name: _window_medians(
            fraction, temperature, PRIMARY_LAYER[1], PRIMARY_LAYER[2]
        )
        for name, fraction in window_fractions.items()
    }
    window_hot = {
        name: _window_medians(fraction, temperature, HOT_LAYER[1], HOT_LAYER[2])
        for name, fraction in window_fractions.items()
    }

    v4r5_total = totals["v4r5"]
    level_sensitivity = []
    control_mask = _slice_mask(temperature, PRIMARY_LAYER[1], PRIMARY_LAYER[2])
    for level in range(1, int(constants.hydrogen_boundfree_level_count) + 1):
        level_bf = hydrogen_boundfree_from_level_populations_local(
            node_frequency,
            stimulated,
            populations,
            rho,
            constants=constants,
            included_levels=(level,),
        )
        sensitivity = _log_sensitivity(weights, v4r5_total, level_bf)
        level_sensitivity.append(
            {
                "principal_quantum_number": level,
                "edge_cross_section_cm2": n2_edge_cross_section_cm2(level),
                "control_median": float(np.median(sensitivity[control_mask])),
            }
        )
    leak_sensitivity = {
        name: float(
            np.median(
                _log_sensitivity(weights, v4r5_total, components[name])[control_mask]
            )
        )
        for name in (
            "hydrogen_boundfree",
            "hydrogen_freefree",
            "hminus_boundfree",
            "hminus_freefree",
            "electron_scattering",
        )
    }

    span = None
    if karzas_ratio is not None:
        span = karzas_ratio["span_p95_minus_p05_dex"]
    decision = decide_balmer_diagnostics(
        control=control_metrics,
        hot=hot_metrics,
        karzas_n2_ratio_span_dex=span,
    )

    hot_grid_record = None
    if hot_grid_path is not None and hot_grid_path.is_file():
        hot_grid_sha = file_sha256(hot_grid_path)
        hot_grid_record = {
            "path": str(hot_grid_path),
            "sha256": hot_grid_sha,
            "sha256_matches_preregistration": hot_grid_sha == EXPECTED_HOT_GRID_SHA256,
        }

    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": "textbook_opacity_v4r5_balmer_diagnostics",
        "version": "v4r5_balmer_diagnostics",
        "status": "diagnostic_only",
        "corpus": str(corpus.path),
        "corpus_sha256": file_sha256(corpus.path),
        "prior_artifacts": {
            "hot_flag_ablation": {
                "path": str(ablation_path),
                "sha256": ablation_sha,
            },
            "v4r5_hot_grid": hot_grid_record,
        },
        "constants": {
            "lyman_edge_cm2": LYMAN_EDGE_CM2,
            "v4_n2_edge_cm2": V4_N2_EDGE_CM2,
            "balmer_literature_edge_cm2": BALMER_LITERATURE_EDGE_CM2,
            "n2_edge_scale": N2_EDGE_SCALE,
            "hydrogen_ionization_eV": constants.hydrogen_ionization_eV,
            "local_boundfree_max_relative_error_vs_v4r5": float(bf_rel),
        },
        "reference_grid": {
            "reference_count": int(grid["indices"].size),
            "reference_indices": [int(index) for index in grid["indices"]],
        },
        "karzas": karzas_ratio,
        "karzas_available": karzas_tables is not None,
        "karzas_path": str(karzas_path) if karzas_tables is not None else None,
        "primary_slice": {
            "name": PRIMARY_LAYER[0],
            "temperature_lower_K": PRIMARY_LAYER[1],
            "temperature_upper_K": PRIMARY_LAYER[2],
            "metrics": control_metrics,
            "inverse_window_fraction_median": window_control,
        },
        "hot_slice": {
            "name": HOT_LAYER[0],
            "temperature_lower_K": HOT_LAYER[1],
            "temperature_upper_K": None,
            "metrics": hot_metrics,
            "inverse_window_fraction_median": window_hot,
        },
        "layer_temperature_slices": slice_metrics,
        "level_log_sensitivity_control": level_sensitivity,
        "component_log_sensitivity_control": leak_sensitivity,
        "decision": decision,
        "production_replay_notes": {
            "iterations": 1,
            "hydrogen_departure_seed": 1.0,
            "textbook_ingests_production_departures": False,
            "lines_enabled": False,
            "molecules_enabled": False,
        },
        "scope_boundary": {
            "production_solver_changed": False,
            "default_initializer_changed": False,
            "textbook_opacity_module_changed": False,
            "new_opacity_version_in_textbook_module": False,
            "gates_changed": False,
            "ode_run": False,
            "funnel_run": False,
            "sealed_holdout_opened": False,
            "cool_mass_work": False,
            "runtime_karzas_load_in_candidate": False,
            "corpus_fit": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "decision": decision,
                    "primary_v4r5": control_metrics["v4r5"],
                    "primary_n2_balmer_edge": control_metrics.get("n2_balmer_edge"),
                    "hot_v4r5": hot_metrics["v4r5"],
                    "hot_n2_balmer_edge": hot_metrics.get("n2_balmer_edge"),
                    "karzas_available": karzas_tables is not None,
                }
            ),
            allow_nan=False,
        ),
        flush=True,
    )
    print(f"wrote {output_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ablation", type=Path, default=ABLATION_JSON)
    parser.add_argument("--hot-grid", type=Path, default=HOT_GRID_JSON)
    parser.add_argument("--karzas", type=Path, default=KARZAS_NPZ)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/analytic_initializer/"
            "textbook_opacity_v4r5_balmer_diagnostics_20260828.json"
        ),
    )
    args = parser.parse_args(argv)
    if not args.ablation.is_file():
        raise SystemExit(f"required ablation JSON is missing: {args.ablation}")
    if not args.corpus.is_file():
        raise SystemExit(f"required corpus is missing: {args.corpus}")
    run_balmer_diagnostics(
        corpus_path=args.corpus,
        ablation_path=args.ablation,
        output_path=args.out,
        karzas_path=args.karzas,
        hot_grid_path=args.hot_grid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
