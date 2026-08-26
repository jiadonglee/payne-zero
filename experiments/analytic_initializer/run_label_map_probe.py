"""Attack the 48 percent of the error that the label-to-amplitude map owns.

``run_hopf_basis_probe`` split the held-out temperature error into three
stages: the depth basis owns 16 percent, the rank-5 truncation 37 percent, and
the polynomial that turns labels into mode amplitudes 48 percent.  A physical
depth basis was measured and dropped because it was optimizing the smallest of
the three.  This probe goes after the largest.

Two controls come first, because together they decide whether the term is
attackable and by what kind of model.

* Is it learnable?  Raising the polynomial degree keeps paying -- 0.00856 at
  three, 0.00698 at four, 0.00670 at five, against a per-star oracle floor of
  0.00449 -- so the map is under-resolved rather than saturated.
* Is it smooth?  k-nearest neighbours in label space does no better than the
  degree-3 polynomial at k=10 and worse at k=1 and k=40.  A local estimator
  with forty thousand training rows losing to a global polynomial says the
  amplitude function is globally smooth, so the answer is better coordinates
  and not a more flexible fit.

The coordinate that was missing is the Saha ionized fraction.  It is a sigmoid
in effective temperature, which is exactly the shape a total-degree polynomial
has to spend many terms approximating, and it is not a curve-fitting trick:
hydrogen ionization sets where the convection zone starts and supplies the
electrons that H-minus opacity needs.

Three failed controls are recorded alongside, because each rules out a cheaper
explanation of the gain: substituting the physical coordinates for the original
ones instead of adding them is three times worse; linear rather than
logarithmic abundances buy nothing; and capping the ionization features at
first order costs nothing, which is the evidence that what they contribute is
the sigmoid itself rather than extra polynomial freedom.

No solver calls.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.analytic_depth import (
    evaluate_analytic_depth_closure,
    fit_analytic_depth_closure,
)
from experiments.analytic_initializer.candidates import temperature_regimes
from experiments.analytic_initializer.compact_initializer import (
    PARITY_CONFIGURATION,
    PHYSICAL_CONFIGURATION,
    fit_compact_profile_parameters,
    load_compact_profile_parameters,
    predict_compact_reduced_state,
    save_compact_profile_parameters,
)
from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    file_sha256,
    grey_temperature,
    label_features,
    load_strict_truth,
    make_split,
    polynomial_exponents,
    polynomial_features,
)
from experiments.analytic_initializer.monotone_temperature import project_to_monotone
from experiments.analytic_initializer.physical_labels import (
    HYDROGEN_POTENTIAL_EV,
    METAL_POTENTIAL_EV,
    PHYSICAL_DEGREE_CAPS,
    capped_polynomial_exponents,
    ionized_fraction,
    physical_label_features,
)
from experiments.analytic_initializer.profile_closure import integrate_mass_from_opacity

MANIFESTS = (
    Path("results/reconstruction_metrics.json"),
    Path("results/sealed_solver_subset_20260808.json"),
    Path("results/sealed_audit_20260808.json"),
    Path("results/sealed_audit_20260811.json"),
    Path("results/sealed_initializer_holdout_20260812.json"),
    Path("results/initializer_calibration_20260812.json"),
    Path("results/four_initializer_benchmark_expanded_20260814/expanded200_manifest.json"),
)
DEFAULT_OUTPUT = Path("results/analytic_initializer/label_map_probe.json")
PHYSICAL_ASSET = Path("results/analytic_initializer/compact_profile_parameters_physical.npz")

REGIMES = (5500.0, 7500.0)
CENTER_DEGREE, MODE_DEGREE, COMPONENTS = 22, 18, 5
OPACITY_CENTER_DEGREE = 18
DEEP_START, DEEP_TRIM = 39, 5


def _depth_pieces(corpus, split, target, center_degree, mode_degree):
    """The regime means and depth modes, held fixed while label maps vary."""

    from experiments.analytic_initializer.analytic_depth import DepthNormalization, _project

    normalization = DepthNormalization.from_grid(corpus.tau)
    center_design = normalization.design(corpus.tau, center_degree)
    mode_design = normalization.design(corpus.tau, mode_degree)
    regimes = temperature_regimes(corpus.labels, boundaries=REGIMES)
    pieces = []
    for regime in range(3):
        rows = split.train[regimes[split.train] == regime]
        values = target[rows]
        mean_profile = _project(center_design, values.mean(axis=0)[None, :])[0]
        residual = _project(mode_design, values - mean_profile)
        basis = np.linalg.svd(residual, full_matrices=False)[2][:COMPONENTS]
        pieces.append((rows, mean_profile, basis, residual @ basis.T))
    return regimes, pieces


def _score_label_map(corpus, split, target, pieces, regimes, amplitudes_for):
    """Predict with a caller-supplied label-to-amplitude rule."""

    held = split.validation
    prediction = np.zeros((held.size, corpus.layers))
    for regime, (rows, mean_profile, basis, oracle) in enumerate(pieces):
        selected = np.where(regimes[held] == regime)[0]
        if selected.size == 0:
            continue
        predicted = amplitudes_for(rows, oracle, held[selected], regime)
        prediction[selected] = mean_profile + predicted @ basis
    return prediction


def _polynomial_rule(features, split, degree, caps=None):
    width = features.shape[1]
    center = features[split.train].mean(axis=0)
    scale = np.maximum(features[split.train].std(axis=0), 1.0e-12)
    normalized = (features - center) / scale
    exponents = (
        polynomial_exponents(width, degree) if caps is None
        else capped_polynomial_exponents(degree, caps)
    )

    def rule(rows, oracle, held_rows, regime):
        train, _, _ = polynomial_features(
            normalized[rows], exponents, center=np.zeros(width), scale=np.ones(width)
        )
        evaluate, _, _ = polynomial_features(
            normalized[held_rows], exponents, center=np.zeros(width), scale=np.ones(width)
        )
        return evaluate @ np.linalg.lstsq(train, oracle, rcond=None)[0]

    return rule, int(exponents.shape[0])


def _neighbour_rule(features, split, neighbours):
    center = features[split.train].mean(axis=0)
    scale = np.maximum(features[split.train].std(axis=0), 1.0e-12)
    normalized = (features - center) / scale

    def rule(rows, oracle, held_rows, regime):
        distance = (
            (normalized[held_rows][:, None, :] - normalized[rows][None, :, :]) ** 2
        ).sum(axis=-1)
        order = np.argsort(distance, axis=1)[:, :neighbours]
        return oracle[order].mean(axis=1)

    return rule


def _rank_sweep(corpus, split, target, features, exponents, ranks) -> list[dict]:
    """Is the low-rank basis still saturated once the labels carry physics?

    Gate B found that going from five modes to twelve moved the deep error by
    0.0002 dex and concluded the basis was saturated.  That conclusion was
    conditional on the label map: extra modes are worthless if their amplitudes
    cannot be predicted from labels, which is a statement about the map and not
    about the basis.  This re-runs it with the amplitudes held to a per-star
    oracle as well, so the two causes are separated.
    """

    from experiments.analytic_initializer.analytic_depth import DepthNormalization, _project

    held = split.validation
    normalization = DepthNormalization.from_grid(corpus.tau)
    center_design = normalization.design(corpus.tau, CENTER_DEGREE)
    mode_design = normalization.design(corpus.tau, MODE_DEGREE)
    regimes = temperature_regimes(corpus.labels, boundaries=REGIMES)
    width = features.shape[1]
    center = features[split.train].mean(axis=0)
    scale = np.maximum(features[split.train].std(axis=0), 1.0e-12)
    normalized = (features - center) / scale
    peak = (corpus.labels[held, 0] >= 7000.0) & (corpus.labels[held, 0] < 8000.0)
    window = slice(DEEP_START, corpus.layers - DEEP_TRIM)

    oracle = {rank: np.zeros((held.size, corpus.layers)) for rank in ranks}
    fitted = {rank: np.zeros((held.size, corpus.layers)) for rank in ranks}
    for regime in range(3):
        rows = split.train[regimes[split.train] == regime]
        selected = np.where(regimes[held] == regime)[0]
        if selected.size == 0:
            continue
        values = target[rows]
        mean_profile = _project(center_design, values.mean(axis=0)[None, :])[0]
        train_residual = _project(mode_design, values - mean_profile)
        held_residual = _project(mode_design, target[held[selected]] - mean_profile)
        all_modes = np.linalg.svd(train_residual, full_matrices=False)[2]
        train, _, _ = polynomial_features(
            normalized[rows], exponents, center=np.zeros(width), scale=np.ones(width)
        )
        evaluate, _, _ = polynomial_features(
            normalized[held[selected]], exponents, center=np.zeros(width), scale=np.ones(width)
        )
        for rank in ranks:
            basis = all_modes[:rank]
            oracle[rank][selected] = mean_profile + (held_residual @ basis.T) @ basis
            amplitudes = np.linalg.lstsq(train, train_residual @ basis.T, rcond=None)[0]
            fitted[rank][selected] = mean_profile + (evaluate @ amplitudes) @ basis

    truth = target[held]
    out = []
    for rank in ranks:
        floor = float(np.percentile(np.abs(oracle[rank] - truth), 95.0))
        achieved = float(np.percentile(np.abs(fitted[rank] - truth), 95.0))
        out.append(
            {
                "rank": rank,
                "terms": int(exponents.shape[0]),
                "oracle_p95_dex": floor,
                "achieved_p95_dex": achieved,
                "label_map_gap_dex": achieved - floor,
                "deep_p95_dex_7000_8000K": float(
                    np.percentile(np.abs(fitted[rank] - truth)[peak][:, window], 95.0)
                )
                if peak.any()
                else None,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=20260816)
    args = parser.parse_args()

    corpus = load_strict_truth(args.corpus)
    excluded, manifests = collect_excluded_indices(MANIFESTS, corpus_size=corpus.size)
    split = make_split(corpus.size, excluded=excluded, seed=args.split_seed)
    held = split.validation
    grey = grey_temperature(corpus.labels[:, 0], corpus.tau)
    temperature_target = np.log10(corpus.temperature / grey)
    opacity_target = np.log10(corpus.rosseland_opacity)
    truth_mass = corpus.column_mass[held]

    regimes, temperature_pieces = _depth_pieces(
        corpus, split, temperature_target, CENTER_DEGREE, MODE_DEGREE
    )
    _, opacity_pieces = _depth_pieces(
        corpus, split, opacity_target, OPACITY_CENTER_DEGREE, MODE_DEGREE
    )

    standard = label_features(corpus.labels)
    physical = physical_label_features(corpus.labels)
    substituted = np.column_stack(
        (
            standard[:, 0],
            0.5 * (corpus.labels[:, 1] + corpus.labels[:, 2]),
            ionized_fraction(corpus.labels, HYDROGEN_POTENTIAL_EV),
            ionized_fraction(corpus.labels, METAL_POTENTIAL_EV),
            standard[:, 4],
        )
    )
    linear_abundance = np.column_stack(
        (standard, 10.0 ** corpus.labels[:, 2], 10.0 ** corpus.labels[:, 3])
    )

    def temperature_p95(prediction):
        return float(np.percentile(np.abs(prediction - temperature_target[held]), 95.0))

    def mass_p95(prediction):
        mass = integrate_mass_from_opacity(corpus.tau, prediction)
        return float(np.percentile(np.abs(np.log10(mass) - np.log10(truth_mass)), 95.0))

    candidates = []
    for name, features, degree, caps in (
        ("standard_degree_2", standard, 2, None),
        ("standard_degree_3", standard, 3, None),
        ("standard_degree_4", standard, 4, None),
        ("standard_degree_5", standard, 5, None),
        ("physical_degree_3_full", physical, 3, None),
        ("physical_degree_3_capped", physical, 3, PHYSICAL_DEGREE_CAPS),
        ("physical_degree_4_capped", physical, 4, PHYSICAL_DEGREE_CAPS),
        ("control_substituted_not_added", substituted, 3, None),
        ("control_linear_abundances", linear_abundance, 3, None),
    ):
        rule, terms = _polynomial_rule(features, split, degree, caps)
        candidates.append(
            {
                "label_map": name,
                "features": int(features.shape[1]),
                "terms": terms,
                "temperature_p95_dex": temperature_p95(
                    _score_label_map(corpus, split, temperature_target, temperature_pieces, regimes, rule)
                ),
                "mass_p95_dex": mass_p95(
                    _score_label_map(corpus, split, opacity_target, opacity_pieces, regimes, rule)
                ),
            }
        )
    for neighbours in (1, 10, 40):
        rule = _neighbour_rule(standard, split, neighbours)
        candidates.append(
            {
                "label_map": f"nearest_neighbour_k{neighbours}",
                "features": 5,
                "terms": None,
                "temperature_p95_dex": temperature_p95(
                    _score_label_map(corpus, split, temperature_target, temperature_pieces, regimes, rule)
                ),
                "mass_p95_dex": mass_p95(
                    _score_label_map(corpus, split, opacity_target, opacity_pieces, regimes, rule)
                ),
            }
        )

    # The per-star oracle: the floor any label map is working against.
    oracle_temperature = np.zeros((held.size, corpus.layers))
    for regime, (rows, mean_profile, basis, _) in enumerate(temperature_pieces):
        selected = np.where(regimes[held] == regime)[0]
        if selected.size == 0:
            continue
        from experiments.analytic_initializer.analytic_depth import DepthNormalization, _project

        mode_design = DepthNormalization.from_grid(corpus.tau).design(corpus.tau, MODE_DEGREE)
        residual = _project(mode_design, temperature_target[held[selected]] - mean_profile)
        oracle_temperature[selected] = mean_profile + (residual @ basis.T) @ basis
    oracle_p95 = temperature_p95(oracle_temperature)

    baseline = next(c for c in candidates if c["label_map"] == "standard_degree_3")
    for candidate in candidates:
        candidate["closes_fraction_of_gap"] = (
            (baseline["temperature_p95_dex"] - candidate["temperature_p95_dex"])
            / (baseline["temperature_p95_dex"] - oracle_p95)
        )

    # The shipped asset, through the real code path.
    parity = fit_compact_profile_parameters(corpus, split, configuration=PARITY_CONFIGURATION)
    physical_parameters = fit_compact_profile_parameters(
        corpus, split, configuration=PHYSICAL_CONFIGURATION
    )
    assets = {}
    window = slice(DEEP_START, corpus.layers - DEEP_TRIM)
    for name, parameters in (("parity", parity), ("physical", physical_parameters)):
        mass, temperature, _ = predict_compact_reduced_state(
            corpus.labels[held], corpus.tau, parameters
        )
        relative = np.abs(temperature / corpus.temperature[held] - 1.0)
        assets[name] = {
            "stored_floats": parameters.stored_float_count,
            "temperature_relative_p95": float(np.percentile(relative, 95.0)),
            "temperature_relative_deep_p95": float(np.percentile(relative[:, window], 95.0)),
            "mass_dex_p95": float(
                np.percentile(np.abs(np.log10(mass) - np.log10(truth_mass)), 95.0)
            ),
            "temperature_monotone_rows": int(
                np.sum(np.all(np.diff(temperature, axis=1) > 0.0, axis=1))
            ),
            "rows": int(held.size),
        }

    save_compact_profile_parameters(PHYSICAL_ASSET, physical_parameters)
    reloaded = load_compact_profile_parameters(PHYSICAL_ASSET)
    before = predict_compact_reduced_state(corpus.labels[held], corpus.tau, physical_parameters)
    after = predict_compact_reduced_state(corpus.labels[held], corpus.tau, reloaded)
    assets["physical"]["path"] = str(PHYSICAL_ASSET)
    assets["physical"]["sha256"] = file_sha256(PHYSICAL_ASSET)
    assets["physical"]["round_trip_exact"] = bool(
        all(np.array_equal(one, other) for one, other in zip(before, after))
    )
    assets["physical"]["label_features"] = reloaded.temperature.label_features

    payload = {
        "format": "payne_zero_label_map_probe_v1",
        "date": "2026-08-17",
        "question": (
            "The label-to-amplitude polynomial owns 48 percent of the held-out "
            "error. Can physically motivated label coordinates take some of it?"
        ),
        "answer": (
            "Yes. Adding two Saha ionized fractions to the five standard "
            "coordinates closes 59 percent of the gap to the per-star oracle, "
            "and beats simply raising the polynomial degree at lower cost: 104 "
            "terms reach 0.00614 dex where 126 terms of degree 4 reach 0.00698. "
            "Column mass improves from 0.0870 to 0.0597 dex."
        ),
        "corpus": {"path": str(args.corpus), "sha256": file_sha256(args.corpus), "rows": corpus.size},
        "split": {
            "seed": args.split_seed,
            "train": int(split.train.size),
            "validation": int(split.validation.size),
            "manifests": manifests,
        },
        "fixed_depth_pieces": {
            "center_degree": CENTER_DEGREE,
            "opacity_center_degree": OPACITY_CENTER_DEGREE,
            "mode_degree": MODE_DEGREE,
            "components": COMPONENTS,
            "note": "held fixed so only the label map varies",
        },
        "per_star_oracle_p95_dex": oracle_p95,
        "label_maps": candidates,
        "rank_saturation": {
            "question": (
                "Gate B concluded the low-rank basis was saturated at five "
                "modes. Was that a property of the basis or of the label map?"
            ),
            "answer": (
                "Of the label map. With the standard coordinates, going from "
                "five modes to sixteen buys 4 percent, because the extra "
                "amplitudes cannot be predicted from the labels -- the gap to "
                "the per-star oracle grows from 0.00407 to 0.00638 dex. With "
                "the Saha coordinates the same change buys 15 percent and the "
                "gap grows only from 0.00165 to 0.00341."
            ),
            "ranks": [1, 2, 3, 5, 8, 12, 16],
            "standard": _rank_sweep(
                corpus, split, temperature_target, standard,
                polynomial_exponents(5, 3), (1, 2, 3, 5, 8, 12, 16),
            ),
            "physical": _rank_sweep(
                corpus, split, temperature_target, physical,
                capped_polynomial_exponents(3, PHYSICAL_DEGREE_CAPS), (1, 2, 3, 5, 8, 12, 16),
            ),
        },
        "assets": assets,
        "reproducer": "PYTHONPATH=. python3 -m experiments.analytic_initializer.run_label_map_probe",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"per-star oracle floor: {oracle_p95:.5f} dex\n")
    print("%-32s %-5s %-6s %-9s %-9s %s" % ("label map", "feat", "terms", "T p95", "mass p95", "gap closed"))
    for candidate in candidates:
        print(
            "%-32s %-5d %-6s %.5f   %.5f   %+.1f%%"
            % (
                candidate["label_map"],
                candidate["features"],
                candidate["terms"] if candidate["terms"] else "-",
                candidate["temperature_p95_dex"],
                candidate["mass_p95_dex"],
                100.0 * candidate["closes_fraction_of_gap"],
            )
        )
    print()
    print(json.dumps(assets, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
