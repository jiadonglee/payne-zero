#!/usr/bin/env python3
"""Extract every number quoted in the manuscript from the recorded result artifacts.

The repository notes (``notes/reduced_state_progress.md`` and the
``solver-in-the-loop-*.md`` files) are a live lab notebook: they interleave
numbers from at least eight rejected candidate checkpoints with the ones that
stand, and their own header warns that "the earlier accuracy and spectral
numbers below describe superseded models".  Transcribing numbers from that prose
into a manuscript silently mixes checkpoints.

This script therefore reads the JSON artifacts directly and emits

* ``paper/numbers.tex``   -- ``\\newcommand`` macros for every inline number,
* ``paper/tables/*.tex``  -- complete, ready-to-``\\input`` table environments,
* ``paper/numbers.json``  -- the same values with their source path and SHA256,
                             which also backs the provenance appendix.

Run ``--check`` to verify that regenerating reproduces ``numbers.json`` exactly;
it exits non-zero if any source file is missing, has changed hash, or if any
emitted value has drifted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
TABLES = PAPER / "tables"

# --------------------------------------------------------------------------
# Sources.  Every number in the manuscript must come from one of these.
# --------------------------------------------------------------------------

SOURCES: dict[str, str] = {
    # -- the corpus and the evaluation protocol ---------------------------
    "training": "results/reduced_state_emulator_training.json",
    # -- solver baseline behavior by label slice --------------------------
    "baseline": "results/baseline_metrics.json",
    # -- Sect. 5.1  sufficiency from converged (m,T) -----------------------
    "recon": (
        "results/paper_physical_seed_20260820/parity/reconstruction_metrics.json"
    ),
    "parity": (
        "results/paper_physical_seed_20260820/parity/"
        "convergence_metrics_reduced_state_parity.json"
    ),
    "parity_reduced_records": (
        "runs/paper_physical_seed_20260820/parity/records/"
        "reduced_state_reconstruction/records.jsonl"
    ),
    "parity_truth_records": (
        "runs/paper_physical_seed_20260820/parity/records/"
        "full_truth_oracle/records.jsonl"
    ),
    # -- Sect. 5.2  spectral sufficiency -----------------------------------
    "gate_truth": (
        "results/paper_physical_seed_20260820/parity/spectral_gate_truth_mT.json"
    ),
    # -- Sect. 5.3  representation resolution and the top boundary ---------
    "resolution": (
        "results/paper_physical_seed_20260820/depth_resolution/"
        "convergence_metrics_depth_resolution.json"
    ),
    "continuity": "runs/continuity/summary.json",
    # The rank correlation between top-boundary seed residual and iteration
    # count is the one quantity the harness reports in prose but does not write
    # to its summary, so it is recomputed here from the corpus with the
    # harness's own seed_residual().  The median it reproduces is checked
    # against the summary's value below.
    "corpus": ("source_data_files/atmosphere_emulator/five_label/"
               "strict_truth_52199.npz"),
    # -- Sect. 5.4-5.5  the learned two-field emulator (ablation arm) ------
    "learned": (
        "results/paper_physical_seed_20260820/learned/"
        "convergence_metrics_learned_monotone.json"
    ),
    "learned_records": (
        "runs/paper_physical_seed_20260820/learned/records/"
        "learned_reduced_state/records.jsonl"
    ),
    "production": "results/convergence_metrics_production_baseline.json",
    "production_records": (
        "runs/reduced_state_emulator/production_six_field/records.jsonl"
    ),
    # The derived-field errors must come from the SAME arm as the solver
    # comparison above.  ``derived_field_accuracy.py`` defaults to
    # ``predicted_monotone.npz``, which is that arm; ``field_consistency_dev.json``
    # scores a different, later checkpoint and is deliberately not used here.
    "derived_learned": (
        "results/paper_physical_seed_20260820/learned/"
        "learned_reduced_state_derived_errors.npz"
    ),
    "derived_learned_summary": (
        "results/paper_physical_seed_20260820/learned/"
        "learned_reduced_state_derived_errors.json"
    ),
    "derived_production": "results/production_sixfield_errors.npz",
    # The bottom row of Fig. 3: the one gated star whose learned start reaches
    # the bar.  The caption claims its residual is one-signed rather than noise
    # about zero, so the fraction is measured here from the two spectra the
    # figure plots rather than asserted.
    "giant_learned": (
        "runs/paper_physical_seed_20260820/learned/spectra/"
        "learned_reduced_state/t04096.6_g+0.72_m-1.91_a-0.10_x3.87.npz"
    ),
    "giant_released": (
        "runs/paper_physical_seed_20260820/learned/spectra/"
        "production_six_field/t04096.6_g+0.72_m-1.91_a-0.10_x3.87.npz"
    ),
    "gate_learned": (
        "results/paper_physical_seed_20260820/learned/spectral_gate.json"
    ),
    # -- non-neural analytic comparison on the same development-60 --------
    "analytic_comparison": (
        "results/paper_physical_seed_20260820/analytic/"
        "paper_dev60_comparison.json"
    ),
    "analytic_comparison_arrays": (
        "results/paper_physical_seed_20260820/analytic/"
        "paper_dev60_comparison.npz"
    ),
    "physical_campaign": (
        "results/paper_physical_seed_20260820/campaign.json"
    ),
    "learned_checkpoint_asset": (
        "artifacts/reduced_state_emulator/checkpoint_monotone.pt"
    ),
    "analytic_parameter_asset": (
        "results/analytic_initializer/compact_profile_parameters_parity.npz"
    ),
    # -- zero-fit first-principles v4r6 development diagnostic ------------
    "vfour_seed_audit": (
        "results/analytic_initializer/"
        "textbook_opacity_v4r6_decoupled_seed_audit_20260828.json"
    ),
    "vfour_decoupled_fifteen": (
        "results/analytic_initializer/"
        "textbook_opacity_v4r6_decoupled_dev60_20260828.json"
    ),
    "vfour_decoupled_policy": (
        "results/analytic_initializer/"
        "textbook_opacity_v4r6_decoupled_dev60_policy60_20260829.json"
    ),
    "vfour_grey_policy": (
        "results/analytic_initializer/"
        "textbook_opacity_v4r6_grey_dev60_policy60_20260829.json"
    ),
    "vfour_convective_policy": (
        "results/analytic_initializer/"
        "textbook_opacity_v4r6_convective_dev60_policy60_20260829.json"
    ),
    "vfour_matched_score": (
        "results/analytic_initializer/"
        "textbook_opacity_v4r6_policy60_matched_dev60_20260829.json"
    ),
    "vfour_offline": (
        "results/analytic_initializer/"
        "textbook_opacity_v4r6_offline_validation_20260828.json"
    ),
    "vfour_policy_source_manifest": (
        "results/analytic_initializer/"
        "textbook_opacity_v4r6_policy60_source_manifest_20260829.json"
    ),
    # -- complete emulator-independent initializer evaluation -------------
    "grey_campaign": "results/paper_grey_convective_20260829/campaign.json",
    "grey_source_manifest": (
        "results/paper_grey_convective_20260829/source_manifest.json"
    ),
    "grey_output_manifest": (
        "results/paper_grey_convective_20260829/output_manifest.json"
    ),
    "grey_dev_seed": (
        "results/paper_grey_convective_20260829/development/seed_summary.json"
    ),
    "grey_dev_replay": (
        "results/paper_grey_convective_20260829/development/replay_check.json"
    ),
    "grey_dev_solver": (
        "results/paper_grey_convective_20260829/development/solver.json"
    ),
    "grey_dev_summary": (
        "results/paper_grey_convective_20260829/development/summary.json"
    ),
    "grey_dev_profiles": (
        "results/paper_grey_convective_20260829/development/profile_metrics.json"
    ),
    "grey_dev_spectra": (
        "results/paper_grey_convective_20260829/development/spectral_gate.json"
    ),
    "grey_post_seed": (
        "results/paper_grey_convective_20260829/posthoc200/seed_summary.json"
    ),
    "grey_post_solver": (
        "results/paper_grey_convective_20260829/posthoc200/solver.json"
    ),
    "grey_post_summary": (
        "results/paper_grey_convective_20260829/posthoc200/summary.json"
    ),
    "grey_post_profiles": (
        "results/paper_grey_convective_20260829/posthoc200/profile_metrics.json"
    ),
    "grey_post_spectra": (
        "results/paper_grey_convective_20260829/posthoc200/spectral_gate.json"
    ),
    # -- Sect. 5.6  what the 5e-3 bar means --------------------------------
    "jitter": "results/convergence_metrics_production_jitter.json",
    "gate_jitter": "results/spectral_gate_jitter_control.json",
    # -- Sect. 5.7  the sealed blind test ----------------------------------
    "blind": ("results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
              "blind200/summary.json"),
    "blind_profile": ("results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
                      "blind200/profile_gate.json"),
    "blind_base_prediction": (
        "results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
        "blind200/base.npz"
    ),
    "blind_final_prediction": (
        "results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
        "blind200/final.npz"
    ),
    "blind_spectra": ("results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
                      "blind200/spectral_gate.json"),
    "blind_open": ("results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
                   "blind200_physical_seed/summary.json"),
    "blind_open_spectra": (
        "results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
        "blind200_physical_seed/spectral_gate.json"
    ),
    "blind_production": ("results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
                         "blind200/summary.json"),
    "dev60_profile": ("results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
                      "profile_dev60.json"),
    "frozen_policy": ("results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
                      "frozen_blind_policy.json"),
    # -- per-star blind solver records, for the paired iteration statistic --
    "blind_cand_records": ("runs/reduced_state_emulator/"
                           "solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
                           "blind200/records/learned_reduced_state/records.jsonl"),
    "blind_open_records": ("runs/reduced_state_emulator/"
                           "solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
                           "blind200_physical_seed/records/"
                           "learned_reduced_state/records.jsonl"),
    "blind_prod_records": ("runs/reduced_state_emulator/"
                           "solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
                           "blind200/production_records/production_six_field/records.jsonl"),
    "blind_manifest": "results/sealed_audit_20260811.json",
    "dev60_gate": ("results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
                   "dev60_solver/spectral_gate.json"),
    # -- bounded native-MARCS M-star science case ------------------------
    "mstar_cases": "results/m_star_science_case_v1/cases.json",
}

# Human-readable descriptions for the provenance appendix.
SOURCE_CAPTIONS: dict[str, str] = {
    "training": "two-field emulator training and validation summary",
    "baseline": "six-field reference convergence by label distribution",
    "recon": (
        "dependent profiles reconstructed from converged $(m,T)$"
    ),
    "parity": (
        "solver convergence from reconstructed $(m,T)$ profiles and "
        "from the reference atmosphere"
    ),
    "parity_reduced_records": (
        "per-star convergence records from reconstructed $(m,T)$"
    ),
    "parity_truth_records": "per-star convergence records from the reference atmosphere",
    "gate_truth": (
        "spectral comparison of reconstructed $(m,T)$ and six-field atmospheres"
    ),
    "resolution": "dependence of reconstruction on depth-grid resolution",
    "continuity": "depth-grid refinement and top-boundary seed survey",
    "corpus": "converged-atmosphere corpus (seed-residual rank correlation)",
    "learned": "solver convergence from the learned two-field initializer",
    "learned_records": (
        "per-star convergence records for the learned two-field initializer"
    ),
    "production": "solver restart from the six-field initializer",
    "production_records": (
        "per-star convergence records for the six-field initializer"
    ),
    "derived_learned": (
        "dependent-field errors after learned two-field reconstruction"
    ),
    "derived_learned_summary": (
        "successful and unsuccessful physical reconstructions"
    ),
    "derived_production": "dependent-field errors, six fields predicted directly",
    "gate_learned": "spectral comparison of learned two-field and six-field states",
    "analytic_comparison": (
        "same-star learned two-field and analytic-initializer comparison"
    ),
    "analytic_comparison_arrays": (
        "learned and analytic profile arrays and paired iterations for the "
        "comparison figure"
    ),
    "physical_campaign": "computational record for the physical-reconstruction calculations",
    "learned_checkpoint_asset": "learned two-field model parameters",
    "analytic_parameter_asset": "analytic-initializer parameters",
    "vfour_seed_audit": (
        "finite-profile checks for the grey--convective initialization"
    ),
    "vfour_decoupled_fifteen": (
        "grey--convective initialization with a 15-iteration limit"
    ),
    "vfour_decoupled_policy": (
        "grey--convective initialization with a 60-iteration limit"
    ),
    "vfour_grey_policy": (
        "grey initialization with a 60-iteration limit"
    ),
    "vfour_convective_policy": (
        "coupled convective initialization with a 60-iteration limit"
    ),
    "vfour_matched_score": (
        "paired convergence comparison of the three physical initializations"
    ),
    "vfour_offline": (
        "opacity and mass-column accuracy of the physical initialization"
    ),
    "vfour_policy_source_manifest": (
        "source and computing environment for the physical comparison"
    ),
    "grey_campaign": (
        "complete emulator-independent initializer evaluation record"
    ),
    "grey_source_manifest": (
        "frozen source, samples, solver settings, and computing environments"
    ),
    "grey_output_manifest": (
        "SHA-256 inventory of all emulator-independent evaluation outputs"
    ),
    "grey_dev_seed": "finite physical seeds on the 60-star development sample",
    "grey_dev_replay": "strict reproduction of the frozen development result",
    "grey_dev_solver": (
        "full development-sample solver outcomes and iteration diagnostics"
    ),
    "grey_dev_summary": (
        "development convergence, iteration, profile, and spectral summary"
    ),
    "grey_dev_profiles": (
        "physical seed and converged-atmosphere profile errors on development stars"
    ),
    "grey_dev_spectra": (
        "physical-initializer spectral differences on development stars"
    ),
    "grey_post_seed": "finite physical seeds on the opened 200-star sample",
    "grey_post_solver": (
        "full post-hoc solver outcomes and iteration diagnostics"
    ),
    "grey_post_summary": (
        "post-hoc convergence, iteration, profile, and spectral summary"
    ),
    "grey_post_profiles": (
        "physical seed and converged-atmosphere profile errors on the opened sample"
    ),
    "grey_post_spectra": (
        "physical-initializer spectral differences on the opened sample"
    ),
    "giant_learned": (
        "Fig. 3 red giant, spectrum from the learned two-field initialization"
    ),
    "giant_released": (
        "Fig. 3 red giant, spectrum from the six-field reference"
    ),
    "jitter": "reference-solver convergence from an alternate accepted start",
    "gate_jitter": "spectral start dependence of the reference solver",
    "blind": "frozen two-field results on the independent 200-star sample",
    "blind_profile": "profile accuracy on the independent 200-star sample",
    "blind_base_prediction": "base two-field prediction on the independent sample",
    "blind_final_prediction": (
        "frozen corrected two-field prediction on the independent sample"
    ),
    "blind_spectra": "spectral accuracy on the independent 200-star sample",
    "blind_open": (
        "post-hoc reconstruction-seed check on the opened 200-star sample"
    ),
    "blind_open_spectra": (
        "spectral result of the post-hoc reconstruction-seed check"
    ),
    "blind_production": "six-field reference on the independent 200-star sample",
    "dev60_profile": "development-sample profile accuracy of the final two-field model",
    "frozen_policy": "final two-field model configuration fixed before testing",
    "blind_cand_records": (
        "two-field convergence records on the independent sample"
    ),
    "blind_open_records": (
        "two-field convergence records for the post-hoc reconstruction-seed check"
    ),
    "blind_prod_records": "six-field convergence records on the independent sample",
    "blind_manifest": "selection record for the independent 200-star sample",
    "dev60_gate": "development-sample spectral accuracy of the final two-field model",
    "mstar_cases": (
        "temperature and gravity dependence across eight native-MARCS M-star atmospheres"
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Artifacts:
    """Lazily loaded JSON artifacts, with their hashes recorded on first read."""

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}
        self.hashes: dict[str, str] = {}
        self.missing: list[str] = []

    def __call__(self, name: str):
        if name not in self._cache:
            path = REPO / SOURCES[name]
            if not path.is_file():
                self.missing.append(SOURCES[name])
                raise FileNotFoundError(path)
            self.hashes[name] = sha256(path)
            if path.suffix == ".npz":
                import numpy as np

                with np.load(path, allow_pickle=False) as data:
                    self._cache[name] = {key: data[key] for key in data.files}
            else:
                with path.open() as handle:
                    self._cache[name] = json.load(handle)
        return self._cache[name]

    def median(self, name: str, key: str) -> float:
        import numpy as np

        return float(np.median(self(name)[key]))


def load_jsonl_records(art: Artifacts, source: str) -> dict[str, dict]:
    """Load a JSONL record artifact with last-record-wins semantics and a hash."""

    path = REPO / SOURCES[source]
    art.hashes[source] = sha256(path)
    rows = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["slug"]] = row
    return rows


# --------------------------------------------------------------------------
# Formatting.  LaTeX macro names may contain letters only.
# --------------------------------------------------------------------------


def sci(value: float, digits: int = 2) -> str:
    """``8.1\\times10^{-5}`` -- always wrapped so it works in text or math mode."""
    if value == 0:
        return r"\ensuremath{0}"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    mantissa = round(mantissa, digits)
    if abs(mantissa) >= 10.0:  # rounding carried, e.g. 9.99 -> 10.0
        mantissa /= 10.0
        exponent += 1
    return rf"\ensuremath{{{mantissa:.{digits}f}\times10^{{{exponent}}}}}"


def dec(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    """Fraction in [0,1] to a percentage, with the percent sign attached."""
    return rf"{100.0 * value:.{digits}f}\%"


def integer(value: float) -> str:
    return f"{int(round(value))}"


def thousands(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", r"\,")


class Macros:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, object]] = {}

    def add(self, name: str, raw, text: str, source: str) -> str:
        if name in self.entries:
            raise KeyError(f"duplicate macro {name}")
        if not name.isalpha():
            raise ValueError(f"macro name must be letters only: {name}")
        self.entries[name] = {"value": raw, "tex": text, "source": SOURCES[source]}
        return text

    def render(self) -> str:
        lines = [
            "% Generated by paper/collect_numbers.py -- do not edit by hand.",
            "% Every macro below traces to a result artifact; see paper/numbers.json.",
            "",
        ]
        for name in sorted(self.entries):
            lines.append(rf"\newcommand{{\{name}}}{{{self.entries[name]['tex']}}}")
        lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

FIELD_KEYS = [
    ("gas_pressure", r"Gas pressure $P_{\mathrm{gas}}$"),
    ("electron_density", r"Electron density $n_{\mathrm{e}}$"),
    ("rosseland_opacity", r"Rosseland opacity $\kappa_{\mathrm{R}}$"),
    ("radiative_acceleration", r"Radiative acceleration $g_{\mathrm{rad}}$"),
]

def _validate_analytic_comparison_arrays(
    comparison: dict, arrays: dict[str, object]
) -> None:
    """Require the figure NPZ to reproduce every JSON metric used in the paper."""

    import numpy as np

    indices = np.asarray(arrays["star_indices"], dtype=np.int64)
    expected_indices = np.asarray(comparison["sample"]["star_indices"], dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise SystemExit("analytic comparison JSON and NPZ use different star order")

    def require_close(label: str, observed: float, expected: float) -> None:
        if not math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise SystemExit(
                f"analytic comparison JSON/NPZ mismatch for {label}: "
                f"{observed} != {expected}"
            )

    arm_arrays = {
        "learned_two_field": (
            "learned_temperature_relative_error",
            "learned_column_mass_dex_error",
            "learned_converged",
            "learned_iterations",
        ),
        "analytic_parity": (
            "analytic_temperature_relative_error",
            "analytic_column_mass_dex_error",
            "analytic_converged",
            "analytic_iterations",
        ),
    }
    derived: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm, (temperature_key, mass_key, converged_key, iterations_key) in (
        arm_arrays.items()
    ):
        temperature_error = np.asarray(arrays[temperature_key], dtype=np.float64)
        mass_error = np.asarray(arrays[mass_key], dtype=np.float64)
        converged = np.asarray(arrays[converged_key], dtype=bool)
        iterations = np.asarray(arrays[iterations_key], dtype=np.float64)
        if (
            temperature_error.shape[0] != indices.size
            or mass_error.shape != temperature_error.shape
            or converged.shape != (indices.size,)
            or iterations.shape != (indices.size,)
        ):
            raise SystemExit(f"analytic comparison NPZ has invalid {arm} shapes")
        if (
            not np.all(np.isfinite(temperature_error))
            or not np.all(np.isfinite(mass_error))
            or np.any(temperature_error < 0.0)
            or np.any(mass_error < 0.0)
        ):
            raise SystemExit(f"analytic comparison NPZ has invalid {arm} errors")
        profile = comparison[arm]["profile_errors"]
        for metric, values, percentile in (
            ("temperature_relative_p50", temperature_error, 50.0),
            ("temperature_relative_p95", temperature_error, 95.0),
            ("column_mass_dex_p50", mass_error, 50.0),
            ("column_mass_dex_p95", mass_error, 95.0),
        ):
            require_close(metric, np.percentile(values, percentile), profile[metric])
        converged_iterations = iterations[converged]
        if converged_iterations.size == 0 or not np.all(
            np.isfinite(converged_iterations)
        ):
            raise SystemExit(
                f"analytic comparison NPZ has invalid {arm} converged iterations"
            )
        solver = comparison[arm]["solver"]
        integer_checks = {
            "star_count": converged.size,
            "converged_count": converged.sum(),
            "failure_count": (~converged).sum(),
        }
        for metric, observed in integer_checks.items():
            if int(observed) != int(solver[metric]):
                raise SystemExit(
                    f"analytic comparison JSON/NPZ mismatch for {arm} {metric}"
                )
        for metric, observed in (
            ("mean_iterations_converged", converged_iterations.mean()),
            ("median_iterations_converged", np.median(converged_iterations)),
            ("p90_iterations_converged", np.percentile(converged_iterations, 90.0)),
        ):
            require_close(f"{arm} {metric}", observed, solver[metric])
        derived[arm] = (converged, iterations)

    learned_ok, learned_iterations = derived["learned_two_field"]
    analytic_ok, analytic_iterations = derived["analytic_parity"]
    common = learned_ok & analytic_ok
    learned_only = learned_ok & ~analytic_ok
    analytic_only = analytic_ok & ~learned_ok
    neither = ~learned_ok & ~analytic_ok
    difference = analytic_iterations[common] - learned_iterations[common]
    paired = comparison["paired_solver"]
    paired_integer_checks = {
        "common_converged_count": common.sum(),
        "learned_only_converged_count": learned_only.sum(),
        "analytic_only_converged_count": analytic_only.sum(),
        "neither_converged_count": neither.sum(),
        "learned_fewer_iterations_count": (difference > 0.0).sum(),
        "analytic_fewer_iterations_count": (difference < 0.0).sum(),
        "tied_count": (difference == 0.0).sum(),
    }
    for metric, observed in paired_integer_checks.items():
        if int(observed) != int(paired[metric]):
            raise SystemExit(
                f"analytic comparison JSON/NPZ mismatch for paired {metric}"
            )
    require_close(
        "paired mean iteration difference",
        difference.mean(),
        paired["mean_analytic_minus_learned_iterations"],
    )
    require_close(
        "paired median iteration difference",
        np.median(difference),
        paired["median_analytic_minus_learned_iterations"],
    )


def build(art: Artifacts, macros: Macros) -> dict[str, str]:
    """Populate macros and return {table filename: table body}."""
    tables: dict[str, str] = {}

    # ---- initializer families -------------------------------------------
    # Static prose table: the numbers it quotes (sizes, epochs, seeds, hashes)
    # are emitted as macros below and assembled here, so the table cannot
    # drift from the artifacts either.
    campaign = art("physical_campaign")
    required_stages = {
        "parity",
        "depth_n40",
        "depth_n80",
        "depth_n160",
        "depth_n320",
        "depth_n640",
        "depth_aggregate",
        "learned_restart",
        "derived_fields",
        "learned_spectral_gate",
        "parity_spectral_gate",
    }
    recorded_stages = {
        Path(path).stem for path in campaign.get("stage_markers", [])
    }
    if campaign.get("campaign") != "paper_physical_seed_20260820":
        raise SystemExit("wrong physical-seed campaign manifest")
    if campaign.get("reconstruction_seed") != "physical":
        raise SystemExit("paper refresh did not use the physical seed")
    if recorded_stages != required_stages:
        missing = sorted(required_stages - recorded_stages)
        extra = sorted(recorded_stages - required_stages)
        raise SystemExit(
            f"incomplete physical-seed campaign markers; missing={missing}, extra={extra}"
        )
    if "unchanged" not in campaign.get("two_field_predictor", ""):
        raise SystemExit("campaign manifest does not freeze the two-field predictor")
    if "reused frozen" not in campaign.get("production_arm", ""):
        raise SystemExit("campaign manifest does not freeze the production arm")
    analytic_comparison = art("analytic_comparison")
    analytic_arrays = art("analytic_comparison_arrays")
    _validate_analytic_comparison_arrays(analytic_comparison, analytic_arrays)
    for asset_name in (
        "learned_checkpoint_asset",
        "analytic_parameter_asset",
        "vfour_policy_source_manifest",
    ):
        art.hashes[asset_name] = sha256(REPO / SOURCES[asset_name])
    col = (r">{\raggedright\arraybackslash}p{0.17\hsize}"
           r">{\raggedright\arraybackslash}p{0.19\hsize}"
           r">{\raggedright\arraybackslash}p{0.29\hsize}"
           r">{\raggedright\arraybackslash}p{0.27\hsize}")
    tables["tab_families.tex"] = table_env(
        label="tab:families",
        caption=(
            "Initialization methods and the evidence available for each. "
            "The independent-test model uses the radial-basis corrections "
            "defined in Sect.~\\ref{sec:learned}; the grey--convective "
            "construction contains no fitted atmospheric profiles."),
        colspec=col,
        header=(r"Initialization & Input $\rightarrow$ state & Fitted information & "
                r"Evidence \\"),
        rows=[
            " & ".join([
                r"Six-field (Paper~I)",
                r"5 labels $\rightarrow$ 6 fields, direct",
                r"Paper~I; not retrained here",
                (r"reference in Tables~\ref{tab:baseline}, "
                 r"\ref{tab:learned} and \ref{tab:blind}"),
            ]) + r" \\",
            " & ".join([
                r"Two-field emulator (this work)",
                (r"5 labels $\rightarrow$ $(\log_{10} m, "
                 r"\log_{10} T)$, monotone increments of "
                 r"Eq.~(\ref{eq:monotone})"),
                (r"\NnetDepth$\times$\NnetWidth\ SiLU network; "
                 r"\NtrainStars\ training and \NvalStars\ validation "
                 r"atmospheres; development stars excluded"),
                (r"development comparisons in Tables~\ref{tab:fields}, "
                 r"\ref{tab:learned}"),
            ]) + r" \\",
            " & ".join([
                r"Fitted analytic formula (this work)",
                (r"5 labels $\rightarrow$ $(T,\kappa_{\rm R})$; "
                 r"$m$ from ${\rm d}m/{\rm d}\tau=1/\kappa_{\rm R}$"),
                (r"\NanalyticTrainStars/\NanalyticValStars\ train/validation "
                 r"rows; three $T_{\rm eff}$ regimes, cubic labels and five "
                 r"Chebyshev depth modes; \NanalyticConstants\ floating-point "
                 r"coefficients and fixed constants"),
                (r"profile and development-sample solver comparison in "
                 r"Table~\ref{tab:analytic}; no spectral or independent test"),
            ]) + r" \\",
            " & ".join([
                r"Grey--convective physical model",
                (r"5 labels $\rightarrow m_{\rm grey},T_{\rm conv},"
                 r"\kappa_{\rm R}$; no post-convection mass reintegration"),
                (r"no fitted profiles; approximate opacity, hydrostatic "
                 r"$P=gm_{\rm grey}$ and Saha-aware convective temperature"),
                (r"development-sample comparison in "
                 r"Fig.~\ref{fig:vfour}; convergence in "
                 r"Table~\ref{tab:physical_initializer}; post-hoc 200-star "
                 r"comparison and paired spectra"),
            ]) + r" \\",
            " & ".join([
                r"Independent-test two-field model",
                r"two-field base + two label-local radial-basis corrections",
                (r"base retrained without the development or independent-test "
                 r"rows; a three-centre $\sigma=0.30$ one-step correction and "
                 r"a 20-centre $\sigma=0.08$ profile correction, all fixed "
                 r"before testing"),
                (r"independent test in Table~\ref{tab:blind} and "
                 r"Fig.~\ref{fig:blind}"),
            ]) + r" \\",
        ],
        wide=True,
        colsep=3.5,
        size="footnotesize",
    )

    analytic_config = analytic_comparison["analytic_parity"]["configuration"]
    analytic_breakdown = analytic_comparison["analytic_parity"]["parameter_breakdown"]
    temp_config = analytic_config["temperature"]
    opacity_config = analytic_config["opacity"]
    tables["tab_analytic_structure.tex"] = table_env(
        label="tab:analytic-structure",
        caption=(
            "Parameterization of the fitted analytic formula. Each closure "
            "uses three effective-temperature regimes and "
            f"{integer(analytic_breakdown['label_polynomial_terms_per_regime'])} "
            "complete cubic label terms per regime. The two closures share the "
            "same ten label-centering and scaling constants. The total counts "
            "these shared constants once and includes eleven constants that "
            "define the support and enforce monotonicity."),
        colspec="llrrrrr",
        header=(
            r"Component & Fitted target & Label degree & Modes & "
            r"Mean degree & Mode degree & Float parameters \\"
        ),
        rows=[
            " & ".join([
                "Temperature",
                r"$\log_{10}(T/T_{\rm grey})$",
                integer(temp_config["degree"]),
                integer(temp_config["components"]),
                integer(temp_config["center_degree"]),
                integer(temp_config["mode_degree"]),
                integer(analytic_breakdown["temperature_closure_raw"]),
            ]) + r" \\",
            " & ".join([
                "Opacity",
                r"$\log_{10}\kappa_{\rm R}$",
                integer(opacity_config["degree"]),
                integer(opacity_config["components"]),
                integer(opacity_config["center_degree"]),
                integer(opacity_config["mode_degree"]),
                integer(analytic_breakdown["opacity_closure_raw"]),
            ]) + r" \\",
            " & ".join([
                "Support and monotonicity", "---", "---", "---", "---", "---",
                integer(analytic_breakdown["support_and_monotonicity_guards"]),
            ]) + r" \\",
            " & ".join([
                r"\textbf{Total}", "---", "---", "---", "---", "---",
                rf"\textbf{{{integer(analytic_breakdown['total'])}}}",
            ]) + r" \\",
        ],
        wide=True,
        colsep=4.0,
        size="small",
    )


    # ---- corpus and protocol -------------------------------------------
    training = art("training")
    macros.add("NtrainSeed", training["seed"], integer(training["seed"]),
               "training")
    macros.add("NcorpusStars", training["star_count"],
               thousands(training["star_count"]), "training")
    macros.add("NtrainStars", training["train_count"],
               thousands(training["train_count"]), "training")
    macros.add("NvalStars", training["validation_count"],
               thousands(training["validation_count"]), "training")
    macros.add("NevalStars", len(training["held_out_indices"]),
               integer(len(training["held_out_indices"])), "training")
    macros.add("NnetWidth", training["width"], integer(training["width"]), "training")
    macros.add("NnetDepth", training["depth"], integer(training["depth"]), "training")
    macros.add("NnetEpochs", training["epochs"], integer(training["epochs"]), "training")

    recon = art("recon")
    macros.add("NgridLayers", len(recon["tau_std"]), integer(len(recon["tau_std"])), "recon")
    macros.add("NsyncPasses", recon["n_synchronizations"],
               integer(recon["n_synchronizations"]), "recon")
    macros.add("NhardFraction", recon["hard_fraction"], pct(recon["hard_fraction"], 0), "recon")

    # ---- Table 1: solver baseline by label slice ------------------------
    baseline = art("baseline")
    slice_titles = {
        "box": "Uniform box",
        "iid": "IID from corpus",
        "boundary": "Label-box boundary",
        "hard": "Low gravity, metal poor",
    }
    rows = []
    for entry in baseline["slices"]:
        rows.append(
            " & ".join([
                slice_titles.get(entry["slice"], entry["slice"]),
                integer(entry["star_count"]),
                pct(entry["converged_fraction"]),
                pct(entry["failure_fraction"]),
                dec(entry["iterations_mean"], 2),
                integer(entry["iterations_p90"]),
                dec(entry["contraction_geomean_q"], 3),
                pct(entry["non_monotonic_fraction"]),
            ]) + r" \\"
        )
    tables["tab_baseline.tex"] = table_env(
        label="tab:baseline",
        caption=(
            "Convergence from the Paper~I six-field initializer for different "
            "stellar-label distributions. The solver and initializer are "
            "identical in every row; only the sampling within the support of "
            "Sect.~\\ref{sec:data} differs. "
            "\\emph{Uniform box} draws each label uniformly over that support, which "
            "covers the box rather than the stellar locus and so includes combinations "
            "no real star occupies; \\emph{IID from corpus} resamples whole corpus rows "
            "with replacement, preserving the joint distribution the initializer was "
            "trained on; \\emph{label-box boundary} pins at least one label within "
            "5\\% of a face of that support; \\emph{low gravity, metal poor} samples "
            "uniformly inside $\\log g \\in [0.7, 2.8]$, $[\\rm M/H] \\in [-2.5, -0.5]$. "
            "Iteration counts include converged calculations only; $q$ is the "
            "geometric-mean ratio of successive temperature residuals, with "
            "$q<1$ indicating contraction."),
        colspec="lrrrrrrr",
        header=(r"Label sampling & $N$ & Converged & Failed & Iterations & $p_{90}$ "
                r"& $q$ & Non-monotonic \\"),
        rows=rows,
        wide=True,
    )
    box = next(e for e in baseline["slices"] if e["slice"] == "box")
    iid = next(e for e in baseline["slices"] if e["slice"] == "iid")
    macros.add("NboxConverged", box["converged_fraction"],
               pct(box["converged_fraction"]), "baseline")
    macros.add("NboxFailed", box["failure_fraction"],
               pct(box["failure_fraction"]), "baseline")
    macros.add("NboxNonmono", box["non_monotonic_fraction"],
               pct(box["non_monotonic_fraction"], 0), "baseline")
    macros.add("NboxQ", box["contraction_geomean_q"],
               dec(box["contraction_geomean_q"], 3), "baseline")
    macros.add("NboxRecoverable", box["recoverable_fraction"],
               pct(box["recoverable_fraction"]), "baseline")
    macros.add("NiidConverged", iid["converged_fraction"],
               pct(iid["converged_fraction"]), "baseline")
    macros.add("NiidPninety", iid["iterations_p90"], integer(iid["iterations_p90"]), "baseline")

    # ---- Table 2: rematerialization parity from converged (m,T) ---------
    rows = []
    for key, title in FIELD_KEYS:
        stats = recon[key]
        rows.append(" & ".join([
            title,
            sci(stats["median_overall"]),
            sci(stats["p90_overall"]),
            sci(stats["max_overall"]),
        ]) + r" \\")
    tables["tab_reconstruction.tex"] = table_env(
        label="tab:reconstruction",
        caption=(
            "Relative errors of the four dependent fields reconstructed from "
            "the converged column-mass and temperature profiles, evaluated "
            r"over \NevalStars\ stars and \NgridLayers\ depth points."),
        colspec="lrrr",
        header=r"Field & Median & $p_{90}$ & Maximum \\",
        rows=rows,
        colsep=3.5,
        size="small",
    )
    for key, _ in FIELD_KEYS:
        stem = "".join(part.capitalize() for part in key.split("_"))
        macros.add(f"NoracleRecon{stem}", recon[key]["median_overall"],
                   sci(recon[key]["median_overall"]), "recon")
    recon_p90 = [recon[key]["p90_overall"] for key, _ in FIELD_KEYS]
    recon_max = [recon[key]["max_overall"] for key, _ in FIELD_KEYS]
    macros.add("NreconPninetyLow", min(recon_p90),
               sci(min(recon_p90)), "recon")
    macros.add("NreconPninetyHigh", max(recon_p90),
               sci(max(recon_p90)), "recon")
    macros.add("NreconMaximumLow", min(recon_max),
               sci(min(recon_max)), "recon")
    macros.add("NreconMaximumHigh", max(recon_max),
               sci(max(recon_max)), "recon")

    # ---- Table 3: restart from converged (m,T) vs the six-field truth -----
    parity = art("parity")
    arm_titles = [
        ("reduced_state_reconstruction", r"Converged $(m,T)$ + physics"),
        ("full_truth_oracle", "Full six-field truth"),
    ]
    rows = []
    for key, title in arm_titles:
        arm = parity[key]
        iters = arm["converging_trial_iterations"]
        rows.append(" & ".join([
            title,
            pct(arm["converged_fraction"], 0),
            dec(iters["mean"], 2),
            integer(iters["p90"]),
            dec(arm["contraction"]["q_ratio"]["geometric_mean"], 3),
            pct(arm["contraction"]["non_monotonic_fraction"]),
            sci(arm["contraction"]["first_iteration_residual"]["p50"]),
            integer(arm["headroom"]["stars_already_at_floor"]),
        ]) + r" \\")
    tables["tab_oracle_restart.tex"] = table_env(
        label="tab:oracle",
        caption=(
            "Convergence from atmospheres reconstructed from exact $(m,T)$ and "
            "from the original six-field converged states. Both initializations "
            "begin within the convergence threshold; their $q$ and non-monotonic "
            "fractions therefore describe numerical behavior near the fixed "
            "point. The last column gives the number of stars at the minimum "
            "of three iterations."),
        colspec="lrrrrrrr",
        header=(r"Restart from & Conv. & Iterations & $p_{90}$ & $q$ & Non-mono. "
                r"& First residual & At floor \\"),
        rows=rows,
        wide=True,
    )
    red = parity["reduced_state_reconstruction"]
    full = parity["full_truth_oracle"]
    macros.add("NoracleIterMean", red["converging_trial_iterations"]["mean"],
               dec(red["converging_trial_iterations"]["mean"], 2), "parity")
    macros.add("NsixIterMean", full["converging_trial_iterations"]["mean"],
               dec(full["converging_trial_iterations"]["mean"], 2), "parity")
    macros.add("NoracleAtFloor", red["headroom"]["stars_already_at_floor"],
               integer(red["headroom"]["stars_already_at_floor"]), "parity")
    macros.add("NsixAtFloor", full["headroom"]["stars_already_at_floor"],
               integer(full["headroom"]["stars_already_at_floor"]), "parity")
    macros.add("NoracleFirstResidual",
               red["contraction"]["first_iteration_residual"]["p50"],
               sci(red["contraction"]["first_iteration_residual"]["p50"]), "parity")
    parity_records = {
        source: load_jsonl_records(art, source)
        for source in ("parity_reduced_records", "parity_truth_records")
    }
    paired_slugs = sorted(
        set(parity_records["parity_reduced_records"])
        & set(parity_records["parity_truth_records"])
    )
    if len(paired_slugs) != red["star_count"]:
        raise SystemExit(
            f"paired parity records contain {len(paired_slugs)} stars, "
            f"expected {red['star_count']}"
        )
    paired_differences = [
        parity_records["parity_reduced_records"][slug][
            "converging_trial_iterations"
        ]
        - parity_records["parity_truth_records"][slug][
            "converging_trial_iterations"
        ]
        for slug in paired_slugs
    ]
    if any(value < 0 for value in paired_differences):
        raise SystemExit("physical-seed oracle restart unexpectedly beat truth")
    macros.add(
        "NoracleGapStars",
        sum(value != 0 for value in paired_differences),
        integer(sum(value != 0 for value in paired_differences)),
        "parity_reduced_records",
    )
    macros.add(
        "NoracleGapMax",
        max(paired_differences),
        integer(max(paired_differences)),
        "parity_reduced_records",
    )

    # ---- Sect. 5.2  spectral sufficiency --------------------------------
    gate_truth = art("gate_truth")
    macros.add("NgateBar", gate_truth["bar"], sci(gate_truth["bar"], 1), "gate_truth")
    macros.add("NgateResolution", gate_truth["resolution"],
               thousands(gate_truth["resolution"]), "gate_truth")
    macros.add("NgateWindowLo", gate_truth["window_nm"][0],
               integer(gate_truth["window_nm"][0]), "gate_truth")
    macros.add("NgateWindowHi", gate_truth["window_nm"][1],
               integer(gate_truth["window_nm"][1]), "gate_truth")
    nf = gate_truth["normalized_flux"]
    macros.add("NtruthGateStars", gate_truth["gated_star_count"],
               integer(gate_truth["gated_star_count"]), "gate_truth")
    macros.add("NtruthGateMedian", nf["median_over_stars"],
               sci(nf["median_over_stars"]), "gate_truth")
    macros.add("NtruthGateMax", nf["max_over_stars"], sci(nf["max_over_stars"]), "gate_truth")
    macros.add("NtruthGateOver", nf["stars_over_bar"], integer(nf["stars_over_bar"]), "gate_truth")
    macros.add("NtruthGateFactor", gate_truth["bar"] / nf["median_over_stars"],
               integer(gate_truth["bar"] / nf["median_over_stars"]), "gate_truth")

    # ---- Table 4: representation resolution -----------------------------
    resolution = art("resolution")
    rows = []
    for key in sorted(resolution["by_resolution"], key=int):
        entry = resolution["by_resolution"][key]
        rep = entry["representation_error"]["column_mass_relative_error"]["median"]
        conv = entry["convergence"]
        rows.append(" & ".join([
            integer(entry["resolution"]),
            sci(rep) if rep > 1e-14 else r"\ensuremath{<10^{-14}}",
            pct(conv["converged_fraction"]),
            dec(conv["converging_trial_iterations"]["mean"], 2),
            integer(conv["converging_trial_iterations"]["p90"]),
            dec(conv["contraction"]["q_ratio"]["geometric_mean"], 3),
            pct(conv["contraction"]["non_monotonic_fraction"]),
        ]) + r" \\")
    tables["tab_resolution.tex"] = table_env(
        label="tab:resolution",
        caption=(
            "The converged $(m,T)$ curves are round-tripped through an intermediate "
            "logarithmic optical-depth grid of $N$ points and back onto the atmosphere "
            "grid before physical reconstruction, so that only the fidelity of the "
            "representation changes. Representation error is the median relative error "
            "in column mass; $N=\\NgridLayers$ is the identity. Restart behavior "
            "remains near the three-iteration floor across the scan."),
        colspec="rrrrrrr",
        header=(r"$N$ & Repr. error & Conv. & Iterations & $p_{90}$ & $q$ "
                r"& Non-mono. \\"),
        rows=rows,
        wide=True,
    )
    coarse = resolution["by_resolution"]["40"]
    finest = resolution["by_resolution"]["640"]
    macros.add("NresCoarse", 40, "40", "resolution")
    macros.add("NresFinest", 640, "640", "resolution")
    macros.add("NresCoarseError",
               coarse["representation_error"]["column_mass_relative_error"]["median"],
               sci(coarse["representation_error"]["column_mass_relative_error"]["median"]),
               "resolution")
    macros.add("NresFinestError",
               finest["representation_error"]["column_mass_relative_error"]["median"],
               sci(finest["representation_error"]["column_mass_relative_error"]["median"]),
               "resolution")

    # ---- Table 5: depth-grid refinement (the top boundary) --------------
    continuity = art("continuity")
    scan = continuity["refinement_scan"]["by_refinement"]
    rows = []
    for key in sorted(scan, key=int):
        entry = scan[key]
        res = entry["top_layer_residual_dex"]
        drift = entry["column_mass_drift_percent"]
        rows.append(" & ".join([
            rf"$\times{int(key)}$",
            integer(entry["grid_points"]),
            dec(res["median"], 5),
            dec(res["p99"], 5),
            dec(res["max"], 5),
            dec(drift["p99"], 2),
            dec(drift["max"], 2),
        ]) + r" \\")
    tables["tab_continuity.tex"] = table_env(
        label="tab:continuity",
        caption=(
            "Optical-depth closure residual in the top five layers as the quadrature "
            "grid is refined, using the solver's own integrator and holding "
            r"$\tau_{\mathrm{min}}$ fixed so that spacing is isolated from the boundary. "
            "Sixteen times the grid density moves the worst case in the fourth decimal "
            "place, because layer~0 involves no quadrature at all."),
        colspec="lrrrrrr",
        header=(r"Refinement & Points & \multicolumn{3}{c}{Closure residual (dex)} "
                r"& \multicolumn{2}{c}{Mass drift (\%)} \\"
                "\n"
                r"& & Median & $p_{99}$ & Max & $p_{99}$ & Max \\"),
        rows=rows,
        wide=True,
    )
    continuity_base = scan["1"]["top_layer_residual_dex"]
    continuity_refined = scan["16"]["top_layer_residual_dex"]
    macros.add("NcontinuityBasePninetyNine", continuity_base["p99"],
               dec(continuity_base["p99"], 5), "continuity")
    macros.add("NcontinuityRefinedPninetyNine", continuity_refined["p99"],
               dec(continuity_refined["p99"], 5), "continuity")
    macros.add("NcontinuityBaseMax", continuity_base["max"],
               dec(continuity_base["max"], 5), "continuity")
    macros.add("NcontinuityRefinedMax", continuity_refined["max"],
               dec(continuity_refined["max"], 5), "continuity")
    seed = continuity["seed_survey"]
    macros.add("NseedMedian", seed["residual_dex"]["median"],
               sci(seed["residual_dex"]["median"]), "continuity")
    macros.add("NseedPninetynine", seed["residual_dex"]["p99"],
               dec(seed["residual_dex"]["p99"], 4), "continuity")
    macros.add("NseedMax", seed["residual_dex"]["max"],
               dec(seed["residual_dex"]["max"], 3), "continuity")
    macros.add("NseedAboveTenth", seed["fraction_above_0.10_dex"],
               pct(seed["fraction_above_0.10_dex"], 2), "continuity")
    macros.add("NseedAboveHundredth", seed["fraction_above_0.01_dex"],
               pct(seed["fraction_above_0.01_dex"], 2), "continuity")
    broken = continuity["failure_population"]["broken"]
    rest = continuity["failure_population"]["rest"]
    macros.add("NbrokenCount", broken["count"], integer(broken["count"]), "continuity")
    macros.add("NbrokenIters", broken["iterations_median"],
               integer(broken["iterations_median"]), "continuity")
    macros.add("NrestIters", rest["iterations_median"],
               integer(rest["iterations_median"]), "continuity")
    macros.add("NbrokenLogg", broken["log_surface_gravity_mean"],
               dec(broken["log_surface_gravity_mean"], 2), "continuity")
    macros.add("NrestLogg", rest["log_surface_gravity_mean"],
               dec(rest["log_surface_gravity_mean"], 2), "continuity")
    macros.add("NbrokenMetal", broken["metallicity_mean"],
               dec(broken["metallicity_mean"], 2), "continuity")
    macros.add("NrestMetal", rest["metallicity_mean"],
               dec(rest["metallicity_mean"], 2), "continuity")

    # Rank correlation of seed residual against iteration count.  The harness
    # reports this in prose only, so it is recomputed from the corpus using the
    # harness's own closed-form seed residual, and cross-checked against the
    # median the summary does record.
    import numpy as np
    from scipy.stats import spearmanr

    from continuity.closure import seed_residual

    corpus = art("corpus")
    fields = [str(name) for name in corpus["target_fields"]]
    profiles = corpus["atmosphere_profiles"]
    residual = np.abs(seed_residual(
        profiles[:, :, fields.index("column_mass")],
        profiles[:, :, fields.index("rosseland_opacity")],
        corpus["standard_rosseland_optical_depth"],
    ))
    iterations = corpus["iterations_to_convergence"].astype(float)
    recomputed_median = float(np.median(residual))
    recorded_median = seed["residual_dex"]["median"]
    if abs(recomputed_median - recorded_median) > 1e-12:
        raise SystemExit(
            "seed-residual recomputation does not reproduce the harness: "
            f"{recomputed_median!r} vs {recorded_median!r}")
    rho = float(spearmanr(residual, iterations).statistic)
    macros.add("NseedSpearman", rho, dec(rho, 3), "corpus")

    # ---- Table 6: the four dependent fields, three ways ------------------
    # All three arms below are scored the same way (median relative error over
    # every star and layer) and the two learned arms are the same checkpoints
    # compared on the solver in Table~\ref{tab:learned}.
    def derived(source: str, field_key: str) -> float:
        return art.median(source, f"{field_key}_relative_error")

    derived_arrays = art("derived_learned")
    derived_summary = art("derived_learned_summary")
    learned_for_fields = art("learned")
    requested_derived = np.asarray(
        derived_arrays["requested_star_indices"], dtype=np.int64
    )
    successful_derived = np.asarray(
        derived_arrays["star_indices"], dtype=np.int64
    )
    failed_derived = np.asarray(
        derived_arrays["failed_star_indices"], dtype=np.int64
    )
    if not np.array_equal(
        requested_derived,
        np.asarray(learned_for_fields["star_indices"], dtype=np.int64),
    ):
        raise SystemExit("derived-field run used a different development sample")
    if set(successful_derived) & set(failed_derived):
        raise SystemExit("derived-field success and failure sets overlap")
    if set(successful_derived) | set(failed_derived) != set(requested_derived):
        raise SystemExit("derived-field success and failure sets do not close")
    if (
        int(derived_summary["requested_count"]) != requested_derived.size
        or int(derived_summary["successful_count"]) != successful_derived.size
        or int(derived_summary["failure_count"]) != failed_derived.size
    ):
        raise SystemExit("derived-field JSON and NPZ counts disagree")
    summary_failed = {
        int(failure["star_index"]) for failure in derived_summary["failures"]
    }
    learned_failed = {
        int(failure["star_index"])
        for failure in learned_for_fields[
            "learned_reduced_state_reconstruction"
        ]["failures"]
    }
    if summary_failed != set(failed_derived) or summary_failed != learned_failed:
        raise SystemExit("derived-field failures disagree with the learned solver run")
    macros.add(
        "NderivedLearnedStars",
        successful_derived.size,
        integer(successful_derived.size),
        "derived_learned_summary",
    )
    macros.add(
        "NderivedLearnedFailures",
        failed_derived.size,
        integer(failed_derived.size),
        "derived_learned_summary",
    )

    arm_order = [
        ("derived_production", "Six fields predicted directly"),
        ("derived_learned", r"Learned two-field $(m,T)$ + physics"),
        ("recon", r"Converged $(m,T)$ + physics"),
    ]
    rows = []
    for source, title in arm_order:
        cells = [title]
        for field_key, _ in FIELD_KEYS:
            value = (recon[field_key]["median_overall"] if source == "recon"
                     else derived(source, field_key))
            cells.append(sci(value))
        rows.append(" & ".join(cells) + r" \\")
    tables["tab_derived_fields.tex"] = table_env(
        label="tab:fields",
        caption=(
            "Median relative error of the four dependent fields over the "
            r"development sample and \NgridLayers\ depth points. The learned "
            r"row contains the \NderivedLearnedStars\ successful physical "
            r"reconstructions; the remaining \NderivedLearnedFailures\ case "
            "has no complete dependent-field profile. The exact-$(m,T)$ row "
            "quantifies the numerical accuracy of the physical reconstruction. "
            "All rows refer to the same sampled stars."),
        colspec="lrrrr",
        header=(r"Source of the four fields & $P_{\mathrm{gas}}$ & $n_{\mathrm{e}}$ "
                r"& $\kappa_{\mathrm{R}}$ & $g_{\mathrm{rad}}$ \\"),
        rows=rows,
        wide=True,
    )
    stems = {"derived_production": "Sixfield", "derived_learned": "Twofield",
             "recon": "Oracle"}
    ratios = []
    for source, _ in arm_order:
        for field_key, _ in FIELD_KEYS:
            fstem = "".join(part.capitalize() for part in field_key.split("_"))
            value = (recon[field_key]["median_overall"] if source == "recon"
                     else derived(source, field_key))
            macros.add(f"N{stems[source]}{fstem}", value, sci(value), source)
    for field_key, _ in FIELD_KEYS:
        ratios.append(derived("derived_production", field_key)
                      / derived("derived_learned", field_key))
    macros.add("NfieldGainLow", min(ratios), dec(min(ratios), 1), "derived_learned")
    macros.add("NfieldGainHigh", max(ratios), dec(max(ratios), 1), "derived_learned")
    macros.add("NoracleGap",
               derived("derived_learned", "gas_pressure")
               / recon["gas_pressure"]["median_overall"],
               integer(derived("derived_learned", "gas_pressure")
                       / recon["gas_pressure"]["median_overall"]), "derived_learned")

    # ---- Table 7: learned two-field against the six-field initializer -----
    learned = art("learned")
    production = art("production")
    jitter = art("jitter")
    if learned.get("synchronization_mode") != "adaptive":
        raise SystemExit("refreshed learned arm is not the adaptive physical-seed run")
    if learned.get("star_indices") != recon.get("star_indices"):
        raise SystemExit("learned and parity refreshes used different development stars")
    macros.add(
        "NsyncMaxPasses",
        learned["max_synchronizations"],
        integer(learned["max_synchronizations"]),
        "learned",
    )
    macros.add(
        "NsyncPressureTolerance",
        learned["pressure_tolerance_dex"],
        sci(learned["pressure_tolerance_dex"]),
        "learned",
    )
    learned_restart = dict(learned["learned_reduced_state"])
    learned_reconstruction = learned["learned_reduced_state_reconstruction"]
    if (
        learned_reconstruction["synchronized_count"]
        != learned_restart["star_count"]
    ):
        raise SystemExit("learned reconstruction and solver star counts disagree")
    learned_restart["converged_fraction"] = (
        learned_restart["converged_count"] / learned["star_count"]
    )
    solver_rows = [
        ("Six-field initializer", production["production_six_field"]),
        ("Learned two-field + physics", learned_restart),
        ("Six-field, alternate converged start", jitter["production_jitter"]),
    ]
    rows = []
    for title, arm in solver_rows:
        iters = arm["converging_trial_iterations"]
        rows.append(" & ".join([
            title,
            pct(arm["converged_fraction"]),
            dec(iters["mean"], 2),
            integer(iters["p90"]),
            dec(iters["p99"], 2),
            dec(arm["contraction"]["q_ratio"]["geometric_mean"], 3),
            pct(arm["contraction"]["non_monotonic_fraction"]),
            sci(arm["contraction"]["first_iteration_residual"]["p50"]),
        ]) + r" \\")
    tables["tab_learned_solver.tex"] = table_env(
        label="tab:learned",
        caption=(
            r"Convergence from three initializations on the same \NevalStars\ "
            "development stars. All begin at comparable initial residuals, so "
            "their contraction statistics can be compared directly. The "
            "learned two-field model is the "
            r"\NnetDepth$\times$\NnetWidth\ monotone network described in "
            r"Sect.~\ref{sec:learned}; the independently tested model in "
            r"Table~\ref{tab:blind} includes the radial-basis corrections of "
            r"Sect.~\ref{sec:learned}."),
        colspec="lrrrrrrr",
        header=(r"Restart from & Conv. & Iterations & $p_{90}$ & $p_{99}$ & $q$ "
                r"& Non-mono. & First residual \\"),
        rows=rows,
        wide=True,
    )
    lrs = learned_restart
    psf = production["production_six_field"]
    pj = jitter["production_jitter"]
    learned_records = load_jsonl_records(art, "learned_records")
    production_records = load_jsonl_records(art, "production_records")
    import numpy as np
    from scipy.stats import spearmanr as _spearmanr

    for label, records, arm in (
        ("learned", learned_records, lrs),
        ("production", production_records, psf),
    ):
        converged_iterations = np.asarray(
            [
                row["converging_trial_iterations"]
                for row in records.values()
                if row["converged"]
            ],
            dtype=float,
        )
        if len(records) != arm["star_count"]:
            raise SystemExit(
                f"{label} records contain {len(records)} stars, "
                f"expected {arm['star_count']}"
            )
        if converged_iterations.size != arm["converged_count"]:
            raise SystemExit(f"{label} record convergence count disagrees with summary")
        if not np.isclose(
            converged_iterations.mean(),
            arm["converging_trial_iterations"]["mean"],
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise SystemExit(f"{label} record mean iterations disagree with summary")
    common_converged = sorted(
        slug
        for slug in set(learned_records) & set(production_records)
        if learned_records[slug]["converged"]
        and production_records[slug]["converged"]
    )
    if not common_converged:
        raise SystemExit("learned and production records have no common converged stars")

    gain = np.asarray(
        [
            production_records[slug]["converging_trial_iterations"]
            - learned_records[slug]["converging_trial_iterations"]
            for slug in common_converged
        ],
        dtype=float,
    )
    gain_teff = np.asarray(
        [
            learned_records[slug]["labels"]["effective_temperature"]
            for slug in common_converged
        ],
        dtype=float,
    )
    gain_rho, gain_p = _spearmanr(gain_teff, gain)
    hot = gain_teff > 9000.0
    if not hot.any() or hot.all():
        raise SystemExit("learned gain comparison lacks a hot or cool subsample")
    macros.add(
        "NlearnedGainRho", gain_rho, dec(float(gain_rho), 2), "learned_records"
    )
    macros.add(
        "NlearnedGainP", gain_p, dec(float(gain_p), 3), "learned_records"
    )
    macros.add(
        "NlearnedHotGain",
        float(gain[hot].mean()),
        dec(float(gain[hot].mean()), 1),
        "learned_records",
    )
    macros.add(
        "NlearnedRestGain",
        float(gain[~hot].mean()),
        dec(float(gain[~hot].mean()), 1),
        "learned_records",
    )
    macros.add("NlearnedIterMean", lrs["converging_trial_iterations"]["mean"],
               dec(lrs["converging_trial_iterations"]["mean"], 2), "learned")
    macros.add("NprodIterMean", psf["converging_trial_iterations"]["mean"],
               dec(psf["converging_trial_iterations"]["mean"], 2), "production")
    macros.add("NlearnedNonmono", lrs["contraction"]["non_monotonic_fraction"],
               pct(lrs["contraction"]["non_monotonic_fraction"]), "learned")
    macros.add("NprodNonmono", psf["contraction"]["non_monotonic_fraction"],
               pct(psf["contraction"]["non_monotonic_fraction"]), "production")
    macros.add("NlearnedQ", lrs["contraction"]["q_ratio"]["geometric_mean"],
               dec(lrs["contraction"]["q_ratio"]["geometric_mean"], 3), "learned")
    macros.add("NprodQ", psf["contraction"]["q_ratio"]["geometric_mean"],
               dec(psf["contraction"]["q_ratio"]["geometric_mean"], 3), "production")
    macros.add("NlearnedFirstResidual",
               lrs["contraction"]["first_iteration_residual"]["p50"],
               sci(lrs["contraction"]["first_iteration_residual"]["p50"]), "learned")
    macros.add("NprodFirstResidual",
               psf["contraction"]["first_iteration_residual"]["p50"],
               sci(psf["contraction"]["first_iteration_residual"]["p50"]), "production")
    learned_solver_failures = int(lrs["failure_count"])
    learned_reconstruction_failures = int(
        learned_reconstruction["failure_count"]
    )
    learned_total_failures = (
        learned_solver_failures + learned_reconstruction_failures
    )
    if learned_total_failures != learned["star_count"] - lrs["converged_count"]:
        raise SystemExit("learned total failure count does not close")
    macros.add("NlearnedFailures", learned_total_failures,
               integer(learned_total_failures), "learned")
    macros.add("NlearnedSolverFailures", learned_solver_failures,
               integer(learned_solver_failures), "learned")
    macros.add("NlearnedReconstructionFailures",
               learned_reconstruction_failures,
               integer(learned_reconstruction_failures), "learned")
    macros.add("NlearnedConverged", lrs["converged_count"],
               integer(lrs["converged_count"]), "learned")
    macros.add("NprodFailures", psf["failure_count"], integer(psf["failure_count"]), "production")
    macros.add("NiterReduction",
               1.0 - lrs["converging_trial_iterations"]["mean"]
               / psf["converging_trial_iterations"]["mean"],
               pct(1.0 - lrs["converging_trial_iterations"]["mean"]
                   / psf["converging_trial_iterations"]["mean"], 0), "learned")
    macros.add("NjitterIterMean", pj["converging_trial_iterations"]["mean"],
               dec(pj["converging_trial_iterations"]["mean"], 2), "jitter")
    macros.add("NjitterConverged", pj["converged_fraction"],
               pct(pj["converged_fraction"]), "jitter")

    # two shared fields, learned against production
    lprof = learned["profile_errors"]["learned_reduced_state"]
    pprof = production["profile_errors"]["production_six_field"]
    macros.add("NlearnedTempPfive", lprof["temperature_relative_p95"],
               sci(lprof["temperature_relative_p95"]), "learned")
    macros.add("NprodTempPfive", pprof["temperature_relative_p95"],
               sci(pprof["temperature_relative_p95"]), "production")
    macros.add("NlearnedMassPfive", lprof["column_mass_dex_p95"],
               sci(lprof["column_mass_dex_p95"]), "learned")
    macros.add("NprodMassPfive", pprof["column_mass_dex_p95"],
               sci(pprof["column_mass_dex_p95"]), "production")

    # ---- same-star analytic formula comparison -------------------------
    learned_cmp = analytic_comparison["learned_two_field"]
    analytic_cmp = analytic_comparison["analytic_parity"]
    paired_cmp = analytic_comparison["paired_solver"]
    for label, observed, expected in (
        (
            "learned temperature p95",
            learned_cmp["profile_errors"]["temperature_relative_p95"],
            lprof["temperature_relative_p95"],
        ),
        (
            "learned column-mass p95",
            learned_cmp["profile_errors"]["column_mass_dex_p95"],
            lprof["column_mass_dex_p95"],
        ),
        (
            "learned convergence count",
            learned_cmp["solver"]["converged_count"],
            lrs["converged_count"],
        ),
        (
            "learned mean iterations",
            learned_cmp["solver"]["mean_iterations_converged"],
            lrs["converging_trial_iterations"]["mean"],
        ),
    ):
        if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-12):
            raise SystemExit(
                f"analytic comparison {label} disagrees with the manuscript source: "
                f"{observed} != {expected}"
            )

    learned_runtime_floats = int(learned_cmp["runtime_float_count"])
    analytic_floats = int(analytic_cmp["logical_float_count"])
    analytic_serialized_floats = int(analytic_cmp["serialized_float_entry_count"])
    analytic_integer_entries = int(
        analytic_cmp["serialized_structural_integer_entry_count"]
    )
    learned_solver = learned_cmp["solver"]
    analytic_solver = analytic_cmp["solver"]
    learned_profile = learned_cmp["profile_errors"]
    analytic_profile = analytic_cmp["profile_errors"]
    rows = [
        " & ".join([
            r"Learned two-field + physics",
            r"$(m,T)$",
            thousands(learned_runtime_floats),
            sci(learned_profile["temperature_relative_p95"]),
            sci(learned_profile["column_mass_dex_p95"]),
            (f"{integer(learned_solver['converged_count'])}/"
             f"{integer(learned_solver['star_count'])}"),
            dec(learned_solver["mean_iterations_converged"], 2),
            dec(learned_solver["median_iterations_converged"], 1),
        ]) + r" \\",
        " & ".join([
            r"Fitted analytic formula",
            r"$(\delta_T,\log\kappa_{\rm R})\rightarrow m$",
            thousands(analytic_floats),
            sci(analytic_profile["temperature_relative_p95"]),
            sci(analytic_profile["column_mass_dex_p95"]),
            (f"{integer(analytic_solver['converged_count'])}/"
             f"{integer(analytic_solver['star_count'])}"),
            dec(analytic_solver["mean_iterations_converged"], 2),
            dec(analytic_solver["median_iterations_converged"], 1),
        ]) + r" \\",
    ]
    tables["tab_analytic_comparison.tex"] = table_env(
        label="tab:analytic",
        caption=(
            "Learned and analytic two-field initializers evaluated on the same "
            "60 development stars. Profile errors are pooled over all "
            r"$\NevalStars\times\NgridLayers$ depth points. Iteration "
            "statistics include converged stars only; unsuccessful "
            "reconstruction and timeouts are counted as non-convergences. "
            "The parameter counts include independent floating-point "
            "coefficients and constants; integer basis indices are not counted. "
            "Solver calculations for the two methods were "
            "performed separately, and no end-to-end timing, spectral "
            "comparison, or independent-sample test is available for the "
            "analytic method."),
        colspec="llrrrrrr",
        header=(
            r"Initializer & Fitted profiles & Float parameters & $T$ $p_{95}$ "
            r"& $\log m$ $p_{95}$ & Converged & Mean iter. & Median \\"
        ),
        rows=rows,
        wide=True,
        colsep=4.0,
        size="small",
    )

    macros.add("NanalyticConstants", analytic_floats,
               thousands(analytic_floats), "analytic_comparison")
    macros.add("NanalyticSerializedFloats", analytic_serialized_floats,
               thousands(analytic_serialized_floats), "analytic_comparison")
    macros.add("NanalyticIntegerEntries", analytic_integer_entries,
               thousands(analytic_integer_entries), "analytic_comparison")
    macros.add("NlearnedRuntimeFloats", learned_runtime_floats,
               thousands(learned_runtime_floats), "analytic_comparison")
    macros.add("NanalyticCompression",
               learned_runtime_floats / analytic_serialized_floats,
               integer(learned_runtime_floats / analytic_serialized_floats),
               "analytic_comparison")
    macros.add("NanalyticTrainStars", analytic_cmp["training_rows"],
               thousands(analytic_cmp["training_rows"]), "analytic_comparison")
    macros.add("NanalyticValStars", analytic_cmp["validation_rows"],
               thousands(analytic_cmp["validation_rows"]), "analytic_comparison")
    macros.add("NanalyticTempPfive",
               analytic_profile["temperature_relative_p95"],
               sci(analytic_profile["temperature_relative_p95"]),
               "analytic_comparison")
    macros.add("NanalyticMassPfive",
               analytic_profile["column_mass_dex_p95"],
               sci(analytic_profile["column_mass_dex_p95"]),
               "analytic_comparison")
    macros.add("NanalyticTempPenalty",
               analytic_profile["temperature_relative_p95"]
               / learned_profile["temperature_relative_p95"],
               dec(
                   analytic_profile["temperature_relative_p95"]
                   / learned_profile["temperature_relative_p95"],
                   1,
               ),
               "analytic_comparison")
    macros.add("NanalyticMassPenalty",
               analytic_profile["column_mass_dex_p95"]
               / learned_profile["column_mass_dex_p95"],
               dec(
                   analytic_profile["column_mass_dex_p95"]
                   / learned_profile["column_mass_dex_p95"],
                   1,
               ),
               "analytic_comparison")
    macros.add("NanalyticConverged", analytic_solver["converged_count"],
               integer(analytic_solver["converged_count"]), "analytic_comparison")
    macros.add("NanalyticFailures", analytic_solver["failure_count"],
               integer(analytic_solver["failure_count"]), "analytic_comparison")
    macros.add("NanalyticTimeouts", analytic_solver["timeout_count"],
               integer(analytic_solver["timeout_count"]), "analytic_comparison")
    macros.add("NanalyticTimeoutSeconds",
               analytic_solver["per_star_timeout_seconds"],
               integer(analytic_solver["per_star_timeout_seconds"]),
               "analytic_comparison")
    macros.add("NanalyticIterationLimit", analytic_solver["iterations_per_trial"],
               integer(analytic_solver["iterations_per_trial"]),
               "analytic_comparison")
    macros.add("NanalyticIterMean", analytic_solver["mean_iterations_converged"],
               dec(analytic_solver["mean_iterations_converged"], 2),
               "analytic_comparison")
    macros.add("NanalyticIterMedian", analytic_solver["median_iterations_converged"],
               dec(analytic_solver["median_iterations_converged"], 1),
               "analytic_comparison")
    macros.add("NanalyticCommon", paired_cmp["common_converged_count"],
               integer(paired_cmp["common_converged_count"]), "analytic_comparison")
    macros.add("NanalyticLearnedOnly",
               paired_cmp["learned_only_converged_count"],
               integer(paired_cmp["learned_only_converged_count"]),
               "analytic_comparison")
    macros.add("NanalyticOnly", paired_cmp["analytic_only_converged_count"],
               integer(paired_cmp["analytic_only_converged_count"]),
               "analytic_comparison")
    macros.add(
        "NanalyticLearnedFaster",
        paired_cmp["learned_fewer_iterations_count"],
        integer(paired_cmp["learned_fewer_iterations_count"]),
        "analytic_comparison",
    )
    macros.add(
        "NanalyticFaster",
        paired_cmp["analytic_fewer_iterations_count"],
        integer(paired_cmp["analytic_fewer_iterations_count"]),
        "analytic_comparison",
    )
    macros.add("NanalyticTied", paired_cmp["tied_count"],
               integer(paired_cmp["tied_count"]), "analytic_comparison")
    macros.add(
        "NanalyticIterDifference",
        paired_cmp["mean_analytic_minus_learned_iterations"],
        dec(paired_cmp["mean_analytic_minus_learned_iterations"], 2),
        "analytic_comparison",
    )

    # ---- zero-fit first-principles v4r6 diagnostic ----------------------
    vfour_seed = art("vfour_seed_audit")
    vfour_historical = art("vfour_decoupled_fifteen")
    vfour_decoupled = art("vfour_decoupled_policy")
    vfour_grey = art("vfour_grey_policy")
    vfour_convective = art("vfour_convective_policy")
    vfour_score = art("vfour_matched_score")
    vfour_offline = art("vfour_offline")

    if vfour_seed.get("decision") != "PASS_STRUCTURAL":
        raise SystemExit("v4r6 seed did not pass its frozen structural audit")
    if vfour_seed["initializer_provenance"]["fitted_parameter_count"] != 0:
        raise SystemExit("v4r6 manuscript arm is not zero-fit")
    if vfour_offline.get("decision") != "FAIL_STOP":
        raise SystemExit("v4r6 offline decision is no longer FAIL_STOP")
    if vfour_historical.get("decision") != "FAIL_STOP_DEVELOPMENT":
        raise SystemExit("v4r6 frozen 15-iteration decision changed")
    if vfour_score.get("decision") != "STOP_POLICY60_MATCHED_DEVELOPMENT":
        raise SystemExit("v4r6 matched policy60 decision changed")
    if vfour_score.get("status") != "matched_development_characterization_complete":
        raise SystemExit("v4r6 matched policy60 study is not complete")
    if vfour_score.get("authorizes_fresh_open_execution") is not False:
        raise SystemExit("v4r6 matched result unexpectedly authorizes fresh-open")
    checks = vfour_score["gate"]["checks"]
    failed_checks = {name for name, passed in checks.items() if not passed}
    if failed_checks != {"cool_net_gain_vs_grey"}:
        raise SystemExit(
            f"v4r6 matched policy60 failed checks changed: {sorted(failed_checks)}"
        )

    vfour_arms = {
        "Decoupled": ("vfour_decoupled_policy", vfour_decoupled),
        "Grey": ("vfour_grey_policy", vfour_grey),
        "Coupled convective": ("vfour_convective_policy", vfour_convective),
    }
    runtime_keys = (
        "policy",
        "trials",
        "iterations",
        "per_star_timeout_seconds",
        "sample",
        "sample_sha256",
        "source_manifest",
        "source_manifest_sha256",
        "source_git_head",
        "source_git_diff_sha256",
        "hostname",
        "operating_system",
        "python",
        "numpy",
        "numba",
        "environment",
    )
    runtime_signatures = []
    for title, (source_name, payload) in vfour_arms.items():
        if payload.get("status") != "policy60_matched_development_only":
            raise SystemExit(f"v4r6 {title} is not a matched development arm")
        runtime = payload["runtime_signature"]
        runtime_signatures.append({key: runtime.get(key) for key in runtime_keys})
        if runtime["source_manifest_sha256"] != art.hashes["vfour_policy_source_manifest"]:
            raise SystemExit(f"v4r6 {title} source-manifest hash mismatch")
        score_arm = vfour_score["arm_artifacts"][payload["arm"]]
        if score_arm["sha256"] != art.hashes[source_name]:
            raise SystemExit(f"v4r6 {title} result hash disagrees with score")
    if not all(signature == runtime_signatures[0] for signature in runtime_signatures):
        raise SystemExit("v4r6 policy60 runtime signatures are not matched")
    if (
        vfour_score["source_manifest_sha256"]
        != art.hashes["vfour_policy_source_manifest"]
    ):
        raise SystemExit("v4r6 score source-manifest hash mismatch")

    def vfour_summary(payload: dict) -> dict[str, float | int]:
        records = payload["records"]
        converged = [
            row for row in records
            if bool(row.get("converged"))
            and row.get("iterations_completed") is not None
        ]
        iterations = [int(row["iterations_completed"]) for row in converged]
        cool = [
            row for row in records
            if float(row["effective_temperature"]) < 7500.0
        ]
        hot = [
            row for row in records
            if float(row["effective_temperature"]) >= 7500.0
        ]
        return {
            "stars": len(records),
            "converged": len(converged),
            "cool_stars": len(cool),
            "cool_converged": sum(bool(row.get("converged")) for row in cool),
            "hot_stars": len(hot),
            "hot_converged": sum(bool(row.get("converged")) for row in hot),
            "timeouts": sum(
                str(row.get("solver_outcome")) == "timeout" for row in records
            ),
            "errors": sum(
                str(row.get("solver_outcome")) == "error" for row in records
            ),
            "mean": float(sum(iterations) / len(iterations)),
            "median": float(np.median(iterations)),
            "max": max(iterations),
            "above_fifteen": sum(value > 15 for value in iterations),
            "late_first": min(
                (value for value in iterations if value > 15),
                default=0,
            ),
        }

    vfour_stats = {
        title: vfour_summary(payload)
        for title, (_, payload) in vfour_arms.items()
    }
    for title, stats in vfour_stats.items():
        if stats["stars"] != 60 or stats["cool_stars"] != 27 or stats["hot_stars"] != 33:
            raise SystemExit(f"v4r6 {title} is not the frozen development-60 split")

    rows = []
    table_titles = {
        "Decoupled": "Grey--convective",
        "Grey": "Grey",
        "Coupled convective": "Coupled convective",
    }
    for title in ("Decoupled", "Grey", "Coupled convective"):
        stats = vfour_stats[title]
        rows.append(" & ".join([
            table_titles[title],
            integer(vfour_decoupled["runtime_signature"]["iterations"]),
            f"{integer(stats['converged'])}/{integer(stats['stars'])}",
            f"{integer(stats['cool_converged'])}/{integer(stats['cool_stars'])}",
            f"{integer(stats['hot_converged'])}/{integer(stats['hot_stars'])}",
            f"{integer(stats['timeouts'])}/{integer(stats['errors'])}",
            dec(stats["mean"], 2),
            dec(stats["median"], 1),
            integer(stats["max"]),
        ]) + r" \\")
    tables["tab_v4r6.tex"] = table_env(
        label="tab:vfour",
        caption=(
            "Convergence of three physical initializations on the same 60 "
            "development stars. Each calculation uses at most "
            "60 iterations and a 900-s limit per star. Means and medians include "
            "converged atmospheres only. The grey--convective construction "
            "converges for three more cool stars than the grey construction, "
            "while the corresponding gain over the full sample is two stars."),
        colspec="lrrrrrrrr",
        header=(
            r"Initial state & Maximum iter. & All & Cool & Hot & Timeout/error & "
            r"Mean iter. & Median & Max \\"
        ),
        rows=rows,
        wide=True,
        colsep=4.0,
        size="small",
    )

    vfour_decoupled_stats = vfour_stats["Decoupled"]
    vfour_grey_stats = vfour_stats["Grey"]
    vfour_convective_stats = vfour_stats["Coupled convective"]
    vfour_gate = vfour_offline["offline_gate"]
    matched_gate = vfour_score["gate"]
    macros.add(
        "NvfourSeedStars",
        vfour_seed["star_count"],
        integer(vfour_seed["star_count"]),
        "vfour_seed_audit",
    )
    macros.add(
        "NvfourFittedParameters",
        vfour_seed["initializer_provenance"]["fitted_parameter_count"],
        integer(vfour_seed["initializer_provenance"]["fitted_parameter_count"]),
        "vfour_seed_audit",
    )
    macros.add(
        "NvfourHistoricalIterationLimit",
        vfour_historical["iterations_per_trial"],
        integer(vfour_historical["iterations_per_trial"]),
        "vfour_decoupled_fifteen",
    )
    macros.add(
        "NvfourHistoricalConverged",
        vfour_historical["converged_count"],
        integer(vfour_historical["converged_count"]),
        "vfour_decoupled_fifteen",
    )
    macros.add(
        "NvfourPolicyIterationLimit",
        vfour_decoupled["runtime_signature"]["iterations"],
        integer(vfour_decoupled["runtime_signature"]["iterations"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourPolicyTimeoutSeconds",
        vfour_decoupled["runtime_signature"]["per_star_timeout_seconds"],
        integer(vfour_decoupled["runtime_signature"]["per_star_timeout_seconds"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourMatchedStars",
        vfour_decoupled_stats["stars"],
        integer(vfour_decoupled_stats["stars"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourCoolStars",
        vfour_decoupled_stats["cool_stars"],
        integer(vfour_decoupled_stats["cool_stars"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourHotStars",
        vfour_decoupled_stats["hot_stars"],
        integer(vfour_decoupled_stats["hot_stars"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourDecoupledConverged",
        vfour_decoupled_stats["converged"],
        integer(vfour_decoupled_stats["converged"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourGreyConverged",
        vfour_grey_stats["converged"],
        integer(vfour_grey_stats["converged"]),
        "vfour_grey_policy",
    )
    macros.add(
        "NvfourConvectiveConverged",
        vfour_convective_stats["converged"],
        integer(vfour_convective_stats["converged"]),
        "vfour_convective_policy",
    )
    macros.add(
        "NvfourDecoupledCoolConverged",
        vfour_decoupled_stats["cool_converged"],
        integer(vfour_decoupled_stats["cool_converged"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourGreyCoolConverged",
        vfour_grey_stats["cool_converged"],
        integer(vfour_grey_stats["cool_converged"]),
        "vfour_grey_policy",
    )
    macros.add(
        "NvfourConvectiveCoolConverged",
        vfour_convective_stats["cool_converged"],
        integer(vfour_convective_stats["cool_converged"]),
        "vfour_convective_policy",
    )
    macros.add(
        "NvfourDecoupledHotConverged",
        vfour_decoupled_stats["hot_converged"],
        integer(vfour_decoupled_stats["hot_converged"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourGreyHotConverged",
        vfour_grey_stats["hot_converged"],
        integer(vfour_grey_stats["hot_converged"]),
        "vfour_grey_policy",
    )
    macros.add(
        "NvfourConvectiveHotConverged",
        vfour_convective_stats["hot_converged"],
        integer(vfour_convective_stats["hot_converged"]),
        "vfour_convective_policy",
    )
    macros.add(
        "NvfourDecoupledIterMean",
        vfour_decoupled_stats["mean"],
        dec(vfour_decoupled_stats["mean"], 2),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourGreyIterMean",
        vfour_grey_stats["mean"],
        dec(vfour_grey_stats["mean"], 2),
        "vfour_grey_policy",
    )
    macros.add(
        "NvfourConvectiveIterMean",
        vfour_convective_stats["mean"],
        dec(vfour_convective_stats["mean"], 2),
        "vfour_convective_policy",
    )
    macros.add(
        "NvfourDecoupledIterMedian",
        vfour_decoupled_stats["median"],
        dec(vfour_decoupled_stats["median"], 1),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourDecoupledIterMax",
        vfour_decoupled_stats["max"],
        integer(vfour_decoupled_stats["max"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourLateFirst",
        vfour_decoupled_stats["late_first"],
        integer(vfour_decoupled_stats["late_first"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourRecoveredLate",
        vfour_decoupled_stats["above_fifteen"],
        integer(vfour_decoupled_stats["above_fifteen"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourTimeouts",
        vfour_decoupled_stats["timeouts"],
        integer(vfour_decoupled_stats["timeouts"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourErrors",
        vfour_decoupled_stats["errors"],
        integer(vfour_decoupled_stats["errors"]),
        "vfour_decoupled_policy",
    )
    macros.add(
        "NvfourGainGreyAll",
        matched_gate["paired_vs_grey"]["all"]["net_gain"],
        integer(matched_gate["paired_vs_grey"]["all"]["net_gain"]),
        "vfour_matched_score",
    )
    macros.add(
        "NvfourGainGreyCool",
        matched_gate["paired_vs_grey"]["cool"]["net_gain"],
        integer(matched_gate["paired_vs_grey"]["cool"]["net_gain"]),
        "vfour_matched_score",
    )
    macros.add(
        "NvfourGainGreyHot",
        matched_gate["paired_vs_grey"]["hot"]["net_gain"],
        integer(matched_gate["paired_vs_grey"]["hot"]["net_gain"]),
        "vfour_matched_score",
    )
    macros.add(
        "NvfourGainConvectiveAll",
        matched_gate["paired_vs_convective"]["all"]["net_gain"],
        integer(matched_gate["paired_vs_convective"]["all"]["net_gain"]),
        "vfour_matched_score",
    )
    macros.add(
        "NvfourGainConvectiveCool",
        matched_gate["paired_vs_convective"]["cool"]["net_gain"],
        integer(matched_gate["paired_vs_convective"]["cool"]["net_gain"]),
        "vfour_matched_score",
    )
    macros.add(
        "NvfourCoolGainRequirement",
        matched_gate["thresholds"]["cool_net_gain_vs_grey_min"],
        integer(matched_gate["thresholds"]["cool_net_gain_vs_grey_min"]),
        "vfour_matched_score",
    )
    macros.add(
        "NvfourCoolMassPfive",
        vfour_gate["cool_mass_observed_p95_dex"],
        dec(vfour_gate["cool_mass_observed_p95_dex"], 3),
        "vfour_offline",
    )
    macros.add(
        "NvfourMiddleMassPfive",
        vfour_gate["middle_mass_observed_p95_dex"],
        dec(vfour_gate["middle_mass_observed_p95_dex"], 3),
        "vfour_offline",
    )
    macros.add(
        "NvfourMassGate",
        vfour_gate["bridge_mass_p95_max_dex"],
        dec(vfour_gate["bridge_mass_p95_max_dex"], 2),
        "vfour_offline",
    )

    # ---- complete hydrostatic grey--convective evaluation ---------------
    grey_campaign = art("grey_campaign")
    grey_source = art("grey_source_manifest")
    grey_output = art("grey_output_manifest")
    grey_replay = art("grey_dev_replay")
    if grey_campaign.get("campaign") != "paper_grey_convective_20260829":
        raise SystemExit("wrong emulator-independent initializer campaign")
    if grey_campaign.get("development_replay_matches") is not True:
        raise SystemExit("development replay is not certified")
    if grey_campaign.get("posthoc200_is_new_blind_test") is not False:
        raise SystemExit("opened 200-star sample is mislabelled as a blind test")
    if grey_replay.get("matches") is not True:
        raise SystemExit("frozen 54/60 development result was not reproduced")
    if grey_campaign["source_manifest_sha256"] != art.hashes["grey_source_manifest"]:
        raise SystemExit("physical campaign source-manifest hash mismatch")
    if grey_output["source_manifest_sha256"] != art.hashes["grey_source_manifest"]:
        raise SystemExit("physical output inventory source hash mismatch")
    policy = grey_source["policy"]
    if (
        policy["trials"] != 1
        or policy["iterations"] != 60
        or policy["per_star_timeout_seconds"] != 900.0
        or policy["mass_reintegrated_after_convection"] is not False
        or policy["requires_neural_checkpoint_at_runtime"] is not False
    ):
        raise SystemExit("physical initializer policy changed")

    grey_payloads = {
        "Development": {
            "source": "grey_dev_summary",
            "seed_source": "grey_dev_seed",
            "solver_source": "grey_dev_solver",
            "profile_source": "grey_dev_profiles",
            "spectral_source": "grey_dev_spectra",
        },
        "Post-hoc 200": {
            "source": "grey_post_summary",
            "seed_source": "grey_post_seed",
            "solver_source": "grey_post_solver",
            "profile_source": "grey_post_profiles",
            "spectral_source": "grey_post_spectra",
        },
    }
    physical_table_rows = []
    for title, names in grey_payloads.items():
        summary = art(names["source"])
        seed = art(names["seed_source"])
        solver = art(names["solver_source"])
        profiles = art(names["profile_source"])
        spectra = art(names["spectral_source"])
        expected = 60 if title == "Development" else 200
        records = solver["records"]
        reference_source = (
            "production_records"
            if title == "Development"
            else "blind_prod_records"
        )
        reference_records = load_jsonl_records(art, reference_source)
        common_rows = [
            (row, reference_records[row["slug"]])
            for row in records
            if bool(row["converged"])
            and row["slug"] in reference_records
            and bool(reference_records[row["slug"]]["converged"])
        ]
        common_physical_iterations = np.asarray(
            [row["iterations_completed"] for row, _ in common_rows],
            dtype=float,
        )
        common_reference_iterations = np.asarray(
            [
                reference["converging_trial_iterations"]
                for _, reference in common_rows
            ],
            dtype=float,
        )
        indices = [int(row["corpus_index"]) for row in records]
        if (
            summary["star_count"] != expected
            or seed["star_count"] != expected
            or solver["star_count"] != expected
            or len(records) != expected
            or len(set(indices)) != expected
        ):
            raise SystemExit(f"{title} physical sample coverage changed")
        if (
            seed["finite_seed_count"] != expected
            or seed["positive_seed_count"] != expected
        ):
            raise SystemExit(f"{title} contains an invalid physical seed")
        if profiles["final_converged_star_count"] != solver["converged_count"]:
            raise SystemExit(f"{title} profile count disagrees with convergence")
        if (
            spectra["gated_star_count"] + spectra["excluded_star_count"]
            != expected
        ):
            raise SystemExit(f"{title} spectral inclusion accounting is incomplete")
        for row in records:
            diagnostics = row.get("diagnostics")
            iterations = row.get("iterations_completed")
            if diagnostics is not None and iterations is not None:
                timings = diagnostics.get("iteration_timings", [])
                if len(timings) != int(iterations):
                    raise SystemExit(
                        f"{title} has an incomplete iteration trace for "
                        f"{row['corpus_index']}"
                    )
        iterations = summary["iterations_among_converged"]
        physical_table_rows.append(
            " & ".join(
                [
                    title,
                    f"{integer(summary['converged_count'])}/"
                    f"{integer(summary['star_count'])}",
                    (
                        f"{integer(summary['not_converged_count'])}/"
                        f"{integer(summary['timeout_count'])}/"
                        f"{integer(summary['error_count'])}"
                    ),
                    dec(iterations["mean"], 2),
                    dec(iterations["median"], 1),
                    integer(iterations["maximum"]),
                    integer(iterations["more_than_15_count"]),
                ]
            )
            + r" \\"
        )

        prefix = "NphysicalDev" if title == "Development" else "NphysicalPost"
        macro_values = (
            (
                "Stars",
                summary["star_count"],
                integer(summary["star_count"]),
                names["source"],
            ),
            (
                "FiniteSeeds",
                seed["finite_seed_count"],
                integer(seed["finite_seed_count"]),
                names["seed_source"],
            ),
            (
                "Converged",
                summary["converged_count"],
                integer(summary["converged_count"]),
                names["source"],
            ),
            (
                "NotConverged",
                summary["not_converged_count"],
                integer(summary["not_converged_count"]),
                names["source"],
            ),
            (
                "Timeouts",
                summary["timeout_count"],
                integer(summary["timeout_count"]),
                names["source"],
            ),
            (
                "Errors",
                summary["error_count"],
                integer(summary["error_count"]),
                names["source"],
            ),
            (
                "IterMean",
                iterations["mean"],
                dec(iterations["mean"], 2),
                names["source"],
            ),
            (
                "IterMedian",
                iterations["median"],
                dec(iterations["median"], 1),
                names["source"],
            ),
            (
                "IterMax",
                iterations["maximum"],
                integer(iterations["maximum"]),
                names["source"],
            ),
            (
                "AboveFifteen",
                iterations["more_than_15_count"],
                integer(iterations["more_than_15_count"]),
                names["source"],
            ),
            (
                "Spectra",
                spectra["gated_star_count"],
                integer(spectra["gated_star_count"]),
                names["spectral_source"],
            ),
            (
                "SpectraExcluded",
                spectra["excluded_star_count"],
                integer(spectra["excluded_star_count"]),
                names["spectral_source"],
            ),
            (
                "SpectraOver",
                spectra["normalized_flux"]["stars_over_bar"],
                integer(spectra["normalized_flux"]["stars_over_bar"]),
                names["spectral_source"],
            ),
            (
                "SpectraMedian",
                spectra["normalized_flux"]["median_over_stars"],
                sci(spectra["normalized_flux"]["median_over_stars"]),
                names["spectral_source"],
            ),
            (
                "SpectraPfive",
                spectra["normalized_flux"]["p95_over_stars"],
                sci(spectra["normalized_flux"]["p95_over_stars"]),
                names["spectral_source"],
            ),
            (
                "SpectraMax",
                spectra["normalized_flux"]["max_over_stars"],
                sci(spectra["normalized_flux"]["max_over_stars"]),
                names["spectral_source"],
            ),
            (
                "SeedMassMedian",
                profiles["seed"]["column_mass"]["median"],
                dec(profiles["seed"]["column_mass"]["median"], 3),
                names["profile_source"],
            ),
            (
                "SeedMassPfive",
                profiles["seed"]["column_mass"]["p95"],
                dec(profiles["seed"]["column_mass"]["p95"], 3),
                names["profile_source"],
            ),
            (
                "SeedTempMedian",
                profiles["seed"]["temperature"]["median"],
                sci(profiles["seed"]["temperature"]["median"]),
                names["profile_source"],
            ),
            (
                "SeedTempPfive",
                profiles["seed"]["temperature"]["p95"],
                sci(profiles["seed"]["temperature"]["p95"]),
                names["profile_source"],
            ),
        )
        for suffix, raw, text, source_name in macro_values:
            macros.add(prefix + suffix, raw, text, source_name)
        if title == "Post-hoc 200":
            mid_temperature_low = 7000.0
            mid_temperature_high = 9000.0
            failed_rows = [row for row in records if not bool(row["converged"])]
            mid_temperature_rows = [
                row
                for row in records
                if mid_temperature_low
                <= float(row["effective_temperature"])
                < mid_temperature_high
            ]
            mid_temperature_failures = [
                row for row in mid_temperature_rows if not bool(row["converged"])
            ]
            if (
                len(failed_rows) != expected - summary["converged_count"]
                or len(mid_temperature_failures) != 16
                or len(mid_temperature_rows) != 62
            ):
                raise SystemExit(
                    "post-hoc physical failure concentration has changed"
                )
            for suffix, raw, text in (
                ("FailureCount", len(failed_rows), integer(len(failed_rows))),
                (
                    "MidTempFailures",
                    len(mid_temperature_failures),
                    integer(len(mid_temperature_failures)),
                ),
                (
                    "MidTempStars",
                    len(mid_temperature_rows),
                    integer(len(mid_temperature_rows)),
                ),
                (
                    "MidTempLow",
                    mid_temperature_low,
                    integer(mid_temperature_low),
                ),
                (
                    "MidTempHigh",
                    mid_temperature_high,
                    integer(mid_temperature_high),
                ),
            ):
                macros.add(
                    "NphysicalPost" + suffix,
                    raw,
                    text,
                    names["solver_source"],
                )
        for suffix, raw, text in (
            ("Common", len(common_rows), integer(len(common_rows))),
            (
                "CommonPhysicalMean",
                float(common_physical_iterations.mean()),
                dec(float(common_physical_iterations.mean()), 2),
            ),
            (
                "CommonReferenceMean",
                float(common_reference_iterations.mean()),
                dec(float(common_reference_iterations.mean()), 2),
            ),
            (
                "CommonDifferenceMedian",
                float(
                    np.median(
                        common_physical_iterations - common_reference_iterations
                    )
                ),
                dec(
                    float(
                        np.median(
                            common_physical_iterations
                            - common_reference_iterations
                        )
                    ),
                    1,
                ),
            ),
        ):
            macros.add(prefix + suffix, raw, text, names["solver_source"])

    tables["tab_physical_initializer.tex"] = table_env(
        label="tab:physical_initializer",
        caption=(
            "Hydrostatic grey--convective initialization on the development "
            "sample and, post hoc, on the same 200-star sample used for the "
            "learned test. Every star uses at most 60 iterations and 900 s. "
            "Iteration statistics are conditional on convergence; the "
            "convergence and failure columns retain the full sample. Spectral "
            "conditioning is reported separately in Table~\\ref{tab:spectral}."
        ),
        colspec="lrrrrrr",
        header=(
            r"Sample & Converged & Non-conv./timeout/error & Mean iter. & "
            r"Median & Max & $>15$ \\"
        ),
        rows=physical_table_rows,
        wide=True,
        colsep=3.5,
        size="small",
    )
    macros.add(
        "NphysicalIterationLimit",
        policy["iterations"],
        integer(policy["iterations"]),
        "grey_source_manifest",
    )
    macros.add(
        "NphysicalTimeoutSeconds",
        policy["per_star_timeout_seconds"],
        integer(policy["per_star_timeout_seconds"]),
        "grey_source_manifest",
    )
    residual = art("grey_dev_summary")["development_residual_iter100"]
    for suffix, key in (
        ("ResidualStars", "star_count"),
        ("ResidualConverged", "converged_count"),
        ("ResidualTimeouts", "timeout_count"),
        ("ResidualIterations", "iterations"),
        ("ResidualTimeoutSeconds", "per_star_timeout_seconds"),
    ):
        macros.add(
            "Nphysical" + suffix,
            residual[key],
            integer(residual[key]),
            "grey_dev_summary",
        )
    macros.add(
        "NphysicalOutputFiles",
        grey_output["result_file_count"] + grey_output["run_file_count"],
        integer(grey_output["result_file_count"] + grey_output["run_file_count"]),
        "grey_output_manifest",
    )

    # ---- monotonicity ablation ------------------------------------------
    mono = training["arms"]["monotone"]["held_out"]
    direct = training["arms"]["direct"]["held_out"]
    rows = []
    for title, arm in [(r"Monotone, Eq.~(\ref{eq:monotone})", mono),
                       (r"Direct, \NgridLayers\ outputs", direct)]:
        rows.append(" & ".join([
            title,
            sci(arm["temperature_relative_error"]["p95_overall"]),
            sci(arm["column_mass_dex_error"]["p95_overall"]),
            rf"{arm['monotonicity_violations']}/{arm['star_count']}",
        ]) + r" \\")
    tables["tab_monotone.tex"] = table_env(
        label="tab:monotone",
        caption=(
            "Comparison of monotonic and direct column-mass parameterizations. "
            r"Both models were trained for \NnetEpochs\ epochs with the same "
            "data split and random seed. The direct parameterization has smaller "
            "profile errors but produces non-monotonic column-mass profiles, "
            "which are not physically admissible."),
        colspec="lrrr",
        header=(r"Parameterisation & $T$ $p_{95}$ & $\log m$ $p_{95}$ & Rejected \\"),
        rows=rows,
        colsep=4.0,
        size="small",
    )
    macros.add("NmonoTempPfive", mono["temperature_relative_error"]["p95_overall"],
               sci(mono["temperature_relative_error"]["p95_overall"]), "training")
    macros.add("NdirectTempPfive", direct["temperature_relative_error"]["p95_overall"],
               sci(direct["temperature_relative_error"]["p95_overall"]), "training")
    macros.add("NmonoMassPfive", mono["column_mass_dex_error"]["p95_overall"],
               sci(mono["column_mass_dex_error"]["p95_overall"]), "training")
    macros.add("NdirectMassPfive", direct["column_mass_dex_error"]["p95_overall"],
               sci(direct["column_mass_dex_error"]["p95_overall"]), "training")
    macros.add("NmonoViolations", mono["monotonicity_violations"],
               integer(mono["monotonicity_violations"]), "training")
    macros.add("NdirectViolations", direct["monotonicity_violations"],
               integer(direct["monotonicity_violations"]), "training")

    # ---- Table 8: spectral comparisons ----------------------------------
    gate_rows = [
        (r"Converged $(m,T)$ vs.\ reference atmosphere", art("gate_truth")),
        (r"Learned two-field vs.\ six-field reference", art("gate_learned")),
        (
            r"Grey--convective vs.\ reference, development",
            art("grey_dev_spectra"),
        ),
        (
            r"Grey--convective vs.\ reference, post-hoc 200",
            art("grey_post_spectra"),
        ),
        (r"Six-field reference vs.\ alternate converged start", art("gate_jitter")),
    ]
    rows = []
    for title, gate in gate_rows:
        nfg = gate["normalized_flux"]
        rows.append(" & ".join([
            title,
            integer(gate["gated_star_count"]),
            sci(nfg["median_over_stars"]),
            sci(nfg["max_over_stars"]),
            rf"{nfg['stars_over_bar']}/{gate['gated_star_count']}",
        ]) + r" \\")
    tables["tab_spectral.tex"] = table_env(
        label="tab:spectral",
        caption=(
            r"Normalized-flux differences between converged atmospheres over "
            r"\NgateWindowLo--\NgateWindowHi\,nm at $R=\NgateResolution$. "
            r"The \NgateBar\ value is retained as a common reference level, "
            "not as a universal acceptability boundary. Only stars with "
            "converged atmospheres from both initial states enter each row. "
            "The final row measures the start dependence of the reference "
            "solver and provides the numerical scale for the other comparisons."),
        colspec="lrrrr",
        header=(r"Comparison & $N$ & Median & Maximum & Above reference \\"),
        rows=rows,
        wide=True,
    )
    gl = art("gate_learned")["normalized_flux"]
    gj = art("gate_jitter")["normalized_flux"]
    macros.add("NlearnedGateMedian", gl["median_over_stars"],
               sci(gl["median_over_stars"]), "gate_learned")
    macros.add("NlearnedGateMax", gl["max_over_stars"], sci(gl["max_over_stars"]), "gate_learned")
    macros.add("NlearnedGateOver", gl["stars_over_bar"],
               integer(gl["stars_over_bar"]), "gate_learned")
    macros.add("NlearnedGateStars", art("gate_learned")["gated_star_count"],
               integer(art("gate_learned")["gated_star_count"]), "gate_learned")

    # Fig. 3, bottom row: how one-sided the worst star's residual actually is.
    _giant = (art("giant_learned")["normalized_flux"]
              - art("giant_released")["normalized_flux"])
    macros.add("NredGiantOneSided", float((_giant < 0).mean()),
               pct(float((_giant < 0).mean())), "giant_learned")
    macros.add("NjitterGateMedian", gj["median_over_stars"],
               sci(gj["median_over_stars"]), "gate_jitter")
    macros.add("NjitterGateMax", gj["max_over_stars"], sci(gj["max_over_stars"]), "gate_jitter")
    macros.add("NjitterGateOver", gj["stars_over_bar"],
               integer(gj["stars_over_bar"]), "gate_jitter")
    macros.add("NjitterGateStars", art("gate_jitter")["gated_star_count"],
               integer(art("gate_jitter")["gated_star_count"]), "gate_jitter")

    # ---- Table 9: development, independent test, and open-set seed check --
    # Keep the frozen 2026-08-11 test separate from the 2026-08-19 rerun.
    # The latter used the same frozen (m,T) predictions but rebuilt the four
    # dependent fields without the six-field emulator after the sample had
    # already been inspected.
    blind = art("blind")
    blind_open = art("blind_open")
    dev_profile = art("dev60_profile")
    policy = art("frozen_policy")
    prof_gate = art("blind_profile")
    bp = {
        "temperature_pointwise_p95":
            prof_gate["pointwise"]["temperature_relative"]["p95"],
        "temperature_pointwise_limit":
            prof_gate["thresholds"]["temperature_pointwise_p95_lte"],
        "column_mass_pointwise_p95_dex": prof_gate["pointwise"]["mass_dex"]["p95"],
        "column_mass_pointwise_limit_dex":
            prof_gate["thresholds"]["mass_pointwise_p95_lte_dex"],
        "temperature_blowout_stars": len(prof_gate["failure_stars"]["temperature"]),
        "column_mass_blowout_stars": len(prof_gate["failure_stars"]["mass"]),
        "union_blowout_stars": len(set(prof_gate["failure_stars"]["temperature"])
                                   | set(prof_gate["failure_stars"]["mass"])),
    }
    import numpy as np

    base_prediction = art("blind_base_prediction")
    final_prediction = art("blind_final_prediction")
    adapter_mass_move = np.max(
        np.abs(
            np.log10(np.maximum(final_prediction["column_mass"], 1.0e-300))
            - np.log10(np.maximum(base_prediction["column_mass"], 1.0e-300))
        ),
        axis=1,
    )
    adapter_temperature_move = np.max(
        np.abs(
            np.log10(np.maximum(final_prediction["temperature"], 1.0e-300))
            - np.log10(np.maximum(base_prediction["temperature"], 1.0e-300))
        ),
        axis=1,
    )
    adapter_move = np.maximum(adapter_mass_move, adapter_temperature_move)
    adapter_moved_millidex = int(np.count_nonzero(adapter_move > 1.0e-3))
    blind_gate = art("blind_spectra")
    open_gate = art("blind_open_spectra")
    bs = {
        "normalized_flux_max":
            blind_gate["normalized_flux"]["max_over_stars"],
        "normalized_flux_stars_over_bar":
            blind_gate["normalized_flux"]["stars_over_bar"],
        "paired_stars": blind_gate["gated_star_count"],
    }
    open_bs = {
        "normalized_flux_max":
            open_gate["normalized_flux"]["max_over_stars"],
        "normalized_flux_stars_over_bar":
            open_gate["normalized_flux"]["stars_over_bar"],
        "paired_stars": open_gate["gated_star_count"],
    }
    bsol = blind["solver"]
    dev_solver = policy["qualification"]["development_solver"]
    dev_spec = policy["qualification"]["development_spectral"]

    # ---- paired iteration statistics ------------------------------------
    # The original and open-set candidate records are paired separately with
    # the same frozen six-field reference records.
    import numpy as np
    from scipy.stats import chi2 as _chi2

    cand_rec = load_jsonl_records(art, "blind_cand_records")
    open_rec = load_jsonl_records(art, "blind_open_records")
    prod_rec = load_jsonl_records(art, "blind_prod_records")

    def paired_iteration_stats(candidate_records, gate):
        gate_slugs = {row["slug"] for row in gate["per_star"]}
        paired_slugs = sorted(
            slug
            for slug in set(candidate_records) & set(prod_rec) & gate_slugs
            if candidate_records[slug]["converged"]
            and prod_rec[slug]["converged"]
        )
        candidate_iterations = np.asarray(
            [
                candidate_records[slug]["converging_trial_iterations"]
                for slug in paired_slugs
            ],
            dtype=float,
        )
        production_iterations = np.asarray(
            [
                prod_rec[slug]["converging_trial_iterations"]
                for slug in paired_slugs
            ],
            dtype=float,
        )
        return {
            "slugs": paired_slugs,
            "candidate": candidate_iterations,
            "production": production_iterations,
            "difference": production_iterations - candidate_iterations,
        }

    paired = paired_iteration_stats(cand_rec, blind_gate)
    paired_open = paired_iteration_stats(open_rec, open_gate)
    diff = paired["difference"]
    n_pair = diff.size
    cand_pair_iters = paired["candidate"]
    prod_pair_iters = paired["production"]
    mean_diff = float(diff.mean())
    med_diff = float(np.median(diff))
    rng = np.random.default_rng(20260817)
    boot = np.empty(10_000)
    for i in range(10_000):
        boot[i] = diff[rng.integers(0, n_pair, n_pair)].mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    n_faster = int((diff > 0).sum())
    n_slower = int((diff < 0).sum())
    n_same = n_pair - n_faster - n_slower
    mcnemar_chi2 = (abs(n_faster - n_slower) - 1.0) ** 2 / (n_faster + n_slower)
    mcnemar_p = float(_chi2.sf(mcnemar_chi2, 1))
    independent_pair = {
        "stars": n_pair,
        "candidate_mean_iterations": float(np.mean(cand_pair_iters)),
        "production_mean_iterations": float(np.mean(prod_pair_iters)),
        "candidate_faster": n_faster,
        "same": n_same,
        "candidate_slower": n_slower,
    }
    open_pair = {
        "stars": len(paired_open["slugs"]),
        "candidate_mean_iterations": float(np.mean(paired_open["candidate"])),
        "production_mean_iterations": float(np.mean(paired_open["production"])),
    }
    independent_usable = int(bsol["candidate"]["usable_products"])
    open_usable = int(sum(row["converged"] for row in open_rec.values()))
    rows = [
        " & ".join([
            r"Stars",
            integer(dev_profile["star_count"]),
            integer(bp["temperature_blowout_stars"] * 0 + art("blind_profile")["star_count"]),
            integer(art("blind_profile")["star_count"]),
        ]) + r" \\",
        " & ".join([
            r"Temperature $p_{95}$",
            sci(dev_profile["pointwise"]["temperature_relative"]["p95"]),
            sci(bp["temperature_pointwise_p95"]),
            r"same prediction",
        ]) + r" \\",
        " & ".join([
            r"Column mass $p_{95}$ (dex)",
            sci(dev_profile["pointwise"]["mass_dex"]["p95"]),
            sci(bp["column_mass_pointwise_p95_dex"]),
            r"same prediction",
        ]) + r" \\",
        " & ".join([
            r"Large profile errors ($T/m$)",
            rf"{len(dev_profile['failure_stars']['temperature'])} / "
            rf"{len(dev_profile['failure_stars']['mass'])}",
            rf"{bp['temperature_blowout_stars']} / {bp['column_mass_blowout_stars']}",
            r"same prediction",
        ]) + r" \\",
        " & ".join([
            r"Usable two-field atmospheres",
            rf"{dev_solver['converged']}/{dev_solver['total']}",
            rf"{independent_usable}/{bsol['candidate']['requested']}",
            rf"{open_usable}/{art('blind_profile')['star_count']}",
        ]) + r" \\",
        " & ".join([
            r"Usable six-field atmospheres",
            rf"{dev_solver['total']}/{dev_solver['total']}",
            rf"{bsol['production']['usable_products']}/{bsol['production']['requested']}",
            rf"{bsol['production']['usable_products']}/{bsol['production']['requested']}",
        ]) + r" \\",
        " & ".join([
            r"Mean two-field iterations",
            dec(dev_solver["mean_iterations"], 2),
            dec(independent_pair["candidate_mean_iterations"], 2),
            dec(open_pair["candidate_mean_iterations"], 2),
        ]) + r" \\",
        " & ".join([
            r"Mean six-field iterations",
            r"---",
            dec(independent_pair["production_mean_iterations"], 2),
            dec(open_pair["production_mean_iterations"], 2),
        ]) + r" \\",
        " & ".join([
            r"Maximum spectral difference",
            sci(dev_spec["normalized_flux_max"]),
            sci(bs["normalized_flux_max"]),
            sci(open_bs["normalized_flux_max"]),
        ]) + r" \\",
        " & ".join([
            r"Above spectral reference level",
            rf"0/{dev_solver['total']}",
            rf"{bs['normalized_flux_stars_over_bar']}/{bs['paired_stars']}",
            rf"{open_bs['normalized_flux_stars_over_bar']}/{open_bs['paired_stars']}",
        ]) + r" \\",
    ]
    tables["tab_blind.tex"] = table_env(
        label="tab:blind",
        caption=(
            "The learned two-field initializer on the development sample "
            "and the independent 200-star test, followed by a post-hoc check "
            "that rebuilt the same predicted $(m,T)$ profiles without fields "
            "from the six-field model. The independent column contains the "
            "original independent calculation. The final column changes only the "
            "reconstruction seed and is not additional independent evidence. "
            "Iteration means for the 200-star columns use stars with both "
            "compared atmospheres available."),
        colspec="lrrr",
        header=(
            r"Quantity & Development & Independent test & Post-hoc reconstruction \\"
        ),
        rows=rows,
        wide=True,
        colsep=3.5,
        size="footnotesize",
    )
    macros.add("NblindStars", art("blind_profile")["star_count"],
               integer(art("blind_profile")["star_count"]), "blind_profile")
    macros.add("NblindTempPfive", bp["temperature_pointwise_p95"],
               sci(bp["temperature_pointwise_p95"]), "blind_profile")
    macros.add("NblindTempLimit", bp["temperature_pointwise_limit"],
               sci(bp["temperature_pointwise_limit"]), "blind_profile")
    macros.add("NblindMassPfive", bp["column_mass_pointwise_p95_dex"],
               sci(bp["column_mass_pointwise_p95_dex"]), "blind_profile")
    macros.add("NblindMassLimit", bp["column_mass_pointwise_limit_dex"],
               sci(bp["column_mass_pointwise_limit_dex"]), "blind_profile")
    macros.add("NblindTempBlowouts", bp["temperature_blowout_stars"],
               integer(bp["temperature_blowout_stars"]), "blind_profile")
    macros.add("NblindMassBlowouts", bp["column_mass_blowout_stars"],
               integer(bp["column_mass_blowout_stars"]), "blind_profile")
    macros.add("NblindUnionBlowouts", bp["union_blowout_stars"],
               integer(bp["union_blowout_stars"]), "blind_profile")
    macros.add("NblindAdapterMovedMillidex", adapter_moved_millidex,
               integer(adapter_moved_millidex), "blind_final_prediction")
    pair = independent_pair
    macros.add(
        "NblindPaired", pair["stars"], integer(pair["stars"]), "blind_cand_records"
    )
    macros.add("NblindCandIters", pair["candidate_mean_iterations"],
               dec(pair["candidate_mean_iterations"], 2), "blind_cand_records")
    macros.add("NblindProdIters", pair["production_mean_iterations"],
               dec(pair["production_mean_iterations"], 2), "blind_prod_records")
    macros.add("NblindFaster", pair["candidate_faster"],
               integer(pair["candidate_faster"]), "blind_cand_records")
    macros.add(
        "NblindSame", pair["same"], integer(pair["same"]), "blind_cand_records"
    )
    macros.add("NblindSlower", pair["candidate_slower"],
               integer(pair["candidate_slower"]), "blind_cand_records")
    macros.add("NblindCandUsable", independent_usable,
               integer(independent_usable), "blind")
    macros.add("NblindProdUsable", bsol["production"]["usable_products"],
               integer(bsol["production"]["usable_products"]), "blind_production")
    macros.add("NblindSpecOver", bs["normalized_flux_stars_over_bar"],
               integer(bs["normalized_flux_stars_over_bar"]), "blind_spectra")
    macros.add("NblindSpecMax", bs["normalized_flux_max"],
               sci(bs["normalized_flux_max"]), "blind_spectra")
    macros.add("NblindSpecMedian",
               art("blind_spectra")["normalized_flux"]["median_over_stars"],
               sci(art("blind_spectra")["normalized_flux"]["median_over_stars"]),
               "blind_spectra")
    macros.add("NblindCandNonmono", bsol["candidate"]["non_monotonic_fraction"],
               pct(bsol["candidate"]["non_monotonic_fraction"]), "blind")
    macros.add("NblindProdNonmono", bsol["production"]["non_monotonic_fraction"],
               pct(bsol["production"]["non_monotonic_fraction"]), "blind_production")
    macros.add("NblindOpenUsable", open_usable, integer(open_usable),
               "blind_open_records")
    macros.add("NblindOpenPaired", open_pair["stars"],
               integer(open_pair["stars"]), "blind_open_records")
    macros.add("NblindOpenCandIters", open_pair["candidate_mean_iterations"],
               dec(open_pair["candidate_mean_iterations"], 2),
               "blind_open_records")
    macros.add("NblindOpenProdIters", open_pair["production_mean_iterations"],
               dec(open_pair["production_mean_iterations"], 2),
               "blind_prod_records")
    macros.add("NblindOpenSpecOver",
               open_bs["normalized_flux_stars_over_bar"],
               integer(open_bs["normalized_flux_stars_over_bar"]),
               "blind_open_spectra")
    macros.add("NblindOpenSpecStars", open_bs["paired_stars"],
               integer(open_bs["paired_stars"]), "blind_open_spectra")
    macros.add("NblindOpenSpecMax", open_bs["normalized_flux_max"],
               sci(open_bs["normalized_flux_max"]), "blind_open_spectra")
    macros.add("NdevIterMean", dev_solver["mean_iterations"],
               dec(dev_solver["mean_iterations"], 2), "frozen_policy")
    macros.add("NdevTempPfive", dev_profile["pointwise"]["temperature_relative"]["p95"],
               sci(dev_profile["pointwise"]["temperature_relative"]["p95"]), "dev60_profile")
    macros.add("NdevMassPfive", dev_profile["pointwise"]["mass_dex"]["p95"],
               sci(dev_profile["pointwise"]["mass_dex"]["p95"]), "dev60_profile")

    # ---- paired blind-test statistics -----------------------------------
    # The paired aggregates (n_pair, mean/median diff, bootstrap interval,
    # faster/same/slower and the McNemar p-value) were recomputed above from
    # the records before the table was emitted; the macros below only report
    # those already-computed values.
    macros.add("NblindPairedDiffMean", mean_diff, dec(mean_diff, 2),
               "blind_cand_records")
    macros.add("NblindPairedDiffMedian", med_diff, dec(med_diff, 2),
               "blind_cand_records")
    macros.add("NblindPairedDiffCiLo", float(ci_lo), dec(float(ci_lo), 2),
               "blind_cand_records")
    macros.add("NblindPairedDiffCiHi", float(ci_hi), dec(float(ci_hi), 2),
               "blind_cand_records")
    macros.add("NblindMcnemarP", mcnemar_p, sci(mcnemar_p, 1),
               "blind_cand_records")

    # Development spectral gate: 0/60 over the bar, with a 95 per cent
    # one-sided binomial (Clopper--Pearson) upper bound, so the zero counts are
    # not read as a zero failure rate.
    dev_gate = art("dev60_gate")
    dev_over = dev_gate["normalized_flux"]["stars_over_bar"]
    dev_n = dev_gate["gated_star_count"]
    if dev_over != 0:
        raise SystemExit(f"development spectral over-bar count is {dev_over}, "
                         "expected 0")
    dev_upper = 1.0 - 0.05 ** (1.0 / dev_n)
    macros.add("NdevSpecOver", dev_over, integer(dev_over), "dev60_gate")
    macros.add("NdevSpecStars", dev_n, integer(dev_n), "dev60_gate")
    macros.add("NdevSpecUpper", dev_upper, pct(dev_upper), "dev60_gate")

    # Sample-selection provenance.
    manifest = art("blind_manifest")
    macros.add("NblindSelectionSeed", manifest["selection"]["seed"],
               integer(manifest["selection"]["seed"]), "blind_manifest")
    macros.add("NdevSelectionSeed", manifest["solver_selection"]["seed"],
               integer(manifest["solver_selection"]["seed"]), "blind_manifest")
    macros.add("NdevQualIters", dev_solver["mean_iterations"],
               dec(dev_solver["mean_iterations"], 2), "frozen_policy")

    # ---- bounded native-MARCS M-star science case ----------------------
    mstar = art("mstar_cases")
    mstar_cases = list(mstar.get("cases", []))
    if mstar.get("campaign") != "m_star_science_case_v1":
        raise SystemExit("wrong M-star case-study artifact")
    if len(mstar_cases) != 8 or mstar.get("case_count") != 8:
        raise SystemExit("M-star case-study artifact does not contain exactly 8 cases")

    def _mstar_status(record: dict) -> str:
        value = record.get("iterations")
        if value is None:
            return r"\ensuremath{--}"
        status = "Yes" if bool(record.get("survives_solver")) else "No"
        return f"{status} ({integer(value)})"

    def _mstar_flux(record: dict) -> float | None:
        value = record.get("flux_imbalance", {}).get("max_percent")
        return None if value is None else float(value)

    def _mstar_korg_p95(case: dict) -> float | None:
        comparison = case.get("korg_comparison", {})
        arm = comparison.get("arms", {}).get("molecular", {})
        result = arm.get("payne_zero_vs_same_atmosphere_korg", {})
        value = result.get("normalized_flux", {}).get("p95")
        return None if value is None else float(value)

    def _mstar_selected(case: dict) -> dict | None:
        route = case.get("selected_route")
        if route == "continuation_primary":
            return case["continuation_primary"]
        if route == "direct":
            return case["direct"]
        return None

    direct_count = sum(bool(case["direct"].get("survives_solver")) for case in mstar_cases)
    continuation_count = sum(bool(case["continuation_primary"].get("survives_solver")) for case in mstar_cases)
    selected_count = sum(bool(case.get("selected_product_path")) for case in mstar_cases)
    korg_count = sum(_mstar_korg_p95(case) is not None for case in mstar_cases)
    selected_flux = [
        _mstar_flux(selected)
        for case in mstar_cases
        if (selected := _mstar_selected(case)) is not None
    ]
    selected_flux = [value for value in selected_flux if value is not None]
    korg_p95 = [value for value in (_mstar_korg_p95(case) for case in mstar_cases) if value is not None]
    dwarf_cases = [case for case in mstar_cases if case["class"] == "M-dwarf"]
    giant_cases = [case for case in mstar_cases if case["class"] == "M-giant"]
    dwarf_direct_count = sum(
        bool(case["direct"].get("survives_solver")) for case in dwarf_cases
    )
    giant_direct_count = sum(
        bool(case["direct"].get("survives_solver")) for case in giant_cases
    )
    dwarf_continuation = [
        case
        for case in dwarf_cases
        if case["continuation_primary"].get("survives_solver")
    ]
    coolest_dwarf_continuation = min(
        dwarf_continuation,
        key=lambda case: float(case["labels"]["effective_temperature"]),
    )
    dwarf_3750 = next(
        case
        for case in dwarf_cases
        if float(case["labels"]["effective_temperature"]) == 3750.0
    )
    coolest_dwarf_flux = _mstar_flux(
        coolest_dwarf_continuation["continuation_primary"]
    )
    warm_dwarf_flux = _mstar_flux(dwarf_3750["continuation_primary"])
    giant_selected_flux = [
        _mstar_flux(selected)
        for case in giant_cases
        if (selected := _mstar_selected(case)) is not None
    ]
    giant_selected_flux = [
        value for value in giant_selected_flux if value is not None
    ]
    if (
        coolest_dwarf_flux is None
        or warm_dwarf_flux is None
        or not giant_selected_flux
    ):
        raise SystemExit("M-star converged cases are missing final flux diagnostics")

    macros.add("NMstarCases", len(mstar_cases), integer(len(mstar_cases)), "mstar_cases")
    macros.add(
        "NMstarDwarfCases",
        len(dwarf_cases),
        integer(len(dwarf_cases)),
        "mstar_cases",
    )
    macros.add(
        "NMstarGiantCases",
        len(giant_cases),
        integer(len(giant_cases)),
        "mstar_cases",
    )
    macros.add("NMstarDirectConverged", direct_count, integer(direct_count), "mstar_cases")
    macros.add("NMstarContinuationConverged", continuation_count, integer(continuation_count), "mstar_cases")
    macros.add(
        "NMstarDwarfDirectConverged",
        dwarf_direct_count,
        integer(dwarf_direct_count),
        "mstar_cases",
    )
    macros.add(
        "NMstarGiantDirectConverged",
        giant_direct_count,
        integer(giant_direct_count),
        "mstar_cases",
    )
    macros.add(
        "NMstarDwarfCoolestContinuation",
        float(coolest_dwarf_continuation["labels"]["effective_temperature"]),
        integer(coolest_dwarf_continuation["labels"]["effective_temperature"]),
        "mstar_cases",
    )
    macros.add(
        "NMstarDwarfFluxCoolest",
        coolest_dwarf_flux,
        integer(round(coolest_dwarf_flux)),
        "mstar_cases",
    )
    macros.add(
        "NMstarDwarfFluxWarm",
        warm_dwarf_flux,
        dec(warm_dwarf_flux, 1),
        "mstar_cases",
    )
    macros.add(
        "NMstarGiantFluxMinimum",
        min(giant_selected_flux),
        dec(min(giant_selected_flux), 2),
        "mstar_cases",
    )
    macros.add(
        "NMstarGiantFluxMaximum",
        max(giant_selected_flux),
        dec(max(giant_selected_flux), 2),
        "mstar_cases",
    )
    macros.add("NMstarSelected", selected_count, integer(selected_count), "mstar_cases")
    macros.add("NMstarKorgCompared", korg_count, integer(korg_count), "mstar_cases")
    macros.add(
        "NMstarFluxMaximum",
        max(selected_flux) if selected_flux else 0.0,
        sci(max(selected_flux)) if selected_flux else r"\ensuremath{--}",
        "mstar_cases",
    )
    macros.add(
        "NMstarKorgPfiveMaximum",
        max(korg_p95) if korg_p95 else 0.0,
        sci(max(korg_p95)) if korg_p95 else r"\ensuremath{--}",
        "mstar_cases",
    )

    mstar_rows = []
    ordered_mstar_cases = sorted(
        mstar_cases,
        key=lambda case: (
            0 if case["class"] == "M-dwarf" else 1,
            -float(case["labels"]["effective_temperature"]),
        ),
    )
    for case in ordered_mstar_cases:
        labels = case["labels"]
        direct = case["direct"]
        continuation = case["continuation_primary"]
        selected = _mstar_selected(case)
        flux = None if selected is None else _mstar_flux(selected)
        title = "M dwarf" if case["class"] == "M-dwarf" else "M giant"
        mstar_rows.append(" & ".join([
            title,
            integer(labels["effective_temperature"]),
            dec(labels["log_surface_gravity"], 1),
            dec(labels["microturbulence_km_s"], 1),
            _mstar_status(direct),
            _mstar_status(continuation),
            sci(flux) if flux is not None else r"\ensuremath{--}",
        ]) + r" \\")
    tables["tab_mstar.tex"] = table_env(
        label="tab:mstar",
        caption=(
            "Temperature continuation across eight solar-composition MARCS "
            "M-star atmospheres. Direct-start and temperature-sequence entries "
            "give convergence status with the iteration count in parentheses; "
            "an em dash means that the sequence was not continued past a failed "
            "warmer node. The final column gives the maximum absolute all-layer "
            "flux imbalance for the retained atmosphere, using the temperature "
            "sequence where it converges and the direct start otherwise. The "
            "large residual at 3500 K prevents interpreting the formally "
            "converged dwarf as a validated atmosphere."
        ),
        colspec="llrrrrr",
        header=(
            r"Class & $T_{\rm eff}$ & $\log g$ & $\xi$ & Direct start & "
            r"$T_{\rm eff}$ sequence & $\max|\Delta F|$ (\%) \\"
        ),
        rows=mstar_rows,
        wide=True,
        colsep=5.0,
        size="footnotesize",
        spaced_rules=False,
    )

    return tables


def table_env(*, label: str, caption: str, colspec: str, header: str,
              rows: list[str], wide: bool = False, colsep: float | None = None,
              size: str | None = None, spaced_rules: bool = True) -> str:
    """Emit a complete A&A table environment.

    ``colsep`` overrides the inter-column padding in points and ``size`` sets a
    font-size command; both exist because several of these tables carry four
    columns of scientific notation and do not otherwise fit an 88 mm column.
    """

    env = "table*" if wide else "table"
    preamble = ""
    if colsep is not None:
        preamble += f"\\setlength{{\\tabcolsep}}{{{colsep}pt}}\n"
    if size is not None:
        preamble += f"\\{size}\n"
    top_rule = "\\hline\\hline\\noalign{\\smallskip}\n" if spaced_rules else "\\hline\\hline\n"
    mid_rule = (
        "\\noalign{\\smallskip}\\hline\\noalign{\\smallskip}\n"
        if spaced_rules else "\\hline\n"
    )
    bottom_rule = "\\noalign{\\smallskip}\\hline\n" if spaced_rules else "\\hline\n"
    return (
        "% Generated by paper/collect_numbers.py -- do not edit by hand.\n"
        f"\\begin{{{env}}}\n"
        "\\caption{" + caption + "}\n"
        f"\\label{{{label}}}\n"
        "\\centering\n"
        f"{preamble}"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        f"{top_rule}"
        f"{header}\n"
        f"{mid_rule}"
        + "\n".join(rows) + "\n"
        f"{bottom_rule}"
        "\\end{tabular}\n"
        f"\\end{{{env}}}\n"
    )


def provenance_table(art: Artifacts) -> str:
    rows = []
    for name in sorted(SOURCES):
        if name not in art.hashes:
            continue
        # Paths run to ~80 characters and \texttt does not hyphenate, so the
        # cell has to be told where it may break: after every separator.
        path = (SOURCES[name].replace("_", r"\_")
                .replace("/", r"/\allowbreak ")
                .replace(r"\_", r"\_\allowbreak "))
        rows.append(
            rf"\texttt{{{path}}} & {SOURCE_CAPTIONS[name]} & "
            rf"\texttt{{{art.hashes[name][:12]}}} \\"
            "\n\\noalign{\\smallskip}"
        )
    caption = (
        "Numerical sources used to generate the reported values and tables. "
        "Hashes are the first 12 hexadecimal digits of each file's SHA-256; "
        "complete values are provided in "
        r"\texttt{paper/numbers.json}.")
    colspec = (r">{\raggedright\arraybackslash}p{0.40\hsize}"
               r">{\raggedright\arraybackslash}p{0.36\hsize}l")
    header = r"Source file & Content & SHA-256 \\"

    # A table* cannot break across pages. Split this audit inventory into two
    # floats so additions to the manifest do not silently run off the page.
    split = (len(rows) + 1) // 2
    first = table_env(
        label="tab:provenance",
        caption=caption,
        colspec=colspec,
        header=header,
        rows=rows[:split],
        wide=True,
        size="footnotesize",
    )
    continuation = (
        "\\begin{table*}\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\textit{Table~\\ref{tab:provenance} continued.}\\par\\smallskip\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\hline\\hline\\noalign{\\smallskip}\n"
        f"{header}\n"
        "\\noalign{\\smallskip}\\hline\\noalign{\\smallskip}\n"
        + "\n".join(rows[split:]) + "\n"
        "\\noalign{\\smallskip}\\hline\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )
    return first + continuation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify regeneration reproduces numbers.json; do not write")
    args = parser.parse_args()

    art = Artifacts()
    macros = Macros()
    try:
        tables = build(art, macros)
    except FileNotFoundError as exc:
        print(f"missing artifact: {exc}", file=sys.stderr)
        return 2

    payload = {
        "sources": {name: {"path": SOURCES[name], "sha256": art.hashes.get(name)}
                    for name in sorted(SOURCES)},
        "macros": {name: macros.entries[name] for name in sorted(macros.entries)},
    }

    numbers_json = PAPER / "numbers.json"
    if args.check:
        if not numbers_json.is_file():
            print("numbers.json does not exist; run without --check first", file=sys.stderr)
            return 3
        with numbers_json.open() as handle:
            previous = json.load(handle)
        drift = []
        for name, entry in payload["macros"].items():
            old = previous["macros"].get(name)
            if old is None:
                drift.append(f"new macro {name}")
            elif old["tex"] != entry["tex"]:
                drift.append(f"{name}: {old['tex']} -> {entry['tex']}")
        for name in previous["macros"]:
            if name not in payload["macros"]:
                drift.append(f"removed macro {name}")
        for name, entry in payload["sources"].items():
            old = previous["sources"].get(name, {})
            if old.get("sha256") and entry["sha256"] and old["sha256"] != entry["sha256"]:
                drift.append(f"source {name} changed hash")
        numbers_tex = PAPER / "numbers.tex"
        if not numbers_tex.is_file():
            drift.append("paper/numbers.tex is missing")
        elif numbers_tex.read_text() != macros.render():
            drift.append("paper/numbers.tex differs from regenerated macros")
        for filename, body in tables.items():
            path = TABLES / filename
            if not path.is_file():
                drift.append(f"paper/tables/{filename} is missing")
            elif path.read_text() != body:
                drift.append(f"paper/tables/{filename} differs from regenerated table")
        if drift:
            print("drift detected:", file=sys.stderr)
            for line in drift:
                print("  " + line, file=sys.stderr)
            return 1
        print(f"ok: {len(payload['macros'])} macros, "
              f"{len([s for s in payload['sources'].values() if s['sha256']])} sources")
        return 0

    TABLES.mkdir(parents=True, exist_ok=True)
    (PAPER / "numbers.tex").write_text(macros.render())
    for filename, body in tables.items():
        (TABLES / filename).write_text(body)
    with numbers_json.open("w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")

    print(f"wrote {len(macros.entries)} macros to paper/numbers.tex")
    print(f"wrote {len(tables)} tables to paper/tables/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
