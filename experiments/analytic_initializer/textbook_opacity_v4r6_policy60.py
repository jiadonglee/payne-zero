"""Frozen policy and provenance helpers for the matched v4r6 60-iteration study."""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Mapping, Sequence

from experiments.analytic_initializer.discovery import file_sha256
from experiments.analytic_initializer.textbook_opacity_v4r6_decoupled_gates import (
    outcome_counts,
    paired_solver_counts,
    records_by_index,
)

POLICY = "v4r6_analytic_warm_start_policy60_v1"
ITERATIONS = 60
PER_STAR_TIMEOUT_SECONDS = 900
SAMPLE = Path(
    "results/paper_physical_seed_20260820/learned/"
    "convergence_metrics_learned_monotone.json"
)
SOURCE_MANIFEST = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r6_policy60_source_manifest_20260829.json"
)

ARM_OUTPUTS = {
    "textbook_v4r6_decoupled": Path(
        "results/analytic_initializer/"
        "textbook_opacity_v4r6_decoupled_dev60_policy60_20260829.json"
    ),
    "textbook_v4r6_grey": Path(
        "results/analytic_initializer/"
        "textbook_opacity_v4r6_grey_dev60_policy60_20260829.json"
    ),
    "textbook_v4r6": Path(
        "results/analytic_initializer/"
        "textbook_opacity_v4r6_convective_dev60_policy60_20260829.json"
    ),
}

# These gates decide only whether a genuinely fresh validation should be
# preregistered. The exposed development-60 cannot validate the policy.
ABSOLUTE_TOTAL_MIN = 54
ABSOLUTE_COOL_MIN = 23
ABSOLUTE_HOT_MIN = 29
ABSOLUTE_TIMEOUTS_MAX = 6
COOL_NET_GAIN_VS_GREY_MIN = 4
HOT_NET_LOSS_VS_GREY_MAX = 2
TOTAL_NET_GAIN_VS_GREY_MIN = 0


def verify_source_manifest() -> tuple[dict[str, object], str]:
    """Require every source hash to equal the frozen policy60 manifest."""

    if not SOURCE_MANIFEST.is_file():
        raise SystemExit(f"missing frozen source manifest {SOURCE_MANIFEST}")
    payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected = dict(payload.get("source_sha256") or {})
    if not expected:
        raise SystemExit("policy60 source manifest has no source hashes")
    mismatches: dict[str, dict[str, str | None]] = {}
    for name, expected_hash in expected.items():
        path = Path(name)
        actual_hash = file_sha256(path) if path.is_file() else None
        if actual_hash != expected_hash:
            mismatches[name] = {
                "expected": None if expected_hash is None else str(expected_hash),
                "actual": actual_hash,
            }
    if mismatches:
        raise SystemExit(
            "policy60 source drift; launch refused: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return payload, file_sha256(SOURCE_MANIFEST)


def runtime_signature(arm: str) -> dict[str, object]:
    """Return the complete source, sample, policy, and runtime identity."""

    if arm not in ARM_OUTPUTS:
        raise ValueError(f"unknown policy60 arm {arm!r}")
    manifest, manifest_hash = verify_source_manifest()
    if not SAMPLE.is_file():
        raise SystemExit(f"missing frozen sample {SAMPLE}")
    try:
        import numba
    except ImportError:  # pragma: no cover
        numba = None
    try:
        import numpy
    except ImportError:  # pragma: no cover
        numpy = None
    return {
        "policy": POLICY,
        "arm": arm,
        "trials": 1,
        "iterations": ITERATIONS,
        "per_star_timeout_seconds": PER_STAR_TIMEOUT_SECONDS,
        "sample": str(SAMPLE),
        "sample_sha256": file_sha256(SAMPLE),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": manifest_hash,
        "source_git_head": manifest.get("git_head"),
        "source_git_diff_sha256": manifest.get("git_diff_sha256"),
        "hostname": socket.gethostname(),
        "operating_system": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": None if numpy is None else numpy.__version__,
        "numba": None if numba is None else numba.__version__,
        "environment": {
            "NUMBA_THREADING_LAYER": os.environ.get("NUMBA_THREADING_LAYER"),
            "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
        },
    }


def enforce_runtime_guard(arm: str) -> dict[str, object]:
    """Freeze one arm's signature before its first streamed row."""

    signature = runtime_signature(arm)
    guard = ARM_OUTPUTS[arm].with_suffix(".runtime.json")
    if guard.is_file():
        previous = json.loads(guard.read_text(encoding="utf-8"))
        if previous != signature:
            raise SystemExit(
                f"{guard} disagrees with the current policy60 signature; resume refused"
            )
    else:
        guard.parent.mkdir(parents=True, exist_ok=True)
        guard.write_text(
            json.dumps(signature, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return signature


def split_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    cool = [
        item for item in records if float(item["effective_temperature"]) < 7500.0
    ]
    hot = [
        item for item in records if float(item["effective_temperature"]) >= 7500.0
    ]
    return {"all": list(records), "cool": cool, "hot": hot}


def paired_split_counts(
    candidate_records: Sequence[Mapping[str, object]],
    control_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return paired candidate/control counts on all, cool, and hot stars."""

    candidate = records_by_index(candidate_records)
    control = records_by_index(control_records)
    if set(candidate) != set(control):
        raise ValueError("policy60 paired arms do not share identical indices")
    candidate_splits = split_records(list(candidate.values()))
    control_splits = split_records(list(control.values()))
    return {
        name: paired_solver_counts(candidate_splits[name], control_splits[name])
        for name in ("all", "cool", "hot")
    }


def continuation_gate(
    candidate: Mapping[str, object],
    grey: Mapping[str, object],
    convective: Mapping[str, object],
) -> dict[str, object]:
    """Score the exposed matched study only for permission to preregister fresh-open."""

    candidate_records = list(candidate["records"])  # type: ignore[arg-type]
    grey_records = list(grey["records"])  # type: ignore[arg-type]
    convective_records = list(convective["records"])  # type: ignore[arg-type]
    splits = split_records(candidate_records)
    counts = outcome_counts(candidate_records)
    cool = outcome_counts(splits["cool"])
    hot = outcome_counts(splits["hot"])
    paired_grey = paired_split_counts(candidate_records, grey_records)
    paired_convective = paired_split_counts(candidate_records, convective_records)
    finite_seeds = int(
        dict(candidate.get("initializer_provenance") or {}).get(
            "finite_seed_count", -1
        )
    )
    checks = {
        "complete_records": counts["star_count"] == 60,
        "solver_errors": counts["error_count"] == 0,
        "finite_seeds": finite_seeds == 60,
        "absolute_total": counts["converged_count"] >= ABSOLUTE_TOTAL_MIN,
        "absolute_cool": cool["converged_count"] >= ABSOLUTE_COOL_MIN,
        "absolute_hot": hot["converged_count"] >= ABSOLUTE_HOT_MIN,
        "timeouts": counts["timeout_count"] <= ABSOLUTE_TIMEOUTS_MAX,
        "total_net_gain_vs_grey": (
            int(paired_grey["all"]["net_gain"]) >= TOTAL_NET_GAIN_VS_GREY_MIN
        ),
        "cool_net_gain_vs_grey": (
            int(paired_grey["cool"]["net_gain"]) >= COOL_NET_GAIN_VS_GREY_MIN
        ),
        "hot_net_loss_vs_grey": (
            -int(paired_grey["hot"]["net_gain"]) <= HOT_NET_LOSS_VS_GREY_MAX
        ),
    }
    decision = (
        "CONTINUE_TO_POLICY60_FRESH_OPEN_PREREGISTRATION"
        if all(checks.values())
        else "STOP_POLICY60_MATCHED_DEVELOPMENT"
    )
    return {
        "decision": decision,
        "authorizes_fresh_open_execution": False,
        "checks": checks,
        "candidate_counts": counts,
        "candidate_cool": cool,
        "candidate_hot": hot,
        "finite_seed_count": finite_seeds,
        "paired_vs_grey": paired_grey,
        "paired_vs_convective": paired_convective,
        "thresholds": {
            "absolute_total_min": ABSOLUTE_TOTAL_MIN,
            "absolute_cool_min": ABSOLUTE_COOL_MIN,
            "absolute_hot_min": ABSOLUTE_HOT_MIN,
            "absolute_timeouts_max": ABSOLUTE_TIMEOUTS_MAX,
            "total_net_gain_vs_grey_min": TOTAL_NET_GAIN_VS_GREY_MIN,
            "cool_net_gain_vs_grey_min": COOL_NET_GAIN_VS_GREY_MIN,
            "hot_net_loss_vs_grey_max": HOT_NET_LOSS_VS_GREY_MAX,
        },
    }
