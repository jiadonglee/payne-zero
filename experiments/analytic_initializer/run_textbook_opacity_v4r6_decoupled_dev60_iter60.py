"""Run the v4r6 decoupled development-60 funnel with 60 solver iterations.

This is a late-convergence diagnostic. It does not overwrite the frozen
15-iteration FAIL_STOP_DEVELOPMENT result, does not relax that gate, and
does not authorize fresh-open 120.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.analytic_initializer.discovery import file_sha256
from experiments.analytic_initializer.run_h2_solver_funnel import main as funnel_main
from experiments.analytic_initializer.textbook_opacity_v4r6_decoupled_gates import (
    CANDIDATE,
    paired_solver_counts,
    records_by_index,
    split_records,
)
from experiments.analytic_initializer.write_textbook_opacity_v4r6_decoupled_source_manifest import (
    OUTPUT as SOURCE_MANIFEST,
)

INDICES_FROM = Path(
    "results/paper_physical_seed_20260820/learned/"
    "convergence_metrics_learned_monotone.json"
)
OUTPUT = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r6_decoupled_dev60_iter60_20260829.json"
)
FIFTEEN_CONTROL = Path(
    "results/analytic_initializer/textbook_opacity_v4r6_decoupled_dev60_20260828.json"
)
GREY_CONTROL = Path(
    "results/analytic_initializer/textbook_opacity_v4r6_grey_dev60_20260828.json"
)
CONVECTIVE_CONTROL = Path(
    "results/analytic_initializer/textbook_opacity_v4r6_dev60_20260828.json"
)
RUNTIME_GUARD = OUTPUT.with_suffix(".runtime.json")
ITERATIONS = 60
PER_STAR_TIMEOUT_SECONDS = 900
DECISION = "ITER60_DIAGNOSTIC_COMPLETE"


def _late_convergence(candidate_records, control_records) -> dict[str, object]:
    """Compare the 60-iteration arm to the frozen 15-iteration decoupled arm."""

    candidate = records_by_index(candidate_records)
    control = records_by_index(control_records)
    shared = sorted(set(candidate) & set(control))
    recovered: list[int] = []
    lost: list[int] = []
    still_converged: list[int] = []
    still_failed: list[int] = []
    for index in shared:
        now = bool(candidate[index].get("converged"))
        before = bool(control[index].get("converged"))
        if now and before:
            still_converged.append(index)
        elif now and not before:
            recovered.append(index)
        elif (not now) and before:
            lost.append(index)
        else:
            still_failed.append(index)
    recovered_records = [candidate[index] for index in recovered]
    recovered_splits = split_records(recovered_records) if recovered_records else {
        "cool": [],
        "hot": [],
    }
    return {
        "shared_count": len(shared),
        "still_converged_count": len(still_converged),
        "recovered_count": len(recovered),
        "lost_count": len(lost),
        "still_failed_count": len(still_failed),
        "recovered_indices": recovered,
        "lost_indices": lost,
        "recovered_cool_count": len(recovered_splits["cool"]),
        "recovered_hot_count": len(recovered_splits["hot"]),
        "paired_vs_15iter_decoupled": paired_solver_counts(
            candidate_records, control_records
        ),
    }


def _runtime_signature() -> dict[str, object]:
    missing = [
        str(path)
        for path in (SOURCE_MANIFEST, INDICES_FROM, FIFTEEN_CONTROL, GREY_CONTROL, CONVECTIVE_CONTROL)
        if not path.is_file()
    ]
    if missing:
        raise SystemExit("missing frozen artifacts: " + ", ".join(missing))
    return {
        "arm": "textbook_v4r6_decoupled",
        "candidate": CANDIDATE,
        "trials": 1,
        "iterations": ITERATIONS,
        "per_star_timeout_seconds": PER_STAR_TIMEOUT_SECONDS,
        "indices_from": str(INDICES_FROM),
        "indices_from_sha256": file_sha256(INDICES_FROM),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": file_sha256(SOURCE_MANIFEST),
        "fifteen_iter_control": str(FIFTEEN_CONTROL),
        "fifteen_iter_control_sha256": file_sha256(FIFTEEN_CONTROL),
        "grey_control": str(GREY_CONTROL),
        "grey_control_sha256": file_sha256(GREY_CONTROL),
        "convective_control": str(CONVECTIVE_CONTROL),
        "convective_control_sha256": file_sha256(CONVECTIVE_CONTROL),
        "frozen_15iter_decision": "FAIL_STOP_DEVELOPMENT",
        "authorizes_fresh_open": False,
    }


def _enforce_runtime_guard() -> dict[str, object]:
    signature = _runtime_signature()
    if RUNTIME_GUARD.is_file():
        previous = json.loads(RUNTIME_GUARD.read_text(encoding="utf-8"))
        if previous != signature:
            raise SystemExit(
                f"{RUNTIME_GUARD} disagrees with the current runtime signature; "
                "resume is refused"
            )
    else:
        RUNTIME_GUARD.parent.mkdir(parents=True, exist_ok=True)
        RUNTIME_GUARD.write_text(
            json.dumps(signature, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return signature


def _enrich(result: dict[str, object], signature: dict[str, object]) -> dict[str, object]:
    fifteen = json.loads(FIFTEEN_CONTROL.read_text(encoding="utf-8"))
    grey = json.loads(GREY_CONTROL.read_text(encoding="utf-8"))
    convective = json.loads(CONVECTIVE_CONTROL.read_text(encoding="utf-8"))
    late = _late_convergence(result["records"], fifteen["records"])
    result["status"] = "development_only"
    result["candidate"] = CANDIDATE
    result["decision"] = DECISION
    result["authorizes_fresh_open"] = False
    result["frozen_15iter_decision"] = "FAIL_STOP_DEVELOPMENT"
    result["late_convergence"] = late
    result["source_manifest"] = str(SOURCE_MANIFEST)
    result["source_manifest_sha256"] = signature["source_manifest_sha256"]
    result["sample_manifest"] = str(INDICES_FROM)
    result["sample_manifest_sha256"] = signature["indices_from_sha256"]
    result["runtime_signature"] = signature
    result["solver_policy"] = {
        "trials": 1,
        "iterations": ITERATIONS,
        "per_star_timeout_seconds": PER_STAR_TIMEOUT_SECONDS,
    }
    result["frozen_controls"] = {
        "v4r6_decoupled_15iter": {
            "path": str(FIFTEEN_CONTROL),
            "sha256": signature["fifteen_iter_control_sha256"],
            "converged_count": fifteen.get("converged_count"),
            "decision": fifteen.get("decision"),
            "iterations_per_trial": fifteen.get("iterations_per_trial", 15),
        },
        "v4r6_convective_15iter": {
            "path": str(CONVECTIVE_CONTROL),
            "sha256": signature["convective_control_sha256"],
            "converged_count": convective.get("converged_count"),
        },
        "v4r6_grey_15iter": {
            "path": str(GREY_CONTROL),
            "sha256": signature["grey_control_sha256"],
            "converged_count": grey.get("converged_count"),
        },
    }
    result["paired_summary"] = {
        "vs_15iter_decoupled": late["paired_vs_15iter_decoupled"],
        "vs_grey_15iter_mixed_policy": paired_solver_counts(
            result["records"], grey["records"]
        ),
        "vs_convective_15iter_mixed_policy": paired_solver_counts(
            result["records"], convective["records"]
        ),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    if argv:
        raise SystemExit("this driver pins its sample, iterations, and output path")
    signature = _enforce_runtime_guard()
    code = funnel_main(
        [
            "--arm",
            "textbook_v4r6_decoupled",
            "--count",
            "60",
            "--indices-from",
            str(INDICES_FROM),
            "--iterations",
            str(ITERATIONS),
            "--per-star-timeout",
            str(PER_STAR_TIMEOUT_SECONDS),
            "--resume",
            "--out",
            str(OUTPUT),
        ]
    )
    if code != 0:
        return code
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    result = _enrich(result, signature)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "authorizes_fresh_open": False,
                "converged_count": result.get("converged_count"),
                "late_convergence": result["late_convergence"],
                "wrote": str(OUTPUT),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
