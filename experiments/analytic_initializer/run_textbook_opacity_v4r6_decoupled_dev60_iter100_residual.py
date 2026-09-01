"""Re-solve the six residual decoupled stars with 100 solver iterations.

The 60-iteration diagnostic recovered 17 of 18 iteration-cap failures.
The six remaining stars were 900 s wall-clock timeouts, so a larger
iteration cap is meaningful only with a larger timeout. This driver does
not overwrite the 15-iteration FAIL_STOP or the 60-iteration JSON, and
it does not authorize fresh-open 120.
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
)
from experiments.analytic_initializer.write_textbook_opacity_v4r6_decoupled_source_manifest import (
    OUTPUT as SOURCE_MANIFEST,
)

RESIDUAL_INDICES = (6152, 33051, 33053, 44167, 46124, 48708)
OUTPUT = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r6_decoupled_dev60_iter100_residual_20260829.json"
)
ITER60_CONTROL = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r6_decoupled_dev60_iter60_20260829.json"
)
FIFTEEN_CONTROL = Path(
    "results/analytic_initializer/textbook_opacity_v4r6_decoupled_dev60_20260828.json"
)
RUNTIME_GUARD = OUTPUT.with_suffix(".runtime.json")
ITERATIONS = 100
PER_STAR_TIMEOUT_SECONDS = 3600
DECISION = "ITER100_RESIDUAL_DIAGNOSTIC_COMPLETE"


def _subset(
    records: list[dict[str, object]],
    indices: tuple[int, ...],
) -> list[dict[str, object]]:
    wanted = set(indices)
    selected = [item for item in records if int(item["corpus_index"]) in wanted]
    found = {int(item["corpus_index"]) for item in selected}
    missing = [index for index in indices if index not in found]
    if missing:
        raise SystemExit(f"control is missing residual indices {missing}")
    extra = sorted(found - wanted)
    if extra:
        raise SystemExit(f"control subset has unexpected indices {extra}")
    return selected


def _residual_outcome(candidate_records, control_records) -> dict[str, object]:
    candidate = records_by_index(candidate_records)
    control = records_by_index(control_records)
    recovered: list[int] = []
    still_failed: list[int] = []
    still_timeout: list[int] = []
    for index in RESIDUAL_INDICES:
        now = bool(candidate[index].get("converged"))
        if now:
            recovered.append(index)
        else:
            still_failed.append(index)
            if str(candidate[index].get("solver_outcome")) == "timeout":
                still_timeout.append(index)
    return {
        "residual_count": len(RESIDUAL_INDICES),
        "recovered_count": len(recovered),
        "still_failed_count": len(still_failed),
        "still_timeout_count": len(still_timeout),
        "recovered_indices": recovered,
        "still_failed_indices": still_failed,
        "paired_vs_60iter_residual": paired_solver_counts(
            candidate_records, control_records
        ),
    }


def _runtime_signature() -> dict[str, object]:
    missing = [
        str(path)
        for path in (SOURCE_MANIFEST, ITER60_CONTROL, FIFTEEN_CONTROL)
        if not path.is_file()
    ]
    if missing:
        raise SystemExit("missing frozen artifacts: " + ", ".join(missing))
    control = json.loads(ITER60_CONTROL.read_text(encoding="utf-8"))
    timeouts = sorted(
        int(item["corpus_index"])
        for item in control["records"]
        if str(item.get("solver_outcome")) == "timeout"
    )
    if timeouts != list(RESIDUAL_INDICES):
        raise SystemExit(
            "residual indices disagree with 60-iteration timeouts: "
            f"{timeouts} vs {list(RESIDUAL_INDICES)}"
        )
    return {
        "arm": "textbook_v4r6_decoupled",
        "candidate": CANDIDATE,
        "trials": 1,
        "iterations": ITERATIONS,
        "per_star_timeout_seconds": PER_STAR_TIMEOUT_SECONDS,
        "residual_indices": list(RESIDUAL_INDICES),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": file_sha256(SOURCE_MANIFEST),
        "iter60_control": str(ITER60_CONTROL),
        "iter60_control_sha256": file_sha256(ITER60_CONTROL),
        "fifteen_iter_control": str(FIFTEEN_CONTROL),
        "fifteen_iter_control_sha256": file_sha256(FIFTEEN_CONTROL),
        "frozen_15iter_decision": "FAIL_STOP_DEVELOPMENT",
        "frozen_60iter_decision": "ITER60_DIAGNOSTIC_COMPLETE",
        "authorizes_fresh_open": False,
        "timeout_raised_because_residuals_are_wallclock": True,
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
    sixty = json.loads(ITER60_CONTROL.read_text(encoding="utf-8"))
    fifteen = json.loads(FIFTEEN_CONTROL.read_text(encoding="utf-8"))
    sixty_residual = _subset(sixty["records"], RESIDUAL_INDICES)
    fifteen_residual = _subset(fifteen["records"], RESIDUAL_INDICES)
    residual = _residual_outcome(result["records"], sixty_residual)
    result["status"] = "development_only"
    result["candidate"] = CANDIDATE
    result["decision"] = DECISION
    result["authorizes_fresh_open"] = False
    result["frozen_15iter_decision"] = "FAIL_STOP_DEVELOPMENT"
    result["frozen_60iter_decision"] = "ITER60_DIAGNOSTIC_COMPLETE"
    result["residual_indices"] = list(RESIDUAL_INDICES)
    result["residual_outcome"] = residual
    result["source_manifest"] = str(SOURCE_MANIFEST)
    result["source_manifest_sha256"] = signature["source_manifest_sha256"]
    result["runtime_signature"] = signature
    result["solver_policy"] = {
        "trials": 1,
        "iterations": ITERATIONS,
        "per_star_timeout_seconds": PER_STAR_TIMEOUT_SECONDS,
    }
    result["frozen_controls"] = {
        "v4r6_decoupled_60iter": {
            "path": str(ITER60_CONTROL),
            "sha256": signature["iter60_control_sha256"],
            "decision": sixty.get("decision"),
        },
        "v4r6_decoupled_15iter": {
            "path": str(FIFTEEN_CONTROL),
            "sha256": signature["fifteen_iter_control_sha256"],
            "decision": fifteen.get("decision"),
        },
    }
    result["paired_summary"] = {
        "vs_60iter_residual": residual["paired_vs_60iter_residual"],
        "vs_15iter_residual": paired_solver_counts(
            result["records"], fifteen_residual
        ),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    if argv:
        raise SystemExit("this driver pins its residual indices, iterations, and output path")
    signature = _enforce_runtime_guard()
    code = funnel_main(
        [
            "--arm",
            "textbook_v4r6_decoupled",
            "--count",
            "6",
            "--indices",
            *[str(index) for index in RESIDUAL_INDICES],
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
                "residual_outcome": result["residual_outcome"],
                "wrote": str(OUTPUT),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
