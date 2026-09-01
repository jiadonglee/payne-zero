"""Run the preregistered v4r6 decoupled development-60 solver funnel.

Historical convective and grey development-60 JSON files are frozen controls
and are not rerun. This driver only sends the decoupled seed
(m_grey, T_conv, no mass re-integration) into the same 60-star list.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.analytic_initializer.discovery import file_sha256
from experiments.analytic_initializer.run_h2_solver_funnel import main as funnel_main
from experiments.analytic_initializer.textbook_opacity_v4r6_decoupled_gates import (
    CANDIDATE,
    development_gate,
    paired_solver_counts,
)
from experiments.analytic_initializer.write_textbook_opacity_v4r6_decoupled_source_manifest import (
    OUTPUT as SOURCE_MANIFEST,
)

INDICES_FROM = Path(
    "results/paper_physical_seed_20260820/learned/"
    "convergence_metrics_learned_monotone.json"
)
OUTPUT = Path(
    "results/analytic_initializer/textbook_opacity_v4r6_decoupled_dev60_20260828.json"
)
GREY_CONTROL = Path(
    "results/analytic_initializer/textbook_opacity_v4r6_grey_dev60_20260828.json"
)
CONVECTIVE_CONTROL = Path(
    "results/analytic_initializer/textbook_opacity_v4r6_dev60_20260828.json"
)
RUNTIME_GUARD = OUTPUT.with_suffix(".runtime.json")
SEED_AUDIT = Path(
    "results/analytic_initializer/textbook_opacity_v4r6_decoupled_seed_audit_20260828.json"
)


def _runtime_signature() -> dict[str, object]:
    missing = [
        str(path)
        for path in (SOURCE_MANIFEST, INDICES_FROM, GREY_CONTROL, CONVECTIVE_CONTROL)
        if not path.is_file()
    ]
    if missing:
        raise SystemExit("missing frozen artifacts: " + ", ".join(missing))
    return {
        "arm": "textbook_v4r6_decoupled",
        "candidate": CANDIDATE,
        "trials": 1,
        "iterations": 15,
        "per_star_timeout_seconds": 900,
        "indices_from": str(INDICES_FROM),
        "indices_from_sha256": file_sha256(INDICES_FROM),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": file_sha256(SOURCE_MANIFEST),
        "grey_control": str(GREY_CONTROL),
        "grey_control_sha256": file_sha256(GREY_CONTROL),
        "convective_control": str(CONVECTIVE_CONTROL),
        "convective_control_sha256": file_sha256(CONVECTIVE_CONTROL),
        "seed_audit": str(SEED_AUDIT) if SEED_AUDIT.is_file() else None,
        "seed_audit_sha256": file_sha256(SEED_AUDIT) if SEED_AUDIT.is_file() else None,
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
    grey = json.loads(GREY_CONTROL.read_text(encoding="utf-8"))
    convective = json.loads(CONVECTIVE_CONTROL.read_text(encoding="utf-8"))
    gate = development_gate(result, grey)
    result["status"] = "development_only"
    result["candidate"] = CANDIDATE
    result["decision"] = gate["decision"]
    result["development_gate"] = gate
    result["source_manifest"] = str(SOURCE_MANIFEST)
    result["source_manifest_sha256"] = signature["source_manifest_sha256"]
    result["sample_manifest"] = str(INDICES_FROM)
    result["sample_manifest_sha256"] = signature["indices_from_sha256"]
    result["runtime_signature"] = signature
    result["solver_policy"] = {
        "trials": 1,
        "iterations": 15,
        "per_star_timeout_seconds": 900,
    }
    result["frozen_controls"] = {
        "v4r6_convective": {
            "path": str(CONVECTIVE_CONTROL),
            "sha256": signature["convective_control_sha256"],
            "converged_count": convective.get("converged_count"),
        },
        "v4r6_grey": {
            "path": str(GREY_CONTROL),
            "sha256": signature["grey_control_sha256"],
            "converged_count": grey.get("converged_count"),
        },
    }
    result["paired_summary"] = {
        "vs_grey": gate["paired_vs_grey"],
        "vs_convective": paired_solver_counts(result["records"], convective["records"]),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    if argv:
        raise SystemExit("this driver pins its sample and output path")
    signature = _enforce_runtime_guard()
    code = funnel_main(
        [
            "--arm",
            "textbook_v4r6_decoupled",
            "--count",
            "60",
            "--indices-from",
            str(INDICES_FROM),
            "--per-star-timeout",
            "900",
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
                "converged_count": result.get("converged_count"),
                "cool": result["development_gate"]["cool"],
                "hot": result["development_gate"]["hot"],
                "paired_vs_grey": result["development_gate"]["paired_vs_grey"],
                "wrote": str(OUTPUT),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
