"""Score the frozen matched v4r6 policy60 development study."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.analytic_initializer.discovery import file_sha256
from experiments.analytic_initializer.textbook_opacity_v4r6_policy60 import (
    ARM_OUTPUTS,
    POLICY,
    SOURCE_MANIFEST,
    continuation_gate,
)

OUTPUT = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r6_policy60_matched_dev60_20260829.json"
)


def _load_arm(arm: str) -> dict[str, object]:
    path = ARM_OUTPUTS[arm]
    if not path.is_file():
        raise SystemExit(f"missing policy60 arm {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("policy") != POLICY:
        raise SystemExit(f"{path} has wrong policy {payload.get('policy')!r}")
    return payload


def main() -> int:
    decoupled = _load_arm("textbook_v4r6_decoupled")
    grey = _load_arm("textbook_v4r6_grey")
    convective = _load_arm("textbook_v4r6")
    signatures = [
        dict(payload.get("runtime_signature") or {})
        for payload in (decoupled, grey, convective)
    ]
    invariant_keys = (
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
    reference = {key: signatures[0].get(key) for key in invariant_keys}
    for signature in signatures[1:]:
        current = {key: signature.get(key) for key in invariant_keys}
        if current != reference:
            raise SystemExit("policy60 arm runtime signatures are not matched")
    gate = continuation_gate(decoupled, grey, convective)
    result = {
        "policy": POLICY,
        "status": "matched_development_characterization_complete",
        "decision": gate["decision"],
        "authorizes_fresh_open_execution": False,
        "gate": gate,
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": file_sha256(SOURCE_MANIFEST),
        "runtime_invariants": reference,
        "arm_artifacts": {
            arm: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for arm, path in ARM_OUTPUTS.items()
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "checks": gate["checks"],
                "wrote": str(OUTPUT),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
