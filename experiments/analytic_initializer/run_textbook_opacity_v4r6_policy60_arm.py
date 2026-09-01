"""Run one frozen arm of the matched v4r6 60-iteration development study."""

from __future__ import annotations

import argparse
import json

from experiments.analytic_initializer.run_h2_solver_funnel import main as funnel_main
from experiments.analytic_initializer.textbook_opacity_v4r6_policy60 import (
    ARM_OUTPUTS,
    ITERATIONS,
    PER_STAR_TIMEOUT_SECONDS,
    POLICY,
    SAMPLE,
    enforce_runtime_guard,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(ARM_OUTPUTS))
    args = parser.parse_args(argv)
    arm = str(args.arm)
    output = ARM_OUTPUTS[arm]
    signature = enforce_runtime_guard(arm)
    code = funnel_main(
        [
            "--arm",
            arm,
            "--count",
            "60",
            "--indices-from",
            str(SAMPLE),
            "--iterations",
            str(ITERATIONS),
            "--per-star-timeout",
            str(PER_STAR_TIMEOUT_SECONDS),
            "--resume",
            "--out",
            str(output),
        ]
    )
    if code != 0:
        return code
    result = json.loads(output.read_text(encoding="utf-8"))
    result["status"] = "policy60_matched_development_only"
    result["policy"] = POLICY
    result["runtime_signature"] = signature
    result["source_manifest"] = signature["source_manifest"]
    result["source_manifest_sha256"] = signature["source_manifest_sha256"]
    result["sample_manifest"] = signature["sample"]
    result["sample_manifest_sha256"] = signature["sample_sha256"]
    result["authorizes_fresh_open_execution"] = False
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "policy": POLICY,
                "arm": arm,
                "converged_count": result.get("converged_count"),
                "timeout_count": result.get("timeout_count"),
                "wrote": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
