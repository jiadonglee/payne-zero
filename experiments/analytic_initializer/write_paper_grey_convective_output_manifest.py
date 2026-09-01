"""Write a complete SHA-256 inventory for the paper campaign outputs."""

from __future__ import annotations

import json
import platform
import socket
import sys
import time
from pathlib import Path

from experiments.analytic_initializer import (
    run_paper_grey_convective_campaign as campaign,
)


def write_manifest(root: Path) -> Path:
    root = root.resolve()
    result_root = root / "results" / campaign.CAMPAIGN
    run_root = root / "runs" / campaign.CAMPAIGN
    replay = json.loads(
        (result_root / "development/replay_check.json").read_text(encoding="utf-8")
    )
    if replay.get("matches") is not True:
        raise SystemExit("development replay is incomplete or does not match")
    for sample, expected in (("development", 60), ("posthoc200", 200)):
        solver = json.loads(
            (result_root / sample / "solver.json").read_text(encoding="utf-8")
        )
        if int(solver["star_count"]) != expected:
            raise SystemExit(f"{sample} has {solver['star_count']} rows, expected {expected}")
        for name in ("summary.json", "profile_metrics.json", "spectral_gate.json"):
            if not (result_root / sample / name).is_file():
                raise SystemExit(f"missing {sample}/{name}")

    output = result_root / "output_manifest.json"
    result_files = sorted(
        path
        for path in result_root.rglob("*")
        if path.is_file() and path != output
    )
    run_files = sorted(path for path in run_root.rglob("*") if path.is_file())
    payload = {
        "campaign": campaign.CAMPAIGN,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "source_manifest_sha256": campaign.sha256(
            result_root / "source_manifest.json"
        ),
        "inventory_source_sha256": campaign.sha256(Path(__file__).resolve()),
        "result_file_count": len(result_files),
        "run_file_count": len(run_files),
        "result_sha256": {
            str(path.relative_to(root)): campaign.sha256(path)
            for path in result_files
        },
        "run_sha256": {
            str(path.relative_to(root)): campaign.sha256(path)
            for path in run_files
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    print(write_manifest(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
