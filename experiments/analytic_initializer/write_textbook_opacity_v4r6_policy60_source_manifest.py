"""Write the frozen source manifest for the matched v4r6 policy60 study."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experiments.analytic_initializer.discovery import file_sha256
from experiments.analytic_initializer.textbook_opacity_v4r6_policy60 import (
    POLICY,
    SAMPLE,
    SOURCE_MANIFEST,
)

SOURCE_PATHS = (
    Path("experiments/analytic_initializer/textbook_opacity.py"),
    Path("experiments/analytic_initializer/run_h2_solver_funnel.py"),
    Path("experiments/analytic_initializer/textbook_opacity_v4r6_decoupled_gates.py"),
    Path("experiments/analytic_initializer/textbook_opacity_v4r6_policy60.py"),
    Path("experiments/analytic_initializer/run_textbook_opacity_v4r6_policy60_arm.py"),
    Path(
        "experiments/analytic_initializer/"
        "run_textbook_opacity_v4r6_policy60_sequence.py"
    ),
    Path("experiments/analytic_initializer/score_textbook_opacity_v4r6_policy60_dev60.py"),
    Path(
        "experiments/analytic_initializer/"
        "write_textbook_opacity_v4r6_policy60_source_manifest.py"
    ),
    Path("tests/test_textbook_opacity_v4r6_decoupled.py"),
    Path("tests/test_textbook_opacity_v4r6_policy60.py"),
    Path(
        "notes/"
        "textbook_opacity_v4r6_policy60_matched_dev60_preregistration_20260829.md"
    ),
)


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    return result.stdout.strip()


def _diff_hash() -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *[str(path) for path in SOURCE_PATHS]],
        check=True,
        capture_output=True,
    )
    digest = hashlib.sha256()
    digest.update(result.stdout)
    for path in SOURCE_PATHS:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            check=False,
            capture_output=True,
        )
        if tracked.returncode != 0:
            digest.update(f"untracked {path}\n".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    missing = [str(path) for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise SystemExit("missing policy60 sources: " + ", ".join(missing))
    payload = {
        "policy": POLICY,
        "run_date": "20260829",
        "git_head": _git(["rev-parse", "HEAD"]) or None,
        "git_branch": _git(["branch", "--show-current"]) or None,
        "git_dirty": bool(_git(["status", "--short"])),
        "git_diff_sha256": _diff_hash(),
        "source_sha256": {
            str(path): file_sha256(path)
            for path in SOURCE_PATHS
        },
        "input_sha256": {
            str(SAMPLE): file_sha256(SAMPLE),
        },
    }
    SOURCE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "policy": POLICY,
                "source_count": len(SOURCE_PATHS),
                "wrote": str(SOURCE_MANIFEST),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
