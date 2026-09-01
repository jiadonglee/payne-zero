"""Write the WP0 source/environment manifest for the decoupled v4r6 candidate."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

from experiments.analytic_initializer.discovery import DEFAULT_CORPUS, file_sha256

CANDIDATE = "v4r6_decoupled_mgrey_tconv_v1"
RUN_DATE = "20260828"
OUTPUT = Path(
    f"results/analytic_initializer/textbook_opacity_v4r6_decoupled_source_manifest_{RUN_DATE}.json"
)

SOURCE_PATHS = (
    Path("experiments/analytic_initializer/textbook_opacity.py"),
    Path("experiments/analytic_initializer/run_h2_solver_funnel.py"),
    Path("experiments/analytic_initializer/textbook_opacity_v4r6_decoupled_gates.py"),
    Path(
        "experiments/analytic_initializer/"
        "write_textbook_opacity_v4r6_decoupled_source_manifest.py"
    ),
    Path("experiments/analytic_initializer/run_textbook_opacity_v4r6_decoupled_seed_audit.py"),
    Path("experiments/analytic_initializer/run_textbook_opacity_v4r6_decoupled_dev60.py"),
    Path("tests/test_textbook_opacity.py"),
    Path("tests/test_textbook_opacity_v4r6_decoupled.py"),
    Path("tests/test_analytic_initializer_multi_arm.py"),
    Path("notes/textbook_opacity_v4r6_decoupled_mgrey_tconv_workplan_20260828.md"),
    Path("notes/textbook_opacity_v4r6_decoupled_dev60_preregistration_20260828.md"),
)

INPUT_PATHS = (
    Path(
        "results/paper_physical_seed_20260820/learned/"
        "convergence_metrics_learned_monotone.json"
    ),
    Path(DEFAULT_CORPUS),
)

CONTROL_PATHS = (
    Path("results/analytic_initializer/textbook_opacity_v4r6_offline_validation_20260828.json"),
    Path("results/analytic_initializer/textbook_opacity_v4r6_dev60_20260828.json"),
    Path("results/analytic_initializer/textbook_opacity_v4r6_grey_dev60_20260828.json"),
)

DIFF_PATHS = (
    Path("experiments/analytic_initializer/textbook_opacity.py"),
    Path("experiments/analytic_initializer/run_h2_solver_funnel.py"),
    Path("experiments/analytic_initializer/textbook_opacity_v4r6_decoupled_gates.py"),
    Path(
        "experiments/analytic_initializer/"
        "write_textbook_opacity_v4r6_decoupled_source_manifest.py"
    ),
    Path("experiments/analytic_initializer/run_textbook_opacity_v4r6_decoupled_seed_audit.py"),
    Path("experiments/analytic_initializer/run_textbook_opacity_v4r6_decoupled_dev60.py"),
    Path("tests/test_textbook_opacity.py"),
    Path("tests/test_textbook_opacity_v4r6_decoupled.py"),
    Path("tests/test_analytic_initializer_multi_arm.py"),
    Path("notes/textbook_opacity_v4r6_decoupled_mgrey_tconv_workplan_20260828.md"),
    Path("notes/textbook_opacity_v4r6_decoupled_dev60_preregistration_20260828.md"),
)


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _hash_existing(paths: tuple[Path, ...]) -> dict[str, str | None]:
    hashed: dict[str, str | None] = {}
    for path in paths:
        hashed[str(path)] = file_sha256(path) if path.is_file() else None
    return hashed


def _git_diff_sha256() -> str:
    existing = [str(path) for path in DIFF_PATHS if path.exists()]
    tracked = subprocess.run(
        ["git", "diff", "HEAD", "--", *existing],
        check=True,
        capture_output=True,
    )
    untracked = bytearray()
    for path in DIFF_PATHS:
        status = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            check=False,
            capture_output=True,
        )
        if status.returncode != 0 and path.is_file():
            untracked.extend(f"untracked {path}\n".encode("utf-8"))
            untracked.extend(path.read_bytes())
    digest = hashlib.sha256()
    digest.update(tracked.stdout)
    digest.update(bytes(untracked))
    return digest.hexdigest()


def build_source_manifest() -> dict[str, object]:
    try:
        import numpy
    except ImportError:  # pragma: no cover
        numpy = None
    try:
        import numba
    except ImportError:  # pragma: no cover
        numba = None
    head = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--short"])
    branch = _git(["branch", "--show-current"])
    return {
        "candidate": CANDIDATE,
        "run_date": RUN_DATE,
        "git_head": head or None,
        "git_branch": branch or None,
        "git_dirty": bool(status),
        "git_status_short": status.splitlines(),
        "git_diff_sha256": _git_diff_sha256(),
        "hostname": socket.gethostname(),
        "operating_system": platform.platform(),
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "numpy": None if numpy is None else numpy.__version__,
        "numba": None if numba is None else numba.__version__,
        "environment": {
            "NUMBA_THREADING_LAYER": os.environ.get("NUMBA_THREADING_LAYER"),
            "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
        },
        "source_sha256": _hash_existing(SOURCE_PATHS),
        "input_sha256": _hash_existing(INPUT_PATHS),
        "control_result_sha256": _hash_existing(CONTROL_PATHS),
    }


def write_source_manifest(path: Path = OUTPUT) -> dict[str, object]:
    payload = build_source_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    payload = write_source_manifest()
    print(json.dumps({"wrote": str(OUTPUT), "git_head": payload["git_head"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
