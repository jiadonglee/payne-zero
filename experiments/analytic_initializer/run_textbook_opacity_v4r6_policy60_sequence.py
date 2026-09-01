"""Run all matched v4r6 policy60 arms sequentially, then score them."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ARM_RUNNER = Path(
    "experiments/analytic_initializer/run_textbook_opacity_v4r6_policy60_arm.py"
)
SCORER = Path(
    "experiments/analytic_initializer/score_textbook_opacity_v4r6_policy60_dev60.py"
)
ARM_ORDER = (
    "textbook_v4r6_decoupled",
    "textbook_v4r6_grey",
    "textbook_v4r6",
)


def main() -> int:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."
    environment["NUMBA_THREADING_LAYER"] = "workqueue"
    for arm in ARM_ORDER:
        subprocess.run(
            [sys.executable, str(ARM_RUNNER), "--arm", arm],
            check=True,
            env=environment,
        )
    subprocess.run([sys.executable, str(SCORER)], check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
