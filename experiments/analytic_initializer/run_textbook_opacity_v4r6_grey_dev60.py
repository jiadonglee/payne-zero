"""Run the preregistered v4r6-grey development-60 solver funnel.

The registered v4r6 grey-plus-adiabatic seed and the offline FAIL_STOP stay
frozen. This driver only sends Eddington-grey T plus the same v4r6 opacity ODE
mass into the same 60-star list.
"""

from __future__ import annotations

from pathlib import Path

from experiments.analytic_initializer.run_h2_solver_funnel import main as funnel_main


INDICES_FROM = Path(
    "results/paper_physical_seed_20260820/learned/"
    "convergence_metrics_learned_monotone.json"
)
OUTPUT = Path(
    "results/analytic_initializer/textbook_opacity_v4r6_grey_dev60_20260828.json"
)


def main(argv: list[str] | None = None) -> int:
    if argv:
        raise SystemExit("this driver pins its sample and output path")
    return funnel_main(
        [
            "--arm",
            "textbook_v4r6_grey",
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


if __name__ == "__main__":
    raise SystemExit(main())
