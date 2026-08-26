"""Small read-only progress summaries for cool-star pilot logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def summarize(path: Path) -> dict[str, list[int]]:
    attempts: dict[str, list[int]] = defaultdict(list)
    current_attempt = 0
    current_iteration = None
    with Path(path).open() as handle:
        for line in handle:
            marker = "iteration "
            if marker not in line:
                continue
            iteration_text = line.split(marker, 1)[1].split("/", 1)[0]
            iteration = int(iteration_text)
            if current_iteration is None:
                current_attempt += 1
            elif iteration <= current_iteration:
                current_attempt += 1
            current_iteration = iteration
            attempts[f"attempt_{current_attempt:02d}"].append(iteration)
    return dict(attempts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    for attempt, iterations in summarize(args.log).items():
        status = "running"
        if len(iterations) >= 30:
            status = "capped"
        print(f"{attempt}\tmax_iteration={max(iterations)}\tcount={len(iterations)}\t{status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
