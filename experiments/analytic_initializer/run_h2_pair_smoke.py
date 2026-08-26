"""Compare the H2 analytic smoke set with the current production initializer."""

from __future__ import annotations

import json
from pathlib import Path

# Configure Numba before importing the reference benchmark.
from bench import environment as _environment  # noqa: F401

from bench.labels import StellarLabels
from bench.run_reference import run_star


SMOKE_PATH = Path("results/analytic_initializer/h2_solver_smoke12.json")
OUTPUT = Path("results/analytic_initializer/h2_pair_smoke12.json")


def main() -> None:
    payload = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
    records = []
    for item in payload["records"]:
        labels = StellarLabels(
            effective_temperature=float(item["effective_temperature"]),
            log_surface_gravity=float(item["log_surface_gravity"]),
            metallicity=float(item["metallicity"]),
            alpha_enhancement=float(item["alpha_enhancement"]),
            microturbulence_km_s=float(item["microturbulence_km_s"]),
        )
        result = run_star(labels, iterations_per_trial=15, max_trials=2)
        record = result.as_json()
        record["corpus_index"] = int(item["corpus_index"])
        print(json.dumps(record, sort_keys=True), flush=True)
        records.append(record)

    result = {
        "candidate": payload["candidate"],
        "comparison": "current_production_emulator_initializer",
        "records": records,
        "production_converged_count": int(sum(bool(item["converged"]) for item in records)),
        "production_finite_count": int(
            sum(
                bool(item["converged"])
                and all(
                    trial.get("converged", False) for trial in item["trials"] if trial.get("converged", False)
                )
                for item in records
            )
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("production_converged_count", "production_finite_count")}, sort_keys=True))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
