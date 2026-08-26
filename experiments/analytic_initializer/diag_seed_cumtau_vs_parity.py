"""Compare cumtau vs parity seed quality on solver-hard stars."""

from __future__ import annotations

import numpy as np

from .compact_initializer import (
    PARITY_CONFIGURATION,
    fit_compact_profile_parameters,
    predict_compact_reduced_state,
)
from .cumulative_tau_initializer import (
    fit_cumulative_tau_parameters,
    fit_oracle_targets,
    predict_cumulative_tau_state,
)
from .discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    load_strict_truth,
    make_split,
)
from .run_h2_solver_funnel import MANIFESTS


def main() -> int:
    corpus = load_strict_truth(DEFAULT_CORPUS)
    excluded, _ = collect_excluded_indices(MANIFESTS, corpus_size=corpus.size)
    split = make_split(corpus.size, excluded=excluded, seed=20260816)

    parity = fit_compact_profile_parameters(
        corpus, split, configuration=PARITY_CONFIGURATION
    )
    oracle = fit_oracle_targets(
        corpus.tau,
        corpus.temperature,
        corpus.column_mass,
        corpus.labels[:, 0],
        width=0.35,
    )
    cumtau = fit_cumulative_tau_parameters(
        corpus.labels,
        corpus.tau,
        oracle,
        split.train,
        degree=2,
        width=0.35,
        label_features_name="physical",
        support_indices=np.arange(corpus.size, dtype=np.int64),
    )

    def diagnose(name: str, mass, temperature, log_opacity) -> None:
        rel = np.abs(temperature / corpus.temperature[[i]] - 1.0)
        mdex = np.abs(np.log10(mass) - np.log10(corpus.column_mass[[i]]))
        kdex = np.abs(log_opacity - np.log10(corpus.rosseland_opacity[[i]]))
        print(name,
              "Trel p50/p95", np.round(np.percentile(rel, 50), 4), np.round(np.percentile(rel, 95), 4),
              "mdex p50/p95", np.round(np.percentile(mdex, 50), 4), np.round(np.percentile(mdex, 95), 4),
              "kdex p50/p95", np.round(np.percentile(kdex, 50), 4), np.round(np.percentile(kdex, 95), 4))
        print("  top T", round(float(temperature[0, 0])), "truth", round(float(corpus.temperature[i, 0])),
              "deep T", round(float(temperature[0, -1])), "truth", round(float(corpus.temperature[i, -1])))

    for i in (2891, 6896, 7811):
        mass, temperature, log_opacity = predict_compact_reduced_state(
            corpus.labels[[i]], corpus.tau, parity, check_support=False
        )
        diagnose("parity", mass, temperature, log_opacity)
        prediction = predict_cumulative_tau_state(
            corpus.labels[[i]], corpus.tau, cumtau, check_support=False
        )
        diagnose("cumtau",
                 prediction.column_mass,
                 prediction.temperature,
                 np.log10(prediction.opacity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
