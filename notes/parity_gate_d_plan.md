# Parity arm Gate D: spectra and expanded open sample

Status: preregistered before the Gate D spectral runs (2026-08-22).

## Candidate

The frozen parity compact-Chebyshev depth initializer
(`results/analytic_initializer/compact_profile_parameters_parity.npz`, 2407
stored floats), run with the product handoff enabled so the spectrum gate can
compare it against the frozen production baseline.

Gate C (development-60 solver funnel) is already satisfied on the parity
stream: 57/60 first-trial convergence, median iterations 8
(`results/analytic_initializer/paper_dev60_parity.json`). Product naming uses
the public solver t-format slug
(`t<Teff>_g<logg>_m<[M/H]>_a<alpha>_x<vt>`) so products pair by stem with the
production-baseline products; verified 200/200 against
`expanded200_manifest.json` `star_slugs`.

## Baseline

The unchanged frozen production six-field products
(`products/production_six_field/`). Only stars with a converged product on
**both** arms are gated; a star that failed one arm is reported as excluded,
not hidden.

## Gates

Both use the full-resolution 400-900 nm, R=20,000, float64 spectrum gate, all
three Ting metrics (normalized flux, total flux, continuum), bar = 5e-3.

1. Development-60: the preselected spectral-60 subset from the opened
   calibration set (`expanded200_manifest.json` `previous_60_indices`),
   equivalently `results/grey_start_benchmark_20260812/calibration_spectral60_manifest.json`.
2. Open-200: the expanded open benchmark (`expanded200_manifest.json`
   `star_indices`, 200 stars).

## Decision rule

- Pass: on the gated intersection, all three metrics have max over stars
  <= 5e-3 (no star above the 0.5% bar). This supports the paper claim that
  the analytic parity initializer is spectrally indistinguishable from the
  production baseline on the development sample (not a sealed holdout).
- Fail: any gated star above 5e-3 on any metric. Then exactly one bounded
  seam-continuation fix is permitted; after it, the same two gates rerun as
  the judgment, with no further tuning.

The sealed holdout (`results/sealed_initializer_holdout_20260812.json`) is not
opened. Paper wording changes only after a passing judgment and the user's
decision.

---

## Executed results (2026-08-22, recorded after the run)

Funnel (dev60 / previous_60, parity arm, per-star timeout 900 s):
- converged_count 57 / 60, first_trial_converged_count 57
- not_converged_indices [1025, 15630, 17444] (1025 = timeout; 15630, 17444 = not_converged)
- median iterations 8
- JSON: /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/results/analytic_initializer/gate_d_parity/solver_parity_development60.json

Spectral gate (dev60, 400-900 nm, R=20000, float64, bar 5e-3, baseline production_six_field):
- gated_star_count 57 (baseline 380 present, candidate 57, paired 57; 323 excluded = production-only, no candidate-only exclusions)
- normalized_flux: max 0.016909, median 0.004052, 24/57 above bar
- flux_total:      max 0.016805, median 0.004334, 24/57 above bar
- flux_continuum:  max 0.005030, median 0.001056,  1/57 above bar
- JSON: /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/results/analytic_initializer/gate_d_parity/spectral_gate_parity_vs_production_development60.json

VERDICT dev60: FAIL (any gated star above 5e-3 fails; 24/57 on two of three metrics).
The failure is systemic (median ~4e-3 on line fluxes), not seam-localized.

open200 gate (started detached, resumable, ~3-4 h funnel + 1-2 h spectra):
- funnel JSONL: /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/results/analytic_initializer/gate_d_parity/solver_parity_open200.jsonl
- funnel JSON:  .../solver_parity_open200.json
- gate JSON:    .../spectral_gate_parity_vs_production_open200.json
