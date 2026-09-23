# Interpolation baseline on development-60 — preregistration (2026-09-23)

## Purpose

The manuscript compares the learned two-field initializer with the six-field
initializer on development-60 (Sect. 5.2, `tab_learned_solver`). A survey user
who already holds a converged atmosphere grid would instead interpolate that
grid. This run supplies that baseline on the same 60 stars, with the same
solver settings, so that the manuscript can state how much of the learned
start's advantage survives against interpolation.

It is a baseline comparison, not a gate. No outcome changes the learned model,
the sample, the solver settings, or any other number in the manuscript.

## Sample

`results/reconstruction_metrics.json`, 60 stars, SHA-256
`042f4d33d9c1e971474e888148bb8648e2adb937365e99d9c162c8e7f6d85753`. These are
the development-60 indices of
`results/paper_physical_seed_20260820/learned/convergence_metrics_learned_monotone.json`
(verified identical, 60/60).

## Arms

Two new arms are solved. Both use the same donor pool, eight neighbours, and
inverse-distance-squared weights; they differ in what is interpolated and, as
implemented, in the temperature coordinate of the donor metric.

| arm | initial state |
|---|---|
| `interpolated_full_state` (I6) | all six stored fields interpolated in the encoded coordinates of `encode_interpolated_full_state`, decoded, and passed to the canonical deck parser; no physical reconstruction |
| `interpolated_grid` (I2) | only `log m` and `log T` interpolated, then the physical reconstruction of the learned arm (adaptive, at most 8 passes, `1e-3` dex pressure tolerance) |

Interpolation, fixed a priori and not tuned on this sample:

- metric labels `5040/Teff` (I6, `inverse_temperature_interpolation_coordinates`)
  or `log Teff` (I2, `interpolation_coordinates`), then `log g`, `[M/H]`, `[α/M]`,
  each scaled by its donor-pool standard deviation; microturbulence enters the
  deck, not the metric. The two temperature coordinates can select slightly
  different neighbours, so the I2–I6 comparison includes that difference;
- `k = 8` nearest donors, inverse-distance weights with power 2
  (`results/interpolated_grid_sensitivity_20260813.json` records why these
  values were fixed before any solve);
- donor pool: the 52,199-star corpus minus every evaluation, audit, calibration,
  and sealed manifest in `DEFAULT_DONOR_EXCLUDE`, 51,139 donors. The run aborts
  if any scored star is in the pool.

Reused, frozen arms (not re-solved):

- six-field: `runs/reduced_state_emulator/production_six_field/records.jsonl`;
- learned two-field: `runs/paper_physical_seed_20260820/learned/records/learned_reduced_state/records.jsonl`.

## Code identity control

The run uses the solver code already in the Garching tree
`/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero`, which produced the reused
arms; no solver file is synced to it, and the MD5 of every solver file is
recorded in `launch_status.json`. The six-field arm is re-solved on the same 60
stars with that code (`runs/interpolation_baseline_dev60_20260923/control/`).
If every re-solved converged/iteration outcome equals the frozen six-field
record, the frozen arms are used as described above. If any differs, the
comparison is not reported against the frozen arms; the difference is recorded
and the choice of reference is made in an amendment before the new arms are
examined.

## Solver settings

Identical to the reused arms, checked from their records: unchanged Payne Zero
solver, one trial, 15-iteration cap, minimum 3 iterations, deep-layer
`|ΔT/T| < 5e-4`, no retry. Workers run single-threaded
(`NUMBA_THREADING_LAYER=workqueue`, `NUMBA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`).

## Reported quantities

All are reported for both new arms whatever their values.

1. Converged count out of 60 with the Wilson 95% interval, and the failure type
   of every other star (reconstruction failure, non-convergence within 15,
   non-finite state, error).
2. Median, mean, and three-iteration floor count of the iterations of converged
   stars.
3. Paired iteration differences against the six-field and against the learned
   arm, on stars converged by both: faster/equal/slower counts and the Wilcoxon
   signed-rank test (`zero_method="wilcox"`, two-sided).
4. Geometric-mean contraction ratio `q`, non-monotonic fraction, and
   first-iteration residual, as in `tab_learned_solver`.
5. Converged spectra against the six-field reference over 400–900 nm at
   `R = 20000` in double precision: median and maximum per-star normalized-flux
   difference and the count above `5e-3`, with the same three quantities as
   `tab_spectral`. The reference is the frozen six-field product of each of
   the 59 converged six-field stars
   (`runs/paper_physical_seed_20260820/learned/products/production_six_field/`
   on Garching); its 56 already synthesized spectra
   (`runs/paper_physical_seed_20260820/learned/spectra/production_six_field/`)
   are reused and the other 3 are synthesized with the same settings. The one
   star without a converged six-field atmosphere is listed as excluded.
6. Seed-profile errors against the converged corpus atmosphere
   (`T` relative and `log m` dex, pointwise p95 and per-star maximum).

## Comparisons fixed in advance

- Co-primary, paired iterations (item 3):
  - I6 against the learned arm: the classical grid-interpolation baseline
    against the method of the paper;
  - I2 against the learned arm: both use the same reconstruction, so this
    separates the network from the two-field route. The I2 seed profiles are
    known to be less accurate than the learned ones on this sample (Prior
    exposure), so an equal iteration count would also test the paper's
    statement that profile accuracy alone does not set convergence.
- Secondary: I2 against I6, which measures what the two-field reconstruction
  does for interpolation itself.

How each outcome is written up:

- The manuscript reports I6 and I2 as two further rows of `tab_learned_solver`
  and two further curves of `fig_convergence`, whatever their values.
- The learned start is described as faster than interpolation only if its
  paired median saving against I6 is positive. If the saving is zero, the text
  says the learned start does not beat interpolation in iterations on this
  sample and states its remaining advantage as not needing a stored grid. If
  the saving is negative, the learned start is described as slower than
  interpolation and the speed claim of the paper is restricted to the
  six-field comparison.
- If I2 and the learned arm have a paired median difference of zero, the text
  attributes the speed of the learned start to the two-field route rather than
  to the network.

## Prior exposure

- Full-state interpolation (I6 settings) has been solved on two other samples:
  calibration-60 (`results/atmosphere_interpolation_benchmark_20260813/`,
  56/60 converged, mean 8.36 iterations; an earlier learned checkpoint had
  57/60 and 3.51, six-field 58/60 and 6.12) and the expanded 200-star
  calibration subset (`results/four_initializer_benchmark_expanded_20260814/`,
  190/200, mean 8.0). Both samples are disjoint from development-60.
- The I2 seed-profile accuracy on development-60 has been examined
  (`results/interpolated_grid_sensitivity_20260813.json`): at `k = 8`, power 2,
  the pointwise p95 errors are `3.12e-2` in `T` and `0.139` dex in `log m`,
  against `3.74e-3` and `0.0152` dex for the learned seeds. No I6 or I2 solver
  outcome on development-60 exists in the repository.
- Development-60 was used to select the learned model, so every comparison
  here is development evidence. The independent 200-star sample is not used.

## Run layout and compute

- Run root: `runs/interpolation_baseline_dev60_20260923/`
- Results: `results/interpolation_baseline_dev60_20260923/`
- 180 solves (I6, I2, and the six-field control, 60 stars each, at most 15
  iterations), then spectra for the converged I6 and I2 stars. On Garching
  Node-06 with 12 single-thread workers.
- Launcher: `experiments/reduced_state_emulator/run_interpolation_baseline_dev60_garching_20260923.sh`.

Solves, one per arm:

```bash
PYTHONPATH=. NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-linux/bin/python -m experiments.reduced_state_emulator.grey_start_benchmark \
  --arm interpolated_full_state \
  --manifest results/reconstruction_metrics.json \
  --workers 12 \
  --run-root runs/interpolation_baseline_dev60_20260923 \
  --result-root results/interpolation_baseline_dev60_20260923 \
  --interpolation-neighbours 8 --interpolation-power 2.0
```

and the same command with `--arm interpolated_grid`. Spectra use
`experiments.reduced_state_emulator.spectral_gate` with
`--baseline-arm production_six_field`, `--candidate-arm` set to each new arm,
400–900 nm, `--resolution 20000`, `--dtype float64`. The launcher and a
`launch_status.json` (host, directory, PIDs, logs, source hashes) are written
before the run starts.
