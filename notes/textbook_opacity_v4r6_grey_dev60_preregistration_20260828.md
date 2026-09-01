# Textbook opacity v4r6-grey development-60 solver funnel

Date: 2026-08-28

This is a user-requested solver-basin diagnostic. It does not reopen or
rewrite the v4r6 offline `FAIL_STOP`. It does not change the registered
v4r6 grey-plus-adiabatic seed, the production solver, the default
initializer, or the sealed holdout. Mean iteration count is reported and
is not a gate.

## Why this task exists

Three stacked reasons explain why cool-star funnel progress stayed at
zero after v4r5 and v4r6:

1. Those opacity repairs act at `T >= 8000 K`. Cool 3200--4000 K
   Rosseland is H-minus dominated (sensitivity ~0.93); hydrogen
   bound-free sensitivity is ~5e-6. v4r6 is bit-identical to v4r3 on
   that layer, so the offline cool-mass p95 (v4r2 0.234 → v4r6 0.247)
   could not move.
2. The formal domain was raised to 4000 K in v4r2, but the mass ODE
   still integrates from the surface. Cool-star median ~31% (p90 ~63%)
   of layers sit below the floor.
3. **Today's finding, largest weight:** the registered seed's dead
   zone is the adiabatic *T* replacement, not only κ. On v4r6 seeds
   versus stored truth, dropping that replacement collapsed cool deep
   (`τ > 10`) log-mass error from ~0.80 dex to ~0.086 dex. Hot stars
   are the opposite: pure grey is already close in the deep radiative
   interior; the replacement inflates deep *T* error, but the basin is
   wide enough that they still converge.

The in-progress registered v4r6 funnel (grey plus adiabatic, same 60
indices) is the paired control. At 57/60 it already showed 0/26 below
7500 K and 19/31 at or above 7500 K, matching v4r3's 0/27 cool. This
task turns convection off and asks whether any cool star enters the
basin.

Target (b), the H-minus implementation-level repair, is **not** this
task.

## Frozen construction

Keep the v4r6 opacity law unchanged. The only difference from the
registered v4r6 seed is `include_convection=False`:

```text
T  = Eddington-grey, no adiabatic replacement
m  = RK4 integral of dm/dtau = 1 / kappa_v4r6(T_grey, g m), 8 substeps
kappa_seed = kappa_v4r6 evaluated at that (T_grey, P=g m)
```

This is a basin diagnostic, not a claim that cool stars should be
radiative. The registered v4r6 arm stays grey-plus-adiabatic.

## Registered sample and solver policy

Same frozen paper development-60 indices as analytic-parity, the v4r3
funnel, and the registered v4r6 funnel:
`results/paper_physical_seed_20260820/learned/convergence_metrics_learned_monotone.json`.
Do not draw a new funnel.

Solver policy matches the formula arms:

- one trial, 15 iterations, 900 s per-star timeout;
- seed the unchanged production solver through `analytic_seed_model`;
- stream JSONL as each star lands; resume is allowed;
- no spectral gate, no 12-star smoke requirement, no sealed holdout.

Do not start this run until the registered v4r6 funnel JSON is on disk,
so the two 60-star solves do not share a node.

Paired control, now on disk:

- `results/analytic_initializer/textbook_opacity_v4r6_dev60_20260828.json`
- SHA-256 `c0c08c9727e522916085941bc5dcb40a96d67fea05852f8d88ddb4cae4cdd3e5`
- 20/60 converged, 0/27 below 7500 K, 20/33 at or above 7500 K, 3 timeouts
- `include_convection: true`

## Reported quantities, not gates

Record, and do not promote into a pass/fail cutoff:

- seed construction finite count;
- seed versus stored truth, split `Teff < 7500 K` versus `>= 7500 K`,
  and split all layers versus `τ > 10`: relative *T* p50/p95 and
  `|Δ log10 m|` p50/p95;
- converged / timeout / error counts on the same Teff split;
- iterations on converged stars, including the mean.

A non-zero cool-star convergence count is the quantity this diagnostic
is for. It does not by itself promote grey-only *T* into the registered
seed. Production remains the default initializer.

## Stop rule for this task

This task stops after the development-60 funnel JSON is written. It does
not run flux/spectral checks, the 200-star sealed holdout, a production
switch, or the H-minus implementation repair.

Output:
`results/analytic_initializer/textbook_opacity_v4r6_grey_dev60_20260828.json`

Runner:
`experiments/analytic_initializer/run_textbook_opacity_v4r6_grey_dev60.py`

Log: `logs/textbook_opacity_v4r6_grey_dev60.log`

## Remote execution

Host `astronode-garching`. Checkout
`/home/jdli/xiasangju/jdli/payne-zero`. Python `.venv-linux/bin/python`.
Do not evaluate production opacity from macOS `.venv`.

## Post-run (2026-08-28)

JSON SHA-256
`caeee639e37600952be8439d259bdb99f68d992bf9d4c2a50749530f68bf015a`

The diagnostic quantity moved: cool-star convergence is no longer zero.

| Split | registered v4r6 (convection on) | v4r6-grey (convection off) |
|---|---:|---:|
| All 60 | 20/60 | **37/60** |
| Teff < 7500 K | **0/27** | **6/27** |
| Teff >= 7500 K | 20/33 | **31/33** |
| Timeouts | 3 | 3 |
| Mean iterations (converged) | 10.7 | 10.3 |

All 60 grey seeds finite. No registered convergent star was lost. The 17
gains are 6 cool plus 11 stars in 7770--9370 K. The two persistent hot
timeouts are the same indices (`6152`, `48708`).

Grey seed versus stored truth, `|Δ log10 m|` p50:

- cool, `τ > 10`: 0.094 dex (all layers 0.113)
- hot, `τ > 10`: 0.035 dex (all layers 0.151)

Paired convective seed on the same 60 indices (recomputed locally, not
stored in the registered funnel JSON):

- cool, `τ > 10`: mass 0.719 → grey 0.094; relative *T* 0.309 → 0.678
- hot, `τ > 10`: mass 0.142 → 0.035; relative *T* 0.249 → 0.002

Cool deep *T* is still poor: grey *T* is not a physical convective
interior. The basin opened because the mass column stopped being
destroyed. This does **not** promote grey-only *T* into the registered
seed. Production remains the default initializer. H-minus
implementation repair was not run.
