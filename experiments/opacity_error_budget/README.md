# Opacity error budget — the experiment that decides whether to build a κ_ν emulator

## Why this exists

The atmosphere solver spends **79.1%** of its wall time in the opacity stage
(`results/solver_in_loop_k1_hard5_linear/real_solver_comparison.json`: 5
iterations, 13.128 s total, opacity 10.379 s, transfer 0.327 s). That share sets
a hard Amdahl ceiling of **4.78×** on anything that only makes opacity cheaper —
an emulator included.

Before spending months on an emulator, two cheap experiments must run, because
they change its marginal value and its accuracy target:

- **A — frequency decimation.** `grid_size = 30000`
  (`payne_zero_atmosphere/continuum_opacity.py:6163`) is an equally-spaced
  log-wavelength opacity-sampling grid, `Δlog λ = 1e-4`, R ≈ 4340. Opacity cost
  is linear in this number, and decimation is the only lever that cuts both the
  continuum and the line halves of the 79%.
- **B — opacity lagging.** Opacity cost is flat from iteration 2 onward
  (1.853 / 1.847 / 1.868 / 1.874 s) while `deep_layer_relative_temperature_change`
  falls 3.075e-3 → 3.019e-4. The solver pays full price on iterations where the
  state barely moves.

A is a *measurement*, not an optimization. Its output is the number an emulator
must hit.

## Cost

12 stars × 4 strides = 48 solves.

The 13.128 s reference solve is a **cluster** number (96 cores). On the dev
machine the same work is 35–50 s per star, with the first iteration spending
~25 s in line selection and later iterations ~5 s (`bench/README.md:48-61`).
Cool stars run 18–30 iterations, so budget several times that for the cool
stratum. Estimate: **~25–40 minutes serial**, plus a one-time ~30 s catalog
load per process.

Peak RSS is ~7.8 GB per process, briefly ~14 GB at startup when
`line_selection.py:381` concatenates the three 1.4 GB catalog shards. Sizing is
`RAM_GB / 16`, so this 16 GB machine runs **exactly one worker** — never launch
a pool here. It still runs comfortably serially, so **experiment A does not need
the cluster.** Only the κ_ν corpus (`experiments/opacity_corpus/`) does.

One thing to watch while measuring: line selection maps lines into frequency
bins, so changing the stride may change the **iteration-1 selection cost** too,
not just the per-iteration accumulation. Report the two separately rather than
folding both into a single speedup number.

## The label set — and why it is stratified, not sampled

`labels.jsonl`, 12 stars, four strata of three. **Deliberately not a random
sample**: the error budget is regime-dependent, and an average over a random
draw would hide the one regime that matters most.

| stratum | stars | opacity character | expected decimation sensitivity |
| --- | --- | --- | --- |
| hot | 9000/4.0/0.0, 10000/4.5/−1.0, 8000/3.0/−2.0 | continuum-dominated (H, H⁻ bf/ff), smooth in ν | lowest |
| solar-type | 5777/4.44/0.0 (the Sun), 6200/4.2/−0.5, 5500/4.5/−1.5 | mixed atomic lines + continuum | low |
| giant | 4800/2.5/0.0, 4500/1.5/−0.5, 5000/2.0/−2.0 | heavily line-blanketed | moderate |
| cool | **4000/5.0/0.0** , 4200/1.5/0.0, 4000/2.0/−1.0 | molecular bands (TiO, H₂O), extreme ν structure | **highest** |

All 12 are inside the five-label support (4000–10500 K, logg 0.7–5.3,
[M/H] −2.5–0.5, [α/M] −0.1–0.5, ξ 0.5–4.0); verified programmatically.

The cool stratum's first entry is the **4000 K anchor of the existing cool-star
continuation ladder** (`results/cool_star_step_test_status.md`: logg 5.0,
[M/H] 0, [α/M] 0, ξ 1 km/s, stepping 4000 → 3750 → 3500 K). Reusing it means
the decimation result lands on the same star as the existing M-dwarf work
instead of a fresh point with no history.

## Protocol

For each star, for stride ∈ {1, 2, 4, 8}, record:

- iterations to convergence (cap and retry policy unchanged from production);
- final `deep_layer_relative_temperature_change` (production threshold `5e-4`);
- per-iteration `opacity_seconds` and total wall time;
- the converged `temperature` and `column_mass` arrays, kept for comparison
  against that star's own stride=1 result;
- the three spectral maxima (normalized / full flux / continuum) against the
  stride=1 spectrum.

Stride=1 must reproduce the production grid **bit-identically** — that is the
regression guard, not a tolerance.

## The decision criterion, and what it returned

Pre-registered before looking at the data: **let `s*` be the largest stride
whose converged atmosphere stays inside the solver's own start-dependence**,
with three anticipated outcomes — free 4x everywhere, a regime split, or
kappa_nu precision being genuinely load-bearing.

**The answer was a fourth outcome that the criterion did not anticipate: `s*`
is not a function of the regime at all.**

## Result (12 stars, `results/opacity_error_budget/ladder.json`)

| `s*` | stars |
| ---: | ---: |
| 1 | **6** |
| 2 | 5 |
| 4 | 1 |

Displacement grows close to linearly in the stride, and the slope is remarkably
uniform: **median 2.42e-4 per unit stride, range 1.16e-4 to 3.72e-4** — a factor
of 3.2 across 4000-10000 K, logg 1.5-5.0, and [M/H] -2.0 to 0.0. Divided into
the 5e-4 threshold this gives `s* ~ 2` at the median.

But the spread straddles the threshold, and **nothing in the labels predicts
which side a star lands on**. Spearman rank correlation of the slope against
each label axis:

| axis | rho | p |
| --- | ---: | ---: |
| effective temperature | −0.01 | 0.97 |
| log surface gravity | +0.40 | 0.20 |
| [M/H] | −0.07 | 0.82 |

All null. The stratification this file was built around — by effective
temperature — is not the axis, and neither is metallicity, which looked
promising at n=9 and died at n=12. The slope behaves like a constant of the
numerical scheme rather than a property of the star.

### Why that is decisive against decimation as a production lever

Stride 2 is free for 6 of 12 stars and not free for the other 6, **and there is
no way to tell in advance which**. Choosing a stride per star would therefore
require solving that star at stride 1 first to have something to compare
against — which costs more than the decimation saves. A single global stride of
2 would silently displace half of all atmospheres past the convergence
threshold.

**There is no free speedup here to subtract from an opacity emulator's value.**

### What this hands the emulator

A quantified target, which is what the experiment was for: an emulator must
displace the converged atmosphere by **less than 5e-4 in deep-layer relative
temperature**. It also inherits two advantages decimation does not have — its
error is tunable by training rather than fixed by the grid, and it removes the
line loop rather than the frequency axis, which is where the cost actually is.

## Two confounds that must be carried forward

**Iteration count is not stable under decimation.** 7 of 12 stars converged in a
different number of iterations at some stride, in both directions, including
down to the 3-iteration floor. Where that happens the measured displacement
mixes resolution loss with landing on a different point inside the fixed point's
own width. The slope medians of the changed and unchanged groups are
indistinguishable (2.45e-4 vs 2.39e-4), so this does not explain the result, but
it does blur individual stars.

**No timing from this campaign may be quoted.** The Sun was run twice with
bit-identical physics — same iteration counts, `deep_layer_relative_temperature_change`
equal to all digits, converged `temperature` and `column_mass` bit-for-bit
equal — and wall times 1.68-2.41x apart, the ratio itself varying with stride.
One star (r6) ran three iterations at strides 2, 4 and 8 and took *longer* at
stride 8 than at stride 2, which is not physically possible. A speed claim needs
a controlled re-run on the cluster: fixed warm state, repeated trials, median
and spread.

## Still missing

The pre-registered bar is **spectral** — the solver's own start-dependence is a
median normalized-spectrum `3.44e-3` — and this campaign only measured the
temperature yardstick. Closing that needs synthesis from each
`converged_state.npz`, which stores the reduced `(m, T)` state and so must go
through `reduced_state/reconstruct.py` first. Both arms take the same
reconstruction path, so its bias largely cancels in the difference. At ~50 min
per comparison on the cluster, scope it to **stride 2 only** (12 comparisons,
~10 h): strides 4 and 8 are already 1.2-5.5x over the temperature threshold and
do not need confirming.

## Companion experiments

| path | experiment |
| --- | --- |
| `experiments/opacity_decimation/` | A — grid stride implementation + runbook |
| `experiments/opacity_lagging/` | B — lagged opacity, verified, not yet merged |
| `experiments/opacity_corpus/` | C — kappa_nu corpus format, numerics tested |
