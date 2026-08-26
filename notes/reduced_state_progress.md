# Reduced-state `(m,T)(τ)` initialization — progress and future plan

> Live status companion to [`reduced_state_existing_work.md`](reduced_state_existing_work.md)
> (the archaeology/what-exists document) and the approved plan at
> `~/.claude/plans/you-are-working-inside-stateful-kurzweil.md`. This file
> tracks what has actually been run, with numbers, and what is planned next.
> Follows the same plan/progress split as `solver-in-the-loop-plan.md` /
> `solver-in-the-loop-progress.md`.

**Status as of 2026-08-10: the reconstruction repair and standalone
default-trajectory implementation are complete.** The frozen regional candidate
passes the 60-star development gate but fails the newly opened 200-star audit
with edge-region profile blowouts. The earlier accuracy and spectral numbers
below describe superseded models. The current production model is unchanged.

**Historical baseline (superseded for acceptance):** the reduced state won on
every axis measured in the old pipeline. Predicting two fields and
deriving the other four beats the shipped six-field network **on all six
fields** (2.6–5.6× on the derived four, and on m and T themselves). Discarding
the four fields and rebuilding them from truth `(m,T)` changes the emergent
spectrum by a median 8e-5 — 0 of 60 stars over the 5e-3 gate. On the solver the
learned initializer converges in 4.07 iterations against production's 5.59, at a
cost of 2 extra failures in 60 that is not statistically resolved (Fisher
p = 0.62). Parts 6–9 are superseded by the repair work below.

The shipped five-label six-field checkpoint is not a “v1.1 checkpoint” as an
older draft of this note said. Its release metadata identifies
`payne_zero_complete_atmosphere_latent_v2`, trained from the 52,199-row corpus
with 50,000 fit rows, 2,000 fit-validation rows, and 199 internal-check rows.
The six-field network is a product comparison only; it overlaps 55/60 stars in
the current development cohort and is not an independent validation set.

The repair fixed the pressure/field pass alignment, added adaptive pressure
synchronization (up to 8 passes, `1e-3` dex tolerance), stabilized the `(m,T)`
coordinates with grey-temperature residuals and positive mass increments, and
added the four-arm comparison and SHA provenance. A labels-only regional
composition then passed the complete 60-star profile/solver/spectral goal, but
failed the 200-star audit on unseen label edges. Therefore **no new model
replaces the current product**. The preselected 60-star solver subset was
consumed for the real solver/spectral goal; the audit diagnostic has been
recorded and a fresh holdout is required for any next correction. Fast tests
pass (`8 passed, 1 skipped`), and the one-star real solver chain passes.

Figures: `figures/reduced_state_sufficiency.png` (six-parameter),
`figures/spectral_sufficiency.png` (spectral), `figures/learned_vs_production.png`
(convergence), `figures/residual_scale.png` (why the oracle arms are excluded
from the contraction comparison), `figures/spectral_gate.png` (per-star detail),
`figures/spectral_gate_default_pull05_g32_full60.png` (the old diagnostic
correction), `figures/spectral_gate_default_trajectory_alpha100_6x1024.png`
(the current candidate), and `figures/field_consistency_comparison.png` (the
repaired four-arm check).

## Regional surface5 audit and retrain scope — 2026-08-10

The regional composer now supports alpha and microturbulence bounds and the
literal training-time `solver_tail` union (`logg<3.2 OR vmic>3.0 OR alpha>0.4`).
The reconstructed frozen candidate reproduces the correct development manifest
(`sealed_solver_subset_20260808.json`) at T/m p95 `2.259e-3` / `1.136e-2` dex
with no blowouts. On the opened audit its baseline is `2.560e-3` / `1.595e-2`,
with 1 temperature and 8 mass blowouts.

The solver-tail surface5 override selects 52/60 development and 160/200 audit
stars. Development p95 improves to `1.956e-3` / `9.601e-3`, but one mass
blowout is introduced, so no solver/spectral run was launched. Audit p95 is
`2.620e-3` / `1.495e-2`; temperature stays at 1 blowout and mass rises from 8
to 11 (2 fixed, 5 newly introduced). The complete chain and hashes are in
`results/goal_regional_20260810/frozen_candidate_policy.json`.

`reconstruction_metrics.json` is not this development set. The next full
retrain is scoped to the existing solver-tail/surface loss with greater
capacity and stronger regional weighting, excluding
`sealed_solver_subset_20260808.json`, `sealed_audit_20260808.json`, and
`sealed_audit_20260811.json`. The fresh holdout remains unopened and is
reserved for one final check after all earlier profile, solver, and spectral
gates pass.

## Three-way-excluded full retrain — 2026-08-11

The planned larger solver-tail model completed 500 CPU epochs: 8×1024,
hard weight 20, surface weight 10, seed 20260808, with the development-60,
opened audit-200, and fresh holdout-200 manifests excluded from fitting. Its
training validation loss reached `8.642e-4`, but external profile accuracy did
not improve robustly. Development T/m p95 is `2.119e-3` / `1.283e-2` dex with
one mass blowout; opened-audit T/m p95 is `2.555e-3` / `1.677e-2` dex with
3 temperature and 8 mass blowouts. It therefore did not proceed to the solver
or spectral gates. The fresh holdout remains unopened. This result closes the
"more capacity + stronger broad-region weighting" lever; the next work returns
to completing the differentiable forward twin and training on solver-aligned
losses rather than profile loss alone.

### First solver-in-loop training result — 2026-08-11

The first K=1 training loop is now executable. Continuum and transfer are
verified differentiable twins; the 24.1-million-record Sun line catalog and
the correction/remap use compact certified forward templates with local
surrogate gradients. End-to-end K=1 gradients are finite and nonzero, and all
88 tests pass (11 skipped).

On five difficult development stars, a label-conditioned adapter changed the
real solver from `4/5` converged at 6.5 mean iterations (converged stars) to
`5/5` at 6.2. This is the first measured real-solver improvement produced by
a solver-in-loop gradient in this project. It is not yet a deployable model:
the global linear adapter badly failed the dev-60 profile gate, while a local
RBF adapter still left star 42147 at 0.364 dex mass error. The immediate next
step is profile-constrained/local K=1 training on a broader development cohort;
the fresh 2026-08-11 holdout remains unopened.

## Frozen development candidate and sealed audit failure — 2026-08-10

The first labels-only regional composition passed the full 60-star development
goal. It uses a low-gravity alpha-0.5 default-trajectory base, a validation-
selected hot/metal-poor regional arm, a refined hot-warm arm, and a narrow
cool metal-poor edge arm. The inference path uses only labels and all decoded
mass profiles remain strictly monotonic. The frozen policy and checkpoint
hashes are recorded in
`results/goal_spectral_20260808/frozen_lowg_alpha050_refined_hotpoor_alpha100_policy.json`.

| check | result |
| --- | --- |
| profile T p95 / m p95 | `2.259e-3` / `1.136e-2` dex |
| profile pointwise blowouts | `0/60` in T; `0/60` in m |
| real solver | `59/60`; mean `3.95`, p90 `6`, non-monotonic `21.7%` |
| spectral pairs | `59`; one solver failure excluded |
| normalized / total / continuum median | `1.230e-3` / `1.241e-3` / `4.253e-4` |
| stars above `5e-3` | `0` in all three metrics |

The frozen 200-star audit then failed before the real-solver stage: one
temperature blowout and nine mass blowouts. The failures cluster in label
edges not represented by the development 60, especially `T≈7600–8200 K`,
`logg≈2–3`, high alpha/microturbulence, plus one hot high-gravity star. The
audit report is `results/goal_spectral_20260808/audit_20260810/profile_audit200.json`.
The production model remains unchanged. The 200-row audit is no longer an
untouched independent test after this diagnostic was opened; the remaining
140 rows were not used for fitting. The table-surface-loss solver-tail
diagnostic completed at 300 epochs. On the opened audit it reduced mass p95
from `1.531e-2` to `1.494e-2` dex, but increased mass blowouts from 9 to 12;
used globally it also worsened the development mass p95 to `3.551e-2` dex.
It is therefore not a replacement model. A fresh 200-star holdout was sealed
as `results/sealed_audit_20260811.json`, excluding both the development set and
the opened audit; its results remain unopened. The next correction must be a
label-defined regional use of the surface model, followed by a full retrain
that excludes the fresh holdout.

## Follow-up solver/spectral correction — 2026-08-09

The pure new physical ensemble from the previous section is still the current
production candidate and remains unchanged. It failed the development goal at
58/60. A diagnostic correction was then tested: in the low-gravity region
`logg < 3.2`, blend the new network's log-space `(m,T)` with the shipped
six-field initializer's own `(m,T)` at weight `alpha=0.5`. This is a
default-trajectory calibration probe, not an independent new model; it is kept
separate so the six-field product is not mistaken for truth.

The 7-star probe passed all three spectral metrics on its 6 paired stars. The
full sealed 60-star solver run used a 16-pass pressure-synchronization
diagnostic limit because one star was still at `1.599e-3 dex` after pass 8.
The pure reconstruction code still has the planned 8-pass limit; this
correction branch therefore does **not** yet replace it as the production
path. A smaller `alpha=0.4` probe still missed the 8-pass limit at
`1.308e-3 dex`, so this is not just a threshold-rounding issue.

| check | result |
| --- | --- |
| candidate solver | `59/60` converged; only `t04995.8_g+2.97_m+0.19_a-0.04_x3.48` failed |
| iterations | mean `3.627`; p90 `5`; max `8` |
| non-monotonic trajectories | `13.3%` |
| spectral pairs | `59`; the failed solver star is excluded |
| normalized flux | median `1.126e-3`; max `5.009e-3`; 1 over `5e-3` |
| total flux / continuum | medians `1.070e-3` / `5.128e-4`; 0 over `5e-3` in each |
| spectral acceptance | passes the stated median `≤1.62e-3` and `≤1` over-bar-star limits; strict zero-over-bar gate is false |

The one normalized-flux over-bar star is
`t05283.1_g+2.31_m-2.25_a-0.06_x1.23`, at `5.0089e-3`. The correction probe
artifacts are `results/goal_spectral_20260808/predicted_default_pull05_g32_probe60.npz`,
`results/goal_spectral_20260808/default_pull05_g32_full60/convergence_metrics_learned_physical_ensemble.json`,
and `results/goal_spectral_20260808/default_pull05_g32_full60_spectral_gate.json`.
The plot is `figures/spectral_gate_default_pull05_g32_full60.png`.

SHA256 provenance: prediction `1e12a2d1b31d9a3a4928bab69d75a5b01c01483030765fe139b98c98d3204a47`,
solver summary `1251ac0a33425e1e4ec8e9d69f619d455f19f367dd073b8590c1826ca0d21382`,
gate `59c0cc9f6fcd471e99a3db4385979d1491c37ae86626da874c1bd51e062b7037`,
figure `612df1cf4cb912ae9406265b1f6f1057ea10d56ed65ab8e78bd28e2e2c878d89`.

The sealed 200-star audit and its frozen 60-star real-solver/spectrum test
have not been opened or run. The current product is therefore not replaced;
the next clean step is to remove the six-field runtime correction by training
or calibrating a standalone `(m,T)` model against the default trajectory, then
repeat this same sealed protocol.

## Standalone default-trajectory model — 2026-08-09

The first clean standalone calibration used three 4×512 models
(`20260807–20260809`). During training only, the target was a log-space blend
with the shipped six-field initializer in the restricted region
`logg < 2.5` and `[M/H] < -1.2`; inference uses labels → `(m,T)` only. The
prediction is strictly monotonic and all 60 stars synchronized within the
planned eight-pass pressure limit.

This version improves the solver basin but does not yet meet the full goal:

| check | result |
| --- | --- |
| real solver | `59/60` converged; only the known `t04995.8_g+2.97_m+0.19_a-0.04_x3.48` failed |
| iterations | mean `3.29`; p90 `4`; non-monotonic `16.7%` |
| profile p95 | T `3.440e-2`; m `2.655e-2` dex; worst m `0.271` dex at `t07650.8_g+1.79_m-1.93_a+0.36_x2.06` |
| normalized flux | median `1.548e-3`; 3/59 above `5e-3` |
| total flux | median `1.552e-3`; 2/59 above `5e-3` |
| continuum | median `4.650e-4`; 0/59 above `5e-3` |

The three normalized-flux outliers are
`t04058.7_g+3.10_m-1.71_a-0.04_x1.64`,
`t04531.6_g+1.43_m-2.32_a+0.16_x0.58`, and
`t08425.4_g+1.06_m-0.84_a+0.47_x1.68`. Thus the median spectrum is close,
but this model is not yet “consistent with payne-zero default” under the
stated gate, and it is not the replacement model.

Artifacts:
`results/goal_spectral_20260808/predicted_default_trajectory_selective_probe60.npz`,
`results/goal_spectral_20260808/default_trajectory_selective_full60/convergence_metrics_learned_physical_ensemble.json`,
`results/goal_spectral_20260808/default_trajectory_selective_spectral_gate.json`,
and `results/default_trajectory_target_selective_20260809.npz`.
Their SHA256 values are, respectively,
`b4fb596a88a64e6b7342691973d1e64f04620dec67fa5abae5784727264f02cc`,
`8338155a194626619d95c750654d43de5f55aab3d7b0a6401b761458b3ad0748`,
`5039da4cab26babf0ea04a428f51b7651356b65cac78d62a9927b40181f000c5`, and
`b8ca0a6edb4c85b7f9e60ead71a779d913de070ccb2ddb13051342902a5de590`.

The 200-star audit remains unopened and the current production model remains
unchanged. A small `alpha=0.10` broad low-gravity diagnostic was used to set
the next training target; it remains diagnostic until a fresh standalone model
passes the complete solver and spectral checks.

## Alpha-0.10 standalone follow-up — 2026-08-10

The next standalone model used the broader training-only target blend in
`logg < 3.2`, with `alpha=0.10`, three 4×512 seeds, and the same train/audit
exclusions. It improves the aggregate profile statistics, but the pointwise
profile gate still fails one development star:

| check | result |
| --- | --- |
| T profile p95 | `2.023e-3` |
| m profile p95 | `7.304e-3` dex |
| worst T profile | `8.87%` |
| worst m profile | `0.438` dex |
| pointwise profile gate | **failed**: `t07650.8_g+1.79_m-1.93_a+0.36_x2.06` |

The failure is not a seed-median accident: the three seed predictions for
that star have maximum m errors `0.438`, `0.249`, and `0.607` dex. Therefore
this model was not sent through the real solver and did not replace the
current product. A single 6×1024 fallback with the same target is training on
the cluster; the sealed audit remains unopened.

Artifacts:
`results/goal_spectral_20260808/predicted_default_trajectory_alpha010_probe60.npz`,
`results/physical_profile_accuracy_alpha010_development.json`,
`results/default_trajectory_target_alpha010_20260809.npz`, and
`artifacts/reduced_state_emulator/physical_default_trajectory_alpha010_cpu/`.
Prediction SHA256 is
`96f9eb2bd55ec0e235d64ff7282c78f23f1d1b04d9e6c7b77cec899e1e584011`; the
profile report SHA256 is
`c6598696bcbd6a71af436d30725c65e03e8c32463fb59fa5d0dfdc82affe606a`.

## Alpha-0.80/1.00 standalone follow-up and final 60-star gate — 2026-08-10

The broader standalone target was increased to `alpha=0.80` and then to
teacher-only `alpha=1.00` during training, still with labels → `(m,T)` at
inference. The 4×512 runs improved the persistent low-gravity mass failure but
did not remove it: the worst mass error was `0.163` dex for `alpha=0.80` and
`0.128` dex for `alpha=1.00`, both at
`t07650.8_g+1.79_m-1.93_a+0.36_x2.06`. A single 6×1024 `alpha=1.00`
fallback was the closest model, with T p95 `3.238e-3` and m p95
`1.704e-2` dex, but its worst mass error was still `0.122` dex. It was sent
through the real solver only as a diagnostic; it is not a production model.

The strict 60-star follow-up used adaptive pressure synchronization (8-pass
limit, `1e-3` dex). The candidate synchronized for `59/60`; the excluded star
was `t08425.4_g+1.06_m-0.84_a+0.47_x1.68`, whose pressure changes were still
`1.912e-3` dex after pass 8. The other 59 solver runs all converged:

| check | result |
| --- | --- |
| real solver | `59/60` overall; `59/59` among synchronized stars |
| iterations | mean `4.356`; p90 `6`; max `14` |
| non-monotonic trajectories | `18.6%` |
| profile p95 | T `3.238e-3`; m `1.704e-2` dex |
| profile pointwise gate | **failed**: one mass blowout, `0.122` dex at `t07650.8...` |
| normalized flux | median `1.280e-3`; max `9.602e-3`; 3 over `5e-3` |
| total flux | median `1.308e-3`; max `9.468e-3`; 2 over `5e-3` |
| continuum | median `4.790e-4`; max `5.255e-3`; 1 over `5e-3` |
| spectral acceptance | **failed**: the median target passes, but the over-bar limits do not |

The normalized-flux outliers are
`t04058.7_g+3.10_m-1.71_a-0.04_x1.64`,
`t04531.6_g+1.43_m-2.32_a+0.16_x0.58`, and
`t05479.2_g+1.90_m-0.96_a+0.40_x2.53`. Thus the remaining problem is no
longer the old pressure-pass misalignment: the network is usually in the
right solver basin, but a small low-gravity/metal-poor tail still moves the
converged spectrum too far, and `t07650.8...` still has a mass-profile error.

The complete current artifacts are
`results/goal_spectral_20260808/predicted_default_trajectory_alpha100_6x1024_probe60.npz`,
`results/physical_profile_accuracy_alpha100_6x1024_development.json`,
`results/goal_spectral_20260808/default_trajectory_alpha100_6x1024_full60/convergence_metrics_learned_physical_ensemble.json`,
`results/goal_spectral_20260808/default_trajectory_alpha100_6x1024_spectral_gate.json`,
and `figures/spectral_gate_default_trajectory_alpha100_6x1024.png`.

SHA256 provenance: prediction `9fec50baf175ab2a5ceebfb9daa0b1faabd79b3507720e0943fb46e24749a415`,
profile `daed9edf2b43fe577d65b83abee6751a4b808841162d90985347c1f475ff4cdf`,
solver `5d75db49ff548a0eb7d7e64f078a962cb67b9f3ed3848f2bcbec01f47635949a`,
spectral gate `b8c03b940c28ed7904f35c4d1cd28a6fef202e6b977a14c3aa825556b9d03bba`,
figure `10e3d2a61bef7274126133b7206b5848d970213974b732749ceb6364000c13a0`.

The candidate remains rejected and the production model remains unchanged.
The sealed 200-star audit was not opened. The next correction should target
the failing label region with a genuinely standalone residual/correction
model, then repeat the same 60-star solver and spectral gate.

## Repair run — 2026-08-08

The exact source, data and prediction hashes for the best development result
are recorded in `results/physical_profile_accuracy_development_fitall.json`,
`artifacts/reduced_state_emulator/physical_fitall_hard_cpu/predicted_physical_ensemble.json`,
and `results/field_consistency_dev.json`.

Provenance: corpus SHA256
`092cf3c4a0c6075c2dde05803bec62b1d0eb6ec36d5794fe400870c505b44284`;
development split SHA256
`042f4d33d9c1e971474e888148bb8648e2adb937365e99d9c162c8e7f6d85753`;
sealed-audit manifest SHA256
`d9b183a30e879c762d04e3624009af584b5db419c3bfe3971c315615f78d4b81`;
best prediction SHA256
`648889bbf7daf053f8d7851a65fcf02af4d0efc436b6d06b4f050e62782a3767`.

| item | result |
| --- | --- |
| best candidate | 3-seed 4×512 physical network, seeds `20260807–20260809`, fit on train+validation rows while excluding the 60 development and 200 sealed-audit rows, hard-region weight 3, tail loss 0 |
| pointwise profile p95 | T `2.611e-3`; m `1.082e-2` dex — both pass |
| per-star failures | T: `46124`, `33051`; m: `46124`, `30143`, `33051`, `24755` |
| worst per-star error | T `68.31%`; m `0.618` dex |
| profile gate | **failed**: 4 unique stars still miss the `<10%` / `<0.10 dex` limits |
| fallback 6×1024 | T p95 `2.781e-3`, m p95 `2.628e-2` dex; rejected because mass worsened and 9 stars failed the mass limit |

The four-arm comparison uses the best 4×512 candidate and 60 development
stars. Median errors are:

| arm | P | n_e | κ_R | g_rad normalized |
| --- | ---: | ---: | ---: | ---: |
| six-field direct | `1.81e-2` | `1.30e-2` | `1.43e-2` | `1.18e-2` |
| six-field `(m,T)` + physics | `1.43e-2` | `1.14e-2` | `8.60e-3` | `3.57e-3` |
| truth `(m,T)` + physics | `4.41e-4` | `5.89e-4` | `8.74e-4` | `2.48e-4` |
| new `(m,T)` + physics | `3.06e-3` | `2.76e-3` | `2.73e-3` | `1.09e-3` |

All 60 stars reconstructed successfully. The adaptive loop stayed synchronized
for every arm: the maximum pass count was 6 for six-field `(m,T)`, 5 for truth
`(m,T)`, and 5 for the new `(m,T)`; the final pressure change stayed below
`1e-3` dex. The comparison figure is
`figures/field_consistency_comparison.png`.

## Sealed solver/spectral goal run — 2026-08-08

The preselected 60-star subset from
`results/sealed_solver_subset_20260808.json` was run on the cluster with the
same real solver policy for the production six-field initializer and the new
physical `(m,T)` ensemble. The production arm converged for all 60 stars. The
candidate converged for 58/60; the failures were
`t04995.8_g+2.97_m+0.19_a-0.04_x3.48` and
`t07650.8_g+1.79_m-1.93_a+0.36_x2.06`.

| arm | converged | mean iterations | p90 | non-monotonic |
| --- | ---: | ---: | ---: | ---: |
| production default | 60/60 | `5.37` | `8` | `41.7%` |
| new physical `(m,T)` | 58/60 | `3.24` | `3` | `16.7%` |

The spectral gate had 58 paired spectra. It **failed**: normalized-flux median
`1.398e-3` with 4 stars above `5e-3`; total-flux median `1.598e-3` with 4
above; continuum median `5.544e-4` with 1 above. The union is 5 stars above
the bar. Full details are in
`results/goal_spectral_20260808/spectral_gate.json`; the solver summaries are
`results/goal_spectral_20260808/convergence_metrics_*.json`.

The five spectral outliers are all in the low-gravity/cool or metal-poor tail.
Forcing a candidate restart to run at least 5 or 8 iterations reduces some
differences but does not pass the gate: the 5-star probe still has 4 failures
at minimum 5 and 4 failures at minimum 8. This isolates the next correction
to the predicted low-gravity `(m,T)` start, not to the pressure synchronization
loop. The current production model remains unchanged. A separate 3-seed 4×512
low-gravity-weighted model (`logg < 3.2`) is training on the cluster; it still
excludes both the 60 development stars and the sealed audit rows.

Verification completed: the fast suite is `7 passed, 1 skipped`; the real
one-star physical solver chain is `6 passed` in about 45 seconds. The full
preselected 60-star solver and spectral gate have now run and failed the
spectral/convergence acceptance gate as recorded above. The remaining sealed
200-star audit rows remain untouched, so no final audit result is being
claimed. The reproducible solver subset manifest is present at
`results/sealed_solver_subset_20260808.json`.

---

## 1. What was run

| part | what | status |
| --- | --- | --- |
| 0 | archaeology + code audit | done — `notes/reduced_state_existing_work.md` |
| 1 | baseline benchmark | done — reused existing `bench/` runs, consolidated |
| 2 | oracle `(m,T)` → 6-field reconstruction | done — new `reduced_state/` library + `experiments/reduced_state_parity/` |
| 3 | depth/representation resolution | done — `experiments/depth_resolution/` |
| 4 | learned `(m,T)` initializer, restarted on the real solver | done — `reduced_state/emulator.py` + `experiments/reduced_state_emulator/` |
| 5 | monotone vs direct column-mass parameterization | done — same run, both arms trained |
| 5b | six-parameter accuracy: two fields + physics vs six predicted fields | done — `experiments/reduced_state_emulator/derived_field_accuracy.py` |
| 5c | spectral dimension of the sufficiency claim | done — `experiments/reduced_state_emulator/truth_arms.py` + `spectral_gate.py` |
| 5d | control: the solver's own start-dependence | done — `experiments/reduced_state_emulator/jitter_control.py` |
| repair | pressure synchronization, physical-coordinate training and unified comparison | done — development gate failed; no model replacement |

**Part 4 was re-scoped before execution, per this file's own instruction that
each part re-verify its premise.** The brief specified a *continuous* coordinate
network queryable at arbitrary depth. Part 3 had just swept representation error
over seven orders of magnitude and found restart behaviour flat within sampling
noise, which removes the measured benefit continuity was supposed to buy. What
survived the evidence — predicting two fields instead of six, on the production
grid, with monotonicity structural rather than guarded — is what was built. The
continuous variant remains untested and is no longer the obvious next thing.

### Part 1 — baseline (no new solver time)

`experiments/baseline_restart/consolidate.py` reads the four existing runs
(`runs/baseline_local/`, `runs/baseline_cluster/{iid,boundary,hard}/`) into
`results/baseline_metrics.{json,csv}`. Confirms the numbers already in
`solver-in-the-loop-progress.md` §4.2–4.3 (box: 85.3% converged, geomean
q≈0.75; IID: 99.2% converged, geomean q=0.644; boundary/hard fail ~1 in 5).

### Part 2 — Gate 1: is `(m,T)` sufficient? **Yes.**

60 stars stratified from `strict_truth_52199.npz` (40 IID + 20 hard-region:
logg∈[0.7,2.8], [M/H]∈[−2.5,−0.5], matching `bench.labels.sample_hard_region`).

`reduced_state/reconstruct.py::reconstruct_full_atmosphere` builds P, n_e,
κ_R, g_rad from exact truth `(m,T,labels)` using only certified
`payne_zero_atmosphere.runner` functions (population → opacity → transfer →
finalize → hydrostatic), 3 synchronization passes, m/T never touched. See
that module's docstring for exactly why this differs from Ting's own
external attempt (13–26% upper-layer drift, `solver-in-the-loop-prior-work.md`
§2.1): this reconstruction never remaps the grid, so it avoids the
top-boundary-seed artifact `continuity/` already diagnosed.

Results (`results/reconstruction_metrics.json`, `figures/reconstruction_parity.png`):

| field | median rel. error | p90 | max (worst star/layer) |
| --- | ---: | ---: | ---: |
| gas_pressure | 3.9e-4 | 2.9e-3 | 0.33 |
| electron_density | 5.5e-4 | 3.5e-3 | 0.28 |
| rosseland_opacity | 8.5e-4 | 5.5e-3 | 0.14 |
| radiative_acceleration | 8.2e-4 | 4.4e-3 | 0.23 |

Error is smallest mid-atmosphere and rises toward both the top (τ≈10⁻⁷) and
bottom (τ≈10³) boundaries — a handful of stars/layers reach 10–30% error.

Restart comparison (`results/convergence_metrics_reduced_state_parity.json`),
same 60 stars, real solver, production policy (single trial, no jitter —
see `reduced_state/restart.py` docstring for why):

| | converged | mean iters | p90 | geomean q | non-monotonic |
| --- | ---: | ---: | ---: | ---: | ---: |
| reduced-state reconstruction | 100% | 3.33 | 3 | 0.642 | 26.7% |
| full six-field truth (oracle) | 100% | 3.23 | 3 | 0.636 | 31.7% |

56/60 and 57/60 stars respectively hit the exact 3-iteration floor. Only 4
stars show a gap (always ≤2 extra iterations, never a failure) — the
Gate-1-mandated honest report of where `(m,T)` alone is not quite enough,
rather than silently patching it.

### Part 3 — Gate 2: does representation resolution matter? **No.**

Reframed with the user's explicit approval (asked via AskUserQuestion,
2026-08-07) to avoid solver surgery: `line_opacity.py` hard-asserts
`layer_count == 80` in two places, and the existing `continuity/` harness
already showed 16× denser quadrature moves the closure residual only in the
4th decimal place. Instead: round-trip the true `(m,T)` curve through an
N-point intermediate log-τ grid (cubic spline in log-log, same method as
`continuity/closure.py`) and back onto the fixed 80-point production grid
(`reduced_state/resample.py`), for N ∈ {40, 80, 160, 320, 640}, then
reconstruct (Part 2's exact pipeline) and restart.

Results (`results/convergence_metrics_depth_resolution.json`,
`figures/resolution_vs_*.png`), same 60 stars:

| N | representation error, median m(τ) | converged | mean iters | p90 | geomean q | non-monotonic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 3.1e-5 | 98.3% | 3.42 | 4 | 0.548 | 15.0% |
| 80 (identity) | ~1e-16 | 100% | 3.33 | 3 | 0.642 | 26.7% |
| 160 | 4.0e-7 | 98.3% | 3.15 | 3 | 0.650 | 26.7% |
| 320 | 4.7e-8 | 98.3% | 3.19 | 3 | 0.652 | 26.7% |
| 640 | 6.6e-9 | 100% | 3.32 | 3 | 0.641 | 26.7% |

Representation error spans **7 orders of magnitude** across N; restart
behavior is flat within sampling noise at n=60 (only N=40, the coarsest,
shows a mild, plausibly real degradation: p90=4 instead of 3, p99≈11 instead
of 7–10). This corroborates rather than contradicts `continuity/`'s finding:
discretization/resolution is not the bottleneck once information reaches the
solver's grid through a proper (non-remapping) reconstruction.

### Part 4 — Gate 3: does a *learned* `(m,T)` still land in the basin? **Yes, at a cost.**

`reduced_state/emulator.py`: 5 labels → 4×512 SiLU → (log10 m, log10 T) on the
fixed 80-point grid, float64, trained on the 52,199-star corpus with the 60
Part-2/3 evaluation stars excluded (41,867 train / 10,272 validation / 60
held out). Prediction is materialized to an `.npz` by
`experiments/reduced_state_emulator/predict.py` and consumed by
`run_learned_restart.py`; see that module's docstring for why the two must be
separate processes.

Held-out profile accuracy — the two-field network beats the shipped six-field
one at the two fields they share:

| initializer | T p95 | m p95 (dex) |
| --- | ---: | ---: |
| production six-field (release) | 4.020e-3 | 2.393e-2 |
| **learned two-field, monotone** | **3.742e-3** | **1.519e-2** |

Real solver, same 60 stars, production policy (single trial, no jitter):

| arm | converged | mean iters | p90 | p99 | geomean q | non-monotonic | first residual p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production six-field (shipped) | 0.983 | 5.59 | 8 | 11.26 | 0.656 | 53.3% | 2.44e-3 |
| **learned reduced state** | **0.950** | **4.07** | **6** | **8.44** | **0.550** | **18.3%** | 2.76e-3 |
| truth `(m,T)` oracle (Part 2) | 1.000 | 3.33 | 3 | 10.23 | 0.642 | 26.7% | 4.20e-4 |
| full six-field truth (Part 2) | 1.000 | 3.23 | 3 | 8.82 | 0.636 | 31.7% | 4.23e-4 |

**Correction to Part 2's reading, from the first-residual column.** The two
oracle arms start at 4.2e-4, already *inside* the 5e-4 convergence threshold —
they are held for three iterations only by
`minimum_iterations_before_convergence`. Their `q` and non-monotonic fractions
therefore measure noise-floor behaviour, not contraction, and **must not be
compared against arms that start an order of magnitude further out.** Only the
first two rows share a starting scale and are directly comparable. This
invalidates the inference in the previous revision of this file that 31.7%
non-monotonicity in the six-field oracle proves the oscillation is intrinsic to
the temperature correction; that number does not support the claim. The
oscillation question is reopened, not settled — see below.

Against the only fair comparison, the learned reduced state wins clearly:
27% fewer iterations, 25% lower p90 and p99, 16% better contraction, and
**two-thirds less oscillation** (53.3% → 18.3%).

The result that most constrains what comes next: the learned initializer starts
*further from truth* than production (2.76e-3 vs 2.44e-3 median first residual)
and still converges faster with less oscillation. Proximity in profile space is
not what makes a start good. That is Ting's Sec 2.2 negative result
(`solver-in-the-loop-prior-work.md`) demonstrated constructively.

**The cost, stated plainly: 3 failures against production's 1.** One is shared
(`t07306.4`, which production also fails), so the net regression is two stars.
Their character differs:

| failed star | T max err | m max (dex) | reading |
| --- | ---: | ---: | --- |
| `t07473.3_g+3.71` | 28.6% | 0.317 | prediction blow-out; ξ=3.94, at the label-box edge |
| `t07306.4_g+3.37` | 38.5% | 0.223 | prediction blow-out; production fails it too |
| `t04336.1_g+4.99` | 1.07% | 0.010 | genuine basin miss — 4336 K, logg 4.99, the cool-dwarf box v1.2.1 excludes |

Converged stars have median 0.46% T error and 1.25e-2 dex; the two blow-outs are
two orders of magnitude outside that. **They are a network coverage failure, not
a reduced-state failure** — nothing about `(m,T)` sufficiency is implicated, and
they are addressable with data or capacity. Only `t04336.1` is a real basin
miss, and it sits in the regime already known to be hard.

### Part 5 — monotone vs direct: **the softplus is required.**

Both arms trained identically, 300 epochs, same seed and split:

| arm | T p95 | m p95 (dex) | non-monotonic profiles (of 60) |
| --- | ---: | ---: | ---: |
| monotone (`eps + softplus`, cumulative) | 3.742e-3 | 1.519e-2 | **0** |
| direct (80 independent outputs) | **3.223e-3** | **1.402e-2** | **3** |

The direct parameterization is genuinely more accurate — 14% better on T, 8% on
m — and still produces three profiles that `reduced_state.reconstruct.
_seed_atmosphere` rejects outright with `ValueError`. A 5% hard-failure rate is
not worth a 14% accuracy gain, so the monotone arm is the one carried forward.

Two details worth keeping. First, full training does most of the work: an
under-trained checkpoint produced 39/60 violations, a fully trained one 3/60.
"Almost always monotone" is what a guard is for, and is exactly what a guard
rejects. Second, plan amendment A4 argued a softplus was unnecessary for the
*six-field* decoder because its decode is a cumulative sum of `10**c`, positive
by construction. That argument does not transfer to predicting log10 m directly,
and this ablation is the measurement of the difference.

### Part 5b — six-parameter accuracy: is the reduced state actually the better product?

Parts 2–4 argued sufficiency from convergence behaviour. This asks the product
question directly: for the four fields the two approaches both have to supply,
which is more accurate against truth? Three ways of obtaining them, same 60
held-out stars, median relative error over all stars and layers:

| field | shipped six-field network | **learned (m,T) + physics** | truth (m,T) + physics |
| --- | ---: | ---: | ---: |
| gas pressure | 1.81e-2 | **4.57e-3**  (4.0×) | 3.89e-4 |
| electron density | 1.30e-2 | **5.04e-3**  (2.6×) | 5.50e-4 |
| Rosseland opacity | 1.43e-2 | **5.05e-3**  (2.8×) | 8.52e-4 |
| radiative acceleration | 3.18e-2 | **5.68e-3**  (5.6×) | 8.19e-4 |

The middle column is the deployable path — the trained network's `(m,T)`, not
truth. With the two shared fields from Part 4 (temperature p95 3.74e-3 against
4.02e-3; column mass p95 1.52e-2 dex against 2.39e-2), **two predicted fields
plus physics beat six predicted fields on all six fields.**

Two things this settles that the earlier framing did not:

- The 17–47× figure obtainable from the third column is an *oracle* result. It
  establishes that `(m,T)` carries the information, not that the reduced state is
  the better product. The middle column is what answers the product question,
  and it still wins, by 2.6–5.6×.
- A concrete worry was checked and did not hold. Hydrostatic equilibrium makes P
  essentially algebraic in m, so predicted-m error propagates straight into P;
  the learned column mass has p95 1.52e-2 dex (≈3.6% relative), which would have
  put derived P *worse* than the network's directly predicted 1.8%. It does not:
  derived P sits at 4.57e-3. The estimate compared a p95 against a median.

**The remaining gap is entirely in the (m,T) predictor.** Middle column 5e-3,
oracle column 5e-4 — an order of magnitude, and the reconstruction step
contributes none of it, since with exact `(m,T)` it reaches 4e-4. Any further
effort belongs in the two-field network, not in the physics path.

### Part 5c — the spectral dimension

Iteration counts and profile errors are both atmosphere-space quantities. The
observable test is whether the reduced state changes the emergent spectrum.
`truth_arms.py` re-runs both truth arms with converged products saved, and
`spectral_gate.py` synthesizes each and applies Ting's three metrics over
400–900 nm at R=20000 in float64. Metric implementations are imported from
`emulator_v1_2.gates.compare_spectra` rather than reimplemented, so a subtle
redefinition cannot make this gate easier than his.

| comparison | n | norm. flux max | median | over 5e-3 |
| --- | ---: | ---: | ---: | ---: |
| **truth (m,T) vs full six-field truth** | 60 | **1.39e-3** | **8.1e-5** | **0 / 60** |
| learned reduced state vs production | 57 | 5.01e-3 | 1.62e-3 | 1 / 57 |
| production vs its own jitter retry start | 56 | 1.50e-2 | 3.44e-3 | 16 / 56 |

The first row is the spectral half of the sufficiency claim: discard P, n_e,
kappa_R and g_rad, rebuild them from `(m,T)`, and the solver converges to an
answer that is spectroscopically indistinguishable from starting with all six —
every star passes, median 62× below the bar.

### Part 5d — control: what does 5e-3 mean for this comparison?

The third row above is why the second is interpretable. Ting's 5e-3 bar was set
for a candidate-versus-reference-*implementation* comparison
(`emulator_v1_2/RESEARCH_LOG.md` Sec 10), not for two runs of the same solver
from different starts. Without a measurement of the solver's own width, a number
near the bar cannot be read either way.

`jitter_control.py` supplies that measurement using production's own retry
policy. `deterministic_initializer_labels` returns "the exact-label initializer
followed by reproducible neighbors" (`warm_start.py:1012`); index 0 is the arm
already on disk, index 1 is the start production would retry from. Both are
starts production is willing to ship a result from.

Their converged products differ by median 3.44e-3, with **16 of 56 stars over
the bar** — three times the maximum and twice the median of the learned-versus-
production comparison, and sixteen times as many exceedances. The learned
initializer moves the solver's fixed point roughly half as far as production's
own jitter does. The single exceedance in the learned gate is well inside the
solver's intrinsic start-dependence.

That jittered start is also worse to solve from than the learned reduced state:
93.3% converged at mean 7.45 iterations, against 95.0% at 4.07.

### Reproducing

Commands only. **How to get code onto the node, launch, poll, and pull results
back — and the seven ways that went wrong — is
[`solver-in-the-loop-cluster.md` §8](../archive/2026-08-19/solver-in-the-loop-cluster.md).** Read it
before running any of this. `.venv/bin/python` is local (macOS);
`.venv-linux/bin/python` is the cluster.

```bash
export NUMBA_THREADING_LAYER=workqueue
PYTHONPATH=. .venv-linux/bin/python -m pytest tests/ -q                                    # fast, ~7s
PAYNE_ZERO_RUN_SOLVER=1 PYTHONPATH=. .venv-linux/bin/python -m pytest tests/test_reduced_state.py -q   # ~75s

PYTHONPATH=. .venv-linux/bin/python -m experiments.baseline_restart.consolidate
PYTHONPATH=. .venv-linux/bin/python -m experiments.reduced_state_parity.run_oracle_parity \
    --count 60 --workers 48 --seed 20260807                                                # ~40 min on 160 cores
PYTHONPATH=. .venv-linux/bin/python -m experiments.depth_resolution.run_depth_resolution \
    --workers 48 --seed 20260807 --reuse-indices-from results/reconstruction_metrics.json  # ~50 min

# Parts 4-5. Training is local (float64 CPU, ~50 min per arm); the solver arms are cluster.
PYTHONPATH=. .venv/bin/python -m experiments.reduced_state_emulator.train --epochs 300
PYTHONPATH=. .venv/bin/python -m experiments.reduced_state_emulator.predict --arm monotone
PYTHONPATH=. .venv-linux/bin/python -m experiments.reduced_state_emulator.run_learned_restart \
    --arm monotone --workers 30 --skip-production-arm          # ~12 min on 30 workers
PYTHONPATH=. .venv-linux/bin/python -m experiments.reduced_state_emulator.run_learned_restart \
    --production-only --workers 30                             # the shipped-initializer baseline
```

```bash
# Parts 5b-5d.
PYTHONPATH=. .venv-linux/bin/python -m experiments.reduced_state_emulator.derived_field_accuracy \
    --workers 28                                               # ~5 min, no solver
PYTHONPATH=. .venv-linux/bin/python -m experiments.reduced_state_emulator.truth_arms \
    --workers 28 --products-dir runs/reduced_state_emulator/products      # ~50 min
PYTHONPATH=. .venv-linux/bin/python -m experiments.reduced_state_emulator.jitter_control \
    --workers 30 --products-dir runs/reduced_state_emulator/products      # ~40 min
for pair in "full_truth_oracle reduced_state_reconstruction truth_mT" \
            "production_six_field learned_reduced_state ''" \
            "production_six_field production_jitter jitter_control"; do
  set -- $pair
  PYTHONPATH=. .venv-linux/bin/python -m experiments.reduced_state_emulator.spectral_gate \
      --dtype float64 --device cpu --baseline-arm $1 --candidate-arm $2   # ~50 min each
done
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python -m experiments.reduced_state_emulator.make_figures
```

**Every atmosphere must be built inside a worker.** A torch forward pass in a
process that later forks a `ProcessPoolExecutor` deadlocks the pool the moment a
worker reaches the solver's structured-product writer, which is itself a torch
path. The signature is workers that load their catalogs (~5 min CPU, ~15 GB RSS)
and then freeze — `ps` shows elapsed advancing while `TIME` does not, and no
product is ever written. It cost three cluster runs to localize. This is why
`predict.py` is separate from `run_learned_restart.py`, why `truth_arms.py`
exists instead of `run_oracle_parity.py --products-dir`, and why the two arms of
`run_learned_restart.py` must be two invocations. Do not merge any of them.

The gate imports its metric implementations from `emulator_v1_2`, so that
package's `gates/` and `lib/` must be on the remote too; its `artifacts/` are
not needed.

---

## 2. Future plan — Parts 6–9

Ordered per the original brief. Each part should re-verify its premise
against the numbers above rather than assume them.

**Parts 4 and 5 below are the brief as written and are kept for provenance;
both have been executed, and §1 supersedes them.** Part 4 was executed without
its continuous-coordinate framing, for the reason recorded at the top of §1;
Part 5's ablation ran and the softplus was retained. The one claim in Part 4's
text that has since been answered is its comparison list: the two-field discrete
emulator beats the six-field production baseline on all six fields (§5b), and
the denser-grid variant it predicted "little gain" from indeed showed none (§3).
The two-field *continuous* emulator remains the only untested arm on that list,
and Part 3 removed the measured motivation for it.

**Part 4 — continuous reduced-state emulator.** `f(theta_star, log tau) ->
(log m, log T)`, lightweight coordinate network (stellar-label encoder +
Fourier features + residual MLP), queryable at arbitrary depth, trained on
randomly-sampled depth coordinates rather than the fixed 80-point vector.
`reduced_state/reconstruct.py` and `restart.py` are already the right
interface to evaluate it: swap the truth `(m,T)` in `ReducedAtmosphere` for
the network's prediction and the rest of the Part 2/3 pipeline applies
unchanged. Compare at minimum: current six-field discrete emulator (existing
production baseline), two-field discrete emulator, two-field denser-grid
emulator (already probed by Part 3 — expect little gain), two-field
continuous emulator.

**Part 5 — monotonic column-mass parameterization.** `d log m / d log tau =
eps + softplus(raw_slope)`, integrated. Compare against direct log-m
prediction. Note `payne_zero_diffatm`'s existing decode
(`solver-in-the-loop-plan.md` amendment A4) found a `softplus` unnecessary
for the *current* six-field decoder because cumulative-sum-of-`10**c` is
positive by construction — check whether the same argument applies to a
continuous coordinate network before assuming Part 5 is needed at all.

**Part 6 — physics-informed training.** Optical-depth residual `L_tau ~
||kappa_R * dm/dtau_R - 1||^2` first (network's autodiff dm/dtau against
`reduced_state.reconstruct`'s κ_R), flux closure only after that works.
Ablate: supervised only / +monotonic / +optical-depth physics / +optical-depth+flux.

**Part 7 — reduced-state solver.** Rewrite the persistent iterative state as
primarily `(m,T)`, with P/n_e/κ_R/g_rad as derived quantities each
iteration — essentially generalizing Part 2's single-pass reconstruction
into the per-iteration inner loop. Require one-step parity (ΔT, flux
residual, opacity structure) against the current solver before any
convergence comparison. `payne_zero_diffatm`'s T1–T2 (grid_math, twin_eos)
are verified and reusable if a differentiable version is wanted; T3–T8 are
unverified drafts (blocked on quota per `solver-in-the-loop-progress.md`
§10) and should not be trusted without their checkers passing.

**Part 8 — final convergence comparison.** The ablation matrix (Baseline,
A=2-field discrete, B=2-field dense discrete, C=2-field continuous,
D=+physics-informed, E=+reduced solver) using the same 60-star sample (or a
larger one) and the same `bench.report`-compatible record schema this
round's experiments already emit, so aggregation is unchanged.

**Part 9 — diagnose remaining oscillation.** If Part 4/6's continuous
initializer substantially improves contraction/non-monotonic fraction over
today's production baseline (`results/baseline_metrics.json`), representation
was a real factor. If not — and Part 2/3's results above already suggest the
*initializer* is not the main lever, since even the oracle reconstruction
sits at the 3-iteration floor with 26–32% non-monotonic trajectories — then
the non-monotonicity is a property of the temperature-correction/damping
scheme itself, not of initialization, and the original solver-in-the-loop
project's learned-damping-controller idea (`solver-in-the-loop-prior-work.md`)
becomes the next thing to scope, not build yet.

### Immediate recommendation

**Withdrawn and replaced.** The previous revision argued from the six-field
oracle's 31.7% non-monotonic fraction that oscillation is intrinsic to the
temperature correction and that a perfect initializer cannot fix it. Part 4's
first-residual column shows why that argument does not hold: the oracle arms
start inside the convergence threshold, so their contraction statistics measure
noise, not dynamics. The learned initializer, starting at a *realistic* residual,
cut non-monotonicity from 53.3% to 18.3% — initialization moves this number a
great deal.

What replaces it, in order:

1. **Fix the two coverage blow-outs before anything else.** They are the whole
   of the convergence regression and they are not a reduced-state problem: their
   predicted temperature is 28.6% and 38.5% off, against 0.46% for converged
   stars. Check whether ξ near 4.0 and the cool-dwarf corner are simply thin in
   the corpus, and whether a capacity or loss-weighting change closes them. Until
   this is done, no comparison against production is fully honest.

2. **Then push the (m,T) predictor, not the physics path.** Part 5b localizes
   the entire remaining accuracy gap: the deployable derived fields sit at 5e-3
   and the oracle at 5e-4, and the reconstruction contributes none of that
   difference. An order of magnitude is available and all of it is in the
   two-field network. This is a sharper target than Part 6's physics-informed
   loss and should be tried first — more capacity, the full corpus, and the
   `intended_outer_role` split are all untouched levers.

3. **Part 5c/5d closed the spectral question; do not re-litigate it.** Truth
   `(m,T)` passes 60/60 at a median 62× below the bar, and the learned arm's
   single exceedance is well inside the solver's own start-dependence (16/56 for
   production against its own retry start). The one thing still worth measuring
   in this direction is the label-recovery test: fit spectra generated from
   converged atmospheres using the reduced-state forward model and check whether
   the labels come back, across several windows. A smooth systematic offset is
   invisible to a flux-residual gate and gets absorbed into the labels instead.

4. **The damping controller is not yet motivated by this file's evidence.**
   18.3% residual non-monotonicity from a realistic start is much weaker support
   than the 53.3% production figure or the box baseline's 64% suggested. Whether
   a learned damper still pays has to be re-argued against the *learned*
   initializer's trajectories, not production's. That is Part 9's question and it
   is genuinely open again.

5. **The comparison still rests on a corpus of stars that converged.** All 60
   evaluation stars come from `strict_truth_52199.npz`, whose admission contract
   requires convergence in fewer than 30 iterations. Nothing here tests whether a
   better initializer rescues the stars production *fails* — those labels are
   absent from the corpus by construction. Generating them is the missing
   experiment behind every convergence claim in this file, and it is unchanged by
   anything in Parts 5b–5d.

6. **Everything is still benchmarked against the v1.1 public release.** The
   production arm calls `emulator_warm_start_model` with no `initializer_label`,
   which loads `five_label/checkpoint.pt` — the bundled v1.1 initializer. Ting's
   v1.2.1 complete-latent checkpoint improves on it substantially (Sun 5, giant 5,
   hot dwarf 7 against bundled 7/4/14, `solver-in-the-loop-prior-work.md` §2.4)
   and is **not in this checkout**. Every "beats production" statement in this
   file means "beats the public release" until that checkpoint is obtained and
   the baseline arm re-run.

---

## Solver-in-the-loop K=1 closure — 2026-08-11

The differentiable T3–T8 twins and K=1 unroll are now checked and wired into a
physical, profile-constrained adapter. Training directly against truth improved
solver convergence to 60/60 but failed the spectral fixed-point gate on two
stars. Retargeting the one-step loss to the converged production atmospheres
and restricting the RBF correction to the two unstable tail cases removed that
regression.

On the frozen development 60, the final candidate passes all three levels:
profile (T p95 `2.299e-3`, m p95 `1.146e-2` dex, no blowouts), real solver
(60/60, mean `3.95`, p90 `6`), and spectra (max normalized/total/continuum
differences `4.230e-3` / `4.926e-3` / `2.059e-3`, all below `5e-3`). The
production initializer is 60/60 at mean `5.37`, p90 `8` on the same set.

Artifacts and reports:

- `artifacts/reduced_state_emulator/solver_in_loop_k1_fixedpoint_tail2/adapter.pt`
- `results/solver_in_loop_k1_fixedpoint_tail2/profile_gate.json`
- `results/solver_in_loop_k1_fixedpoint_tail2/full60/convergence_metrics_learned_physical_ensemble.json`
- `results/solver_in_loop_k1_fixedpoint_tail2/full60/spectral_gate.json`

That candidate could not be used for a valid fresh blind claim because its base
checkpoint did not explicitly exclude the fresh rows. A replacement candidate
was retrained from a provenance-qualified base, passed dev-60 profile/solver/
spectral qualification, passed the previously opened audit profile gate, and
was frozen before opening `sealed_audit_20260811.json`.

## Final independent blind test — 2026-08-11

**Failed; no post-blind tuning was performed.** On the fresh 200 stars, the
profile gate failed: temperature p95 `3.380e-3`, column-mass p95 `2.656e-2` dex,
with 6 temperature and 18 mass blowouts. The candidate yielded 192 usable
atmospheres versus production's 193. It still reduced the mean iteration count
on the 189 paired usable atmospheres from `5.68` to `4.14`, and was
faster/same/slower on 137/32/20 stars, so the speed signal is real but is not
deployable at the present failure rate.

The full-resolution 400–900 nm, R=20,000 spectral gate on 189 usable pairs also
failed: maximum normalized/total/continuum differences were `1.419e-2` /
`1.523e-2` / `6.249e-3`; 25/24/1 stars exceeded the `5e-3` bar. One candidate
run exposed a validator hole: a non-finite atmosphere could satisfy the
temperature-only stop, be reported as converged, then fail product serialization.
Product-requested restart validation now counts that outcome as failure.

The complete machine-readable result is
`results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/blind200/summary.json`.

## Initializer improvement campaign — launched 2026-08-12

The follow-up now targets the blind failure mode rather than increasing network
capacity again. The planned product is a clean two-field base, a small bounded
K=1 correction, and a reliability gate that falls back to the shipped six-field
initializer when the two-field proposal is unsafe. The detailed plan is
`notes/initializer_improvement_plan_20260812.md`.

Before training, two disjoint validation-role sets were selected. The opened
calibration set has 400 stars: 200 ordinary, 100 solver-tail, and 100 label-edge
stars. It has a fixed 300/100 gate split and a preselected 60-star spectral
subset. A new 200-star holdout with a preselected 60-star solver/spectrum subset
is sealed and will not be predicted or inspected until the complete policy is
frozen. Neither set overlaps the earlier development, solver, or audit
manifests.

The long calibration run was launched at `2026-08-12T10:03:28Z` on
`astronode-garching-gpu` (`Shared-GPU-01-A100`), not Node05, because Node05 is
running eight SPHEREx workers. The payne-zero job is limited to six solver
workers and one CPU thread per worker. At launch, the driver PID was `3717784`,
training PID `3717785`, and the first seed had reached epoch 70 with decreasing
validation loss. The remote log is
`logs/initializer_improvement_20260812.log` under the payne-zero checkout.

Manifests and SHA256:

- `results/initializer_calibration_20260812.json`:
  `4e6e20bec9b98e01ed1b68b2eb2cf945417960710b1c9d9c6872ed1a73dc1030`
- `results/sealed_initializer_holdout_20260812.json`:
  `63d968832e01022829b3ef0782207df6339f24c16ed4ca6d2edbf261eef356ee`

Local verification before launch: manifest overlap checks passed, command-chain
dry run passed, and the fast suite reported `91 passed, 11 skipped`. The current
production initializer remains unchanged.

## Grey-start solver control — prepared 2026-08-12

A separate control now tests Ting's question directly: can the unchanged
solver still converge quickly from an analytic Eddington-grey atmosphere? The
start uses only the requested labels: `T(tau)` is the Eddington-grey law,
`m=tau/0.34`, and `P=g*m`. It does not use the six-field network, the two-field
network, or a truth atmosphere. The same 60-star development manifest is used
for production six-field, learned `(m,T)`, truth `(m,T)`, grey with the normal
15-iteration cap, and a 30-iteration continuation for grey failures. Twelve
fixed representative stars also receive three deterministic smooth grey
perturbations.

The local real-solver smoke test already answers one case: the first star
failed at the normal 15-iteration cap but converged at iteration 22 when the
cap was raised to 30. This is only one star, so the population and spectral
conclusion remains open, but it demonstrates that grey can be recoverable
without being fast. Fast verification reports `95 passed, 11 skipped` for the
main test suite.

The resumable cluster driver is
`experiments/reduced_state_emulator/run_grey_start_benchmark_long.py`. It waits
for the active 400-star initializer campaign and all of its workers to finish,
then runs on `astronode-garching-gpu` with at most six CPU workers. It writes
only under `runs/grey_start_benchmark_20260812/` and
`results/grey_start_benchmark_20260812/`; no SPHEREx directory or process is
touched. Final atmosphere profiles, first-versus-later iteration timing, and
full `400--900 nm`, `R=20,000` spectra are all retained.

Update on 2026-08-13: grey15 converged for 12/60 stars. Rerunning its 48
failures with a 30-iteration cap recovered 10 more, giving 22/60 in total.
The learned two-field initializer converged for 57/60 stars on the same run;
the six-field and spectral stages are still running. Of the 38 grey30
failures, 28 already had a non-finite final state and cannot recover merely by
adding iterations. The remaining 10 finite failures are queued for a 60-step
diagnostic after the primary three-way comparison finishes. This extension
does not change the solver or the primary 15-step comparison.

## Expanded four-initializer benchmark — 2026-08-14

The open benchmark now contains 200 stars: the original 60 plus 140 new stars.
All four initializers were run with the unchanged solver and the same
15-iteration cap. The original 60-star results were reproduced exactly.

- Learned two-field: 195/200 converged, mean 3.63 iterations, p90 5.
- Production six-field: 192/200 converged, mean 6.00 iterations, p90 9.
- Full-state interpolation: 190/200 converged, mean 8.02 iterations, p90 11.
- Grey atmosphere: 48/200 converged, mean 12.44 iterations among successes.

The learned two-field initializer is therefore the fastest and most reliable
of the tested starts on this open set. Its trajectories are also more stable:
16% are non-monotonic, compared with 58% for production, 71% for interpolation,
and 96% for grey.

On 189 stars where both learned two-field and production six-field converged,
the median maximum normalized spectral difference over 400--900 nm at
R=20,000 is `1.345e-3`. Seven stars exceed the strict `5e-3` normalized-flux
threshold, and ten exceed it in total flux. Full-state interpolation has a
larger median difference (`2.724e-3`) and 41/186 normalized-flux failures.
Grey starts are not production-ready: their median normalized difference is
`1.968e-2`, with 38/48 failures.

The result supports the scientific claim that a compact `(m,T)` initializer
can be faster and at least as reliable as the shipped six-field initializer,
while usually reaching a very similar spectrum. It does not yet support
universal equivalence: the open sample is not an independent sealed test, the
six-field result is a product reference rather than physical truth, and a small
set of two-field spectral outliers remains. Most reused product files also lack
saved final `kappa_R` and `g_rad`, so the final six-field comparison is complete
for `m,T,P,n_e` but only partial for those two fields.

Deliverables:

- `results/four_initializer_benchmark_expanded_20260814_en.pdf`
- `results/four_initializer_benchmark_expanded_20260814_en.manifest.json`
- `results/four_initializer_benchmark_expanded_20260814/expanded_four_initializer_comparison.json`

## Physical reconstruction seed; six-field runtime dependency removed — 2026-08-19

`reduced_state/reconstruct.py` no longer needs the six-field emulator to seed
the reconstruction. The new default seed (`seed="physical"`) is built from
labels and hydrostatic balance alone: `P_gas = g*m`, `n_e = 1e-4*P/(k_B T)`,
`kappa_R = 0.34`, `g_rad = 0`, label-faithful microturbulence, and the
production deck formatter for metadata/abundances. The pinned (m, T) are
restored to their exact values after the deck round-trip. The old path
remains available as `seed="emulator"`, and a pre-built `ModelAtmosphere`
can be passed as the seed for checks that must construct it in a separate
process.

The physical basis: the molecular EOS re-solves charge conservation from its
own per-layer seeds (`molecular_equilibrium.py`), so the seed's `n_e` is a
validation placeholder, and the first hydrostatic update
(`hydrostatic.py::integrate_hydrostatic_pressure`) already implements
`P_gas = g*m - P_rad - P_turb`, making `P_rad = P_turb = 0` the natural
first guess. The six-field seed was an implementation shortcut, not a
physical requirement.

Seed-independence evidence
(`experiments/reduced_state_emulator/seed_independence_check.py`, truth (m,T)
from `strict_truth_52199.npz`, 12 Teff-spaced stars, six seeds per star:
emulator, physical, physical with electron fraction 1e-6 / 1e-2, physical
with pressure scale x0.5 / x2):

- Fixed 3-pass synchronization: 10/12 stars agree to <= 8e-6 dex across all
  arms; the two slowest (`6751`, `40696`) sit at 1.45e-4 / 6.85e-4 dex.
- The residual contracts ~0.6-2 orders of magnitude per additional pass:
  at 4 passes those two drop to 2.3e-5 / 1.7e-4 dex, at 5 passes to
  2.3e-6 / 4.1e-5 dex — all six seeds share one fixed point.

One mechanism worth recording: the adaptive pressure-only stop halts as soon
as pressure settles, one pass before `n_e`/`kappa_R` finish contracting,
because `build_runtime_state` computes `total_nuclei = P/kT - n_e` from the
*seed's* electron density before the molecular solve. At the production
tolerance (1e-3 dex) this leaves O(tolerance) seed memory in `n_e`/`kappa_R`
for any seed choice, including the emulator's; it is a stopping-tolerance
effect, not evidence of distinct fixed points.

Local environment caveat: on this workstation a process that has run the
torch-based emulator warm start segfaults once the numba continuum kernels
load afterwards, so the check builds emulator seed decks in a separate
subprocess that never imports the solver pipeline. This also means the
solver-gated `tests/test_reduced_state.py` reconstruction test cannot run
locally (its module imports torch at top level); it is unaffected on torch-
free or cluster environments.

Artifacts: `results/seed_independence_20260819.json` (12 stars, 3 passes),
`results/seed_independence_20260819_slow2_pass4.json`,
`results/seed_independence_20260819_slow2_pass5.json`, fast tests in
`tests/test_physical_seed_reconstruction.py`.

## Clean-pipeline blind200 rerun on node07 — 2026-08-19

The frozen 2026-08-11 candidate was rerun end-to-end on
`astronode-garching-node07` with the physical reconstruction seed: same
manifest (`sealed_audit_20260811.json`), same prediction
(`blind200/final.npz`), same solver and production baseline; only the learned
arm's reconstruction seed changed. The audit set was already opened, so this
is an open-set confirmation, not a new blind claim.

| metric | emulator seed (2026-08-11) | physical seed (2026-08-19) |
| --- | --- | --- |
| reconstruction failures | 1 | 2 (19807, 49580) |
| solver converged | 193/199 (192 usable) | 190/198 |
| mean / p90 iterations | 4.16 / 7 | 4.11 / 7 |
| non-monotonic | 27.1% | 26.8% |
| spectral max normalized / total / continuum | 1.4188e-2 / 1.5229e-2 / 6.249e-3 | 1.4188e-2 / 1.5206e-2 / 6.202e-3 |
| spectral stars over 5e-3 bar | 25 / 24 / 1 | 25 / 23 / 1 |

The three extra marginal failures (sync: 49580; solver: `t08641.5`,
`t08705.8`) are boundary flips consistent with O(tolerance) seed memory at
the production 1e-3 dex stop, in the known tail region. Gate outcomes are
unchanged; the blind failure remains driven by network (m,T) accuracy, not
the seed. The two-field pipeline is now end-to-end free of the six-field
network at inference.

Artifacts: `results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/blind200_physical_seed/summary.json`
(SHA256 `d8cb052dd5e1b5cf46a1fded7a2b5f4ccffa0231c716377b0a256e3fd241f324`),
`spectral_gate.json` (`466f6a33e3150eb6ffdfa93424b7353db3b3895a3b7bdfeff1dad760a78e93ee`),
`convergence_metrics_learned_monotone.json` (`a28e428d2e3e869b80894fb73bb982d848153a7de27ea7392426f024603e5eee`).
Run record: `runs/reduced_state_emulator/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/blind200_physical_seed/`.

Operational note for node07: the node was heavily oversubscribed (load ~900
on 160 cores); uncapped numba threading per pool worker thrashed. The
spectral gate only progressed after relaunch with
`NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and 32 workers.
