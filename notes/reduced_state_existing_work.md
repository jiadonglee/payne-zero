# Reduced-state `(m,T)(τ)` initialization — existing work

> Canonical entry point for "has this been tried before." This file does not
> duplicate the five `solver-in-the-loop-*.md` documents at the repo root —
> it indexes them, adds the code-level facts a 2026-08-07 session verified
> against the checkout at commit `9c44001`, and states plainly what is
> reusable vs. missing for the `(m,T)(τ)` reduced-state hypothesis. Read the
> originals for full narrative and numbers; this file is the map.

## 1. Where the hypothesis came from

The reduced-state idea is **not internal speculation** — it is Yuan-Sen
Ting's own proposal, made directly in a 2026-08-06 Slack exchange, fully
transcribed in [`solver-in-the-loop-prior-work.md`](../archive/2026-08-19/solver-in-the-loop-prior-work.md):

> *"instead of the 6 fields, all of that 6 fields can collapse just to
> (m,T)(tau), the other fields from physical relation are just function of
> these two … so by construction if we solve things in these space and
> emulate them here, then the physical relation is satisfied. But the
> current solver actually have a hard time to use this 2D → 6D space as a
> initialization."*

He raised two distinct sub-hypotheses in the same exchange, and it matters
that they are kept separate (see §3):

1. **Spacing**: the fixed 80-point τ grid is too coarse — "more densely
   sampled ... can solve the problem."
2. **Boundary**: "tau itself can be predicted as well ... i.e. depth / step
   size" — the top boundary condition, not the interior spacing.

## 2. What already exists in this checkout

| area | location | status |
| --- | --- | --- |
| Full narrative + decision log | `solver-in-the-loop-{plan,progress,prior-work,continuity,cluster}.md` | current, read first |
| Reference-solver restart benchmark harness | `bench/` | done, tested, 4 full runs in `runs/` |
| Depth-grid closure (static) harness | `continuity/` | done, answered the spacing question |
| Torch differentiable twin (for gradients/training, not reconstruction) | `payne_zero_diffatm/` | partial: T1–T2 verified, T3–T6 unverified drafts, T7–T8 not started |
| Pure-function single-iteration refactor | `payne_zero_atmosphere/runner.py` | done — this is the reusable "synchronization pass" |
| Chinese-language physics walkthrough of the whole solver | `payne-zero-notes.md` | reference, not experimental |

### 2.1 What works — Part 1 (baseline) is fully done, do not redo it

`bench/` is a complete, tested harness:

- `bench/run_reference.py` — mirrors the production trial loop (`cli.py:383-448`), writes `records.jsonl` with per-star, per-trial, per-iteration diagnostics (`deep_layer_relative_temperature_change`, flux-error percentiles, per-stage timings — see docstrings; **no full per-iteration field arrays** unless `--traces` is set).
- `bench/report.py` — computes exactly the metrics this project asks for: `q_k = r_{k+1}/r_k` contraction (`_contraction`, geometric mean because the arithmetic mean is dominated by near-zero denominators — verified necessary), non-monotonic fraction, iteration percentiles (p50/75/90/95/99), failure/retry fraction, `headroom.recoverable_fraction`, per-stage wall time, tail-vs-population label comparison.
- `bench/labels.py` — four samplers: `sample_uniform` (box), `sample_iid_from_corpus`, `sample_boundary`, `sample_hard_region`.
- Four completed runs already sit in `runs/`: `baseline_local/` (300 stars, box), `baseline_cluster/{iid,boundary,hard}/` (1000/500/500 stars). Numbers: IID converges 99.2% with `recoverable_fraction=51.8%`; boundary/hard fail ~1 in 5 with `recoverable_fraction≈73%`; geometric-mean contraction 0.644 (IID) → 0.811 (hard); 56–70% non-monotonic trajectories depending on slice. Full tables in `solver-in-the-loop-progress.md` §4.2–4.3, §9.3.

**Consequence**: Part 1 needs no new solver time — only aggregation of the four existing `summary.json` files into one table (`experiments/baseline_restart/consolidate.py`).

### 2.2 What was tried and failed — the (m,T)→6-field bridge

Ting built this externally (`emulator_v1.2.1_20260712.zip`, not in this
checkout, transcribed in `solver-in-the-loop-prior-work.md` §2.1): a native
fixed-τ profile format plus "one exact population/opacity/transfer
synchronization" before the ordinary iteration — i.e. exactly the
reconstruction this project's Part 2 asks for.

**Result: it did not work.** Passing *exact truth* `(m,T)` through that
bridge still moved column mass **~13% at the most sensitive upper layer**
(his number) / **~26% relative, ~0.258 absolute** (the fresh live-pyk gate
number quoted in the same section), and the resulting normalized spectrum
disagreed with truth by `2.27e-2` — **35× worse** than the bundled
production six-field emulator's `6.43e-4`. His own conclusion: *"Passing
exact truth temperature and mass through the same bridge still moved column
mass by about 13% at the most sensitive upper layer."* Exact truth in, 13%
out — this is not a prediction error, the 2→6 map itself enters a different
branch of the solution near the surface.

**What this checkout independently found about *why*** — the `continuity/`
harness (§3 below) shows the error is **not** a spacing artifact. It
localizes exactly at the closed-form top-boundary seed, which is the
"boundary" half of Ting's own hypothesis (§1, item 2), not the "spacing"
half (item 1). The scope of what is closed is narrow: *the specific
64-temperature+32-mass checkpoint* and *the production-format handoff*. The
coordinate itself is not closed — it is "promising" and explicitly not on
his do-not-repeat list.

### 2.3 What is missing — no reconstruction code exists in this repo

A code-level audit of every module in `payne_zero_diffatm/` (three parallel
Explore agents, 2026-08-07) confirms: **there is no function anywhere in
this checkout, complete or partial, that takes `(column_mass, temperature,
labels)` and reconstructs `pressure, electron_density, kappa_R, g_rad` via
certified physics.**

- `payne_zero_diffatm/twin_eos.py::solve_populations(temperature, gas_pressure, ...)` — `gas_pressure` is a **required input**, not an output. Given m,T alone there is no gas pressure to hand it.
- `payne_zero_diffatm/twin_continuum.py`, `twin_lines.py`, `twin_transfer.py` — all downstream of populations/opacity, none produce a Rosseland mean or radiative acceleration; there is no torch hydrostatic port at all.
- `payne_zero_diffatm/initializer.py::DifferentiableInitializer` — a torch port of the *existing* six-field joint emulator (`payne_zero_atmosphere/warm_start.py::AtmosphereInitializer.predict`, `warm_start.py:614-693`). It predicts all six fields jointly from labels; it is the direct emulator being ported, not a 2→6 physics reconstruction.

**Why pressure genuinely cannot come from `(m,T)` alone with a single closed-form step**, confirmed by reading `payne_zero_atmosphere/runner.py:274-449` (`run_single_iteration`): on the very first call the atmosphere's gas pressure is simply copied from the warm-start guess (`runner.py:291-307`); on every subsequent call, `hydrostatic.integrate_hydrostatic_pressure` needs `integrated_radiation_pressure` and `turbulent_pressure`, both of which are **outputs of a full opacity+transfer pass**, not of `(m,T)` alone. Pressure, opacity, and populations are mutually coupled through the transfer solve — this is the physical content of Ting's 13% drift, and it is why "one synchronization pass" (not a closed-form formula) is the correct minimal reconstruction procedure.

**What is directly reusable, at zero new-physics cost**: `run_single_iteration(config, setup, carry, iteration_index)` (`runner.py:274`, paired with `initialize_iteration_carry(setup)` at `runner.py:252`) is exactly Ting's "one synchronization pass," already refactored (Stage 1a, `solver-in-the-loop-progress.md` §7) into a pure function verified **bit-identical** to the pre-refactor solver on real traces. Feed it a `ModelAtmosphere` whose `column_mass`/`temperature` columns are pinned to truth (labels' warm start used only to seed the pressure guess), call it once, and read `P, n_e, κ_R, g_rad` off the result. This is the `ReducedAtmosphere → FullAtmosphere` interface Part 2 asks for, built from code that already exists and is already certified — no new physics is written.

## 3. The depth-resolution question — half-answered, and how

[`solver-in-the-loop-continuity.md`](../archive/2026-08-19/solver-in-the-loop-continuity.md)
(harness: `continuity/`, output: `runs/continuity/summary.json`) directly
tests Ting's "spacing" hypothesis using the solver's own closure relation
`τ = ∫κ dm`, seeded as `τ[0] = κ[0]·m[0]` (`rosseland_mean.py:72-78`, the
seed is a closed form, not a quadrature — `radiative_transfer.py:132`).

Method: cubic-spline-in-log-log interpolation of `m` and `κ` from the
converged 52,199-star corpus (`strict_truth_52199.npz`) onto grids 1×–16×
denser, re-applying the **unmodified** `integrate_on_depth_grid`, reading
back at the original 80 nodes.

**Result: refining the grid 16× changes the closure residual in the 4th
decimal place** (median 0.00059 → 0.00081 dex, p99 0.0357 → 0.0350 dex, max
identical at 0.66143 dex across every refinement level). The mechanism is
mechanical and airtight: layer 0 is the closed-form seed, refinement
preserves `τ_min` by construction, so the seed — and hence the dominant
share of the residual — is untouched at any resolution.

**Consequence, explicit in the doc**: *"'more densely sampled' is dead; do
not build it. 'tau itself can be predicted ... i.e. depth / step size'
survives and is now the only live option."* The two broken-seed / hard
populations from two unrelated measurements land in the same corner of
label space (logg 2.38–2.54, [M/H] −1.09 to −1.25) — the continuity
harness's severe-seed-residual stars and this project's baseline iteration
tail (`solver-in-the-loop-progress.md` §4.2).

**What this harness does *not* test** (its own §6, "Not tested here"):
recomputing κ from `(m,T)` through the real EOS/opacity path — the
`continuity/` harness reads only stored fields, it never runs solver
physics — and it never restarts the *iterative* solver at any resolution.
It is a static closure-residual measurement, not a convergence-trajectory
measurement. `bench/run_reference.py` has no depth-count parameter, and
`layer_count == 80` is hard-asserted in the certified solver itself
(`line_opacity.py:2391-2393` for selected-line opacity, `:2526-2528` for
transition-line opacity; `equation_of_state.py:962,1413` document/assume it
architecturally in the parallel electron-density sweep). Actually running
the solver end-to-end at `N ≠ 80` would mean modifying certified physics in
several places — out of scope for testing a hypothesis the static evidence
already weighs against.

**This project's Part 3 reframing** (agreed with the user 2026-08-07):
resample the true `(m,T)` curve at N ∈ {40,80,160,320,640} using the same
cubic-spline-in-log-log method, then remap back onto the fixed 80-point
production grid via the existing `remap_to_grid`
(`payne_zero_atmosphere/radiative_transfer.py` /
`payne_zero_diffatm/grid_math.py`, the latter verified bit-exact against the
former) before reconstruction and restart. This measures whatever a future
continuous emulator's internal resolution does to restart behavior, once
its output is materialized on the grid the solver actually consumes — with
zero solver modification, and it directly extends the finding above from
"closure residual" to "iteration count / contraction / non-monotonic
fraction."

## 4. What is reusable vs. what is genuinely new (summary)

| need | reuse | new code |
| --- | --- | --- |
| Restart benchmark, aggregation, samplers | `bench/*` — 100% reusable | consolidation script only |
| `(m,T)` interpolation at arbitrary resolution | `continuity/closure.py`'s cubic-spline-in-log-log method | thin wrapper generalizing it to `T` as well as `m`, and to up/down-sampling |
| Remap onto the 80-point production grid | `payne_zero_atmosphere/radiative_transfer.py` / `payne_zero_diffatm/grid_math.py::remap_to_grid` | none |
| `(m,T,labels) → (P,n_e,κ_R,g_rad)` reconstruction | `runner.py::run_single_iteration` + `initialize_iteration_carry` — zero new physics | thin adapter that builds the seed `ModelAtmosphere` and pins `m,T` to the reduced state before the call |
| Restart-from-custom-atmosphere driver | `AtmosphereInput.initial_atmosphere` accepts any `ModelAtmosphere` (`config.py:12-15`); pattern already used by `bench/perturb_deck.py` and `runs/probe_line_counts.py` | thin driver mirroring `bench/run_reference.py`'s `_solver_config` + trial loop, emitting `bench.report`-compatible records so aggregation is reused too |
| Torch-differentiable version of any of the above (needed only from Part 6 onward) | `payne_zero_diffatm/grid_math.py`, `twin_eos.py` (T1–T2, verified) | Rosseland-mean finalization and hydrostatic integration have **no torch port at all** — the one genuinely missing piece if/when a differentiable reduced-state pipeline is needed later |

## 5. Open items carried forward from prior sessions

Unchanged from `solver-in-the-loop-prior-work.md` §5 and
`solver-in-the-loop-progress.md` §5/§10 — not addressed by this round
(Parts 0–3 only):

- Obtaining Ting's `complete_atmosphere_latent_checkpoint.pt` / `PLAN.md` (not in this checkout).
- Resuming Stage 1b (`payne_zero_diffatm` T3–T8) for the differentiable twin — only needed once gradient-based training (Part 6+) is in scope.
- The honest form of "spectral parity" (generate spectra from converged atmospheres, refit with the emulated-atmosphere forward model, check the labels come back) — not yet tested anywhere.
- Deciding, after this round's Part 2/3 results, whether "predict τ" (extending the boundary rather than resampling the interior) is worth a dedicated experiment — the continuity harness explicitly flags this as untested and the natural next step on the boundary side.
