# Extensions in This Branch Compared with Upstream Payne Zero

Snapshot date: 2026-09-01

Branch: `codex/sync-20260826`

Comparison base: `tingyuansen/payne-zero` at `origin/main@9c44001`

## Overview

Upstream Payne Zero already computes one-dimensional LTE stellar atmospheres
and synthetic spectra. This branch does not replace its production defaults.
It adds a research layer for comparing alternative atmosphere initializers
with the same certified solver, studying a reduced two-field atmosphere
representation, developing first-principles analytic warm starts, testing
bounded extensions toward M stars, and training through a differentiable
PyTorch twin.

## Main additions

| Area | Added in this branch | Main locations |
|---|---|---|
| Reproducible benchmarks | Environment capture, label sampling, perturbations, reference solves, restart comparisons, and machine-readable reports. Alternative initializers are compared with the same Payne Zero solver and stopping rules. | `bench/`, `continuity/` |
| Two-field atmosphere representation | Resampling, physical reconstruction, training, prediction, restart, and solver adapters for `(m,T)(tau)`. Pressure, electron density, opacity, and radiative acceleration are rematerialized through existing physics. | `reduced_state/`, `experiments/reduced_state_emulator/` |
| Learned warm starts | One-step residual models, solver-in-the-loop training, physical labels, candidate blending, independent validation, resolution controls, and spectral gates. | `experiments/reduced_state_emulator/` |
| Analytic warm starts | Hopf, grey-atmosphere, convective, cumulative-optical-depth, entropy, hydrostatic-polytropic, and multi-arm comparison candidates. Every candidate is ultimately tested with the real solver. | `experiments/analytic_initializer/` |
| Textbook opacity experiments | v2/v3 were extended through v4-v4r6 with local Saha ionization, metal electron donors, H-minus and hydrogen/helium continua, scattering, frequency-node Rosseland harmonic means, and hydrostatic integration. Oracle-only quantities remain diagnostic. | `experiments/analytic_initializer/textbook_opacity.py` and associated runners |
| Bounded M-star work | Native MARCS-node loading, `(m,T)`-only rematerialization, temperature continuation, fixed M-dwarf/M-giant cases, cool-corpus construction, candidate evaluation, and a preregistered MARCS-seeded v1r2 protocol. | `experiments/reduced_state_emulator/m_star_*`, `cool_star_step_test.py` |
| Differentiable solver twin | PyTorch implementations of the initializer, EOS, continuum, lines, radiative transfer, and temperature correction for gradient-based training. This twin is not the certified production solver. | `payne_zero_diffatm/` |
| Research interfaces in the production solver | Controlled warm-start injection, temperature-correction policies, convergence diagnostics, and research run configuration without changing default routing. | `payne_zero_atmosphere/` |
| Evidence visualizations | Reusable plotting and report-generation tools for initializer comparisons and failure diagnostics. | `payne_zero_figures/reports/` |
| Tests | Additional coverage for analytic initializers, two-field reconstruction, the differentiable twin, M-star workflows, and registered controls. | `tests/` |

## Key results so far

### Two-field representation and warm starts

The branch separates prediction from physical convergence:

- a network or analytic construction supplies only the initial state;
- dependent fields are rebuilt with Payne Zero physics;
- success is decided by the unchanged solver, full flux diagnostics, and
  prospectively fixed gates;
- development, diagnostic, post-hoc, and sealed-holdout evidence are kept
  separate.

These results support specific representation and warm-start claims on their
tested samples. They do not show that every finite output is physically valid,
and they do not authorize replacing the production initializer.

### v4r6 analytic warm start

Under a matched 60-iteration budget on development-60:

| Initializer | Total converged | Cool | Hot |
|---|---:|---:|---:|
| Decoupled `m_grey + T_conv` | 54/60 | 24/27 | 30/33 |
| Grey `m_grey + T_grey` | 52/60 | 21/27 | 31/33 |
| Coupled convective | 49/60 | 18/27 | 31/33 |

The decoupled initializer passed every absolute warm-start gate. Its paired
cool-star net gain over the grey arm was `+3`, however, below the preregistered
requirement of `>= +4`. The final status is therefore
`STOP_POLICY60_MATCHED_DEVELOPMENT`. The threshold was not relaxed, and no
fresh-open sample, spectral promotion, production switch, or sealed holdout was
authorized.

### Bounded M-star work

- On eight fixed solar-composition native MARCS M-star nodes, the direct MARCS
  `(m,T)` start converged in 5/8 cases and the primary temperature-continuation
  route converged in 4/8. This is evidence for eight representative points, not
  broad M-star parameter-space support.
- M-star emulator v1 and v1r1 both stopped before training because the frozen
  cool-corpus gate failed. The final training split contained 14 giants and
  zero dwarfs. No candidate network, sealed track, old-domain retention test,
  Korg comparison, or production routing change followed.
- v1r2 is currently a preregistered MARCS-seeded 100-row protocol. MARCS is
  used only as an initializer; admitted training truth must be a converged
  Payne Zero result. A preregistration is not a completed validation.
- M-dwarf solver repairs (2026-09-23, opt-in, all off by default): the
  convective inner loop wrote ∇ with a one-sided difference while MLT reads
  it centred (flux overshoot 1.7–1.95× in the H₂ zone); its physics callback
  kept round-input density and populations at trial temperatures; the global
  correction re-ranked column mass in inner-loop layers (∇ perturbed by 13%
  of δ); round-to-round steps needed under-relaxation (λ = 0.5); and the D
  start carries a two-layer subadiabatic hole inside the deep convective zone
  that radiation cannot carry, where the radiative correction took +110–137 K
  steps. With all five repairs, both 10-round development trajectories descend
  monotonically in input-state deep-window mean |R_raw|: A 0.1698 → 0.0045,
  D 0.4349 → 0.0084 (D previously failed to converge in every arm: S0, S3w
  and S3wr diverged, S3 stalled near 0.8).
  Development sweep (five points, candidate against a paired S0 on the pchip
  overlay, strict 60-iteration solves, frozen flux gate and dual-path TiO):
  candidate eligible at 4/5 (flux-error p95 0.30–0.67%), S0 at 2/5; the 3200 K
  point fails restart column-mass consistency in both arms. Validation on the
  eight held-out safezone v2 dwarf points from the frozen emulator warm starts:
  candidate eligible at 7/8 (p95 0.24–1.17%), S0 at 2/8. Start independence
  (warm start against the certified reference product as a second start)
  holds at 4/7; the three failures are marginal, in upper-atmosphere column
  mass and in temperature where TiO forms. At g4.50 m+0.5 3600 K the candidate
  fails from both starts and moves away from S0's converged solution. Not
  used for library products. Chain of records:
  `notes/m_star_platform_decomposition_20260923.md` (correction section) →
  `notes/m_star_inner_loop_fill_holes_trajectories_closeout_20260923.md` →
  `notes/m_star_inner_loop_dev_sweep_closeout_20260923.md` →
  `notes/m_star_inner_loop_validation_closeout_20260924.md`.

## Scientific and production boundaries

The following remain unchanged:

- the three upstream initializer families and default production routing;
- the requirement that convergence, flux closure, spectral parity, domain
  coverage, and production readiness be evaluated separately;
- failed preregistered thresholds, which were not moved after results were
  observed;
- the status of the differentiable twin, analytic initializers, and M-star
  candidates as research paths rather than production defaults.

## What this GitHub snapshot contains

The branch includes source code, tests, preregistrations, closeout records,
report plotting, and selected compact machine-readable results.

The following remain local or on the compute nodes:

- large `runs/` products and full solver trajectories;
- complete M-star NPZ products and training corpora;
- `SDSS_MARCS_atmospheres.h5`;
- the historical `archive/`;
- downloaded references, reference-check images, build caches, and local
  editor or agent state;
- the private Paper II manuscript and its dedicated figures.

Replaying every numerical campaign from scratch therefore requires retrieving
the data identified by the hashes and paths in the corresponding
preregistration and closeout records.
