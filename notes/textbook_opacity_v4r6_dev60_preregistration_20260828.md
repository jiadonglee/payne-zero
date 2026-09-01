# Textbook opacity v4r6 development-60 solver funnel

Date: 2026-08-28

This is a user-requested solver-basin diagnostic. It does not reopen or
rewrite the v4r6 offline `FAIL_STOP`. It does not change the production
solver, default initializer, or sealed holdout. Mean iteration count is
reported and is not a gate.

## Why this task exists

The registered v4r6 offline validation
(`notes/textbook_opacity_v4r6_preregistration_20260828.md`, JSON SHA-256
`ad58d6da5ec046401b55655ca60b96cb6e123f12111e330417228ffdeda4909b`) passed
both opacity gates and failed both true-`(P,T)` mass gates (cool 0.2468,
middle 0.2074, limit 0.20). The stop rule therefore blocked the ODE, 12-star
smoke, and 60-star funnel.

The v4r3 development-60 funnel
(`notes/textbook_opacity_v4r3_dev60_preregistration_20260827.md`, JSON
`textbook_opacity_v4r3_dev60_20260827.json`) converged 25/60 with all
failures at `Teff < 7500 K` (0/27). That run predates the v4r5 ground-anchor
H I population repair and the v4r6 per-n threshold repair; both repairs move
the seed on exactly the mid/hot layers where the v4r3 funnel already worked,
and neither touches the `T < 4000 K` surface column where cold seeds fail.

The user's stated goal is a two-field analytic initializer, for which the
solver basin is the direct measurement. The user asked to send v4r6 into the
same development-60 solver test, in parallel with the offline residual
diagnostics. An offline mass miss is not a solver-basin measurement; the
funnel is.

## Frozen construction

Keep the v4r6 opacity law unchanged: per-n hydrogenic thresholds with
published n=1/2/3 edges (6.30e-18, 1.40e-17, 2.16e-17 cm2) below 15000 K and
v4r5 hydrogen above, ground-anchored H I populations, ideal-gas
particle-count density, seven-donor Saha, John H-minus, H2+, He-minus, no
molecular bands, no stored `n_e` input, no corpus fit. The solver seed is:

```text
T  = Eddington-grey, then Saha-aware adiabatic replacement where nabla_rad > nabla_ad
m  = RK4 integral of dm/dtau = 1 / kappa_v4r6(T, g m), 8 substeps per layer
kappa_seed = kappa_v4r6 evaluated at that (T, P=g m)
```

The formal v4r6 domain remains `T >= 4000 K`. Colder photospheric layers are
still evaluated and handed to the solver; they are not clipped. Helium is not
added as a Saha donor.

## Registered sample and solver policy

Use the frozen paper development-60 indices from
`results/paper_physical_seed_20260820/learned/convergence_metrics_learned_monotone.json`
(60 stars, split seed `20260816`). This is the same list as analytic-parity
and the v4r3 funnel, so the rows pair. Do not draw a new funnel.

Solver policy matches the formula arms:

- one trial, 15 iterations, 900 s per-star timeout;
- seed the unchanged production solver through `analytic_seed_model`;
- stream JSONL as each star lands; resume is allowed;
- no spectral gate, no 12-star smoke requirement, no sealed holdout.

## Reported quantities, not gates

Record, and do not promote into a pass/fail cutoff in this task:

- seed construction finite count;
- seed versus stored truth `T` relative p50/p95 and `log m` p50/p95;
- converged / timeout / error counts;
- iterations on converged stars, including the mean.

Compare descriptively against the frozen analytic-parity, production, and
v4r3 development-60 records, including the `Teff < 7500 K` split that decided
the v4r3 funnel. A high or low mean iteration count does not decide this
task. Production remains the default initializer regardless of the outcome.

## Stop rule for this task

This task stops after the development-60 funnel JSON is written. It does not
run flux/spectral checks, the 200-star sealed holdout, or a production
switch.

Output:
`results/analytic_initializer/textbook_opacity_v4r6_dev60_20260828.json`

Runner:
`experiments/analytic_initializer/run_textbook_opacity_v4r6_dev60.py`

Log: `logs/textbook_opacity_v4r6_dev60.log`

## Remote execution

Host `astronode-garching`. Checkout
`/home/jdli/xiasangju/jdli/payne-zero`. Python `.venv-linux/bin/python`.
Do not evaluate production opacity from macOS `.venv`.

## Post-run (2026-08-28)

JSON SHA-256
`c0c08c9727e522916085941bc5dcb40a96d67fea05852f8d88ddb4cae4cdd3e5`

20/60 converged (mean 10.7 iterations on converged stars), 3 timeouts, 0
errors. `Teff < 7500 K`: **0/27**. `Teff >= 7500 K`: 20/33. All 60 seeds
finite. This is the grey-plus-adiabatic control for the v4r6-grey
(`include_convection=False`) diagnostic. Production remains the default
initializer.
