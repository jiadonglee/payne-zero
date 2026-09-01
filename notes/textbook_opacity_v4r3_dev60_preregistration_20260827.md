# Textbook opacity v4r3 development-60 solver funnel

Date: 2026-08-27

This is a user-requested solver-basin diagnostic. It does not reopen or
rewrite the v4r3 offline `FAIL_STOP`. It does not change the production
solver, default initializer, or sealed holdout. Mean iteration count is
reported and is not a gate.

## Why this task exists

The registered v4r3 offline validation
(`notes/textbook_opacity_v4r3_preregistration_20260827.md`, JSON SHA-256
`04432ac37667fb7ad1f49de23ef72e6de2cd94b419c8e2959abf2ffecef867cc`) failed the
middle opacity gate and both true-`(P,T)` mass gates. The stop rule therefore
blocked the ODE, 12-star smoke, and 60-star funnel.

The user then asked to send v4r3 into the same development-60 solver test used
by analytic-parity, and stated that mean iterations are not the decision
quantity. This task answers that request. An offline mass miss is not a
solver-basin measurement; the funnel is.

## Frozen construction

Keep the v4r3 opacity law unchanged: ideal-gas particle-count density, H2+
and He-minus continua, seven-donor Saha, John H-minus, no molecular bands,
no stored `n_e` input, no corpus fit. The solver seed is:

```text
T  = Eddington-grey, then Saha-aware adiabatic replacement where nabla_rad > nabla_ad
m  = RK4 integral of dm/dtau = 1 / kappa_v4r3(T, g m), 8 substeps per layer
kappa_seed = kappa_v4r3 evaluated at that (T, P=g m)
```

The formal v4r3 domain remains `T >= 4000 K`. Colder photospheric layers are
still evaluated and handed to the solver; they are not clipped. Helium is not
added as a Saha donor.

## Registered sample and solver policy

Use the frozen paper development-60 indices from
`results/paper_physical_seed_20260820/learned/convergence_metrics_learned_monotone.json`
(60 stars, split seed `20260816`). This is the same list as analytic-parity,
so the rows pair. Do not draw a new funnel.

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

Compare descriptively against the frozen analytic-parity and production
development-60 records. A high or low mean iteration count does not decide
this task. Production remains the default initializer regardless of the
outcome.

## Stop rule for this task

This task stops after the development-60 funnel JSON is written. It does not
run flux/spectral checks, the 200-star sealed holdout, or a production
switch. The offline `FAIL_STOP` remains the opacity-physics record.

The registered output is
`results/analytic_initializer/textbook_opacity_v4r3_dev60_20260827.json`.
The streamed records are the sibling `.jsonl`.

## Remote execution

The funnel runs on `astronode-garching` (Node-06), checkout
`/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero` (same tree as
`/home/jdli/xiasangju/jdli/payne-zero`). Parent PID `1452834`, solver
worker spawned under that process. Log:
`logs/textbook_opacity_v4r3_dev60.log`. The local laptop run was stopped
after four stars so the node owns the full 60.
