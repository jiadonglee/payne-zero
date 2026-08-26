# Parity + EOS polytrope handoff

## Decision

The one-shot EOS-driven polytropic projection is vetoed at Gate 1. The first
two stars in the frozen 12-star smoke both reached the 15-iteration limit
without convergence. The parity baseline converged on all 12 stars. Since the
Gate 1 requirement permits at most one lost convergence, the maximum possible
polytrope result was already 10/12 and could not pass. The remaining smoke
stars were not run, and the 60-star funnel was not opened.

This is a solver-stability failure, not a non-finite-state failure. Both
completed polytrope states were finite and the projection itself was physical
and numerically consistent: one contiguous EOS-selected component in each
case, trapezoidal adiabatic residual below `9e-14`, and no fitted closure
parameters. The first two projections changed the deep profile by about
`0.504` in `|delta ln T|`.

## Implemented files

- `experiments/analytic_initializer/eos_polytrope.py`: pure EOS projection,
  post-iteration pressure construction, and one-shot hook.
- `payne_zero_atmosphere/runner.py`: private after-iteration hook; the public
  `run_atmosphere_model(config)` wrapper and default path remain unchanged.
- `experiments/analytic_initializer/run_h2_solver_funnel.py`: new
  `parity_polytrope` arm and per-star projection diagnostics.
- `tests/test_analytic_initializer_eos_polytrope.py`: Gate 0 mathematical and
  dependency tests.
- `notes/parity_eos_polytrope_plan.md`: fixed method and gates.

## Verification

- Gate 0 focused suite: `92 passed`.
- Runner import through `.venv`: passed.
- `git diff --check`: passed.
- Parity smoke: 12/12 converged, 12/12 finite, 0 timeout.
- Polytrope smoke: 0/2 completed stars converged, 2/2 finite, 0 timeout;
  early stop after arithmetic Gate 1 veto.
- Detailed streamed records: `results/analytic_initializer/funnel12_parity_polytrope.jsonl`.
- Decision record: `results/analytic_initializer/funnel12_parity_polytrope.json`.

## Prohibited next steps

Do not tune damping, crossing thresholds, transition widths, fixed gamma, or
fitted amplitudes after this veto. Do not run the 60-star funnel or alter the
production default. The result does not justify productizing
`initializer="analytic-parity-polytrope"`.

If the research question is reopened, it needs a new preregistered physical
formulation. The next plausible direction is a solver-in-the-loop entropy or
temperature correction that uses the solver's full energy/flux balance, rather
than imposing a full deep adiabatic profile after one iteration.
