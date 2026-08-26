# Experiment B — opacity lagging

## Why

Opacity is 79.1% of the solve
(`results/solver_in_loop_k1_hard5_linear/real_solver_comparison.json`), and its
per-iteration cost is flat while the state stops moving:

| iteration | opacity_seconds | deep_layer_relative_temperature_change |
| ---: | ---: | ---: |
| 2 | 1.853 | 3.075e-3 |
| 3 | 1.847 | 1.478e-3 |
| 4 | 1.868 | 9.751e-4 |
| 5 | 1.874 | 3.019e-4 |

The solver pays full price to rebuild an 80x30000 opacity on iterations where
the atmosphere barely moves. This is the classical Kantorovich lagged-operator
situation.

**Unlike decimation and unlike an emulator, lagging introduces no approximation
at the fixed point.** That is now the reason it matters: experiment A found that
frequency decimation has no safely exploitable stride (`s* = 1` for 6 of 12
stars, and nothing in the labels predicts which — see
`experiments/opacity_error_budget/README.md`). Lagging is the only speedup on
the table that does not trade accuracy for time.

## The invariant

Enforced in one place, `convergence.evaluate_convergence_stop`:

> `converged` is never `True` for an iteration with `opacity_recomputed=False`.

A lagged iteration measured its temperature change against an opacity operator
built from an earlier atmosphere, so its residual is not evidence about the true
fixed point. The policy makes a lagged iteration **unable to create confidence
but still able to destroy it**: it never increments the consecutive-converged
counter, it still resets that counter when the state is visibly moving, and when
it *looks* converged it raises `force_exact_opacity` so the next iteration
re-tests the candidate against the true operator.

With lagging off, `opacity_recomputed` is always `True` and the policy reduces
branch for branch to the historical one.

## Verified

- 29 unit tests, `tests/test_opacity_lagging.py`.
- **End-to-end bit-identity with the flag off.** The Sun re-solved after the
  merge reproduces the pre-merge converged `temperature` and `column_mass`
  bit-for-bit, same iteration count, same convergence flag. This is the
  regression that matters: the default path is untouched.

## Running

```bash
NUMBA_THREADING_LAYER=workqueue PYTHONPATH=. .venv/bin/python -m bench.run_reference \
  --labels experiments/opacity_error_budget/labels.jsonl --out runs/lagging_baseline
```

then the same with `enable_opacity_lagging=True` in the config, and compare.
`opacity_recompute_interval` defaults to 2 (recompute every other iteration).

## Do not measure the speedup on the dev machine

This is the trap experiment A fell into, and B is *entirely* a speed claim, so
it matters more here.

Wall clock is not reproducible on the 8-core dev machine. The Sun was run twice
with bit-identical physics — same iteration counts, `deep_layer_relative_temperature_change`
equal to every digit, converged arrays bit-for-bit equal — and wall times
**1.68 to 2.41x apart**, with the ratio itself varying by condition. In one run
(`r6`, 5500/4.50/-1.50) three iterations at strides 2, 4 and 8 took 11.6, 15.5
and 17.2 s: strictly decreasing work, increasing time. The OS page cache over
7.1 GB of catalogs and the numba JIT cache both move between runs and neither is
controlled here.

A speedup claim needs the cluster, a fixed warm state, repeated trials per
point, and a reported median and spread. Anything measured locally is an
anecdote.

## What to compare, and against what bar

Per star, lagged versus baseline:

1. iterations to convergence, and how many of them were lagged (the diagnostics
   record both counts);
2. the converged atmosphere, against that star's baseline.

The bar is **not** zero. Lagging changes the trajectory, so the converged
product will not be bit-identical — only the fixed point it is converging to is
unchanged. The relevant scale is the solver's own start-dependence: two
legitimate production runs of the same star differ by a median normalized
spectrum `3.44e-3`, max `1.50e-2`, with 16 of 56 stars over `5e-3`. A lagged run
inside that band is indistinguishable from ordinary solver noise.

Do not report a lagged run as "identical" and do not report it as "failed" for
being non-zero. Report where it sits against `3.44e-3`.
