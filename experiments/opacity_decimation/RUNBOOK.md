# Experiment A — frequency-grid decimation

Measures how much κ_ν frequency resolution the converged atmosphere actually
needs. The output is an **error budget**, not a speedup: it is the number an
opacity emulator would have to hit, and it decides whether one is worth
building. See `experiments/opacity_error_budget/README.md` for the
pre-registered decision criterion.

## What was changed

`build_opacity_sampling_grid` (`payne_zero_atmosphere/continuum_opacity.py`)
takes `frequency_grid_stride`, surfaced as `AtmosphereConfig
.opacity_frequency_grid_stride` (default `1`). `runner.py` differs from
production by **two lines**: the call site and one diagnostic field. Everything
else — trial policy, initializer, convergence test, record format — is
`bench.run_reference` unchanged.

**Stride 1 is bit-identical to the production grid.** Verified independently of
the test suite across all nine effective-temperature branches of the edge-start
selector (3500 / 4000 / 5777 / 7000 / 8000 / 10500 / 12000 / 20000 / 35000 K),
on both wavelengths and weights, with `np.array_equal`.

### The one correction applied to the first weight

The undecimated first weight is `1.5 * (nu[0] - nu[1])`. That factor is two
things added together: a half cell of the sampled grid, which must scale with
the stride, and a full extra cell standing in for the band just blueward of the
grid, which represents a *fixed physical band* and must not.

Scaling both drifts the total quadrature measure by exactly
`(stride - 1) * 2.3025e-4` — measured `+2.302e-4 / +6.904e-4 / +1.610e-3` at
strides 2 / 4 / 8 — entirely through that one weight. Since this experiment
exists to isolate the cost of lost frequency resolution, a boundary term
growing linearly with the stride would confound precisely the quantity being
measured. The stand-in is therefore taken on the undecimated grid.

| stride | n points | measure drift, scaled stand-in | measure drift, fixed stand-in |
| ---: | ---: | ---: | ---: |
| 1 | 30000 | 0 | 0 |
| 2 | 15001 | +2.302e-4 | 0 |
| 4 | 7501 | +6.904e-4 | −1.15e-7 |
| 8 | 3751 | +1.610e-3 | −3.46e-7 |
| 16 | 1876 | +3.447e-3 | −8.08e-7 |

Residual drift is now the second-to-last sampled point only. Stride 1 stays
bit-identical because the two terms sum back to the original `1.5`.

## Running

Environment is not optional: `NUMBA_THREADING_LAYER=workqueue` or the solver
segfaults in the opacity stage with no traceback, and `.venv/bin/python` — the
conda base interpreter cannot import numba.

Single star, the whole ladder:

```bash
NUMBA_THREADING_LAYER=workqueue PYTHONPATH=. .venv/bin/python \
  -m experiments.opacity_decimation.run_decimation \
  --strides 1 2 4 8 \
  --effective-temperature 5777 --log-surface-gravity 4.44 \
  --metallicity 0.0 --alpha-enhancement 0.0 --microturbulence-km-s 1.0 \
  --out runs/opacity_decimation_smoke \
  --summary runs/opacity_decimation_smoke/summary.json
```

The regime-stratified 12-star set is `experiments/opacity_error_budget/labels.jsonl`.

## Sizing

Peak RSS is ~7.8 GB per process, briefly ~14 GB at startup when
`line_selection.py` concatenates the three 1.4 GB catalog shards. Worker sizing
is `RAM_GB / 16`, so a 16 GB machine runs **exactly one** — never launch a pool
there. Measured on the dev machine: the Sun at stride 1 converged in 4
iterations in **99.8 s wall**, which includes the one-time catalog load.

## Reporting

Per star and stride: iterations to convergence, final
`deep_layer_relative_temperature_change`, per-iteration `opacity_seconds`,
total wall time, and the converged `temperature` / `column_mass` against that
star's own stride-1 result.

Two things must be reported separately rather than folded into one speedup
number:

1. **iteration-1 line selection**, which maps lines into frequency bins and so
   changes with the stride;
2. **per-iteration opacity accumulation**, the recurring cost.

The comparison bar is the solver's own start-dependence — median normalized
spectrum `3.44e-3`, max `1.50e-2`, 16/56 stars over `5e-3` — not zero. A
decimated run inside that band is not distinguishable from ordinary solver
noise.
