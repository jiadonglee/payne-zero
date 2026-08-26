# 32-layer physical homotopy pilot

Date: 2026-08-22

## Scope

This is a development-only initializer experiment. It does not replace the
production solver, does not use a training target or emulator checkpoint, and
does not open the sealed holdout.

The pilot uses:

- 32 logarithmic Rosseland-depth points;
- four equal-measure frequency groups formed from the real continuum-opacity
  sampling grid;
- the existing EOS/population path and continuum opacity kernels;
- a two-stream transfer update;
- two damped radiative updates followed by two damped MLT/EOS-convection
  updates;
- interpolation of only `(m, T)` to the standard 80-layer solver grid.

The current selected/detailed line-opacity kernels require exactly 80 layers.
The coarse pre-solver therefore defers line opacity to the unchanged exact
solver. This is an explicit limitation of pilot v0, not a claim of full
line-opacity closure.

## Fixed gate

Use the existing 12-star smoke list and the same solver configuration as the
`parity`/production controls. Count first-trial convergence only. The pilot
passes the smoke gate only if it reaches 11/12, has no timeout or non-finite
final state, and records four physical evaluations per star. The seed builder
must remain emulator-free and must emit finite positive, strictly increasing
`m`.

If the gate fails, stop before a development-60 run. If it passes, the next
run is a paired development-60 comparison; the sealed holdout remains closed.

## Executed v0 result

The seed-only structural check passed on all 12 fixed stars:

- 12/12 finite and strictly increasing after resampling to 80 layers;
- four physical evaluations per star;
- no emulator checkpoint was loaded.

The real-solver gate was stopped as soon as it became impossible to pass:

- index 2891: finite final state, but not converged within 15 iterations;
- index 6896: timeout under the 120 s diagnostic protection.

The existing paired `parity` control covers both rows on the identical smoke
list and converges them in 7 and 6 iterations, respectively. This makes the
v0 failure an initializer-quality failure on these rows, rather than evidence
that the solver was globally hard or unavailable.

The gate therefore fails before a development-60 run. The early-stop result is
`results/analytic_initializer/physical_homotopy_gateB_early_stop.json`; the
12-star structural result is
`results/analytic_initializer/physical_homotopy_seed12.json`.

This closes pilot v0 as a solver initializer. It does not show that the
physics-driven idea is impossible; it shows that this particular continuum-
only, four-group two-stream approximation is not yet inside the required
basin. No tuning or sealed-holdout run follows this failed gate.
