# Physics-driven initializer: next-step audit

Date: 2026-08-22

## Decision

The coarse physical homotopy is closed as a solver-initializer candidate. The
low-dimensional physical-residual version is also closed as a two-star
prototype. Neither result opens development-60 or the sealed holdout.

## Evidence

- The original 32-layer, four-group pilot passed the structural seed check on
  12/12 stars, but the real-solver gate failed immediately: 2891 did not
  converge in 15 iterations and 6896 timed out.
- A transfer audit found a real discretization defect: midpoint source
  propagation across the deepest logarithmic cells created a roughly 225x
  spurious deep flux. Linear source integration and a diffusion lower boundary
  remove that defect in a grey benchmark.
- Three bounded follow-up variants were tested without opening a larger
  sample: corrected transfer, joint mass relaxation, and mass-first then
  temperature relaxation. The 2891 real-solver check still timed out within
  120 s, with downstream EOS/molecular overflow warnings.
- The reduced physical-residual prototype uses 12 coefficients and the real
  EOS/continuum/4-group transfer. On 2891 and 6896 it reduced residual RMS from
  1.652 to 0.315 and from 1.453 to 0.336, respectively, while preserving
  finite positive monotone `(m,T)`. This is a residual improvement only: both
  2891 and 6896 still timed out in the unchanged 80-layer solver within 120 s.

## Interpretation

The residual objective is not yet aligned with the production solver's basin.
Residual reduction cannot be promoted to initializer success. A cluster-side
80-layer handoff audit now separates the two possible failure modes:

- For both 2891 and 6896, the exact line slab, continuum, and population
  arrays remained finite after one unchanged production iteration. The line
  opacity handoff is therefore not failing because of a missing catalog or a
  non-finite line kernel.
- The grey seed stayed positive after that iteration. The physical-residual
  seed instead produced negative temperatures (`T ~ -71 to -8 K`) after the
  first exact correction for both stars. It is finite numerically but not an
  acceptable atmosphere state.
- A grey seed followed by one exact production iteration improved the 15-step
  trajectory for both stars, but neither arm passed the 15-iteration gate.
  The residuals at 15 steps changed from 0.0277 to 0.0136 for 2891 and from
  0.00949 to 0.00573 for 6896 (direct to preconditioned).
- On a longer trajectory, 6896 converged at production iteration 26 after the
  extra preconditioning step. For 2891, the 30-step preconditioned run was
  still at `5.32e-4`; a paired 35-step diagnostic converged at production
  iteration 32 for direct-grey and 31 for preconditioned-grey. Because the
  preconditioner itself costs one exact iteration, this is not a <=15-step
  initializer.

The remaining technical boundary is therefore the exact solver's correction
policy and basin placement, not a missing polynomial degree or an absent line
opacity table. Changing that policy would be a separate solver experiment,
not another initializer fit.

## Cluster correction-policy audit

An opt-in global temperature-correction damping factor was tested without
changing the default value (`damping=1`). The audit used the same two stars and
the same unchanged line-opacity solver:

- At `damping=0.5`, the residual handoff became acceptable for 2891 but not
  6896; the latter still ended at negative temperatures.
- At `damping=0.25`, all four one-iteration handoff rows (grey and residual
  for both stars) were finite and positive.
- The qualifying `damping=0.25` grey direct/preconditioned 15-step run still
  had no converged row. Deep-layer changes were about `0.0080--0.0089`, while
  p95 flux errors were `198--247%`. At the original damping, the corresponding
  p95 flux errors were `2.0--104%`.

Thus global correction damping can prevent one handoff failure in this small
sample, but it damages flux balance and does not satisfy the original gate.
This branch is closed; further global damping tuning is not justified.

## Artifacts

- `experiments/analytic_initializer/physical_homotopy.py`
- `experiments/analytic_initializer/physical_residual_initializer.py`
- `experiments/analytic_initializer/run_physical_residual_smoke.py`
- `results/analytic_initializer/physical_homotopy_gateB_early_stop.json`
- `results/analytic_initializer/physical_residual_seed2.json`
- `results/analytic_initializer/physical_residual_gate_2891.json`
- `results/analytic_initializer/physical_residual_gate_6896.json`
- `results/analytic_initializer/physical_handoff_audit_v2.json`
- `results/analytic_initializer/exact_preconditioned_smoke.json`
- `results/analytic_initializer/exact_preconditioned_trajectory30.json`
- `results/analytic_initializer/exact_preconditioned_trajectory35_2891.json`
- `results/analytic_initializer/physical_handoff_damping05_v2.json`
- `results/analytic_initializer/physical_handoff_damping025_v2.json`
- `results/analytic_initializer/exact_preconditioned_damping025.json`

No further coefficient tuning or dev-60 run follows this audit. The next
bounded experiment, if pursued, must preserve flux balance while enforcing
positive remapped states; it should be compared against the same direct-grey
control and should not be described as a new initializer success unless it
passes the original 15-iteration gate.
