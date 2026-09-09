# Fixed-point continuation probe closeout

Date: 2026-09-09

## Question

The v3 TiO gate passed 7/9; `t3850` and `t3900` (`g2.0 [M/H]=0`) missed at
`1.36-1.40x` the bar. This probe tests whether that offset is a loose
stopping condition, a genuine emulator fixed-point offset, or an unstable
truth reference. Both arms - the emulator endpoint and the truth product -
were continued ten iterations with the stopping rule released (physics
untouched), checkpoints at iterations 0, 1, 3, 5, 10; `t3950` ran as the
passing control.

## Result

`results/m_star_fixed_point_probe_v1/probe.json`,
figure `figures/m_star_fixed_point_probe_v1.png`.

- The cross-arm TiO error falls monotonically for every star:
  `t3850 6.79e-3 -> 4.81e-3` (crosses the bar near k=8),
  `t3900 6.96e-3 -> 5.28e-3`, control `4.40e-3 -> 2.85e-3`.
- The two arms converge on each other: candidate-vs-live-truth collapses to
  `1.2-1.6e-3` by k=10.
- The truth arm itself drifts more than the emulator arm
  (max |dT/T| after ten released iterations: truth `0.10-0.21%`,
  emulator `0.02-0.05%`); neither arm oscillates.

## Reading

The `5e-3` offsets are under-relaxation, not an emulator bias: both endpoints
sit on the same slow approach path, and the formal stop fires while the
spectral residual is still shrinking. The truth reference is not a fixed
point at this tolerance either - it creeps further per released iteration
than the emulator endpoint does. Fast warm-start convergence is therefore not
the sloppy side; the convergence tolerance of the training targets and the
spectral gate probe comparable scales (`a few x 1e-3`).

Practical consequence: before adding any training data for the two misses,
align the tolerance - either fix a small number of forced post-convergence
iterations on both arms, or tighten the all-layer criterion for truth
generation and gate evaluation together. Data volume is not the binding
constraint for these two points.
