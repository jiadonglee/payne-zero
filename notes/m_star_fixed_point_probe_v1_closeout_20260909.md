# Fixed-point continuation probe closeout

Date: 2026-09-09

## Question

The v3 TiO gate passed 7/9; `t3850` and `t3900` (`g2.0 [M/H]=0`) missed at
`1.36-1.40x` the bar. This probe tests whether that offset is a loose
stopping condition, a genuine emulator fixed-point offset, or an unstable
truth reference. Both arms - the emulator endpoint and the truth product -
were continued past their formal stops with the stopping rule released
(physics untouched), first to ten iterations, then to twenty; checkpoints at
iterations 0, 1, 3, 5, 10, 15, 20; `t3950` ran as the passing control.

## Result

`results/m_star_fixed_point_probe_v1_k20/probe.json`,
figure `figures/m_star_fixed_point_probe_v1_k20.png`
(the ten-iteration run is `results/m_star_fixed_point_probe_v1/`).

- Continuing alone removes the misses: `t3850` crosses the `5e-3` bar near
  `k=8` (`4.53e-3` at k=20), `t3900` reaches it near `k=20` (`4.99e-3`);
  the control falls `4.40e-3 -> 2.57e-3`. No retraining involved.
- The two arms converge onto one another: candidate-vs-live-truth in the TiO
  window collapses to `2.8-3.7e-4` at k=20.
- Flux residuals fall monotonically in both arms, with no oscillation. At
  their formal stops the truth products carried the larger residual
  (`p95 1.0-1.3%`) and the emulator endpoints the smaller one
  (`0.44-0.65%`); by k=20 both sit at `0.10-0.21%`.
- The truth arm keeps creeping after the emulator arm has saturated
  (max |dT/T| from own start at k=20: truth `0.21-0.22%` and still rising,
  emulator saturated at `~0.05%`).

## Reading

The spectral offset is at least partly caused by the iterations stopping too
early: continuation alone removes it, and both arms relax onto one and the
same state. What can be said additionally is directional, not an
exoneration: at their formal stops the truth targets were the
under-relaxed side, so the gate compared the candidate against a reference
that was itself `~4.3-4.6e-3` away (in the TiO window) from where both arms
settle. A systematic emulator component is not ruled out by this probe - the
truth arm was still moving (`~2e-4` updates, falling flux residual) at k=20,
and the evidence covers one window, three stars, one (logg, [M/H]) track.

## Practical consequence

Before adding training data for the two misses, align the tolerance on both
sides - e.g. a fixed number of forced post-convergence iterations for truth
generation and gate evaluation alike, or a tightened all-layer criterion.
The remaining uncertainty in the attractor location (`~1e-3` scale) is the
next thing to pin down if the bar is to mean more than the reference's own
convergence error.
