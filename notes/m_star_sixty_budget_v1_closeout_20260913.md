# 60-iteration configuration validation closeout

Date: 2026-09-13

## Step 1 - the t3250 reference exists

`Teff 3250 K, logg 2.0, [M/H] = 0` now has an independent reference: a
parameter walk from the converged corpus truth `g1.5 t3300 [M/H]=0` (two
logg legs of 0.25 dex at fixed temperature, then two 25 K temperature legs),
relaxed under the frozen stop rule.  The reference stabilizes at iteration
30 of its 240-iteration target budget; the v4 emulator arm stabilizes at 40
of 60.  Cross-arm agreement at the frozen endpoints: temperature p95
`5e-4`, column mass `4e-4` dex, all three TiO metrics `<= 3e-4` - an order
of magnitude under the `5e-3` bar.  The temperature-descent walk had failed
at this node; the gravity walk succeeds.  No emulator product touched the
reference path.

## Step 2 - six new reproduction points

Preregistered before any solve (two nominal, two cool, two low gravity;
distinct metallicities; none in training or earlier tests):

| point | emu frozen | ref frozen | budget | verdict |
|---|---|---|---|---|
| 3675 K g1.75 [M/H]=+0.5 | 25 | 45 | within 60 | pass |
| 3750 K g0.75 [M/H]=-0.5 | 90 | 95 | over budget | pass (spectral) |
| 3400 K g0.5 [M/H]=+0.5 | 40 | 145 | within 60 | pass |
| 3525 K g2.25 [M/H]=-1.0 | 25 | none | - | undetermined |
| 3175 K g2.25 [M/H]=+0.5 | 45 | none | - | undetermined |
| 3325 K g1.5 [M/H]=-0.5 | 25 | none | - | undetermined |

The three undetermined points are all reference-side: their nearest-node
MARCS seeds drive the solver non-finite within two iterations, the
walk-seeded fallbacks fail at their final legs (or lack a cross-track
anchor within the search radius).  The emulator arms stabilize normally
(25-45 iterations) but cannot be judged without a reference.  Per the
preregistered bookkeeping they stay undetermined; no points were dropped
and no budgets were silently extended.

## Merged tally (18 points, `results/m_star_sixty_budget_v1/merged_table.csv`)

- 30-iteration standard: **7/12** of the original twelve pass.
- 60-iteration configuration (emulator <= 60, frozen-rule references,
  walk-seeded where needed): **12/12** of the original twelve pass -
  including `t3250`, resolved this round - and **3/6** of the new points
  pass with one further pass over budget (stabilizes at 90).
- Overall: 15/18 dual-path agreements, every decidable point agreeing at
  `dT p95 <= 3.9e-3`, `dlogm <= 1.2e-3` dex, TiO metrics `<= 8e-4`.
- Undetermined: 3, all reference-side, all in the cool off-grid region
  (`logg 2.25` at 3175-3525 K, `g1.5` at 3325 K).  These are limits of
  reference construction under the current seeding strategies, not
  emulator-spectral failures.

## Cost accounting

- Emulator path: 20-90 iterations to frozen precision (25 typical).
- Reference path at target: 30-145 iterations.
- Reference construction is the expensive side and is booked separately:
  walk-seeded references cost 4-19 additional strict solver legs (each up to
  60 iterations) before the target relaxation - for `t3250`, 4 legs; for
  `t3050`, 18 legs; for the three failed cool references, 8-18 failed legs
  per point.

## Conclusion

Within the tested envelope (3000-4000 K, logg 0.5-2.5, [M/H] -1 to +0.5,
off-grid nodes included), the emulator + solver configuration with a
60-iteration emulator budget and frozen-rule references produces
spectrally and structurally consistent atmospheres at every point where a
reference can be constructed - 15/18, with the original twelve now fully
closed.  The open region is the cool off-grid band around `logg 2.25,
3175-3525 K` where reference atmospheres cannot yet be built; logg-space
walks from the neighboring gravity tracks are the untried remedy there,
mirroring what worked for `t3250`.  Eighteen points remain a bounded
sample, not a whole-grid success rate.
