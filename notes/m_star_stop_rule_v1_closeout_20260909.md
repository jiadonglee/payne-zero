# Stop-rule freeze and twelve-point validation closeout

Date: 2026-09-09

## Step 1 - frozen stop rule

The three fixed-point probe stars (both arms, from their k=20 checkpoints)
were continued forty released iterations.  Two consecutive five-iteration
segments were reached whose TiO metric changes are all below `5e-4`
(one tenth of the gate bar), temperature-change p95 below `1e-4`,
column-mass-change p95 below `1e-3 dex`, and with the frozen flux gate
passing throughout.  All six arms satisfy this by ten continuation
iterations (`results/m_star_stop_rule_v1/stop_rule.json`).  The rule was
then frozen:

> iterate until two consecutive 5-iteration segments each satisfy
> (max dTiO < 5e-4, dT p95 < 1e-4, dlogm p95 < 1e-3 dex, flux gate pass);
> budgets: emulator path 30 iterations, MARCS reference path 240.

## Step 2 - twelve dual-path validation points

Twelve preregistered off-grid points (four per group: nominal
`3500-4000 K / logg 1.5-2.5`, cool `3000-3400 K / logg 1.5-2.5`,
low gravity `logg 0.5-1.0`; each group spanning metal-poor, solar, and
metal-rich) were solved twice - emulator warm start (cap 30) and
nearest-native-MARCS initialization (cap 240) - into the unchanged solver,
then judged by the frozen rule and compared against each other.

Verdicts (`results/m_star_stop_rule_v1/validation_table.csv`):

- **7/12 pass.**  Wherever both paths reached the frozen precision, the two
  independent starts landed on the same atmosphere: cross-arm
  `dT p95 3e-4 - 4.4e-3`, `dlogm p95 <= 1.2e-3 dex`, and all three TiO
  metrics `<= 9e-4` (five times under the gate bar).  Emulator path reached
  the frozen precision in 20-30 iterations; the reference path needed
  30-125.
- **5/12 undetermined.**  Three reference arms diverged outright
  (non-positive column mass within 2-10 iterations from the nearest-node
  MARCS seed: `t3250 g2.0 [M/H]=0`, `t3050 g2.0 [M/H]=+0.5`,
  `t3850 g0.5 [M/H]=-0.5`) - the unchanged solver cannot even hold a valid
  structure from a distant node seed at these corners.  Two further points
  (`t3150 g1.5 [M/H]=-0.5`, `t3300 g0.75 [M/H]=+0.5`) had a stable reference
  (60 and 85 iterations) but the emulator path did not stabilize within its
  30-iteration budget.

## Reading

- The original convergence criterion stops at 3-19 iterations everywhere
  while the frozen precision needs 20-125 - confirming, on independent
  points, that the formal stop is the dominant source of the earlier
  spectral offsets.
- The 11/12 bar is not met as stated: 7 passes, 5 undetermined.  Within the
  decidable subset the initializer plus solver behaves consistently across
  nominal, cool, and low-gravity points (passes exist in all three groups,
  including `g0.75` and `3250-3450 K` cool nodes); the undetermined points
  cluster at the two hardest corners.
- The two emulator-unstable points (`t3150 g1.5 m-0.5`, `t3300 g0.75 m+0.5`)
  match the "warm start slow, reference stable" case - candidates for added
  training coverage.  The three reference-diverged points fall under
  "solver/seed" - they need continuation seeding as in the training
  generation, not more iterations.
- Twelve points remain a small sample; these fractions do not estimate the
  success rate over the whole parameter range.

## Retrieved evidence

`results/m_star_stop_rule_v1/` (probe continuations, frozen rule,
per-point solver records, validation table),
`figures/m_star_stop_rule_v1_validation.png`,
`logs/mstar_stop_rule_v1_*.log`.
