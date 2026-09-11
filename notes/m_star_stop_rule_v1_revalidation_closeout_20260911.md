# Stop-rule twelve-point revalidation closeout (v4 checkpoints)

Date: 2026-09-11

## Outcome

After adding the three corner truth rows and retraining (corpus
`m_star_cool_corpus_mgiant_v2`, 75 train + 9 validation giants; checkpoints
`m_star_emulator_mgiant_v4`), the twelve dual-path points were re-solved with
the frozen stop rule (emulator budget 30, reference budget 240; reference
arms reused unchanged).

Verdicts are unchanged in composition: **7/12 pass, 5/12 undetermined**
(`results/m_star_stop_rule_v1/validation_table.{json,csv}`,
figure `figures/m_star_stop_rule_v1_revalidation_v4.png`).  The seven passes
again show two independent starts landing on one atmosphere (cross-arm
`dT p95 <= 3.9e-3`, `dlogm <= 1.0e-3 dex`, all TiO metrics `<= 8e-4`, five
times under the `5e-3` bar), with the emulator reaching the frozen precision
in 20-30 iterations - including the low-gravity `g0.75` point whose MARCS
reference needed 125 iterations.

The five undetermined points are the same five, and they do not respond to
the added data:

- Three reference arms diverge from the nearest-node MARCS seed regardless of
  checkpoint (`t3250 g2.0 m0`, `t3050 g2.0 m+0.5`, `t3850 g0.5 m-0.5`) -
  solver/seed class, needs continuation seeding.
- Two emulator arms do not stabilize within 30 iterations although the
  reference stabilizes (60 and 85 iterations; `t3150 g1.5 m-0.5`,
  `t3300 g0.75 m+0.5`) - the corner rows added next to them did not change
  this, indicating a convergence-speed property of these cool nodes rather
  than data sparsity.

Profile gates for v4: temperature p95 `2.27e-3` (pass), column-mass p95
`1.23e-2` (`1.6x` the bar; the `g1.5 m+0.5 t3500` carved validation star
remains the dominant outlier).  Monotonicity violations zero.

## Reading

The scope-limited conclusion stands and is now reproduced on independently
trained checkpoints: within the decidable subset the M-giant initializer
plus solver is consistent across nominal, cool, and low-gravity points.  The
goal of `11/12 within 30 iterations` is not reachable as long as the budget
is 30 at nodes whose reference needs 60-125: the remaining five points need
one of three distinct remedies (a larger emulator budget at slow nodes,
continuation-seeded references, or both), and that choice has not been made.
