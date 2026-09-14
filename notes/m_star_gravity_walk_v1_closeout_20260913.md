# M-giant gravity-walk closeout and campaign conclusion

Date: 2026-09-13

## Outcome

The three remaining undetermined points are resolved by gravity walks, all
passing: `t3525 g2.25 m-1.0` (emulator 25 / reference 40 iterations; worst
TiO metric 1.6e-3), `t3175 g2.25 m+0.5` (45 / 30; worst 1.8e-5), `t3325
g1.5 m-0.5` (25 / 20; worst 5.1e-5).  All three emulator arms stabilize
within the 60-iteration budget and land on the walked references with
cross-arm temperature p95 between 4.8e-4 and 6.6e-4.

The reference for each point is a parameter walk from the converged corpus
truth at the same metallicity and nearly the same temperature but
neighboring logg (2.5 -> 2.25 with an intermediate 2.0 where needed;
1.5 -> 1.75 -> 2.0 -> 2.25), followed by one 25 K temperature leg - the
remedy that closed t3250, applied from neighboring gravity tracks.

## Final M-giant validation tally

Across all dual-path validations of this campaign (12 original points, 6
reproduction points, 3 gravity-walk points): **21/21 pass** wherever both
paths exist, with cross-arm TiO metrics at or below 1.6e-3 - three times
under the 5e-3 bar - and no consistency failures.  Three points required
walk-seeded references (t3250, t3050, t3850 g0.5) and three more the
gravity walks above; every reference that could be constructed agrees with
the independently initialized emulator-solver endpoint.

## Boundaries

`t3250 g2.0 [M/H]=0` remains the one node where no reference can be built
from any tested seeding strategy (temperature descent, nearest-node MARCS,
temperature walk); its emulator arm is stable, so the gap is a solver
convergence-structure issue on that specific branch, documented as terminal.

## Verdict

**M-giant emulator + solver is complete.**  The configuration (v4
checkpoints, 60-iteration emulator budget, frozen-rule references,
walk-seeded where necessary) is validated across 3000-4000 K, logg
0.5-2.25, [M/H] -1 to +0.5 including off-grid nodes.  Fit-for-purpose for
exploration calculations within the tested envelope.

`results/m_star_gravity_walk_v1/` holds the walk legs, relaxations, and
verdicts.
