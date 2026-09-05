# M-star interpolated (m,T) seed v1 preregistration

This exploratory initializer arm leaves the terminal v1r2 `FAIL_STOP` and the
v1r3 120-iteration MARCS rescue unchanged. Its single purpose is to test
whether already-gated Payne-Zero/ATLAS dwarf `(m,T)` profiles can warm-start
failed same-track dwarfs.

## Frozen input

- Parent: `m_star_emulator_v1r2_marcs100`.
- Parent result: 26/108 eligible dwarfs and an exhausted fixed pool.
- Donors: parent dwarf nodes with `training_eligible`. Giants and ungated
  nodes are not donors.
- Candidates: parent dwarf nodes that are not `training_eligible` and that
  have at least one same-track donor. Same track means identical `logg`,
  `[M/H]`, and `vmic`.
- Background, not a parent: v1r3 recovered 0/51 additional dwarfs from the
  same MARCS `(m,T)` seed with a 120-iteration ceiling. This arm changes the
  initializer, not the iteration budget.

## Interpolation rule

Do not interpolate the six-field state. Mix only column mass and temperature
in the log, then reconstruct the remaining fields.

- If the target Teff is strictly between two gated donor temperatures on the
  same track, linearly interpolate `(log m, log T)` in `log Teff` between the
  nearest cooler donor and the nearest hotter donor.
- If donors exist on only one side, copy the nearest donor `(m,T)`. That is a
  one-sided start, not an interpolation.
- Do not use cross-track kNN. Mixing metallicities that differ by 1 dex is
  reserved for a later arm if this one fails.

A target cooler than every same-track donor is outside the donor convex hull.
3000–3300 K dwarfs remain extrapolations wherever the coldest gated donor is
warmer. Probe rows must keep that label; a later success there is not
interpolation coverage.

## Solver and admission

Reuse v1r2 unchanged: 60 iterations, all-layer relative temperature `5e-4`,
imported flux gate, finite/positive/monotone six-field state, and
primary-versus-restart path consistency. Do not retune thresholds.

The terminal ATLAS atmosphere is the only training target. MARCS is retained
only as the parent-campaign comparison already stored in v1r2 `case.json`.

## Staged execution

0. From existing v1r2 `cases/dwarf/**/case.json`, write the same-track
   neighbor table. No solver.
1. Run only the frozen six-star probe:
   - easy: first three same-track failures with nearest `|ΔT| ≤ 100 K` in
     frozen v1r2 priority;
   - hard: first three same-track failures with nearest `|ΔT|` in 200–300 K
     in frozen v1r2 priority, which includes the metal-poor 3800→3500 node.
2. Scale only if at least two of the three easy probes become eligible. Then
   run every remaining same-track failure. The target is 24 new eligible
   dwarfs. Batch overshoot is reserve evidence.
3. If the easy trio recovers 0 or 1 row, stop. Do not add neighbors, mix
   tracks, or loosen gates. The next tool is the existing adaptive
   continuation runner, not a denser interpolator.

## Frozen probe identifiers

These identities are the rule above applied to frozen v1r2, written here
before this arm's solver runs.

Easy:

- `g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3500` (3600 K donor, ΔT = 100 K, hull)
- `g+4.50_m-0.50_a+0.00_c+0.00_x1.00_t3800` (3900 K donor, ΔT = 100 K, hull)
- `g+5.00_m+0.50_a+0.00_c+0.00_x1.00_t3300` (3400 K donor, ΔT = 100 K, hull)

Hard:

- `g+4.50_m-1.00_a+0.00_c+0.00_x1.00_t3500` (3800 K donor, ΔT = 300 K, hull)
- `g+5.00_m+0.00_a+0.00_c+0.00_x1.00_t3500` (3750 K donor, ΔT = 250 K, hull)
- `g+5.50_m+0.50_a+0.00_c+0.00_x1.00_t3500` (3800 K donor, ΔT = 300 K, hull)

## Boundaries

This run does not train an emulator, run candidate validation, open a sealed
holdout, run Korg, change production routing, overwrite v1r2 or v1r3, or use
MARCS as truth. Even 50 eligible dwarfs would still not train a network and
would still not support a Teff < 4000 K generalization claim.
