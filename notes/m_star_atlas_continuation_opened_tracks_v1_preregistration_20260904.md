# M-star ATLAS continuation on opened tracks v1 preregistration

This arm leaves the terminal v1r2 `FAIL_STOP`, the v1r3 120-iteration rescue,
and the interpolated `(m,T)` arm unchanged. Its single purpose is to test
whether small-step continuation from newly gated ATLAS `(m,T)` products can
carry the two interpolation-opened dwarf tracks to the next frozen v1r2 grid
node through the same admission gates.

## Frozen input

- Parent: `m_star_emulator_v1r2_marcs100` (26/108 eligible dwarfs, pool
  exhausted, frozen flux gate).
- Seed campaign: `m_star_interpolated_mt_seed_v1` with a drained scale pool.
  Its six gated dwarfs are the only donors this arm may start from.
- Opened tracks:
  - A: log g=4.5, [M/H]=0, vmic=1 km/s, gated at 4000-3500 and 3300 K.
  - B: log g=4.5, [M/H]=-0.5, vmic=1 km/s, gated at 4000-3700 K.
- Everything else (denser interpolation, more iterations, cross-track
  neighbours) is out of scope for this arm.

## Continuation rule

Carry only column mass and temperature from a gated ATLAS product; Payne-Zero
reconstructs the other four fields through the exact physical reconstruction
path (`reduced_rematerialized`). No six-field carry, no MARCS seed.

- Approach the next v1r2 grid node in 50 K steps. A waypoint between grid
  nodes only needs solver convergence and a finite valid six-field state.
- After a failed step, halve it once: 50 K to 25 K. No step smaller than
  25 K. A failure at 25 K closes the cell and the track stops there.
- Every v1r2 grid node that should count toward the corpus must pass the
  full v1r2 eligibility: primary 60 iterations at all-layer relative
  temperature `5e-4`, strict self-restart, imported flux gate on both,
  finite/positive/monotone six-field state, and path consistency.
- Same track only: identical `logg`, `[M/H]`, `vmic`. Walk seeds are always
  the coldest gated ATLAS on that track at that moment; the walk never
  restarts from a failed node and never skips a node.

## Frozen probes

Only two targets run before any walk decision:

- Probe A: seed `g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3500`, target 3400 K
  (the hole between the gated 3500 and 3300 K nodes; the interpolation arm
  failed it from a single-sided 3600 K copy).
- Probe B: seed `g+4.50_m-0.50_a+0.00_c+0.00_x1.00_t3700`, target 3600 K
  (the interpolation arm failed it from a single-sided 3900 K copy).

Decision: 2/2 eligible walks both tracks; 1/2 walks the passing track only;
0/2 stops the arm. The walk order is A: 3200, 3100, 3000; B: 3500, 3400,
3300, 3200, 3100, 3000. A track stops at its first ineligible node.

## Frozen seed products

Gated interpolation-arm primaries on Garching, hashed at plan time:

| role | candidate_id | Teff K | sha256 |
| --- | --- | ---: | --- |
| probe A seed | `g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3500` | 3500 | `295fa503b4296414a408040d48e3c646d0be27304f33aaa8dafe22077232b306` |
| probe B seed | `g+4.50_m-0.50_a+0.00_c+0.00_x1.00_t3700` | 3700 | `ef75d3245a5462e381c0a343dd007b66e67cee79aedc3d03bf9ca1960e245889` |
| track A backup | `g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3300` | 3300 | `54790c865451ec25704a4b380f2a56173f1715ca9f2b4b9e2f3b3f54644cb2b5` |

The runner verifies the probe seed hashes against this table before solving.
The walk uses this campaign's own gated products as soon as they exist.

## Boundaries

This run does not train an emulator, run candidate validation, open a sealed
holdout, run Korg, change production routing, overwrite v1r2/v1r3/interpolation
results, use MARCS as a seed or truth, mix tracks, or use `full_carry`.
Even if both tracks walk to 3000 K, the geometric ceiling is about 17 new
dwarfs (11 walk nodes plus 6 interpolation rows on top of v1r2's 26); this
arm makes no 50-row quota promise and no Teff < 4000 K generalization claim.
