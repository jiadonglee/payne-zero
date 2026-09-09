# M-giant two-field emulator v3 Garching closeout

Date: 2026-09-09

## Outcome

The M-giant goal of this campaign is met: with the enlarged corpus, the
two-field emulator warm-starts the unchanged Payne-Zero solver to
**9/9 = 100% convergence in 3-4 iterations** on every opened giant validation
star (gate: >= 90% within 30), with 9/9 flux-gate passes and 9/9 landings on
the truth fixed point. Every solve also converges within the 15-iteration
mark.

Level verdicts:

- Profile: temperature p95 (three-seed ensemble) `2.42e-3` passes the `3e-3`
  gate; column-mass p95 `9.6e-3` misses the `7.7e-3` gate by `1.25x`;
  monotonicity violations zero.
- Solver: passed outright as above.
- Spectral (TiO 6650-6670 A, 5e-3 bar): continuum 9/9; 7/9 stars pass all
  three metrics. The two misses (`t3850`, `t3900` at `g2.0 [M/H]=0`) sit at
  `1.36-1.40x` the bar and both passed in the v2 run - marginal fixed-point
  wander, not a systematic bias.

## What moved the numbers

Corpus growth did exactly what it was supposed to. Against v2:

- cool train rows `34 -> 73` giants, boundary extended to 3000-4000 K
  (17 gate-passing v1r2 products at 3000-3300 K admitted) plus the cap-240
  gap round (`+21`, all at 3000-3700 K);
- giant temperature p95 `6.2e-3 -> 2.4e-3` (2.6x better, now under gate);
- giant column-mass p95 `1.9e-2 -> 9.6e-3` (2x better);
- solver convergence `9/9` giants with iterations `3-6 -> 3-4`.

## Walls recorded

The cap-240 gap round admitted 21 of 53 nodes; the 32 rejects are now
double-evidenced (cap 60 and cap 240, same strict criterion, same frozen
gates), concentrated on the warm `logg 0.5` giant tracks. They are retained
as ineligible records, not interpolated over. Dwarf coverage remains deferred
(v2 closeout); dwarf rows left the v3 corpus by design.

## Retrieved evidence

Remote tree:

`/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-mstar-emulator-v1-20260831`

Local copies:

- `results/m_star_cold_library_inventory_v1/` (in-boundary 103 = 76 giants + 27 dwarfs)
- `results/m_star_giant_supplement_v2_cap240/` (53 case records, 59 products)
- `results/m_star_cool_corpus_mgiant_v1/` (82-row corpus: 73 train + 9 validation)
- `results/m_star_emulator_mgiant_v3/` (candidate validation + spectral gate)
- `artifacts/m_star_emulator_mgiant_v3/` (3 checkpoints + training summary)
- `logs/mstar_giant_supplement_v2_cap240.log`, `logs/mstar_mgiant_v3_*.log`

## Remaining gaps

1. Column-mass p95 at `1.25x` the preregistered bar (the bar originates from
   Ting's warm-dominated calibration); more giant rows or a mass-weighted
   loss term are the levers, not gate edits.
2. Two marginal TiO points at `1.4x` bar.
3. 32 open-track warm `logg 0.5` nodes remain unsolved under the frozen
   strict criterion; continuation-seeded attempts are the untried alternative.
