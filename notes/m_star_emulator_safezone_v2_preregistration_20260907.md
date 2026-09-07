# Cold safe-zone emulator v2 preregistration

Date: 2026-09-07

## Boundary

The EOS donor-exhaust probes closed the dwarf walls: below 3275 K at
`[M/H] = 0` and below 3675 K at `[M/H] = -1.0` no convergence path exists
(`notes/m_star_donor_exhaust_v1_closeout_20260907.md`). Training therefore
abandons the square grid and targets the safe zone only:

- Giants: `3500-4000 K`, `logg 0.5-2.5`, `vmic 2 km/s`, `[M/H]` unrestricted.
- Dwarfs (`logg >= 4.5`, `vmic 1 km/s`): `[M/H] = +0.5` at `Teff >= 3600 K`,
  `0.0` at `>= 3300 K`, `-1.0` at `>= 3800 K`.

Rows outside the boundary are never admitted to training even when eligible.

## Certified library (inventory)

`m_star_cold_library_inventory_v1` joins the admission records of the v1r2,
pipeline scaleout/complete, tomography, v1r3, atlas-continuation, and walk
campaigns against the boundary table. Result on both the local mirror and the
Garching frozen tree:

- 64 admitted in-boundary products (giants 37, dwarfs 27).
- 86 admitted total; 22 eligible-but-out-of-boundary rows are retained as
  evidence and excluded from training.
- 17 open-track giant nodes missing in `3500-4000 K`; they are filled by the
  `m_star_giant_supplement_v1` campaign using the unchanged MARCS-seeded
  truth generation, cap 60, strict self-restart, and the same frozen flux gate.
- Converged-but-never-admitted waypoint products stay excluded; the dwarf
  `3800 K / [M/H] = 0` node keeps its phase-guard failure record.

## Corpus

`m_star_cool_corpus_safezone_v1` builds the corpus from the inventory:

- Train: all in-boundary admitted products minus the six carved validation
  nodes below. After the supplement lands the pool is rebuilt from the updated
  inventory; the carve-out list stays fixed.
- Carved cool validation (fixed before training): dwarf `g4.5 m-1.0 t3800`,
  `g4.5 m0.0 t3400`, `g4.5 m+0.5 t3600`; giant `g1.5 m-1.0 t3500`,
  `g2.5 m0.0 t3500`, `g1.5 m+0.5 t3500`. These probe the cold corners that the
  imported validation set does not cover.
- Imported opened validation: the 12 rows of the v1r1 policy60 corpus
  (unchanged, `3750-4000 K`, disjoint nodes).

The v1 smoke corpus therefore has 58 train (34 giant + 24 dwarf) and 18
validation (9 + 9) rows.

## Training

Unchanged: the 4x512 physical two-field architecture with by-construction
monotone column mass, the immutable 52,199-row base corpus, 50:50 group
sampling, and balanced group standardization. Changed: checkpoint selection
and early stopping now minimize the maximax-normalized profile score

```
max( T_p95/3e-3 , logm_p95/7.7e-3 )
```

over the existing and cool validation groups (`minimax_normalized_profile_p95_v1`),
replacing best-validation-loss selection.

Execution: 50-epoch single-seed smoke first (loss stability, zero monotonicity
violations), then three seeds `20260831, 20260901, 20260902`, 300 epochs,
A100 on the Garching GPU node.

## Acceptance

1. Profile: cool validation `temperature_relative_p95 <= 3e-3` and
   `mass_dex_p95 <= 7.7e-3` (median over the three seeds).
2. Solver: emulator warm start solved by the unchanged 30-iteration policy;
   convergence fraction over the cool validation rows `>= 0.90` with the
   existing per-class `>= 0.90` reported alongside (`evaluate_mstar_candidate_v1`
   keeps its stricter preregistered constants).
3. Spectral: TiO window `6650-6670 A` residual gate at `5e-3` on cold
   validation products.
