# Per-element abundances in M giants, 32-label pilot: closeout (2026-10-15)

Preregistration: `notes/mgiant_direct_abundance_pilot_preregistration_20261014.md`.
All 40 cases (8 null, 32 changed) solved and evaluated on `astro-highcpu09`
and a GPU node; the evaluation finished 2026-10-14 18:52 UTC. Thresholds
unchanged. Results: `results/coauthor_validation_20261005/mgiant_direct_abundance_pilot/summary.json`.

## Composition

At the null compositions the 81-element vector equals the eight-label
composition (0.0 dex). In every solved reference endpoint,
log(N_X/N_X,null) equals the requested change to within 1.5e-15 dex for all
81 elements: the per-element abundances reach the solved atmosphere.

## Solver

| change | labels | stable reference | eight-label success | median iteration (reference / eight-label) |
| --- | ---: | ---: | ---: | ---: |
| null | 8 | 7 | – | – |
| alpha_split | 8 | 5 | 4 | 30 / 27.5 |
| titanium | 8 | 7 | 5 | 40 / 40 |
| cno | 8 | 7 | 7 | 45 / 30 |
| all | 8 | 7 | 7 | 35 / 45 |

The eight-label start succeeds in 23 of the 26 changed labels with a stable
reference; every successful endpoint agrees with the reference within 8e-4 in
the three TiO metrics (limit 5e-3). The three failures (alpha_split at
3000 K, log g 1.5; titanium at 3800 K and 4000 K) run the full 90 iterations
without meeting the stability rule; none diverges. The reference start has no
stable endpoint within 120 iterations in 6 changed labels: the 3200 K,
log g 2.5, [M/H] = −0.5 parent, whose null is not stable either (alpha_split,
cno), the 3000 K, log g 2.5 parent (alpha_split, titanium), and one label each
at 3800 K (alpha_split) and 4000 K (all). Median wall time per start at eight
threads is 31–33 min (eight-label) and 39–47 min (reference).

## Structural response

Measured where both the changed case and its null have a stable reference
(24 labels; the 3200 K parent has no null endpoint).

| change | labels | median T p95 | largest T max | TiO structure-only max | H-band structure-only max |
| --- | ---: | ---: | ---: | ---: | ---: |
| alpha_split | 5 | 0.038 | 0.077 | 0.020 | 0.009 |
| titanium | 6 | 0.0057 | 0.014 | 0.035 | 0.014 |
| cno | 7 | 0.041 | 0.117 | 0.154 | 0.105 |
| all | 6 | 0.054 | 0.203 | 0.154 | 0.120 |

Every one of the 24 responses exceeds the null's difference from the parent
product of the same parent (0.0007–0.0080), so all are resolved. The
temperature response exceeds the two-profile network's validation T p95
(0.0053) in 22 of 24 labels. The structure-only spectral effect exceeds 5e-3
in 21 of 24 labels at 665–667 nm and in 21 of 24 at 1555–1570 nm. The two
starts agree at their endpoints: in the all-element change at 3300 K,
log g 2.5, [M/H] = −1.0, both reach a surface temperature of 1934 K against
2086 K for the null, with the column-mass scale larger by 0.15 dex.

## Reading

Both preregistered criteria are exceeded. Changes of 0.1 dex outside the
eight-label relation, CNO and the electron donors most of all, change the
M-giant structure by more than the eight-label network error. A synthesis
that changes these abundances on an eight-label atmosphere misses up to 0.15
in TiO normalized flux. The full composition therefore belongs in the
atmosphere solve, which the existing interface already supports, and the
existing eight-label start, with the solver, reaches the full-composition
solution in 23 of 26 labels at an iteration count comparable to the reference
start. Ti changes the structure least and the spectrum mostly directly.

## Limitations

- The temperature response is taken on the null column-mass grid, as
  preregistered, while the network validation error is on the standard
  Rosseland optical-depth grid. A shift of the column-mass scale enlarges the
  deep-layer differences, so the comparison with 0.0053 is not like for
  like. The structure-only spectral effect does not depend on the depth
  coordinate and gives the same reading.
- Eight parents and one draw of signs per change; 0.1 dex only. The pilot
  does not test whether an 84-input network would shorten the solve, and it
  does not cover wider abundance ranges or C/O near 1.
