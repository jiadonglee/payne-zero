# M-giant CNO bootstrapping round 2: preregistration (2026-10-09)

Round 2 trains matched two- and six-profile M-giant networks on all admitted
atmospheres and then solves random labels over the full M-giant range from
the new two-profile network. Round 1:
`notes/mgiant_cno_multiparent_round1_closeout_20261009.md`.

## Holdout boxes

The 10 holdout parents of round 1 each define a box
$|\Delta T_{\rm eff}|\le75$ K, $|\Delta\log g|\le0.25$, $|\Delta[{\rm M/H}]|\le0.25$.
The final M-giant test draws its labels inside these boxes. No training row
and no round-2 label lies inside any box.

## Corpus

| source | rows | in a holdout box | kept |
| --- | ---: | ---: | ---: |
| v4 cool corpus (scaled-solar CNO, α=0, ξ=2) | 84 | 10 | 74, if regenerated |
| pilot, walk and local CNO batches | 29 | 0 | 29 |
| round 1 | 147 | 10 | 137 |

The v4 cool corpus stores only (m,T). Each kept row is restarted from its
stored (m,T) at its node labels (rebuild with $P=g\,m$, convergence stop,
cap 120) and enters only if the restart converges, is finite and positive,
has monotone column mass, passes the flux gate, and stays within
temperature p95 ≤ 3e-3 and mass p95 ≤ 7.7e-3 dex of the stored structure.
The six initializer profiles of the restart are the training target.

Rows are grouped by parent node (cool node id, round-1 parent id, and the
single `t3800_g1.00_m-0.75` group). 15 per cent of the groups, drawn with
seed 20261010, are M-giant validation; the rest are training. There is no
M-giant test split; the holdout-box solver test is the test.

## Networks

- Inputs: the eight labels of the cno8 family; cool rows carry
  [C/M]=[N/M]=[O/M]=0.
- Encoding, architecture (cno8 reference checkpoint), loss and optimizer as
  `coauthor_train_20261005.py`; 300 epochs, batch 512, learning rate 1e-3,
  seed 20261010. The two networks differ only in their outputs.
- Training rows: cno8 training split of the Sect. 3 validation plus the M-giant
  training rows. Each batch takes 256 cno8 rows without replacement and 256
  M-giant rows with replacement. Standardization and the six-profile PCA use
  the union of training rows without weighting.
- Selection: one checkpoint per network at the minimum M-giant validation
  loss. Reported: temperature and column-mass p95 on the M-giant validation
  and cno8 validation rows.
- Output: `results/coauthor_validation_20261005/mgiant_round2/`; the Sect. 3
  networks are not touched.

## Round-2 solve (after the networks exist)

- 172 labels drawn uniformly over 3000–4000 K, $\log g$ 0.5–2.5,
  [M/H] −1.0–0.5, [α/M] ±0.1, ξ 1.5–2.5 km/s, [C/M] −0.3–0.1,
  [N/M] −0.1–0.5, [O/M] ±0.1, with rejection inside the holdout boxes;
  seed 20261011.
- Learned start: round-2 two-profile network, (m,T) rebuilt with $P=g\,m$;
  cap 90.
- Reference start: the nearest admitted atmosphere in scaled label distance
  (scales 250 K, 0.25, 0.20, 0.10, 0.50 km/s, and 0.10 for each of C, N, O),
  its (m,T) rebuilt at the target composition; cap 120.
- Stability, independent-path, strict-restart and spectral criteria as round 1.
- Reported as round 1: admitted fraction over all labels and over labels with
  a stable reference, outcome counts, and counts by $T_{\rm eff}$ bin.

Round 1 drew labels near parents and round 2 draws over the full range, so the
two admitted fractions describe different populations. The comparable number
for `fig:mconv` is the holdout-box test, which runs the two-profile,
six-profile and nearest-MARCS-node starts.

## Compute

Cool-row regeneration on `astro-highcpu09`, 8 workers × 8 threads. Training on
`astro-h100gpu02` (two threads). Drivers:
`experiments/reduced_state_emulator/coauthor_mgiant_round2_corpus_20261009.py`,
`experiments/reduced_state_emulator/coauthor_mgiant_train_20261009.py`.
Production defaults are unchanged.

## Corpus and networks (2026-10-09, before the round-2 solve)

The corpus holds 239 rows (73 of 74 regenerated cool rows pass), 209 training
and 30 validation rows in 74 parent groups. Each network trains in about
75 s on the H100; the selected checkpoints are at epochs 18 (two-profile) and
22 (six-profile). After those epochs the M-giant validation loss rises while
the cno8 validation loss keeps falling, so the M-giant rows are overfitted.

| network | M-giant validation T p95 | M-giant validation $m$ p95 (dex) |
| --- | ---: | ---: |
| frozen v4 (round-1 start) | 0.0190 | 0.126 |
| round-2 two-profile | 0.0113 | 0.089 |
| round-2 six-profile | 0.0131 | 0.077 |

The round-2 two-profile network is the learned start of the round-2 solve.
