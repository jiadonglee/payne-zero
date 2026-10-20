# 84-input M-giant bootstrapping and final test: closeout (2026-10-20)

Preregistration: `notes/mgiant_direct_abundance_bootstrap_preregistration_20261015.md`.
Pilot: `notes/mgiant_direct_abundance_pilot_closeout_20261015.md`.
Results: `results/coauthor_validation_20261005/mgiant_direct_abundance/`
(`round{1,2}/rounds_summary.json`, `round{1,2}/corpus/summary.json`,
`round{1,2}/training_{2,6}profile.json`, `test/summary.json`).
Runs: `runs/mgiant_direct_abundance_{round1,round2,test}_20261015/` on Garching.

## Rounds

| round | w (dex) | labels | admitted | fraction | with reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1 | 256 | 196 | 76.6 % | 86.7 % |
| 2 | 0.1 | 128 | 94 | 73.4 % | 82.5 % |
| 2 | 0.2 | 128 | 90 | 70.3 % | 79.6 % |

By temperature (admitted / labels, 3000–3300 / 3300–3600 / 3600–4000 K):
round 1 52/80, 73/88, 71/88; round 2 at 0.1 dex 23/40, 34/44, 37/44;
round 2 at 0.2 dex 20/40, 36/44, 34/44.
Labels without a stable reference: 30 in round 1 (16 below 3300 K), 14 and 15
in round 2. Other failure classes: learned path not stable 24, 8, 18; paths
disagree 1, 0, 0; strict restart or spectrum 5, 12, 5; no evaluation errors.

## Corpus and networks

The 455 eight-label atmospheres plus the admitted labels: 651 rows after
round 1 (576 training, 75 validation), 835 after round 2 (732 / 103).

M-giant validation errors (temperature relative p95, column mass dex p95):

| network | after round 1 | final |
| --- | --- | --- |
| two-profile | 0.0142, 0.115 | 0.0190, 0.133 |
| six-profile | 0.0115, 0.076 | 0.0177, 0.125 |

The final validation set includes the ±0.2 dex labels of round 2. The
eight-label two-profile network has 0.0053 on its own validation set.

## Final test

120 labels, 12 in each of the 10 holdout boxes, w = 0.1 dex. The reference
path is stable at 107 labels; 13 have none, 8 of them below 3300 K.

| start | success / 107 | 3000–3300 (19) | 3300–3600 (33) | 3600–4000 (55) | median iteration | median start difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 84-input two-profile | 103 (96.3 %) | 15 | 33 | 55 | 30 | 0.052 |
| 84-input six-profile | 100 (93.5 %) | 16 | 30 | 54 | 30 | 0.031 |
| eight-label two-profile | 102 (95.3 %) | 18 | 32 | 52 | 25 | 0.027 |
| MARCS grid node | 67 (62.6 %) | 6 | 24 | 37 | 45 | 0.175 |

Every failure is a path that does not meet the stability rule within its
cap; no stable endpoint disagrees with the reference. At the 13 labels
without a reference, the starts are stable at 5, 8, 8 and 3 labels.
Median wall time per solve 1355, 1436, 1435 and 1326 s.

Paired over labels where both succeed (faster / equal / slower):

- 84-input two-profile vs eight-label: 11 / 25 / 62 of 98; success of one only 5 and 4.
- 84-input six-profile vs eight-label: 15 / 24 / 57 of 96; 4 and 6.
- 84-input two-profile vs six-profile: 32 / 22 / 43 of 97; 6 and 3.

## Reading

The three network starts succeed at fractions that differ by a few per cent,
which the preregistration reports as indistinguishable. All three are far
ahead of the grid node. The eight-label start, evaluated at the projected
labels and rebuilt at the full composition, needs fewer iterations than both
84-input starts and starts closer to the reference. The test changes each
element by at most 0.1 dex, so the projected eight-label labels lie close to
the test composition. Below 3300 K the eight-label start succeeds at 18 of
19 labels, the 84-input starts at 15 and 16.

## Limitations

- One test width (0.1 dex); the ±0.2 dex labels enter only the corpus.
- 19 test labels with a reference below 3300 K.
- Production defaults are unchanged.

Paper: `NdirTest*`, `NdirRound*`, `NdirCorpus*` macros, Sect. 4 and Sect. 6.
