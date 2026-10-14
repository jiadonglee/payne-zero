# Final M-giant test in the holdout boxes: closeout (2026-10-13)

Preregistration: `notes/mgiant_holdout_test_preregistration_20261012.md`.
Results: `results/coauthor_validation_20261005/mgiant_holdout_test/summary.json`
(100 rows with per-arm stability, start and endpoint differences). Labels,
starts, caps and the success rule are as preregistered.

| | two-profile | six-profile | MARCS grid node |
| --- | ---: | ---: | ---: |
| success over 93 labels with a reference | 83 (89%) | 90 (97%) | 65 (70%) |
| below 3300 K (19) | 16 | 19 | 15 |
| 3300–3600 K (27) | 24 | 25 | 18 |
| 3600–4000 K (47) | 43 | 46 | 32 |
| median iteration at success | 25 | 25 | 45 |
| median start difference, normalized flux | 4.8e-3 | 3.9e-3 | 4.4e-2 (70 measured) |
| starts above 5e-3 before solving | 42 | 36 | 61 |

- The reference path is stable at 93 labels (median 35 iterations); 6 of the
  7 without one lie below 3300 K.
- All network failures reach the cap without meeting the stability rule. 23
  grid-node starts diverge within 20 iterations (non-finite temperature,
  density or populations, or non-monotone column mass); 5 reach the cap.
- Paired against the grid node, the two-profile start is faster at 56 of 57
  labels and the six-profile start at 54 of 62. Two- against six-profile: 30
  faster, 26 equal, 24 slower over 80 labels.
- The six-profile network is at least as robust as the two-profile start in
  every temperature bin. The expected cool-atmosphere failure of the
  six-profile start does not occur once the network is trained on the
  bootstrapped corpus.

Paper: `NreviewTest*` macros, `fig_mconv.pdf`, `fig_mspectral.pdf`, Sect. 4
test text.
