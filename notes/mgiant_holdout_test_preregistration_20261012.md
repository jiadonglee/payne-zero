# Final M-giant test in the holdout boxes: preregistration (2026-10-12)

The test of Sect. 4 (`fig:mconv`, `fig:mspectral`, coauthor comments
L856–897). The 10 holdout parents and their boxes were fixed before
bootstrapping round 1 (`notes/mgiant_cno_multiparent_round1_preregistration_20261008.md`);
no corpus atmosphere lies in a box.

## Networks

The round-2 recipe on the cumulative corpus of 455 atmospheres (73 regenerated
v4 cool rows, 29 early rows, 137 round-1, 141 round-2 and 75 round-3 rows);
one two-profile and one six-profile network, checkpoint at the minimum M-giant
validation loss. Output: `results/coauthor_validation_20261005/mgiant_final/`.

## Labels

Ten labels per holdout parent, 100 in total, seed 20261013:
$T_{\rm eff}$, $\log g$ and [M/H] uniform within the parent's box (clipped to
3000–4000 K, 0.5–2.5, −1.0–0.5); [α/M] ±0.1, ξ 1.5–2.5 km/s, [C/M] −0.3–0.1,
[N/M] −0.1–0.5, [O/M] ±0.1. By temperature: 25 below 3300 K, 27 at
3300–3600 K, 48 at 3600–4000 K.

## Starts

| start | profiles | cap |
| --- | --- | ---: |
| two-profile network | (m,T), rebuilt with $P=g\,m$ at the target composition | 90 |
| six-profile network | all six predicted profiles, no rebuild | 90 |
| MARCS grid node | native MARCS (m,T) of the parent's grid node, rebuilt at the target composition | 90 |
| reference | the parent's converged atmosphere (m,T), rebuilt at the target composition | 120 |

## Success

A start succeeds at the first iteration where it meets the stability rule of
the bootstrapping rounds (two consecutive five-iteration segments with
temperature p95 < 1e-4, mass p95 < 1e-3 dex, TiO three-metric change < 5e-4,
flux gate on every iteration), provided its endpoint agrees with the reference
endpoint within 5e-3 in all three TiO metrics (665–667 nm, R=20,000).

## Reported

- Labels with a stable reference, and those without (L875).
- For each start: successes and success fraction over the labels with a
  reference, counts by $T_{\rm eff}$ bin, the success fraction against
  iteration (`fig:mconv`), and the median iteration at success.
- For each start: the three-metric TiO difference between the spectrum before
  any iteration and the reference endpoint (`fig:mspectral`).
- The median iteration at success of the two-profile start against that of the
  MARCS start (W and V of L887).

## Compute

Solve split over two CPU nodes by case index (`astro-highcpu09` and one more
node with free capacity), evaluation in two GPU shards. Remote directory
`runs/mgiant_holdout_test_20261012/`. Driver:
`experiments/reduced_state_emulator/coauthor_mgiant_holdout_test_20261012.py`;
launcher `run_coauthor_mgiant_final_garching_20261012.sh`. Production defaults
are unchanged.

## Compute record (2026-10-12)

Final networks on `astro-a100gpu01`: corpus 455 rows (409 training, 46
validation); M-giant validation T p95 / column-mass p95 0.0053 / 0.043 dex
(two-profile, epoch 66) and 0.0059 / 0.045 dex (six-profile, epoch 46).
The solve started 09:50 UTC on `astro-highcpu09` (even cases, 4 workers on a
loaded node) and `astro-highmem01` (odd cases, 9 workers). At 13:10 UTC the
even cases from 036 upward, none of them started, moved to a third process on
`astro-highcpu07` (10 workers, `--indices 36:100`, skipping existing case
directories); `astro-highcpu09` keeps cases 000–034. The single evaluation
process on `astro-a100gpu01` now takes cases as their solves finish rather
than in index order. Labels, starts and criteria are unchanged.
At 16:45 UTC the dispatcher on `astro-highcpu09` was stopped so that it could
not start cases from 036 upward that `astro-highcpu07` holds; its four workers
finish their current cases 022–028. Cases 030–034, not started, run on
`astro-highcpu05` (3 workers, `--indices 30:36`).
All 100 solves finished at 03:23 UTC on 2026-10-13. Cases 030 and 034 had
been started by both the stopped `astro-highcpu09` workers and
`astro-highcpu05`; the second solve rewrote their `solve.json` after the
evaluation. Both cases were re-evaluated against the final files (034
unchanged; 030 had read a reference file mid-write and now has a stable
reference), and the summary was collected again at 09:40 UTC. The
`astro-highcpu09` process group was killed; no workers remain on any node.
