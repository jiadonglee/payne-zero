# M-giant CNO bootstrapping round 1: preregistration (2026-10-08)

Round 1 of the M-giant corpus bootstrapping (coauthor comments L831–842).
It extends the single-parent development batches
(`mgiant_cno_pilot`, `mgiant_cno_walk`, `mgiant_cno_local`) to many parents
and to a CNO range that covers APOGEE M giants. Thresholds are fixed here
before any solve.

## Parents and split

- Parents: the 53 recovered MARCS giants of
  `results/coauthor_validation_20261005/marcs_structure/summary.json`,
  each with a converged Payne Zero atmosphere (`reference_product`).
- Sorted by `candidate_id`; position `i % 5 == 4` is a holdout parent
  (10 parents), the rest are development parents (43).
- Holdout parents are not run in any bootstrapping round. Their
  neighbourhoods are reserved for the final M-giant test (`fig:mconv`), so no
  atmosphere near them enters the corpus.

Holdout parents: g+0.50_m-0.50 t3000, t4000; g+1.50_m+0.00 t3500, t4000;
g+1.50_m+0.50 t4000; g+1.50_m-1.00 t3600; g+2.50_m+0.00 t3300;
g+2.50_m-0.50 t3000, t3900; g+2.50_m-1.00 t3600.

## Labels

Four random labels per development parent, 172 in total, seed 20261009,
drawn in sorted parent order:

| label | draw |
| --- | --- |
| $T_{\rm eff}$ | parent ± U(75 K), clipped to 3000–4000 K |
| $\log g$ | parent ± U(0.25), clipped to 0.5–2.5 |
| [M/H] | parent ± U(0.25), clipped to −1.0–0.5 |
| [α/M] | U(−0.1, 0.1) |
| ξ | U(1.5, 2.5) km/s |
| [C/M] | U(−0.3, 0.1) |
| [N/M] | U(−0.1, 0.5) |
| [O/M] | U(−0.1, 0.1) |

C, N, O enter as absolute abundance offsets [M/H]+[X/M]. The C and N ranges
are set by the APOGEE M-giant labels of Sect. 5, not by the earlier ±0.1 dex
batches.

## Arms

- Learned: frozen five-label v4 (median of the three seed checkpoints,
  `artifacts/m_star_emulator_mgiant_v4/`), (m,T) only, rebuilt with
  $P=g\,m$ at the target composition; cap 90 iterations.
- Reference: the parent's converged (m,T), rebuilt the same way; cap 120.

## Criteria (unchanged from the earlier batches)

- Stability: five-iteration segments, two consecutive stable segments;
  temperature p95 < 1e-4, mass p95 < 1e-3 dex, three-metric spectral change
  < 5e-4 over 665–667 nm at R=20,000, flux gate of `flux_gate.json` on every
  iteration of the segment.
- Independent paths: three-metric endpoint difference ≤ 5e-3.
- Strict restart from the learned endpoint: numerical convergence within 120,
  finite, positive, monotone column mass, flux gate, temperature p95 ≤ 3e-3,
  mass p95 ≤ 7.7e-3 dex, and three-metric restart spectral difference ≤ 5e-3.
- A label is admitted to the corpus only if it passes all three.

## Reported statistics

- Admitted fraction over all 172 labels. This is the round-1 number for
  L836; it describes random labels near development parents.
- Admitted fraction over labels with a stable reference path; labels without
  one are counted separately (L875).
- Outcome per label: admitted, no reference, learned path not stable, paths
  disagree, strict restart or restart spectrum failed, evaluation error.
- Counts by $T_{\rm eff}$ bin 3000–3300, 3300–3600, 3600–4000 K (L874), and by
  parent.
- Median stability iterations of the two paths on paired successes.

## Next rounds

Round 2 retrains the two-profile and six-profile M-giant networks on the v4
cool corpus plus every admitted atmosphere (pilot, walk, local, round 1),
then draws labels over the full range and starts from the round-2 network;
the reference is the nearest admitted atmosphere. The final holdout test
draws random labels around the holdout parents and runs the two-profile,
six-profile and nearest raw MARCS node arms.

## Compute

`astro-highcpu09` (192 cores, cap 96): 12 workers × 8 Numba threads,
about 240 GB resident. Evaluation in two shards, one process per idle GPU node
(`astro-h100gpu02` and one more), within each node's half-CPU cap.
Remote directory: `runs/mgiant_cno_multiparent_20261008/` under
`payne-zero-coauthor-20261005`. Driver:
`experiments/reduced_state_emulator/coauthor_mgiant_cno_multiparent_20261008.py`.
Production defaults are unchanged.
