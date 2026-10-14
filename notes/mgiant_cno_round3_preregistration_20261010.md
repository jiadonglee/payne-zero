# M-giant CNO bootstrapping round 3: preregistration (2026-10-10)

Round 3 fills the cool, high-gravity corner where round 2 lost most labels
(`notes/mgiant_cno_round2_closeout_20261010.md`: 32/53 admitted below 3300 K,
13 of them without a stable reference).

## Corpus and networks

The round-2 recipe (`notes/mgiant_cno_round2_preregistration_20261009.md`) is
repeated on the cumulative corpus: the 73 regenerated v4 cool rows, the 29
early development rows, the 137 round-1 rows outside the holdout boxes and the
141 round-2 rows, 380 in total. The parent-group validation split (15 per
cent, seed 20261010), architecture, loss, batch balance and checkpoint
selection are unchanged. Output: `results/coauthor_validation_20261005/mgiant_round3/`.

## Labels

120 labels, seed 20261012, uniform over $T_{\rm eff}$ 3000–3300 K and
$\log g$ 1.5–2.5, with the round-2 ranges for the other labels
([M/H] −1.0–0.5, [α/M] ±0.1, ξ 1.5–2.5 km/s, [C/M] −0.3–0.1,
[N/M] −0.1–0.5, [O/M] ±0.1); labels inside a holdout box are redrawn.

## Starts

- Learned: the round-3 two-profile network; cap 90.
- Reference: nearest corpus atmosphere with temperature scale 100 K (round 2
  used 250 K); the other scales are unchanged (0.25, 0.20, 0.10, 0.50 km/s,
  0.10 for each of C, N, O). In round 2 the cool labels without a stable
  reference differed from the admitted cool labels only in temperature
  distance to the reference (median 146 K against 62 K; $\log g$, [M/H], C and
  N distances alike). With the 380-row corpus the 100 K scale gives a median
  temperature distance of 60 K for the round-3 labels, against 114 K at 250 K.
- Criteria, outcome classes and reported statistics as rounds 1 and 2. The
  round-3 admitted fraction describes the cool corner only.

## Compute

Training on `astro-h100gpu02`; solve on `astro-highcpu09`, 10 workers ×
8 threads; evaluation in two GPU shards. Remote solve directory:
`runs/mgiant_cno_round3_20261010/`. Launcher:
`experiments/reduced_state_emulator/run_coauthor_mgiant_round3_garching_20261010.sh`.
Production defaults are unchanged.
