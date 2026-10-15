# M-giant bootstrapping with individual abundances (84 inputs): preregistration (2026-10-15)

Coauthor items A1, L831–842 and L1157: after the eight-label bootstrapping,
bootstrap the M-giant corpus with the full individual abundances, train
84-input two- and six-profile networks, and test them. The 32-label pilot
(`notes/mgiant_direct_abundance_pilot_closeout_20261015.md`) showed that
0.1 dex changes outside the eight-label relation change the structure, and
that the eight-label start still reaches the full-composition solution at
23 of 26 labels with a reference.

Drivers in `experiments/reduced_state_emulator/`:
`mgiant_direct_abundance_bootstrap_20261015.py` (rounds, corpus),
`mgiant_direct_abundance_train_20261015.py` (networks),
`mgiant_direct_abundance_test_20261015.py` (final test); launcher
`run_mgiant_direct_abundance_garching_20261015.sh`. Output:
`results/coauthor_validation_20261005/mgiant_direct_abundance/`.

## Labels

Eight-label coordinates as in round 2 of the eight-label bootstrapping:
$\log g$ 0.5–2.5, [M/H] −1.0–0.5, [α/M] ±0.1, ξ 1.5–2.5 km/s, [C/M]
−0.3–0.1, [N/M] −0.1–0.5, [O/M] ±0.1, uniform, with $T_{\rm eff}$ stratified
over 3000–3300, 3300–3600 and 3600–4000 K. Labels inside a holdout box are
redrawn. On top of the eight-label composition, each of the 81 elements of
the direct-abundance encoding changes independently by a uniform amount
within ±w dex, rounded to 0.01 dex. Labels with a number ratio C/O ≥ 1 are
redrawn, so the corpus stays oxygen-rich. Every abundance goes to the solver
as an absolute [X/H].

| round | seed | w | labels (3000–3300 / 3300–3600 / 3600–4000 K) |
| --- | --- | --- | --- |
| 1 | 20261015 | 0.1 | 80 / 88 / 88 |
| 2 | 20261016 | 0.1 | 40 / 44 / 44 |
| 2 | 20261016 | 0.2 | 40 / 44 / 44 |

## Starts and admission

- Learned start: round 1, the final eight-label two-profile network of
  Sect. 4 at the projected labels ([M/H] = [Fe/H], [α/M] the mean [X/Fe] of
  Ne, Mg, Si, S, Ca, Ti, [C/M], [N/M], [O/M] = [X/Fe]); round 2, the
  84-input two-profile network of round 1. Rebuilt from (m,T) at the full
  composition, cap 90.
- Reference start: the nearest corpus atmosphere in the scaled distance of
  the projected labels (scales 250 K, 0.25, 0.20, 0.10, 0.50, 0.10, 0.10,
  0.10; 100 K below 3300 K), rebuilt at the full composition, cap 120.
- Stability rule, spectral agreement, strict restart and admission are those
  of the eight-label rounds. Every failure stays in the denominator.

## Corpus and networks

The 455 eight-label atmospheres are converted to the 84-input form from
their compositions (checked: 0.0 dex against the solver composition) and
keep their training and validation roles. Admitted atmospheres of each round
are added; their development groups are split 85/15 into training and
validation (seed 20261017). The networks take T_eff, log g, ξ, [Fe/H] and the
80 [X/Fe] and use the set-encoded architecture of the normal-star
direct-abundance family. Training rows are the normal-star direct-abundance
training split plus the M-giant training rows, half of each per batch;
standardization and the six-profile PCA use the training rows only; learning
rate 3e-4, 300 epochs; the checkpoint minimizes the M-giant validation loss.
Networks are trained after round 1 (used in round 2) and after round 2
(final).

## Final test

Twelve labels in the box of each of the 10 holdout parents (120, seed
20261018), with the eight-label coordinates of the eight-label holdout test
and w = 0.1 dex. The test width is fixed now and does not depend on round 2.
Starts, each solved and rebuilt at the full composition:

| start | profiles | cap |
| --- | --- | ---: |
| 84-input two-profile | (m,T), rebuilt | 90 |
| 84-input six-profile | six profiles, no rebuild | 90 |
| eight-label two-profile | final network of Sect. 4 at the projected labels, rebuilt | 90 |
| MARCS grid node | native (m,T) of the parent's node, rebuilt | 90 |
| reference | the parent's converged (m,T), rebuilt | 120 |

Success: the stability rule of the eight-label holdout test and endpoint
agreement with the reference within 5e-3 in the three TiO metrics.

## Reported

- Each round: admitted fraction over all labels and over labels with a
  reference, by temperature bin, with the 0.1 and 0.2 dex groups separate;
  the outcome classes of the failures.
- Corpus sizes; validation T p95 and column-mass p95 of each network.
- Final test: labels with and without a reference; for each start the
  successes, success fraction, counts by temperature bin, median iteration at
  success, median start difference and median wall time; paired comparisons
  of the 84-input starts with the eight-label start (faster, equal, slower,
  success of one only).

No threshold for "better" is set. With 120 labels, success fractions that
differ by a few per cent are reported as indistinguishable.

## Compute

Rounds and the test on Garching CPU nodes (eight threads per solve),
training and spectral evaluation on a GPU node. Production defaults are
unchanged.
