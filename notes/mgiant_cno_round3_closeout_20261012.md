# M-giant CNO bootstrapping round 3: closeout (2026-10-12)

Preregistration: `notes/mgiant_cno_round3_preregistration_20261010.md`.
All 120 labels in the cool, high-gravity corner completed (2026-10-10 20:10 to
2026-10-11 17:56 UTC); thresholds unchanged. No process remains on any node.

## Networks

Trained on the 380-row cumulative corpus (326 training, 54 validation rows).
M-giant validation T p95 / column-mass p95: two-profile 0.0047 / 0.025 dex
(checkpoint epoch 75), six-profile 0.0044 / 0.028 dex (epoch 27).

## Outcomes

| outcome | labels |
| --- | ---: |
| admitted | 75 |
| no stable reference within 120 | 34 |
| learned path not stable within 90 | 6 |
| strict restart or restart spectrum failed | 5 |
| paths disagree | 0 |

Admitted fraction 75/120 = 62.5 per cent; over the 86 labels with a stable
reference, 75/86 = 87.2 per cent.

| range | admitted / labels |
| --- | --- |
| $T_{\rm eff}$ 3000–3100 / 3100–3200 / 3200–3300 K | 29/43, 14/41, 32/36 |
| $\log g$ 1.5–2.0 / 2.0–2.5 | 57/81, 18/39 |
| [M/H] −1.0 to −0.5 / −0.5 to 0 / 0 to 0.5 | 23/41, 22/36, 30/43 |

The tighter reference temperature scale does not reduce the share of labels
without a stable reference (34/120, against 13/53 below 3300 K in round 2).
In these paths the three-metric spectral change over the last five-iteration
segment is about 1e-5 and every flux gate passes, while the 95th-percentile
temperature change is 1e-4 to 7e-4 per segment, above the 1e-4 stability
limit. The same holds for the round-2 cool failures. Below 3300 K at high
gravity the solver therefore approaches the solution slowly in the deep
temperature structure after the spectrum has settled; the cap of 120
iterations, not the distance of the reference, sets the failures.

On the 80 labels whose paths agree, the median stability iteration is 35 from
the round-3 two-profile start and 55 from the reference; the learned start is
faster for 72, equal for 4 and slower for 4.

## Products

`results/coauthor_validation_20261005/mgiant_round3/` (corpus, networks) and
`results/coauthor_validation_20261005/mgiant_cno_round3/` (solve records,
`cno_development_corpus.npz` with 75 atmospheres). The cumulative corpus
outside the holdout boxes holds 380 + 75 = 455 atmospheres.
