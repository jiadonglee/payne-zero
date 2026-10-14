# M-giant CNO bootstrapping round 1: closeout (2026-10-09)

Preregistration: `notes/mgiant_cno_multiparent_round1_preregistration_20261008.md`.
All 172 labels around 43 development parents completed (2026-10-08 11:16 to
2026-10-09 03:03 UTC); thresholds unchanged. The solve stage ran 11 workers
× 8 threads on `astro-highcpu09`; evaluation ran on `astro-h100gpu02` and
`astro-a100gpu01`. No process remains on any node.

## Outcomes

| outcome | labels |
| --- | ---: |
| admitted | 147 |
| no stable reference within 120 | 14 |
| learned path not stable within 90 | 8 |
| strict restart or restart spectrum failed | 3 |
| paths disagree | 0 |
| evaluation error | 0 |

Admitted fraction 147/172 = 85.5 per cent; over the 158 labels with a stable
reference, 147/158 = 93.0 per cent. Every development parent contributes at
least one admitted atmosphere (43 parent groups).

| $T_{\rm eff}$ (K) | labels | admitted | no reference |
| --- | ---: | ---: | ---: |
| 3000–3300 | 43 | 33 | 6 |
| 3300–3600 | 49 | 44 | 4 |
| 3600–4000 | 80 | 70 | 4 |

| range | admitted / labels |
| --- | --- |
| $\log g$ 0.5–1.0 / 1.0–2.0 / 2.0–2.5 | 38/40, 55/64, 54/68 |
| [C/M] −0.3 to −0.1 / −0.1 to 0.1 | 72/93, 75/79 |
| [N/M] −0.1–0.1 / 0.1–0.3 / 0.3–0.5 | 48/55, 51/61, 48/56 |

Nine of the 14 labels without a reference come from $\log g=2.5$ parents.
Failures rise with gravity and with carbon depletion; the nitrogen range up
to +0.5 dex does not change the admitted fraction.

On the 150 labels whose paths agree, the median stability iteration is 30
from the learned start and 35 from the parent reference; the learned start is
faster for 90, equal for 32, and slower for 28.

## Products

`results/coauthor_validation_20261005/mgiant_cno_multiparent/`: `setup.json`,
`summary.json`, `cno_development_corpus.npz` (147 × 8 labels, 147 × 80 × 6
profiles, parent group per row), per-case JSON records, logs. Remote:
`runs/mgiant_cno_multiparent_20261008/`.

Round 2 trains the M-giant two-profile and six-profile networks on the v4 cool
corpus plus the 29 earlier and 147 round-1 atmospheres. The 10 holdout
parents remain unrun.
