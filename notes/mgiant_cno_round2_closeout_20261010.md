# M-giant CNO bootstrapping round 2: closeout (2026-10-10)

Preregistration: `notes/mgiant_cno_round2_preregistration_20261009.md`.
All 172 full-range labels completed (2026-10-09 09:56 to 2026-10-10 04:26 UTC);
thresholds unchanged. The solve ran 10 workers × 8 threads on
`astro-highcpu09`; evaluation ran on `astro-h100gpu02` and `astro-a100gpu01`.
No process remains on any node.

## Outcomes

| outcome | labels |
| --- | ---: |
| admitted | 141 |
| no stable reference within 120 | 15 |
| learned path not stable within 90 | 10 |
| strict restart or restart spectrum failed | 6 |
| paths disagree | 0 |
| evaluation error | 0 |

Admitted fraction 141/172 = 82.0 per cent; over the 157 labels with a stable
reference, 141/157 = 89.8 per cent.

| $T_{\rm eff}$ (K) | labels | admitted | no reference |
| --- | ---: | ---: | ---: |
| 3000–3300 | 53 | 32 | 13 |
| 3300–3600 | 43 | 40 | 1 |
| 3600–4000 | 76 | 69 | 1 |

| range | admitted / labels |
| --- | --- |
| $T_{\rm eff}$ 3000–3150 / 3150–3300 K | 18/26, 14/27 |
| $\log g$ 0.5–1.0 / 1.0–1.5 / 1.5–2.0 / 2.0–2.5 | 36/40, 33/38, 27/36, 45/58 |
| [C/M] −0.3 to −0.1 / −0.1 to 0.1 | 61/80, 80/92 |
| [N/M] −0.1–0.1 / 0.1–0.3 / 0.3–0.5 | 48/56, 51/65, 42/51 |

Above 3300 K, 109 of 119 labels are admitted. Below 3300 K, 21 of 53 fail:
13 without a stable reference, 7 with an unstable learned path, 1 at the
strict restart. The cool labels without a reference are mostly at
$\log g>2$ and sit a median 146 K from their reference atmosphere, against
62 K for the admitted cool labels; above 3300 K, references a median 115 K
away still succeed. The corpus is sparse in the cool, high-gravity corner,
where two holdout parents (3000 K at $\log g=0.5$ and 2.5, [M/H]=−0.5) also lie.

On the 147 labels whose paths agree, the median stability iteration is 25
from the round-2 two-profile start and 35 from the reference; the learned
start is faster for 128, equal for 11 and slower for 8 (round 1: 30 against
35, faster for 90 of 150).

Round 1 drew labels near parents and round 2 over the full range, so 85.5 and
82.0 per cent describe different populations.

## Products

`results/coauthor_validation_20261005/mgiant_cno_round2/`: `setup.json`,
`summary.json`, `cno_development_corpus.npz` (141 atmospheres), per-case JSON
records, logs. Remote: `runs/mgiant_cno_round2_20261009/`. The cumulative
M-giant corpus outside the holdout boxes holds 239 + 141 = 380 atmospheres.
