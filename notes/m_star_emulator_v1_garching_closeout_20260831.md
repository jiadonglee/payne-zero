# M-star emulator v1 Garching closeout

## Outcome

`FAIL_STOP` before emulator training. The open Payne-Zero continuation campaign
produced a valid but preregistered-insufficient cool corpus. No emulator
checkpoint was trained, opened candidate validation and old-domain retention
were not reached, and no sealed track or existing sealed holdout was opened.
Production routing was unchanged and Korg was not run.

The exact training preflight failure was:

```text
FAIL_STOP: sparse cool corpus: ['cool_train_row_count', 'cool_train_dwarf_count']
```

The frozen requirements were at least 20 cool training rows and at least five
training rows in each luminosity class. The observed split was:

| split | giant | dwarf | total |
|---|---:|---:|---:|
| train | 14 | 0 | 14 |
| opened validation | 6 | 6 | 12 |
| total | 20 | 6 | 26 |

The validation dwarf rows remain validation rows. They were not moved into the
training split after seeing the result.

## Open-spine result

- Reference gate: 15/16 successful solves; passed.
- Frozen all-layer absolute flux-error thresholds:
  median 0.234215%, p95 9.557343%, maximum 23.151469%.
- Eight opened tracks completed in 9567 s.
- 33 nodes were attempted, 31 primary solves converged, 26 rows passed every
  truth-admission gate, and 24 of those converged within 15 iterations.
- Corpus SHA-256:
  `8b34a3b852cc7d268dbff4a8d7eba539e56173f2a51036afa0986c0f5ce56656`.

First ineligible node on each opened track:

| log g | class | first ineligible Teff | boundary |
|---:|---|---:|---|
| 0.50 | giant | 4000 K | self-restart/flux/path-consistency failure |
| 1.50 | giant | 3500 K | 3550 K is the last admitted node |
| 2.00 | giant | 3700 K | 3750 K is the last admitted node |
| 2.50 | giant | 3800 K | 3850 K is the last admitted node |
| 4.50 | dwarf | 4000 K | path-consistency failure |
| 4.75 | dwarf | 3700 K | 3750 K is the last admitted node |
| 5.00 | dwarf | 4000 K | path-consistency failure |
| 5.50 | dwarf | 4000 K | reference anchor did not converge |

This establishes useful local M-giant and validation M-dwarf continuation
boundaries, but it is not enough to train the preregistered balanced v1
emulator.

## Validation-launcher evidence

The pre-launched validation supervisor also exited with a shell syntax error:

```text
==: -c: line 2: syntax error: unexpected end of file
```

This did not change the scientific boundary because training had already
stopped before producing a candidate checkpoint. It should be repaired only in
a separately versioned follow-up campaign.

## Retrieval verification

The immutable Garching payload was copied to
`results/m_star_emulator_v1_garching_20260831/`.

- Remote and local payload: 105 files, 111,897,210 bytes.
- Results: 97 files, 111,889,265 bytes.
- Logs: 8 files, 7,945 bytes.
- Remote/local tree SHA-256:
  `4a3bbd955f662971e55a752ad1ac0bf8581e8da5194a5e3537cac5e781308376`.
- Per-file hashes:
  `results/m_star_emulator_v1_garching_20260831/TRANSFER_CHECKSUMS.sha256`.

Primary evidence:

- `results/m_star_emulator_v1_garching_20260831/results/m_star_emulator_v1/open_spine_summary.json`
- `results/m_star_emulator_v1_garching_20260831/results/m_star_emulator_v1/cool_truth_corpus.json`
- `results/m_star_emulator_v1_garching_20260831/logs/mstar_training.log`
- `results/m_star_emulator_v1_garching_20260831/logs/mstar_validation.log`
