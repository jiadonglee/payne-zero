# M-star v1r2 MARCS-100 Garching closeout

Date: 2026-09-02

## Outcome

The preregistered 100-row campaign ended at `FAIL_STOP`.

- Giant pool: 80/108 candidates attempted, 53 eligible, 50-row quota reached.
- Dwarf pool: 108/108 candidates attempted, 26 eligible, pool exhausted before the 50-row quota.
- The exact 50 giant + 50 dwarf corpus was therefore not constructed.
- Emulator training did not start.
- Candidate validation did not run.

The terminal evidence is recorded by:

- `results/m_star_emulator_v1r2_marcs100/GIANT_QUOTA_REACHED`
- `results/m_star_emulator_v1r2_marcs100/DWARF_QUOTA_FAILED`
- `results/m_star_emulator_v1r2_marcs100/CORPUS_BUILD_FAILED`
- `artifacts/m_star_emulator_v1r2_marcs100/TRAINING_FAILED`
- `results/m_star_emulator_v1r2_marcs100/VALIDATION_BLOCKED_BY_TRAINING`

`status_giant.json` and `status_dwarf.json` are the terminal class-level records. The top-level `status.json` is an earlier non-terminal snapshot and should not be used for the final counts.

## Direct interpretation

Same-node native MARCS `(m,T)` starts produced enough eligible giant atmospheres under the frozen restart and flux gates, but not enough dwarfs. The current bottleneck is dwarf truth-generation eligibility, not GPU emulator training. MARCS was used only as an initializer; the eligible targets remain terminal Payne-Zero/ATLAS atmospheres.

No threshold, candidate ordering, quota, sealed holdout, or production routing was changed. Batch overshoot in the giant run was retained as reserve and was not admitted to training.

## Retrieved evidence

Remote source:

`/home/jdli/xiasangju/jdli/payne-zero-mstar-emulator-v1-20260831`

Local copies:

- `results/m_star_emulator_v1r2_marcs100/`
- `artifacts/m_star_emulator_v1r2_marcs100/`
- `logs/mstar_v1r2_marcs100_*.log`

Transfer verification over these campaign files:

- File count: 383 on both remote and local.
- Total bytes: 427,007,857 on both remote and local.
- Aggregate SHA-256 over the sorted per-file SHA-256 manifest:
  `0e46aea9749fc665b46da6aa9b4aad594416643e395023c78c5f448fda7b1e03`

