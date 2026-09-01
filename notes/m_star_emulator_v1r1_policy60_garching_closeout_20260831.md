# M-star emulator v1r1 policy60 Garching closeout

Final decision: **`FAIL_STOP_SPARSE_COOL_CORPUS`**

The preregistered open policy60 campaign reached a terminal state without
opening either sealed M-star track, the existing sealed holdout, Korg, or
production routing. The A100 training process stopped at its frozen corpus gate
before fitting any epoch, so candidate solver validation did not run.

## Open-spine result

- Eight of eight opened tracks completed.
- 33 of 168 planned temperature nodes were attempted. Each continuation track
  stopped after its first ineligible node, as preregistered.
- 31/33 primary solves converged.
- 26 nodes were training-eligible.
- 24/33 attempted nodes converged within 15 iterations.
- The generated corpus contains 26 rows: 14 train and 12 validation.
- The split is:
  - train: 14 giants, 0 dwarfs;
  - validation: 6 giants, 6 dwarfs.

The frozen training gate requires at least 20 cool training rows and at least
five training rows per stellar class. It therefore failed exactly
`cool_train_row_count` and `cool_train_dwarf_count`.

This is not mainly a failure to solve the atmospheres. Both opened training
dwarf anchors at 4000 K converged and passed the frozen flux gates:

- log g = 4.5: primary 8 iterations, restart 3;
- log g = 5.0: primary 10 iterations, restart 3.

They were excluded because primary/restart path consistency failed. Since each
training continuation stops at its first ineligible node, both lower-temperature
training-dwarf sequences were blocked and the corpus received no dwarf training
rows.

## Strict-settling diagnostic

The separate, preregistered all-layer settling diagnostic passed 2/2:

- log g = 4.5: primary 12 iterations, restart 3; column-mass p95 difference
  0.001067 dex and temperature p95 difference 0.000392;
- log g = 5.0: primary 15 iterations, restart 3; column-mass p95 difference
  0.001761 dex and temperature p95 difference 0.000402.

These diagnostic products were not admitted to training. They show that stricter
settling can repair the 4000 K dwarf path-consistency problem, but using that
policy to produce training truth requires a new prospective campaign and
preregistration.

## Terminal downstream state

- A100 trainer: `TRAINING_FAILED`, stopped before model fitting.
- Candidate checkpoint: none.
- Opened candidate validation: not run because training did not pass.
- Existing v1 results remain immutable.
- No gate or threshold was loosened or refitted.

## Retrieved evidence

Local archive:
`results/m_star_emulator_v1r1_policy60_garching_20260831/`

The retrieved remote payload contains 121 files and 118,485,961 bytes, excluding
the local transfer manifest. Remote and local relative-path SHA-256 tree hashes
match for all four components:

- policy60: `bf1395299572b4dd54bb46c85c8feb2645c85601495869f0d368bcf359a64319`;
- settling diagnostic: `62bad5e65d06a6ca052271e043f3190b7889da9f9f6d1ccb95c29458699362d1`;
- logs: `5fe9528272b0619f7b64faab9515d3d41ee30719d61062b596e8cfcb090f79d0`;
- artifacts: `8557c1c0295db09d2422e6373f7c254016049911514bf5fd35e9af85042b4f63`.

The cool corpus itself has SHA-256
`94e8a8ba4b31d3ab5f8735ed819ff0dc83ae754dfe8cff906f1d2cfd9bbc820d`.
