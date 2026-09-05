# M-star emulator v1r3 dwarf iter120 preregistration

This exploratory rescue keeps the terminal v1r2 `FAIL_STOP` unchanged. Its
single purpose is to test whether a larger iteration budget recovers additional
dwarf truth rows.

## Frozen input

- Parent: `m_star_emulator_v1r2_marcs100`.
- Parent result: 26/108 eligible dwarfs and an exhausted fixed pool.
- Rescue pool: parent dwarf nodes that were ineligible because the primary
  solver reached iteration 60, while retaining a valid finite six-field state.
- Candidate order: unchanged v1r2 priority.
- Initial state: same-node native MARCS column mass and temperature only.

## One changed solver setting

The primary and strict self-restart iteration ceiling is increased from 60 to
120. The all-layer relative-temperature threshold remains `5e-4`. The v1r2
flux thresholds, state-quality checks, and primary-versus-restart consistency
checks remain unchanged.

The target is 24 new eligible dwarf rows. Together with the 26 parent rows,
this would provide 50 dwarf rows. Batch overshoot is reserve evidence and is
not automatically admitted.

## Boundaries

This run does not train an emulator, run candidate validation, open a sealed
holdout, run Korg, change production routing, or use MARCS as truth. A later
corpus or training step requires the rescue result first.
