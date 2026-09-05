# M-star v1r3 dwarf iter120 Garching closeout

Date: 2026-09-03

## Outcome

- Terminal marker: `RESCUE_POOL_EXHAUSTED`
- Candidates attempted: 51/51
- New full-gate eligible dwarfs: 0
- Combined v1r2 + v1r3 dwarf eligible count: 26
- Target of 24 additional dwarfs: not reached

Increasing the iteration cap from 60 to 120 did not recover additional training rows. One case reached formal primary and self-restart convergence, but failed the unchanged primary flux, restart flux, and path-consistency gates. The other 50 did not reach formal primary convergence and also failed the downstream gates.

## Frozen setup

- Same-node native MARCS `(m,T)` start
- Iteration cap: 120
- Flux, state, restart, path-consistency gates and candidate order unchanged from v1r2
- Protocol hash: `726d30f8a8274cf1086c72994831395a9826265921dc77e16992a0595b540f72`
- Final status hash: `f2900e07282c7f9780885600fafe8aa89b4be30b048124c633f17133839ff974`

## Retrieved artifacts

- Results: `results/m_star_emulator_v1r3_dwarf_iter120/`
- Log: `logs/mstar_v1r3_dwarf_iter120.log`
- Verified local versus Garching: 58 files, 37,540,768 bytes
- Aggregate content hash: `8c654dca1531c705d9e8e9a3d5af17b33b6ced773472f6127540c674457d8350`

The v1r3 result is diagnostic only. It does not authorize emulator training, production routing, Korg runs, or sealed-holdout use.
