# Textbook opacity v4r6 decoupled residual 100-iteration diagnostic

Date: 2026-08-29

This run re-solves only the six stars that remained unconverged after the
60-iteration diagnostic. It does not overwrite the 15-iteration
`FAIL_STOP_DEVELOPMENT` JSON or the 60-iteration
`ITER60_DIAGNOSTIC_COMPLETE` JSON. It cannot authorize fresh-open 120.

## Why timeout must rise with the iteration cap

The 60-iteration diagnostic recovered 17 of 18 stars that had exhausted
15 iterations. The six residuals were **900 s wall-clock timeouts**, not
iteration-cap failures (`iterations_completed` is `null`). Raising the
cap to 100 while keeping 900 s would repeat those timeouts. The per-star
timeout is therefore raised to **3600 s** so that 100 iterations can
become the binding constraint.

This is still a restart from the same decoupled seed, not a mid-solve
checkpoint resume.

## Frozen residual set

Taken from 60-iteration timeouts, in order:

`6152`, `33051`, `33053`, `44167`, `46124`, `48708`

Five of these also timed out at 15 iterations. `33053` finished 15
iterations without converging and then timed out when given 60.

## Solver policy

- arm: `textbook_v4r6_decoupled` (`m_grey`, `T_conv`, no mass re-integration)
- one trial
- **100** iterations
- **3600 s** per-star timeout
- only the six residual indices
- resume allowed only onto this run's JSONL and runtime signature

Primary pair: the same six rows from the frozen 60-iteration JSON.
The 15-iteration residual rows are a secondary reference.

## What this run may conclude

Record, and do not promote into `PASS_TO_FRESH_OPEN`:

- recovered / still-failed / still-timeout counts on the six stars
- iterations completed on any recovery
- whether a star still returns `iterations_completed = null` at 3600 s

Machine label: `ITER100_RESIDUAL_DIAGNOSTIC_COMPLETE`.
`authorizes_fresh_open` is false.

## Stop rule

This task stops after the residual JSON is written. It does not rerun
grey or convective at 100 iterations. It does not run fresh-open 120.

Output:
`results/analytic_initializer/textbook_opacity_v4r6_decoupled_dev60_iter100_residual_20260829.json`

Runner:
`experiments/analytic_initializer/run_textbook_opacity_v4r6_decoupled_dev60_iter100_residual.py`

Log: `logs/textbook_opacity_v4r6_decoupled_dev60_iter100_residual_20260829.log`

Host: `astronode-garching`.
