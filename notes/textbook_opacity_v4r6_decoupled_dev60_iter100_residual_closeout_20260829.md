# v4r6 decoupled residual 100-iteration diagnostic closeout

Date: 2026-08-29

Machine decision: `ITER100_RESIDUAL_DIAGNOSTIC_COMPLETE`.

This closes only the six-star residual diagnostic registered in
`notes/textbook_opacity_v4r6_decoupled_dev60_iter100_residual_preregistration_20260829.md`.
It does not rewrite the historical 15-iteration
`FAIL_STOP_DEVELOPMENT` or the 60-iteration
`ITER60_DIAGNOSTIC_COMPLETE` result. It does not authorize fresh-open,
production, spectra, or sealed holdout.

## Result

All six residual stars were restarted from the same decoupled analytic seed
with one trial, at most 100 iterations, and a 3600 s per-star wall-clock
timeout.

| Outcome | Count |
|---|---:|
| Recovered | 0/6 |
| Still failed | 6/6 |
| Wall-clock timeout | 6/6 |
| Solver error | 0/6 |

Residual indices:

`6152`, `33051`, `33053`, `44167`, `46124`, `48708`.

Every row has `solver_outcome="timeout"` and
`iterations_completed=null`. This means the subprocess did not return a
terminal solver result within 3600 s. The current funnel has no partial
per-iteration telemetry, so this diagnostic cannot determine how many
internal iterations completed or identify a particular slow iteration.

The bounded conclusion is therefore:

> Raising the cap from 60 to 100 together with the timeout from 900 s to
> 3600 s did not recover any of the six residual stars. Their binding observed
> limit is wall-clock return time, not a demonstrated 60- or 100-iteration
> ceiling.

## Returned artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| JSON | 11297 | `a19a6812fa3e9b6c00a1671d82b716c5fe93b9398b0eb0e5d8e8f968ea27cbc3` |
| JSONL | 2879 | `d9a577441ea38da10e2c5cca4315ee6cbc916d0265bdbb2b8674b6833714fcb3` |
| Runtime guard | 1083 | `38d508c10f8f6c5c70d174b14bf86ed544ff7c9d67c8d75820f16c7dc89c31c5` |
| Log | 41365 | `2317327392cc4391720b7648ffed13e8cddaa8ff0ec174939bdfa7eeb8d4dc23` |

Remote and local byte sizes and SHA-256 values were equal after transfer.

## Provenance limitation

The runtime guard records the original decoupled source-manifest hash
`ebc932f2...`, but that manifest does not include the iter100 driver and does
not describe the later funnel plumbing used by this diagnostic. The numerical
rows and returned artifacts are complete, but this run remains a diagnostic
with the previously documented source-sidecar gap. It is not used as a
promotion result.

The new matched policy60 study has a separate source manifest that verifies
every source file at launch and is unaffected by this limitation.
