# Textbook opacity v4r6 decoupled 60-iteration late-convergence diagnostic

Date: 2026-08-29

This is a solver-budget diagnostic on the already-built decoupled seed. It
does not overwrite
`results/analytic_initializer/textbook_opacity_v4r6_decoupled_dev60_20260828.json`.
It does not relax the frozen 15-iteration continuation gate. The 15-iteration
decision `FAIL_STOP_DEVELOPMENT` remains authoritative. This run cannot
authorize fresh-open 120, a coupled ODE, a production switch, spectra, or
the sealed holdout.

## Question

Eighteen of the sixty decoupled stars finished 15 solver iterations without
converging. Five more hit the 900 s wall-clock timeout. The question is
whether raising the iteration cap from 15 to 60, with the same 900 s
timeout, recovers any of those 18 stars.

The 15-iteration continuation gate is not reopened. A 60-iteration score
against 15-iteration grey or convective controls is mixed-policy
characterization, not a pass.

## Frozen identities

- Seed: `v4r6_decoupled_mgrey_tconv_v1` (`m_grey`, `T_conv`, no mass
  re-integration). Unchanged.
- Sample: the same paper development-60 indices.
- 15-iteration decoupled JSON SHA-256
  `81211e9e61cf9e2ab39a517d3bc4c455a9a1fd702a9a92db35d70abd7609afb0`
  (`FAIL_STOP_DEVELOPMENT`, 37/60, cool 12/27, hot 25/33, 5 timeouts).
- Timeout remains 900 s. Stars that already failed to complete 15
  iterations in 900 s cannot be rescued by a larger iteration cap under
  this timeout.

## Solver policy

- one trial
- **60** iterations
- 900 s per-star timeout
- resume allowed only onto this run's JSONL and runtime signature

Primary paired control: the frozen 15-iteration decoupled funnel, same
seed. Grey and convective 15-iteration JSON files are mixed-policy
references and are not rerun.

## What this run may conclude

Record, and do not promote into `PASS_TO_FRESH_OPEN`:

- 60-iteration cool / hot / total convergence
- recovered / lost / still-failed counts versus the 15-iteration decoupled arm
- iterations completed on recovered stars
- timeout count under the unchanged 900 s budget

Machine label: `ITER60_DIAGNOSTIC_COMPLETE`.
`authorizes_fresh_open` is false.

## Stop rule

This task stops after the 60-iteration JSON is written and paired against
the frozen 15-iteration decoupled JSON. It does not rewrite the 15-iteration
closeout. It does not run fresh-open 120.

Output:
`results/analytic_initializer/textbook_opacity_v4r6_decoupled_dev60_iter60_20260829.json`

Runner:
`experiments/analytic_initializer/run_textbook_opacity_v4r6_decoupled_dev60_iter60.py`

Log: `logs/textbook_opacity_v4r6_decoupled_dev60_iter60_20260829.log`

Host: `astronode-garching`.

## Post-run formal result

The diagnostic completed on 2026-08-29 on `astronode-garching`. JSON
SHA-256 `9926fd75ba1ba948407a37c3886803e177f5cfe033d8754e6e2f64f42c7d595f`.
JSONL SHA-256
`7c0041c649d35ce1acd585ddc07f0fdb5bfc924ffb128d0c85c2d2d5960215c5`.
Machine label **`ITER60_DIAGNOSTIC_COMPLETE`**.
`authorizes_fresh_open` is false. The 15-iteration
`FAIL_STOP_DEVELOPMENT` was not rewritten.

| Split | 15-iter decoupled | 60-iter decoupled | Grey 15-iter (mixed) | Convective 15-iter (mixed) |
|---|---:|---:|---:|---:|
| All 60 | 37/60 | **54/60** | 37/60 | 20/60 |
| Teff < 7500 K | 12/27 | **24/27** | 6/27 | 0/27 |
| Teff >= 7500 K | 25/33 | **30/33** | 31/33 | 20/33 |
| Cool dwarf | 8/11 | 10/11 | 2/11 | 0/11 |
| Cool giant | 4/16 | **14/16** | 4/16 | 0/16 |
| Timeouts | 5 | 6 | 3 | 3 |

Primary pair versus the frozen 15-iteration decoupled arm: recovered 17,
lost 0, still failed 6, net +17. All 17 recovered stars finished in
16--29 iterations (mean 20.8). Of the 18 stars that exhausted 15
iterations without converging, 17 entered the basin and one (`33053`)
became a 900 s timeout. The five original wall-clock timeouts remained
timeouts.

The 15-iteration cap, not the decoupled seed, was the binding
constraint on those 17 stars. Grey and convective were not rerun at 60
iterations, so mixed-policy counts against them are not a pass. Fresh-open
120 was not run.
