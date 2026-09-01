# Textbook opacity v4r6 matched 60-iteration development study

Date: 2026-08-29

Policy identity: `v4r6_analytic_warm_start_policy60_v1`.

This is a new solver-budget policy. It does not rewrite the historical
15-iteration `FAIL_STOP_DEVELOPMENT`, the v4r6 offline `FAIL_STOP`, or any
previous JSON. The already-exposed decoupled 54/60 result motivates the policy
but cannot validate it. All three arms are therefore rerun from frozen source
under one matched policy before any fresh sample is considered.

## Scientific question

With one trial, 60 solver iterations, and 900 s per-star timeout:

1. Is the decoupled analytic seed a usable warm start on the exposed
   development-60?
2. Does it retain its cold-star advantage over grey without materially
   degrading the hot or total solution rate?
3. Is its gain specifically associated with preserving `m_grey` while using
   the convective temperature construction?

The intervention is the complete seed tuple, not temperature alone:

```text
grey:       (m_grey, T_grey, kappa(T_grey, P_grey))
convective: (m_reintegrated, T_conv, kappa(T_conv, P_reintegrated))
decoupled:  (m_grey, T_conv, kappa(T_conv, P_grey))
```

## Frozen arms and execution policy

- same paper development-60 indices;
- one trial per star;
- 60 iterations per trial;
- 900 s wall-clock timeout per star;
- exact same source manifest and runtime environment;
- arms run sequentially, never concurrently on the node;
- order: decoupled, grey, convective;
- streamed JSONL may resume only when its runtime guard is unchanged.

Outputs:

```text
results/analytic_initializer/textbook_opacity_v4r6_decoupled_dev60_policy60_20260829.json
results/analytic_initializer/textbook_opacity_v4r6_grey_dev60_policy60_20260829.json
results/analytic_initializer/textbook_opacity_v4r6_convective_dev60_policy60_20260829.json
results/analytic_initializer/textbook_opacity_v4r6_policy60_matched_dev60_20260829.json
```

## Frozen continuation gate

This exposed study can authorize only writing a fresh-open preregistration.
It cannot authorize executing fresh-open, production, spectra, or a sealed
holdout.

All conditions are mandatory:

| Metric | Requirement |
|---|---:|
| Complete decoupled records | 60/60 |
| Solver errors | 0 |
| Finite decoupled seeds | 60/60 |
| Decoupled total convergence | >= 54/60 |
| Decoupled cool convergence | >= 23/27 |
| Decoupled hot convergence | >= 29/33 |
| Decoupled timeouts | <= 6 |
| Total paired net gain versus grey | >= 0 |
| Cool paired net gain versus grey | >= 4 |
| Hot paired net loss versus grey | <= 2 |

Pass:
`CONTINUE_TO_POLICY60_FRESH_OPEN_PREREGISTRATION`.

Fail:
`STOP_POLICY60_MATCHED_DEVELOPMENT`.

The absolute 90% target reflects the explicitly adopted 60-iteration
definition of a successful analytic warm start. Because development-60 is
already exposed, passing is development evidence only.

## Source and runtime integrity

Before any arm runs, write and transfer:

`results/analytic_initializer/textbook_opacity_v4r6_policy60_source_manifest_20260829.json`

Every arm driver verifies every source hash against that manifest before
launch. Each output also records the remote Python, NumPy, Numba, operating
system, hostname, environment, source-manifest hash, and sample hash. Any
source mismatch or cross-arm runtime mismatch is `INCONCLUSIVE_RUNTIME`.

## Stop boundaries

- Do not edit or rescore the historical 15- or 60-iteration JSON files.
- Do not tune opacity, convection, thresholds, timeout, or solver damping
  after seeing a matched arm.
- Do not open a fresh sample until all three matched arms are complete and a
  separate fresh-open protocol is frozen.
- Do not run spectra, production switching, coupled ODE work, or sealed
  holdout.
- The v4r6 offline opacity/mass `FAIL_STOP` remains a separate unresolved
  physical-validity limitation.
