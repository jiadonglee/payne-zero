# Textbook opacity v4r6 decoupled development-60 solver funnel

Date: 2026-08-28

This is a solver-basin diagnostic. It does not reopen or rewrite the v4r6
offline `FAIL_STOP`. It does not change the production solver, the default
initializer, or the sealed holdout. It does not repair H-minus. Mean
iteration count is reported and is not a gate.

WP0–WP3 passed before this note was frozen. This note is frozen before any
candidate solver execution.

## Why this task exists

The registered convective v4r6 development-60 converged `20/60` with
`0/27` cool. The grey v4r6 development-60 converged `37/60` with `6/27`
cool. Seed-versus-truth errors showed that re-integrating column mass
after the Saha-aware convective *T* replacement is the largest cool-star
mass error. The work plan
(`notes/textbook_opacity_v4r6_decoupled_mgrey_tconv_workplan_20260828.md`)
isolates that mechanism with one candidate:

```text
m_seed     = v4r6 mass integrated with Eddington-grey temperature
T_seed     = current Saha-aware convective temperature evaluated on P_grey = g m_grey
kappa_seed = kappa_v4r6(T_seed, P_grey)
```

Column mass is not re-integrated after the temperature replacement. The
candidate name is `v4r6_decoupled_mgrey_tconv_v1`.

Primary hypothesis `H1_MASS_REINTEGRATION_DAMAGE`: preserving `m_grey`
while retaining `T_conv` will improve cool-star convergence and retain
most of the grey arm's hot-star convergence.

## Frozen construction

The candidate is a new function. Historical v4r6 convective and grey
paths are unchanged.

```text
T_grey     = Eddington
m_grey     = RK4 integral of dm/dtau = 1 / kappa_v4r6(T_grey, g m), 8 substeps
P_grey     = g m_grey
T_conv     = current Saha-aware adiabatic replacement on (T_grey, P_grey)
m_seed     = m_grey
kappa_seed = kappa_v4r6(T_conv, P_grey)
```

Identities required on the registered seed audit:

- `m_decoupled == m_grey` bitwise
- `T_decoupled == T_convective` bitwise
- `kappa_decoupled` matches a fresh v4r6 recomputation at `rtol <= 1e-12`,
  `atol = 0`

## Structural gate (already passed)

Local project `.venv`, `NUMBA_THREADING_LAYER=workqueue`.

- Focused plus historical tests: 64 passed
  (`tests/test_textbook_opacity_v4r6_decoupled.py`,
  `tests/test_textbook_opacity.py`,
  `tests/test_analytic_initializer_multi_arm.py`)
- Seed-only audit JSON
  `results/analytic_initializer/textbook_opacity_v4r6_decoupled_seed_audit_20260828.json`
- SHA-256 `4d7179fce0af46889cf753dc462b28bcd66436d0efcec274bf653e5ed533e2be`
- Decision `PASS_STRUCTURAL`
- Finite and positive: `60/60`
- Mass identity: bitwise on 4800 layers, max abs difference `0`
- Temperature identity: bitwise on 4800 layers, max abs difference `0`
- Opacity identity: max relative residual `0`
- Fitted parameters: `0`
- `mass_reintegrated_after_convection`: `false`

Seed-versus-truth errors are characterization, not gates. Cool deep
(`τ > 10`) median `|Δ log10 m|` is `0.094` dex on both grey and
decoupled, versus `0.719` dex on the convective arm. Cool deep median
relative *T* is `0.309` on both decoupled and convective, versus `0.678`
on grey.

## Frozen sample and solver policy

Same frozen paper development-60 indices as analytic-parity, v4r3, the
registered convective v4r6 funnel, and the grey v4r6 funnel:
`results/paper_physical_seed_20260820/learned/convergence_metrics_learned_monotone.json`.
Do not draw a new funnel.

Solver policy matches the formula arms:

- one trial, 15 iterations, 900 s per-star timeout;
- seed the unchanged production solver through `analytic_seed_model`;
- stream JSONL as each star lands; resume is allowed only if the runtime
  signature is unchanged;
- no spectral gate, no 12-star smoke requirement, no sealed holdout.

Historical controls are read from frozen JSON and are not rerun:

- convective
  `results/analytic_initializer/textbook_opacity_v4r6_dev60_20260828.json`
  SHA-256 `c0c08c9727e522916085941bc5dcb40a96d67fea05852f8d88ddb4cae4cdd3e5`
  (`20/60`, cool `0/27`, hot `20/33`, 3 timeouts)
- grey
  `results/analytic_initializer/textbook_opacity_v4r6_grey_dev60_20260828.json`
  SHA-256 `caeee639e37600952be8439d259bdb99f68d992bf9d4c2a50749530f68bf015a`
  (`37/60`, cool `6/27`, hot `31/33`, 3 timeouts)

## Development-60 continuation gate

The development-60 is exposed. This gate decides only whether a fresh
open validation is worth the compute cost. All conditions are mandatory
and will not be relaxed after seeing the result:

| Metric | Requirement |
|---|---:|
| Complete records | `60/60` |
| Solver errors | `0` |
| Finite seeds | `60/60` |
| Cool convergence | `>= 11/27` |
| Hot convergence | `>= 30/33` |
| Total convergence | `>= 41/60` |
| Losses among 37 grey-converged stars | `<= 2` |
| Net paired gains minus losses | `>= 4` |
| Timeouts | `<= 3` |

Pass label: `PASS_TO_FRESH_OPEN`. Fail label: `FAIL_STOP_DEVELOPMENT`.
Incomplete records or a changed source/runtime signature:
`INCONCLUSIVE_RUNTIME`.

Mean iterations, seed-truth errors, and individual-star narratives are
reported and are not gates.

## Stop rule for this task

This task stops after the development-60 JSON is written and the
continuation gate is scored. It does not run the fresh-open 120 unless
that gate passes. It does not run flux/spectral checks, the sealed
holdout, a production switch, or the H-minus implementation repair.

Output:
`results/analytic_initializer/textbook_opacity_v4r6_decoupled_dev60_20260828.json`

Runner:
`experiments/analytic_initializer/run_textbook_opacity_v4r6_decoupled_dev60.py`

Log: `logs/textbook_opacity_v4r6_decoupled_dev60_20260828.log`

## Remote execution

Host `astronode-garching`. Checkout
`/home/jdli/xiasangju/jdli/payne-zero`. Python `.venv-linux/bin/python`.
Do not evaluate production opacity from macOS `.venv`. Before launch,
remote source hashes must equal the frozen source manifest. Do not
launch this candidate and a control simultaneously on the same node.

## Post-run formal result

The development-60 completed on 2026-08-28 on `astronode-garching`
(parent PID `2688371`). JSON SHA-256
`81211e9e61cf9e2ab39a517d3bc4c455a9a1fd702a9a92db35d70abd7609afb0`.
JSONL SHA-256
`bc3ec0583cd45cf5566ffd4ae8cd37d63bb96ff6bf86630cbd78a60948966bf3`.
The machine decision is **`FAIL_STOP_DEVELOPMENT`**. Fresh-open 120 was
not run.

| Split | Convective | Grey | Decoupled |
|---|---:|---:|---:|
| All 60 | 20/60 | 37/60 | 37/60 |
| Teff < 7500 K | 0/27 | 6/27 | **12/27** |
| Teff >= 7500 K | 20/33 | 31/33 | 25/33 |
| Timeouts | 3 | 3 | 5 |
| Mean iterations (converged) | 10.7 | 10.3 | 11.73 |

Continuation-gate failures: hot `25/33` (need ≥30), total `37/60` (need
≥41), losses among grey-converged `10` (need ≤2), net paired gain `0`
(need ≥4), timeouts `5` (need ≤3). Cool `12/27` passed the ≥11 floor.

Paired versus grey: both 27, decoupled-only 10, grey-only 10, neither
13. All 10 decoupled-only wins are cool. Paired versus convective: net
+17, convective-only 0.

Closeout:
`notes/textbook_opacity_v4r6_decoupled_dev60_closeout_20260828.md`.
Production remains the default initializer. H-minus implementation
repair was not run.
