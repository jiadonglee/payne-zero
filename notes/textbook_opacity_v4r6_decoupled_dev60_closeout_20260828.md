# Textbook opacity v4r6 decoupled development-60 closeout

Date: 2026-08-28

Candidate: `v4r6_decoupled_mgrey_tconv_v1`

Final label: **`FAIL_STOP_DEVELOPMENT`**

This note closes WP4–WP6 of
`notes/textbook_opacity_v4r6_decoupled_mgrey_tconv_workplan_20260828.md`.
It does not reopen the v4r6 offline `FAIL_STOP`. It does not authorize
fresh-open 120, a coupled ODE, a production switch, spectra, or the
sealed holdout. It does not repair H-minus.

## Construction

```text
m_seed     = v4r6 mass integrated with Eddington-grey temperature
T_seed     = current Saha-aware convective temperature on P_grey = g m_grey
kappa_seed = kappa_v4r6(T_seed, P_grey)
```

Column mass was not re-integrated after the temperature replacement.
Historical convective and grey v4r6 paths were not modified.

## Hashes

| Artifact | SHA-256 |
|---|---|
| Source/environment manifest | `ebc932f2402d15a936cd1a96465c15262f4f5197ad3636794fe24245db7f152c` |
| Sample manifest (`convergence_metrics_learned_monotone.json`) | `5e0238098f5811de7738d6e8fcf5b9eb5d94fe85a8fa9505ad3513769179e27e` |
| Seed-only audit | `4d7179fce0af46889cf753dc462b28bcd66436d0efcec274bf653e5ed533e2be` |
| Development-60 JSON | `81211e9e61cf9e2ab39a517d3bc4c455a9a1fd702a9a92db35d70abd7609afb0` |
| Development-60 JSONL | `bc3ec0583cd45cf5566ffd4ae8cd37d63bb96ff6bf86630cbd78a60948966bf3` |
| Runtime signature | `3f8fa295e3391c63a1e4bb6a7087f1b85221646e42fc9f92938082bbf491c99c` |
| Log | `ebe0d832f60be8c6d8675d9d8fa78e1e9f7537fa1b53b0d42e250268c3020384` |
| Frozen convective control | `c0c08c9727e522916085941bc5dcb40a96d67fea05852f8d88ddb4cae4cdd3e5` |
| Frozen grey control | `caeee639e37600952be8439d259bdb99f68d992bf9d4c2a50749530f68bf015a` |

Host: `astronode-garching`. Checkout
`/home/jdli/xiasangju/jdli/payne-zero`. Python `.venv-linux/bin/python`.
Parent PID `2688371`. One trial, 15 iterations, 900 s per-star timeout.

## Gate decisions

Structural gate (WP3, before any solver execution): **`PASS_STRUCTURAL`**.
Finite and positive `60/60`. Mass identity bitwise to grey. Temperature
identity bitwise to convective. Opacity identity max relative residual
`0`. Fitted parameters `0`.

Development-60 continuation gate: **`FAIL_STOP_DEVELOPMENT`**.

| Metric | Need | Observed | Check |
|---|---:|---:|---|
| Complete records | 60/60 | 60/60 | pass |
| Solver errors | 0 | 0 | pass |
| Finite seeds | 60/60 | 60/60 | pass |
| Cool convergence | ≥ 11/27 | **12/27** | pass |
| Hot convergence | ≥ 30/33 | 25/33 | fail |
| Total convergence | ≥ 41/60 | 37/60 | fail |
| Losses among 37 grey-converged | ≤ 2 | 10 | fail |
| Net paired gain vs grey | ≥ 4 | 0 | fail |
| Timeouts | ≤ 3 | 5 | fail |

Fresh-open 120: **not run**. Coupled ODE: **not run**. Spectral gate:
**not run**. Sealed holdout: **not run**. H-minus repair: **not run**.
Production solver and default initializer: **unchanged**.

## Solver counts

| Split | Decoupled | Grey | Convective |
|---|---:|---:|---:|
| All 60 | 37/60 | 37/60 | 20/60 |
| Cool \(T_\mathrm{eff} < 7500\,\mathrm{K}\) | **12/27** | 6/27 | 0/27 |
| Hot \(T_\mathrm{eff} \ge 7500\,\mathrm{K}\) | 25/33 | 31/33 | 20/33 |
| Dwarf \(\log g \ge 3.5\) | 13/17 | 8/17 | 3/17 |
| Giant \(\log g < 3.5\) | 24/43 | 29/43 | 17/43 |
| Cool dwarf | **8/11** | 2/11 | 0/11 |
| Cool giant | 4/16 | 4/16 | 0/16 |
| Timeouts | 5 | 3 | 3 |
| Mean iterations (converged) | 11.73 | 10.3 | 10.7 |

Paired versus grey: both 27, decoupled-only 10, grey-only 10, neither 13,
net gain 0. All 10 decoupled-only wins are cool. All 6 extra grey-only
losses on the hot set are hot; the remaining 4 grey-only losses are
cool. Paired versus convective: both 20, decoupled-only 17, convective-only
0, net gain 17. No registered convective success was lost.

## Hypothesis score

The causal question was: does the current convective temperature remain
useful when the grey-derived mass column is preserved?

- **Cool stars: yes.** Cool convergence moved from convective `0/27` and
  grey `6/27` to `12/27`. The cool-dwarf cell is `8/11` against grey
  `2/11`. Keeping `m_grey` and not re-integrating mass is sufficient to
  put the current convective `T` inside a cool basin that the coupled
  seed never reached.
- **Hot stars: no.** Hot convergence fell from grey `31/33` to `25/33`.
  Decoupled-only hot wins: 0. Grey-only hot losses: 6. The same
  convective `T` that helps cool stars knocks hot stars out of the grey
  basin.

`H1_MASS_REINTEGRATION_DAMAGE` is therefore split and, as stated, is
rejected: the conjunction required both a cool gain and retention of
most of the grey hot basin. `H0_NO_GAIN` holds against grey on the
total (`37/60 = 37/60`, net 0). `T_CONSTRUCTION_STILL_BAD` is rejected
for cool stars and supported for hot stars.

## Allowed next action

Stop. Do not run fresh-open 120. Do not promote this candidate into the
registered seed. Do not treat grey-only `T` or this decoupled pair as a
production initializer.

A later candidate may keep grey mass and replace the convective
temperature construction; that is a new preregistration, not a
relaxation of this gate. The separate H-minus implementation branch
remains open and is not authorized by this closeout.
