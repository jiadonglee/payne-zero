# Textbook opacity v4r4 hot-layer production-flag ablation

Date: 2026-08-28

This is a development-only diagnostic. It does not change v4r3 or v4r4, the
production solver, the default initializer, or any sealed holdout. The v4r3
offline `FAIL_STOP` remains authoritative. The v4r4 He II node law remains
frozen in the tree; this task does not reopen it as a middle-gate repair.
No 10,228-star offline run, ODE, smoke, funnel, or sealed holdout is run.

## Why this task exists

v4r3 failed the middle opacity gate. The fail is isolated to `T >= 15000 K`
layers (signed median `-0.299` dex, p95 `1.063` dex). v4r4 added hydrogenic
He II (Flag 5) as the registered candidate for that box. A pre-offline
`(log g=4, P=10^5)` T-scan then showed:

- `kappa_v4r4 / kappa_v4r3 = 1.000` at 15000, 18000, 20000, and 25000 K;
- those Rosseland means sit in `balmer_to_lyman`;
- v4r3 hydrogen bound-free carries Rosseland weight `0.87--0.98` there;
- He II n=1 is EUV; He III free-free is closed until `T ≳ 30000 K`.

The T-scan falsifies Flag 5 as the Rosseland mechanism in the registered
failure box. It does not identify which production continuum *does* carry
the extra opacity. This ablation asks that question on the same 20-star
reference grid used by v4 sanity and the v4r1 molecule ablation.

## Question

On `T >= 15000 K` layers, with lines off, which production IFOP flag
accounts for `log10(kappa_continuum / kappa_v4r3)`?

Competing, distinguishable answers:

1. Flag 0 (H bf/ff) — textbook hydrogenic Balmer continuum is the mismatch.
2. Flag 4 (He I, including e+He+ free-free) — helium neutral continuum.
3. Flag 5 (He II) — T-scan predicted null; include as a negative control.
4. Flag 9 (lukewarm metals) or Flag 10 (hot metals).
5. None of the above (residual persists after every single-flag knockout).

## Fixed protocol

Use the v4 sanity / molecule-ablation 20-star `Teff x [M/H]` reference grid,
split seed `20260816`, stride 16, no temperature iteration, no sealed rows.

Replay the production continuum with:

- lines `IFOP(15)=0` and `IFOP(17)=0` (flags 14 and 16 in the 20-vector);
- molecules off;
- baseline: remaining DEFAULT continuum flags unchanged;
- then six knockouts, one flag at a time set to 0: Flag 0, 4, 5, 8, 9, 10.

Evaluate frozen v4r3 on the stored true `(T, P)` layers. Stored `n_e` is not
an input to v4r3. Do not retune gates. Do not construct a new opacity law.

Primary slice: every finite reference layer with `T >= 15000 K`.
Control slice: `8000 <= T < 15000 K`. Report `[M/H]` breakdowns but do not
change the rule after seeing them.

## Registered residuals

```text
flag_X_effect     = log10(kappa_baseline / kappa_flag_X_off)
v4r3_minus_base   = log10(kappa_v4r3 / kappa_baseline)
```

`flag_X_effect` signed median `>= 0.05` dex on the primary slice means that
flag carries Rosseland weight there. The T-scan predicts Flag 5 effect
`< 0.05` dex.

## Registered verdict

On the primary `T >= 15000 K` slice:

- `HYDROGEN_CONTINUUM_MISMATCH` if Flag 0 effect signed median `>= 0.15` dex
  and is strictly larger than every other listed flag effect;
- `HELIUM_NEUTRAL_CONTINUUM` if Flag 4 effect signed median `>= 0.15` dex
  and is strictly larger than Flag 0, 5, 8, 9, and 10;
- `HOT_METAL_CONTINUUM` if Flag 9 or Flag 10 effect signed median `>= 0.15`
  dex and is strictly larger than Flag 0, 4, 5, and 8;
- `HELIUM_IONIZED_CONFIRMED_NULL` is recorded independently if Flag 5
  effect signed median `< 0.05` dex (this does not by itself decide the
  missing-opacity identity);
- `INCONCLUSIVE` if no single flag satisfies the exclusive rules above.

A hydrogen verdict licenses a later registered repair of the textbook H I
bound-free (Karzas/Gaunt or Balmer-edge law), not a corpus fit. A metal
verdict does not license a line haze. A Flag-5 null does not license
deleting the frozen v4r4 functions.

The registered output is
`results/analytic_initializer/textbook_opacity_v4r4_hot_flag_ablation_20260828.json`.

## Remote execution

The 20-star replay ran on `astronode-garching` (Node-06), checkout
`/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero`. Log:
`logs/textbook_opacity_v4r4_hot_flag_ablation.log`. Local Numba/NumPy
could not import the production continuum; the node `.venv-linux` could.

## Post-run formal result

The diagnostic completed on 2026-08-28. The JSON SHA-256 is
`c136b076d5f135733e4d7e43081d2ed8040f3586b0f4cbd01283628dda613b66`.
Production, solver, v4r3/v4r4 physics, gates, ODE, smoke, funnel, and
sealed boundaries remained closed. Lines and molecules were off.

Primary slice `T >= 15000 K`: 133 layers. Control `8000--15000 K`: 324
layers. The machine verdict is `HYDROGEN_CONTINUUM_MISMATCH`. Flag 5
(He II) is independently `helium_ionized_confirmed_null`.

| flag | primary signed median (dex) | primary p95 |abs| (dex) |
|---:|---:|---:|
| 0 H bf/ff | 0.736 | 1.219 |
| 4 He I | 0.040 | 0.061 |
| 5 He II | 0.00008 | 0.110 |
| 8 metals+CIA | 0.00003 | 0.021 |
| 9 lukewarm metals | 0.00045 | 0.009 |
| 10 hot metals | 0.00005 | 0.0009 |

`v4r3_minus_base` on the primary slice: signed median `-0.119` dex, mean
`-0.251`, p95 `1.043`, positive fraction `0.361`. On the control slice
the same residual flips sign: signed median `+0.208` dex, p95 `0.315`,
positive fraction `1.0`. The hydrogenic H I law is high at
`8000--15000 K` and low at `T >= 15000 K`. Metallicity does not change
the identity: Flag 0 remains largest in every `[M/H]` bin.

The next registered stage is a hydrogen bound-free repair
(Karzas/Gaunt or Balmer-edge law), not He II, not a metal haze, and not
a 10,228-star v4r4 offline run.
