# Textbook opacity v4r2 preregistration

Date: 2026-08-27

v4r2 is a new development-only applicability candidate. The historical v4 and
v4r1 code paths, preregistrations, and `FAIL_STOP` records remain
authoritative and are not overwritten. v4r2 does not change the production
solver, default initializer, sealed holdout, five frequency windows, 32 nodes
per window, `u=100` truncation, Saha donors, John H-minus free-free units, or
any named constant. No molecular or metal bound-free term is added.

## Applicability change

The formal atomic-opacity domain is changed explicitly from `T >= 3200 K` in
v4r1 to `T >= 4000 K` in v4r2. Layers below 4000 K remain in the output and
are reported separately, split into the historical `T < 3200 K` tail and the
newly excluded `3200--4000 K` band. They do not enter the formal gate.

This is a declared domain change, not a hidden molecular-opacity correction.
The v4r1 molecule-on/off ablation already returned `ATOMIC_IR_REMAINS` on
`3200--4000 K`; raising the floor does not claim that those layers are
molecular, and it does not relax the cool or middle mass-bridge limits.

The opacity law evaluated at `T < 4000 K` is unchanged v4r1 physics. Stored
electron density is not an input.

## Registered formal validation and stop rule

Use the original manifest-excluded development validation split: 10,228 stars,
80 layers, split seed `20260816`. Evaluate the v4r1 node construction on true
`(P, T)`. Report stored-total-opacity residuals, true-`P,T` integrated-mass
residuals, electron-density residuals, inverse-opacity window fractions, and
component log sensitivities on `T >= 4000 K`, with `3200--4000 K` and
`T < 3200 K` reported separately.

All gates are required on the new domain:

- cool (`Teff < 6000 K`) opacity p95 `<= 0.30 dex`;
- middle (`6000 <= Teff < 10000 K`) opacity p95 `<= 0.50 dex`;
- cool and middle true-`P,T` mass p95 both `<= 0.20 dex`.

The historical `0.10 dex` bridge allowance is still reported but does not
replace any required gate. The measured line floor does not relax the mass
bridge. The mass integral still starts at the surface, so excluding
`T < 4000 K` from the p95 mask does not remove those layers from `m(tau)`.

If any gate fails, record `FAIL_STOP`. If all gates pass, record
`QUALIFIED_BUT_STOP_AFTER_OFFLINE`. In either case this task does not run the
ODE temperature ablation, 12-star smoke, 60-star funnel, production solver, or
sealed holdout.

The registered output is
`results/analytic_initializer/textbook_opacity_v4r2_offline_validation_20260827.json`.

Required prior artifacts, cited not overwritten:

- `results/analytic_initializer/textbook_opacity_v4r1_offline_validation_20260827.json`;
- `results/analytic_initializer/textbook_opacity_v4r1_molecule_ablation_20260827.json`.

## Post-run formal result

The preregistered full development validation completed on 2026-08-27. The
JSON SHA-256 is
`2a48fb9d8111fbf5ef752818db272f58952ca1dc49fc6221250c88944b7a44d0`.
It cites the v4r1 offline result and the molecule-ablation verdict
`ATOMIC_IR_REMAINS`. The construction is unchanged v4r1. No molecular term was
added. Production/solver/ODE/smoke/funnel/sealed boundaries remained closed.

The run covered 10,228 stars and 818,240 layers. The registered `T >= 4000 K`
domain retained 703,273 layers. The newly excluded `3200--4000 K` band has
82,859 layers; the historical `T < 3200 K` tail has 32,108 layers. The
non-finite count was zero.

| gate | observed p95 (dex) | limit (dex) | result |
|---|---:|---:|---|
| cool opacity | 0.2137 | 0.30 | pass |
| middle opacity | 0.4061 | 0.50 | pass |
| cool true-`P,T` mass | 0.2337 | 0.20 | fail |
| middle true-`P,T` mass | 0.3057 | 0.20 | fail |

The newly excluded `3200--4000 K` opacity p95 is 0.422 dex with signed median
`-0.189` dex; those layers are reported and are not in the formal gate. The
mass integral still includes them, which is why the cool mass p95 remains
above 0.20 dex after the domain raise.

The machine decision is `FAIL_STOP`. No ODE temperature ablation, 12-star
smoke, 60-star funnel, production solver, or sealed holdout was run.
