# Textbook opacity v4r3 preregistration

Date: 2026-08-27

v4r3 is a new development-only mass-bridge candidate. The historical v4,
v4r1, and v4r2 code paths, preregistrations, and `FAIL_STOP` records remain
authoritative and are not overwritten. v4r3 does not change the production
solver, default initializer, sealed holdout, five frequency windows, 32 nodes
per window, `u=100` truncation, seven-donor Saha charge balance, John H-minus
coefficients, or the v4r2 formal domain `T >= 4000 K`. No molecular band
(H2O/CO/TiO) term is added. Stored electron density is not an input.

## Registered diagnosis

The v4r2 domain raise passed both opacity gates and failed both mass gates.
Stored-`kappa_R` integration recovers column mass to p95 `0.006` dex, so the
mass miss is opacity physics, not the trapezoid or the `m_0 = tau_0/kappa_0`
surface rule.

Two independent mechanisms are already in the v4r1/v4r2 artifacts:

1. **Ionized density.** v4r1 deferred the neutral `mu=1.30` closure. D1 already
   measured that this density is high relative to production (signed median
   `+0.023` dex, p95 `0.314` dex, positive fraction `0.984`). The v4r2 hot
   `n_e` residual is the same number (signed `+0.199`, p95 `0.294`, positive
   fraction `1.0`). Middle-band opacity changes sign with layer temperature
   (`4000--6000 K` low, `7000--15000 K` high), and middle mass is mixed-sign
   scatter (positive fraction `0.533`, p95 `0.306` dex).
2. **Cool photospheric `1/kappa`.** Cool mass is one-sided (positive fraction
   `1.0`, signed median `+0.124`, p95 `0.234`). Forty-three percent of the
   cool `T >= 4000 K` gate layers sit in `4000--5000 K`, where the Rosseland
   mean is H-minus and the column still feels the more transparent
   `T < 4000 K` integral. The molecule-ablation verdict remains
   `ATOMIC_IR_REMAINS`. John H-minus bound-free and free-free already match
   the production tables at the coefficient level; the omitted production
   atomic continua in that window are H2+ (`IFOP(2)`) and He-minus
   (`IFOP(7)`).

Neither mechanism licenses a corpus fit, a line haze, or a silent gate
change. He ionization is not added to the charge balance in this candidate;
helium enters only as a nucleus in the particle count and as the He-minus
absorber.

## Fixed v4r3 construction

Keep the v4r1 seven-donor Saha (H, Na, K, Ca, Mg, Fe, Al, Si), AGSS09
abundances, ground-term `2 U_II / U_I`, and John H-minus units. Replace only
the density closure and add the two named atomic continua.

Ideal-gas particle count:

```text
n_tot = P / (k T)
n_nuclei = n_tot - n_e
n_H = n_nuclei / (1 + Y/(4X) + sum_donors n_d/n_H)
rho = n_H m_H / X
```

`X=0.7381` and `Y=0.2485` are the existing named mass fractions. The loop
is the existing damped Saha update, now with `rho` recomputed from `n_e`
each iteration. Neutral layers recover `mu ~ 1.25` instead of the round
`1.30`; ionized hydrogen layers fall to `mu ~ 0.65`. That is a composition
identity, not a fitted cutoff.

H2+ uses the Bates/ATLAS infrared-through-Lyman polynomial already in the
production continuum, evaluated at the v4r3 nodes from `n(H0)` and `n(H+)`.
He-minus uses the Kurucz/ATLAS free-free expansion with the documented
`10^{15} cm^{-3}` density convention:

```text
kappa_He- = (a(nu) T + b(nu) + c(nu)/T) n_e n(He) / rho * 10^{-45}.
```

Negative polynomial samples are clipped to zero. No extra stimulated-emission
factor is applied to He-minus; H2+ keeps the nodal `1-exp(-u)` factor used
by the production branch.

## Registered formal validation and stop rule

Use the original manifest-excluded development validation split: 10,228 stars,
80 layers, split seed `20260816`. Evaluate v4r3 on true `(P, T)`. Report
stored-total-opacity residuals, true-`P,T` integrated-mass residuals,
electron-density residuals, mean-molecular-weight, inverse-opacity window
fractions, and component log sensitivities on `T >= 4000 K`, with
`3200--4000 K` and `T < 3200 K` reported separately.

All gates are required on the v4r2 domain:

- cool (`Teff < 6000 K`) opacity p95 `<= 0.30` dex;
- middle (`6000 <= Teff < 10000 K`) opacity p95 `<= 0.50` dex;
- cool and middle true-`P,T` mass p95 both `<= 0.20` dex.

The historical `0.10` dex bridge allowance is still reported but does not
replace any required gate. The measured line floor does not relax the mass
bridge. The mass integral still starts at the surface.

If any gate fails, record `FAIL_STOP`. If all gates pass, record
`QUALIFIED_BUT_STOP_AFTER_OFFLINE`. In either case this task does not run the
ODE temperature ablation, 12-star smoke, 60-star funnel, production solver, or
sealed holdout.

The registered output is
`results/analytic_initializer/textbook_opacity_v4r3_offline_validation_20260827.json`.

Required prior artifacts, cited not overwritten:

- `results/analytic_initializer/textbook_opacity_v4r1_offline_validation_20260827.json`;
- `results/analytic_initializer/textbook_opacity_v4r1_molecule_ablation_20260827.json`;
- `results/analytic_initializer/textbook_opacity_v4r2_offline_validation_20260827.json`.

## Post-run formal result

The preregistered full development validation completed on 2026-08-27. The
JSON SHA-256 is
`04432ac37667fb7ad1f49de23ef72e6de2cd94b419c8e2959abf2ffecef867cc`.
It cites the v4r1 and v4r2 offline results and the molecule-ablation verdict
`ATOMIC_IR_REMAINS`. Helium was not added as a Saha donor. Production, solver,
ODE, smoke, funnel, and sealed boundaries remained closed.

The run covered 10,228 stars and 818,240 layers. The registered `T >= 4000 K`
domain retained 703,273 layers. The non-finite count was zero. The particle-count
density did what D1 predicted: domain mean molecular weight has median `1.181`,
p05 `0.650`, p95 `1.250`, and the stored-`n_e` residual collapsed (hot p95
`0.025` dex, signed median `+0.000`; v4r2 hot p95 was `0.294`).

| gate | v4r2 p95 (dex) | v4r3 p95 (dex) | limit (dex) | result |
|---|---:|---:|---:|---|
| cool opacity | 0.2137 | 0.2170 | 0.30 | pass |
| middle opacity | 0.4061 | 0.5127 | 0.50 | fail |
| cool true-`P,T` mass | 0.2337 | 0.2375 | 0.20 | fail |
| middle true-`P,T` mass | 0.3057 | 0.3231 | 0.20 | fail |

H2+ carries median log sensitivity `0.037` in the cool band and `0.003` in
the middle band. He-minus is `0.003` and `0.0004`. Neither moved the cool mass
offset (still signed `+0.122` dex, positive fraction `0.999`). The new middle
opacity fail is localized to `T >= 15000 K` layers (middle signed median
`-0.299` dex, p95 `1.063` dex). Correcting `mu` removed an accidental
high-density compensation; those layers are now missing the still-absent
He II continuum. Cool `4000--5000 K` opacity remains signed `-0.120` dex.

The machine decision is `FAIL_STOP`. No ODE temperature ablation, 12-star
smoke, 60-star funnel, production solver, or sealed holdout was run.

A later, separately registered task
(`notes/textbook_opacity_v4r3_dev60_preregistration_20260827.md`) was opened
on explicit user request to send this same frozen v4r3 seed into the
development-60 solver funnel. That task does not reopen or rewrite this
offline `FAIL_STOP`.
