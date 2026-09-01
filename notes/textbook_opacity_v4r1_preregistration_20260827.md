# Textbook opacity v4r1 preregistration

Date: 2026-08-27

v4r1 is a new development-only repair candidate. The historical v4 code path,
preregistration, and `FAIL_STOP` result remain authoritative and are not
overwritten. v4r1 does not change the production solver, default initializer,
sealed holdout, five frequency windows, 32 nodes per window, `u=100`
truncation, or formal `T >= 3200 K` domain.

## Registered diagnosis

The external review identified two falsifiable cold-opacity defects:

1. the compact v4 charge balance omits Si and Al and applies one implicit
   partition ratio to every donor;
2. the John H-minus free-free coefficient is already normalized per neutral
   hydrogen atom and electron pressure, but v4 applies an additional
   stimulated-emission factor.

Before the formal offline gate, the fixed manifest-excluded development split
will be used for D1 electron-density diagnostics and the existing deterministic
20-star `Teff x [M/H]` reference grid will be used for D2. Stored electron
density is allowed only in an explicitly labeled oracle diagnostic. It is
forbidden as an input to the candidate.

D1 reports historical-v4 and v4r1 `log10(n_e / n_e,stored)` by
`Teff x layer temperature x [M/H]`, the v4r1 charge-balance residual, and a
stored-`n_e` opacity substitution on the 20 reference stars. D2 reports, at
every v4r1 node,

```text
raw window weight = sum_window w
inverse-opacity window fraction =
    sum_window(w / kappa_nu) / sum_all(w / kappa_nu)
component log sensitivity =
    sum(w kappa_component / kappa_total^2) / sum(w / kappa_total).
```

The reference comparison uses the production continuum replay with line flags
15 and 17 disabled, molecules enabled, stride 16, no temperature iteration,
and no sealed rows.

## Fixed v4r1 construction

The neutral mean molecular weight remains `mu=1.30`; density closure is
reported but is not changed in the same candidate.

The local charge balance includes H plus Na, K, Ca, Mg, Fe, Al, and Si. Solar
number abundances follow the existing AGSS09 production convention. `[M/H]`
scales every donor; `[alpha/M]` additionally scales Mg, Si, and Ca. The
first-ionization energies and fixed ground-term approximations to
`2 U_II / U_I` are:

| donor | ionization energy (eV) | `2 U_II / U_I` |
|---|---:|---:|
| Na | 5.1391 | 1 |
| K | 4.3407 | 1 |
| Ca | 6.1132 | 4 |
| Mg | 7.6462 | 4 |
| Fe | 7.9024 | 20/9 |
| Al | 5.9858 | 1/3 |
| Si | 8.1517 | 4/3 |

These are named atomic constants and degeneracy ratios, not corpus-fit
parameters.

The historical v4 H-minus bound-free, hydrogen bound/free-free, Thomson, and
Rayleigh branches are unchanged. H-minus bound-free retains its
stimulated-emission factor. John H-minus free-free is evaluated as

```text
kappa_Hminus_ff = k_John(lambda, T) n(H0) P_e / rho
                  = k_John(lambda, T) n(H0) n_e k_B T / rho.
```

No additional `1-exp(-h nu/kT)` factor is applied to that branch. No molecular
or metal bound-free opacity is added.

## Registered formal validation and stop rule

After the diagnostic JSON exists, run the original manifest-excluded
development validation split: 10,228 stars, 80 layers, split seed `20260816`.
Report stored-total-opacity residuals, true-`P,T` integrated-mass residuals,
electron-density residuals, inverse-opacity window fractions, and component
log sensitivities.

All gates are required:

- cool (`Teff < 6000 K`) opacity p95 `<= 0.30 dex`;
- middle (`6000 <= Teff < 10000 K`) opacity p95 `<= 0.50 dex`;
- cool and middle true-`P,T` mass p95 both `<= 0.20 dex`.

The historical `0.10 dex` bridge allowance is still reported but does not
replace any required gate. The measured line floor does not relax the mass
bridge.

If any gate fails, record `FAIL_STOP`. If all gates pass, record
`QUALIFIED_BUT_STOP_AFTER_OFFLINE`. In either case this task does not run the
ODE temperature ablation, 12-star smoke, 60-star funnel, production solver, or
sealed holdout.

The registered outputs are:

- `results/analytic_initializer/textbook_opacity_v4r1_diagnostics_20260827.json`;
- `results/analytic_initializer/textbook_opacity_v4r1_offline_validation_20260827.json`.

## Post-run formal result

The preregistered diagnostic and full development validation completed on
2026-08-27. The diagnostic JSON SHA-256 is
`eb1d390aac5829a04501d92443652c2fb663fad210c3a584d98ab6417fa3fadc`.
It confirms that stored electron density was used only by the oracle branch,
the candidate charge-balance maximum relative residual was
`6.93e-14`, and the production/solver/ODE/smoke/funnel/sealed boundaries
remained closed.

The formal run covered 10,228 stars and 818,240 layers. The registered
`T >= 3200 K` domain retained 786,132 layers, with 32,108 colder layers
reported separately. The non-finite count was zero.

| gate | observed p95 (dex) | limit (dex) | result |
|---|---:|---:|---|
| cool opacity | 0.3117 | 0.30 | fail |
| middle opacity | 0.4060 | 0.50 | pass |
| cool true-`P,T` mass | 0.3768 | 0.20 | fail |
| middle true-`P,T` mass | 0.3061 | 0.20 | fail |

The machine decision is `FAIL_STOP`. No ODE temperature ablation, 12-star
smoke, 60-star funnel, production solver, or sealed holdout was run. The
formal-result JSON SHA-256 is
`2d95b15d573b67a7c22aa91ac12b432d0a93326c0c4620e5ea79dc51f889dcd3`.
