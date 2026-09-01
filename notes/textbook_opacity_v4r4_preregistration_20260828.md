# Textbook opacity v4r4 preregistration

Date: 2026-08-28

v4r4 is a new development-only He II continuum candidate. The historical v4,
v4r1, v4r2, and v4r3 code paths, preregistrations, and `FAIL_STOP` records
remain authoritative and are not overwritten. v4r4 does not change the
production solver, default initializer, sealed holdout, five frequency
windows, 32 nodes per window, `u=100` truncation, seven-donor Saha charge
balance, John H-minus coefficients, particle-count density, H2+ or He-minus
formulae, or the formal domain `T >= 4000 K`. No molecular band is added.
Stored electron density is not an input. Helium is still not a Saha electron
donor. No Rayleigh He, Rayleigh H2, or metal bound-free term is added in this
round.

## Registered diagnosis

The v4r3 offline validation
(`notes/textbook_opacity_v4r3_preregistration_20260827.md`, JSON SHA-256
`04432ac37667fb7ad1f49de23ef72e6de2cd94b419c8e2959abf2ffecef867cc`) failed
middle opacity and both true-`(P,T)` mass gates. Cool opacity passed.

Three independent read-only probes (2026-08-28) split that record:

1. **Middle opacity is isolated to `T >= 15000 K`.** In the middle Teff band
   on the `T >= 4000 K` domain, the `at_least_15000K` box (71,539 layers,
   15.7%) has signed median `-0.299` dex and p95 `1.063` dex. Every other
   middle-band layer-T box has p95 `<= 0.258` dex. Dropping `T >= 15000 K`
   would pass the middle opacity gate. v4r2 had the same box at signed
   `-0.077` / p95 `0.873` and still passed middle opacity because the fixed
   `mu=1.30` density was accidentally high. Particle-count density removed
   that compensation. Those layers are missing He II (`IFOP`/`Flag 5`).
2. **Cool mass is not an He II problem, and it is not a missing Rayleigh
   scatterer.** Cool 4000–5000 K opacity remains signed `-0.120` dex. Cool
   mass on the gated domain is signed `+0.122` dex, p95 `0.238`, positive
   fraction `0.999`. The mass integral still starts at the surface.
   Out-of-gate cool layers are worse: 3200–4000 K opacity signed `-0.194`
   and mass `+0.251`; `T < 3200 K` opacity `-0.488` and mass `+0.596`.
   H2+ and He-minus did not move this offset. Rayleigh He (`Flag 7`) and
   Rayleigh H2 (`Flag 12`) are implementable as named constants but are
   λ⁻⁴ in the H-minus IR window and are not expected to close 0.12 dex.
   He I and metal bound-free are table-heavy and UV/optical. The cool-band
   inverse-opacity window on the gated domain is `paschen_to_balmer`
   (median 0.454), not `below_hminus_threshold` (median 0.070).
3. **Four prior cool-mass interventions have failed** (v4, v4r1 charge
   balance, v4r2 domain raise, v4r3 H2+/He-minus). This round does not add
   a fifth unmeasured cool absorber. A later, separately registered 20-star
   production continuum flag ablation may identify the remaining atomic IR.
   That ablation is not this candidate.

The two FAIL_STOP mechanisms are therefore separable. v4r4 tests only the
He II hypothesis.

## Fixed v4r4 construction

Keep the frozen v4r3 state: particle-count density, seven-donor Saha, John
H-minus, H2+, He-minus, no He donor. After that state converges, compute a
two-step hydrogenic helium ionization with the frozen `n_e`:

```text
phi_1 = n(He II)/n(He I)  = 4 * (2 pi m_e kT / h^2)^{3/2} / n_e * exp(-24.587 eV / kT)
phi_2 = n(He III)/n(He II) = 1 * (2 pi m_e kT / h^2)^{3/2} / n_e * exp(-54.418 eV / kT)
n_I   = n_He / (1 + phi_1 + phi_1 phi_2)
n_II  = n_He * phi_1 / (1 + phi_1 + phi_1 phi_2)
n_III = n_He * phi_1 phi_2 / (1 + phi_1 + phi_1 phi_2)
```

`4` and `1` are ground-term `2 U_{r+1}/U_r` values, not corpus fits.
`24.587 eV` and `54.418 eV` are NIST first and second helium ionization
energies. Production Flag 5 uses `54.403 eV` in its Boltzmann factor; the
0.015 eV difference is recorded and is not a free parameter.

He II bound-free is the existing textbook H I node law at `Z=2`:

```text
sigma_edge(n) = (6.30e-18 cm2 / Z^2) * n^2
nu_n          = (54.418 eV / n^2) / h
kappa_bf      = sum_n n_n(He II)/rho * sigma_edge(n) * (nu_n/nu)^3 * Theta(nu>=nu_n) * (1-e^{-u})
```

with `n = 1..10` and the same Boltzmann level weights as H I. He II
free-free is the existing H I free-free law at `Z^2=4` with Gaunt factor 1:

```text
kappa_ff = 4 * 3.6919e8 * n_e * n(He III) / (rho * sqrt(T) * nu^3) * (1-e^{-u})
```

Do not load Karzas-Latter or Coulomb Gaunt tables. Do not implement the
production `1.31522e14 Hz` high-n extension (textbook H I has no analogue).
Do not replace the v4r3 He-minus density with `n_I` in this round; that
would confound the He II ablation. Cool He-minus already uses total helium,
which equals `n_I` in the cool photosphere.

## Registered formal validation and stop rule

Use the original manifest-excluded development validation split: 10,228
stars, 80 layers, split seed `20260816`. Evaluate v4r4 on true `(P, T)`.
Report the same quantities as v4r3, plus He II bound-free and free-free
log sensitivities, and the middle-band `at_least_15000K` signed residual.

All gates remain the v4r2/v4r3 gates on `T >= 4000 K`:

- cool (`Teff < 6000 K`) opacity p95 `<= 0.30` dex;
- middle (`6000 <= Teff < 10000 K`) opacity p95 `<= 0.50` dex;
- cool and middle true-`P,T` mass p95 both `<= 0.20` dex.

The historical `0.10` dex bridge allowance is still reported but does not
replace any required gate. The mass integral still starts at the surface.

Falsification for the He II hypothesis, independent of the mass gates: in
the middle Teff band, `T >= 15000 K` layers, the signed median opacity
residual must increase algebraically from v4r3's `-0.299` dex (textbook
kappa too low) toward zero. If it stays `<= -0.25` dex, or becomes more
negative, the hydrogenic He II node law is the wrong mechanism. Cool-band
opacity p95 must remain `<= 0.30` dex; He II must be negligible there.

If any gate fails, record `FAIL_STOP`. If all gates pass, record
`QUALIFIED_BUT_STOP_AFTER_OFFLINE`. In either case this task does not run
the ODE temperature ablation, 12-star smoke, 60-star funnel, production
solver, or sealed holdout. A later cool-mass flag ablation, if opened, is a
separate note.

The registered output is
`results/analytic_initializer/textbook_opacity_v4r4_offline_validation_20260828.json`.

Required prior artifacts, cited not overwritten:

- `results/analytic_initializer/textbook_opacity_v4r3_offline_validation_20260827.json`;
- `results/analytic_initializer/textbook_opacity_v4r2_offline_validation_20260827.json`;
- `results/analytic_initializer/textbook_opacity_v4r1_molecule_ablation_20260827.json`.

## Pre-offline T-scan (registered falsification, 10k run not started)

After the v4r4 functions were implemented, a one-point `(log g=4, [M/H]=0, P=10^5 dyn cm^{-2})` scan was run *before* the 10,228-star offline validation. Hydrogenic He II does not move the Rosseland mean in the registered failure box:

| T (K) | kappa_v4r4 / kappa_v4r3 | He II/He | Rosseland window | v4r3 H I bf Rosseland weight |
|---:|---:|---:|---|---:|
| 15000 | 1.0000 | 0.005 | balmer_to_lyman | 0.975 |
| 18000 | 1.0000 | 0.138 | balmer_to_lyman | 0.965 |
| 20000 | 1.0000 | 0.501 | balmer_to_lyman | 0.949 |
| 25000 | 1.0000 | 0.968 | balmer_to_lyman | 0.866 |
| 40000 | 1.677 | 0.764 He II, 0.236 He III | above_lyman | 0.499 |

He II n=1 sits at 54.418 eV (EUV). He III free-free is exponentially closed until `T ≳ 30000 K`. The middle-band `T >= 15000 K` residual is therefore not a Flag-5 He II hole. It is a `balmer_to_lyman` residual, and the textbook Rosseland weight there is hydrogen bound-free. A Flag-4 He+ free-free trial at the same points moved kappa_R by at most 0.008.

The 10,228-star v4r4 offline run is **not started**. Starting it would spend the full corpus to remeasure a null that the T-scan already shows. The He II node law remains in the tree as a frozen, tested construction; it is not a middle-gate repair. The next registered diagnostic is the 20-star hot-layer production-flag ablation
(`notes/textbook_opacity_v4r4_hot_flag_ablation_preregistration_20260828.md`).
