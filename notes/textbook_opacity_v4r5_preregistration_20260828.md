# Textbook opacity v4r5 preregistration

Date: 2026-08-28

v4r5 is a new development-only hydrogen bound-free population candidate.
Historical v4 through v4r4 paths, preregistrations, and `FAIL_STOP` records
remain authoritative and are not overwritten. v4r5 does not change the
production solver, default initializer, sealed holdout, five frequency
windows, 32 nodes, `u=100` truncation, seven-donor Saha, John H-minus,
particle-count density, H2+, He-minus, the n^2 hydrogenic edge law, or
the formal domain `T >= 4000 K`. Helium is still not a Saha donor. v4r4
He II is not added in this round. No molecular band, Rayleigh extra, or
corpus fit is added.

## Registered diagnosis

The v4r4 hot-layer flag ablation
(`notes/textbook_opacity_v4r4_hot_flag_ablation_preregistration_20260828.md`,
JSON SHA-256
`c136b076d5f135733e4d7e43081d2ed8040f3586b0f4cbd01283628dda613b66`)
returned `HYDROGEN_CONTINUUM_MISMATCH` with Flag 5 independently null.

A layer-temperature split of that 20-star, lines-off continuum replay
shows the mixed primary median is not a single 15000 K flip:

| layer T | n | v4r3 minus production continuum |
|---|---:|---:|
| 8000--15000 K | 324 | +0.208 dex |
| 15000--22000 K | 43 | +0.085 |
| 22000--30000 K | 38 | -0.103 |
| `T >= 30000 K` | 52 | **-0.537** |

Electron density on those stars matches stored `n_e` to 0.02 dex, so the
hot hole is not a charge-balance miss. v4r4 He II recovers 0.16 dex only
for `T >= 30000 K` and leaves the primary median unchanged.

The v4 hydrogenic bound-free loop renormalizes `n(H I)` across ten bound
levels. At 8000--15000 K the ten-level partition is `U_10 ~ 2` and the
ground holds essentially all neutrals. At 40000 K, `U_10 ~ 18` and the
ground fraction falls to `n_1 / n(H I) ~ 0.11`. Those extra high-n levels
are not bound in a real plasma; ATLAS/Flag 0 instead anchors excited
populations on the ground-term factor `n(H I)/U` with `U ~ 2` after
dissolved high-n are removed. Stealing the ground to fund ten fictitious
bound shells underpredicts the Lyman continuum by about a factor of nine
at 40000 K, which is the size of the `T >= 30000 K` residual.

A Kramers `n^1` edge-scaling trial on the same 20 stars *worsens* the
primary slice (median `-0.119` to `-0.250`) and is not this candidate.
The 8000--15000 K Balmer overprediction (`+0.208`) is a separate, later
edge-law question; ground anchoring does not move that slice.

## Fixed v4r5 construction

Keep frozen v4r3 state and continua. Replace only the H I bound-free
level closure. Excited-state populations are Boltzmann from the ground,
with the ground holding the neutral hydrogen density:

```text
n_1 = n(H I)
n_n = n(H I) * n^2 * exp(-chi_H (1 - 1/n^2) / kT)    n = 1..10
```

because `g_n / g_1 = n^2`. Do not divide by `sum_{k=1}^{10} g_k
exp(-E_k/kT)`. The sum of these `n_n` may exceed `n(H I)`; that is the
occupation-probability statement that high-n shells are not a closed
bound reservoir. Edge cross sections, `nu^{-3}` frequency law, ten
levels, and stimulated emission `1-exp(-u)` are unchanged from v4.

## Registered formal validation and stop rule

Use the original manifest-excluded development validation split: 10,228
stars, 80 layers, split seed `20260816`. Evaluate v4r5 on true `(P, T)`.
Report the same quantities as v4r3, plus the middle-band `at_least_15000K`
signed residual and, if stored, the `T >= 30000 K` slice.

All gates remain the v4r2/v4r3 gates on `T >= 4000 K`:

- cool (`Teff < 6000 K`) opacity p95 `<= 0.30` dex;
- middle (`6000 <= Teff < 10000 K`) opacity p95 `<= 0.50` dex;
- cool and middle true-`P,T` mass p95 both `<= 0.20` dex.

Falsification for the ground-anchor hypothesis, independent of the mass
gates: on the frozen 20-star ablation grid, `T >= 30000 K` signed median
`log10(kappa_v4r5 / kappa_production_continuum)` must increase
algebraically from v4r3's `-0.537` dex toward zero, and the 8000--15000 K
median must stay within 0.02 dex of v4r3's `+0.208`. If the hot tail does
not move, or the cool-warm Balmer slice moves by more than 0.02 dex, the
population closure is the wrong mechanism.

If any gate fails, record `FAIL_STOP`. If all gates pass, record
`QUALIFIED_BUT_STOP_AFTER_OFFLINE`. This task does not run ODE ablation,
12-star smoke, 60-star funnel, production solver, or sealed holdout.

The registered output is
`results/analytic_initializer/textbook_opacity_v4r5_offline_validation_20260828.json`.

Required prior artifacts, cited not overwritten:

- `results/analytic_initializer/textbook_opacity_v4r3_offline_validation_20260827.json`;
- `results/analytic_initializer/textbook_opacity_v4r4_hot_flag_ablation_20260828.json`;
- `results/analytic_initializer/textbook_opacity_v4r1_molecule_ablation_20260827.json`.

## Remote execution

Code and the 10,228-star offline run executed on `astronode-garching`
(Node-06), checkout `/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero`,
Python `.venv-linux/bin/python`. Local pytest: 32 passed. Remote pytest:
31 passed; the historical v4 golden values differ at ~1e-15, the same
Linux float discrepancy seen on v4r3, and is not a v4r5 change. Log:
`logs/textbook_opacity_v4r5_offline.log`.

## Post-run formal result

The 20-star production-continuum falsification completed first
(`results/analytic_initializer/textbook_opacity_v4r5_hot_grid_20260828.json`,
SHA-256
`d663496ab9128aa2b4b0ec58560c7def449e44d66f45987bdf5540c31ef67dad`).
Decision `HYPOTHESIS_HOLD`. Control `8000--15000 K` signed median
`+0.2079` (v4r3) to `+0.2086` (v4r5). Hot tail `T >= 30000 K` signed
median `-0.537` to `-0.004`. He II was not added.

The preregistered full development validation then completed on
2026-08-28. The JSON SHA-256 is
`c3ea2b6091d5fcd0f23ab20ca26a69025888cae1a0a9a02870ac12f1562e9df6`.
It cites the v4r3 offline result, the hot-flag ablation
`HYDROGEN_CONTINUUM_MISMATCH`, and the molecule-ablation verdict
`ATOMIC_IR_REMAINS`. Production, solver, ODE, smoke, funnel, and sealed
boundaries remained closed. The n^2 edge law was not changed.

The run covered 10,228 stars. The registered `T >= 4000 K` domain is the
same cutoff as v4r3. The non-finite count was zero.

| gate | v4r3 p95 (dex) | v4r5 p95 (dex) | limit (dex) | result |
|---|---:|---:|---:|---|
| cool opacity | 0.2170 | 0.2170 | 0.30 | pass |
| middle opacity | 0.5127 | 0.2341 | 0.50 | **pass** |
| cool true-`P,T` mass | 0.2375 | 0.2375 | 0.20 | fail |
| middle true-`P,T` mass | 0.3231 | 0.2157 | 0.20 | fail |

Formal opacity against stored total `kappa_R` now passes. Cool mass is
unchanged because the integral still includes `T < 4000 K` surface
layers. Middle mass moved toward the gate but remains 0.016 dex over.
On the 10k split, `T >= 15000 K` versus stored total `kappa_R` has
signed median `-0.075` dex and p95 `0.426` (v4r3 middle-band
`T >= 15000 K` p95 was `1.063`). The 8000--15000 K Balmer overprediction
versus production continuum is unchanged and was not this candidate.

The machine decision is `FAIL_STOP`. No ODE temperature ablation,
12-star smoke, 60-star funnel, production solver, or sealed holdout was
run.
