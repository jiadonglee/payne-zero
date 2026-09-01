# Textbook opacity v4 preregistration

Date: 2026-08-26

This is a new development-only candidate. It does not change the production
solver, the default initializer, or any sealed holdout. The v3
`FAIL_STOP` record remains unchanged.

## Pre-run sanity result and target definition

The strict five-label corpus is converged solver output. The production
warm-start opacity flags enable both selected and detailed line opacity
(`IFOP(15)=1` and `IFOP(17)=1`). A fixed 20-row development reference grid was
replayed with the production continuum and line paths, and then with both
line flags disabled. The replay used the same Rosseland inverse-opacity
algebra on a stride-16 frequency grid; it did not run a temperature iteration
and did not inspect sealed rows.

The machine-readable result is
`results/analytic_initializer/textbook_opacity_v4_sanity_20260826.json`.
Across 1,600 reference layers, the measured line-plus-continuum minus
continuum-only effect has p50 `0.0949` dex, p95 `0.2381` dex, maximum `0.4901`
dex, and positive fraction `1.0`. In the registered 4000--7000 K Teff and
4000--7000 K layer-temperature slice, the four increasing `[M/H]` bins have
line-effect medians `0.0182`, `0.0418`, `0.1020`, and `0.1489` dex. This
confirms that the stored `kappa_R` target is a total-opacity target with a
metallicity-dependent line contribution; it is not a continuum-only truth
array.

The v4 candidate is therefore a continuum construction. The full-corpus
comparison to stored `kappa_R` remains reported as an operational
comparability diagnostic, but a failure of that comparison is not by itself a
claim that the continuum construction failed. The true-`P,T` mass bridge is
still an operational warm-start test against the stored converged column-mass
profile and is not relaxed.

## Fixed node-level construction

The five frequency windows and 32-point Gauss--Legendre order are unchanged
from v3. The nodes are mapped in `u = h nu / (k T)` separately in each window.
Every node receives the analytic Rosseland weight

```text
R(u) = u^4 exp(-u) / (1 - exp(-u))^2
```

and the normalized node weights are used directly in

```text
1 / kappa_R = sum_nodes w_node / kappa_nu.
```

The `u=100` tail truncation and 5 x 32 quadrature order are implementation
constants, not corpus-derived parameters. This follows the Rosseland
inverse-opacity definition and analytic weight used in the published
[Rosseland mean formulation](https://www.mdpi.com/2218-2004/6/3/35).

The only local thermodynamic closure is the existing Saha H-plus-donor
calculation. The fixed monochromatic components are:

1. Hydrogenic bound-free for `n=1,...,10`. The level energy is
   `E_n = 13.5984 eV (1 - 1/n^2)`, the statistical weight is `g_n = 2 n^2`,
   and the ten-level neutral-H partition is normalized before applying the
   Boltzmann populations. The threshold is
   `nu_n = 13.5984 eV / (h n^2)`, the threshold cross section is
   `sigma_edge,n = 6.30e-18 n^2 cm2`, and
   `sigma_n(nu) = sigma_edge,n (nu_n/nu)^3` above threshold and zero below.
   The threshold scaling is the explicit low-order hydrogenic approximation;
   it is not fitted to the corpus.
2. H free-free uses the standard cgs coefficient
   `3.6919e8 n_e n_p T^(-1/2) nu^(-3) (1-exp(-u))/rho`, with unit Gaunt
   factor. The proton density comes from the same Saha hydrogen-ionization
   fraction.
3. H-minus bound-free uses the John (1988) photodetachment polynomial with
   `lambda` in micrometres, `lambda_0=1.6419 micrometres`, and fixed
   coefficients `(152.519, 49.534, -118.858, 92.536, -34.194, 4.982)`.
   The formula is evaluated over its declared `0.125 <= lambda < 1.6419`
   micrometre range and is zero outside that range. The H-minus population is
   the existing Saha closure.
4. H-minus free-free uses the published John (1988) six-term polynomial with
   its fixed short- and long-wavelength coefficient tables, rather than the
   v3 window constant. It is used in its declared `1400 <= T <= 10080 K`
   temperature range and zero outside it.
5. Thomson scattering is frequency independent. Hydrogen Rayleigh scattering
   uses the explicit 500 nm normalization and its `nu^4` dependence.

For every absorptive component the stimulated-emission factor is evaluated at
the node's actual `u`; there is no fixed photon-energy pivot. All constants
are exposed in `TextbookOpacityConstants` and are literature/physics
constants or stated validity limits. No corpus fit, label polynomial, window
amplitude, or learned parameter is introduced.

The John (1988) H-minus formula is documented with its coefficient tables in
the [John-opacity implementation reference](https://pyratbay.readthedocs.io/en/docs2.0/cookbooks/opacity_h_ion.html).
The earlier Wishart (1979) calculation is the physical photodetachment
reference underlying this family of H-minus cross sections; see the
[Wishart paper record](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/187.1.59P).

## Applicability

The formal v4 domain remains `T >= 3200 K`, inherited explicitly from v3.
Layers below 3200 K remain in output and are reported separately; no H2O/CO/TiO
term is added in this round. The John H-minus formula's wider validity range
does not silently lower the formal domain because molecular opacity is still
outside this candidate.

## Registered validation sequence

1. Run the full manifest-excluded development validation split: 10,228 stars,
   80 layers, split seed `20260816`, with signed residuals, absolute p50/p95/
   max, positive fraction, component fractions, and Teff-band x layer-
   temperature tables.
2. Report the stored-total-opacity comparison with the original `0.30 dex`
   cool and `0.50 dex` middle p95 limits. This is explicitly labeled a
   total-opacity comparability diagnostic because the sanity replay measured a
   line floor.
3. Run the true-`P,T` integrated-mass bridge against the stored converged
   column mass. The bridge requirement remains p95 `<= 0.20 dex`; no allowance
   is created by the line-floor diagnosis. A bridge failure is `FAIL_STOP`.
4. If the operational bridge passes, evaluate ODE mass error for truth `T` and
   grey-plus-adiabatic `T`, then run the registered 12-star smoke (`>=11/12`)
   and 60-star funnel against grey15, parity, and production.

The fixed 20-row line-free replay is a reachability diagnostic, not a license
to tune the full-corpus thresholds after observing results. No gate is changed
silently and no sealed holdout is opened. A finite output or a better offline
profile does not establish solver convergence, flux parity, or production
readiness.

## Formal offline result

The full manifest-excluded validation run used 10,228 stars and 818,240
layers. The formal `T >= 3200 K` domain retained 786,132 layers and reported
32,108 colder layers separately. There were no non-finite v4 opacity or
integrated-mass values.

The stored-total-opacity comparison did not pass the cool p95 gate, and the
true-`P,T` mass bridge did not pass:

| Teff band | opacity signed median dex | opacity absolute p95 dex | true-`P,T` mass p95 dex | dominant component |
|---|---:|---:|---:|---|
| `Teff < 6000 K` | -0.231 | 0.474 | 0.534 | H-minus bound-free |
| `5500--7000 K` | -0.131 | 0.393 | 0.407 | H-minus bound-free |
| `6000 <= Teff < 10000 K` | +0.010 | 0.412 | 0.337 | hydrogen bound-free |
| `Teff >= 10000 K` | +0.124 | 0.656 | 0.397 | hydrogen bound-free |

The formal cool and middle opacity limits remain `0.30` and `0.50` dex. The
cool opacity p95 is outside the `0.10 dex` bridge allowance, so the registered
bridge allowance does not apply. The v4 result is therefore `FAIL_STOP` before
the grey-plus-adiabatic ODE ablation, 12-star smoke, and 60-star funnel. The
machine-readable result is
`results/analytic_initializer/textbook_opacity_v4_offline_validation.json`.

The failure is not erased by the line-floor diagnosis: the line-free 20-point
sanity comparison had overall v4-minus-continuum p95 `0.448` dex, while the
operational mass bridge against the actual stored target still reached `0.534`
dex in the cool band. The next action is bounded diagnosis of the remaining
continuum/ODE construction, not threshold tuning or a solver run.
