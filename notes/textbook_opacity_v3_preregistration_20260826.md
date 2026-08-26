# Textbook opacity v3 preregistration

Date: 2026-08-26

This is a new development-only candidate. It does not change the production
solver, the default initializer, or any sealed holdout. The v2 result remains
an independent failure record.

## Fixed construction

The local state remains the Saha-aware `(T, P)` state with `P = g m`. The v3
change is the opacity synthesis, not a fitted correction to the corpus.

The frequency axis is divided into five fixed windows, in increasing
frequency:

1. below the H-minus photodetachment threshold;
2. H-minus threshold to the hydrogen Paschen limit (`n=3`);
3. Paschen to the hydrogen Balmer limit (`n=2`);
4. Balmer to the hydrogen Lyman limit (`n=1`);
5. above the Lyman limit.

The boundaries are derived from the named H-minus affinity and hydrogen
ionization energy. There are no corpus-derived edge positions.

Within each window, the allowed positive local components are added:

- H-minus free-free, gated Kramers free-free, electron scattering, and the
  fixed hydrogen Rayleigh term are present in every window;
- H-minus bound-free starts at the H-minus threshold;
- the n=3 Paschen bound-free component starts at the Paschen limit;
- the n=2 Balmer bound-free component starts at the Balmer limit.

The five window opacities are combined with the Rosseland harmonic rule

```text
1 / kappa_R = sum_i w_i / kappa_i,
R(u) = [15 / (4 pi^4)] u^4 exp(u) / [exp(u) - 1]^2,
u = h nu / (k T).
```

The weights are integrated over the fixed windows using a fixed 32-point
Gauss-Legendre quadrature and a `u=100` upper truncation, followed by exact
renormalization. The quadrature order and truncation are implementation
constants, not fitted parameters. This uses the Rosseland inverse-opacity
definition and its analytic weight function; see the published derivation in
[STA mean opacities](https://www.mdpi.com/2218-2004/6/3/35) and the continuum
frequency treatment in [Rosseland and Planck mean opacities for primordial
matter](https://academic.oup.com/mnras/article/358/2/614/1001175).

The H-minus free-free coefficient, gated Kramers free-free law, Saha donor
closure, H-minus bound-free cross section, Balmer/Paschen edge cross sections,
and stimulated-emission pivot are unchanged from v2. The hydrogen Rayleigh
cross section is the fixed `5.799e-29 cm2` value at 500 nm; it is included as
a named low-order scattering floor and is not calibrated to the corpus.

## Applicability change

The formal atomic-opacity domain is changed explicitly from `T >= 2500 K` in
v2 to `T >= 3200 K` in v3. Layers below 3200 K remain in the output and are
reported separately, but do not enter the formal gate. This is a declared
domain change, not a hidden molecular-opacity correction; H2O/CO/TiO opacity is
not added in this round.

## Fixed validation and gates

1. Use the same manifest-excluded development validation split, target 10,228
   stars, split seed `20260816`, and corpus hash as v2. Do not open a sealed
   holdout.
2. Report signed residuals, absolute p50/p95/max, positive fraction, component
   dominance, and a fixed Teff-band × layer-temperature table.
3. Formal opacity gate: absolute pooled p95 `<= 0.30 dex` for Teff `< 6000 K`
   and `<= 0.50 dex` for `6000 <= Teff < 10000 K`. The 5500--7000 K transition
   band remains a separate diagnostic.
4. If a formal p95 exceeds its limit by at most `0.10 dex`, proceed only when
   the true-`P,T` integrated-mass p95 is `<= 0.20 dex`. A larger failure is
   `FAIL_STOP` before any solver run.
5. If the opacity gate/bridge passes, run the ODE mass error with true `T` and
   grey-plus-adiabatic `T`, then the fixed 12-star smoke (`>=11/12`), followed
   by the fixed 60-star funnel against grey15, parity, and production.

The unchanged real-solver contract remains: at most 15 iterations, at least
three iterations before convergence, positive finite final state, and the
existing flux/spectral checks. An offline opacity pass is not a production
claim.

## Formal result

The fixed validation run used 10,228 stars and 818,240 layers. The explicit
`T >= 3200 K` domain retained 786,132 layers and reported the remaining 32,108
layers separately. There were no non-finite opacity or integrated-mass values.

The result is `FAIL_STOP` at the offline opacity/bridge stage:

| Teff band | signed median dex | signed mean dex | absolute p95 dex | dominant component |
|---|---:|---:|---:|---|
| `Teff < 6000 K` | -0.096 | -0.111 | 0.478 | H-minus bound-free |
| `5500--7000 K` | +0.002 | -0.008 | 0.473 | gated Kramers free-free |
| `6000 <= Teff < 10000 K` | -0.070 | -0.016 | 0.682 | H Balmer bound-free |
| `Teff >= 10000 K` | -0.078 | -0.033 | 0.705 | H Balmer bound-free |

The formal limits are `0.30 dex` and `0.50 dex`. The true-`P,T` integrated-mass
p95 is `0.406 dex` in the cool band and `0.558 dex` in the middle band, so the
failures are outside the preregistered `0.10 dex` bridge allowance. The
Teff-by-layer-temperature table localizes the remaining error mainly to
`3200--4000 K` layers in the cool band and `>=15000 K` layers for stars with
`6000 <= Teff < 10000 K`; the transition-band median is near zero but its tail
does not rescue the two formal gates.

No ODE temperature ablation, solver smoke, or 60-star funnel was run after
this result. The output is diagnostic evidence only:

`results/analytic_initializer/textbook_opacity_v3_offline_validation.json`
