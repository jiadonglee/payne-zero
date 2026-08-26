# Textbook opacity v2 preregistration

Date: 2026-08-26

This is a new development-only candidate. It does not change the production
solver, the default initializer, or any sealed holdout.

## Fixed construction

The local state is the Saha-aware `(T, P)` state with `P = g m` and a positive
log-coordinate hydrostatic ODE. The v2 ablation fixes the following constants
before reading its validation result:

1. H-minus free-free uses the standard low-order Rosseland estimate
   `2.5e-31 (Z/0.02) rho^0.5 T^9` in cgs units. It has full strength through
   6000 K and a fixed linear taper to zero at 7000 K, which is the declared
   validity transition of this cool-atmosphere approximation.
2. The old ungated Kramers bound-free/free-free term is removed. A separate
   Kramers free-free fallback remains only as
   `3.68e22 (1+X) rho T^-3.5 x_H+`, where `x_H+` is the Saha hydrogen-ionized
   fraction. It has no metallicity fit term.
3. H-minus bound-free stays at the v1 fixed `1e-17 cm2` cross-section so this
   ablation isolates the requested H-minus free-free change.
4. Neutral-H Balmer bound-free uses the n=2 edge value `1.4e-17 cm2`, the
   existing Paschen value `1.2e-18 cm2`, and the fixed representative photon
   energy `h nu / kT = 3.8` in the stimulated-emission factor.
5. Saha donors remain H, Na, K, Ca, Mg, and Fe with the existing solar number
   anchors; no corpus-derived coefficients are fitted.

Layers with local `T < 2500 K` are outside the declared atomic-opacity domain.
They remain in the output and are reported separately, but do not enter the
formal cool/middle opacity gate. This is an applicability declaration, not a
molecular-opacity correction.

The Gray-style H-minus scaling is consistent with the standard cool-star
Rosseland estimate, while detailed H-minus frequency-dependent treatments
remain outside this compact seed. See the cited standard discussion:

- https://www.astro.princeton.edu/~burrows/classes/514/514.2025.pdf
- https://www.aanda.org/articles/aa/pdf/2017/09/aa30856-17.pdf

## Fixed acceptance sequence

1. Run the excluded-manifest development validation set (target 10,228
   stars), writing signed residual summaries, Teff bands, low-temperature
   counts, and component dominance.
2. Formal opacity gate: absolute pooled p95 `<= 0.30 dex` for Teff `< 6000 K`
   and `<= 0.50 dex` for `6000 <= Teff < 10000 K`. The transition band
   `5500--7000 K` is reported separately and does not replace either gate.
3. If a formal p95 exceeds its limit by at most 0.10 dex, the candidate may
   proceed only when the true-`P,T` integrated-mass p95 is `<= 0.20 dex`.
   Larger failures stop the branch before any solver run.
4. Only after that bridge check, run ODE mass errors for truth `T` and for
   grey-plus-adiabatic `T`; then run the fixed 12-star smoke (`>=11/12`) and
   fixed 60-star funnel. The controls are grey15, parity, and production.

The output is diagnostic evidence only until the unchanged real-solver
15-iteration, positivity, flux-balance, and spectral checks all pass.

## Formal result

The 10,228-star validation was run with the command:

```text
PYTHONPATH=. python experiments/analytic_initializer/run_textbook_opacity_offline.py \
  --out results/analytic_initializer/textbook_opacity_v2_offline_validation.json
```

The result is `FAIL_STOP` for the opacity bridge:

| band | signed median dex | signed mean dex | absolute p95 dex | dominant component |
|---|---:|---:|---:|---|
| `Teff < 6000 K` | -0.016 | -0.020 | 0.581 | H-minus bound-free |
| `5500--7000 K` | +0.177 | +0.212 | 0.545 | gated Kramers free-free |
| `6000 <= Teff < 10000 K` | +0.585 | +0.521 | 0.969 | H Balmer/Paschen bound-free |
| `Teff >= 10000 K` | +0.707 | +0.578 | 0.990 | H Balmer/Paschen bound-free |

The formal limits are `0.30 dex` and `0.50 dex`. The true-`P,T` mass p95 is
`0.609 dex` in the cool band and `0.921 dex` in the middle band, so neither
formal failure is within the preregistered 0.10-dex bridge allowance. No ODE
temperature ablation, solver smoke, funnel, or oracle expansion was run after
this result. The prior v1 result remains a separate failure record.
