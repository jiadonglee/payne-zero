# Textbook opacity v4r6 preregistration

Date: 2026-08-28

v4r6 is a new development-only hydrogenic *threshold* candidate. Historical
v4 through v4r5 paths, preregistrations, and `FAIL_STOP` records remain
authoritative. v4r6 does not change the production solver, default
initializer, sealed holdout, five frequency windows, 32 nodes, `u=100`
truncation, seven-donor Saha, John H-minus, particle-count density, H2+,
He-minus, ground-anchored H I populations, He II, or the formal domain
`T >= 4000 K`. Helium is still not a Saha donor. No molecular band, line
haze, corpus fit, or runtime Karzas/Coulomb/continuum-table load is added.

## Rejected sibling: T<4000 Bell & Berrington H-minus free-free

The cool-mass decomposition licensed a T<4000 continuum
(`notes/textbook_opacity_v4r5_cool_mass_preregistration_20260828.md`,
JSON SHA-256
`6115c8c78c3ab583fac2fa47b964224f0346588713ac55a39ced54bad3c0bcf1`).
The letter-of-rule Flag-8 metal mapping is physically null (knockout
0.0008 dex). The caveat construction was Bell & Berrington (1987) H-
free-free below 4000 K, John (1988) frozen at `T >= 4000 K`.

That sibling is **not implemented**. John (1988) is already a published
fit to Bell & Berrington with stated ~1% accuracy, in the same 1400--10080 K
window as the ATLAS θ-grid (0.5--3.6). Swapping the polynomial for the
tabulated ATLAS numbers cannot close the 0.067 dex cool 3200--4000 K hole
versus production continuum, and loading `continuum_opacity_tables.npz` at
runtime is forbidden. Cool mass therefore stays a later, separate problem.

## Registered diagnosis (this candidate)

The Balmer diagnostic
(`notes/textbook_opacity_v4r5_balmer_preregistration_20260828.md`,
JSON SHA-256
`4b696cbf6435d1c4bc8292a725a28b8ea132d33e2c7e3eab0ac4742701e8e699`)
returned `INCONCLUSIVE` for a *global* n=2-only edge. On the frozen 20-star
grid, versus production continuum:

- 8000--15000 K: v4r5 **+0.209** dex
- n=2 edge → 1.40e-17: leftover **+0.076**, but `T >= 30000 K` moved
  **-0.081** (reopens the Lyman hole)
- n=3 textbook edge is `9 × 6.30e-18 = 5.67e-17`; Karzas-Latter threshold
  is **2.16e-17**
- n>=7 is negligible; n=1 does not carry 8000--15000 K

v4r5 already froze ground-anchored populations. The remaining Balmer/Paschen
overprediction is the n=2 and n=3 *threshold values*, not the 10-level
partition. Applying those published thresholds at `T >= 15000 K` is the
isolation failure. v4r6 therefore **gates the per-n law**:

```text
T <  15000 K : σ_n(threshold) = published n=1,2,3; n>=4 keep n^2 σ_1
T >= 15000 K : hydrogen bound-free identical to v4r5
```

Frequency law remains ν^{-3}. n=1 stays 6.30e-18. Populations stay
ground-anchored. The 15000 K ceiling is the Balmer diagnostic's control/hot
split, not a fitted number.

## Named constants

| n | v4/v4r5 edge | v4r6 edge | source |
|---:|---:|---:|---|
| 1 | 6.30e-18 | 6.30e-18 | Lyman / Karzas 6.31e-18 |
| 2 | 2.52e-17 | **1.40e-17** | literature Balmer (Karzas 1.39e-17) |
| 3 | 5.67e-17 | **2.16e-17** | Karzas-Latter threshold, offline npz read |
| >=4 | n^2 × 6.30e-18 | unchanged | v4 Kramers |

## Formal validation and stop rule

Same 10,228-star development split, seed `20260816`, true `(P, T)`, domain
`T >= 4000 K`. Gates unchanged:

- cool opacity p95 `<= 0.30`
- middle opacity p95 `<= 0.50`
- cool and middle true-`(P,T)` mass p95 `<= 0.20`

20-star isolation (production continuum, lines off), independent of mass:

- 8000--15000 K signed median must fall by **at least 0.10 dex** from v4r5
  `+0.209`
- `T >= 30000 K` signed median must stay within **0.03 dex** of v4r5
  `-0.004`
- layers with `T >= 15000 K` must have `kappa_v4r6 / kappa_v4r5 = 1`

If isolation fails, record `FAIL_STOP` even if a mass or opacity gate
moves. If isolation holds and any registered gate fails, `FAIL_STOP`. If
all gates pass, `QUALIFIED_BUT_STOP_AFTER_OFFLINE`. No ODE, funnel, or
sealed holdout.

Output:
`results/analytic_initializer/textbook_opacity_v4r6_offline_validation_20260828.json`

Cite, do not overwrite:

- v4r5 offline `c3ea2b6091d5fcd0f23ab20ca26a69025888cae1a0a9a02870ac12f1562e9df6`
- hot-flag ablation `c136b076d5f135733e4d7e43081d2ed8040f3586b0f4cbd01283628dda613b66`
- Balmer diagnostics `4b696cbf6435d1c4bc8292a725a28b8ea132d33e2c7e3eab0ac4742701e8e699`
- cool-mass decomposition `6115c8c78c3ab583fac2fa47b964224f0346588713ac55a39ced54bad3c0bcf1`

## Post-run (2026-08-28)

Isolation (20-star production continuum, lines off) **holds**:

- 8000--15000 K signed median: v4r5 `+0.2086` → v4r6 `+0.0200` (drop `0.1886` ≥ 0.10)
- T ≥ 30000 K: `-0.0042` unchanged (change `0.0` ≤ 0.03)
- T ≥ 15000 K layers: `kappa_v4r6 / kappa_v4r5 = 1` (`frozen_max_abs_log10_ratio = 0`)

Hot-grid JSON:
`results/analytic_initializer/textbook_opacity_v4r6_hot_grid_20260828.json`
SHA-256 `ea2b012c4623923f4c41ed0a28d72eac665fadc71b8e83cad31066f79767b1c7`

Full offline (10,228 stars, seed 20260816, T ≥ 4000 K): **FAIL_STOP**.
Isolation held; opacity gates passed; mass gates failed. No ODE, funnel,
or sealed holdout.

| Gate | v4r5 p95 | v4r6 p95 | Limit | v4r6 |
|---|---:|---:|---:|---|
| Cool κ (Teff < 6000 K) | 0.2170 | 0.2415 | 0.30 | pass |
| Middle κ (6000--10000 K) | 0.2341 | 0.2373 | 0.50 | pass |
| Cool true-(P, T) mass | 0.2375 | 0.2468 | 0.20 | fail |
| Middle true-(P, T) mass | 0.2157 | 0.2074 | 0.20 | fail |

Offline JSON:
`results/analytic_initializer/textbook_opacity_v4r6_offline_validation_20260828.json`
SHA-256 `ad58d6da5ec046401b55655ca60b96cb6e123f12111e330417228ffdeda4909b`

Against stored total κ_R (lines included), 8000--15000 K signed median
flipped from v4r5 `+0.099` to v4r6 `−0.090`. Frozen T ≥ 15000 K slices
are bit-identical to v4r5. Cool-star signed median became more negative
(`−0.074` → `−0.109`); that is why cool κ p95 rose while still passing.

The remaining blocker is still the mass integral through T < 4000 K
surface columns, not the Balmer threshold. Bell & Berrington H− ff stays
rejected.
