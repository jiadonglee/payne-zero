# Textbook opacity v4r5 cool-mass decomposition

Date: 2026-08-28

This is a development-only diagnostic. It does not change v4 through v4r5, the
production solver, the default initializer, any sealed holdout, the cool/middle
p95 gates, or the hydrostatic integral. Column mass is still integrated from
the surface with `m0 = tau0 / kappa0`. No new opacity version is constructed.
No ODE, funnel, or sealed holdout is run.

Historical v4r5 offline `FAIL_STOP` remains authoritative. Cool and middle
opacity gates passed; cool true-`(P, T)` mass did not. v4r3 already had the
same cool-mass p95; H2+ / He-minus and the v4r5 ground-anchored H I bound-free
did not move it.

## Question

The cool mass gate scores per-layer column-mass residual on
`Teff < 6000 K` and `T >= 4000 K`. The integral that produces that column
still starts at the surface. The claim to test, not assume, is:

**the cool-gate mass fail is the T<4000 contribution to a surface-started
column, not an in-domain cool-photosphere opacity miss.**

Competing, distinguishable answers:

1. `SURFACE_INTEGRAL_DOMINATED` — in-domain cool `kappa` already passes; the
   0.20 dex mass miss is mostly outer-layer `1/kappa` (including `m0`).
2. `IN_DOMAIN_COOL_OPACITY_DOMINATED` — after removing the outer-layer
   contribution, `T >= 4000 K` cool mass still fails. Name the continuum
   component on `4000--5000 K` layers.
3. `MIXED` — both, with a quantitative split of the p95 excess.
4. `INCONCLUSIVE` — the gate is not reproduced, the stored-`kappa_R` integral
   sanity fails, or a primary slice is empty.

This task does not work the middle-band `8000--15000 K` Balmer / He II
problem. Middle-band hybrid/oracle numbers are a control only: they are
reported if the same T<4000 surface mechanism is present, and are otherwise
ignored.

## Prior artifacts (cited, not overwritten)

| artifact | SHA-256 |
|---|---|
| `results/analytic_initializer/textbook_opacity_v4r5_offline_validation_20260828.json` | `c3ea2b6091d5fcd0f23ab20ca26a69025888cae1a0a9a02870ac12f1562e9df6` |
| `results/analytic_initializer/textbook_opacity_v4r1_molecule_ablation_20260827.json` | `881c2cf9df7139c0bc190980a8c7e9ff89452ee63f63fae6964fc15adee73787` |
| `results/analytic_initializer/textbook_opacity_v4r4_hot_flag_ablation_20260828.json` | `c136b076d5f135733e4d7e43081d2ed8040f3586b0f4cbd01283628dda613b66` |

v4r5 offline (10,228 stars, split seed `20260816`, domain `T >= 4000 K`):

- cool `kappa` p95 0.217 (limit 0.30) pass
- middle `kappa` p95 0.234 (limit 0.50) pass
- cool true-`(P, T)` mass p95 **0.2375** (limit 0.20) fail
- cool in-domain opacity signed median `-0.074` dex
- `3200--4000 K` mass signed median `+0.252` dex, p95 0.499, positive fraction 1.0
- `T < 3200 K` mass signed median `+0.596` dex, p95 1.393, positive fraction 1.0

Molecule-ablation verdict remains `ATOMIC_IR_REMAINS`.

## Fixed diagnostic (not a new opacity law)

Evaluate frozen **v4r5** (`textbook_rosseland_opacity_v4r5` /
`textbook_opacity_node_components_v4r5`) on stored true `(P, T)`. Stored
`n_e` is not an input. The registered sample is the same manifest-excluded
development validation split as v4r5: 10,228 stars, seed `20260816`.
`--limit` is allowed for a smoke pass; the registered JSON is the full split.

Reuse `integrate_mass_from_opacity` in
`experiments/analytic_initializer/profile_closure.py` for every surface-started
column. The cool-gate mask is unchanged:

```text
cool_mask = (Teff < 6000 K) and (T >= 4000 K)
```

scored on per-layer `log10(m_pred) - log10(m_stored)`.

### A. Outer-layer contribution

Three complementary splits, all diagnostics, none of them a gate change:

1. **Hybrid `kappa`.** Replace predicted `kappa` with stored total `kappa_R`
   on `T < 4000 K` layers; keep v4r5 on `T >= 4000 K`; integrate from the
   surface. This is the counterfactual "correct outer opacity, same integral."
2. **Oracle boundary.** Let `k` be the first layer with `T >= 4000 K`. Set
   `m_cf[i] = m_stored[k] + (m_pred[i] - m_pred[k])` for `i >= k`. Linear
   mass is additive, so the outer offset cancels and the remainder is the
   predicted in-domain integral on top of the stored column at the domain
   edge.
3. **Restart `tau/kappa`.** Re-integrate from layer `k` with
   `m[k] = tau[k] / kappa[k]`. Report restart-versus-stored (the construction
   asked for) and restart-versus-truth-restart (same boundary form on both
   sides). Restart-versus-stored is *not* a fair residual against the stored
   column, because stored `m[k]` is not `tau[k]/kappa[k]`.

The registered excess split is

```text
explained_excess_dex = p95_surface - p95_hybrid
explained_fraction   = explained_excess_dex / (p95_surface - 0.20)
```

on the cool-gate mask.

### B. In-domain local `kappa` versus cumulative column

On cool stars, the local increment residual is

```text
dm[i] = m[i] - m[i-1]
increment_residual[i] = log10(dm_pred[i]) - log10(dm_stored[i])
```

for `i >= 1`. **Wholly in-domain** increments are those with both `T[i]` and
`T[i-1] >= 4000 K`. Crossing increments (`T[i-1] < 4000 <= T[i]`) are reported
separately and do not decide the in-domain bit.

If wholly in-domain increment p95 `<= 0.20` while the surface-started column
p95 is `0.2375`, the gate is cumulative, not local.

### C. Component identity at `3200--5000 K`

Report v4r5 Rosseland log-sensitivity (existing node components; molecule
bands remain off) and inverse-window fraction, on cool stars, in

- `3200 <= T < 4000 K`
- `4000 <= T < 5000 K`
- `T >= 4000 K` (the gate domain)

This identifies what the *candidate already carries*. A missing term has
near-zero self-sensitivity; identity of a missing continuum uses D.

### D. Stored total versus production continuum (20-star grid)

The frozen 20-star ablation grid
(`textbook_opacity_v4r4_hot_flag_ablation_20260828.json`) stores
`production_continuum_baseline`, `temperature_K`, and `corpus_index`. Eight of
those stars have `Teff < 6000 K`. Do **not** call production opacity in this
task. Evaluate v4r5 on those rows' stored `(P, T)` and compare, on
`3200 <= T < 4000 K` cool-star layers:

```text
v4r5_minus_stored      = log10(kappa_v4r5 / kappa_stored_total)
v4r5_minus_production  = log10(kappa_v4r5 / kappa_production_continuum)
production_minus_stored = log10(kappa_production_continuum / kappa_stored_total)
flag_X_effect          = already stored in the ablation JSON
```

Flag identity on that slice uses the existing knockouts 0, 4, 5, 8, 9, 10.

## Sanity that must hold before a scientific verdict

Integrate stored total `kappa_R` from the surface and score the same cool
mask. v4r3 recorded that this recovers column mass to p95 `0.006` dex on the
full domain. If this cool-mask p95 exceeds `0.05` dex, the trapezoid or the
`m0` rule is the problem, not v4r5, and the verdict is `INCONCLUSIVE`.

The surface-started v4r5 cool-mass p95 must reproduce `0.2375` dex within
`0.002` dex on the full 10,228-star split. Failure to reproduce is
`INCONCLUSIVE`.

If the cool-gate surface p95 is already `<= 0.20`, the diagnostic has no fail
to explain: `INCONCLUSIVE`.

## Registered verdict

Let `LIMIT = 0.20`. On the cool-gate mask:

- `hybrid_pass` if hybrid-`kappa` mass p95 `<= LIMIT`
- `oracle_pass` if oracle-boundary mass p95 `<= LIMIT`
- `increment_pass` if wholly in-domain increment p95 `<= LIMIT`

Then, after the sanity checks:

- `SURFACE_INTEGRAL_DOMINATED` if all three pass
- `IN_DOMAIN_COOL_OPACITY_DOMINATED` if all three fail
- `MIXED` if the three bits disagree
- `INCONCLUSIVE` otherwise (sanity / reproduce / empty)

Do not relax `LIMIT`. Do not move the integral start. Do not drop `T < 4000 K`
layers from the production mass rule.

## v4r6 continuum license (note only; not implemented)

A later T<4000 continuum candidate is licensed only if **all** of:

1. verdict is `SURFACE_INTEGRAL_DOMINATED` or `MIXED`;
2. `explained_fraction >= 0.50`;
3. on the 20-star cool `3200--4000 K` slice, v4r5 minus production continuum
   signed median `<= -0.05` dex (a real continuum miss, not lines).

If `|v4r5_minus_production| < 0.05` and v4r5 minus stored total signed median
`<= -0.05`, the outer miss is **lines** in stored `kappa_R`. That does **not**
license a continuum candidate (and this program forbids a line haze / corpus
fit).

Named construction, chosen by the dominant production flag on that same
slice, not by v4r5 self-sensitivity:

| dominant flag | named construction (literature, not a fit; T<4000 only) |
|---|---|
| Flag 0 `H_bf_ff` | Bell & Berrington (1987) H-minus free-free in the below-H-minus-threshold window. John (1988) stays frozen at `T >= 4000 K`. Molecular bands stay off. |
| Flag 8 `C_Mg_Al_Si_Fe_plus_CIA` | Published metal bf/ff (C, Mg, Al, Si, Fe) only. Not CIA, not H2O/CO/TiO. |
| Flag 9 `lukewarm_metals` | ATLAS lukewarm-metal continuum, T<4000 only. |
| Flag 4 / 5 / 10 | not licensed for cool T<4000 |

H2+ and He-minus are already in v4r3/v4r5 and are not a new license. Karzas
or Coulomb tables are not loaded at runtime. John H-minus coefficients are
not refit.

## Stop rule and output

Record the machine verdict, the four to six deciding numbers, and whether
v4r6 is licensed. Stop. Do not implement the named construction.

Registered output:
`results/analytic_initializer/textbook_opacity_v4r5_cool_mass_decomposition_20260828.json`

Runner:
`experiments/analytic_initializer/run_textbook_opacity_v4r5_cool_mass_decomposition.py`

Log: `logs/textbook_opacity_v4r5_cool_mass_decomposition.log`

## Remote execution

Host `astronode-garching`. Checkout
`/home/jdli/xiasangju/jdli/payne-zero`. Python `.venv-linux/bin/python`.
Do not evaluate production opacity from macOS `.venv`.

## Post-run formal result

The diagnostic completed on 2026-08-28 on `astronode-garching`. Remote
pytest: 10 passed. Full split: 10,228 stars, seed `20260816`. Non-finite
count 0. Non-monotonic cool stars 0. Stored-`kappa_R` cool-mask mass p95
`0.00626` dex (sanity `<= 0.05`). Surface-started v4r5 cool-mass p95
`0.23750253783251785` dex, identical to the cited v4r5 offline JSON.

JSON: `results/analytic_initializer/textbook_opacity_v4r5_cool_mass_decomposition_20260828.json`
SHA-256: `6115c8c78c3ab583fac2fa47b964224f0346588713ac55a39ced54bad3c0bcf1`
Log: `logs/textbook_opacity_v4r5_cool_mass_decomposition.log`

Cool stars: 3,550 / 10,228. Of those, 3,383 have at least one `T < 4000 K`
layer. Median first in-domain layer index is 36 / 80.

| diagnostic (cool gate unless noted) | p95 dex | vs 0.20 |
|---|---:|---|
| surface-started v4r5 mass | 0.2375 | fail |
| hybrid: stored `kappa` on `T < 4000 K` | 0.2028 | fail |
| oracle boundary at first `T >= 4000 K` | 0.1993 | pass |
| wholly in-domain increment | 0.2189 | fail |
| stored-`kappa` integral sanity | 0.0063 | pass |

`explained_excess = 0.0347` dex. `explained_fraction = 0.924`.
`hybrid_pass=false`, `oracle_pass=true`, `increment_pass=false`.

The machine verdict is **`MIXED`**.

Outer layers still dominate the *excess*: swapping stored total `kappa_R`
into `T < 4000 K` removes 92% of the 0.0375 dex over-limit. The remaining
in-domain increment p95 is 0.219, and hybrid remains 0.003 dex over the
gate, so the three registered bits do not all pass. Cool in-domain opacity
versus stored total remains a pass-level hole (signed median `-0.074`,
p95 `0.217`). `T < 4000 K` opacity versus stored total is larger (signed
median `-0.194` at `3200--4000 K`, `-0.488` below 3200 K).

Middle-band control: surface p95 0.216, hybrid 0.214, increment 0.229.
Hybrid does not move the middle mass gate. That miss is not this
T<4000 surface mechanism and was not diagnosed further.

20-star cool `3200--4000 K` (8 stars, 199 layers), no production call:

| residual | signed median dex |
|---|---:|
| v4r5 minus production continuum | **-0.0668** |
| v4r5 minus stored total | -0.196 |
| production continuum minus stored total | -0.081 |

All six stored flag knockouts on that slice have signed median
`< 0.001` dex. The algebraic maximum is Flag 8
(`C_Mg_Al_Si_Fe_plus_CIA`) at `+0.00083` dex. Flag 0 (H bf/ff) is
`4e-7` dex. IFOP(2) H2+ and IFOP(7) He-minus were not in this knockout
set.

v4r5 self-sensitivity on cool `3200--4000 K`: H-minus free-free median
0.495, H-minus bound-free 0.401. Inverse-window fraction in
`below_hminus_threshold` is 0.466. On cool `4000--5000 K` the in-domain
carrier is H-minus bound-free (0.640).

### v4r6 license (machine, then a physical caveat)

The pre-registered license bits all fire: verdict `MIXED`, explained
fraction `0.924 >= 0.50`, v4r5 minus production signed median
`-0.0668 <= -0.05`. The machine therefore records
`licensed=true` with the Flag-8 mapping:

> Published metal bound-free/free-free (C, Mg, Al, Si, Fe) for T<4000 K
> only; not CIA and not H2O/CO/TiO bands; not a corpus fit.

That mapping is **letter-of-rule, not a demonstrated production
identity**. Flag 8 cannot carry a 0.067 dex continuum miss with a 0.0008
dex knockout. The available ablation flags are a null set at
`3200--4000 K`. A later candidate that actually sits in the IR window
the candidate already uses is Bell & Berrington (1987) H-minus free-free
for `T < 4000 K` only, with John (1988) frozen at `T >= 4000 K`. That
H-minus construction was not the machine-selected name because the
pre-registered D-arm names by dominant *production flag*, and every
listed flag is null.

No v4r6 code was implemented. The mass integral still starts at the
surface. Gates were not relaxed. Production, ODE, funnel, and sealed
holdout stayed closed.
