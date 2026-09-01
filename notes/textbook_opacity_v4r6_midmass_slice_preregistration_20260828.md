# Textbook opacity v4r6 middle-mass layer-temperature slice

Date: 2026-08-28

This is a development-only diagnostic. It does not change v4 through v4r6, the
production solver, the default initializer, any sealed holdout, the cool/middle
p95 gates, or the hydrostatic integral. Column mass is still integrated from
the surface with `m0 = tau0 / kappa0`. No new opacity version is constructed.
No ODE, funnel, or sealed holdout is run. Production opacity is not evaluated
(the 20-star production-continuum comparisons cited below are read from prior
frozen JSON artifacts).

Historical v4r6 offline `FAIL_STOP` remains authoritative. Both opacity gates
passed; both true-`(P, T)` mass gates failed. The v4r5 cool-mass decomposition
(verdict `MIXED`, 92% of the cool excess from the `T < 4000 K` surface column)
reported its middle-band control only as three aggregates — surface p95
0.2157, hybrid p95 0.2144, wholly-in-domain increment p95 0.2293 — never
sliced by layer temperature. This note diagnoses the middle band.

## Question

The middle mass gate scores per-layer column-mass residual on
`6000 <= Teff < 10000 K` (5,685 stars) and layer `T >= 4000 K`, and v4r6 fails
it at p95 `0.2074` against limit `0.20`. Which layer-temperature range carries
the failure?

Competing hypotheses:

1. `LINE_FLOOR_8_15KK` — carried by layers at `8000--15000 K`, where on the
   frozen 20-star grid v4r6 underpredicts *stored total* `kappa_R` by
   `-0.090` dex signed median (missing line blanketing that a continuum
   candidate cannot express; vs production *continuum* the same slice is
   `+0.020`, fixed). If this carries the gate, no continuum edit can close
   the middle mass gate.
2. `DEEP_HOT_15_30KK` — carried by deep hot layers at `15000--30000 K`, where
   v4r6 is `+0.12` dex vs production continuum (`+0.123` at `15000--22000 K`,
   `+0.122` at `22000--30000 K`; over-opaque), i.e. a fixable continuum issue.
3. `SURFACE_COLUMN` — like the cool stars, the surface-started integral
   through `T < 4000 K` layers carries it. The v4r5 decomposition's middle
   control says hybrid does NOT move the middle gate, so this is expected to
   be falsified; verify anyway at the gate level (hybrid p95) and note the
   increment level (wholly in-domain increments exclude the surface column by
   construction, so they cannot carry this hypothesis).
4. `MIXED` / `INCONCLUSIVE` with quantitative splits.

## Prior artifacts (cited, not overwritten)

| artifact | SHA-256 |
|---|---|
| `results/analytic_initializer/textbook_opacity_v4r6_offline_validation_20260828.json` | `ad58d6da5ec046401b55655ca60b96cb6e123f12111e330417228ffdeda4909b` |
| `results/analytic_initializer/textbook_opacity_v4r5_cool_mass_decomposition_20260828.json` | `6115c8c78c3ab583fac2fa47b964224f0346588713ac55a39ced54bad3c0bcf1` |

v4r6 offline (10,228 stars, split seed `20260816`, domain `T >= 4000 K`):

- cool `kappa` p95 0.241 (limit 0.30) pass; middle `kappa` p95 0.237 (limit
  0.50) pass
- cool true-`(P, T)` mass p95 **0.2468** (limit 0.20) fail
- middle true-`(P, T)` mass p95 **0.20741398414881243** (limit 0.20) fail

## Fixed diagnostic (not a new opacity law)

Evaluate frozen **v4r6** (`textbook_rosseland_opacity_v4r6` /
`textbook_opacity_node_components_v4r6`) on stored true `(P, T)`. Stored
`n_e` is not an input. The registered sample is the same manifest-excluded
development validation split as v4r6: 10,228 stars, seed `20260816`, via
`collect_excluded_indices(MANIFESTS)` + `make_split(..., seed=20260816)`,
reusing the exact machinery of the v4r5 decomposition and v4r6 offline
runners. `--limit` is allowed for a smoke pass; the registered JSON is the
full split.

Reuse `integrate_mass_from_opacity` from
`experiments/analytic_initializer/profile_closure.py` for every
surface-started column (`m0 = tau0 / kappa0`, trapezoid `dm = dtau/kappa`).
The middle-gate mask is unchanged:

```text
middle_mask = (6000 <= Teff < 10000 K) and (T >= 4000 K)
```

scored on per-layer `log10(m_pred) - log10(m_stored)`.

### A. Reproduce the failure and the increment control

1. Surface-started v4r6 column-mass residual on the middle gate must
   reproduce the v4r6 offline value `0.20741398414881243` dex within
   `0.002` dex (full split only), else `INCONCLUSIVE`.
2. Wholly-in-domain increment residuals
   `increment[i] = log10(dm_pred[i]) - log10(dm_stored[i])` with
   `dm[i] = m[i] - m[i-1]`, restricted to middle-band stars and increments
   with both `T[i] >= 4000 K` and `T[i-1] >= 4000 K`, must reproduce the
   v4r5 decomposition's middle control `0.22927611663282257` dex within
   `0.01` dex (recomputed with v4r6, whose mid-layer opacity differs slightly
   from v4r5; full split only), else `INCONCLUSIVE`.
3. Hybrid check (gate level): replace predicted `kappa` with stored total
   `kappa_R` on `T < 4000 K` layers, keep v4r6 on `T >= 4000 K`, integrate
   from the surface, score on the middle gate. Report p95.

### B. Layer-temperature slice of wholly-in-domain increments

On middle-band stars, bin wholly-in-domain increments by the **upper-endpoint
layer temperature** `T[i]` (registered choice; both endpoints are
`>= 4000 K` by construction):

```text
[4000,6000) [6000,8000) [8000,10000) [10000,15000) [15000,22000)
[22000,30000) [30000,inf)
```

Per bin: count, signed median, p95 of `|residual|`, positive fraction. Also
report, for connection to the gate, the per-layer *cumulative* column-mass
residual restricted to middle-band layers whose temperature falls in each bin
(same statistics). The cumulative per-bin numbers are context only; the
verdict uses the increment bins.

### C. Sanity

Integrate stored total `kappa_R` with the same surface-started rule and score
the middle gate. v4r3 recorded p95 `0.006` dex on the full domain. If the
middle-mask p95 exceeds `0.05` dex, the trapezoid or the `m0` rule is the
problem, not v4r6, and the verdict is `INCONCLUSIVE`.

If the middle-gate surface p95 is already `<= 0.20`, there is no failure to
explain: `INCONCLUSIVE`.

## Registered verdict

Let `LIMIT = 0.20`. After the reproduce/sanity checks above:

1. `SURFACE_COLUMN` if hybrid-`kappa` middle-gate mass p95 `<= LIMIT` and the
   wholly-in-domain increment p95 `<= LIMIT` (surface-started gate fails while
   both in-domain diagnostics pass).
2. Otherwise, on the wholly-in-domain increment bins, define per bin `b`:
   `excess_b = max(p95_b - LIMIT, 0)`, `weight_b = count_b / count_total`,
   `contribution_b = excess_b * weight_b`, and
   `share_b = contribution_b / sum(contribution)`. Consider every contiguous
   interval of bins (adjacent in the registered order); let `S*` be the
   interval with the maximum total share. If that maximum share is
   `>= 0.70`:
   - `S*` contained in `{[8000,10000), [10000,15000)}` -> `LINE_FLOOR_8_15KK`
   - `S*` contained in `{[15000,22000), [22000,30000)}` -> `DEEP_HOT_15_30KK`
   - any other `S*` (e.g. spans both ranges, includes `[30000, inf)`, or the
     cool bins) -> `MIXED`
3. Otherwise `MIXED`.
4. `INCONCLUSIVE` if any reproduce/sanity check fails, a primary statistic is
   non-finite, or the total increment count is zero. In smoke mode
   (`--limit`), the reproduce checks are skipped and `full_registered_split`
   is false; the verdict is still computed for inspection but is not
   registered.

Do not relax `LIMIT`. Do not move the integral start. Do not drop
`T < 4000 K` layers from the mass rule. Do not edit
`experiments/analytic_initializer/textbook_opacity.py`, any gate, or any
existing file besides this note, the runner, and the registered JSON.

## Stop rule and output

Record the machine verdict, the per-bin table, the reproduce/sanity numbers,
and one sentence on what closes (or cannot close) the `0.0074` dex
middle-mass gap. Stop. Do not implement any opacity edit.

Registered output:
`results/analytic_initializer/textbook_opacity_v4r6_midmass_slice_20260828.json`

Runner:
`experiments/analytic_initializer/run_textbook_opacity_v4r6_midmass_slice.py`

Log: `logs/textbook_opacity_v4r6_midmass_slice.log`

## Post-run formal result

The diagnostic completed on 2026-08-28 locally on macOS
(`PYTHONPATH=. .venv/bin/python`). Local pytest: `tests/test_textbook_opacity.py`
37 passed. Full split: 10,228 stars, seed `20260816`; middle band 5,685 stars,
454,253 gate layers, 448,568 wholly-in-domain increments. Non-finite count 0.

JSON: `results/analytic_initializer/textbook_opacity_v4r6_midmass_slice_20260828.json`
SHA-256: `07d4dacfccffd60d794eecfab9e784804a0aae7f87795e1c4ebc1aa345816624`
Log: `logs/textbook_opacity_v4r6_midmass_slice.log`

Reproduce and sanity checks (all pass):

| check | observed | target | tolerance | pass |
|---|---:|---:|---:|---|
| surface-started middle mass p95 | 0.2074139841488115 | 0.20741398414881243 | 0.002 | yes |
| wholly-in-domain increment p95 | 0.2328289827180433 | 0.22927611663282257 | 0.01 | yes |
| stored-`kappa_R` integral sanity p95 | 0.005935131334758002 | `<= 0.05` | — | yes |

Gate-level diagnostics: surface-started v4r6 p95 **0.2074** (fail, as
required); hybrid (stored `kappa_R` below 4000 K) p95 **0.2052** — still over
the 0.20 limit, so `SURFACE_COLUMN` is falsified at the gate level, matching
the v4r5 decomposition's middle control. Wholly-in-domain increment p95
**0.2328** — the in-domain increments themselves are over limit, so the gate
failure is not purely cumulative-from-the-surface either.

Per-bin wholly-in-domain increment residuals (middle band, binned by
upper-endpoint layer temperature):

| bin | count | signed median | p95 \|res\| | positive fraction | excess share |
|---|---:|---:|---:|---:|---:|
| [4000,6000) | 143,335 | +0.0616 | 0.2255 | 0.992 | 0.277 |
| [6000,8000) | 141,662 | −0.0003 | 0.1035 | 0.496 | 0.000 |
| [8000,10000) | 42,859 | +0.0349 | 0.1617 | 0.700 | 0.000 |
| [10000,15000) | 49,173 | +0.1329 | 0.2620 | 0.980 | 0.231 |
| [15000,22000) | 24,279 | +0.0127 | 0.1651 | 0.590 | 0.000 |
| [22000,30000) | 18,910 | +0.0453 | 0.2026 | 0.852 | 0.004 |
| [30000,inf) | 28,350 | +0.1715 | 0.4263 | 0.999 | 0.488 |

Per-layer cumulative mass residual by bin (context): 4000–6000 K p95 0.2371,
6000–8000 K 0.1064, 8000–10000 K 0.1575, 10000–15000 K 0.2010, 15000–22000 K
0.1538, 22000–30000 K 0.1836, ≥30000 K 0.2730.

The machine verdict is **`MIXED`**.

No hypothesis-named adjacent bin set carries ≥ 70% of the over-limit excess:
the two `LINE_FLOOR_8_15KK` bins sum to share 0.231 (and the 8000–10000 K bin
is itself under limit at p95 0.162), the two `DEEP_HOT_15_30KK` bins sum to
0.004 (both essentially at or under limit), and the excess is spread across
4000–6000 K (0.277), 10000–15000 K (0.231), and ≥30000 K (0.488) — the last
being outside every registered hypothesis. The registered maximum-share
interval is the full bin range (share 1.0), which maps to `MIXED`; the
smallest ≥ 0.70 interval, [10000_15000K, at_least_30000K] at share 0.723,
also spans hypothesis boundaries and likewise maps to `MIXED`. The verdict is
therefore robust to that reading of the registered rule.

Implication for the 0.0074 dex middle-mass gap: it is not a localizable
opacity hole — not the 8–15 kK line floor (those bins are at or under limit)
and not the 15–30 kK over-opaque continuum (share 0.004) — but a broad,
mostly positive low-level increment bias peaking in the ≥30000 K deep layers
and the 4000–6000 K upper domain, so no single continuum edit in any one
temperature range is expected to close the middle mass gate.

No v4r7 code was implemented. The mass integral still starts at the surface.
Gates were not relaxed. Production opacity, ODE, funnel, and sealed holdout
stayed closed.
