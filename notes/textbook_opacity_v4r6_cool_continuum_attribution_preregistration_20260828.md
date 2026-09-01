# Textbook opacity v4r6 cool 3200–4000 K continuum attribution (B2)

Date: 2026-08-28

This is a development-only diagnostic. It does not change v4 through v4r6,
the production solver, the default initializer, any sealed holdout, any gate,
the hydrostatic integral, or `textbook_opacity.py`. No new opacity version is
constructed. No ODE, funnel, or sealed holdout is run. Production opacity is
evaluated only on the remote Linux host (`astronode-garching`), never from
the macOS `.venv`.

## Question

Which production continuum component carries the **−0.0668 dex** signed-median
gap (frozen candidate v4r5 versus production continuum, lines off) on the
20-star cool `3200 <= T < 4000 K` slice?

Competing, distinguishable answers:

1. `H2PLUS_IMPLEMENTATION` — knocking out IFOP(2) in production collapses the
   gap (production H2+ ≫ candidate H2+ on this slice).
2. `HEMINUS_IMPLEMENTATION` — same for IFOP(7).
3. `BASE_CONTINUUM` — all single-flag knockouts stay null; the gap is in the
   always-on base (production H-minus free-free/bound-free coefficient
   details, electron scattering, Rayleigh scattering, or the
   stimulated-emission treatment). The specific base component is named only
   if the registered bookkeeping isolates one.
4. `UNRESOLVED` — none of the above separates it.

## Prior artifacts (cited, not overwritten)

| artifact | SHA-256 |
|---|---|
| `results/analytic_initializer/textbook_opacity_v4r4_hot_flag_ablation_20260828.json` | `c136b076d5f135733e4d7e43081d2ed8040f3586b0f4cbd01283628dda613b66` |
| `results/analytic_initializer/textbook_opacity_v4r5_cool_mass_decomposition_20260828.json` | `6115c8c78c3ab583fac2fa47b964224f0346588713ac55a39ced54bad3c0bcf1` |

Established facts this diagnostic starts from:

- On the slice (8 cool stars, 199 layers): v4r5 − production continuum
  (lines off) = **−0.0668 dex** signed median; production continuum − stored
  total = −0.081 dex; v4r5 − stored total = −0.196 dex.
- All six stored flag knockouts (0 `H_bf_ff`, 4 `He_I`, 5 `He_II`,
  8 `C_Mg_Al_Si_Fe_plus_CIA`, 9 `lukewarm_metals`, 10 `hot_metals`) are null
  on the slice (signed median < 0.001 dex; algebraic max Flag 8 at
  +0.00083 dex; Flag 0 at 4e-7 dex).
- IFOP(2) = H2+ and IFOP(7) = He-minus were **never knocked out**. The
  candidate carries its own H2+/He- implementations since v4r3, so an
  implementation mismatch there is invisible to the stored knockout set.
- Candidate v4r5 Rosseland log-sensitivity on cool `3200–4000 K`: H-minus
  free-free 0.495, H-minus bound-free 0.401 (combined ≈ 0.90); H2+ median
  sensitivity 0.042 on cool `Teff < 6000 K` (slice-specific value computed
  here).

## Flag-index convention (registered)

Production `OPACITY IFOP` is a 20-element vector; the replay override key is
the 0-based Python index. Verified against
`payne_zero_atmosphere/continuum_opacity.py`: index 0 = IFOP(1) H I bf/ff,
index 1 = IFOP(2) H2+, index 2 = IFOP(3) H-minus, index 3 = IFOP(4) H
Rayleigh, index 4 = IFOP(5) He I, index 5 = IFOP(6) He II, index 6 = IFOP(7)
He-minus, index 7 = IFOP(8) He Rayleigh, index 11 = IFOP(12) electron
scattering. The two new knockouts are therefore `opacity_flag_overrides={1: 0}`
(IFOP(2)) and `{6: 0}` (IFOP(7)). The stored ablation baseline flags are
`[1]*13 + [0, 0, 0, 0, 0, 0, 0]` with IFOP(15) = IFOP(17) = 0 (lines off).

## Method (registered before running)

Reuse the frozen 20-star ablation grid and its stored rows
(`corpus_index`, `labels`, `temperature_K`, `production_continuum_baseline`,
`production_continuum_flag_off`, `flag_effect_dex`).

1. **Slice of record.** Cool stars (`Teff < 6000 K`), layers
   `3200 <= T < 4000 K`. The stored JSON has 8 cool stars and 199 such
   layers. Reproduce that count first; if it differs, stop and report
   `INCONCLUSIVE` with the counts. Recomputing `_reference_indices` from the
   corpus must return the stored 20 `corpus_index` values in order, and
   corpus temperatures must match stored `temperature_K` to 1e-6 K; else
   `INCONCLUSIVE` (corpus drift).
2. **Baseline reproduction.** Replay the all-flags-on production continuum
   (stride-16 grid, lines off via IFOP(15) = IFOP(17) = 0, molecules off, no
   temperature iteration — identical settings to the stored ablation) on all
   20 rows. On the slice, median |log10(replayed/stored)| must be
   `< 1e-6 dex` before any verdict; else `INCONCLUSIVE`.
3. **Two new knockouts.** Replay the same rows with IFOP(2) off and with
   IFOP(7) off. Per-layer effect = `log10(kappa_all_on / kappa_knockout)`.
   Report signed median and p95 of |effect| on the slice. Replay contract
   assertions are the same as the stored ablation (line flags off, zero
   positive line cells, molecules off, knockout flag took).
4. **Candidate-side magnitudes.** Evaluate `textbook_opacity_node_components_v4r5`
   on the same rows' stored `(P, T)` (v4r6 as a control; its continua differ
   from v4r5 only in H I bf thresholds). Report on the slice, for `h2plus`
   and `heminus` (and all other components for context):
   - Rosseland log-sensitivity (existing `_rosseland_diagnostics`);
   - subset-removal effect `log10(kappa_v4r5 / kappa_v4r5_without_X)`,
     the candidate-side analog of the production knockout effect.
5. **Base bookkeeping (for the BASE_CONTINUUM arm only).** Linear Rosseland
   bookkeeping: implied production flag contribution
   `delta_X = kappa_all_on − kappa_X_off` for the 6 stored flags plus the 2
   new ones; implied production base `= kappa_all_on − Σ delta_X` (H-minus,
   H/He Rayleigh, electron scattering, IFOP(13)-class terms, stimulated
   emission). Candidate base = v4r5 subset Rosseland without `h2plus` and
   `heminus`. Report the base-to-base gap
   `log10(kappa_candidate_base / kappa_implied_production_base)` on the
   slice, the implied-base fraction of the production continuum, and the
   candidate base sensitivity split (H-minus ff+bf versus scattering).

## Verdict rule (registered exactly)

Let `m2`, `m7` be the IFOP(2) / IFOP(7) knockout-effect signed medians on the
slice, in dex.

- if `m2 >= 0.03` → `H2PLUS_IMPLEMENTATION`
- elif `m7 >= 0.03` → `HEMINUS_IMPLEMENTATION`
- elif `m2 < 0.01` and `m7 < 0.01` → `BASE_CONTINUUM`
- else → `UNRESOLVED`

Sanity gates evaluated first: slice count != 199, reference-index or
temperature drift, baseline reproduction median |diff| >= 1e-6 dex, or any
non-finite primary number → `INCONCLUSIVE` with the failing counts.

Base-component naming (only inside the `BASE_CONTINUUM` arm): the bookkeeping
names **production H-minus free-free/bound-free** (versus the candidate's
John 1988 implementation) if and only if all three hold on the slice:

1. implied production base carries >= 90% of the production continuum
   (median linear ratio `implied_base / kappa_all_on >= 0.90`);
2. candidate combined H-minus (ff + bf) Rosseland log-sensitivity
   >= 0.80;
3. base-to-base gap signed median <= −0.05 dex (the gap survives
   base-to-base, not just total-to-total).

Otherwise the note states that the bookkeeping does not isolate a single
base component. Electron scattering and Rayleigh scattering are not
separately knocked out; their candidate-side sensitivities are reported so
the residual ambiguity is quantified, not hidden.

## Stop rule and output

Record the machine verdict, the deciding numbers (m2, m7 with p95; candidate
H2+/He- magnitudes; baseline reproduction; slice count), the named base
component if isolated, and one sentence on whether any legal pure-continuum
edit remains for the T<4000 K hole. Stop. Do not implement any construction.

Registered output:
`results/analytic_initializer/textbook_opacity_v4r6_cool_continuum_attribution_20260828.json`

Runner:
`experiments/analytic_initializer/run_textbook_opacity_v4r6_cool_continuum_attribution.py`

## Remote execution

Host `astronode-garching`. Checkout `/home/jdli/xiasangju/jdli/payne-zero`.
Python `PYTHONPATH=. .venv-linux/bin/python` with
`NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Single-threaded; the dev60 solver
funnel (remote PID 2682042) must not be disturbed. Do not evaluate production
opacity from macOS `.venv`.

## Post-run formal result

The diagnostic completed on 2026-08-28 on `astronode-garching`,
single-threaded under the registered thread caps (14 s; the dev60 solver
funnel, remote PID 2682042, was not disturbed). Production opacity was not
evaluated from macOS.

JSON: `results/analytic_initializer/textbook_opacity_v4r6_cool_continuum_attribution_20260828.json`
SHA-256: `4657d4a79a654a63c05feee2e1a6db235c64c0fadf00232098d6b9220f179121`
Log: `logs/textbook_opacity_v4r6_cool_continuum_attribution.log`

Sanity gates all passed: recomputed reference indices identical to the stored
20 (in order); corpus temperature drift 0.0 K; 8 cool stars; **199** slice
layers; all-on baseline replay bit-identical to the stored
`production_continuum_baseline` (median |diff| = 0.0 dex, max 0.0); all 60
replays used 1876 frequencies; replay contract assertions held (lines off,
zero positive line cells, molecules off, knockout flags took).

Deciding numbers on the slice (199 layers, signed median dex / p95 |dex|):

| quantity | signed median | p95 |
|---|---:|---:|
| IFOP(2) H2+ knockout effect | **+4.8e-6** | 0.0039 |
| IFOP(7) He-minus knockout effect | **+0.00459** | 0.0056 |
| candidate v4r5 H2+ subset-removal effect | +6.5e-6 | 0.0053 |
| candidate v4r5 He- subset-removal effect | +0.00460 | 0.0059 |
| v4r5 − production continuum (reproduced) | −0.06675 | 0.362 |
| candidate base − implied production base | −0.06229 | 0.363 |

Candidate v4r5 slice log-sensitivity medians: H-minus ff 0.5146, H-minus bf
0.4134 (combined **0.9280**), He- 0.0105, electron scattering 0.0200, H
Rayleigh 0.0046, H2+ 1.5e-5, H I bf 4.9e-6. v4r6 control is identical to
v4r5 on this slice (−0.06675). Production and candidate He- magnitudes agree
to ~1e-5 dex (0.004586 versus 0.004599): no He- implementation mismatch.
H2+ is null on both sides.

The machine verdict is **`BASE_CONTINUUM`**. Both new knockouts are null
(< 0.01 dex), so neither H2+ nor He-minus carries the gap. The registered
bookkeeping isolates the base component: implied production base carries
**0.9852** of the production continuum (>= 0.90), candidate combined H-minus
sensitivity is **0.9280** (>= 0.80), and the base-to-base gap is
**−0.0623** dex (<= −0.05). Named component: **production H-minus
free-free/bound-free implementation versus the candidate's John (1988)**.
Electron-scattering and Rayleigh terms are not separately knocked out; their
candidate-side leverage (0.020 and 0.0046) is too small to carry a 0.062 dex
gap without an implausible multi-dex component error, so the H-minus naming
is the tightest statement the data supports. If the entire base gap sits in
H-minus, the component-level difference is ~0.067 dex (~17%), far beyond the
~1% John–Bell & Berrington mutual agreement.

Legal-edit consequence: exactly one class of pure-continuum edit remains for
the T<4000 K hole — an implementation-level H-minus repair (matching the
production/ATLAS stimulated-emission and coefficient evaluation), which is
neither a runtime table load nor a corpus fit. A published-coefficient swap
(John → Bell & Berrington) is foreclosed: the required ~17% component change
exceeds their ~1% difference by an order of magnitude, consistent with the
v4r6 rejected-sibling analysis.

No construction was implemented. Gates, the mass integral, production, ODE,
funnel, and sealed holdout stayed closed.
