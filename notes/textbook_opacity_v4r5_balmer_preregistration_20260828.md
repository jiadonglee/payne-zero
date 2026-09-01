# Textbook opacity v4r5 Balmer-window hydrogen bound-free diagnostic

Date: 2026-08-28

This is a development-only diagnostic. It does not change v4 through v4r5,
`experiments/analytic_initializer/textbook_opacity.py`, the production solver,
the default initializer, gates, or any sealed holdout. The v4r5 offline
`FAIL_STOP` and the v4r4 hot-flag verdict `HYDROGEN_CONTINUUM_MISMATCH` remain
authoritative. No 10,228-star offline run, ODE, smoke, funnel, or cool-mass
repair is run. No new opacity version is added to the textbook module.

## Why this task exists

v4r5 ground-anchored H I populations closed the `T >= 30000 K` Lyman hole
(signed median `-0.537` to `-0.004` dex versus production continuum) and left
the 8000--15000 K control slice unchanged at `+0.209` dex. A global Kramers
`n^1` edge-power trial on the same 20 stars moved 8000--15000 K from `+0.208`
to `-0.096` (overshoot) and worsened `T >= 15000 K`. That global power is not
a candidate.

The remaining mismatch sits where the Rosseland harmonic mean lives in
`balmer_to_lyman`. The question is which hydrogenic ingredient is high, with
Lyman (`n=1`) and `T >= 30000 K` held fixed.

## Hypotheses (exclusive, not a kitchen sink)

All residuals are `log10(kappa_variant / kappa_production_continuum)` on the
frozen 20-star, lines-off, molecules-off grid.

- **H1** `BALMER_EDGE_CROSS_SECTION`. The v4 node law sets the `n=2` threshold
  to `n^2 * 6.30e-18 = 2.52e-17` cm2. The unused named constant
  `hydrogen_balmer_cross_section_cm2 = 1.40e-17` is the literature Balmer-edge
  scale. Scaling **only** `n=2` by `1.40e-17 / 2.52e-17`, leaving `n=1` at
  `6.30e-18` and `n >= 3` at `n^2 * 6.30e-18`, should close most of the
  `+0.21` dex without reopening the Lyman hole.
- **H2** `BALMER_GAUNT_SHAPE`. The mismatch is the `nu^{-3}` frequency law
  inside `balmer_to_lyman`, not the threshold value. A same-edge Karzas shape
  (or a frequency-dependent Gaunt ratio that is not flat) would then move the
  control slice after the H1 scale has been applied, or instead of it.
- **H3** `HIGH_N_OVERCOUNT`. Extra `n > 2` or `n > 6` bound-free, which
  production dissolves or does not tabulate the same way, carries the
  `+0.21` dex. Dropping `n >= 3` or `n >= 7` would then close the control
  slice, and an `n=2`-only reconstruction would sit near production.
- **H4** `NLTE_OR_STIMULATED`. Production replay `iterations=1` either applies
  non-unit `b_n` or a stimulated-emission algebra that textbook `1-exp(-u)`
  misses. Textbook must not ingest production departures as inputs. The
  registered H4 test is textbook-side: drop stimulated emission, and record
  that the runtime seeds `b_n = 1`.
- **H5** `NOT_HYDROGEN_BF`. H I free-free or H-minus (or another named
  continuum already in v4r5) leaks into 8000--15000 K at the `0.10` dex level.

A prior global `n^2 -> n^1` trial is cited as a negative control. It is not
re-run as a candidate and is not a licensed construction.

## Frozen inputs

- Ablation JSON
  `results/analytic_initializer/textbook_opacity_v4r4_hot_flag_ablation_20260828.json`
  SHA-256 `c136b076d5f135733e4d7e43081d2ed8040f3586b0f4cbd01283628dda613b66`.
  Use `reference_grid.reference_indices` / `references[].corpus_index`,
  `temperature_K`, `production_continuum_baseline`, and
  `v4r3_rosseland_opacity`.
- v4r5 hot-grid citation
  `results/analytic_initializer/textbook_opacity_v4r5_hot_grid_20260828.json`
  SHA-256 `d663496ab9128aa2b4b0ec58560c7def449e44d66f45987bdf5540c31ef67dad`.
- Corpus `source_data_files/atmosphere_emulator/five_label/strict_truth_52199.npz`
  on `astronode-garching`. Recompute v4r5 on stored `(P, T)`.
- Karzas tables may be read **offline** from
  `source_data_files/atmosphere_tables/karzas_latter_tables.npz` inside the
  diagnostic runner only. They are not loaded by the textbook candidate.

Primary slice: `8000 <= T < 15000 K` (expected `n = 324`).
Hot isolation slice: `T >= 30000 K` (expected `n = 52`). Both must be
reported for every variant.

## Named constants (not fit targets)

```text
hydrogen_ground_edge_cross_section_cm2 = 6.30e-18   # Lyman n=1, frozen
hydrogen_balmer_cross_section_cm2      = 1.40e-17   # literature n=2; unused by the v4 node loop
v4 n=2 edge                            = 2.52e-17   # 4 * 6.30e-18
n=2 scale                              = 1.40e-17 / 2.52e-17
```

## Local H I bound-free copy (runner only)

Reimplement the v4r5 H I bf loop inside
`experiments/analytic_initializer/run_textbook_opacity_v4r5_balmer_diagnostics.py`.
Do not edit `textbook_opacity.py`. Variants:

| name | construction |
|---|---|
| `v4r5` | frozen ground-anchored 10-level `n^2` edge, `nu^{-3}` |
| `n2_only` | zero `n != 2` |
| `n2_balmer_edge` | scale only `n=2` edge to `1.40e-17`; `n=1` and `n>=3` unchanged |
| `drop_n_ge_3` | drop `n >= 3` |
| `drop_n_ge_7` | drop `n >= 7` |
| `no_stimulated` | stimulated factor `= 1` on H I bf |
| `no_hminus` | zero H-minus bf and ff |
| `no_hi_ff` | zero H I free-free |
| `n2_karzas_shape` | `n=2` uses Karzas `sigma(nu)/sigma(edge)` times the **textbook** `2.52e-17` edge; skip if the npz is absent |
| `n2_karzas_full` | `n=2` uses Karzas `sigma(nu)` including its edge; skip if absent |

Also report window-resolved inverse-`kappa` fractions (`WINDOW_NAMES`,
especially `balmer_to_lyman`) and per-level Rosseland log-sensitivity on the
primary slice.

## Registered verdict

Let `delta = median_control(v4r5) - median_control(variant)` (positive means
the variant reduced the overprediction). Hot change is
`median_hot(variant) - median_hot(v4r5)`.

Thresholds, frozen before seeing the new JSON:

- identity move on the primary slice: `delta >= 0.15` dex
- high-n move: `delta >= 0.10` dex
- leak / stimulated move: `delta >= 0.10` dex
- hot isolation: `|hot change| < 0.03` dex for any Balmer-edge claim
- `n=2` is the primary-slice carrier if `n2_only` remains `>= +0.10` dex high
  versus production (removing Lyman and high-n does not fix 8000--15000 K)
- `n=1` still owns the hot tail if `n2_only` hot median `<= -0.30` dex

Machine verdict, first exclusive match:

1. `NOT_HYDROGEN_BF` if `no_hminus` or `no_hi_ff` has `delta >= 0.10` and
   `n2_balmer_edge` has `delta < 0.10`. Name the component.
2. `NLTE_OR_STIMULATED` if `no_stimulated` has `delta >= 0.10` and
   `n2_balmer_edge` has `delta < 0.10`.
3. `HIGH_N_OVERCOUNT` if `drop_n_ge_3` or `drop_n_ge_7` has `delta >= 0.10`
   and `n2_balmer_edge` has `delta < 0.10`.
4. `BALMER_EDGE_CROSS_SECTION` if `n2_balmer_edge` has `delta >= 0.15`,
   `|hot change| < 0.03`, and `n2_only` control remains `>= +0.10` dex.
   Quantify dex closed and leftover. If Karzas is present, the Balmer-window
   `log10(sigma_textbook_n2 / sigma_karzas_n2)` span (`p95 - p05`) must be
   `< 0.08` dex **or** `n2_karzas_shape` must have `delta < 0.10`; otherwise
   the edge claim is not exclusive of shape and the verdict is
   `BALMER_GAUNT_SHAPE` instead.
5. `BALMER_GAUNT_SHAPE` if the Karzas same-edge shape moves the control slice
   by `>= 0.15` dex while `n2_balmer_edge` moves it by `< 0.10`, or if H1
   would have matched except the Karzas ratio is not flat.
6. `INCONCLUSIVE` otherwise.

If `BALMER_EDGE_CROSS_SECTION` holds, a later candidate is licensed only as
the named-constant construction:

```text
n = 1  edge  6.30e-18          # unchanged Lyman
n = 2  edge  1.40e-17          # published Balmer; not n^2 * 6.30e-18
n >= 3 edge  n^2 * 6.30e-18    # still the v4 node law
frequency law                nu^{-3}
populations                  v4r5 ground-anchored
tables                       none
```

Do not implement that construction in `textbook_opacity.py` in this task.
Do not license a global `n^2 -> n^1` power.

The registered output is
`results/analytic_initializer/textbook_opacity_v4r5_balmer_diagnostics_20260828.json`.

## Remote execution

Code and the 20-star diagnostic ran on `astronode-garching` (Node-06),
checkout `/home/jdli/xiasangju/jdli/payne-zero`, Python
`.venv-linux/bin/python`. Local pytest: 4 passed. Remote pytest: 4 passed.
Log: `logs/textbook_opacity_v4r5_balmer_diagnostics.log`. Env:
`NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=.`.
The runner imports frozen v4r5 helpers and a local H I bf copy. It does not
import production `continuum_opacity.py`. Karzas tables were read offline
with NumPy from `source_data_files/atmosphere_tables/karzas_latter_tables.npz`.

## Post-run formal result

The diagnostic completed on 2026-08-28. The JSON SHA-256 is
`4b696cbf6435d1c4bc8292a725a28b8ea132d33e2c7e3eab0ac4742701e8e699`.
Prior ablation and hot-grid hashes matched the preregistration. The local H I
bf copy matched frozen v4r5 to relative error `0`. Production, solver, v4--v4r5
physics, gates, ODE, smoke, funnel, sealed, and cool-mass boundaries remained
closed. No new opacity version was added to `textbook_opacity.py`.

Primary slice `8000--15000 K`: 324 layers. Hot isolation `T >= 30000 K`: 52
layers. The machine verdict is `INCONCLUSIVE`. No later candidate is licensed.

| variant | 8000--15000 signed median (dex) | delta vs v4r5 | `T >= 30000` signed median | hot change |
|---|---:|---:|---:|---:|
| v4r5 (frozen) | **+0.209** | -- | **-0.004** | -- |
| `n2_balmer_edge` | +0.076 | 0.133 | -0.085 | **-0.081** |
| `n2_only` | -0.223 | 0.431 | -0.843 | -0.839 |
| `drop_n_ge_3` | -0.220 | 0.429 | -0.116 | -0.112 |
| `drop_n_ge_7` | +0.194 | 0.015 | -0.017 | -0.013 |
| `n2_karzas_shape` | +0.228 | -0.019 | +0.015 | +0.020 |
| `n2_karzas_full` | +0.097 | 0.111 | -0.067 | -0.063 |
| `no_stimulated` | +0.217 | -0.008 | +0.007 | +0.011 |
| `no_hminus` | +0.155 | 0.054 | -0.004 | 0.000 |
| `no_hi_ff` | +0.201 | 0.008 | -0.042 | -0.037 |

Exclusive-rule failures, frozen before the JSON:

- H1 `BALMER_EDGE_CROSS_SECTION` needs `delta >= 0.15`, `|hot change| < 0.03`,
  and `n2_only` still `>= +0.10` high. Observed: closed **0.133**, leftover
  **+0.076**, hot change **-0.081**, `n2_only` **undershoots** to **-0.223**.
- H2 `BALMER_GAUNT_SHAPE`: Karzas/textbook n=2 ratio span **0.069** `< 0.08`;
  same-edge shape moves the control slice the wrong way (`-0.019`).
- H3 `HIGH_N_OVERCOUNT`: `drop_n_ge_3` moves 0.429 dex, but `n2_balmer_edge`
  also moves 0.133 `>= 0.10`, so the identity is not exclusive. `drop_n_ge_7`
  is null (0.015). The extra bound-free is `n = 3..6`, not `n > 6`.
- H4 `NLTE_OR_STIMULATED`: stimulated ablation `-0.008` dex. Replay
  `iterations=1` seeds `b_n = 1`. Textbook did not ingest departures.
- H5 `NOT_HYDROGEN_BF`: H-minus 0.054 dex, H I ff 0.008 dex, both `< 0.10`.
  Control log-sensitivity: H I bf 0.86, H-minus bf 0.077, H I ff 0.016.

Supporting, non-verdict measurements:

- Control inverse-`kappa` weight is split: `balmer_to_lyman` **0.55**,
  `paschen_to_balmer` **0.42**. n=1 sensitivity is `~0`. n=2 is 0.44, n=3 is
  0.20, n=4..6 sum to 0.10, n=7..10 sum to 0.027.
- Offline Karzas total-column edges: n=1 `6.31e-18` (Lyman named constant
  holds), n=2 `1.39e-17` (literature `1.40e-17` holds), n=3 `2.16e-17` versus
  textbook `9 * 6.30e-18 = 5.67e-17`.
- Hot inverse-`kappa` remains **0.74** in `balmer_to_lyman` and only 0.26 in
  `above_lyman`. A Balmer-only n=2 edge change cannot be isolated from
  `T >= 30000 K`.

A Balmer-only named-constant n=2 candidate is **not** licensed. A global
`n^2 -> n^1` power remains forbidden. A later diagnostic would have to
preregister a joint n=2 plus n=3..6 threshold-constant hypothesis, still
freezing n=1 at `6.30e-18`, still `nu^{-3}`, still no runtime table load,
with an explicit hot-isolation gate. That construction is not implemented
here.
