# payne_zero_figures

Every figure this repository produces. Before this package the plotting code was
eleven standalone scripts under `paper/` and `experiments/`, each carrying its
own copy of the palette, the rcParams and the artifact loaders.

## Layout

```
payne_zero_figures/
    style.py        palettes and rcParams presets
    data.py         artifact loading (results/, runs/, artifacts/)
    paper/          the nine manuscript figures
    reports/        lab reports and evidence figures
```

## Building

Both entry points want an interpreter with matplotlib. `.venv` is the *solver*
environment and does not carry one; it is needed for `paper/collect_numbers.py`
instead, which recomputes one quantity with numba kernels.

```bash
# from the repository root
MPLCONFIGDIR=/tmp/mpl python3 -m payne_zero_figures.paper.figures      # paper/figs/*.pdf
MPLCONFIGDIR=/tmp/mpl python3 -m payne_zero_figures.reports.make_figures  # figures/*.png
```

The report scripts take arguments; run any of them with `--help`. They resolve
inputs relative to the repository root, so run them from there.

## Two audiences, deliberately kept apart

`paper/` figures ship. They are vector PDF at A&A column widths, follow the
Paper I house style, and re-plot arrays already written to `results/` so they
cannot disagree with the tables beside them.

`reports/` figures are lab notebook. They are screen-first PNG and multi-page
PDF, and they are free to reference superseded checkpoints — which is exactly
why a manuscript number must never be read off one of them.

## Two things a future cleanup could silently break

Both are covered by `tests/test_figure_package.py`; that file exists to make
these fail loudly rather than quietly.

**The palettes clash on purpose.** `PaperPalette.LEARNED` is burnt orange and
`EvidencePalette.LEARNED` is blue; `PRODUCTION` is grey in one and orange in the
other. The names collided long before they lived in the same file. Collapsing
them into one palette would recolour every figure in the repository without
raising anything.

**There are two record loaders with opposite semantics.** `load_records` keeps
the last entry for a slug, because `run_many_restarts` appends and the file
legitimately holds reruns. `load_records_strict` rejects a duplicate slug as a
corrupt input. Merging them breaks whichever caller loses.

## Style presets

| Preset | Used by | Look |
|---|---|---|
| `PAPER` | manuscript figures | serif (TeX Gyre Termes), no grid, four inward-tick spines |
| `REPORT` | the initializer-comparison PDFs | Okabe-Ito on sans-serif, top/right spines dropped |
| `COOL_STAR` | the cool-star PDFs | the same one notch smaller |
| `EVIDENCE` | the screen PNGs | off-white surface, recessive grid |

`style.configure(preset, **overrides)` takes per-caller rcParams on top, which
is how the handful of scripts that differ from their preset by one key keep that
difference without minting a near-duplicate preset.

One style stayed local: `reports/plot_cool_star_3500_comparison.py` sets its own
rcParams because it is a genuine one-off that shares no font sizes with
`COOL_STAR`, and routing it through a preset would have meant overriding almost
every key.
