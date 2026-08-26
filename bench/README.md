# Reference-solver benchmark

Measures the **unmodified** `payne_zero_atmosphere` production solve. This is the
Stage 0 baseline of the differentiable-initializer project: every later claim
about iteration reduction is compared against numbers produced here, by running
the real solver, not a differentiable twin.

Nothing in this package imports torch or is differentiable.

## Running

```bash
python -m bench.run_reference --count 300 --out runs/baseline
python -m bench.report runs/baseline/records.jsonl --json runs/baseline/summary.json
```

Labels can come from a file instead of the sampler:

```bash
python -m bench.run_reference --labels my_labels.jsonl --out runs/baseline
```

Add `--traces` to dump the runner's per-iteration state
(`runner.py:1367`), which is what the Stage 1 twin is validated against. It is
large and slow, so it is off by default.

Records stream to `records.jsonl` as they land, so a run killed partway is still
usable.

## Environment

`bench.environment` is imported before anything that pulls in Numba and sets
`NUMBA_THREADING_LAYER=workqueue`. **This is required, not a tuning knob.** With
the default `omp` layer the solver segfaults during the opacity stage of the
first iteration, with no Python traceback, because a second OpenMP runtime is
already loaded by torch or an Anaconda-linked NumPy. An explicit
`NUMBA_THREADING_LAYER` in the caller's environment is honoured.

Thread count does not meaningfully change results: 1-thread and 8-thread solves
agree bit-for-bit on 26 of 27 output arrays, with `column_mass` differing at
~1e-8 from float reassociation in the chunked reductions
(`transfer_kernels.py:34`).

The solver needs `scipy` at runtime — Numba lowers `np.dot` in the transfer
kernel through it. It was missing from `pyproject.toml` dependencies and has
been added.

## Sizing a run

Peak resident memory is about **7.8 GB per process**, peaking near 14 GB during
startup: `line_selection.py:381` concatenates the three 1.4 GB predicted-atomic
line-catalog shards, briefly holding both the parts and the result.

- **Workers must be sized by memory, not cores**: roughly `RAM_GB / 16`.
  A 17 GB machine runs exactly one.
- Catalog reads are process-resident (`line_selection.py:343`), so a worker that
  handles many stars pays the ~30 s load once. `run_many` keeps pool workers
  alive across items, so this amortization already happens — do not switch to
  one-process-per-star.
- Warm cost is roughly 35–50 s per star on 8 Apple-silicon cores: the first
  iteration spends ~25 s in opacity (line selection), later iterations ~5 s.

## What the report measures, and why

The headline is the **tail and the retry rate**, not the mean.
`cli.py:427` floors every solve at three iterations and caps each trial at
fifteen, with two trials allowed. A mean iteration count therefore cannot drop
below three however good the initializer becomes, so `--report` leads with:

- `p90/p95/p99` of the converging trial's iteration count;
- the fraction of stars that needed a second trial, each of which costs a whole
  extra fifteen-iteration budget;
- `recoverable_iterations`: iterations actually burned above the floor, which is
  the real upper bound on what a better initializer can save;
- the contraction ratio `q = r_next / r` on the deep-layer `dT/T` that
  `convergence.py:60` actually tests — including how often it is non-monotonic.

## Files

| file | purpose |
| --- | --- |
| `environment.py` | Numba threading configuration; import before the solver |
| `labels.py` | support bounds read from the released checkpoint; samplers: uniform box, IID from the training corpus, boundary, hard region |
| `run_reference.py` | the production trial loop, mirroring `cli.py:383-448` |
| `report.py` | aggregation into the baseline report |
| `perturb_deck.py` | deck-perturbation sensitivity experiment (§4.1 of the progress notes): does `N_iter` respond to noise at the deck format's own resolution? |

## Benchmark slices

The headline benchmark is split into three slices (`runs/run_cluster_baseline.py`
drives them into `runs/baseline_cluster/<slice>/`):

- **IID** (`sample_iid_from_corpus`, 1000 stars): resampled from the five-label
  training corpus (`strict_truth_52199.npz`, v1.3 release, SHA-256 verified).
  The corpus is rank-coupled to the H–R diagram — the distribution the
  initializer was trained on — so this slice measures the production solve on
  the population the initializer is actually for.
- **Boundary** (`sample_boundary`, 500 stars): at least one coordinate pinned
  within 5% of a support-box face, the rest uniform. Measures the edges, where
  the initializer extrapolates worst.
- **Hard region** (`sample_hard_region`, 500 stars): uniform inside
  logg 0.7–2.8, [M/H] −2.5…−0.5 — the sub-box where the Stage 0 tail lives.

Report each slice separately (`python -m bench.report .../records.jsonl`);
never pool them — the failure rate is a property of the sampling distribution,
and pooling mixes populations by construction.

## Deck-perturbation experiment

```bash
python -m bench.perturb_deck runs/baseline/records.jsonl --out runs/deck_perturbation --workers 32
python -m bench.perturb_deck --report runs/deck_perturbation/records.jsonl
```

Picks baseline stars stratified by trial-0 iteration count (fast / mid / slow),
re-runs an in-harness control twice per star (determinism check), then solves
with the four `%.3E` deck columns (gas pressure, electron density, Rosseland
opacity, radiative acceleration) multiplied by `1 + eps * U(-1, 1)` per layer
for `eps` in 1e-4, 5e-4, 1e-3 — the rounding-noise model of the fixed-width
format. All solves use the production policy with `max_trials = 1`, so the
report shows the raw first-trial `N_iter` response (`dN` vs control) and
convergence flips. Records stream to `records.jsonl`; re-running the same
command skips `(slug, epsilon, rep)` jobs already on disk.
