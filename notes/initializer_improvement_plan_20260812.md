# `(m,T)` initializer improvement plan — 2026-08-12

## Short answer

The next model will keep the current two-field initializer as a fast proposal,
add only a small bounded solver-aligned correction, and fall back to the shipped
six-field initializer whenever the proposal is not demonstrably safe. The main
goal is to remove rare failures without losing the measured iteration speedup.

The previous blind test remains a failure: the two-field candidate was faster
on paired usable stars (`4.14` versus `5.68` iterations), but had 19/200 profile
outliers and 25/189 normalized spectra above 0.5%. No production checkpoint is
changed by this work.

## Data contract

Two sets are selected once, before training:

- Opened calibration set: 400 validation-role stars, split into 200 ordinary,
  100 solver-tail, and 100 label-edge stars.
- New sealed holdout: 200 disjoint stars with the same 2:1:1 proportions. A
  60-star solver/spectrum subset is fixed in advance.

All previous development and audit manifests are excluded. The new sealed 200
may be read only as an exclusion list until the complete model and fallback
threshold are frozen.

## Execution stages

1. Train a clean three-seed `4x512` `(m,T)` ensemble on training-role rows only.
   It uses the stable grey-temperature and positive mass-increment coordinates.
2. Predict the opened calibration 400 and run the real solver for both the new
   two-field proposal and the shipped six-field initializer.
3. Synthesize full `400–900 nm`, `R=20,000` spectra for a preselected 60-star
   calibration subset.
4. Train a small bounded K=1 residual correction on the 300-star gate-training
   split. The correction is clipped in log space and keeps mass strictly
   monotonic.
5. Train a conservative reliability gate using labels, ensemble disagreement,
   distance from the training set, reconstruction status, and real-solver
   outcomes. The remaining 100 calibration stars select one fixed fallback
   threshold.
6. Freeze the two-field model, correction, and gate. Only then open the sealed
   200 once and run its preselected 60-star real-solver/spectrum test.

## Acceptance

The hybrid policy may replace the current initializer only if all are true on
the sealed test:

- no new `(m,T)` profile blowouts;
- usable solve count is not lower than the six-field initializer;
- at least 59/60 preselected solver stars converge;
- mean iterations remain below the six-field baseline;
- median spectral difference is at most `1.62e-3` and no more than one star is
  above `5e-3`;
- every fallback decision is recorded, including why it was triggered.

If the sealed test fails, the current production model stays unchanged. The
sealed results are retained as a final diagnostic and are not reused as another
development set.

## Cluster execution

The long calibration run is launched only under
`/home/jdli/xiasangju/jdli/payne-zero`. Node05 is not used while its SPHEREx
workers are active. The selected host is `astronode-garching-gpu`; training and
spectral synthesis use its A100, while the real solver is limited to six CPU
workers and one Numba thread per worker.

Main command:

```bash
NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=1 \
  nohup .venv-linux/bin/python -m \
  experiments.reduced_state_emulator.run_initializer_improvement_long \
  --workers 6 > logs/initializer_improvement_20260812.log 2>&1 < /dev/null &
```

Primary outputs:

- `results/initializer_calibration_20260812.json`
- `results/sealed_initializer_holdout_20260812.json`
- `results/initializer_improvement_20260812/calibration400/`
- `runs/initializer_improvement_20260812/calibration400/`
- `artifacts/reduced_state_emulator/initializer_improvement_20260812/`
