# Line-by-line calibration of the APOGEE CN, CO and OH lines: preregistration (2026-10-15)

Coauthor item L995–998: calibrate the CN, CO and OH oscillator strengths on
part of the APOGEE sample, keep the rest as the test, and reach a fit that is
no worse than the ASPCAP model spectra. The per-group calibration of
`notes/mgiant_apogee_fit_closeout_20261014.md` (four offsets: CN, CO, OH and
the 13C isotopologues) leaves the fits behind the ASPCAP models: median χ²
per pixel on common pixels 351 against 271 on the 44 calibration stars, and
332 against 272 on the 20 control stars.

Driver: `experiments/reduced_state_emulator/mgiant_apogee_linebyline_20261015.py`;
launcher `run_mgiant_apogee_linebyline_garching_20261015.sh`; refits with
`coauthor_mgiant_apogee_fit_20261013.py fit --offsets .../linebyline/linebyline.json`.
Output: `results/coauthor_validation_20261005/mgiant_apogee_fit/linebyline/`
and the arms `linebyline_aspcap` and `linebyline_fixed`.

## Lines

Every APOGEE DR17 CN, CO and OH line in 1510–1700 nm, including the 13C
isotopologues: 32,715 lines (CN 7,749, CO 23,006, OH 1,960). The other
molecules and the atomic lines stay as in the per-group arm.

## Calibration

- Stars: the calibration stars of the sample split with a solved reference
  atmosphere (49 of 50), each at the solved
  atmosphere, radial velocity, extra broadening and v sin i of its reference
  fit at the raw ASPCAP labels. The atmospheres are not re-solved.
- Model: the per-group calibrated list plus one additive log gf correction
  per line. The opacity of all other lines is evaluated once per star; each
  evaluation adds the 32,715 lines, solves the transfer, applies the fit's
  rotation and instrument operators, and profiles the per-chip B-spline
  continuum by weighted least squares.
- Objective: summed χ² of these stars divided by their median χ² per pixel
  at zero correction, plus Σ(δ/0.25 dex)²; δ bounded to ±1 dex through a tanh
  map. Gradients by automatic differentiation (checked against central
  differences on a 1 nm window: relative agreement 1e-4).
- Optimizer: L-BFGS (history 20, strong Wolfe line search), at most 50
  iterations, stopped when the objective improves by less than 1e-4 of its
  value.
- Check before the calibration: the differentiable spectrum at zero
  correction agrees with the fit's own model of the same atmosphere.

## Refits and reported

All 300 stars are refitted from the ASPCAP labels with the line-by-line list
(`linebyline_aspcap`), and the 20 control stars also from the fixed start
(`linebyline_fixed`), with the fit, refinement and stability rule unchanged.
Reported as for the per-group arm: label differences from ASPCAP (raw and
calibrated) for the test stars, fits at the iteration limit and on bounds,
and the median χ² per pixel on the pixels where the ASPCAP model is defined,
for the calibration stars and for the control stars, against that of the
ASPCAP model. The test of the calibration is the 20 control stars and the
250 test stars, which take no part in it. The criterion of the coauthor, a
median χ² on common pixels no larger than that of the ASPCAP model, is
evaluated on the control stars and reported as met or not met.

## Compute

Calibration on one GPU (float64); refits on CPU nodes, four threads per
fit. Production defaults are unchanged.
