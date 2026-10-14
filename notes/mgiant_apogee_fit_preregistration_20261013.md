# APOGEE M-giant fits with the final networks: preregistration (2026-10-13)

The H-band fits of Sect. 5 (`fig:mgiant_apogee`, coauthor comments L1059,
L1081, L1138–1167). They replace the eight-star fits of
`notes/mgiant_apogee_pzfit_v1_closeout_20260927.md`, which used the v4
five-label network and kept the structure at scaled-solar abundances.
Driver: `experiments/reduced_state_emulator/coauthor_mgiant_apogee_fit_20261013.py`.
Output: `results/coauthor_validation_20261005/mgiant_apogee_fit/`.

## Sample

From the DR20 Mg-cohort quality pool with mwmStar coadds at S/N > 500
(`data/dr20/mgiant_quality_pool.fits`, `mgiant_snr500_files_qc.fits`):
apo25m only (one LSF), good-pixel fraction ≥ 0.9, ASPCAP `flag_bad` = 0,
finite raw FERRE labels, raw $T_{\rm eff}$ 3000–4000 K, raw $\log g$ 0.5–2.5,
raw [M/H] −1.0 to 0.5. Eligible: 146 / 1376 / 4810 stars at 3000–3300 /
3300–3600 / 3600–4000 K. 100 stars are drawn per bin with seed 20261014
(`sample.json`). In each bin the first 17/17/16 drawn are calibration stars
(50 in all) and the next 7/7/6 are control stars (20); the other 230 and the
control stars form the test set of 250.

The labels compared are the raw FERRE values, because the calibrated
ASPCAP table has no [C/M] and [N/M] for these stars and the raw values are
the ones of the ASPCAP model spectra. Calibrated $T_{\rm eff}$, $\log g$ and
[M/H] are kept in `sample.json` and reported alongside.

## Forward model

Ten free parameters: $T_{\rm eff}$, $\log g$, [M/H], [α/M], [C/M], [N/M], ξ,
RV, extra Gaussian broadening, v sin i. [O/M] = [α/M].

- Fast model: the final six-profile network
  (`mgiant_final/checkpoint_6profile.pt`, md5 fc9f122e) at the eight labels,
  used without a rebuild, so the structure carries [α/M], [C/M], [N/M] and ξ.
  Synthesis 1510–1700 nm at R_grid = 120,000, rotation kernel, Gaussian LSF
  R = 22,500 plus the extra broadening.
- Solved atmosphere: the final two-profile network
  (`checkpoint_2profile.pt`, md5 7f955a15) rebuilt at the composition, solved
  until the stability rule of the M-giant test holds (two consecutive
  five-iteration segments, T p95 < 1e-4, mass p95 < 1e-3 dex, TiO change
  < 5e-4 over 665–667 nm at R = 20,000, flux gate of
  `results/m_star_emulator_v1r2_marcs100/flux_gate.json` on every iteration),
  cap 90. A solve that does not meet the rule within the cap ends the
  refinement of that fit; it is counted as a fit that ends at the cap.
- Bounds: $T_{\rm eff}$ 3000–4000, $\log g$ 0.5–2.5, [M/H] −1.0–0.5 (the corpus
  range); [α/M] −0.2–0.4, [C/M] −0.5–0.3, [N/M] −0.3–0.8, ξ 1.0–3.0 km/s,
  wider than the corpus (±0.1, −0.3–0.1, −0.1–0.5, 1.5–2.5), where the network
  start is extrapolated and the solver decides; RV ±5 km/s around the scan
  start, extra broadening 0–10 km/s, v sin i 0–15 km/s.
- Continuum: data divided by the file continuum only. A per-chip cubic
  B-spline with 10 nm knot spacing (26 coefficients) multiplies the model and
  is solved by weighted linear least squares at every model evaluation. No
  running median on data or models.
- Optimizer: bounded Gauss–Newton with one-sided finite differences and a
  backtracking line search (`fitter.fit_normalized_spectrum`, at most 12
  iterations), then `refine_with_physical_atmosphere` with the gates of the
  earlier fits (discrepancy RMS ≤ 2e-3, χ² degradation ≤ 0.1, at most four
  solved atmospheres).

## Line lists and calibration

Atomic lines: native Payne Zero list with the Sun/Arcturus overlay
`linelist_calibration/data/sun_arcturus_fts_hband_shared.npz`. Molecular
lines: CN, CO, OH, C2 and FeH from the APOGEE DR17 list
(`turbospec.20180901t20.molec`) at solar isotopic fractions; other molecules
native.

Calibration on the 50 calibration stars: each star is solved at its raw
ASPCAP labels, and RV, extra broadening and v sin i are fitted at those
labels. Four log gf offsets — all CN lines, all CO lines, all OH lines, and
an extra offset on the ¹³C isotopologues of CN and CO, which stands for the
¹²C/¹³C ratio — are then fitted to minimize the summed χ² over the 50 stars
at fixed atmospheres (Gauss–Newton, step 0.05 dex, at most 6 iterations,
bound ±1 dex). The atmospheres are not re-solved with the offsets.

## Fits

| arm | stars | start |
| --- | --- | --- |
| calibrated | 250 test and 50 calibration | raw ASPCAP labels, clipped to the bounds |
| calibrated, fixed start | 20 control | 3600 K, 1.0, −0.3, α=C=0, N=+0.2, ξ=2 |
| uncalibrated | 20 control | raw ASPCAP labels |

RV starts from a ±30 km/s scan of the start spectrum; extra broadening at
3.5 km/s, v sin i at 3 km/s.

## Reported

- Fraction of test fits whose refinement ends at the solve cap, and of fits
  that end on a bound.
- For the 250 test stars: mean and RMS of fitted minus raw ASPCAP for
  $T_{\rm eff}$, $\log g$, [M/H], [α/M], [C/M], [N/M]; the same against the
  calibrated $T_{\rm eff}$, $\log g$, [M/H]; by temperature bin.
- χ² per pixel of the calibrated fits on the calibration and the test stars,
  of the uncalibrated fits on the control stars, and of the ASPCAP model
  spectra where an `astraStarASPCAP` file is available, on common pixels.
- Fixed against ASPCAP start on the control stars: label differences.
- Fit cost: syntheses, solves and wall time per star, hardware.

No accuracy threshold is set; the comparison with ASPCAP is descriptive.
Production defaults are unchanged.

## Compute record (2026-10-13)

Run directory `/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-apogee-fit-20261013`
(local code, line lists, the 300 spectra and both final checkpoints;
`source_data_files` hard-linked to the coauthor run directory). CPU float64,
four threads per process; one fast-model evaluation takes 8–15 s, a rebuilt
start 35 s, a solve about 15–25 s per iteration. Local smoke test on one star
(MPS float32, two iterations, one solve): the solve met the stability rule at
iteration 30. At 15:05 CEST: reference solves of the 50 calibration stars on
`astro-highcpu09` (12 workers) and the uncalibrated fits of the 20 control
stars on `astro-highcpu07` (10 workers).
At 15:40 CEST the uncalibrated control fits were stopped: a second
dispatcher had started the same 20 stars on `astro-highcpu07` an hour after
the first, so both wrote the same directories. The output was deleted and the
20 fits restarted with one dispatcher (10 workers). The reference solves met
the stability rule at 49 of 50 calibration stars (65371511, 3217 K, did not);
the calibration runs on the 49 on `astro-a100gpu01` (CUDA float64).
65 of the 70 calibration and control stars have an `astraStarASPCAP` file on
the DR20 SAS (downloaded to `data/dr20/results/star/`); 85721195, 84248284,
81660964, 64018661 and 67448859 have none.
Calibration (49 stars, four Gauss–Newton steps): CN −0.053, CO −0.003,
OH +0.120 dex, ¹³C isotopologues +0.822 dex (¹²C/¹³C ≈ 14 instead of 89);
χ² per pixel at fixed atmospheres 484 → 392 (`calibration/offsets.json`).
Calibrated fits launched 16:30 CEST: `astro-highcpu05` (11 workers) and
`astro-highcpu09` (12 workers) with a third of the 300 stars each;
`astro-highcpu07` (10 workers) takes the 20 fixed-start control fits and then
the last third once the uncalibrated control fits end (`chain07.sh`).
