# APOGEE M-giant fits with the final networks: closeout (2026-10-14)

Preregistration: `notes/mgiant_apogee_fit_preregistration_20261013.md`.
Results: `results/coauthor_validation_20261005/mgiant_apogee_fit/summary.json`
(per-arm rows and statistics), `calibration/offsets.json`, per-star
`summary.json` and `physical_model_flux.npy`. Sample, bounds, line lists,
continuum, starts and gates are as preregistered.

| | calibrated, ASPCAP start | calibrated, fixed start | uncalibrated |
| --- | ---: | ---: | ---: |
| fits | 300 | 20 control | 20 control |
| end at the solve cap | 14 | 1 | 0 |
| on a label bound | 4 (log g) | 0 | 0 |

Calibration (49 stars): CN −0.05, CO 0.00, OH +0.12, ¹³C +0.82 dex
(¹²C/¹³C ≈ 14); χ² per pixel 484 → 392 at fixed atmospheres.

Test stars (250) against the raw ASPCAP labels, mean (standard deviation):
Teff +24 K (27), log g +0.07 (0.20), [M/H] −0.07 (0.07), [α/M] +0.10 (0.04),
[C/M] +0.08 (0.08), [N/M] +0.08 (0.03), ξ +0.04 (0.15) km/s. Against the
calibrated ASPCAP Teff: +167 K. By bin, Teff +13 / +39 / +21 K.

- Fixed against ASPCAP start (20): RMS 19 K, 0.03 dex in log g, 0.02 in [M/H].
- Control χ² per pixel (median): calibrated 335, uncalibrated 385; calibrated
  better for 18 of 20. Uncalibrated [C/M], [N/M] offsets +0.06, +0.12.
- ASPCAP model spectra on the pixels where they are defined (median χ² per
  pixel): 272 against 332 for Payne Zero on the 20 control stars, 271 against
  351 on the 44 calibration stars with a model file.
- Cost: median 21 min per fit on four CPU cores, 40 syntheses, one solved
  atmosphere; about 620 core hours for the 300 fits.

The calibrated Payne Zero list does not reach the ASPCAP models even on the
calibration stars; four offsets per molecule are coarser than the line-by-line
APOGEE calibration. The ASPCAP model spectra were fitted to an earlier coadd
of each star.
