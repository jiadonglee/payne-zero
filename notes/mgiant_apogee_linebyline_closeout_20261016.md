# Line-by-line calibration of the APOGEE CN, CO and OH lines: closeout (2026-10-16)

Preregistration: `notes/mgiant_apogee_linebyline_preregistration_20261015.md`.
Calibration on `astro-a100gpu01` (2026-10-15); refits on `astro-highcpu05` and
`astro-highmem01` (2026-10-15 12:13 UTC onward). Results:
`results/coauthor_validation_20261005/mgiant_apogee_fit/linebyline/linebyline.json`,
`.../line_offsets.npz`, arms `linebyline_aspcap` and `linebyline_fixed` in
`.../mgiant_apogee_fit/summary.json` (the per-group summary is kept as
`summary_before_linebyline.json`). The first launch stopped after one
iteration because the stopping rule read the objective from before the
L-BFGS step; it is kept in `linebyline/first_run_one_iteration/` and not used.

## Calibration

- Check: the differentiable spectrum at zero correction matches the fit's own
  model of the same atmosphere to 6.7e-16.
- 49 calibration stars with a solved reference atmosphere; 32,715 lines
  (CN 7,749, CO 23,006, OH 1,960).
- 50 L-BFGS iterations (the cap; the last step improved the objective by
  6.6e-4 of its value), 25 min on one A100. Objective 408,442 → 165,476.
- Median χ² per pixel of the calibration stars at fixed atmospheres:
  337 → 130.
- 7,189 lines changed by more than 1e-3 dex; rms correction 0.13 dex; 21 at
  the ±1 dex bound; |δ| percentiles 50/90/99 %: 0 / 0.08 / 0.69 dex.

## Refits (all 300 from ASPCAP labels, 20 controls also from the fixed start)

| arm | fits | ends at cap | on bound | χ²/pixel (test) | χ² common pixels | ASPCAP model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| per-group, ASPCAP start | 300 | 14 | 4 | 370 | 332 (20 controls) | 272 |
| line-by-line, ASPCAP start | 300 | 12 | 2 | 133 | 129 (20 controls) | 272 |
| per-group, fixed start | 20 | 1 | 0 | 319 | 302 | 268 |
| line-by-line, fixed start | 20 | 0 | 0 | 123 | 122 | 272 |

On the 42 calibration stars with an ASPCAP model and a converged fit, the
common-pixel median is 124 against 266 for the ASPCAP model.

Differences from the raw ASPCAP labels for the 250 test stars (mean, scatter):

| | per-group | line-by-line |
| --- | --- | --- |
| T_eff (K) | +24, 27 | +9, 19 |
| log g | +0.07, 0.20 | +0.09, 0.11 |
| [M/H] | −0.07, 0.07 | −0.01, 0.06 |
| [α/M] | +0.10, 0.04 | +0.04, 0.03 |
| [C/M] | +0.08, 0.08 | +0.05, 0.05 |
| [N/M] | +0.08, 0.03 | +0.03, 0.05 |

## Reading

The preregistered criterion, a median χ² on common pixels no larger than that
of the ASPCAP model on the control stars, is met: 122 against 272 from the
fixed start, and Payne Zero fits better than the ASPCAP model at all 20
control stars. The control and test stars take no part in the calibration.
The labels also move closer to ASPCAP in every mean and in the scatter of
T_eff and log g.

## Limitations

- The corrections were fitted at the atmospheres of the reference fits and
  are not iterated with re-solved atmospheres.
- The optimizer stopped at its 50-iteration cap with the objective still
  decreasing slowly.
- One 12C/13C ratio for all stars, from the per-group step (offset +0.82 dex).
- The ASPCAP model spectra come from an interpolated grid with its own line
  list and instrument model; the comparison is on common pixels only.

Paper: `Nline*`, `NobsGroup*` and updated `Nobs*` macros; Sect. 5, abstract,
conclusions, `fig_mgiant_apogee`.
