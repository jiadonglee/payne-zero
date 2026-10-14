# Gaia benchmark giants with the final networks: preregistration (2026-10-13)

The optical fits of Sect. 5 (`fig:mgiant_obs`, coauthor comments L1106–1128).
Driver: `experiments/reduced_state_emulator/coauthor_mgiant_benchmark_fit_20261013.py`.
Output: `results/coauthor_validation_20261005/mgiant_benchmark_fit/`.

## Stars

The three Gaia FGK benchmark giants below 4000 K in the local GBS v3
catalogue: α Cet (3796 K, 0.68, −0.45) and α Tau (3927 K, 1.11, −0.37),
with the UVES-POP spectra of `notes/mgiant_eso_bestfit_v1_closeout_20260924.md`
(R = 74,450), calibrate the TiO lines; γ Sge (3807 K, 1.05, −0.17), with the
NARVAL spectrum of the benchmark library (taken as R = 70,000; the fitted
extra broadening absorbs a different instrument width), is the test and is
not used in the calibration.

## Fit

Window 6480–6735 Å (air), Hα 6555–6572 Å and Li I 6707.0–6708.8 Å masked;
cubic B-spline continuum with 25 Å knots (13 coefficients) profiled at every
evaluation. Free: $T_{\rm eff}$, $\log g$, [M/H], RV, extra Gaussian
broadening. Fixed: [α/M] = [C/M] = [N/M] = [O/M] = 0, ξ at the benchmark
value, no rotation. Bounds as in the APOGEE fits. Start 3700 K, 1.0, −0.3
for every star and both line lists. Fast model, solved atmosphere, stability
rule and refinement gates as in `notes/mgiant_apogee_fit_preregistration_20261013.md`,
with the synthesis at R_grid = 600,000.

## TiO calibration

α Cet and α Tau are solved at their benchmark labels; RV and extra
broadening are fitted there with the default list. Five log gf offsets on the
Schwenke TiO lines, one per 51 Å bin of the window, are fitted to both spectra
at once at fixed atmospheres (Gauss–Newton, step 0.05 dex, at most 6
iterations, bound ±1 dex). Other lines are unchanged; the default list has
no ZrO.

## Reported

For each star and list: fitted $T_{\rm eff}$, $\log g$, [M/H] against the
benchmark values, χ² per pixel, and whether the refinement ends at the solve
cap. The calibration stars recover their labels partly by construction;
γ Sge is the independent test. No accuracy threshold is set.

## Compute record (2026-10-13)

`astro-highcpu09`, run directory of the APOGEE fits, launcher
`run_coauthor_mgiant_benchmark_fit_garching_20261013.sh`: default-list fits
of the three stars alongside the two reference solves, then the TiO
calibration, then the calibrated fits. CPU float64, four threads per process.
The benchmark library files used (catalogue and `NARVAL_gamSge.fits`) are
copied to `data/gbs/`.

## Benchmark labels (2026-10-14)

The values quoted above are those of the benchmark library catalogue
(`data/gbs/metadata/catalog_with_local_paths.csv`), which are the earlier
Heiter et al. (2015) / Jofré et al. (2014) labels, not the third version.
The reference solves, the TiO calibration and the calibrated fits were rerun
at the third-version labels of Soubiran et al. (2024; VizieR J/A+A/682/A145,
saved in `data/gbs/soubiran2024_cool_giants.json`): α Cet 3738 K, 0.66,
−0.24; α Tau 3921 K, 1.17, −0.20; γ Sge 3904 K, 1.06, −0.26. Microturbulence
stays at the library values. The default-list fits start from fixed labels
and are unchanged. The third version lists nine giants below 4000 K; ψ Phe
lies outside the label range of the corpus, and the others have no spectra in
this work.
