# Gaia benchmark giants with the final networks: closeout (2026-10-14)

Preregistration: `notes/mgiant_benchmark_fit_preregistration_20261013.md`
(benchmark labels: third version, Soubiran et al. 2024).
Results: `results/coauthor_validation_20261005/mgiant_benchmark_fit/summary.json`.

TiO calibration on α Cet and α Tau: offsets +0.40 to +0.43 dex in all five
51 Å bins (bands about 2.6 times stronger); χ² per pixel of the two stars
174 → 148 at fixed atmospheres.

Fitted minus benchmark (Teff / log g / [M/H]), χ² per pixel:

| star | default list | calibrated TiO |
| --- | --- | --- |
| α Cet (calibration) | −242 K / −0.16 / −0.64, 120 (log g bound; second solve at the cap) | −141 K / −0.16 / −0.71, 122 (log g bound) |
| α Tau (calibration) | −278 K / −0.49 / −0.54, 106 | −185 K / −0.50 / −0.55, 106 |
| γ Sge (test) | −176 K / −0.49 / −0.10, 951 | −68 K / −0.42 / −0.11, 946 |

The default list places the three giants 180–280 K too cool. The TiO
calibration moves all three about 100 K warmer, including γ Sge, whose χ²
does not rise, but none reaches its benchmark temperature, and the
metallicities stay 0.1–0.7 dex low. TiO strength and temperature are nearly
degenerate in this window; one offset per bin cannot replace missing lines.
