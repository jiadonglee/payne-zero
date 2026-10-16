# The original ATLAS12 on M dwarfs: preregistration (2026-10-16)

Coauthor item E5 (L1125), second part: compare the original Kurucz code on
the M dwarfs that Payne Zero does not converge. The Fortran chain is the
reference chain of Paper I (github.com/tingyuansen/kurucz, commit b27cf82,
prebuilt `bin_linux` executables, gfpred, low/high/diatomic, TiO (Schwenke)
and H2O line lists). Driver:
`experiments/reduced_state_emulator/kurucz_mdwarf_comparison_20261016.py`.
Output: `results/coauthor_validation_20261005/kurucz_mdwarf_comparison/`.

## Stars

ξ = 1 km/s, [α/M] = 0, all MARCS grid nodes:

| T_eff (K) | log g | [M/H] | Payne Zero |
| ---: | ---: | ---: | --- |
| 3000 | 4.5 | 0.0 | not converged |
| 3200 | 4.5 | 0.0 | not converged |
| 3000 | 5.0 | 0.0 | not converged |
| 3200 | 5.0 | 0.0 | not converged |
| 3200 | 5.0 | +0.5 | not converged |
| 3600 | 4.5 | −1.0 | not converged |
| 3600 | 4.5 | 0.0 | converged (control) |
| 3800 | 4.5 | 0.0 | converged (control) |
| 3400 | 5.0 | +0.5 | converged (control) |

## Starts

- `payne_zero_marcs`: the MARCS node rebuilt from (m,T) by Payne Zero, the
  start of the Payne Zero recovery, written as a Kurucz deck.
- `kurucz_grid`: the nearest model of the package's initial-atmosphere grid,
  the package default.

## Run and criterion

The package's `atlas12` procedure: one line-reading iteration, then blocks of
30 iterations with molecules, line and extra-line opacity, convection with
l/H = 1.25 and overshoot, `SCALE MODEL 80 -6.875 0.125`; a new block starts
while the package's convergence test fails, at most 4 blocks (120
iterations, the Payne Zero cap). SYNTHE is not run. Converged: no crash,
finite output, and the package's test, the largest relative temperature
change between the last two iterations in layers 40–75 below 1e-4.

## Reported

For each star and start: converged or not, crash, blocks used, the final
largest temperature change, the largest flux and flux-derivative errors of
the last iteration, and the final temperature range. For the controls, the
difference between the ATLAS12 and the Payne Zero converged temperatures on
the common column-mass range.

## Reading

If ATLAS12 converges dwarfs that Payne Zero does not, the failure lies in
the Payne Zero iteration and not in the physics of these atmospheres. If it
fails at the same dwarfs, the failure is shared by the original code. No
other threshold is set.

## Compute

One CPU node, one single-threaded ATLAS12 process per run, 18 runs.
Production defaults are unchanged.
