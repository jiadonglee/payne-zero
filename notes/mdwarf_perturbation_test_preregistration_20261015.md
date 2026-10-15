# Small perturbations of converged M dwarfs: preregistration (2026-10-15)

Coauthor item E5 (L1125): from converged atmospheres, add a small
perturbation and see whether the residual grows. Driver:
`experiments/reduced_state_emulator/mdwarf_perturbation_test_20261015.py`;
launcher `run_mdwarf_perturbation_test_garching_20261015.sh`. Output:
`results/coauthor_validation_20261005/mdwarf_perturbation_test/`.

## States

Eleven dwarfs that the production solver converged in the MARCS-grid
recovery (`results/m_star_emulator_v1r2_marcs100/status_dwarf.json`), along
three tracks toward the cool boundary, ξ = 1 km/s, [α/M] = 0:

| log g | [M/H] | T_eff (K) |
| ---: | ---: | --- |
| 4.5 | 0.0 | 4000, 3800, 3700, 3600 |
| 5.0 | +0.5 | 4000, 3800, 3600, 3500, 3400 |
| 4.5 | −1.0 | 4000, 3800 |

3600 K and 3400 K are the coolest converged dwarfs of their tracks.

## Paths

Four restarts per state with the production solver, no early stop, 60
iterations, each start rebuilt from (m,T) as in the M-giant tests:

| path | temperature of the start |
| --- | --- |
| control | converged |
| plus | converged × (1 + 1e-3) in every layer |
| minus | converged × (1 − 1e-3) in every layer |
| alternating | converged × (1 ± 1e-3), sign alternating with layer |

## Measured

- Deviation d_k: p95 over layers of |T/T_control − 1| at iteration k, on the
  control's column-mass grid, so that the drift of the control itself does
  not enter.
- Contraction factor per five iterations, d_{k+5}/d_k, while d_k > 1e-5;
  its median and its first value per path.
- The control's own drift (p95 of |T_60/T_0 − 1|) and final flux error p95.

## Reading

A factor below one means the iteration contracts near the converged
solution; a factor at or above one means the perturbation does not decay.
The trend of the factor with T_eff along each track, toward the cool
boundary, is set beside the M-giant factors of the paper (median 0.68 on
stable paths, 0.88 on capped paths), which measure the change between
iterations rather than the distance from a control. No threshold is set in advance; the factors are reported
as measured. The test cannot reach dwarfs at 3200 K or cooler, which have no
converged atmosphere. It does not compare the original Kurucz code.

## Compute

One CPU node, 44 solves with eight threads each. Production defaults are
unchanged.
