# The original ATLAS12 on M dwarfs: closeout (2026-10-16)

Preregistration: `notes/kurucz_mdwarf_comparison_preregistration_20261016.md`.
18 runs on `astro-highmem02`, 9 single-threaded processes. Results:
`results/coauthor_validation_20261005/kurucz_mdwarf_comparison/`
(`summary.json`, `comparison.json`, per-run iteration files and models).
The first launch called the package's `atlas12` loop, which copies the output
model onto its own input path when a block does not converge
(`SameFileError`); the driver now runs the same input cards in its own block
loop with the input kept apart, and all 18 runs were repeated with it.

## Result

| T_eff | log g | [M/H] | Payne Zero | ATLAS12, MARCS start | ATLAS12, Kurucz grid start |
| ---: | ---: | ---: | --- | --- | --- |
| 3800 | 4.5 | 0.0 | converged | converged, gate (1 block) | converged, gate (1 block) |
| 3600 | 4.5 | 0.0 | converged | converged, gate (1 block) | converged, gate (1 block) |
| 3400 | 5.0 | +0.5 | converged | converged, gate (2 blocks) | converged, gate (3 blocks) |
| 3200 | 4.5 | 0.0 | not converged | package test passed, gate failed | not converged |
| 3200 | 5.0 | 0.0 | not converged | not converged | not converged |
| 3200 | 5.0 | +0.5 | not converged | not converged (ΔT 3.5e-3) | not converged |
| 3000 | 4.5 | 0.0 | not converged | not converged | not converged |
| 3000 | 5.0 | 0.0 | not converged | not converged | not converged |
| 3600 | 4.5 | −1.0 | not converged | not converged | not converged |

"Converged" is the package criterion (preregistered): largest relative
temperature change between the last two iterations in layers 40–75 below
1e-4 within 4 blocks of 30 iterations. "Gate" is a diagnostic added after
the runs: the frozen Payne Zero flux gate of the MARCS recovery (median, p95
and maximum absolute flux error below 0.23 %, 9.56 % and 23.2 %), applied to
the last ATLAS12 iteration.

- Controls: both starts converge and pass the gate. The ATLAS12 temperatures
  agree with the converged Payne Zero atmospheres within 0.50 % at most
  (0.04–0.50 % by star and start, on the common column-mass range).
- The six dwarfs that Payne Zero does not converge: 1 of 6 meets the package
  criterion (3200 K, log g 4.5, from the MARCS start), none from the Kurucz
  grid start; none passes the flux gate from either start. The non-converged
  runs end with flux errors of 10^4–10^6 % at their worst layer.
- The 3200 K, log g 4.5 endpoint has stopped changing but is not in flux
  balance: flux-error p95 12.8 % and maximum 82 %, 2.2 % at τ_R = 1e-3–1 and
  12 % at τ_R = 1–100, against at most 0.6 % and 1.6 % for the controls from
  the same start, and
  5165 K at τ_R = 1000 against 6290–8790 K for the controls. The package test
  checks only the temperature change.

## Reading

The original code fails at the same dwarfs as Payne Zero, from the Payne
Zero start and from its own grid start. The failure is shared by the Kurucz
iteration and is not specific to the Payne Zero implementation.

## Limitations

- Nine dwarfs, two starts, the package's input cards (mixing length 1.25 with
  overshoot) and its line data; no other settings of ATLAS12 were tried.
- The flux-gate reading was added after the runs; the preregistered criterion
  alone gives 1 of 6 from the MARCS start.

Paper: `Nkur*` macros, Sect. 6 (limits and future work).
