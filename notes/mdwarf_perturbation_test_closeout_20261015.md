# Small perturbations of converged M dwarfs: closeout (2026-10-15)

Preregistration: `notes/mdwarf_perturbation_test_preregistration_20261015.md`.
All 44 restarts ran 60 iterations on `astro-highcpu05` (10 workers × 8
threads, 2026-10-15 09:45 UTC onward), none with a solver error. Results:
`results/coauthor_validation_20261005/mdwarf_perturbation_test/summary.json`
(the in-run collect stopped at a non-finite path; `collect` now records the
first non-finite iteration and ran on the fetched paths).

## Deviation from the control after 60 iterations (p95 of |T/T_control − 1|)

| T_eff | log g | [M/H] | control drift | control flux p95 (%) | plus | minus | alternating |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 4.5 | 0.0 | 6.5e-4 | 0.22 | 1.8e-4 | 9.7e-5 | 1.9e-3 |
| 3800 | 4.5 | 0.0 | 5.7e-4 | 0.26 | 1.3e-4 | 3.4e-4 | 2.8e-3 |
| 3700 | 4.5 | 0.0 | 7.8e-4 | 0.15 | 7.7e-5 | 4.1e-5 | 2.6e-3 |
| 3600 | 4.5 | 0.0 | 5.9e-4 | 0.33 | 2.3e-4 | 3.1e-4 | 1.5e-3 |
| 4000 | 5.0 | +0.5 | 9.6e-4 | 0.28 | 8.7e-5 | 8.7e-5 | 1.4e-3 |
| 3800 | 5.0 | +0.5 | 3.4e-4 | 0.51 | 4.5e-5 | 5.8e-5 | 2.3e-3 |
| 3600 | 5.0 | +0.5 | 5.3e-4 | 0.59 | 3.7e-5 | 4.4e-5 | 0.57 |
| 3500 | 5.0 | +0.5 | 5.2e-4 | 1.07 | 6.6e-5 | 4.7e-5 | 2.2e-3 |
| 3400 | 5.0 | +0.5 | 2.3e-3 | 2.71 | 2.6e-4 | 1.6e-4 | 0.48 |
| 4000 | 4.5 | −1.0 | 7.7e-4 | 0.38 | 4.9e-5 | 7.2e-5 | 1.0, non-finite from iteration 42 |
| 3800 | 4.5 | −1.0 | 1.2e-3 | 1.87 | 1.2e-4 | 4.5e-4 | 0.51 |

All paths start at 1e-3.

## Reading

- A temperature change of the same sign in every layer decays within the first
  five iterations, to the level of the control's own drift (at
  iteration 60, at most 4.5e-4). The iteration contracts for this pattern at every dwarf.
- A change that alternates in sign from layer to layer does not decay at any
  dwarf (median factor per five iterations 0.99–1.01 where it stays bounded).
  At 7 dwarfs it grows by 1.4–3.2× in the first ten iterations and then holds at
  1.4–2.8e-3. At 4 dwarfs, 3400 and 3600 K at log g 5.0, [M/H] = +0.5 and 3800
  and 4000 K at [M/H] = −1.0, it grows by two to three orders of magnitude to
  0.48–1.0, and at 4000 K, [M/H] = −1.0 the atmosphere becomes non-finite at
  iteration 42.
- The growth is not monotone in T_eff: at log g 5.0, [M/H] = +0.5 it occurs at
  3600 and 3400 K but not at 3500 K. The metal-poor track converges only at
  3800–4000 K, and both of its tested dwarfs grow. Near the edge of the
  converged dwarfs, the production iteration is not locally stable for the
  alternating pattern.

## Limitations

- Dwarfs at 3200 K or cooler have no converged atmosphere, so the test does
  not reach them. The original Kurucz code is not compared.
- Each start passes through the (m,T) rebuild; one amplitude (1e-3) and two
  shapes. The test shows non-decay and growth of one pattern; it does not
  identify the mechanism.
- The M-giant contraction factors of the paper measure the change between
  iterations and are not directly comparable with these deviations.

Paper: `Npert*` macros, Sect. 6 (limits and future work).
