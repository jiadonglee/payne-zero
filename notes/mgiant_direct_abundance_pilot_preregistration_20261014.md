# Per-element abundances in M giants, 32-label pilot: preregistration (2026-10-14)

Coauthor item A1 (84-input networks). This pilot asks whether abundance
changes outside the eight-label relation reach the solved atmosphere, whether
the solver handles them, and how large the structural response is compared
with the error of the existing eight-label start. No network is trained.
Driver: `experiments/reduced_state_emulator/mgiant_direct_abundance_pilot_20261014.py`;
launcher `run_mgiant_direct_abundance_pilot_garching_20261014.sh`.
Output: `results/coauthor_validation_20261005/mgiant_direct_abundance_pilot/`.

## Parents

Eight development parents of bootstrapping round 1 (none is a holdout
parent), at their own $T_{\rm eff}$, $\log g$, [M/H] and ξ = 2 km/s, all with
[α/M] = 0:

| $T_{\rm eff}$ | $\log g$ | [M/H] |
| ---: | ---: | ---: |
| 3000 | 1.5 | 0.0 |
| 3000 | 2.5 | 0.0 |
| 3200 | 2.5 | −0.5 |
| 3300 | 2.5 | −1.0 |
| 3500 | 0.5 | 0.0 |
| 3500 | 1.5 | +0.5 |
| 3800 | 1.5 | 0.0 |
| 4000 | 2.5 | −1.0 |

## Compositions

Each element listed changes by 0.1 dex with random sign (seed 20261014),
relative to the parent's [X/H] = [M/H]:

| change | elements |
| --- | --- |
| alpha_split | Mg, Si, Ca |
| titanium | Ti |
| cno | C, N, O |
| all | all 81 elements of the direct-abundance encoding |

A null case per parent keeps the parent composition. Every case passes all 81
elements to the solver as absolute [X/H]. C/O stays below 0.9 in every case.
At the null compositions the 81-element vector equals the eight-label
composition of the parent (largest difference 0.0 dex, checked before any
solve).

## Starts

| start | profiles | cap |
| --- | --- | ---: |
| eight-label | final two-profile network of Sect. 4 at the projected labels, (m,T) rebuilt at the full composition | 90 |
| reference | the parent's converged (m,T), rebuilt at the full composition | 120 |

Projection: [M/H] = [Fe/H]; [α/M] the mean [X/Fe] of Ne, Mg, Si, S, Ca and
Ti; [C/M], [N/M], [O/M] = [X/Fe]. The null case solves the reference start
only.

## Success

Stability rule and endpoint agreement as in
`notes/mgiant_holdout_test_preregistration_20261012.md`: two consecutive
five-iteration segments with temperature p95 < 1e-4, mass p95 < 1e-3 dex,
TiO three-metric change < 5e-4 and the flux gate on every iteration; the
eight-label start succeeds when its endpoint agrees with the reference
endpoint within 5e-3 in all three TiO metrics.

## Reported

- Composition check: largest |log(N_X/N_X,null) − Δ_X| over the 81 elements
  in the solved reference endpoints.
- For each change: labels with a stable reference, eight-label successes,
  median stable iteration of each start, median wall time per start.
- Structural response: |T/T_null − 1| between the reference endpoint and the
  null endpoint of the same parent on the null column-mass grid (maximum and
  p95), set against the M-giant validation T p95 of the two-profile network
  (0.0053) and against the null endpoint's difference from the parent
  product.
- Structure-only spectral effect: three-metric difference between the
  reference endpoint and the null endpoint synthesized with the target
  composition, at 665–667 nm (R = 20,000) and 1555–1570 nm (R = 22,500),
  set against the 5e-3 endpoint limit.

## Reading

If the structural response stays below the network validation error and the
structure-only spectral effect below 5e-3, the eight-label start already
covers 0.1 dex per-element changes and an 84-input network adds no
demonstrated benefit at this range. A structural response counts as measured
only where it exceeds the null endpoint's difference from the parent product
of the same parent; below that it is reported as not resolved. If either exceeds them, the elements and
parents involved define the corpus of a later round. No corpus is generated
and no network is trained in this pilot.

## Compute

One CPU node for the solve (reference and eight-label starts per case, nulls
first), one GPU process for the evaluation, which waits for each solve.
Production defaults are unchanged.
