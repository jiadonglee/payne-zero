# M-star two-field emulator v1 preregistration

## Boundary

This is a development bootstrap campaign. It does not change production
routing, open the existing sealed holdout, use Korg, or claim general
`Teff < 4000 K` support. MARCS is not a training target. Every admitted target
must be a terminal Payne-Zero ATLAS atmosphere.

## Solar-composition spine

- `Teff = 4000, 3950, ..., 3000 K` (21 temperatures).
- Giants: `logg = 0.5, 1.0, 1.5, 2.0, 2.5`, `vmic = 2 km/s`.
- Dwarfs: `logg = 4.5, 4.75, 5.0, 5.25, 5.5`, `vmic = 1 km/s`.
- `[M/H] = [alpha/M] = 0`.
- Total: 210 nodes in 10 complete temperature tracks.

The split unit is a complete temperature track. With ten tracks, a balanced
70/15/15 split is impossible while retaining giants and dwarfs in both
validation and sealed sets. The frozen first-round realization is therefore
60/20/20: six train, two validation, and two sealed tracks. Sealed tracks are
not run by default and require an explicit flag.

## Frozen flux gate

Before any cool continuation, solve the 4000 and 4500 K reference nodes on the
eight opened tracks. At least 12 of 16 reference solves must converge. For each
of the median, p95, and maximum absolute all-layer flux-error metrics, freeze
the threshold at 1.25 times the largest successful reference value. If fewer
than 12 reference solves succeed, stop before the cool spine.

## Truth admission

Proceed from 4000 to 3000 K in 50 K reduced/rematerialized `(m,T)` steps. Every
candidate is independently restarted from its own final `(m,T)`. A row enters
the truth corpus only if:

1. primary and self-restart solves converge under the unchanged 30-iteration
   policy;
2. both final states are finite, positive, and monotonic in column mass;
3. both pass all three frozen flux thresholds;
4. primary/restart differences have temperature-relative p95 no larger than
   `3e-3` and column-mass p95 no larger than `7.7e-3 dex`.

An ineligible step blocks all cooler nodes on that track. Failures remain in
the manifest.

## Candidate training and opened validation

Train three seeds of the existing 4x512 physical two-field architecture from
scratch on the immutable 52,199-row corpus plus admitted cool rows. Existing
and cool groups each receive one half of the expected training-sample weight.
Sealed cool rows are rejected by the loader.

Opened validation requires profile p95 errors below `3e-3` in temperature and
`7.7e-3 dex` in column mass, no monotonicity violation, at least 95% solver
convergence overall, at least 90% per luminosity class, at least 80% within
15 iterations, and flux/path-consistency passes for every validation solve.
Existing-domain retention is required separately before any sealed run.
