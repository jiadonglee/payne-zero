# M-star emulator v1r2 MARCS-seeded 100-row preregistration

## Purpose

This is a new prospective campaign following the immutable v1 and v1r1
`FAIL_STOP` results. Its only training-data target is exactly 100 new,
quality-gated Payne-Zero ATLAS atmospheres: 50 giants and 50 dwarfs.

MARCS is an initializer, not a training target. For every candidate, only the
native MARCS column-mass and temperature profiles are converted to Payne-Zero's
80-layer grid. Gas pressure, electron density, Rosseland opacity, radiative
acceleration, and convective fields are reconstructed by Payne-Zero and then
iterated by the ATLAS solver.

## Fixed candidate pool

- Native MARCS temperatures, in frozen priority order:
  `4000, 3500, 3000, 3800, 3300, 3900, 3600, 3200, 3750, 3400, 3700, 3100 K`.
- Giants: `logg = 0.5, 1.5, 2.5`, `vmic = 2 km/s`.
- Dwarfs: `logg = 4.5, 5.0, 5.5`, `vmic = 1 km/s`.
- `[M/H] = -1.0, -0.5, 0.0, +0.5`.
- `[alpha/M] = [C/M] = 0`.
- Complete `(class, logg, metallicity)` tracks have fixed train, validation,
  and sealed roles before solver execution.
- The opened train pool contains 108 giant and 108 dwarf candidates.

The temperature order is deliberately non-monotonic so that an early quota
still spans the full 3000--4000 K interval. Within each temperature, tracks are
ordered deterministically by metallicity and gravity.

## Solver and admission

Every node is independent; no failed temperature blocks another node.

1. Load the same-node native MARCS `(m,T)` and reconstruct the other fields.
2. Run Payne-Zero with a 60-iteration ceiling and require all-layer relative
   temperature change below `5e-4`.
3. Independently restart from the terminal ATLAS `(m,T)` under the same strict
   settling rule.
4. Keep the v1 frozen flux thresholds without refitting.
5. Keep the v1 finite/positive/monotone and primary/restart consistency gates:
   temperature p95 at most `3e-3` and column-mass p95 at most `7.7e-3 dex`.

Giant and dwarf pools may run concurrently on separate Garching CPU nodes.
Each class stops when at least 50 eligible rows exist. Corpus construction
selects the first 50 eligible rows per class in the frozen priority order.
Batch overshoot remains as versioned reserve evidence and is not admitted to
training. Pool exhaustion before either quota is `FAIL_STOP`.

## Training and validation

The cool corpus contains exactly 100 new train rows. The 12 already-opened v1r1
validation rows are imported unchanged: six giants and six dwarfs. No v1r1
strict-settling diagnostic product enters training.

After the corpus gate passes, train the unchanged three-seed 4x512 physical
two-field architecture on the immutable 52,199-row existing corpus plus the 100
cool rows. Existing and cool groups keep equal expected sampling weight.
Training runs on the Garching A100.

Opened candidate validation keeps the 30-iteration solver policy and all
existing profile, solver, flux, and fixed-point gates.

## Boundaries

- Do not run new sealed tracks or open the existing sealed holdout.
- Do not refit or loosen flux, path, corpus, or validation gates.
- Do not use MARCS outputs as truth.
- Do not run Korg.
- Do not change production routing.
- Do not claim general support for every star below 4000 K.
- Preserve every failed and unselected candidate.
