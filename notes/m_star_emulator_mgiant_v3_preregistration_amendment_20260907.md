# Cold safe-zone emulator v3 (M-giant focus) preregistration amendment

Date: 2026-09-07

Amends `notes/m_star_emulator_safezone_v2_preregistration_20260907.md` after
the v2 acceptance verdict: the giant branch is solver-grade (9/9 warm-start
convergence in 3-6 iterations) and the dwarf branch is not (3/9). The
campaign goal narrows to M giants; dwarf coverage is deferred.

## Boundary amendment

The giant boundary moves from 3500-4000 K to 3000-4000 K (`logg 0.5-2.5`,
all metallicities, `vmic 2 km/s`). Grounds: 17 v1r2 giant products between
3000 and 3300 K already pass the unchanged frozen gates, and both EOS walls
are dwarf phenomena - no giant wall exists in the evidence. Dwarf boundaries
are unchanged; dwarf rows simply leave the training corpus.

## Giant gap round 2

The remaining open-track gap is 53 nodes (v1r2 train grid in 3000-4000 K
minus admitted products), run by `m_star_giant_supplement_v1 --from-v1r2-gap
--iteration-cap 240` into `results/m_star_giant_supplement_v2_cap240`.
Everything stays frozen except the compute budget: same MARCS-seed
machinery, strict criterion, self-restart leg, imported flux gate. Cap 240
follows the `pipeline_complete` precedent of certifying stubborn nodes at
higher caps.

## M-giant corpus and training

`m_star_cool_corpus_safezone_v1 --stellar-class giant --campaign
m_star_cool_corpus_mgiant_v1`: giant-only pool, the giant carve-out
(`g1.5 m-1.0 t3500`, `g2.5 m0.0 t3500`, `g1.5 m+0.5 t3500`) held out as cool
validation, plus the six giant rows imported from the v1r1 validation corpus.
`validate_cool_corpus` now checks per-class minima only for classes present
in the corpus. Training otherwise unchanged: 4x512 two-field architecture,
52,199-row base corpus, 50:50 group sampling, minimax profile-p95 checkpoint
selection, three seeds, A100.

## Acceptance (M-giant scope)

1. Profile: cool (giant) validation `T p95 <= 3e-3`, `log m p95 <= 7.7e-3`
   (median over seeds).
2. Solver: emulator warm start, 30-iteration policy, giant validation rows
   converge `>= 90%`.
3. Spectral: TiO `6650-6670 A` gate at `5e-3` on converged giant validation
   products versus their truth products.
