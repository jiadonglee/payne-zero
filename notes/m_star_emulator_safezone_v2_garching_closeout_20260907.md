# Cold safe-zone emulator v2 Garching closeout

Date: 2026-09-07

## Outcome

The safe-zone chain ran end to end: certified inventory, giant-track
supplement, corpus assembly, 3-seed training, and the three-level acceptance.
The verdict is split by stellar class.

- Giants: the warm start converges 9/9 validation giants in 3-6 iterations
  (30-cap policy), lands on the truth fixed point spectrally in 8/9 cases
  (the ninth, `g1.5 m-1.0 t3500`, sits at 1.6x the 5e-3 TiO bar). The giant
  branch is acceptance-quality today.
- Dwarfs: 3/9 validation solves converge within 30 iterations. Failures are
  temperature-graded: `g4.75` converges at 4000/3900 K and diverges at
  3850/3800/3750 K; the three cold-corner carved dwarfs
  (`m-1.0 t3800`, `m0.0 t3400`, `m+0.5 t3600`) all diverge. Even converging
  dwarf solves land off the truth fixed point in the TiO window (0/3 pass).
- Profile gates: not met. Cool validation p95 over the three-seed median:
  temperature `6.2e-3` (gate `3e-3`), column mass `1.9e-2 dex`
  (gate `7.7e-3`). Monotonicity violations are zero everywhere, and training
  was stable in both the 50-epoch smoke and the 300-epoch runs.

`all_gates_pass = False` (`status = fail_opened_validation`).

## Direct interpretation

The two-field emulator is solver-grade for giants and not yet for dwarfs.
The dwarf failures mirror the data: 24 dwarf training rows against 35 giant
rows, with the cold corners exactly where the corpus is thinnest. The
preregistered profile bars (3e-3 / 7.7e-3) came from Ting's basin calibration
on the warm-dominated single-corpus training; the minimax selection score was
in fact bound by the existing-group mass p95, not the cool group. No gate,
threshold, or production routing was changed to move these numbers.

The giant supplement campaign itself hit a second wall: of the 17 open-track
giant nodes between 3500 and 4000 K, 16 failed the unchanged truth-generation
admission (mostly the primary solver under the strict criterion on `logg 0.5`
tracks) and one (`g2.5 m-0.5 t3500`) was admitted. The certified library
therefore stands at 65 in-boundary products (38 giants, 27 dwarfs); the 16
failed nodes are retained as ineligible evidence.

## Retrieved evidence

Remote tree:

`/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-mstar-emulator-v1-20260831`

Local copies:

- `results/m_star_cold_library_inventory_v1/` (inventory, both trees agree)
- `results/m_star_giant_supplement_v1/` (17 case records, 7 products)
- `results/m_star_cool_corpus_safezone_v1/` (77-row corpus: 59 train + 18 validation)
- `results/m_star_emulator_safezone_v2/candidate_validation/` (18 solver records)
- `results/m_star_emulator_safezone_v2/spectral_stage` evidence in
  `results/m_star_emulator_safezone_v2/spectral_gate.json`
- `artifacts/m_star_emulator_safezone_v2/` (3 checkpoints + training summary)
- `logs/mstar_giant_supplement_v1.log`, `logs/mstar_safezone_v2_smoke.log`,
  `logs/mstar_safezone_v2_training.log`, `logs/mstar_safezone_v2_candidate.log`,
  `logs/mstar_safezone_v2_spectral.log`

## Next lever

Dwarf coverage, not gate tuning: more dwarf rows on the open tracks above the
walls, dwarf-weighted sampling inside the cool group, or a longer/dwarf-heavy
schedule. The giant branch needs no further work for warm-start use.
