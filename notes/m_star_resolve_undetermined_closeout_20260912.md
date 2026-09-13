# Undetermined-point resolution closeout

Date: 2026-09-12

## Outcome

Four of the five undetermined twelve-point validation entries are resolved,
all four pass; the fifth is terminal.

| point | remedy | emulator frozen | reference frozen | cross-arm | verdict |
|---|---|---|---|---|---|
| t3150 g1.5 m-0.5 | budget 30 -> 60 | 40 | 60 | dT p95 8e-4, TiO <= 1e-4 | pass |
| t3300 g0.75 m+0.5 | budget 30 -> 60 | 45 | 85 | dT p95 1.9e-3, TiO <= 1e-4 | pass |
| t3050 g2.0 m+0.5 | walk reference + budget 60 | 55 | 30 | dT p95 7e-4, TiO <= 1e-4 | pass |
| t3850 g0.5 m-0.5 | walk reference + budget 60 | 50 | 25 | dT p95 1.4e-3, TiO <= 2e-4 | pass |
| t3250 g2.0 m0 | walk reference | 40 (stable) | none | - | undetermined, terminal |

(`results/m_star_resolve_undetermined_v1/resolution.json`)

## What the resolution established

- The two slow points needed iterations, not data: their reference arms were
  already stable, and with a 60-iteration budget the v4 emulator seeds
  converge to the same atmosphere the reference finds (agreement at the
  1e-4 spectral level).
- The two hard corners (`t3050`, `t3850`) have working references once the
  seed is a continuation walk from the nearest converged truth; the v4
  emulator arms then stabilize at 50-55 iterations and agree with those
  references at the same tight level.
- `t3250 g2.0 m0` is terminal: the walk from the t3600 same-track truth dies
  entering t3575 even with 25 K legs, while the v4 emulator arm itself
  stabilizes by 40 iterations.  There is no reference atmosphere to validate
  against - the same phenomenology as the dwarf walls, now mapped on the
  cool solar-composition g2.0 branch.

## Final twelve-point tally

Superseded by `notes/m_star_sixty_budget_v1_closeout_20260913.md`: the
`t3250` reference was subsequently built by a gravity walk and the point
passes, so the original twelve stand at **12/12 under the 60-iteration
configuration** (7/12 under the 30-iteration standard).
