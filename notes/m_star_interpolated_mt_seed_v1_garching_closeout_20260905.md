# M-star interpolated (m,T) seed v1 Garching closeout

Date: 2026-09-05

## Outcome

- Terminal marker: `SCALE_POOL_EXHAUSTED`
- Candidates attempted: 58/58 same-track dwarf failures
- New full-gate eligible dwarfs: 6
- Combined v1r2 + interpolation dwarf eligible count: 32
- Target of 24 additional dwarfs: not reached
- The last four writes (all 3100 K, ΔT 300–700 K extrapolations) all failed.

## Gated dwarfs

| candidate | Teff K | nearest-donor ΔT K | hull | primary iters |
| --- | ---: | ---: | --- | ---: |
| `g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3500` | 3500 | 100 | outside | 38 |
| `g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3300` | 3300 | 300 | outside | 28 |
| `g+4.50_m-0.50_a+0.00_c+0.00_x1.00_t3800` | 3800 | 100 | outside | 20 |
| `g+4.50_m-0.50_a+0.00_c+0.00_x1.00_t3750` | 3750 | 150 | outside | 32 |
| `g+4.50_m-0.50_a+0.00_c+0.00_x1.00_t3700` | 3700 | 200 | outside | 21 |
| `g+5.00_m+0.00_a+0.00_c+0.00_x1.00_t3900` | 3900 | 100 | inside | 14 |

Five of six are outside the donor convex hull (one-sided copies); the 3300 K
pass is a one-sided extrapolation success, not interpolation coverage of
3000–3300 K.

## Per-track state after this arm

| track | gated (this arm) | still failed, warm to cold |
| --- | --- | --- |
| log g=4.5, [M/H]=0 | 3500, 3300 | 3400, 3200, 3100, 3000 |
| log g=4.5, [M/H]=−0.5 | 3800, 3750, 3700 | 3600, 3500, 3400, 3300, 3200, 3100, 3000 |
| log g=5.0, [M/H]=0 | 3900 | 3700–3000 |
| log g=4.5, [M/H]=−1; log g=4.5, [M/H]=+0.5; log g=5.0, [M/H]=±0.5; log g=5.5, [M/H]=+0.5 | — | unchanged from v1r2 |

3400 K on the rich dwarf track failed between two gated neighbours (3500 and
3300) — an isolated jump that did not land in the basin, not a missing seed.

## Probe bins

- Easy probes (ΔT ≤ 100 K): 2/3 eligible (`t3500` rich, `t3800` metal-poor),
  which triggered the scale rollout per the preregistration.
- Hard probes (ΔT 200–300 K): 0/3 eligible.
- In the scale rollout the hard bin still recovered two rows (`t3300` rich at
  ΔT 300, `t3700` metal-poor at ΔT 200); every other same-track failure
  failed, and 3000–3200 K stays almost entirely closed.

The 3500 K (rich) and 3800 K (metal-poor) products are the seeds for the
small-step continuation arm; the 3300 K product is its cold backup.

## Frozen setup

- Donors: v1r2 gated dwarf `(m,T)` only, same track, log mix linear in
  log Teff; other four fields reconstructed by Payne-Zero.
- Solver and gates identical to v1r2: 60 iterations, all-layer relative
  temperature `5e-4`, imported frozen flux gate, strict self-restart, path
  consistency.
- Protocol hash: `6be8b7d4eeb0d96bdd00cccc7888340e6e3855773e16205a8d54c50a714336c3`
- Final status hash: `325fca2221c581ad421ce0b85bafc5525b3f5ef8d6375a44f6cce91c51434637`
- Flux gate hash: `ae0d384e9f2a0c97d5a3325275b3fc024e903c8aec61dd4857f5a49fe7bde69b`

## Retrieved artifacts

- Results: `results/m_star_interpolated_mt_seed_v1/`
- Log: `logs/mstar_interpolated_mt_seed_v1.log`
- Verified local versus Garching: 95 files, 98,317,436 bytes

The interpolation result does not by itself authorize emulator training,
production routing, Korg runs, or sealed-holdout use. It opened two dwarf
tracks that the continuation arm (`m_star_atlas_continuation_opened_tracks_v1`)
now probes one v1r2 node further.
