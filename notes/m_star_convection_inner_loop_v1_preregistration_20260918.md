# Convective inner loop v1 — preregistration (2026-09-18)

COOLTLUSTY (Hubeny) inserts a convective-zone temperature correction
between global radiation iterations because a global step can push
convective layers into a thermodynamically inconsistent state and damage
the next energy-transport evaluation.  The same paper warns that, near
final convergence, the inner correction can cancel the global iteration
and oscillate.

This arm tests **solver order**, not mixing length, opacity, or the EOS.
It is an experiment on deep coupling.  It does not assume that convective
on/off chatter is the root cause; existing tomography does not prove that.

## Order

```
global radiative transfer
    → convective inner loop:
          trial temperature gradient
          → update T (P held)
          → recompute EOS + MLT convective flux
          → local mismatch H_rad + H_conv - H_target
    → global temperature correction / next radiation field
```

``H_rad`` is held during the inner loop.  The next global iteration
recomputes the radiation field.  The true residual is that later global
residual, not the inner-loop local mismatch.

## Hard limits

1. ``F_conv^{required} = σ T_eff^4 - F_rad`` is a target.  The stored
   convective flux is whatever the original MLT recomputes after the
   structure update.  Assigning the required value and calling that
   energy conservation is forbidden.
2. The convective-zone mask may be frozen while the inner loop runs.
   After the inner passes it is released and rebuilt from
   ``∇ - ∇_ad > 0``.  A radiative layer must not be painted convective
   to force convergence.

EOS, opacity tables, and ``α_MLT = 1.25`` stay at production values.
The config default is ``convection_zone_inner_loop_passes = 0``.

## Cases

Tomography A/B/C/D and the 3200 K continuation node.  One experimental
arm (S3): eight inner passes, frozen mask during the passes, mask
released after.  Production tomography remains the S0 reference; this
campaign does not re-run S0.

## Diagnostics (not success criteria)

- Local inner-loop mismatch versus the subsequent global flux error.
- Sign correlation of inner ``ΔT`` and the following global ``ΔT`` on
  convective layers in the last ten iterations (Hubeny cancellation).
- Initial versus released Schwarzschild mask.  A mask change is a
  diagnostic, not evidence that on/off chatter was the cause.

## Decision table

| Result | Next action |
|---|---|
| A/C stay eligible and B or D deep flux error falls without oscillation | Interesting numerical result. Still not a production default. |
| Inner ``ΔT`` and global ``ΔT`` systematically oppose near the end | Hubeny cancellation. Stop the arm; do not freeze the zone to hide it. |
| No material change versus production | Solver order is not the missing deep-coupling ingredient. |
| Eligibility of A/C is lost | Arm fails, even if a failing dwarf looks quieter. |

Do not relax a flux gate after seeing the arm.  Do not retrain.
