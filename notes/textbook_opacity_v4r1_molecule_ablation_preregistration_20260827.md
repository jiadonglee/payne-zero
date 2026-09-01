# Textbook opacity v4r1 molecule-ablation preregistration

Date: 2026-08-27

This is a development-only diagnostic. It does not change the v4 or v4r1
candidates, the production solver, the default initializer, or any sealed
holdout. The v4r1 `FAIL_STOP` record remains authoritative. No ODE ablation,
12-star smoke, or 60-star funnel is run.

## Question

v4r1 closed the cool photospheric electron-density defect. The remaining cool
gate miss is localized to `3200--4000 K` layers, where the Rosseland harmonic
mean is set by the infrared window below the H-minus threshold. The v4r1 D2
continuum comparison used production replay with lines off and molecules on.
v4r1 itself has no molecular term. The question is whether that remaining
`3200--4000 K` residual is molecular continuum or atomic infrared opacity.

## Fixed protocol

Use the same manifest-excluded development split seed `20260816` and the same
deterministic 20-star `Teff x [M/H]` reference grid as v4 sanity / v4r1 D2.
Replay the production continuum twice on each reference star:

- lines `IFOP(15)=0` and `IFOP(17)=0`;
- stride 16;
- no temperature iteration;
- no sealed rows;
- molecules on, then molecules off.

Evaluate historical v4r1 on the stored true `(T, P)` layers. Stored electron
density is not an input. Do not retune the `0.30 / 0.50 / 0.20` gates and do
not construct a v4r2 opacity law in this round.

The primary slice is every finite reference layer with
`3200 <= T < 4000 K`. The `4000--5000 K` slice is a control and does not
decide the verdict. Report `[M/H]` breakdowns, but do not change the rule
after seeing them.

## Registered residuals

```text
molecular_effect      = log10(kappa_molecules_on / kappa_molecules_off)
v4r1_minus_atomic     = log10(kappa_v4r1 / kappa_molecules_off)
v4r1_minus_molecular  = log10(kappa_v4r1 / kappa_molecules_on)
```

`v4r1_minus_molecular` is the historical D2 comparison. The algebraic identity
`v4r1_minus_atomic = v4r1_minus_molecular + molecular_effect` is a sanity
check, not a scientific result.

## Registered verdict

On the primary `3200--4000 K` slice:

1. `atomic_aligned` if `v4r1_minus_atomic` p50 absolute `<= 0.10` dex and its
   p95 absolute is strictly smaller than `v4r1_minus_molecular` p95 absolute.
2. `atomic_ir_remains` if `molecular_effect` signed median `< 0.05` dex, or
   `v4r1_minus_atomic` p95 absolute `>= 0.20` dex, or `v4r1_minus_atomic`
   signed median `<= -0.05` dex.

Then:

- `MOLECULAR_CONTINUUM_DOMINATES` if (1) holds and (2) does not;
- `ATOMIC_IR_REMAINS` if (2) holds and (1) does not;
- `MIXED_MOLECULAR_PLUS_ATOMIC_IR` if both hold;
- `INCONCLUSIVE` if neither holds.

A molecular verdict does not license adding H2O/CO/TiO inside the current
`T >= 3200 K` domain, nor silently raising that floor. Those are later
registered constructions. An atomic-IR verdict does not license a corpus fit
to John H-minus free-free.

The registered output is
`results/analytic_initializer/textbook_opacity_v4r1_molecule_ablation_20260827.json`.

## Post-run formal result

The diagnostic completed on 2026-08-27. The JSON SHA-256 is
`881c2cf9df7139c0bc190980a8c7e9ff89452ee63f63fae6964fc15adee73787`.
Line flags 15 and 17 stayed off, the molecules-on/off population flags matched
the requested replay, the log identity residual was `2.2e-16` dex, and the
production/solver/ODE/smoke/funnel/sealed/v4r2 boundaries remained closed.

The primary `3200--4000 K` slice has 199 layers, all of them on stars with
`Teff < 6000 K`:

| residual | signed median dex | p50 abs dex | p95 abs dex |
|---|---:|---:|---:|
| `molecular_effect` | -0.0035 | 0.0035 | 0.049 |
| `v4r1_minus_atomic` | -0.059 | 0.059 | 0.367 |
| `v4r1_minus_molecular` | -0.054 | 0.054 | 0.367 |

`atomic_aligned` is false. `atomic_ir_remains` is true because the atomic
comparison signed median is `-0.059 <= -0.05` and its p95 is `0.367 >= 0.20`.
The machine verdict is `ATOMIC_IR_REMAINS`.

The control `4000--5000 K` slice has a still smaller molecular effect
(signed median `-0.0008` dex, p95 `0.018` dex) while v4r1 remains low against
the atomic continuum (signed median `-0.065` dex, p95 `0.187` dex). Molecular
continuum is not the remaining cool-gate mechanism. The next registered stage
is an atomic infrared diagnosis of the H-minus-threshold window, not a
molecular term and not a silent domain change to `T >= 4000 K`.
