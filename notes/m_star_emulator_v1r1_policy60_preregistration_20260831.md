# M-star emulator v1r1 policy60 preregistration

## Purpose

This is a versioned follow-up to the immutable
`m_star_emulator_v1` `FAIL_STOP`. It tests whether the 30-iteration ceiling,
rather than the initializer basin, caused recoverable open-track failures.

## Frozen comparison

- Reuse the exact v1 complete-track train/validation/sealed split.
- Do not run either sealed M-star track.
- Reuse the exact v1 flux thresholds; do not refit them from policy60 results.
- Keep the exact v1 finite/positive/monotonic, primary/self-restart, flux, and
  path-consistency truth-admission gates.
- Keep the production solver stopping rule.
- Change only the per-attempt iteration ceiling from 30 to 60.
- Recompute the opened 4000/4500 K references under policy60 so a reference
  that hit the old cap may recover, but judge it with the imported v1 flux gate.
- An ineligible step still blocks every cooler node on that track.

The v1 results and corpus remain immutable. Policy60 products go under
`results/m_star_emulator_v1r1_policy60/`.

## Strict-settling diagnostic

The v1 `log g=4.5` and `5.0` training dwarfs already reached the production
stopping rule at 4000 K but failed primary/restart column-mass consistency.
They therefore cannot be diagnosed by merely raising the iteration cap.

A separate diagnostic reruns only those two 4000 K nodes with:

- a 60-iteration ceiling;
- the unchanged deep-layer threshold;
- an additional all-layer relative-temperature-change threshold of `5e-4`;
- the unchanged v1 flux and path-consistency thresholds.

These diagnostic products go under
`results/m_star_emulator_v1r1_settling_diagnostic/` and are never admitted to
the policy60 training corpus. Any use in a future training campaign requires a
new version and preregistration.

## Downstream gate

Build the policy60 cool corpus only from opened train/validation rows that pass
all frozen gates. Emulator training may start only if the existing preflight
still passes: at least 20 cool train rows, at least six opened-validation rows,
at least five train rows per luminosity class, and at least two validation rows
per luminosity class.

If training starts, candidate solver validation retains the existing
30-iteration production policy. Policy60 is a truth-bootstrap experiment, not
a production iteration-policy change.

No threshold relaxation, post-hoc track reassignment, production routing
change, existing sealed-holdout opening, Korg run, or general
`Teff < 4000 K` support claim is authorized.
