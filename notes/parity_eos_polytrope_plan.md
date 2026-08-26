# Parity + EOS polytrope validation record

Status: preregistered before the solver runs.

## Fixed method

The `parity_polytrope` arm starts from the 2407-float parity reduced state. It
runs one ordinary exact solver iteration, then applies one temperature-only
projection before iteration 2. The projection uses the post-iteration total
pressure and the solver's EOS adiabatic gradient. It selects layers 36 and
deeper where the solver's logarithmic temperature-pressure gradient is above
the adiabatic gradient. Each contiguous component is integrated with the
trapezoidal adiabatic relation and is continued below its lower boundary by a
constant log-temperature offset. No fitted constants, fixed gamma, damping,
transition width, or post-hoc threshold is allowed.

The first physical iteration remains part of the 15-iteration budget. The
production initializer and public default path are unchanged.

## Gates

Gate 0 requires the pure projection tests to cover constant and variable EOS
gradients, no crossing, multiple components, bottom-reaching convection,
continuity, invalid pressure/gradient rejection, finite positive output, and
the no-emulator import guard.

Gate 1 uses the frozen 12-star smoke set. Both parity and parity_polytrope must
finish without exception, nonfinite state, or timeout; the new arm may lose at
most one convergence and one median iteration relative to parity. At least
one star must trigger the projection and at least one star must be a no-op.

Gate 2 reuses the seed-20260817 60-star draw. The new arm must converge on at
least 52/60 stars, lose at most two stars to the same-run parity baseline,
have no exception, nonfinite state, or timeout, have median iterations at most
9, and have a common-star mean iteration difference no larger than +1.0. Each
predefined temperature regime may lose at most one convergence relative to
parity. The result may support only a stability claim, not superiority.

Failure vetoes this exact projection. Only clear implementation errors may be
fixed before rerunning; no damping, crossing retuning, fixed-gamma variant,
new fit, or threshold adjustment is permitted.
