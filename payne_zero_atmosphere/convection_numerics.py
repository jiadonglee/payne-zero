"""Named finite-difference scales used by convection.

Kept in a tiny module so evidence-chain tests can import the numbers without
loading Numba or the continuum kernels.
"""

import numpy as np

# Finite-difference samples in the runner are taken at relative steps of
# ±0.001.  A central difference over that span is
# (f+ - f-) / (2 * 0.001 x) = 500 * (f+ - f-) / x.
CONVECTION_FINITE_DIFFERENCE_RELATIVE_STEP = 1.0e-3
CONVECTION_FINITE_DIFFERENCE_CENTER_WEIGHT = 1.0 / (
    2.0 * CONVECTION_FINITE_DIFFERENCE_RELATIVE_STEP
)

# Diagnostic scan for the TLUSTY-style derivative check.  Production convection
# still uses the named 1e-3 step above; these values are not solver defaults.
THERMODYNAMIC_FINITE_DIFFERENCE_RELATIVE_STEPS = (
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
)


def convection_finite_difference_center_weight(relative_step: float) -> float:
    """Central-difference weight for samples taken at ``x * (1 ± relative_step)``."""

    step = float(relative_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    return 1.0 / (2.0 * step)
