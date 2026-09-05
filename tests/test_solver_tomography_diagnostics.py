"""Per-iteration solver tomography diagnostics exposed on the iteration result.

Opt-in, like ``test_integration.py``: each solve costs ~1 min and ~8 GB.

    PAYNE_ZERO_RUN_SOLVER=1 python -m pytest tests/test_solver_tomography_diagnostics.py -q

These fields exist so the M-star iteration tomography can record, per
iteration: the raw (pre-heuristic, pre-damping) temperature correction, the
convective flux ratio, the superadiabatic gradient, and the per-layer
molecular-equilibrium Newton pass count / least-squares fallback flag. The
production solve path is unchanged; the fields ride along on
``SingleIterationResult``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from bench.run_reference import _solver_config
from bench.labels import StellarLabels
from payne_zero_atmosphere.runner import run_atmosphere_model
from payne_zero_atmosphere.warm_start import emulator_warm_start_model


pytestmark = pytest.mark.skipif(
    os.environ.get("PAYNE_ZERO_RUN_SOLVER") != "1",
    reason="set PAYNE_ZERO_RUN_SOLVER=1 to run the real solver (~1 min, ~8 GB)",
)

SOLAR = StellarLabels(
    effective_temperature=5777.0,
    log_surface_gravity=4.44,
    metallicity=0.0,
    alpha_enhancement=0.0,
    microturbulence_km_s=2.0,
)


def test_iteration_result_carries_tomography_diagnostics():
    warm_start, _deck = emulator_warm_start_model(**SOLAR.as_kwargs(), device="cpu")
    config = _solver_config(
        warm_start,
        iterations_per_trial=3,
        structured_atmosphere_path=None,
        debug_state_path=None,
    )

    captured: list = []

    def hook(iteration_index, setup, step):
        captured.append(step)
        return {"iteration": int(iteration_index)}

    result = run_atmosphere_model(config, after_iteration_hook=hook)
    assert len(captured) == result.iterations_completed

    layers = int(warm_start.temperature.size)
    for step in captured:
        timing = step.timing
        assert "maximum_abs_raw_relative_temperature_correction" in timing
        assert np.isfinite(
            timing["maximum_abs_raw_relative_temperature_correction"]
        )

        raw = step.raw_temperature_correction
        assert raw is not None and raw.shape == (layers,)
        assert np.all(np.isfinite(raw))

        assert step.flux_ratio is not None and step.flux_ratio.shape == (layers,)

        assert step.superadiabatic_gradient is not None
        assert step.superadiabatic_gradient.shape == (layers,)

        newton = step.molecular_newton_iterations
        assert newton is not None and newton.shape == (layers,)
        assert np.all(newton >= 1) and np.all(newton <= 200)

        lstsq = step.molecular_newton_used_lstsq
        assert lstsq is not None and lstsq.shape == (layers,)
        assert lstsq.dtype == np.bool_

    # Iteration 1 skips the sign/acceleration heuristics on every layer and
    # production damping is 1.0, so the raw correction must equal the applied
    # one exactly on the first iteration.
    applied_first = (
        captured[0].remapped.finalization.temperature_correction_result
    )
    np.testing.assert_allclose(
        captured[0].raw_temperature_correction,
        applied_first.temperature_correction,
        rtol=0.0,
        atol=0.0,
    )
