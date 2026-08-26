from types import SimpleNamespace

import numpy as np

from bench.labels import StellarLabels
from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere
from reduced_state import restart


def _finite_atmosphere() -> ModelAtmosphere:
    layer = np.array([1.0, 2.0])
    return ModelAtmosphere(
        column_mass=layer,
        temperature=layer,
        gas_pressure=layer,
        electron_density=layer,
        rosseland_opacity=layer,
        radiative_acceleration=layer,
        microturbulence=layer,
        convective_flux=layer,
        convective_velocity=layer,
    )


def test_missing_requested_product_is_not_counted_as_converged(
    monkeypatch, tmp_path
):
    labels = StellarLabels(5777.0, 4.44, 0.0, 0.0, 2.0)
    monkeypatch.setattr(restart, "_solver_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        restart,
        "run_atmosphere_model",
        lambda config: SimpleNamespace(
            converged=True,
            iterations_completed=3,
            diagnostics={},
            atmosphere=_finite_atmosphere(),
        ),
    )

    record = restart.run_restart_trial(
        labels,
        initial_atmosphere=None,
        product_dir=tmp_path,
    )

    assert not record.converged
    assert not record.trials[0].converged
    assert any("no usable structured atmosphere" in w for w in record.warnings)


def test_nonfinite_atmosphere_is_not_counted_as_converged(monkeypatch):
    """The structural stop only checks a temperature layer window, so a
    solver-reported convergence can still hide a non-finite state elsewhere
    (e.g. gas pressure). This must be caught even with no product requested,
    independent of the file-existence proxy above."""

    labels = StellarLabels(5777.0, 4.44, 0.0, 0.0, 2.0)
    nonfinite_atmosphere = _finite_atmosphere()
    nonfinite_atmosphere.gas_pressure = np.array([1.0, np.nan])
    monkeypatch.setattr(restart, "_solver_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        restart,
        "run_atmosphere_model",
        lambda config: SimpleNamespace(
            converged=True,
            iterations_completed=3,
            diagnostics={},
            atmosphere=nonfinite_atmosphere,
        ),
    )

    record = restart.run_restart_trial(
        labels,
        initial_atmosphere=None,
        product_dir=None,
    )

    assert not record.converged
    assert not record.trials[0].converged
    assert any("non-finite values" in w for w in record.warnings)
