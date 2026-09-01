import numpy as np

from experiments.reduced_state_emulator.evaluate_mstar_candidate_v1 import (
    PROFILE_MASS_P95_DEX_LIMIT,
    PROFILE_TEMPERATURE_P95_LIMIT,
    profile_metrics,
)


def test_profile_metrics_match_candidate_gates() -> None:
    mass = np.repeat(np.geomspace(1.0e-6, 100.0, 80)[None, :], 4, axis=0)
    temperature = np.repeat(np.linspace(2500.0, 6500.0, 80)[None, :], 4, axis=0)
    predicted_mass = mass * 10.0 ** (0.5 * PROFILE_MASS_P95_DEX_LIMIT)
    predicted_temperature = temperature * (
        1.0 + 0.5 * PROFILE_TEMPERATURE_P95_LIMIT
    )
    metrics = profile_metrics(
        predicted_mass,
        predicted_temperature,
        mass,
        temperature,
    )
    assert metrics["temperature_relative"]["p95"] < PROFILE_TEMPERATURE_P95_LIMIT
    assert metrics["column_mass_dex"]["p95"] < PROFILE_MASS_P95_DEX_LIMIT
    assert metrics["monotonicity_violations"] == 0
