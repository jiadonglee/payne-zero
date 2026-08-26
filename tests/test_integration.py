"""End-to-end tests that actually run the reference solver.

Opt-in. Each solve costs roughly 30-70 s and peaks near 8 GB resident, so these
are skipped unless ``PAYNE_ZERO_RUN_SOLVER=1`` is set, and they should not be
run concurrently with a benchmark sweep on a machine that cannot hold two
solver processes at once.

    PAYNE_ZERO_RUN_SOLVER=1 python -m pytest tests/test_integration.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from bench import run_reference
from bench.labels import StellarLabels


pytestmark = pytest.mark.skipif(
    os.environ.get("PAYNE_ZERO_RUN_SOLVER") != "1",
    reason="set PAYNE_ZERO_RUN_SOLVER=1 to run the real solver (~1 min, ~8 GB each)",
)


# A star that is known to converge quickly, to keep these affordable.
SOLAR = StellarLabels(
    effective_temperature=5777.0,
    log_surface_gravity=4.44,
    metallicity=0.0,
    alpha_enhancement=0.0,
    microturbulence_km_s=2.0,
)


@pytest.fixture(scope="module")
def solar_record():
    return run_reference.run_star(SOLAR)


def test_solar_star_converges(solar_record):
    assert solar_record.converged
    assert not solar_record.needed_retry
    assert solar_record.converging_trial_iterations >= 3, "the production floor"
    assert solar_record.converging_trial_iterations <= 15


def test_record_carries_per_iteration_diagnostics(solar_record):
    diagnostics = solar_record.trials[0].diagnostics
    timings = diagnostics["iteration_timings"]
    assert len(timings) == solar_record.converging_trial_iterations
    for step in timings:
        for key in (
            "deep_layer_relative_temperature_change",
            "p95_absolute_flux_error_percent",
            "opacity_seconds",
            "total_seconds",
        ):
            assert key in step
    final = timings[-1]["deep_layer_relative_temperature_change"]
    assert final < 5e-4, "the converged iteration must satisfy the stopping rule"


def test_record_is_standard_json(solar_record):
    """No bare NaN or Infinity tokens, so jq and pandas can read the output."""

    encoded = json.dumps(solar_record.as_json(), allow_nan=False)
    assert json.loads(encoded)["converged"] is True


def test_run_star_is_deterministic():
    first = run_reference.run_star(SOLAR)
    second = run_reference.run_star(SOLAR)
    assert (
        first.converging_trial_iterations == second.converging_trial_iterations
    ), "the same labels must give the same iteration count"


def test_traces_are_written(tmp_path):
    record = run_reference.run_star(SOLAR, trace_dir=tmp_path)
    assert record.converged
    debug = tmp_path / SOLAR.slug / "trial_00" / "debug_state.npz"
    assert debug.is_file(), "--traces must dump the runner's per-iteration state"
    with np.load(debug, allow_pickle=False) as arrays:
        assert len(arrays.files) > 0


def test_keep_product_writes_the_structured_atmosphere(tmp_path):
    record = run_reference.run_star(SOLAR, trace_dir=tmp_path, keep_product=True)
    assert record.converged
    product = (
        tmp_path / SOLAR.slug / "trial_00" / "payne_zero_structured_atmosphere.npz"
    )
    assert product.is_file()
    with np.load(product, allow_pickle=False) as arrays:
        assert arrays["temperature"].shape == (80,)
        assert (arrays["temperature"] > 0).all()


def test_a_star_outside_support_is_recorded_not_raised():
    """A failed star is a data point; the sweep must not die on it."""

    impossible = StellarLabels(
        effective_temperature=100000.0,  # far outside initializer support
        log_surface_gravity=4.44,
        metallicity=0.0,
        alpha_enhancement=0.0,
        microturbulence_km_s=2.0,
    )
    record = run_reference.run_star(impossible, max_trials=1)
    assert not record.converged
    assert record.trials[0].error is not None
    json.dumps(record.as_json(), allow_nan=False)


def test_run_many_with_two_workers(tmp_path):
    """Exercises the ProcessPoolExecutor path; needs ~16 GB for two workers."""

    if os.environ.get("PAYNE_ZERO_RUN_SOLVER_PARALLEL") != "1":
        pytest.skip("set PAYNE_ZERO_RUN_SOLVER_PARALLEL=1; needs ~16 GB")

    labels = [
        SOLAR,
        StellarLabels(4500.0, 2.0, -1.0, 0.3, 1.5),
    ]
    out = tmp_path / "records.jsonl"
    records = run_reference.run_many(labels, workers=2, out_path=out)
    assert len(records) == 2
    written = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(written) == 2
    assert {r["slug"] for r in written} == {item.slug for item in labels}


def test_cli_end_to_end(tmp_path):
    """The documented invocation, on one star."""

    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(json.dumps(SOLAR.as_kwargs()) + "\n")
    out_dir = tmp_path / "run"

    assert run_reference.main(["--labels", str(labels_path), "--out", str(out_dir)]) == 0

    assert (out_dir / "labels.jsonl").is_file()
    assert (out_dir / "run_config.json").is_file()
    config = json.loads((out_dir / "run_config.json").read_text())
    assert config["numba_threading_layer"] == "workqueue"
    assert config["production_policy"]["minimum_iterations_before_convergence"] == 3

    records = [
        json.loads(line)
        for line in (out_dir / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 1 and records[0]["converged"]

    from bench import report

    summary = report.summarize(records)
    assert summary["star_count"] == 1
    report.format_report(summary)


def test_torch_initializer_reproduces_the_solver_input(tmp_path):
    """The deck the solver actually consumes must match the torch prediction.

    This is the seam between ``payne_zero_diffatm`` and the reference path: the
    torch initializer is only a valid starting point for training if, after
    deck quantization, it lands on the same atmosphere the solver would have
    been given.
    """

    import torch

    from payne_zero_atmosphere.warm_start import emulator_warm_start_model
    from payne_zero_diffatm.initializer import DifferentiableInitializer
    from payne_zero_diffatm.quantize import quantize_prediction

    warm_start, _deck = emulator_warm_start_model(**SOLAR.as_kwargs(), device="cpu")

    model = DifferentiableInitializer.from_checkpoint(network_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predicted = quantize_prediction(
            model(
                {
                    key: torch.tensor([value], dtype=torch.float64)
                    for key, value in SOLAR.as_kwargs().items()
                }
            ),
            straight_through=False,
        )

    for field, expected in (
        ("temperature", warm_start.temperature),
        ("gas_pressure", warm_start.gas_pressure),
        ("electron_density", warm_start.electron_density),
        ("rosseland_opacity", warm_start.rosseland_opacity),
        ("column_mass", warm_start.column_mass),
    ):
        np.testing.assert_allclose(
            getattr(predicted, field)[0].numpy(),
            np.asarray(expected, dtype=np.float64),
            rtol=1e-12,
            err_msg=f"{field} does not match the deck the solver receives",
        )
