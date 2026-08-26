"""Restart the reference solver from a caller-supplied atmosphere.

``bench/run_reference.py``'s ``run_star`` always goes through the neural
initializer (``emulator_warm_start_model``) plus its jittered-retry policy.
Parts 2-3 need to restart from atmospheres built by
``reduced_state.reconstruct`` or ``reduced_state.resample`` instead. This
reuses the exact production ``AtmosphereConfig`` builder
(``bench.run_reference._solver_config``) and the exact
``StarRecord``/``TrialRecord`` schema, so ``bench.report.summarize()``
aggregates these runs without modification -- one less thing to
reimplement or get subtly wrong.

Unlike ``run_star``, there is no second-trial jitter here: a reconstructed
or resampled atmosphere has no natural "nearby" perturbation to retry with,
so every restart is exactly one trial. This is a deliberate, documented
deviation from the production policy, not an omission.
"""

from __future__ import annotations

import dataclasses
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere
from payne_zero_atmosphere.runner import run_atmosphere_model
from payne_zero_atmosphere.synthesis_bridge import save_product_structured_atmosphere

from bench.labels import StellarLabels
from bench.run_reference import (
    PRODUCTION_ITERATIONS_PER_TRIAL,
    StarRecord,
    TrialRecord,
    _as_plain,
    _solver_config,
)


def run_restart_trial(
    labels: StellarLabels,
    initial_atmosphere: ModelAtmosphere,
    *,
    iterations_per_trial: int | None = PRODUCTION_ITERATIONS_PER_TRIAL,
    minimum_iterations_before_convergence: int | None = None,
    maximum_all_layer_relative_temperature_change: float | None = None,
    write_unconverged_product: bool = False,
    source: str | None = None,
    product_dir=None,
    atmosphere_profile_dir=None,
) -> StarRecord:
    """Restart the production solver from ``initial_atmosphere`` and record it.

    ``source`` is an arbitrary tag (e.g. ``"reduced_state_reconstruction"``,
    ``"depth_resolution_n160"``, ``"full_truth_oracle"``) carried through as
    the sole trial's ``initializer_label`` slot so it survives to
    ``records.jsonl`` for provenance, even though it is not an initializer
    label in the production sense.

    ``product_dir`` writes the converged structured atmosphere to
    ``<product_dir>/<slug>.npz``. It defaults to off because the Part 2/3 runs
    only needed iteration counts, and the products are ~100 KB per star. It is
    required for the spectral gate, which has to synthesize from the converged
    product rather than from the diagnostics.
    """

    star_start = time.perf_counter()
    trial_start = time.perf_counter()
    if iterations_per_trial is None:
        iterations_per_trial = PRODUCTION_ITERATIONS_PER_TRIAL
    trials: list[TrialRecord] = []
    captured_warnings: list[str] = []

    product_path = None
    if product_dir is not None:
        product_dir = Path(product_dir)
        product_dir.mkdir(parents=True, exist_ok=True)
        product_path = product_dir / f"{labels.slug}.npz"

    atmosphere_profile_path = None
    if atmosphere_profile_dir is not None:
        atmosphere_profile_dir = Path(atmosphere_profile_dir)
        atmosphere_profile_dir.mkdir(parents=True, exist_ok=True)
        atmosphere_profile_path = atmosphere_profile_dir / f"{labels.slug}.npz"

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = _solver_config(
                initial_atmosphere,
                iterations_per_trial=iterations_per_trial,
                structured_atmosphere_path=product_path,
                debug_state_path=None,
            )
            if minimum_iterations_before_convergence is not None:
                minimum = int(minimum_iterations_before_convergence)
                if minimum < 1:
                    raise ValueError(
                        "minimum_iterations_before_convergence must be positive"
                    )
                config = dataclasses.replace(
                    config, minimum_iterations_before_convergence=minimum
                )
            if maximum_all_layer_relative_temperature_change is not None:
                maximum_all = float(maximum_all_layer_relative_temperature_change)
                if maximum_all <= 0.0:
                    raise ValueError(
                        "maximum_all_layer_relative_temperature_change must be positive"
                    )
                config = dataclasses.replace(
                    config,
                    maximum_all_layer_relative_temperature_change=maximum_all,
                )
            result = run_atmosphere_model(config)
        captured_warnings.extend(str(w.message) for w in caught)
    except Exception as exc:  # noqa: BLE001 - a failed restart is a data point
        trials.append(
            TrialRecord(
                trial_index=0,
                initializer_label={"source": source} if source else None,
                iterations_completed=0,
                converged=False,
                seconds=time.perf_counter() - trial_start,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        )
        return StarRecord(
            labels=labels,
            trials=trials,
            seconds=time.perf_counter() - star_start,
            warnings=captured_warnings,
        )

    trial_converged = bool(result.converged)
    if product_path is not None and not product_path.is_file():
        if write_unconverged_product:
            try:
                save_product_structured_atmosphere(
                    result.atmosphere,
                    product_path,
                    molecular_lines=True,
                    device="cpu",
                    dtype="float64",
                )
                captured_warnings.append(
                    "saved final non-converged atmosphere for diagnostic use"
                )
            except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
                captured_warnings.append(
                    f"diagnostic non-converged product was not written: {exc}"
                )
        else:
            captured_warnings.append(
                f"solver did not write the structured product to {product_path}"
            )
        if trial_converged and not product_path.is_file():
            # A finite temperature delta can satisfy the structural stop even
            # when another state field has become non-finite.  Such a result
            # cannot be serialized or used downstream, so a product-requested
            # validation run must count it as a failure.
            trial_converged = False
            captured_warnings.append(
                "solver reported convergence but produced no usable structured "
                "atmosphere; counted as not converged"
            )
    if trial_converged and atmosphere_profile_path is not None:
        temporary = atmosphere_profile_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            column_mass=np.asarray(result.atmosphere.column_mass, dtype=np.float64),
            temperature=np.asarray(result.atmosphere.temperature, dtype=np.float64),
            gas_pressure=np.asarray(result.atmosphere.gas_pressure, dtype=np.float64),
            electron_density=np.asarray(
                result.atmosphere.electron_density, dtype=np.float64
            ),
            rosseland_opacity=np.asarray(
                result.atmosphere.rosseland_opacity, dtype=np.float64
            ),
            radiative_acceleration=np.asarray(
                result.atmosphere.radiative_acceleration, dtype=np.float64
            ),
        )
        temporary.replace(atmosphere_profile_path)
    trials.append(
        TrialRecord(
            trial_index=0,
            initializer_label={"source": source} if source else None,
            iterations_completed=int(result.iterations_completed),
            converged=trial_converged,
            seconds=time.perf_counter() - trial_start,
            diagnostics=_as_plain(result.diagnostics),
        )
    )
    return StarRecord(
        labels=labels,
        trials=trials,
        seconds=time.perf_counter() - star_start,
        warnings=captured_warnings,
    )


def _worker(payload):
    """Process-pool entry point mirroring ``bench.run_reference._worker``."""

    labels, initial_atmosphere, options = payload
    return run_restart_trial(labels, initial_atmosphere, **options).as_json()


def run_many_restarts(
    items: list[tuple[StellarLabels, ModelAtmosphere]],
    *,
    workers: int = 1,
    out_path=None,
    **options,
) -> list[dict]:
    """Restart many (labels, atmosphere) pairs, streaming records like ``run_many``."""

    import json
    from pathlib import Path

    handle = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        handle = out_path.open("a")

    records: list[dict] = []
    executor = None
    try:
        if workers <= 1:
            iterator = (
                run_restart_trial(labels, atmosphere, **options).as_json()
                for labels, atmosphere in items
            )
        else:
            from concurrent.futures import ProcessPoolExecutor

            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(
                _worker,
                [(labels, atmosphere, options) for labels, atmosphere in items],
            )

        for index, record in enumerate(iterator, start=1):
            records.append(record)
            if handle is not None:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            print(
                f"[{index}/{len(items)}] {record['slug']} "
                f"converged={record['converged']} "
                f"iters={record['converging_trial_iterations']} "
                f"{record['seconds']:.1f}s",
                flush=True,
            )
    finally:
        if handle is not None:
            handle.close()
        if executor is not None:
            executor.shutdown()

    return records
