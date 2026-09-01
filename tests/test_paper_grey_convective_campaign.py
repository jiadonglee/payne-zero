"""Contract tests for the manuscript grey--convective campaign."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from experiments.analytic_initializer import run_paper_grey_convective_campaign as campaign
from experiments.analytic_initializer import recover_paper_grey_convective_replay


def test_campaign_is_versioned_away_from_historical_results() -> None:
    assert campaign.CAMPAIGN == "paper_grey_convective_20260829"
    assert campaign.ITERATIONS == 60
    assert campaign.PER_STAR_TIMEOUT_SECONDS == 900.0
    assert campaign.ARM == "hydrostatic_grey_convective"
    assert campaign.CAMPAIGN not in str(campaign.HISTORICAL_DEVELOPMENT)
    assert campaign.CAMPAIGN not in str(campaign.HISTORICAL_RESIDUAL)


def test_historical_replay_and_posthoc_thread_environments_are_distinct() -> None:
    assert campaign.HISTORICAL_REPLAY_ENV == {
        "NUMBA_THREADING_LAYER": "workqueue"
    }
    assert campaign.THREAD_ENV["NUMBA_NUM_THREADS"] == "1"
    assert "NUMBA_NUM_THREADS" in campaign.THREAD_LIMIT_KEYS


def test_campaign_uses_exact_frozen_samples() -> None:
    development, development_manifest = campaign._indices("development")
    posthoc, posthoc_manifest = campaign._indices("posthoc200")
    assert development_manifest == campaign.DEVELOPMENT_MANIFEST
    assert posthoc_manifest == campaign.POSTHOC_MANIFEST
    assert development.size == np.unique(development).size == 60
    assert posthoc.size == np.unique(posthoc).size == 200


def test_posthoc_manifest_is_described_as_opened_not_blind() -> None:
    source = inspect.getsource(campaign)
    assert "post-hoc evaluation on a previously opened sample" in source
    assert '"posthoc200_is_new_blind_test": False' in source


def test_solver_path_never_loads_an_atmosphere_checkpoint() -> None:
    source = inspect.getsource(campaign._solve_one)
    assert "emulator_warm_start_model" not in source
    assert "checkpoint" not in source
    assert "analytic_seed_model" in source
    assert "iterations_per_trial=ITERATIONS" in source


def test_profile_error_uses_dex_only_for_column_mass() -> None:
    truth = np.asarray([[1.0, 10.0]])
    prediction = np.asarray([[10.0, 100.0]])
    np.testing.assert_allclose(
        campaign._profile_error(prediction, truth, "column_mass"),
        np.ones_like(truth),
    )
    np.testing.assert_allclose(
        campaign._profile_error(prediction, truth, "temperature"),
        9.0 * np.ones_like(truth),
    )


def test_development_seed_artifact_matches_its_summary() -> None:
    root = Path(__file__).resolve().parents[1]
    summary_path = (
        root
        / "results"
        / campaign.CAMPAIGN
        / "development"
        / "seed_summary.json"
    )
    seeds_path = summary_path.with_name("seeds.npz")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["star_count"] == 60
    assert summary["finite_seed_count"] == 60
    assert summary["positive_seed_count"] == 60
    assert summary["seed_sha256"] == campaign.sha256(seeds_path)
    with np.load(seeds_path, allow_pickle=False) as seeds:
        assert seeds["corpus_indices"].shape == (60,)
        assert seeds["labels"].shape == (60, 5)
        assert seeds["column_mass"].shape == (60, 80)
        assert seeds["temperature"].shape == (60, 80)
        assert seeds["log_rosseland_opacity"].shape == (60, 80)


def test_recovery_imports_only_exact_matching_rows(
    tmp_path: Path, monkeypatch
) -> None:
    result_root = tmp_path / "results/formal"
    run_root = tmp_path / "runs/formal"
    paths = {
        "result": result_root,
        "seeds": result_root / "seeds.npz",
        "shards": run_root / "record_shards",
        "profiles": run_root / "profiles" / campaign.ARM,
        "products": run_root / "products" / campaign.ARM,
    }
    result_root.mkdir(parents=True)
    np.savez_compressed(
        paths["seeds"],
        labels=np.zeros((3, len(campaign.LABEL_FIELDS)), dtype=np.float64),
    )
    monkeypatch.setattr(
        recover_paper_grey_convective_replay.campaign,
        "_write_seeds",
        lambda root, sample: (np.asarray([10, 20, 30]), paths),
    )
    monkeypatch.setattr(
        recover_paper_grey_convective_replay.campaign,
        "HISTORICAL_DEVELOPMENT",
        Path("history.json"),
    )
    (tmp_path / "history.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "corpus_index": 10,
                        "converged": True,
                        "iterations_completed": 3,
                    },
                    {
                        "corpus_index": 20,
                        "converged": False,
                        "iterations_completed": 60,
                    },
                    {
                        "corpus_index": 30,
                        "converged": True,
                        "iterations_completed": 4,
                    },
                ]
            }
        )
    )
    diagnostic_result = tmp_path / "results/diagnostic"
    diagnostic_run = tmp_path / "runs/diagnostic"
    shards = diagnostic_run / "record_shards"
    profiles = diagnostic_run / "profiles" / campaign.ARM
    products = diagnostic_run / "products" / campaign.ARM
    for directory in (diagnostic_result, shards, profiles, products):
        directory.mkdir(parents=True)
    slug = "accepted"
    np.savez_compressed(profiles / f"{slug}.npz", value=np.asarray([1.0]))
    np.savez_compressed(products / f"{slug}.npz", value=np.asarray([2.0]))
    (shards / "shard_00.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "corpus_index": 10,
                        "position": 2,
                        "slug": slug,
                        "converged": True,
                        "iterations_completed": 3,
                    }
                ),
                json.dumps(
                    {
                        "corpus_index": 20,
                        "position": 0,
                        "slug": "timeout",
                        "converged": False,
                        "iterations_completed": None,
                        "solver_outcome": "timeout",
                    }
                ),
            ]
        )
        + "\n"
    )

    recovered = recover_paper_grey_convective_replay.recover(
        tmp_path, diagnostic_result, diagnostic_run
    )

    assert recovered["imported_row_count"] == 1
    assert recovered["remaining_row_count"] == 2
    assert [row["corpus_index"] for row in recovered["rejected_rows"]] == [20]
    imported = [
        json.loads(line)
        for line in paths["shards"].joinpath("shard_00.jsonl").read_text().splitlines()
    ]
    assert imported[0]["corpus_index"] == 10
    assert imported[0]["position"] == 0
    assert paths["profiles"].joinpath(f"{slug}.npz").is_file()
    assert paths["products"].joinpath(f"{slug}.npz").is_file()
