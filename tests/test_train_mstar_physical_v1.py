from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.reduced_state_emulator.train_mstar_physical_v1 import (
    _equal_group_moments,
    _train_one,
    fit_balanced_standardization,
    load_cool_corpus,
    validate_cool_corpus,
)


def _profiles(rows: int) -> tuple[np.ndarray, np.ndarray]:
    mass = np.geomspace(1.0e-6, 100.0, 80)
    temperature = np.linspace(2500.0, 6500.0, 80)
    return (
        np.repeat(mass[None, :], rows, axis=0),
        np.repeat(temperature[None, :], rows, axis=0),
    )


def test_equal_group_moments_do_not_let_large_group_dominate() -> None:
    existing = np.zeros((1000, 2))
    cool = np.full((10, 2), 10.0)
    mean, std = _equal_group_moments(existing, cool)
    np.testing.assert_allclose(mean, 5.0)
    np.testing.assert_allclose(std, 5.0, rtol=1.0e-10)


def test_balanced_standardization_accepts_positive_monotone_profiles() -> None:
    old_labels = np.array(
        [[5000.0, 4.5, 0.0, 0.0, 1.0], [4500.0, 2.0, 0.0, 0.0, 2.0]]
    )
    cool_labels = np.array(
        [[3500.0, 5.0, 0.0, 0.0, 1.0], [3300.0, 1.5, 0.0, 0.0, 2.0]]
    )
    old_mass, old_temperature = _profiles(2)
    cool_mass, cool_temperature = _profiles(2)
    standardization = fit_balanced_standardization(
        old_labels,
        old_mass,
        old_temperature,
        cool_labels,
        cool_mass,
        cool_temperature,
    )
    assert standardization.feature_mean.shape == (5,)
    assert standardization.log_temperature_ratio_mean.shape == (80,)
    assert np.all(standardization.feature_std > 0.0)
    assert np.all(standardization.log_mass_increment_std > 0.0)


def test_cool_corpus_rejects_sealed_rows(tmp_path: Path) -> None:
    mass, temperature = _profiles(1)
    path = tmp_path / "cool.npz"
    np.savez(
        path,
        labels=np.array([[3500.0, 5.0, 0.0, 0.0, 1.0]]),
        column_mass=mass,
        temperature=temperature,
        roles=np.array(["sealed"]),
        track_ids=np.array(["track"]),
        node_ids=np.array(["node"]),
        source_product_paths=np.array(["product.npz"]),
        protocol_hash=np.array(["a" * 64]),
        flux_gate_hash=np.array(["b" * 64]),
    )
    with pytest.raises(ValueError, match="sealed"):
        load_cool_corpus(path)


def test_cool_corpus_gate_requires_both_classes_in_both_open_roles() -> None:
    train_giant = np.repeat(
        [[3500.0, 1.5, 0.0, 0.0, 2.0]], 10, axis=0
    )
    train_dwarf = np.repeat(
        [[3500.0, 5.0, 0.0, 0.0, 1.0]], 10, axis=0
    )
    val_giant = np.repeat([[3600.0, 2.0, 0.0, 0.0, 2.0]], 3, axis=0)
    val_dwarf = np.repeat([[3600.0, 4.75, 0.0, 0.0, 1.0]], 3, axis=0)
    labels = np.vstack([train_giant, train_dwarf, val_giant, val_dwarf])
    roles = np.array(["train"] * 20 + ["validation"] * 6)
    result = validate_cool_corpus({"labels": labels, "roles": roles})
    assert result["passes"]

    result = validate_cool_corpus(
        {"labels": np.vstack([train_giant, val_giant]), "roles": np.array(["train"] * 10 + ["validation"] * 3)}
    )
    assert result["passes"] is False
    assert "cool_train_dwarf_count" in result["failures"]


def test_tiny_balanced_training_smoke() -> None:
    old_labels = np.array(
        [
            [4500.0 + 50.0 * i, 1.5 if i % 2 == 0 else 4.5, 0.0, 0.0, 1.5]
            for i in range(8)
        ]
    )
    cool_labels = np.array(
        [
            [3300.0 + 50.0 * i, 1.5 if i % 2 == 0 else 5.0, 0.0, 0.0, 1.5]
            for i in range(8)
        ]
    )
    old_mass, old_temperature = _profiles(8)
    cool_mass, cool_temperature = _profiles(8)
    existing = {
        "labels": old_labels,
        "column_mass": old_mass,
        "temperature": old_temperature,
    }
    cool = {
        "labels": cool_labels,
        "column_mass": cool_mass,
        "temperature": cool_temperature,
    }
    model, standardization, summary = _train_one(
        existing=existing,
        cool=cool,
        existing_train_index=np.arange(6),
        existing_validation_index=np.arange(6, 8),
        cool_train_index=np.arange(6),
        cool_validation_index=np.arange(6, 8),
        seed=7,
        width=8,
        depth=2,
        batch_size=4,
        learning_rate=1.0e-3,
        epochs=1,
        patience=0,
        dtype=torch.float64,
        device="cpu",
    )
    assert model.layer_count == 80
    assert standardization.feature_mean.shape == (5,)
    assert summary["epochs_completed"] == 1
    assert np.isfinite(summary["best_validation_loss"])
