"""Preregistered verdict for the v4r4 hot-layer production-flag ablation."""

from __future__ import annotations

from typing import Mapping


PRIMARY_LAYER = ("at_least_15000K", 15000.0, float("inf"))
CONTROL_LAYER = ("8000_15000K", 8000.0, 15000.0)
FLAG_KNOCKOUTS = (0, 4, 5, 8, 9, 10)
EXCLUSIVE_IDENTITY_MIN_DEX = 0.15
HELIUM_IONIZED_NULL_MAX_DEX = 0.05


def _signed_median(metrics: Mapping[str, float]) -> float:
    return float(metrics["signed_median_dex"])


def decide_hot_flag_ablation(
    primary_effects: dict[int, dict],
    v4r3_minus_base: dict,
) -> dict:
    """Return the preregistered hot-layer flag-ablation verdict.

    ``primary_effects`` maps knockout flag indices to ``_metrics``-like dicts
    with at least ``signed_median_dex``. ``v4r3_minus_base`` is the same shape
    for ``log10(kappa_v4r3 / kappa_baseline)`` on the primary slice.
    """

    signed = {
        int(flag): _signed_median(primary_effects[flag]) for flag in FLAG_KNOCKOUTS
    }
    metal_effect = max(signed[9], signed[10])
    hydrogen_exclusive = bool(
        signed[0] >= EXCLUSIVE_IDENTITY_MIN_DEX
        and all(signed[0] > signed[flag] for flag in FLAG_KNOCKOUTS if flag != 0)
    )
    helium_neutral_exclusive = bool(
        signed[4] >= EXCLUSIVE_IDENTITY_MIN_DEX
        and all(signed[4] > signed[flag] for flag in (0, 5, 8, 9, 10))
    )
    hot_metal_exclusive = bool(
        metal_effect >= EXCLUSIVE_IDENTITY_MIN_DEX
        and all(metal_effect > signed[flag] for flag in (0, 4, 5, 8))
    )
    if hydrogen_exclusive:
        verdict = "HYDROGEN_CONTINUUM_MISMATCH"
        identity_flag: int | None = 0
    elif helium_neutral_exclusive:
        verdict = "HELIUM_NEUTRAL_CONTINUUM"
        identity_flag = 4
    elif hot_metal_exclusive:
        verdict = "HOT_METAL_CONTINUUM"
        identity_flag = 9 if signed[9] >= signed[10] else 10
    else:
        verdict = "INCONCLUSIVE"
        identity_flag = None
    return {
        "verdict": verdict,
        "identity_flag": identity_flag,
        "helium_ionized_confirmed_null": bool(
            signed[5] < HELIUM_IONIZED_NULL_MAX_DEX
        ),
        "hydrogen_exclusive": hydrogen_exclusive,
        "helium_neutral_exclusive": helium_neutral_exclusive,
        "hot_metal_exclusive": hot_metal_exclusive,
        "metal_effect_signed_median_dex": float(metal_effect),
        "flag_effects_signed_median_dex": {
            flag: float(value) for flag, value in signed.items()
        },
        "v4r3_minus_base_signed_median_dex": _signed_median(v4r3_minus_base),
        "thresholds": {
            "exclusive_identity_min_dex": EXCLUSIVE_IDENTITY_MIN_DEX,
            "helium_ionized_null_max_dex": HELIUM_IONIZED_NULL_MAX_DEX,
        },
    }
