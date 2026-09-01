"""Preregistered verdict for the v4r1 molecule-on/off continuum ablation."""

from __future__ import annotations

from typing import Mapping


PRIMARY_LAYER = ("3200_4000K", 3200.0, 4000.0)
CONTROL_LAYER = ("4000_5000K", 4000.0, 5000.0)
ATOMIC_ALIGNED_P50_MAX_DEX = 0.10
MOLECULAR_EFFECT_MEDIAN_MIN_DEX = 0.05
ATOMIC_IR_P95_MIN_DEX = 0.20
ATOMIC_IR_SIGNED_MEDIAN_MAX_DEX = -0.05


def decide_molecule_ablation(primary_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    """Return the preregistered molecule-ablation verdict.

    ``primary_metrics`` must contain ``molecular_effect``,
    ``v4r1_minus_atomic``, and ``v4r1_minus_molecular``, each with
    ``signed_median_dex``, ``p50_abs_dex``, and ``p95_abs_dex``.
    """

    molecular_effect = primary_metrics["molecular_effect"]
    atomic = primary_metrics["v4r1_minus_atomic"]
    molecular = primary_metrics["v4r1_minus_molecular"]
    atomic_aligned = bool(
        atomic["p50_abs_dex"] <= ATOMIC_ALIGNED_P50_MAX_DEX
        and atomic["p95_abs_dex"] < molecular["p95_abs_dex"]
    )
    atomic_ir_remains = bool(
        molecular_effect["signed_median_dex"] < MOLECULAR_EFFECT_MEDIAN_MIN_DEX
        or atomic["p95_abs_dex"] >= ATOMIC_IR_P95_MIN_DEX
        or atomic["signed_median_dex"] <= ATOMIC_IR_SIGNED_MEDIAN_MAX_DEX
    )
    if atomic_aligned and not atomic_ir_remains:
        verdict = "MOLECULAR_CONTINUUM_DOMINATES"
    elif atomic_ir_remains and not atomic_aligned:
        verdict = "ATOMIC_IR_REMAINS"
    elif atomic_aligned and atomic_ir_remains:
        verdict = "MIXED_MOLECULAR_PLUS_ATOMIC_IR"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "atomic_aligned": atomic_aligned,
        "atomic_ir_remains": atomic_ir_remains,
        "verdict": verdict,
        "thresholds": {
            "atomic_aligned_p50_max_dex": ATOMIC_ALIGNED_P50_MAX_DEX,
            "molecular_effect_median_min_dex": MOLECULAR_EFFECT_MEDIAN_MIN_DEX,
            "atomic_ir_p95_min_dex": ATOMIC_IR_P95_MIN_DEX,
            "atomic_ir_signed_median_max_dex": ATOMIC_IR_SIGNED_MEDIAN_MAX_DEX,
        },
    }
