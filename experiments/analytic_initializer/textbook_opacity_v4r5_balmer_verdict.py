"""Preregistered verdict for the v4r5 Balmer-window hydrogen bf diagnostic."""

from __future__ import annotations

from typing import Mapping


PRIMARY_LAYER = ("8000_15000K", 8000.0, 15000.0)
HOT_LAYER = ("at_least_30000K", 30000.0, float("inf"))

LYMAN_EDGE_CM2 = 6.30e-18
BALMER_LITERATURE_EDGE_CM2 = 1.40e-17
V4_N2_EDGE_CM2 = LYMAN_EDGE_CM2 * 4.0
N2_EDGE_SCALE = BALMER_LITERATURE_EDGE_CM2 / V4_N2_EDGE_CM2

IDENTITY_MIN_DEX = 0.15
HIGH_N_MIN_DEX = 0.10
LEAK_MIN_DEX = 0.10
STIMULATED_MIN_DEX = 0.10
HOT_STABILITY_DEX = 0.03
N2_CARRIER_MIN_CONTROL_DEX = 0.10
N2_ONLY_HOT_MAX_DEX = -0.30
KARZAS_SHAPE_SPAN_MAX_DEX = 0.08

VERDICTS = (
    "BALMER_EDGE_CROSS_SECTION",
    "BALMER_GAUNT_SHAPE",
    "HIGH_N_OVERCOUNT",
    "NLTE_OR_STIMULATED",
    "NOT_HYDROGEN_BF",
    "INCONCLUSIVE",
)


def _signed(metrics: Mapping[str, float] | None) -> float | None:
    if metrics is None:
        return None
    value = metrics.get("signed_median_dex")
    if value is None:
        return None
    number = float(value)
    if number != number:  # NaN
        return None
    return number


def _delta(baseline: float | None, variant: float | None) -> float | None:
    if baseline is None or variant is None:
        return None
    return float(baseline) - float(variant)


def n2_edge_cross_section_cm2(
    principal_quantum_number: int,
    *,
    n2_edge_cm2: float | None = None,
) -> float:
    """Return the diagnostic n-shell threshold cross section.

    Lyman ``n=1`` is always ``6.30e-18``.  When ``n2_edge_cm2`` is set, only
    ``n=2`` uses that value; ``n>=3`` keep the frozen v4 ``n^2`` law.
    """

    level = int(principal_quantum_number)
    if level < 1:
        raise ValueError("principal quantum number must be >= 1")
    if level == 2 and n2_edge_cm2 is not None:
        return float(n2_edge_cm2)
    return float(LYMAN_EDGE_CM2 * (level**2))


def decide_balmer_diagnostics(
    *,
    control: Mapping[str, Mapping[str, float]],
    hot: Mapping[str, Mapping[str, float]],
    karzas_n2_ratio_span_dex: float | None = None,
) -> dict[str, object]:
    """Return the preregistered Balmer-window identity verdict."""

    control_v4r5 = _signed(control.get("v4r5"))
    hot_v4r5 = _signed(hot.get("v4r5"))
    deltas = {
        name: _delta(control_v4r5, _signed(control.get(name)))
        for name in (
            "n2_only",
            "n2_balmer_edge",
            "drop_n_ge_3",
            "drop_n_ge_7",
            "no_stimulated",
            "no_hminus",
            "no_hi_ff",
            "n2_karzas_shape",
            "n2_karzas_full",
        )
    }
    hot_changes = {
        name: (
            None
            if hot_v4r5 is None or _signed(hot.get(name)) is None
            else float(_signed(hot.get(name))) - float(hot_v4r5)
        )
        for name in deltas
    }
    n2_only_control = _signed(control.get("n2_only"))
    n2_only_hot = _signed(hot.get("n2_only"))
    n2_is_carrier = bool(
        n2_only_control is not None and n2_only_control >= N2_CARRIER_MIN_CONTROL_DEX
    )
    lyman_owns_hot_tail = bool(
        n2_only_hot is not None and n2_only_hot <= N2_ONLY_HOT_MAX_DEX
    )
    edge_delta = deltas["n2_balmer_edge"]
    edge_hot_change = hot_changes["n2_balmer_edge"]
    edge_hot_stable = bool(
        edge_hot_change is not None and abs(edge_hot_change) < HOT_STABILITY_DEX
    )
    shape_delta = deltas["n2_karzas_shape"]
    shape_moves = bool(shape_delta is not None and shape_delta >= IDENTITY_MIN_DEX)
    edge_moves = bool(edge_delta is not None and edge_delta >= IDENTITY_MIN_DEX)
    edge_weak = bool(edge_delta is None or edge_delta < IDENTITY_MIN_DEX)
    high_n_delta = max(
        (value for value in (deltas["drop_n_ge_3"], deltas["drop_n_ge_7"]) if value is not None),
        default=None,
    )
    leak_name = None
    leak_delta = None
    for name in ("no_hminus", "no_hi_ff"):
        value = deltas[name]
        if value is not None and (leak_delta is None or value > leak_delta):
            leak_delta = value
            leak_name = name
    leftover = None
    if _signed(control.get("n2_balmer_edge")) is not None:
        leftover = float(_signed(control.get("n2_balmer_edge")))
    h1_core = bool(edge_moves and edge_hot_stable and n2_is_carrier)
    shape_allows_h1 = bool(
        karzas_n2_ratio_span_dex is None
        or float(karzas_n2_ratio_span_dex) < KARZAS_SHAPE_SPAN_MAX_DEX
        or (shape_delta is not None and shape_delta < HIGH_N_MIN_DEX)
    )

    if (
        leak_delta is not None
        and leak_delta >= LEAK_MIN_DEX
        and (edge_delta is None or edge_delta < LEAK_MIN_DEX)
    ):
        verdict = "NOT_HYDROGEN_BF"
        component = "hminus" if leak_name == "no_hminus" else "hydrogen_freefree"
    elif (
        deltas["no_stimulated"] is not None
        and deltas["no_stimulated"] >= STIMULATED_MIN_DEX
        and (edge_delta is None or edge_delta < STIMULATED_MIN_DEX)
    ):
        verdict = "NLTE_OR_STIMULATED"
        component = "stimulated_emission"
    elif (
        high_n_delta is not None
        and high_n_delta >= HIGH_N_MIN_DEX
        and (edge_delta is None or edge_delta < HIGH_N_MIN_DEX)
    ):
        drop7 = deltas["drop_n_ge_7"]
        drop3 = deltas["drop_n_ge_3"]
        mostly_n_ge_7 = bool(
            drop7 is not None
            and drop3 is not None
            and drop7 >= 0.5 * high_n_delta
            and (drop3 - drop7) < 0.05
        )
        verdict = "HIGH_N_OVERCOUNT"
        component = "n_ge_7" if mostly_n_ge_7 else "n_ge_3"
    elif h1_core and shape_allows_h1:
        verdict = "BALMER_EDGE_CROSS_SECTION"
        component = "n2_edge_1.40e-17"
    elif h1_core and not shape_allows_h1:
        verdict = "BALMER_GAUNT_SHAPE"
        component = "n2_frequency_law"
    elif shape_moves and edge_weak:
        verdict = "BALMER_GAUNT_SHAPE"
        component = "n2_frequency_law"
    else:
        verdict = "INCONCLUSIVE"
        component = None

    recommended = None
    if verdict == "BALMER_EDGE_CROSS_SECTION":
        recommended = {
            "n1_edge_cm2": LYMAN_EDGE_CM2,
            "n2_edge_cm2": BALMER_LITERATURE_EDGE_CM2,
            "n_ge_3_edge_law": "n_squared_times_lyman_edge",
            "frequency_law": "nu_to_the_minus_3",
            "populations": "v4r5_ground_anchored",
            "karzas_table_load": False,
            "global_edge_power_n1": False,
        }

    return {
        "verdict": verdict,
        "component": component,
        "control_v4r5_signed_median_dex": control_v4r5,
        "hot_v4r5_signed_median_dex": hot_v4r5,
        "n2_balmer_edge_leftover_dex": leftover,
        "n2_balmer_edge_closed_dex": edge_delta,
        "deltas_dex": deltas,
        "hot_changes_dex": hot_changes,
        "n2_is_primary_carrier": n2_is_carrier,
        "lyman_owns_hot_tail": lyman_owns_hot_tail,
        "edge_hot_stable": edge_hot_stable,
        "karzas_n2_ratio_span_dex": karzas_n2_ratio_span_dex,
        "recommended_construction": recommended,
        "thresholds": {
            "identity_min_dex": IDENTITY_MIN_DEX,
            "high_n_min_dex": HIGH_N_MIN_DEX,
            "leak_min_dex": LEAK_MIN_DEX,
            "stimulated_min_dex": STIMULATED_MIN_DEX,
            "hot_stability_dex": HOT_STABILITY_DEX,
            "n2_carrier_min_control_dex": N2_CARRIER_MIN_CONTROL_DEX,
            "n2_only_hot_max_dex": N2_ONLY_HOT_MAX_DEX,
            "karzas_shape_span_max_dex": KARZAS_SHAPE_SPAN_MAX_DEX,
        },
    }
