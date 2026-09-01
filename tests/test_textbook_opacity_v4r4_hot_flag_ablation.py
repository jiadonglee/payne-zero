"""Preregistered verdict tests for the v4r4 hot-layer flag ablation."""

from experiments.analytic_initializer.textbook_opacity_v4r4_hot_flag_verdict import (
    CONTROL_LAYER,
    FLAG_KNOCKOUTS,
    PRIMARY_LAYER,
    decide_hot_flag_ablation,
)


def _effects(signed: dict[int, float]) -> dict[int, dict[str, float]]:
    values = {flag: {"signed_median_dex": 0.01} for flag in FLAG_KNOCKOUTS}
    for flag, median in signed.items():
        values[int(flag)] = {"signed_median_dex": float(median)}
    return values


def _v4r3_minus_base(signed_median_dex: float = -0.30) -> dict[str, float]:
    return {"signed_median_dex": float(signed_median_dex)}


def test_hot_flag_ablation_constants_match_preregistration():
    assert PRIMARY_LAYER == ("at_least_15000K", 15000.0, float("inf"))
    assert CONTROL_LAYER == ("8000_15000K", 8000.0, 15000.0)
    assert FLAG_KNOCKOUTS == (0, 4, 5, 8, 9, 10)


def test_hydrogen_continuum_mismatch_is_exclusive():
    decision = decide_hot_flag_ablation(
        _effects({0: 0.20, 4: 0.08, 5: 0.02, 8: 0.04, 9: 0.07, 10: 0.06}),
        _v4r3_minus_base(),
    )
    assert decision["verdict"] == "HYDROGEN_CONTINUUM_MISMATCH"
    assert decision["identity_flag"] == 0


def test_helium_neutral_continuum_is_exclusive():
    decision = decide_hot_flag_ablation(
        _effects({0: 0.07, 4: 0.22, 5: 0.03, 8: 0.05, 9: 0.08, 10: 0.09}),
        _v4r3_minus_base(),
    )
    assert decision["verdict"] == "HELIUM_NEUTRAL_CONTINUUM"
    assert decision["identity_flag"] == 4


def test_hot_metal_continuum_uses_larger_of_flags_nine_and_ten():
    flag9 = decide_hot_flag_ablation(
        _effects({0: 0.06, 4: 0.05, 5: 0.02, 8: 0.04, 9: 0.21, 10: 0.11}),
        _v4r3_minus_base(),
    )
    assert flag9["verdict"] == "HOT_METAL_CONTINUUM"
    assert flag9["identity_flag"] == 9
    assert flag9["metal_effect_signed_median_dex"] == 0.21

    flag10 = decide_hot_flag_ablation(
        _effects({0: 0.06, 4: 0.05, 5: 0.02, 8: 0.04, 9: 0.11, 10: 0.21}),
        _v4r3_minus_base(),
    )
    assert flag10["verdict"] == "HOT_METAL_CONTINUUM"
    assert flag10["identity_flag"] == 10
    assert flag10["metal_effect_signed_median_dex"] == 0.21


def test_flag_five_null_bit_is_independent_of_identity():
    hydrogen_with_null = decide_hot_flag_ablation(
        _effects({0: 0.20, 4: 0.08, 5: 0.01, 8: 0.04, 9: 0.07, 10: 0.06}),
        _v4r3_minus_base(),
    )
    assert hydrogen_with_null["verdict"] == "HYDROGEN_CONTINUUM_MISMATCH"
    assert hydrogen_with_null["helium_ionized_confirmed_null"] is True

    hydrogen_without_null = decide_hot_flag_ablation(
        _effects({0: 0.20, 4: 0.08, 5: 0.06, 8: 0.04, 9: 0.07, 10: 0.06}),
        _v4r3_minus_base(),
    )
    assert hydrogen_without_null["verdict"] == "HYDROGEN_CONTINUUM_MISMATCH"
    assert hydrogen_without_null["helium_ionized_confirmed_null"] is False

    inconclusive_with_null = decide_hot_flag_ablation(
        _effects({0: 0.04, 4: 0.03, 5: 0.01, 8: 0.02, 9: 0.03, 10: 0.02}),
        _v4r3_minus_base(),
    )
    assert inconclusive_with_null["verdict"] == "INCONCLUSIVE"
    assert inconclusive_with_null["helium_ionized_confirmed_null"] is True
    assert inconclusive_with_null["identity_flag"] is None


def test_inconclusive_when_no_exclusive_identity():
    below_threshold = decide_hot_flag_ablation(
        _effects({0: 0.14, 4: 0.10, 5: 0.02, 8: 0.08, 9: 0.09, 10: 0.11}),
        _v4r3_minus_base(),
    )
    assert below_threshold["verdict"] == "INCONCLUSIVE"

    hydrogen_helium_tie = decide_hot_flag_ablation(
        _effects({0: 0.20, 4: 0.20, 5: 0.02, 8: 0.04, 9: 0.05, 10: 0.06}),
        _v4r3_minus_base(),
    )
    assert hydrogen_helium_tie["verdict"] == "INCONCLUSIVE"

    flag8_largest = decide_hot_flag_ablation(
        _effects({0: 0.08, 4: 0.07, 5: 0.02, 8: 0.25, 9: 0.09, 10: 0.10}),
        _v4r3_minus_base(),
    )
    assert flag8_largest["verdict"] == "INCONCLUSIVE"

    metal_tied_with_hydrogen = decide_hot_flag_ablation(
        _effects({0: 0.18, 4: 0.05, 5: 0.02, 8: 0.04, 9: 0.18, 10: 0.10}),
        _v4r3_minus_base(),
    )
    assert metal_tied_with_hydrogen["verdict"] == "INCONCLUSIVE"


def test_flag_five_threshold_is_strictly_below_0_05():
    at_threshold = decide_hot_flag_ablation(
        _effects({0: 0.20, 4: 0.08, 5: 0.05, 8: 0.04, 9: 0.07, 10: 0.06}),
        _v4r3_minus_base(),
    )
    assert at_threshold["verdict"] == "HYDROGEN_CONTINUUM_MISMATCH"
    assert at_threshold["helium_ionized_confirmed_null"] is False
