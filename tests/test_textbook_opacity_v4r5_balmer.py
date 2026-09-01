"""Tiny tests for the v4r5 Balmer-window diagnostic helpers."""

import numpy as np

from experiments.analytic_initializer.run_textbook_opacity_v4r5_balmer_diagnostics import (
    hydrogen_boundfree_from_level_populations_local,
)
from experiments.analytic_initializer.textbook_opacity import (
    DEFAULT_TEXTBOOK_CONSTANTS,
    _hydrogen_ground_anchored_level_populations,
    saha_electron_diagnostics_v4r3,
    textbook_opacity_node_components_v4r5,
)
from experiments.analytic_initializer.textbook_opacity_v4r5_balmer_verdict import (
    BALMER_LITERATURE_EDGE_CM2,
    HOT_LAYER,
    LYMAN_EDGE_CM2,
    N2_EDGE_SCALE,
    PRIMARY_LAYER,
    V4_N2_EDGE_CM2,
    decide_balmer_diagnostics,
    n2_edge_cross_section_cm2,
)


def _signed(value: float) -> dict[str, float]:
    return {"signed_median_dex": float(value)}


def test_n2_edge_helper_isolates_balmer_from_lyman():
    assert PRIMARY_LAYER == ("8000_15000K", 8000.0, 15000.0)
    assert HOT_LAYER[1] == 30000.0
    assert n2_edge_cross_section_cm2(1) == LYMAN_EDGE_CM2
    assert n2_edge_cross_section_cm2(2) == V4_N2_EDGE_CM2
    assert n2_edge_cross_section_cm2(3) == LYMAN_EDGE_CM2 * 9.0
    assert n2_edge_cross_section_cm2(1, n2_edge_cm2=BALMER_LITERATURE_EDGE_CM2) == (
        LYMAN_EDGE_CM2
    )
    assert n2_edge_cross_section_cm2(2, n2_edge_cm2=BALMER_LITERATURE_EDGE_CM2) == (
        BALMER_LITERATURE_EDGE_CM2
    )
    assert n2_edge_cross_section_cm2(3, n2_edge_cm2=BALMER_LITERATURE_EDGE_CM2) == (
        LYMAN_EDGE_CM2 * 9.0
    )
    np.testing.assert_allclose(N2_EDGE_SCALE, 1.40e-17 / 2.52e-17, rtol=0.0, atol=0.0)


def test_local_boundfree_copy_matches_frozen_v4r5():
    labels = np.asarray([[9000.0, 4.0, 0.0, 0.0, 1.0]])
    temperature = np.asarray([[11000.0]])
    pressure = np.asarray([[1.0e5]])
    components = textbook_opacity_node_components_v4r5(labels, temperature, pressure)
    state = saha_electron_diagnostics_v4r3(labels, temperature, pressure)
    populations = _hydrogen_ground_anchored_level_populations(
        temperature,
        state["hydrogen_neutral_density_cm3"],
        constants=DEFAULT_TEXTBOOK_CONSTANTS,
    )
    local = hydrogen_boundfree_from_level_populations_local(
        components["frequency_nodes_hz"],
        -np.expm1(-components["frequency_nodes_u"]),
        populations,
        state["rho_g_cm3"],
    )
    np.testing.assert_allclose(
        local, components["hydrogen_boundfree"], rtol=1.0e-12, atol=0.0
    )


def test_balmer_edge_verdict_requires_hot_isolation():
    control = {
        "v4r5": _signed(0.209),
        "n2_only": _signed(0.200),
        "n2_balmer_edge": _signed(-0.046),
        "drop_n_ge_3": _signed(0.190),
        "drop_n_ge_7": _signed(0.205),
        "no_stimulated": _signed(0.205),
        "no_hminus": _signed(0.207),
        "no_hi_ff": _signed(0.208),
        "n2_karzas_shape": _signed(0.200),
        "n2_karzas_full": _signed(-0.040),
    }
    hot = {
        "v4r5": _signed(-0.004),
        "n2_only": _signed(-0.55),
        "n2_balmer_edge": _signed(-0.005),
        "drop_n_ge_3": _signed(-0.004),
        "drop_n_ge_7": _signed(-0.004),
        "no_stimulated": _signed(-0.004),
        "no_hminus": _signed(-0.004),
        "no_hi_ff": _signed(-0.004),
        "n2_karzas_shape": _signed(-0.004),
        "n2_karzas_full": _signed(-0.004),
    }
    decision = decide_balmer_diagnostics(
        control=control, hot=hot, karzas_n2_ratio_span_dex=0.02
    )
    assert decision["verdict"] == "BALMER_EDGE_CROSS_SECTION"
    assert decision["n2_is_primary_carrier"] is True
    assert decision["lyman_owns_hot_tail"] is True
    assert decision["edge_hot_stable"] is True
    assert decision["recommended_construction"]["n1_edge_cm2"] == LYMAN_EDGE_CM2
    assert decision["recommended_construction"]["n2_edge_cm2"] == (
        BALMER_LITERATURE_EDGE_CM2
    )
    assert decision["recommended_construction"]["global_edge_power_n1"] is False

    reopened = dict(hot)
    reopened["n2_balmer_edge"] = _signed(-0.20)
    assert (
        decide_balmer_diagnostics(
            control=control, hot=reopened, karzas_n2_ratio_span_dex=0.02
        )["verdict"]
        == "INCONCLUSIVE"
    )


def test_nonflat_karzas_ratio_blocks_edge_verdict():
    control = {
        "v4r5": _signed(0.209),
        "n2_only": _signed(0.200),
        "n2_balmer_edge": _signed(-0.046),
        "drop_n_ge_3": _signed(0.190),
        "drop_n_ge_7": _signed(0.205),
        "no_stimulated": _signed(0.205),
        "no_hminus": _signed(0.207),
        "no_hi_ff": _signed(0.208),
        "n2_karzas_shape": _signed(0.050),
        "n2_karzas_full": _signed(-0.010),
    }
    hot = {
        name: _signed(-0.004 if name != "n2_only" else -0.55)
        for name in control
    }
    decision = decide_balmer_diagnostics(
        control=control, hot=hot, karzas_n2_ratio_span_dex=0.20
    )
    assert decision["verdict"] == "BALMER_GAUNT_SHAPE"
