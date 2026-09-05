import numpy as np

from experiments.reduced_state_emulator.m_star_interpolated_mt_seed_v1 import (
    EXPECTED_PROBE_IDS,
    SCALE_EASY_MIN_ELIGIBLE,
    interpolate_same_track_mt,
    probe_decision,
    select_probe_candidates,
    select_same_track_donor_indices,
    track_key,
)


def test_same_track_key_ignores_temperature() -> None:
    left = {
        "track": {
            "log_surface_gravity": 4.5,
            "metallicity": -1.0,
            "microturbulence_km_s": 1.0,
        }
    }
    right = {
        "track": {
            "log_surface_gravity": 4.5,
            "metallicity": 0.0,
            "microturbulence_km_s": 1.0,
        }
    }
    assert track_key(left) != track_key(right)


def test_bracketing_uses_nearest_cooler_and_hotter_not_two_nearest() -> None:
    selection = select_same_track_donor_indices(
        3500.0,
        np.asarray([3000.0, 3900.0, 4000.0]),
    )
    assert selection["kind"] == "bracketed"
    assert selection["indices"] == [0, 1]
    assert selection["outside_convex_hull"] is False


def test_one_sided_copies_nearest_and_marks_hull() -> None:
    selection = select_same_track_donor_indices(
        3500.0,
        np.asarray([3800.0, 4000.0]),
    )
    assert selection["kind"] == "one_sided"
    assert selection["indices"] == [0]
    assert selection["outside_convex_hull"] is True


def _profile(teff: float) -> np.ndarray:
    depth = np.linspace(0.0, 1.0, 80)
    column_mass = 10.0 ** np.linspace(-6.0, 2.0, 80)
    temperature = teff * (1.0 + 0.2 * depth)
    return np.stack([column_mass, temperature], axis=-1)


def _labels(teff: float, metallicity: float = 0.0) -> dict[str, float]:
    return {
        "effective_temperature": float(teff),
        "log_surface_gravity": 4.5,
        "metallicity": float(metallicity),
        "alpha_enhancement": 0.0,
        "microturbulence_km_s": 1.0,
    }


def test_bracketed_log_mix_is_linear_in_log_teff() -> None:
    low, high = 3600.0, 4000.0
    target = float(np.sqrt(low * high))
    mass, temperature, diagnostics = interpolate_same_track_mt(
        _labels(target),
        [_labels(low), _labels(high)],
        np.stack([_profile(low), _profile(high)], axis=0),
    )
    expected_mass, expected_temperature = np.sqrt(
        _profile(low)[:, 0] * _profile(high)[:, 0]
    ), np.sqrt(_profile(low)[:, 1] * _profile(high)[:, 1])
    assert diagnostics["kind"] == "bracketed"
    assert np.allclose(mass, expected_mass)
    assert np.allclose(temperature, expected_temperature)
    assert np.all(np.diff(mass) > 0.0)


def test_one_sided_mix_copies_nearest_profile() -> None:
    mass, temperature, diagnostics = interpolate_same_track_mt(
        _labels(3500.0),
        [_labels(3800.0), _labels(4000.0)],
        np.stack([_profile(3800.0), _profile(4000.0)], axis=0),
    )
    assert diagnostics["kind"] == "one_sided"
    assert diagnostics["outside_convex_hull"] is True
    assert diagnostics["delta_t_K"] == 300.0
    assert np.allclose(mass, _profile(3800.0)[:, 0])
    assert np.allclose(temperature, _profile(3800.0)[:, 1])


def test_probe_takes_first_three_easy_and_hard_by_priority() -> None:
    rows = []
    for priority, teff, delta, bin_name in (
        (9, 3500.0, 300.0, "hard"),
        (13, 3500.0, 100.0, "easy"),
        (14, 3500.0, 250.0, "hard"),
        (17, 3500.0, 300.0, "hard"),
        (29, 3800.0, 100.0, "easy"),
        (40, 3300.0, 300.0, "hard"),
        (43, 3300.0, 100.0, "easy"),
        (50, 3900.0, 100.0, "easy"),
    ):
        rows.append(
            {
                "candidate_id": f"star-{priority}",
                "priority": priority,
                "temperature_K": teff,
                "interpolation": {
                    "kind": "one_sided",
                    "delta_t_K": delta,
                    "probe_bin": bin_name,
                    "outside_convex_hull": True,
                },
            }
        )
    selected = select_probe_candidates({"rows": rows})
    assert [row["candidate_id"] for row in selected] == [
        "star-13",
        "star-29",
        "star-43",
        "star-9",
        "star-14",
        "star-17",
    ]
    assert len(EXPECTED_PROBE_IDS) == 6


def test_scale_requires_two_easy_recoveries() -> None:
    easy_pass = [
        {"probe_bin": "easy", "training_eligible": True},
        {"probe_bin": "easy", "training_eligible": True},
        {"probe_bin": "easy", "training_eligible": False},
        {"probe_bin": "hard", "training_eligible": False},
    ]
    easy_fail = [
        {"probe_bin": "easy", "training_eligible": True},
        {"probe_bin": "easy", "training_eligible": False},
        {"probe_bin": "easy", "training_eligible": False},
    ]
    assert SCALE_EASY_MIN_ELIGIBLE == 2
    assert probe_decision(easy_pass)["scale"] is True
    assert probe_decision(easy_fail)["decision"] == "stop_switch_to_continuation"
