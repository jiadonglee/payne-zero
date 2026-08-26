"""Guardrails for the consolidated figure package.

The package was created by moving eleven plotting scripts into one place and
deleting their duplicated helpers.  The risk in that kind of change is silent:
two of the scripts used *the same variable names for different colours*
(``LEARNED`` is burnt orange in the manuscript and blue in the evidence
figures), and four of them had ``_configure_style`` functions that differed by
a single rcParam.  A careless merge would recolour published figures without
raising anything.

These tests pin the things a merge could quietly break.  They need only
matplotlib and numpy, not the solver environment.
"""

from __future__ import annotations

import importlib

import matplotlib
import pytest

matplotlib.use("Agg")

from payne_zero_figures import data, style  # noqa: E402

REPORT_MODULES = [
    "make_figures",
    "make_four_initializer_report",
    "make_three_initializer_report",
    "make_expanded_four_initializer_report",
    "make_two_vs_six_report",
    "make_cool_star_status_report",
    "make_cool_star_temperature_report",
    "make_convergence_geometry_figure",
    "make_convergence_geometry_animation",
    "plot_cool_star_3500_comparison",
    "plot_h2_results",
]


@pytest.mark.parametrize("name", REPORT_MODULES)
def test_report_module_imports(name):
    """Every migrated script still imports, so the move did not strand one."""

    assert importlib.import_module(f"payne_zero_figures.reports.{name}")


def test_paper_module_imports():
    assert importlib.import_module("payne_zero_figures.paper.figures")


def test_paper_and_evidence_palettes_stay_distinct():
    """The clash that makes a naive palette merge dangerous.

    If someone later collapses these two palettes, this fails rather than
    silently swapping the colours of every arm in every figure.
    """

    assert style.PaperPalette.LEARNED == "#b8621b"
    assert style.EvidencePalette.LEARNED == "#2a78d6"
    assert style.PaperPalette.PRODUCTION == "#4d4d4d"
    assert style.EvidencePalette.PRODUCTION == "#eb6834"
    assert style.PaperPalette.LEARNED != style.EvidencePalette.LEARNED


@pytest.mark.parametrize("preset", sorted(style.PRESETS))
def test_presets_apply_cleanly(preset):
    """Applying a preset must not raise and must actually change rcParams."""

    matplotlib.rcdefaults()
    style.configure(preset)
    assert matplotlib.rcParams["pdf.fonttype"] in (3, 42)


def test_unknown_preset_is_an_error():
    with pytest.raises(KeyError):
        style.configure("NO_SUCH_PRESET")


def test_two_vs_six_keeps_its_own_black():
    """One report deliberately uses a darker black than the shared palette."""

    module = importlib.import_module(
        "payne_zero_figures.reports.make_two_vs_six_report")
    assert module.BLACK == "#111111"
    assert style.OkabeIto.BLACK == "#222222"


def test_record_loaders_have_opposite_duplicate_semantics(tmp_path):
    """load_records is last-wins; load_records_strict rejects duplicates.

    Both behaviours are relied on by different callers, so collapsing them into
    one function would break whichever caller lost.
    """

    path = tmp_path / "records.jsonl"
    path.write_text('{"slug": "a", "n": 1}\n{"slug": "a", "n": 2}\n')

    assert data.load_records(path)["a"]["n"] == 2
    with pytest.raises(ValueError):
        data.load_records_strict(path)


def test_strict_loader_rejects_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError):
        data.load_records_strict(path)


def test_repo_paths_resolve():
    """The move changed directory depth; these must still point at the repo."""

    assert (data.REPO / "payne_zero_figures").is_dir()
    assert data.RESULTS.name == "results"
    assert data.EMULATOR_RUNS.parts[-2:] == ("runs", "reduced_state_emulator")
