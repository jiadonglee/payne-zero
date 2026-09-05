"""Compare eligible cool-star ATLAS endpoints with same-node native MARCS."""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bench.labels import StellarLabels

from .marcs_h5 import inspect_marcs_grid, load_marcs_node


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN_ROOT = (
    REPO_ROOT / "results" / "m_star_emulator_v1r2_marcs100"
)
DEFAULT_MARCS_GRID = REPO_ROOT / "SDSS_MARCS_atmospheres.h5"
DEFAULT_OUT_ROOT = (
    REPO_ROOT / "results" / "m_star_atlas_marcs_comparison_v1"
)
CLASS_COLORS = {"giant": "#E69F00", "dwarf": "#0072B2"}
METRIC_FIELDS = (
    "common_mass_temperature_abs_relative_percent_p95",
    "common_mass_gas_pressure_abs_dex_p95",
    "common_mass_electron_density_abs_dex_p95",
    "seed_index_temperature_abs_relative_percent_p95",
    "seed_index_column_mass_abs_dex_p95",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _absolute_stats(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "median": float(np.median(absolute)),
        "p95": float(np.percentile(absolute, 95.0)),
        "max": float(np.max(absolute)),
    }


def _resolve_product(case_path: Path, record: dict[str, Any]) -> Path:
    recorded = Path(record["primary"]["product_path"])
    if recorded.is_file():
        return recorded
    local = case_path.parent / "products" / "primary" / recorded.name
    if not local.is_file():
        raise FileNotFoundError(local)
    return local


def _load_atlas(path: Path) -> dict[str, np.ndarray]:
    fields = ("column_mass", "temperature", "gas_pressure", "electron_density")
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in fields if field not in data.files]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        atmosphere = {
            field: np.asarray(data[field], dtype=np.float64)
            for field in fields
        }
    shape = atmosphere["column_mass"].shape
    if shape != (80,) or any(values.shape != shape for values in atmosphere.values()):
        raise ValueError(f"{path} does not contain one consistent 80-layer atmosphere")
    if any(not np.all(np.isfinite(values)) for values in atmosphere.values()):
        raise ValueError(f"{path} contains non-finite values")
    if any(np.any(values <= 0.0) for values in atmosphere.values()):
        raise ValueError(f"{path} contains non-positive values")
    if np.any(np.diff(atmosphere["column_mass"]) <= 0.0):
        raise ValueError(f"{path} has non-monotone column mass")
    return atmosphere


def _common_mass_comparison(
    atlas: dict[str, np.ndarray],
    node: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    atlas_logm = np.log10(atlas["column_mass"])
    marcs_logm = np.log10(np.asarray(node.native_column_mass, dtype=np.float64))
    lower = max(float(atlas_logm[0]), float(marcs_logm[0]))
    upper = min(float(atlas_logm[-1]), float(marcs_logm[-1]))
    overlap = (marcs_logm >= lower) & (marcs_logm <= upper)
    if int(np.count_nonzero(overlap)) < 8:
        raise ValueError("ATLAS and MARCS share fewer than eight native mass layers")
    x = marcs_logm[overlap]

    atlas_temperature = np.interp(x, atlas_logm, atlas["temperature"])
    marcs_temperature = np.asarray(node.native_temperature)[overlap]
    temperature_percent = (
        100.0 * (atlas_temperature - marcs_temperature) / marcs_temperature
    )

    atlas_log_pressure = np.interp(
        x,
        atlas_logm,
        np.log10(atlas["gas_pressure"]),
    )
    marcs_log_pressure = np.log10(np.asarray(node.native_gas_pressure)[overlap])
    pressure_dex = atlas_log_pressure - marcs_log_pressure

    atlas_log_ne = np.interp(
        x,
        atlas_logm,
        np.log10(atlas["electron_density"]),
    )
    marcs_log_ne = np.log10(np.asarray(node.native_electron_density)[overlap])
    electron_density_dex = atlas_log_ne - marcs_log_ne

    metrics = {
        "overlap_native_layers": int(x.size),
        "log_column_mass_min": float(x[0]),
        "log_column_mass_max": float(x[-1]),
        "temperature_relative_percent": _absolute_stats(temperature_percent),
        "temperature_signed_percent_median": float(np.median(temperature_percent)),
        "gas_pressure_dex": _absolute_stats(pressure_dex),
        "gas_pressure_signed_dex_median": float(np.median(pressure_dex)),
        "electron_density_dex": _absolute_stats(electron_density_dex),
        "electron_density_signed_dex_median": float(np.median(electron_density_dex)),
    }
    profile = {
        "log_column_mass": x,
        "temperature_relative_percent": temperature_percent,
    }
    return metrics, profile


def _compare_case(
    case_path: Path,
    record: dict[str, Any],
    *,
    schema: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels = record["labels"]
    stellar_labels = StellarLabels(
        float(labels["effective_temperature"]),
        float(labels["log_surface_gravity"]),
        float(labels["metallicity"]),
        float(labels["alpha_enhancement"]),
        float(labels["microturbulence_km_s"]),
    )
    node = load_marcs_node(
        schema.path,
        stellar_labels,
        carbon_enhancement=float(record["track"]["carbon_enhancement"]),
        verify_sha256=False,
        expected_sha256=None,
        schema=schema,
        depth_coordinate="log_mass",
    )
    product = _resolve_product(case_path, record)
    atlas = _load_atlas(product)
    common, profile = _common_mass_comparison(atlas, node)

    seed_temperature_percent = 100.0 * (
        atlas["temperature"] - np.asarray(node.reduced_temperature)
    ) / np.asarray(node.reduced_temperature)
    seed_mass_dex = np.log10(atlas["column_mass"]) - np.log10(
        np.asarray(node.reduced_column_mass)
    )
    output = {
        "candidate_id": record["candidate_id"],
        "class": record["class"],
        "effective_temperature_K": float(labels["effective_temperature"]),
        "log_surface_gravity": float(labels["log_surface_gravity"]),
        "metallicity": float(labels["metallicity"]),
        "primary_iterations": int(record["primary"]["iterations"]),
        "restart_iterations": int(record["restart"]["iterations"]),
        "product_path": str(product.relative_to(REPO_ROOT)),
        "common_mass": common,
        "seed_index": {
            "temperature_relative_percent": _absolute_stats(
                seed_temperature_percent
            ),
            "column_mass_dex": _absolute_stats(seed_mass_dex),
        },
    }
    profile["class"] = np.asarray(record["class"])
    return output, profile


def _flatten_row(record: dict[str, Any]) -> dict[str, Any]:
    common = record["common_mass"]
    seed = record["seed_index"]
    row = {
        key: record[key]
        for key in (
            "candidate_id",
            "class",
            "effective_temperature_K",
            "log_surface_gravity",
            "metallicity",
            "primary_iterations",
            "restart_iterations",
            "product_path",
        )
    }
    row.update(
        {
            "overlap_native_layers": common["overlap_native_layers"],
            "common_mass_temperature_abs_relative_percent_median": common[
                "temperature_relative_percent"
            ]["median"],
            "common_mass_temperature_abs_relative_percent_p95": common[
                "temperature_relative_percent"
            ]["p95"],
            "common_mass_temperature_abs_relative_percent_max": common[
                "temperature_relative_percent"
            ]["max"],
            "common_mass_temperature_signed_percent_median": common[
                "temperature_signed_percent_median"
            ],
            "common_mass_gas_pressure_abs_dex_p95": common[
                "gas_pressure_dex"
            ]["p95"],
            "common_mass_electron_density_abs_dex_p95": common[
                "electron_density_dex"
            ]["p95"],
            "seed_index_temperature_abs_relative_percent_p95": seed[
                "temperature_relative_percent"
            ]["p95"],
            "seed_index_column_mass_abs_dex_p95": seed["column_mass_dex"][
                "p95"
            ],
        }
    )
    return row


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for stellar_class in ("giant", "dwarf", "all"):
        selected = (
            rows
            if stellar_class == "all"
            else [row for row in rows if row["class"] == stellar_class]
        )
        metrics: dict[str, Any] = {}
        for field in METRIC_FIELDS:
            values = np.asarray([row[field] for row in selected], dtype=np.float64)
            metrics[field] = {
                "median_across_cases": float(np.median(values)),
                "p95_across_cases": float(np.percentile(values, 95.0)),
                "max_across_cases": float(np.max(values)),
            }
        summary[stellar_class] = {"count": len(selected), "metrics": metrics}
    return summary


def _profile_envelope(
    profiles: list[dict[str, np.ndarray]],
    stellar_class: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in profiles if str(row["class"]) == stellar_class]
    lower = min(float(np.min(row["log_column_mass"])) for row in selected)
    upper = max(float(np.max(row["log_column_mass"])) for row in selected)
    grid = np.linspace(lower, upper, 240)
    stack = np.full((len(selected), grid.size), np.nan)
    for index, row in enumerate(selected):
        x = row["log_column_mass"]
        y = row["temperature_relative_percent"]
        mask = (grid >= x[0]) & (grid <= x[-1])
        stack[index, mask] = np.interp(grid[mask], x, y)
    count = np.sum(np.isfinite(stack), axis=0)
    keep = count >= 3
    return (
        grid[keep],
        np.nanmedian(stack[:, keep], axis=0),
        np.nanpercentile(stack[:, keep], 16.0, axis=0),
        np.nanpercentile(stack[:, keep], 84.0, axis=0),
        count[keep],
    )


def _make_figure(
    rows: list[dict[str, Any]],
    profiles: list[dict[str, np.ndarray]],
    out_root: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), constrained_layout=True)
    profile_axis = axes[0, 0]
    for stellar_class in ("giant", "dwarf"):
        color = CLASS_COLORS[stellar_class]
        selected = [row for row in profiles if str(row["class"]) == stellar_class]
        for profile in selected:
            profile_axis.plot(
                profile["log_column_mass"],
                profile["temperature_relative_percent"],
                color=color,
                alpha=0.11,
                linewidth=0.55,
            )
        x, median, low, high, _count = _profile_envelope(profiles, stellar_class)
        profile_axis.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        profile_axis.plot(
            x,
            median,
            color=color,
            linewidth=1.8,
            label=f"{stellar_class} (n={len(selected)})",
        )
    profile_axis.axhline(0.0, color="0.35", linewidth=0.7, linestyle="--")
    profile_axis.set_xlabel(r"$\log_{10}$ column mass (g cm$^{-2}$)")
    profile_axis.set_ylabel(r"$(T_{\rm ATLAS}-T_{\rm MARCS})/T_{\rm MARCS}$ (%)")
    profile_axis.set_title("Temperature structure at common column mass")
    profile_axis.legend(frameon=False)

    panels = (
        (
            axes[0, 1],
            "common_mass_temperature_abs_relative_percent_p95",
            r"p95 $|\Delta T|/T$ (%)",
            "Temperature difference",
        ),
        (
            axes[1, 0],
            "seed_index_column_mass_abs_dex_p95",
            r"p95 $|\Delta\log_{10} m|$ (dex)",
            "Column-mass grid displacement",
        ),
        (
            axes[1, 1],
            "common_mass_electron_density_abs_dex_p95",
            r"p95 $|\Delta\log_{10} n_e|$ (dex)",
            "Electron-density difference",
        ),
    )
    for axis, field, ylabel, title in panels:
        for stellar_class, marker in (("giant", "o"), ("dwarf", "s")):
            selected = [row for row in rows if row["class"] == stellar_class]
            axis.scatter(
                [row["effective_temperature_K"] for row in selected],
                [row[field] for row in selected],
                s=22,
                marker=marker,
                facecolor=CLASS_COLORS[stellar_class],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.82,
                label=stellar_class,
            )
        axis.set_xlabel(r"$T_{\rm eff}$ (K)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", color="0.9", linewidth=0.5)
    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
    figure.savefig(out_root / "atlas_vs_marcs_structure.png", dpi=300)
    figure.savefig(out_root / "atlas_vs_marcs_structure.pdf")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--marcs-grid", type=Path, default=DEFAULT_MARCS_GRID)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args(argv)

    case_paths = sorted(args.campaign_root.glob("cases/*/*/t*/case.json"))
    eligible: list[tuple[Path, dict[str, Any]]] = []
    for path in case_paths:
        record = _read_json(path)
        if bool(record.get("training_eligible")):
            eligible.append((path, record))
    if not eligible:
        raise SystemExit("no eligible ATLAS cases found")

    schema = inspect_marcs_grid(args.marcs_grid, verify_sha256=True)
    records: list[dict[str, Any]] = []
    profiles: list[dict[str, np.ndarray]] = []
    for case_path, record in eligible:
        comparison, profile = _compare_case(case_path, record, schema=schema)
        records.append(comparison)
        profiles.append(profile)

    rows = [_flatten_row(record) for record in records]
    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "atlas_vs_marcs_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "format": "payne_zero_mstar_atlas_marcs_structure_comparison_v1",
        "campaign": _read_json(args.campaign_root / "protocol.json")["campaign"],
        "eligible_case_count": len(records),
        "marcs_grid": str(schema.path),
        "marcs_sha256": schema.sha256,
        "reference_role": "same-node native MARCS; diagnostic only",
        "comparison_coordinate": "common log10 column mass",
        "groups": _group_summary(rows),
        "cases": records,
    }
    (args.out_root / "atlas_vs_marcs_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _make_figure(rows, profiles, args.out_root)
    print(
        json.dumps(
            {
                "eligible_case_count": len(records),
                "giant_count": summary["groups"]["giant"]["count"],
                "dwarf_count": summary["groups"]["dwarf"]["count"],
                "out_root": str(args.out_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
