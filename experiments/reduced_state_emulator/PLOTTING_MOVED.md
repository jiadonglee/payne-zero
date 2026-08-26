# The plotting scripts moved

Everything that drew a figure moved to `payne_zero_figures/`, so that the
palette, the rcParams and the artifact loaders live in one place instead of
eleven copies. The computation scripts in this directory did not move.

| Was | Now |
|---|---|
| `make_figures.py` | `payne_zero_figures/reports/make_figures.py` |
| `make_two_vs_six_report.py` | `payne_zero_figures/reports/` |
| `make_three_initializer_report.py` | `payne_zero_figures/reports/` |
| `make_four_initializer_report.py` | `payne_zero_figures/reports/` |
| `make_expanded_four_initializer_report.py` | `payne_zero_figures/reports/` |
| `make_cool_star_status_report.py` | `payne_zero_figures/reports/` |
| `make_cool_star_temperature_report.py` | `payne_zero_figures/reports/` |
| `make_convergence_geometry_figure.py` | `payne_zero_figures/reports/` |
| `make_convergence_geometry_animation.py` | `payne_zero_figures/reports/` |
| `plot_cool_star_3500_comparison.py` | `payne_zero_figures/reports/` |
| `../analytic_initializer/plot_h2_results.py` | `payne_zero_figures/reports/` |

Run them as modules from the repository root; the arguments are unchanged:

```bash
MPLCONFIGDIR=/tmp/mpl python3 -m payne_zero_figures.reports.make_figures
MPLCONFIGDIR=/tmp/mpl python3 -m payne_zero_figures.reports.make_two_vs_six_report --help
```

See `payne_zero_figures/README.md` for the two things a future cleanup of that
package could silently break.
