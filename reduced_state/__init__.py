"""Reduced-state (m,T)(tau) reconstruction, resampling, and restart tooling.

This package builds a thin ``ReducedAtmosphere -> FullAtmosphere`` interface
entirely out of existing certified physics in ``payne_zero_atmosphere`` --
no new EOS, opacity, transfer, or hydrostatic code is written here. It also
reuses ``bench/``'s restart-benchmark machinery (``_solver_config``,
``StarRecord``) so aggregation via ``bench.report`` keeps working on the new
experiments' output.
"""
