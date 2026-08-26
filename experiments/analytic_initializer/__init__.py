"""Offline discovery tools for a physics-constrained analytic initializer.

This package is deliberately separate from the production initializer.  It
contains no checkpoint loading and no solver dispatch; the first stage only
asks whether the converged atmosphere corpus admits compact, physically
motivated coordinates.
"""

from .discovery import (
    LABEL_FIELDS,
    TARGET_FIELDS,
    Corpus,
    collect_excluded_indices,
    fit_low_rank_surrogate,
    load_strict_truth,
    make_split,
)
from .candidates import (
    ScalarOpacityParameters,
    build_h1_reduced_state,
    fit_scalar_opacity_parameters,
    predict_effective_opacity,
)
from .profile_closure import (
    ProfileClosureParameters,
    fit_profile_closure,
    integrate_mass_from_opacity,
    predict_profile_closure,
)
from .profile_initializer import (
    AnalyticProfileParameters,
    fit_analytic_profile_parameters,
    load_analytic_profile_parameters,
    predict_analytic_reduced_state,
    save_analytic_profile_parameters,
)

__all__ = [
    "LABEL_FIELDS",
    "TARGET_FIELDS",
    "Corpus",
    "collect_excluded_indices",
    "fit_low_rank_surrogate",
    "load_strict_truth",
    "make_split",
    "ScalarOpacityParameters",
    "build_h1_reduced_state",
    "fit_scalar_opacity_parameters",
    "predict_effective_opacity",
    "ProfileClosureParameters",
    "fit_profile_closure",
    "integrate_mass_from_opacity",
    "predict_profile_closure",
    "AnalyticProfileParameters",
    "fit_analytic_profile_parameters",
    "load_analytic_profile_parameters",
    "predict_analytic_reduced_state",
    "save_analytic_profile_parameters",
]
