# Cumulative-\(\tau\) analytic initializer: frozen pilot plan

Date: 2026-08-21

## Question

Can a continuous, closed-form initializer in \(x=\ln\tau\) produce positive,
strictly monotone \(m(x)\) and \(T(x)\), avoid the hard 5500/7500 K seams, and
enter the unchanged solver basin without a neural checkpoint?

This pilot does not claim a first-principles atmosphere solution. It tests a
physics-constrained analytic warm start.

## Frozen v0.1 family

For \(y\in\{m,T\}\),

\[
\frac{{\rm d}\ln y}{{\rm d}x}
=\sum_{j=1}^{4} A_{y,j}(\ell)W_j(x),
\qquad A_{y,j}>0 ,
\]

where the four non-negative logistic windows partition optically thin,
photospheric, sub-photospheric, and deep layers with fixed boundaries
\(\tau=(10^{-2},1,100)\). Their analytic integrals are softplus differences.
There are no hard effective-temperature regimes.

The temperature anchor is fitted as a correction to the Eddington-grey
temperature near \(\tau=0.013335\). Column mass is fitted directly. The
provisional opacity is derived, not independently fitted:

\[
\kappa_{\rm R}=\frac{\tau}{m}
\left(\frac{{\rm d}\ln m}{{\rm d}\ln\tau}\right)^{-1}.
\]

Only continuous degree-1 and degree-2 label maps and fixed window widths
\(0.35,0.70,1.20\) in \(\ln\tau\) are screened.

## Data boundary

- Fit and validation use the strict-truth corpus with the existing
  manifest-aware exclusions.
- Existing development samples may be used only after the offline family is
  frozen.
- `sealed_initializer_holdout_20260812.json` remains unopened and is used only
  as an exclusion manifest.

## Pre-registered gates

### Gate A: offline eligibility

All conditions must pass:

- finite, positive \(m,T,\kappa_{\rm R}\) on every validation row;
- strictly increasing \(m\) and \(T\) on every validation row;
- temperature relative p95 \(\le 0.05\);
- column-mass dex p95 \(\le 0.20\);
- representative infinitesimal seam checks at 5500 and 7500 K:
  maximum relative \(T\) jump and maximum \(m\) dex jump \(<10^{-3}\);
- at most 200 fitted floating constants.

If no screened candidate passes, stop before solver work.

### Gate B: 12-star solver smoke

- at least 11/12 first-trial convergence;
- no timeout or non-finite final atmosphere.

If Gate B fails, do not launch the 60-star funnel.

### Gate C: development-60 solver funnel

Minimum usable result:

- at least 52/60 first-trial convergence;
- median iterations at most 10.

Paper-candidate target:

- at least 57/60 first-trial convergence;
- median iterations at most 8.

Profile error is diagnostic only. The solver gates decide the result.

### Gate D: spectra and expanded open sample

Only a frozen Gate-C paper candidate may run:

- the existing 60-star 0.5% spectral gate;
- the open 200-star benchmark.

The sealed holdout is not opened without a separate promotion decision.

## v0.2 activation after the frozen v0.1 result

The 2,000-row v0.1 smoke failed Gate A, but its per-star NNLS oracle nearly
reached the profile thresholds: temperature p95 \(=0.0525\) and column-mass
p95 \(=0.179\) dex. The fitted continuous degree-2 label map, rather than the
four-window depth family, caused most of the loss.

Before seeing any v0.2 result, one bounded follow-up is therefore activated:

- retain the same four windows, anchors, invariants, split and Gate-A limits;
- test only the existing seven-coordinate physical map (the five standard
  coordinates plus hydrogen and metal Saha ionized fractions);
- screen degrees 1 and 2; degree 2 is diagnostic because it necessarily
  exceeds the 200-float gate;
- do not alter window boundaries or add a hard regime.

If the degree-1 physical map does not pass Gate A, v0.2 stops. A higher-degree
diagnostic may identify label-map capacity but cannot advance to the solver.
