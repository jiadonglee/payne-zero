# Fitting spectra pipeline smoke test closeout

Date: 2026-09-13

## Outcome

**PASSED.**  The synthesis/fitting pipeline recovers the correct M-giant
atmosphere from a noisy mock observation at 50 K library resolution.

Setup: the mock observation is the validated `t3750 g2.0 [M/H]=0` relaxed
truth atmosphere synthesized over 640-680 nm at R=20000 with Gaussian noise
at SNR 100 per pixel.  The library is the four relaxed truth atmospheres on
the same branch (3750/3800/3850/3900 K).  The fit minimizes the
normalized-flux chi-square over the library.

Result (`results/m_star_fit_smoke_v1/fit_report.json`):

- Best fit: `Teff = 3750 K`, `chi2/dof = 1.009` - the noise level is
  recovered exactly as designed, and the true library member is selected
  with delta Teff = 0.
- Discrimination is decisive: the nearest neighbor (3800 K) sits at
  `chi2/dof = 52` - the 50 K library spacing is resolved at ~50 sigma.

## Finding worth keeping

The label-based synthesis path (`synthesize_from_labels`) cannot serve M
giants: its neural atmosphere initializer only supports `5040/T <= 1.26`
(Teff >= ~4000 K).  M-giant fitting must fit over atmosphere libraries - the
validated converged atmospheres this campaign produced.  The smoke test now
does exactly that.

## Scope

This is a pipeline smoke test on four library atmospheres with synthetic
noise; it is not a retrieval demonstration over the full parameter space.
