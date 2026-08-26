# Paper II physical-seed refresh (2026-08-20)

## Decision

The new two-field initialization is **runtime-independent of the released
six-field emulator**. The candidate predicts only \((m,T)\). Before the ordinary
energy iteration starts, the unchanged atmosphere solver seeds
\(P_{\rm gas}=gm\), positive \(n_e\) and \(\kappa_R\) placeholders, and
\(g_{\rm rad}=0\), then recomputes the four dependent fields through its own
EOS, opacity, transfer, and hydrostatic-closure routines.

This is not training independence: the frozen two-field network was trained
against converged full-atmosphere targets. It is also not exact numerical
seed-invariance, because synchronization stops adaptively when the pressure
change is below \(10^{-3}\) dex; other fields can retain differences of that
order.

## Reused and recomputed

- Reused: the frozen two-field checkpoint, unchanged atmosphere solver, released
  six-field production arm, and the 2026-08-19 post-opening blind-200
  physical-seed rerun.
- Recomputed: exact \((m,T)\) parity, learned two-field restarts, four-field
  rematerialization accuracy, depth-grid controls, both development spectral
  gates, the analytic comparison join, generated numbers/tables, all affected
  figures, and the paper PDF.
- The blind-200 rerun is a post-opening implementation confirmation, not a
  second blind test. `sealed_initializer_holdout_20260812.json` remained
  unopened.

## Remote execution and provenance

- Host: `astronode-garching-node08`
- Remote checkout:
  `/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero`
- Workers: 24, with all numerical-library thread counts fixed to one per worker.
- Result root: `results/paper_physical_seed_20260820/`
- Run root: `runs/paper_physical_seed_20260820/`
- Initial campaign PID: `1029953`
- Corrected resume PID: `1033559`
- The first driver treated a scientifically failed spectral gate (exit 1) as an
  execution failure. The resume accepts exit 1 only for that gate and records
  it explicitly; parity still requires exit 0.
- Executed first driver SHA-256:
  `74ba3e213f0266e6f178962316e41c0f464b48f191536f318620d704a717ceb2`
- Resume driver SHA-256:
  `40579283b7ae6d14a76f2aec7d2257638a3048ce9298739841e73f2bbc7f6055`
- Current canonical driver SHA-256:
  `b01149c93ea31cd2632e66ca7270e973157db4dc61c17a2dd87225bf8d92d156`
- Campaign manifest SHA-256:
  `4a185ae697ebb7f922757f26f2720dfab89c2fe764fa802af373bc47ca56cc78`
- All 14 outputs listed in the campaign manifest were synced locally and
  verified against their SHA-256 hashes. Superseded first-pass artifacts remain
  recoverable under `runs/paper_physical_seed_20260820/provenance/pre_final_rerun/`.

## Main results

- Exact physical reconstruction from converged \((m,T)\): 60/60 solver
  convergence, mean 3.37 iterations, versus 3.23 from full six-field truth.
  Fifty-five stars have the same iteration count; the largest gap is two.
- Learned two-field end-to-end path: 56/60 usable solver products. There is one
  rematerialization failure and three later solver failures. Mean iterations
  among converged stars are 4.05; the non-monotonic fraction is 18.6%.
- Released production arm: 59/60 convergence, mean 5.59 iterations, and 53.3%
  non-monotonic trajectories.
- On the 56 common converged stars, learned is faster for 40, tied for 8, and
  slower for 8. The iteration saving correlates with \(T_{\rm eff}\)
  (\(\rho=0.412\), \(p=0.00161\)); stars above 9000 K save 2.41 iterations on
  average versus 0.68 for the rest.
- Successful learned rematerializations: 59/60. Median relative errors are
  \(4.51\times10^{-3}\) for \(P_{\rm gas}\),
  \(5.24\times10^{-3}\) for \(n_e\),
  \(5.21\times10^{-3}\) for \(\kappa_R\), and
  \(5.59\times10^{-3}\) for \(g_{\rm rad}\).
- The failed rematerialization is corpus index 6152: pressure synchronization
  remained at \(5.034\times10^{-3}\) dex after eight passes.
- Exact-\((m,T)\) parity spectra pass all three 0.5% gates on 60/60 stars. The
  worst normalized, total, and continuum differences are respectively
  \(1.394\times10^{-3}\), \(1.416\times10^{-3}\), and
  \(3.205\times10^{-4}\).
- Learned spectra remain a scientific gate failure: 1/56 normalized-flux and
  2/56 total-flux exceedances; the continuum gate passes.
- Changing the intermediate grid from 40 to 640 points does not systematically
  improve solver convergence. The residual limitation is therefore not
  insufficient interior sampling.

## Interpretation

The refreshed evidence supports the representation claim: two coordinates are
enough to rebuild a solver-consistent atmosphere and preserve spectra when the
coordinates are exact. The learned initializer improves iteration count and
dependent-field accuracy over the released six-field network, but its current
profile and spectral tail still prevent a deployment claim.

## Final validation

- `paper/collect_numbers.py --check`: 171 macros and 37 hashed sources, passed.
- Relevant tests: 34 passed.
- LaTeX: 17 pages, no undefined references/citations, no overfull boxes, and no
  clipped provenance rows after splitting the generated table across two pages.
- Final PDF SHA-256:
  `3be4744447791ffbbb2bbc6c8c04d324c655067b9cd55bd922bd6a6ff92e679f`
