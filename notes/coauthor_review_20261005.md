# Coauthor revision: normal-star validation

The working manuscript is the coauthor's 2026-10-05 `main.tex` and `refs.bib`.
The normal-star experiment uses the complete v1.3 atmosphere corpora and one
fixed split per input family. Parent groups remain together for CNO and direct
abundances. Both output sets share their family's split and network architecture.

## Calculation

- Remote directory: `/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-coauthor-20261005`.
- Training: `astro-h100gpu01`, NVIDIA H100 NVL, existing `.venv-linux` environment.
- First physical restart: `astro-highcpu03` (Node-07), eight threads per star.
- Full restarts: `astro-highmem01` (Node-01, five labels), `astro-highcpu07`
  (Node-11, CNO), and `astro-highcpu09` (Node-13, direct abundances).
  Each runs 16 stars at a time, with eight threads per star. The output directories
  are `runs/full_<family>/` under the remote directory; logs are
  `runs/full_<family>.log`.
- Training driver: `experiments/reduced_state_emulator/run_coauthor_training_garching_20261005.sh`.
- Training and predictions: `results/coauthor_validation_20261005/<family>/`.
- Five-label split: 42,199 training, 5,000 validation, 5,000 test.
- CNO split: 42,199 training, 5,003 validation, 5,000 test; 1,622 unused rows.
- Direct-abundance split: 42,214 training, 5,019 validation, 5,045 test;
  29,738 unused rows. Whole parent groups account for the small count differences.
- Column mass uses positive linear-mass increments; temperature is relative to
  the grey profile, as in Paper I. No softplus increment of log mass is used.
- Training loss: decoded standardized profile smooth-L1 plus 0.1 times the
  adjacent-layer smooth-L1. One checkpoint is selected by validation loss.
- Adam, 300 epochs, batch 512, cosine learning-rate schedule; initial learning
  rate 1e-3 for five labels and CNO, 3e-4 for direct abundances. Both output sets
  within a family use the same learning rate. Direct-family learning-rate
  selection uses the sum of the two validation losses.
- Five- and eight-label architecture: six hidden layers of width 1,024, SiLU.
  The 84-input family uses the released element-aware architecture for both outputs.
- Two profiles use their 160 coordinates directly. Six profiles use 160 PCA
  coefficients; the PCA basis and all scalings use training rows only.
- Normal-star restarts: one trial per start, cap 15, minimum three iterations,
  unchanged deep-layer temperature threshold of 5e-4. Numerical convergence,
  flux diagnostics, and spectral agreement are reported separately.

## Results

All three pairs have completed training. Test-depth pooled errors:

| Family | Two-profile T p95 | Six-profile T p95 | Two-profile m p95 (dex) | Six-profile m p95 (dex) |
| --- | ---: | ---: | ---: | ---: |
| Five labels | 0.0039313 | 0.0033680 | 0.0257276 | 0.0175271 |
| CNO | 0.0031768 | 0.0030116 | 0.0195627 | 0.0146389 |
| Direct abundances | 0.0099503 | 0.0088598 | 0.0782229 | 0.0683267 |

Sources: `results/coauthor_validation_20261005/<family>/training_{2,6}profile.json`.
The six-profile prediction is more accurate in these two coordinates; the
restart comparison must determine the effect of physical reconstruction.

The direct prediction diagnostic is
`results/coauthor_validation_20261005/five_label/profile_accuracy.pdf`.
Its ordinate is the fraction of the 5,000 test stars below a given maximum
depth error. It is not a solver or spectral test.

The first five-label restart is test row 3 (8222.52 K, log g 1.6263,
[M/H] -1.6224). The two-profile, six-profile, and exact-rebuild starts each
converge in three iterations; the two nearest nodes need eight and eleven,
and the grey–convective start needs fourteen. Both neural trajectories
decrease from n=1 to n=2. This single star does not establish the population
ordering. Its diagnostics are under
`results/coauthor_validation_20261005/solver_smoke/`.

Matched timings use an AMD EPYC 9454 CPU, eight threads, and batch one, with
warm-up excluded. Eight evenly spaced test-temperature ranks give median
times of 0.6503 ms for the six-profile network plus decoding, 11.4656 s for
the rebuild, and 4.2291 s for one solver iteration. The median of the paired
cost ratios is 1.9502. The individual rebuilds require one to three evaluations.
Source: `results/coauthor_validation_20261005/timing.json`.

Full five-label spectra run in four shards, two each on `astro-h100gpu01` and
`astro-h100gpu02`. Each shard waits for the corresponding physical restart,
then evaluates the truth atmosphere, all six starts, and the two neural final
states over 400–900 nm at R=20,000. Outputs are
`results/coauthor_validation_20261005/spectra/star_*.json`; logs are
`runs/spectra_{0,1,2,3}.log`. Numerical convergence is recorded separately
from the spectral difference.

`coauthor_collect_20261005.py` on Node-07 waits for complete campaigns, then
generates the per-family reconstruction, convergence, and mismatch figures
under `results/coauthor_validation_20261005/<family>/restarts/`. It also
generates the five-label spectral CDF and `spectral_summary.json` after all
5,000 spectral records are available. Its log is `runs/collect.log`.

## Limitations seen in the new calculations

- At the first 28 completed five-label stars, 27 two-profile and 28 six-profile
  starts converge within 15 iterations. Their conditional iteration medians
  are five and four. Six-profile residuals rise from n=1 to n=2 for two of these
  28 stars; two-profile residuals rise for none. Completion order and corpus row
  order are not random, so these counts are diagnostics rather than population
  estimates. The corresponding figures are under
  `results/coauthor_validation_20261005/partial_five_label/`.
- The fixed eight-pass rebuild limit rejects some exact CNO (m,T) inputs.
  CNO test row 8 (8433.77 K, log g 0.8365, [M/H] -1.5456) reaches a pressure
  change of 0.001512 dex at pass eight. An independent post-hoc diagnostic
  reaches 0.000751 dex at pass ten, with the same 0.001 dex threshold.
  The fixed-cap test retains its failed outcome. Source:
  `results/coauthor_validation_20261005/rebuild_diagnostic.json`.
- On five-label test row 3, the maximum normalized-flux error before solving
  is 0.007121 for the two-profile start, 0.001531 for the six-profile start,
  and 0.0001001 for exact reconstruction. The two final neural-start spectra
  differ by 0.003247 in normalized flux. Source:
  `results/coauthor_validation_20261005/spectral_smoke/star_000003.json`.

- Five-label test row 12 (4354.66 K, log g 1.3562, [M/H] -1.9856) reaches the
  numerical stopping criterion from both neural starts, in seven and five
  iterations. Their final spectra differ by 0.005920 in normalized flux,
  exceeding the existing 0.005 spectral-parity threshold. Final-iteration
  maximum flux errors are 5.11% and 1.02%, respectively. Numerical convergence
  alone does not establish spectral parity. Source:
  `results/coauthor_validation_20261005/spectra/star_000012.json` and the
  corresponding `runs/full_five_label/star_000012/record.json` on Garching.

## Method checks

- `payne_zero_atmosphere/rosseland_mean.py:72` integrates opacity in column mass
  from the top with surface value `kappa[0] * m[0]`; rebuild calls this path.
- `payne_zero_atmosphere/temperature_correction.py:939` obtains the mass
  correction from the fractional pressure response to the temperature correction.
  `runner.remap_finalized_iteration_state` then remaps corrected mass and temperature.
- The stopping equation uses completed iteration n, starting state T(0), and
  final count N, with N >= 3 in the normal-star comparison.
- Paper I, Section V and Algorithm 1, use direct synthesis in the fast search
  and physical atmosphere convergence at the candidate minimum.
- `marcs_h5.load_marcs_node` reads 56 layers of T, electron and total particle
  number densities, tau5000, and height. It estimates gas pressure as ntot*kB*T
  and mass as Pgas/g. The MARCS100 protocol resamples log(T) on 80 uniformly
  spaced log-mass points with PCHIP. The extract does not contain Rosseland
  opacity or radiative acceleration. Source:
  `results/coauthor_validation_20261005/marcs_method.json`.
- The [MARCS public software page](https://marcs.astro.uu.se/software.html)
  distributes model readers and a structure interpolator. The manuscript
  describes this public distribution. APOGEE's use of MARCS is documented by
  [SDSS ASPCAP](https://www.sdss4.org/dr17/irspec/aspcap/) and Jönsson et al.
  (2020, AJ 160, 120; DOI 10.3847/1538-3881/aba592).

The complete normal-star restart and spectral comparisons are available. The M-giant extension and observed-star calculations remain incomplete.


## Complete normal-star comparisons and M-giant diagnostics (2026-10-07)

All 15,045 physical restart targets and 5,000 five-label spectral records completed.
The complete matched results, four normal-star figures, and generated numerical
macros are incorporated in Sect. 3. The corresponding abstract, Discussion, and
Conclusions now distinguish iteration savings, CPU work, numerical success,
and spectral accuracy. On shared neural successes, iteration savings are
3.9%, 8.8%, and 8.0% for five labels, CNO, and direct abundances. Including
reconstruction and solving, paired mean process CPU costs are 1.57, 1.59,
and 1.31 times those of the six-profile start. These totals exclude network
inference and use the concurrent eight-thread physical campaign.

Before solving, six-profile spectra have lower conditional median errors
in all four reported temperature intervals. Among 4,794 paired numerical
successes, 553 exceed the 0.005 maximum normalized-flux difference threshold.
Sources: `results/coauthor_validation_20261005/manuscript_summary.json`,
`experiments/reduced_state_emulator/coauthor_summary_20261007.py`.

The specified 3750 K, log g=0.75, [M/H]=-0.50 giant is tested with the
frozen v4 predictor and a hydrostatic rebuild seed. The formal numerical
stop is at four iterations. Released iteration with the original
five-step, two-consecutive-segment stability rule freezes at 65 iterations;
the independent stored nearest-MARCS path freezes at 95 under the same
synthesis backend. The 65-iteration endpoint remains outside the preferred
60-iteration budget, so the older budget-compliance tally is not upgraded. Three-metric endpoint differences over 665--667 nm
at R=20,000 are 0.00053375, 0.00065735, and 0.00015030, within 0.005.
The result is confined to that window and does not attribute a change in
cost to seed choice alone. Sources: `mgiant_cap.json`,
`mgiant_stability.json` under the new result root. The original reference
checkpoints remain intact; their copies and the new physical checkpoints
are in the remote `runs/mgiant_stability_20261007/`.

The MARCS selection description reports nine training tracks, a 108-point
training candidate pool, 80 attempted, 53 eligible, and 28 unrun after the
50-row quota was reached. The HDF5 extract omits original model geometry
and mass headers. On common log-column-mass support, the median per-star
maximum temperature difference from MARCS is 6.476%; 51 of 53 maxima fall
in the deeper half of that support. The new two-panel `fig_mstruct` is
in Sect. 4 and the structure value is in Discussion. The coordinate is
column mass because the extract supplies tau5000 rather than native
Rosseland optical depth. Script: `experiments/plot_coauthor_mstruct_20261007.py`;
products: `results/coauthor_validation_20261005/marcs_structure/`.

The first CNO development pilot contains eight corners at 3800 K,
log g=1.0, [M/H]=-0.75, with C/M, N/M, O/M each at -0.1 or +0.1.
Each uses frozen-v4 (m,T) and native-MARCS (m,T) starts with independent
hydrostatic rebuilds, released caps of 90 and 120, and the existing
stability and three-metric spectral criteria. All eight belong to one
development group. No sealed holdout is opened and no candidate is yet
admitted to training. The saved elemental abundances carry the prescribed
CNO changes: the opposite corners have C/N/O ratios of 10^0.2, while the
other elements are unchanged. Strict restart remains required for corpus
admission. Driver: `coauthor_mgiant_cno_pilot_20261007.py`.

Remote pilot directory: `runs/mgiant_cno_pilot_20261007/` under the
coauthor work directory. CPU Node-11: parent 715953, eight workers,
eight threads each. GPU H10002: evaluation process 1465033, waiting
for per-target solve completion. Logs: `solve.log`, `spectra.log`;
final report: `summary.json`. Production defaults and historical
products are unchanged.

The pilot is followed by strict self-restarts and three-metric restart spectral
checks. CPU process 719101 on Node-11 and GPU process 1467850 on H10002 wait
for the paired pilot report, then evaluate every target with the existing
flux and structure limits. Passed rows produce `cno_development_corpus.npz`,
with eight labels, six profiles, the common Rosseland grid, and a single
development parent group. `admission.json` retains all eight outcomes.
Driver: `coauthor_mgiant_cno_restart_20261007.py`.

The first completed pilot point, case_03 (C/M=-0.1, N/M=O/M=+0.1),
finishes 90 released iterations from the learned start. Its direct MARCS
reference produces nonpositive column mass at iteration five and fails.
This is a reference-path failure, retained in the eight-target denominator;
it does not imply absence of an equilibrium atmosphere. No criterion is
changed and the point is not admitted to training.


## 2026-10-08: first CNO development batch

All eight corners at Teff=3800 K, logg=1.0, [M/H]=-0.75 completed. Six pass the independent-path comparison, strict self-restart, and all three TiO restart spectral metrics. The retained development corpus has labels (6,8) and atmosphere profiles (6,80,6), with one parent group. The 6/8 fraction describes these corners, not random-label coverage. The accepted subset has median stabilization iterations 35 (learned) and 80 (MARCS); every strict restart numerically stops at iteration 3. Maximum restart spectral metric is 8.67e-5. The fixed flux selection limits and 665–667 nm window remain unchanged.

Case 03 has a direct MARCS failure from non-positive column mass at iteration 5; its learned-path stability time was not assessed by the original evaluator. Case 05 has learned-path stabilization at iteration 35 but no MARCS stabilization within 120 iterations. Both are retained in the original denominator and excluded from the corpus.

A separate development diagnostic follows independent stable MARCS products from cases 01 and 07 to cases 03 and 05 in eight nitrogen-abundance steps of 0.025 dex. Each step uses target chemistry and hydrostatic-seeded reconstruction. Final endpoints undergo the original fixed stability, path comparison, and strict restart checks. The original batch is preserved. Remote directory: `runs/mgiant_cno_walk_20261008/`; CPU Node-11 process 1133333, GPU H10002 process 1472037. Local driver: `experiments/reduced_state_emulator/coauthor_mgiant_cno_walk_20261008.py`.

The first-batch results are synchronized under `results/coauthor_validation_20261005/mgiant_cno_pilot/`. Sect. 4 now describes this measured development batch; its numerical values come from `collect_numbers.py`. The eight-label network, random-label coverage, and individual-abundance expansion remain unfinished.


The next local development batch fixes 24 random labels before solving (seed 20261008): Teff 3700–3900 K, logg 0.85–1.15, [M/H] -0.9 to -0.6, alpha ±0.05, microturbulence 1.75–2.25 km/s, and each CNO enhancement ±0.1 dex. The frozen five-label predictor supplies learned starts; reference starts use the nearest accepted independent MARCS-derived pilot endpoint in CNO coordinates. All retain the original `t3800_g1.00_m-0.75` ancestry group and development-training role. No independent validation is claimed. Remote directory: `runs/mgiant_cno_local_20261008/`; Node-11 process 1134315 (eight workers), H10002 process 1472734 (queued after the two-point walk diagnostic). The setup is synchronized under `results/coauthor_validation_20261005/mgiant_cno_local/setup.json`. The original accepted corpus remains six atmospheres until the new checks complete.


## 2026-10-08 10:08 CEST: CNO development runs complete

The two nitrogen-abundance walks both pass independent-path, strict-restart, and TiO spectral checks. Original direct-MARCS failures remain in the initial 6/8 accounting. The local random batch completes all 24 cases and accepts 21 (87.5%). Case 00 does not stabilize from the learned start within 90 iterations; case 23 does not stabilize from the MARCS-derived reference within 120; case 10 passes the paired spectral comparison but its strict restart does not satisfy numerical convergence within 120. None is included in the corpus. All 21 accepted strict restarts numerically stop at iteration 3; their largest three-channel restart spectral metric is 1.1966e-4.

On the 22 paired spectral successes, median stability iterations are 35 for learned starts and 30 for references. Here references start from already converged MARCS-derived pilot endpoints; they are not raw MARCS grid starts. The earlier 35/80 comparison and this 35/30 comparison use different reference initializers.

All results are synchronized. `coauthor_mgiant_cno_merge_20261008.py` combines 6 initial, 2 recovered, and 21 random accepted atmospheres into `results/coauthor_validation_20261005/mgiant_cno_development/corpus.npz`. The 29 labels are distinct and retain one common parent group and development-training role. There is no independent validation claim. All four remote parent processes have ended. Multi-parent coverage, new M-giant networks, and observed-star applications remain unfinished.
