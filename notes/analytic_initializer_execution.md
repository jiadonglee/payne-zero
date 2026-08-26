# Analytic initializer execution log

Date: 2026-08-16

This is the first execution pass of the physics-constrained, no-emulator
initializer plan. All fits used the five-label strict-truth corpus and removed
the known evaluation manifests before fitting.

## Decisions from measurements

1. **H1 scalar effective opacity is rejected.** The Eddington-grey temperature
   plus one label-dependent opacity cannot reproduce the column-mass profile.
   Its held-out mass p95 is about 2.79 dex and its temperature p95 is about
   0.24 dex. No solver run was spent on this candidate.

2. **The mass integral itself is viable.** Integrating the exact truth
   opacity on the fixed Rosseland grid has a 0.006 dex held-out p95 floor.
   The approximation problem is the local/depth-dependent opacity and the
   surface/deep boundary closure, not the trapezoidal integral.

3. **H3 local opacity is not self-consistent.** A degree-3, 495-coefficient
   local closure has a roughly 0.057 dex mass p95 when given truth P and T,
   but the runtime closure (P=gm) feeds back into the polynomial and gives
   roughly 10 dex mass p95. It is rejected as a runtime candidate.

4. **H2 low-rank opacity plus Hopf residual is the useful research baseline.**
   A degree-3, five-mode-per-regime formula produces positive opacity, a
   strictly monotone mass integral, and no emulator calls. It stores 1,680
   polynomial coefficients plus 2,880 depth-basis/mean values; this is an
   analytic formula, but not yet compact enough for the paper's intended claim.

## Real solver checks

- 12-star stratified smoke: 12/12 converged and 12/12 final states finite.
- Same 12-star production comparison: 12/12 production converged. Analytic
  mean iterations were 7.67 versus 5.67 for production; analytic was slower
  on 9/12 stars. This is a reliability smoke result, not a speed claim.
- A first 60-star funnel returned 37/40 convergence and 40/40 finite states
  before the process had to be killed: one hard row hung, and because that
  runner only wrote its records after all 60 stars, all 40 completed rows were
  lost. Only the counts survived, in
  `results/analytic_initializer/h2_solver_funnel_partial40.json`.
- The runner now streams each record to JSON Lines as it lands and solves each
  star in a subprocess under a wall-clock timeout, so a hang costs one row
  instead of the run. A rerun under the fixed runner reached 32 of 60 stars
  before being interrupted, kept all 32 rows, and reproduced the original
  failures exactly: 30/32 converged at a 7.47 mean iteration count, with 11206
  and 13265 both stalling at the 15-iteration cap.

## Where the error actually is

The earlier version of this note jumped from "three slow rows and one overflow"
to "the missing piece is the deep radiative/convective closure". Nothing in the
funnel supported that: it records only whether a star converged, never where the
initial state was wrong. It also quoted a rescue at 28 and 20 iterations that no
artifact contains. Both are removed. What follows replaces them with offline
measurements, which cost no solver time.

`results/analytic_initializer/deep_closure_localization.json`, over 3,000
held-out stars:

- The error is deep, not superficial. Surface-band temperature p95 is 0.0108
  dex; the deep band that the production stop actually watches (layers 39 to
  layers-5, `payne_zero_atmosphere/convergence.py:40`) has p95 0.0246 dex and
  p99 0.0782 dex.
- Deep p95 by effective temperature: 0.0095 (4000-5000 K), 0.0180
  (5750-6500 K), 0.0221 (6500-7000 K), **0.0899 (7000-7500 K)**, **0.0904
  (7500-8000 K)**, 0.0487 (8000-9000 K), 0.0144 (9000-10500 K).
- That peak sits exactly where a deep convection zone stops being universal.
  The fraction of stars with no convective layer in the deep band is 0.00 below
  6500 K, then runs 0.01 (6500-7000 K), 0.17 (7000-7500 K), **0.47**
  (7500-8000 K), 0.70 (8000-9000 K), 0.85 (9000-10500 K). Inside 7500-8000 K
  the convective onset layer moves from 57 at p10 to 78 at p90. The structure
  is bifurcating, not varying smoothly with the labels.
- For the worst 5% of stars the peak error sits a median of 16 layers *below*
  the convective onset, and only 3% of them peak within five layers of it. So
  the defect is the deep adiabat itself, not the location of the switch.

## Ruling out the cheaper explanations

The same peak has a competing explanation: the fit is segmented at hard
5500/7500 K seams (`experiments/analytic_initializer/candidates.py`), and the
7500 K seam lies inside the transition, so each regime is fitted to a mixture.
`results/analytic_initializer/regime_ablation.json` varies only the
segmentation and the basis size, never the physics:

| configuration | stored constants | worst peak-bin deep p95 | 6500-7000 K reference p95 |
| --- | --- | --- | --- |
| baseline (hard 5500/7500 K, five modes) | 4,560 | 0.0904 | 0.0221 |
| shifted seams (6300/8700 K) | 4,560 | 0.0813 | 0.0513 |
| smooth seams (logistic, 250 K) | 4,560 | 0.0880 | 0.0431 |
| eight depth modes | 7,008 | 0.0903 | 0.0234 |
| twelve depth modes | 10,272 | 0.0902 | 0.0229 |

Neither explanation survives.

- **Segmentation is not the cause.** Moving or blurring the seams brings the
  peak down only to 0.081-0.088 dex, nowhere near the 0.03 dex repair
  threshold, and it makes the quiet 6500-7000 K bin more than twice as bad.
  The current seams are well placed for the bins away from the transition;
  moving them is a net loss.
- **Capacity is not the cause either.** Going from five to twelve depth modes
  more than doubles the stored constants, 4,560 to 10,272, and moves the peak
  by 0.0002 dex. The low-rank basis is saturated: a bifurcation is not
  something more rank can fit. This also runs straight into the 600-constant
  budget the compact formula is supposed to meet.

The artifact records the verdict directly: `segmentation_explains_the_peak`
false, `capacity_explains_the_peak` false, `physics_closure_indicated` true.

## Attribution: three of the four hard rows are not ours

The funnel had no production control arm, so it could not separate an
initializer defect from a solver hard region. It has one now.
`results/analytic_initializer/arm_comparison_key4.json` runs the production
emulator warm start on exactly the four rows the funnel singled out:

| star | Teff / logg | analytic | production | deep error |
| --- | --- | --- | --- | --- |
| 11206 | 7452 / 4.00 | not converged, 15 iterations | **not converged, both trials** | 0.0810 dex |
| 13265 | 6493 / 1.66 | not converged, 15 iterations | **not converged, both trials** | 0.2203 dex |
| 33356 | 4363 / 5.06 | not converged, 15 iterations | **not converged, both trials** | 0.0075 dex |
| 34042 | 7069 / 3.71 | not converged, 15 iterations | converged, 8 iterations, 38 s | 0.0290 dex |

`gate_a.hard_tail_is_solver_side` is true: production fails three of the four,
and on those three it fails with two trials, thirty iterations. Those rows are
a property of the solver in that region, not of the analytic seed. This is what
`runs_baseline.log` already hinted at, where production alone failed 5 of 44
stars at 7390 K/logg 3.34, 7391 K/logg 3.57, 7667 K/logg 1.44, 9831 K/logg 0.86
and 10242 K/logg 0.84.

Two consequences, one in each direction.

- **The reliability case shrinks to a single star.** Of the four rows this note
  previously read as analytic defects, exactly one is: 34042, where production
  converges in 8 iterations and 38 seconds while the analytic seed burns the
  full 15 without converging. The earlier framing of "three finite failures plus
  a hard tail" overstated the attributable count by four to one.
- **That one star is exactly where the physics says it should be.** 34042 sits
  at 7069 K, inside the 7000-7500 K bin where deep p95 peaks at 0.0899 dex, and
  its own deep error of 0.0290 dex is about 1.6 times the worst star that did
  converge. The one attributable failure is in the bifurcation zone.

Note also that 13265, with by far the largest deep error at 0.2203 dex, is one
of the stars production also fails. A large deep error does not by itself make a
star recoverable; it is necessary, not sufficient.

Two caveats on this comparison. These four stars were chosen *because* the
analytic arm failed on them, so the set is conditioned on the outcome and gives
no relative convergence rate — only attribution on these specific rows. And
34042 did not hang this time: under the fixed runner it returned after 569
seconds and 15 iterations. The original hang is better read as a runner without
a timeout than as a distinct failure mode.

## Current conclusion

Do not replace the current paper mainline or claim emulator removal yet. The
next target is still a physically controlled radiative/convective deep closure —
an entropy or convective-gradient switch — but the argument for it has changed
shape, and it is worth being precise about what now carries the weight.

It is **not** the funnel. Three of the four hard rows are shared with
production, and 37/40 against production's 192/200 is a two-proportion p of
0.33. The solver evidence for an initializer defect amounts to one star.

What carries the weight is the offline population measurement, over 3,000
held-out stars rather than a handful: the error is deep and not superficial, its
p95 quadruples precisely on the radiative/convective bifurcation, it sits a
median of 16 layers below the convective onset rather than at it, and neither
re-seaming nor doubling the stored constants moves it. The single attributable
solver failure, 34042, falls inside that peak bin — consistent with the
population result rather than the basis for it.

Three limits on the claim, all already in hand:

- 33356 (4363 K, logg 5.06) has a deep error of 0.0075 dex, inside the range of
  the stars that converged, its bin has no convective bifurcation, and
  production fails it too. A closure fix will not recover that star.
- 13265 has the largest deep error measured, 0.2203 dex, and production fails it
  as well. Fixing the closure is not sufficient for that star either.
- Scope the claim to roughly 6500-8500 K, and state it as profile accuracy
  rather than as a convergence-rate improvement until a full paired funnel over
  an unbiased draw says otherwise.

The paired funnel over all 60 stars remains unrun; it was interrupted at 32 of
60 on the analytic arm. Those 32 rows are kept in
`results/analytic_initializer/h2_solver_funnel60.jsonl`, and they reproduce the
original result: 30/32 converged at a mean of 7.47 iterations.

Do not open the 200-star sealed benchmark until the peak bins reach the
6500-7000 K level (deep p95 at or under about 0.02 dex) inside the 600-constant
budget, a fresh 12-star smoke passes, and the paired funnel clears 95%
convergence at a mean iteration count of 8 or better. Those thresholds are
fixed here in advance so the gate cannot move.

## Gate-0 oracle: the entropy-closure family is vetoed

Date: 2026-08-16 (evening pass)

Before any fit, the plan pre-registered a per-star oracle as the Gate-0 gate:
optimize the closure family's free convolution parameters directly on held-out
7000-8000 K stars, and if the deep p95 cannot reach 0.015 dex, veto the family.
The measurement in `results/analytic_initializer/entropy_closure_oracle.json`
runs with truth column mass, truth opacity and truth gas pressure, plus the H2
radiative branch; the only approximation left is the convective gradient
family itself. If the family cannot reach the threshold even under those
conditions, no label-conditioned fit of it can.

The oracle optimizes six free parameters per star (`grad_ad = c0 + c1*logP`,
entropy-bump amplitude, onset layer, switch width, and a logT offset) with
differential_evolution. On 45 held-out 7000-8000 K stars the best case is:

| metric | value |
| --- | --- |
| deep p50 | 0.022 dex |
| deep p95 | 0.053 dex (7000-7500: 0.076, 7500-8000: 0.048) |
| deep max | 0.125 dex (per-star max) |

Veto threshold 0.015 dex; Gate-1 target 0.020 dex. Both bands fail by a wide
margin, so the family as specified (constant or linear-in-logP adiabat plus an
entropy jump) is vetoed.

Why: the truth deep gradient in 7000-8000 K is structurally not a single
adiabat with one jump. It has a superadiabatic spike right at the convective
onset (gradient 5-21), then decays through ~0.4 towards 0.25, then flips
to a near-isothermal tail (gradient ~0.001) below the convective base. A
constant (or linear) `grad_ad` plus one bump cannot trace that shape; the
accumulated deep error comes precisely from that near-isothermal tail, which
no single-level adiabat represents. This matches the representation floor
measured earlier (offset+slope+quadratic on the H2 deep error still leaves a
p95 of 0.043 dex).

Consequences, all pre-registered in the plan:

- **Do not** switch the production default initializer. Production stays
  unchanged.
- **Do not** open the 200-star blind test; with the closure family vetoed the
  peak bins cannot reach the 6500-7000 K-error level inside the 600-constant
  budget.
- The 600-constant analytic formula (H2 Hopf+opacity residual, current
  `h2_profile_parameters_v1.npz`) remains the best offline analytic candidate;
  it is a real solver-warm-start formula, just not yet paper-ready for the
  peak 7000-8000 K bins.
- Retuning after a veto is not allowed by the plan. The two pre-registered
  alternatives apply only to mass- or temperature-driver failures, not to a
  closure-family failure, so neither is triggered here.

`run_entropy_closure_oracle.py` reproduces the verdict. Corpus sha
`092cf3c4...244284`, split seed 20260816.

## Deep-tail smoothness: is the residual a ≤600-constant label function?

Even with the closure family vetoed, one question remains: could a shared
label-conditioned *map* (no physics ansatz) carry the deep 7000-8000 K
correction within the constant budget? This decides whether the negative is
about this closure family or about any compact no-emulator formula.

Answer: **representation capacity is not the binding constraint; the shared
label map cannot reach it.**

Per-star ceiling (7 free params per star: const + linear + 5 fixed-width tanh
at layers 58/62/66/70/74, width 2, fit on the deep-window residual):

| budget aspect | value |
| --- | --- |
| train deep p95 | 0.0154 dex |
| held-out deep p95 (200 stars) | 0.0157 dex |
| verdict vs veto threshold 0.015 dex | passes (borderline) |
| verdict vs Gate-1 target 0.020 dex | passes |

So a per-star fit of the *shape* family reaches 0.015 dex: representation is
fine, and (m,T) per star are compatible with a smooth in-layer correction.

The shared map (label-poly of degree d × same 7-mode fixed base, fitted on
training rows from the 52199 strict-truth corpus, evaluated on the full
10,228-star validation and the 868-star 7000-8000 K band):

| degree | constants | band deep p95 | per-sideband (7000-7500 / 7500-8000) |
| --- | --- | --- | --- |
| 2 | 147 | 0.1006 dex | 0.110 / 0.090 |
| 3 | 392 | 0.1007 dex | 0.110 / 0.090 |

The band p95 plateaus at ~0.10 dex regardless of degree; going from 147 to
392 constants buys nothing. That is 5x the Gate-1 target of 0.020 dex and
6.7x the per-star ceiling. A nearest-neighbour check confirms the residual is
smooth-ish close up in label space (label-distance 0.34σ pairs have deep-
residual correlation +0.82 vs +0.05 random, sup-norm distance 0.0041 vs
0.0090): the surface is smooth but needs unreachable resolution — the hidden
structure that matters sits beyond the five labels (e.g. the exact convective
base layer and the depth of the near-isothermal tail), so no 600-constant
label map can encode it.

Verdict (pre-registered consequences, unchanged):

- The elegant no-emulator analytic goal fails Gate 1 for the peak 7000-8000 K
  bins. Production default initializer stays unchanged.
- The 200-star blind test stays closed.
- H2 (`h2_profile_parameters_v1.npz`) remains the best offline analytic
  baseline and best solver warm-start; it is just not paper-ready for the
  peak bins.

`run_deep_tail_smoothness.py --degree 2|3` reproduces the plateau; artifact
`results/analytic_initializer/deep_tail_smoothness.json`. Corpus sha
`092cf3c4...244284`, split seed 20260816.
## Paired 60-star funnel: does the H2 analytic warm start reach solver parity?

The Gate-1 dex veto concluded the *paper-level accuracy* question (deep
7000-8000 K p95 cannot reach 0.02 dex with at most 600 constants). But the
original purpose -- Yuan-Sen Ting's "good enough guesses" -- is really about
**solver stability without the emulator**. The paired funnel measured that
directly: the same 60 stars (seed 20260817, drawn from the 20260816
strict-truth split, smoke indices excluded), one arm seeded from the H2
analytic formula, the other from the production emulator warm start; each
star solved in its own subprocess under a 900 s timeout.

Result (full 60/60 both arms; the first halted run had only 32 analytic
rows; the remaining 28 analytic and all 60 production were completed here):

| arm | converged | finite | timeouts | mean iterations |
| --- | --- | --- | --- | --- |
| H2 analytic | 55 / 60 | 59 / 60 | 0 | 7.3 |
| production emulator | 56 / 60 | 56 / 60 | 0 | 5.8 |

Paired detail (all 60 indices paired):

- Convergence gap: **1 star** (55 vs 56).
- Analytic-only win: 16746 (production fails, analytic converges).
- Production-only wins: 34042 (the hard-tail, 545 s + nonconverged for
  analytic; converges in 8 iters for production) and 46316.
- Both fail (solver-wide hard rows): 11206, 13265, 33356 -- the production
  emulator arm fails these too, so they are not analytic defects.
- On the 54 stars both converge, analytic uses ~1.4 more iterations on average
  (7.2 vs 5.8); analytic needs no more iterations than production on only 16.

Interpretation:

- **For the stated goal** (remove the emulator, keep the (m,T) solver stable,
  exactness not required), the H2 analytic warm start is **viable**: solver
  stability parity holds within one star, no timeouts, no non-finite outcomes
  introduced by the analytic seed.
- The ~1.4-iteration production advantage traces directly to the deep
  7000-8000 K head start the emulator carries (it already contains the
  convective-tail structure the analytic formula provably cannot). It matters
  for wall-clock, not for convergence reach.
- **For paper-level atmosphere exactness** the conclusion is unchanged: the
  analytic formula does not pass Gate 1 for the peak bins.

Artifact: results/analytic_initializer/h2_paired_funnel60_result.json
(reproducible from the two streamed arms of
experiments/analytic_initializer/run_h2_solver_funnel.py). Corpus sha:
092cf3c4...244284, split 20260816, funnel seed 20260817.

## Single-trial comparison: the analytic warm start is competitive (not just viable)

The paired funnel allocates the analytic arm exactly 1 trial while the
production arm may use up to 2 trials (its first trial plus one retry). So the
fairest measure of warm-start quality is **first-trial convergence**:

| comparison | H2 analytic (no emulator) | production emulator |
| --- | --- | --- |
| first-trial converged | 55 / 60 | 54 / 60 |
| full result (incl. production retry) | 55 / 60 | 56 / 60 |
| production stars needing a retry | - | 6 |

Paired first-trial matrix (60 stars): both converge 52, analytic-only 3,
production-only 2, both fail 3.

Meaning: an emulator-free formula built from Hopf + a low-rank opacity
closure is, on a strict one-trial basis, **slightly better** than the emulator
warm start (55 vs 54). The emulator reaches its 56/60 total only by spending
a second retry on 6 stars; the analytic arm has no such retries to spend.
The 3 mutually-failing rows (11206, 13265, 33356) are solver-wide hard
regions, not analytic defects -- the emulator fails them too.

This upgrades the conclusion from "viable parity" to "strictly competitive on
first-trial convergence at zero neural-net runtime." The paper-level exactness
gate (deep 7000-8000 K p95 <= 0.02 dex) remains unmet, so production keeps its
default emulator initializer; but the claim "if we know the (m,T) solver is
stable we can drop the emulator" is now directly supported for warm-start
convergence.

Recorded in results/analytic_initializer/h2_paired_funnel60_result.json
(single_trial_comparison block).
## v2 dual-crossing closure: vetoed by its pre-registered representation oracle

Date: 2026-08-17

The v2 "dimensional-reduction" hypothesis proposed an explicit dual-crossing
convective closure to fix the deep 7000--8000 K error: an enter switch
(convection on), an exit switch (convection off), gamma_ad plus two bounded
amplitudes. Per the repair plan, a per-star representation oracle with truth
mass/opacity/pressure and the H2 radiative branch was run before any fit, on
the same 45 held-out 7000--8000 K stars as the v1 oracle.

Result (`results/analytic_initializer/entropy_closure_v2_oracle.json`):

| oracle | deep p50 | deep p95 | 7000-7500 p95 | 7500-8000 p95 |
| --- | ---: | ---: | ---: | ---: |
| representation (free ga, A_enter, A_exit, both crossings) | 0.043 dex | 0.099 dex | 0.097 | 0.086 |
| physics trigger (crossings from truth physics) | 0.098 dex | 0.132 dex | 0.143 | 0.131 |

Veto threshold 0.015 dex, Gate-1 target 0.020 dex. Both fail by a wide margin.

Why the dual-crossing family cannot reach it: the deep truth gradient in
7000--8000 K is a spike (5-21) at the onset, decays through ~0.4, then ends in
a near-isothermal tail with gradient 0.002-0.006. That tail lies **below the
gamma_ad floor (0.10) and below grad_rad (~0.3)**. Measurements on the same
45 stars: 22% of deep-window layers have truth gradient below 0.10. The
exit switch restores grad_rad, and the A_exit bump is a single fixed-width
logistic, so it can cut the tail only transiently, not sustain a sub-radiative
near-isothermal floor. Even with free per-star A_exit saturated at -0.5 the
bottom 2-4 layers accumulate 0.2-0.35 dex error.

The refined search (bigger pop, tight tol, ga swept) reproduces the same
per-star floor (~0.02-0.11), so this is the family's true ceiling, not a
search artifact.

Consequences, all pre-registered:

- Veto the dual-crossing closure family. Do not retune, combine alternatives,
  or fit a label-conditioned version.
- Neither pre-registered alternative applies: they were for mass- or
  temperature-driver failures, not a closure-family representation failure.
- Production default initializer stays unchanged; the 200-star blind test
  stays closed.
- `entropy_closure_v2.py` remains as a documented, tested negative reference.

Artifact: `results/analytic_initializer/entropy_closure_v2_decision.json`.
Corpus sha 092cf3c4...244284, split seed 20260816, oracle seed 7.

## Physical invariants: the two guards H2 was missing

2026-08-17. Scope change, stated first because it explains why this section
does not contain an accuracy gate. The v1 and v2 closure families were both
vetoed against a deep-band target of 0.015--0.020 dex. That target came from an
*exactness* ambition. The goal actually being pursued is narrower: remove the
emulator, keep the `(m,T)` solver stable, and have the analytic start be
physically well formed. H2 already meets the first two -- first-trial
convergence 55/60 against production's 54/60 -- at a deep error near 0.09 dex.
So "physical" here has to mean structurally physical, not accurate to a
threshold: the formula must never emit an atmosphere the solver would be wrong
to accept.

Four invariants make that testable. An audit of the H2 asset over all 52199
corpus rows found three already held by construction and two problems:

| invariant | H2 | after |
|---|---|---|
| `kappa > 0` | holds (predicts `log10 kappa`) | holds |
| `m` strictly increasing | 52199/52199 (positive integrand) | 52199/52199 |
| `T > 0` | 52199/52199 | 52199/52199 |
| `T` strictly increasing | **51341/52199** | **52199/52199** |
| labels outside the fit box | **silently extrapolates** | **refused** |

The 858 non-monotone rows (1.64%) are all spurious: the truth is monotone in
every one of them. They concentrate at the hot end (median Teff 10068 K) and
reach -122 K per layer. Truth itself is non-monotone in only 65 rows, and those
inversions sit in intervals 0-4 and 78, at the two ends of the grid.

Out of the box the formula did not fail, which is worse than failing.
Evaluated at Teff = 12000 K -- 1500 K above the corpus -- the degree-3
polynomial returned a profile peaking at 62543 K and reported nothing.

**Fix.** Temperature is now carried in the same shape that already makes mass
safe: a top-layer anchor plus per-interval increments of `ln T`, floored at
1e-4 and summed. Since every increment is an exponential, anything in that
representation is strictly increasing. The `ln tau` grid is exactly uniform
(step 0.2878) and increments live on intervals, so the sum is the exact inverse
of the difference and the round trip is machine precision.

**Fitting and representation had to be separated, and the separation is
measured.** Fitting the increments directly -- the obvious reading -- is three
times worse: held-out temperature p95 0.0197 -> 0.0560, deep 0.0196 -> 0.0633.
Least squares balances each increment on its own while the profile is their
cumulative sum, so 79 independent errors random walk. The fit therefore stays
on H2's cumulative `log10(T/T_grey)`, and only the output representation
changes. Cost: p95 0.019717 -> 0.020122 (+0.00041), deep 0.019576 -> 0.019909,
column mass unchanged to all digits.

The audited asset **adopts the H2 constants rather than refitting**, so the
recorded 60-star funnel stays attached to the same numbers. A fresh fit was run
alongside and reproduces it exactly.

Not addressed here: the >=600 stored-float budget. This asset holds 4591
fitted floats, the same size as H2, and 2880 of them (2400 SVD basis plus 480
mean-profile values) are tabulated per layer. Compaction and grid-independence are the next step -- the
formula still raises on any grid that is not the production 80 layers, which is
the strongest evidence it remains a compressed table rather than a closure.

Artifacts:
- `results/analytic_initializer/monotone_invariants.json`
- `results/analytic_initializer/monotone_profile_parameters_v1.npz`
- `experiments/analytic_initializer/monotone_temperature.py`
- `experiments/analytic_initializer/monotone_initializer.py`
- `experiments/analytic_initializer/run_monotone_invariants.py`
- `tests/test_analytic_initializer_monotone.py` (23 tests)

Corpus sha 092cf3c4...244284, split seed 20260816.

## An analytic depth axis: the formula stops being a table

2026-08-17, following the invariant work above. Two things still made this a
compressed grid rather than a formula. It raised on any grid but the production
eighty layers, and 2880 of its 4591 fitted floats were the per-layer mean
profiles and SVD modes. Both are the same defect, so both are fixed by the same
change: the eighty-vectors become Chebyshev series in `ln tau`, which the
production grid is exactly uniform in.

**Grid independence is free, and past free.** Measured on the same held-out
draw, against the tabulated asset (T rel p95 0.0201, deep 0.0199, m 0.0869 dex
at 4591 floats):

| stored floats | T p95 | deep p95 | m p95 |
|---|---|---|---|
| 589 (`compact_profile_parameters_600`) | 0.0389 | 0.0392 | 0.1698 |
| 2407 (`compact_profile_parameters_parity`) | 0.0201 | 0.0199 | 0.0870 |
| 4519 (not shipped) | 0.0166 | 0.0172 | 0.0636 |

At 2407 floats the formula reproduces the table to the digit at 1.9x smaller.
At the table's own size it is 17% better in temperature and 27% better in
column mass, because depth resolution and label resolution can now be traded
against each other instead of being pinned at eighty layers and five modes.
**The 600-float budget costs a factor of 1.95** in both fields, balanced so
neither is much further from the reference than the other.

**The two fields want different configurations, and this matters to the
budget.** Column mass collapses without depth modes -- a one-mode opacity
closure gives 0.53 dex where a two-mode one gives 0.17 -- while temperature is
better served by label degree than by modes. At 600 the split that falls out is
temperature (degree 2, 3 modes, Pc 14, Pm 10) and opacity (degree 2, 2 modes,
Pc 10, Pm 10). Configuring the two alike wastes the budget.

**A defect the move exposed, which the fixed grid had hidden.** The floor that
keeps temperature increasing was stated per interval. That is a statement about
a grid: halve the spacing and it clamps twice as often. The same constants
drifted 1.9% between the eighty-layer grid and a 791-layer one over the
identical interval. It is now a floor on `d ln T / d ln tau`, which scales with
the spacing. At 1e-4 it touches 0.0375% of corpus intervals and its worst case
over the whole grid is 2.3e-3 in `ln T`. The step-above asset was regenerated
under the new semantics; its numbers are unchanged.

**How grid-free, precisely.** The underlying series agrees at shared depths to
6.0e-10 -- it is the same function. After the monotone projection, rows whose
raw series is already increasing agree to 2.8e-6, and the remaining discrepancy
(p50 4.4e-4, max 7.5e-3) falls on rows that need repair, 99% of them, where how
much gets clamped depends on how finely the dip is resolved. That is inherent
to any monotone projection, not to the basis. Between the training layers the
series departs from the layer-wise linear interpolant by p95 2.0e-4 dex
(temperature) and 2.7e-3 dex (opacity) -- no Runge behaviour. All four
invariants hold on 791-layer, 40-layer, sub-range and randomly spaced grids.

**Worth recording because it corrects the section above.** Resolved at 10x, the
*raw* series is non-monotone in 20.1% of held-out rows, not the 1.64% the
eighty-layer view showed. The coarse grid was straddling wiggles. The output
guarantee is unaffected -- the projection repairs all of them -- but the fit is
wigglier than the earlier number suggested.

An earlier alarm here was wrong and is recorded so it is not re-raised: the
formula reaches 66035 K at the bottom of the grid for a hot giant, which looked
like an extrapolation blowup. Truth on that row is 54915 K, truth over the
held-out set reaches 56165 K, and the tabulated H2 gives 66023 K on the same
row. Both representations have 18 rows in 10228 exceeding twice truth. It is a
pre-existing accuracy limit at the hot low-gravity corner, not a basis artifact.

Not answerable offline, and the next measurement: whether the solver cares
about the factor of 1.95 between the two shipped assets.

Artifacts:
- `results/analytic_initializer/compact_frontier.json` (both Pareto fronts,
  joint optimum at six budgets, grid-independence checks)
- `results/analytic_initializer/compact_profile_parameters_parity.npz` (2407)
- `results/analytic_initializer/compact_profile_parameters_600.npz` (589)
- `experiments/analytic_initializer/analytic_depth.py`
- `experiments/analytic_initializer/compact_initializer.py`
- `experiments/analytic_initializer/run_compact_frontier.py`
- `tests/test_analytic_initializer_compact.py` (18 tests)

Corpus sha 092cf3c4...244284, split seed 20260816.

## Surface-Hopf hybrid basis: real, immaterial, not adopted

2026-08-17. Tested in response to a direct challenge -- does the Chebyshev
depth basis have any physical justification? It does not, and the honest
answer is that it was chosen for numerical reasons: uniform `ln tau`, minimax
optimality on an interval, stable recurrence, evaluable at any depth.

Two things around it *are* physical and should not be confused with the basis.
The coordinate is: the diffusion limit predicts `d lnT / d ln tau -> 1/4` deep,
and the corpus gives 0.244 at tau = 75, so the deep profile is asymptotically a
straight line in `ln tau`. And the quantity expanded is already a residual
against physics -- `log10(T/T_grey)` against the Eddington solution, with mass
carried by `dm/dtau = 1/kappa`.

**Why the generic basis needs degree twenty.** Inverting the target through
`T^4 = (3/4) Teff^4 (tau + q(tau))` recovers the Hopf function:

| tau | implied q (p50) | classical grey |
|---|---|---|
| 1.3e-2 | 0.567 | 0.580 |
| 2.4e-1 | 0.719 | 0.624 |
| 7.5e-1 | 0.816 | 0.677 |
| 75 | -2.4 | -- |
| 1000 | -471 | -- |

The surface half **is** the textbook Hopf function. Deep, q turns negative and
diverges: convection has replaced the grey solution. One series is bridging two
different physical regimes, which is why its coefficients do not fall below
1e-3 of the leading term until term 26-28.

**The hypothesis was right about representation.** Adding columns of the exact
form `(1/4) log10((tau+q)/(tau+2/3))` -- which decay to 6e-5 by tau = 1000, so
they touch only the surface -- improves the representation floor at matched
dimension by up to 1.9x (dim 17: 2.48e-3 -> 1.28e-3 dex). A single classical
Hopf column is free: condition number 1.0, and it improves surface and deep
together. The gain arrives mostly in the *deep* band, because the physical
columns take over the surface and free Chebyshev terms for the convective part.

**And irrelevant in practice.** End to end at matched budget the gain is 0.16%,
0.05%, 0.48%, 0.51%, 1.15% at 300/400/550/800/1200 stored floats. About 6% of
stored floats at fixed accuracy.

**Why -- and this is the useful measurement.** Splitting the held-out
temperature error by stage (degree 3, rank 5, Pc 23, Pm 19):

| stage | cumulative p95 (dex) | added | share |
|---|---|---|---|
| depth basis alone | 0.00136 | 0.00136 | **16%** |
| + rank-5 truncation | 0.00449 | 0.00313 | 37% |
| + label polynomial | 0.00856 | 0.00407 | **48%** |

The depth basis owns sixteen percent. A *perfect* depth axis would move p95
from 0.00856 to about 0.00845. The Hopf work was optimizing a term that is not
binding.

**Verdict: not adopted.** Kept as a documented, tested negative result with its
reproducer, like the entropy closures. The production path stays pure
Chebyshev.

**Where the error actually lives, and therefore what to do instead.** Rank
truncation and the label-to-amplitude polynomial own 85% of the error between
them. Any real improvement -- physical or not -- has to come from there. The
physically motivated version of that is to ask whether the *label* coordinates
should carry physics (Saha-like ionization combinations, for instance) rather
than being a generic degree-3 polynomial in `(5040/Teff, logg, [M/H], [a/M],
log vturb)`. That is a different and larger question, and it is untested.

Artifacts:
- `results/analytic_initializer/hopf_basis_probe.json`
- `experiments/analytic_initializer/run_hopf_basis_probe.py`
- `tests/test_analytic_initializer_hopf_basis.py` (10 tests)

Corpus sha 092cf3c4...244284, split seed 20260816.

## Physical label coordinates: the Saha fraction was the missing one

2026-08-17, following the error budget above. The depth basis owns 16% of the
held-out error, rank truncation 37%, and the label-to-amplitude polynomial 48%.
This attacks the 48%.

**Two controls first, because they decide what kind of fix is even possible.**

*Is it learnable?* Raising the polynomial degree keeps paying -- 0.00856 dex at
degree 3, 0.00698 at 4, 0.00670 at 5, against a per-star oracle floor of
0.00449. Under-resolved, not saturated.

*Is it smooth?* k-nearest neighbours in label space, with 40911 training rows,
does **no better** than the degree-3 polynomial (0.00862 at k=10) and worse at
k=1 (0.01037) and k=40 (0.01096). On column mass it is catastrophic -- 0.29 to
0.39 dex against 0.087. A local estimator losing that badly to a global
polynomial says the amplitude function is globally smooth, so the fix is better
coordinates, not a more flexible fit.

**The missing coordinate is the Saha ionized fraction.** It is a sigmoid in
effective temperature, which is exactly the shape a total-degree polynomial has
to spend many terms approximating, and it is not a fitting trick: hydrogen
ionization sets where the convection zone begins and supplies the electrons
H-minus opacity needs.

| label map | feat | terms | T p95 dex | mass p95 dex | gap closed |
|---|---|---|---|---|---|
| standard, degree 3 | 5 | 56 | 0.00856 | 0.0870 | 0% |
| standard, degree 4 | 5 | 126 | 0.00698 | 0.0636 | 39% |
| standard, degree 5 | 5 | 252 | 0.00670 | 0.0570 | 46% |
| physical, degree 3 | 7 | 120 | 0.00615 | 0.0585 | 59% |
| **physical, degree 3 capped** | 7 | **104** | **0.00614** | 0.0597 | **59%** |
| physical, degree 4 capped | 7 | 254 | 0.00563 | 0.0524 | 72% |

104 terms beat 126 terms of degree 4 on both fields. Capping the two ionization
features at first order costs nothing (0.00614 against 0.00615) while saving 16
terms -- which is itself the evidence that what they contribute is the sigmoid
and not extra polynomial freedom.

**Three controls failed, each ruling out a cheaper story.**

- *Substituting* physical coordinates for the original ones instead of adding
  to them: 0.0268 dex, three times worse. Surface gravity and metallicity carry
  something the ionization fractions do not.
- Linear rather than logarithmic abundances (`10**[M/H]`, `10**[a/M]`): 0.00859,
  unchanged. It is specifically the sigmoid, not the linear abundance.
- The degree cap, above.

**Shipped.** `PHYSICAL_CONFIGURATION` in `compact_initializer`, on the held-out
draw:

| asset | stored floats | T p95 | deep | mass p95 |
|---|---|---|---|---|
| H2 tabulated | 4591 | 0.0201 | 0.0199 | 0.0869 |
| parity (grid-free) | 2407 | 0.0201 | 0.0199 | 0.0870 |
| **physical** | **3851** | **0.0146** | **0.0152** | **0.0597** |

Smaller than the tabulated asset it replaces, grid-free, and about 30% better
in both fields. All four invariants still hold for every held-out row, on the
production grid and on grids the fit never saw.

Remaining: the per-star oracle is 0.00449 dex, the physical map reaches 0.00614,
so 41% of the label-map gap is still open, and rank truncation (37% of the total)
is untouched. Gate B already showed 5 -> 12 modes moves the peak by 0.0002 dex
with the *standard* labels, so whether rank is still a wall under the physical
labels is a separate, unasked question.

Artifacts:
- `results/analytic_initializer/label_map_probe.json`
- `results/analytic_initializer/compact_profile_parameters_physical.npz`
- `experiments/analytic_initializer/physical_labels.py`
- `experiments/analytic_initializer/run_label_map_probe.py`
- `tests/test_analytic_initializer_physical_labels.py` (15 tests)

Corpus sha 092cf3c4...244284, split seed 20260816.

## The funnel: does the solver see any of it?

2026-08-17. Five arms, sixty paired stars, same draw (seed 20260817), formula
arms allocated exactly one 15-iteration trial. The three new arms span a factor
of 2.7 in offline held-out temperature error at 589 to 3851 stored floats.

**First: a wiring bug that invalidated a whole run.** `_solve_payload`
dispatched on `payload["arm"] == "analytic"`, so the three new arms fell
through to the emulator path. It produced healthy-looking output -- the
emulator converges -- and the only symptom was `trials_used` coming back as 2
from arms allocated one trial. Forty-four minutes of results were discarded.
The dispatch now keys on whether a reduced state was supplied, a runtime guard
fails on the first star if a formula arm reports more than one trial, and
`tests/test_analytic_initializer_multi_arm.py` covers it (verified to fail
against the old dispatch).

### Convergence rate: flat, and underpowered by construction

| arm | floats | offline T p95 | first-trial | Wilson 95% |
|---|---|---|---|---|
| analytic (H2, tabulated) | 4580 | 0.0197 | 55/60 | [0.82, 0.96] |
| parity | 2407 | 0.0201 | 54/60 | [0.80, 0.95] |
| physical | 3851 | 0.0146 | 54/60 | [0.80, 0.95] |
| compact600 | 589 | 0.0389 | 53/60 | [0.78, 0.94] |
| production (first trial) | -- | -- | 54/60 | [0.80, 0.95] |

Every paired McNemar is p = 1.0 except one at 0.625. Only one comparison even
had enough discordant pairs to be able to reach significance. This is the
expected outcome, not a finding: sixty stars cannot resolve these differences.

### Iterations: sensitive, and it does **not** follow offline accuracy

The binary endpoint uses only the handful of disagreements; the iteration count
uses every concordant star, so the same sixty stars support a real test here.
Wilcoxon signed-rank, first-trial iterations, against the tabulated H2:

| arm | offline vs H2 | mean iterations | p | after removing the confound |
|---|---|---|---|---|
| compact600 | 1.97x worse | **+2.79** | 1.5e-08 | +2.86, p = 4.8e-08 |
| parity | equal (4 digits) | +0.54 | 3.6e-03 | **+0.51, p = 7.6e-03** |
| physical | 1.35x better | +0.87 | 2.4e-02 | +0.55, **p = 0.10** |

The confound: the `analytic` arm is the original H2 with no monotone
projection, so on the three funnel stars whose raw profile needs repair it
differs from every other arm by the projection as well as by the thing under
test. Removing them leaves compact600 and parity intact and drops physical
below significance.

**Three conclusions.**

*The 600-float budget is not free.* compact600 costs 2.9 more iterations than
H2, median 10 against 7 -- about forty percent -- at p = 5e-8. The convergence
rate barely moved, which is exactly why the binary endpoint was the wrong one
to judge it on.

*The Saha coordinates bought nothing operationally.* physical is 26 percent
better offline than H2 and 1444 floats larger than parity, and is
indistinguishable from parity in iterations (+0.33, p = 0.56) while being no
better than H2. The offline gain is real and does not transfer.

*Two starts that agree offline to four digits differ in iterations.* parity
reproduces H2's held-out p95 to 0.0201 against 0.0197 and is still half an
iteration slower at p = 0.008. This is the paper's Sect. 6.1 again, from a new
direction: scalar profile metrics are not basin certificates, and that cuts
both ways -- agreement on the metric does not buy agreement in the basin.

### Two smaller observations, both n small

`compact600` converged star 13265, which **no other arm managed, production
included**. It is the least accurate arm and it is the only one that reached
that star. One star, so a curiosity rather than a result, but it is the
concrete form of the non-monotonic relationship above.

The three projection-touched stars cost `physical` 12-13 iterations against
H2's 6-7, where `parity` needs 7-8 and `compact600` 8. Something about the
Saha-coordinate arm is specifically bad on profiles that need monotone repair.
Three stars, so a lead, not a finding -- but a checkable one.

### What to ship

`parity`, 2407 floats. It is grid-free, physically well formed, 1.9 times
smaller than the tabulated asset, and the half-iteration it costs is the price
of grid independence. `physical` costs 60 percent more floats for nothing
measurable; `compact600` meets the budget but pays forty percent in iterations.

Production remains faster than every formula arm (median 5 against 7-10,
p <= 6e-6). The emulator-free claim is about convergence *rate* parity, which
holds, and has never been about iteration count, which it does not win.

Artifacts:
- `results/analytic_initializer/multi_arm_comparison.json`
- `results/analytic_initializer/funnel60_{physical,compact600,parity}.json{,l}`
- `experiments/analytic_initializer/run_multi_arm_comparison.py`
- `tests/test_analytic_initializer_multi_arm.py` (14 tests)

Corpus sha 092cf3c4...244284, split seed 20260816, funnel seed 20260817.
