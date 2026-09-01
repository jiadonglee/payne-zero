# v4r6 policy60 matched development-60 closeout

Date: 2026-08-29

Policy: `v4r6_analytic_warm_start_policy60_v1`

Final machine decision: **`STOP_POLICY60_MATCHED_DEVELOPMENT`**

This closes the matched 60-iteration study preregistered in
`notes/textbook_opacity_v4r6_policy60_matched_dev60_preregistration_20260829.md`.
It does not rewrite the historical 15-iteration
`FAIL_STOP_DEVELOPMENT`, the earlier 60-iteration diagnostic, or the v4r6
offline `FAIL_STOP`.

## Result

All three analytic arms used the same 60 development stars, one trial,
60 iterations, a 900 s per-star timeout, identical source manifest, and the
same remote runtime.

| Arm | All | Cool, Teff < 7500 K | Hot | Timeouts | Errors |
|---|---:|---:|---:|---:|---:|
| Decoupled: `m_grey + T_conv` | **54/60** | **24/27** | 30/33 | 0 | 0 |
| Grey: `m_grey + T_grey` | 52/60 | 21/27 | **31/33** | 0 | 0 |
| Coupled convective | 49/60 | 18/27 | **31/33** | 0 | 0 |

The decoupled arm passed every frozen absolute warm-start gate:

- total `54/60 >= 54/60`;
- cool `24/27 >= 23/27`;
- hot `30/33 >= 29/33`;
- finite seeds `60/60`;
- timeout `0 <= 6`;
- solver errors `0`.

Against the matched grey arm:

| Split | Both | Decoupled only | Grey only | Neither | Net |
|---|---:|---:|---:|---:|---:|
| All | 48 | 6 | 4 | 2 | **+2** |
| Cool | 18 | 6 | 3 | 0 | **+3** |
| Hot | 30 | 0 | 1 | 2 | **-1** |

The sole failed frozen check was cool paired net gain versus grey:
observed `+3`, required `>= +4`. Therefore the exposed development study
does not authorize a fresh-open preregistration or execution.

## Physical interpretation

Under the adopted 60-iteration budget, the decoupled analytic construction is
a successful warm start in the absolute sense: it converges 90% of the
development sample and performs especially well on cool stars.

The matched controls support the intended mechanism qualitatively:

- preserving `m_grey` while applying the convective temperature improves
  total convergence over grey by two stars and cool convergence by three;
- it improves over the coupled convective seed by five stars overall and six
  cool stars;
- the price is one hot-star convergence relative to either control.

The evidence is not strong enough for promotion because the preregistered
cool paired advantage missed its threshold by one star. This is a bounded
development result, not evidence that the opacity law, flux balance, spectra,
or production behavior are scientifically validated.

## Integrity verification

- source manifest SHA-256:
  `fee015711e42bbca970e7e68ff66da57feb81eb21c786706a08739a8dfeffaa8`;
- source files covered: 11;
- sample SHA-256:
  `5e0238098f5811de7738d6e8fcf5b9eb5d94fe85a8fa9505ad3513769179e27e`;
- all three runtime signatures matched exactly;
- remote and local sizes and SHA-256 values matched after transfer;
- local focused verification: `21 passed`.

Runtime: host `Node-06`, Python `3.12.9`, NumPy `2.4.6`, Numba `0.66.0`,
`NUMBA_THREADING_LAYER=workqueue`.

## Authoritative artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Decoupled JSON | 50613 | `58c679cb207fa7edde7774d4449e8ce2853068f7c4e819922406238115f422e6` |
| Decoupled JSONL | 37422 | `ddd506b88fd0390bd12e78eab37176f5f152631e924444190d39d113d8a483ae` |
| Decoupled runtime | 987 | `2d2b368a8f11bb958af536b93c3a8cad27009222464e1d2cbae7793f6de17a9e` |
| Grey JSON | 49849 | `e730a496a6b41d82f1f2ad537a264876d18db1e8835dd76029ddb0a42f574db4` |
| Grey JSONL | 37160 | `788b28a78a464543dc61ac303fcdbf1b668494b861142478705c90f98b78c688` |
| Grey runtime | 982 | `24465376fed71a170be12d4815cd3d800fddbefd76fa5045cc4b4e5db1d05db4` |
| Convective JSON | 47730 | `69a77248656264a624f34d4c0bef8a21005d0789e88c9c02c4f2e6fbbe33f911` |
| Convective JSONL | 36863 | `7ed7a74221fc37065180b72ce86e0acaa8cc8bdb606e5f829291d82c2de39335` |
| Convective runtime | 977 | `8ae67e3ceb28744c9994cd2c39d5eb3f6f18739c83b8ba720dc007cf170d9c32` |
| Matched score JSON | 5016 | `c0fb80effd2f49f05f8e74c6bf73cf22224a3f768decd964ba9ad912a0023b70` |
| Sequence log | 62576 | `3398d43be068e23af8cad0b68cfcb2ca97fcbdda2bb23f4c9bd97ef0799d67f2` |

## Stop boundary

Do not tune the cool-gain threshold, opacity, convection law, solver damping,
or timeout on this exposed result. Do not open fresh validation, spectra,
production switching, coupled ODE work, or sealed holdout under this policy.

A future attempt requires a new physical candidate and a new preregistration;
simply accepting `+3` after observing it would be post-hoc threshold movement.
