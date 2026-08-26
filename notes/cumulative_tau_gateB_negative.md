# Cumulative-tau initializer: Gate B solver verdict (negative)

Date: 2026-08-21. Registered in notes/cumulative_tau_initializer_plan.md.

## Verdict

The frozen four-window cumulative-tau family was connected to the real solver
(new cumtau arm in run_h2_solver_funnel.py) and run on the 12-star smoke list.
Gate B fails: 0/3 completed stars converged; all three hit the 600 s timeout;
the remaining nine would not change the gate.

Paired control: the parity arm (compact Chebyshev closures) converged 12/12 on the
identical star list in 21-88 s, median about 40 s. The solver and the star draws are the
same in both arms, so the failure is the seed, not the environment.

## Why (offline diagnostic, same 3 stars)

| field | parity p95 | cumtau p95 |
|---|---|---|
| T rel | 0.007-0.015 | 0.051-0.069 |
| m dex | 0.05-0.08 | 0.12-0.28 |
| log kappa dex | 0.055-0.077 | 0.75-0.92 |

The decisive defect is opacity. parity predicts log kappa directly with its own
Chebyshev closure and derives m by integrating dm/dtau = 1/kappa. cumtau instead
fits m and T and derives kappa from kappa = tau/(m * dlnm/dlntau), a byproduct
that inherits every m/T slope error and lands p95 0.75-0.92 dex off truth. The solver
walks on opacity; a seed whose kappa is an order of magnitude wrong sits outside the
basin.

## What this closes

- Gate B denied: no 60-star funnel, no spectra gate, no open-200 run for cumtau.
- The four-window cumulative-tau label-map family is closed. Do not raise its degree.

Streamed records: results/analytic_initializer/funnel12_cumtau.jsonl (3 rows).
Probe: results/analytic_initializer/cumulative_tau_probe.json.
