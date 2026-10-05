# Cumulative A0000 evidence

`cumulative_baseline.py` prepares fresh matched A0000/candidate offload and
folded-resident processes for all16 workloads. Use a retained implementation;
a testing or rejected attempt cannot supply the final cumulative curve. This
helper has only CPU/synthetic validation so far. No real cumulative plan or
measurements have been launched by adding it.

From the repository root, with the campaign Python environment:

```text
python profile/results/optimization/offload-gap-001/continuation-03/cumulative_baseline.py prepare --attempt ATTEMPT
```

The plan uses the selected attempt's predeclared independent block count
(historical default3; A0030 has6), with the same formal inner sampling as the
continuation. It covers16 workloads ×2 implementations ×2 placements ×blocks:
192 processes at3 blocks or384 at6. Preparation exposes the estimated work,
freezes source and helper hashes, exact commands, order and output paths, and
refuses to replace a previous plan. Do not prepare a speculative real plan
while the retained final implementation remains undecided.

Commit the exact `ATTEMPT/cumulative-A0000/plan.json` and initial manifest.
Wait until no other GPU timing/profile/correctness task is active, then run:

```text
python profile/results/optimization/offload-gap-001/continuation-03/cumulative_baseline.py run --attempt ATTEMPT
python profile/results/optimization/offload-gap-001/continuation-03/cumulative_baseline.py audit --attempt ATTEMPT
```

The runner rejects uncommitted plans, changed jobs/helpers/sources, preexisting
raw outputs and automatic restarts. Audit checks every unique matched cell,
process order, source/configuration/sampling/memory, execution timestamps and
one environment, and recomputes means from raw samples. It writes full paired
statistics to `summary.json`, plus figure-oriented `analysis.json` and
`summary.csv`. Every speedup is a within-block A0000-offload/candidate-offload
ratio. Never multiply parent-relative speedups to invent cumulative evidence.
Signed latency and gap reductions retain two-sided95% descriptive intervals;
these are not new multiplicity-corrected local-gain claims.

This comparison answers the cumulative-benefit question. The separate
`final_target.py` experiment answers whether the candidate's offload latency
meets the user-confirmed tolerance against its own resident placement. Neither
helper automatically accepts a source or completes the overall goal. The final
cumulative figures still need actual audited measurements and rendering.

Validation: seven CPU tests cover3/6-block plans, rejected source/changed job/
missing cell/count guards, old output rejection, the real shared-driver handoff
with fake benchmark children, exact A0000 ratios and refusal to rerun completed
work. Synthetic evidence stays in pytest temporary directories.
