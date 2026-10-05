# Independent final latency verification

`final_target.py` prepares and audits fresh evidence for the user-confirmed
all16 criterion. It does not select a source, choose a sample size automatically,
or establish overall campaign completion. No real final plan has been prepared
or launched as part of this helper's CPU-only validation.

After choosing a retained source, predeclare at least three independent process
pairs per workload. Counts may differ by workload but must cover all16. Choose
counts before observing the new evidence, using existing diagnostic variability
and the remaining margin for planning. Screening/retention results cannot serve
as final pairs. Do not optionally extend a failed or inconclusive plan.

From the repository root, use the Python environment used for the campaign:

```text
python profile/results/optimization/offload-gap-001/continuation-03/final_target.py prepare --attempt ATTEMPT --blocks N
```

Replace `ATTEMPT` with the selected accepted attempt and `N` with the predeclared
count. Alternatively, pass `--block-counts PATH` containing a JSON list of all16
`mode`, `batch`, `length`, `blocks` entries. Do not use both options. Preparation
records exact commands, the frozen source, audit-helper hashes, job count and
estimated runtime under `ATTEMPT/final-independent-target/`. It refuses to
replace an existing plan.

Commit `plan.json` and the initial `manifest.json` before running. Confirm that
no other GPU benchmark/profile/correctness job is live, then execute:

```text
python profile/results/optimization/offload-gap-001/continuation-03/final_target.py run --attempt ATTEMPT
```

Every matched pair uses fresh consecutive offload/GPU processes; placement order
alternates across blocks. Each process uses the existing formal sampling plan.
The independent sample count is the number of process pairs, not the number of
inner measurements. The runner requires the exact committed plan, unchanged
source/helpers, pending jobs and absent raw result files. It never resumes or
extends a plan implicitly. Execution failures preserve their raw evidence and
require review.

The terminal audit checks complete coverage, balanced consecutive pairs, source,
configuration, sampling, memory and one environment; recomputes process means
from raw samples; and evaluates signed per-pair residuals
`offload - GPU - max(0.1 ms, 0.01 * GPU)`. It uses one-sided95% Student-t upper
bounds with Bonferroni correction over the fixed family16. This parametric model
assumes independent approximately normal process-block differences. Every upper
bound must be at most zero. Analysis JSON and summary CSV preserve every result,
including failures; GPU memory savings are reported separately.

```text
python profile/results/optimization/offload-gap-001/continuation-03/final_target.py audit --attempt ATTEMPT
```

An all-workload latency pass remains separate from overall completion, which
also needs the verified implementation, cumulative A0000 evidence, ledger,
figures, commands and publication requirements in GOAL.md. The tool deliberately
never sets `goal_accepted` true.

CPU-only validation: `test_final_target.py`, 13 passed. The tests include a case
where the mean and ordinary one-sided95% bound pass but the required simultaneous
bound fails; they also reject missing/duplicate coverage, changed commands,
preexisting output and uncommitted/modified plans. Synthetic audit evidence is
confined to pytest temporary directories and does not count as real measurements.
The runner handoff test also executes the real shared driver with fake benchmark
children, verifies all96 processes and the final audit, and refuses a rerun.
It caught and fixed a missing `formal` field assumption introduced by the
variable-block parent-comparison update; no real final measurement was affected.
