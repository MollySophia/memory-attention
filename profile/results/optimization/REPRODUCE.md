# Reproducing the paper-001 campaign

Use the dedicated `attempts/paper-001` branch and the Python environment at
`/home/molly/miniconda3/envs/fla-bench/bin/python`. Model and protocol details
are fixed in the repository's `GOAL.md`; all experiments use seeded random
weights and establish numerical equivalence/performance, not text quality.

## Source and imports

Keep a sibling checkout named `baseline-paper-001` at
`d949640ebf2f13f56021bd08c5c9f10e65d571c3`. If it does not exist, from the
campaign checkout run:

```sh
git worktree add --detach ../baseline-paper-001 d949640ebf2f13f56021bd08c5c9f10e65d571c3
```

Set `PYTHONPATH` to the checkout being measured for every benchmark child.
Changing only the working directory is insufficient: an editable installation
can otherwise import the candidate model while running the frozen script.
A0001/R02 exposed this error; its results are retained but invalid for comparison.
Current paired controllers set the path explicitly and record an import probe.
Do not modify model source or advance the campaign HEAD while a controller is
running; several controllers verify the exact expected commit before each job.

## Correctness

From the candidate checkout, run the three GOAL.md gates and the candidate's
additional cache tests before timing:

```sh
/home/molly/miniconda3/envs/fla-bench/bin/python -m pytest -q tests/test_memory_offload_regressions.py tests/test_decode_benchmark.py tests/test_memory_kv_cache.py
/home/molly/miniconda3/envs/fla-bench/bin/python profile/test_memory_offload.py
/home/molly/miniconda3/envs/fla-bench/bin/python profile/verify_primary.py --output /tmp/memory-primary-check.json
```

`profile/frozen_cache_reference.py` loads the baseline cache updates directly
from the frozen Git object. Growing-cache comparisons therefore cannot pass
merely because both placements accidentally use the same new cache logic.
A0002 adds an explicit batch1 fallback gate; its helper must not be called.

## Timing and raw records

Use a new output directory on every invocation. Reproduce A0002 screening with:

```sh
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/A0002/run_screening.py /tmp/a0002-replay/R01-screening
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/A0002/analyze_screening.py /tmp/a0002-replay/R01-screening
```

The manifest preserves all child commands, process IDs, exact source commits,
import paths, stage/counts, result statuses and timestamps. Each JSON retains
raw samples, round means, model/config/environment and memory snapshots.
Screening uses3 warmups,5 samples,1 round. Formal non-generation measurements
use10 warmups,10 samples per round,3 rounds; generation uses2 full-trajectory
warmups,5 trajectories per round,3 rounds. Frozen scripts lack the new stage
argument; controllers pass the matching counts explicitly without altering
original baseline metadata. Independent confirmation requires separate paired
processes, not merely several rounds in one process.

## Attempt ledger and interpretation

- A0000: original completed48-point baseline,7200 samples under30/30/5.
  Preserve it as recorded; new reduced-plan comparisons require matched counts.
- A0001: unconditional bounded KV reuse. Primary confirmation showed about2x
  decode speedup, but batch1 resident decode regressed in three independent
  pairs. Rejected and reverted; all evidence and `candidate-model.patch` remain.
  R06 was deliberately interrupted to investigate, then remaining points were
  marked not_run. No growing-generation performance was collected for A0001.
- A0002: batch1 retains concatenation; larger batches reuse bounded capacity.
  Its current phase/outcome is authoritative in `A0002/attempt.json`. A passing
  screen alone is never an accepted optimization.

Analyses describe round ranges separately from process-pair confidence
intervals. Final acceptance and publication figures require full matrix,
growing-generation validation, unresolved-regression review and all correctness
gates. Never relabel partial reports or a rejected attempt as final evidence.

## Full validation and regression followup

A0002/R06 contains all96 baseline/candidate records (48 each),2700 samples.
Nineteen records were reused only after checking source, scope, counts and
recorded environment. Reused paths are explicit in the manifest. Candidate
A0001 records are never reused as A0002 evidence. To audit these stored data:

```sh
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/A0002/analyze_full_validation.py --run profile/results/optimization/A0002/R06-full-validation --output /tmp/a0002-full-audit
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/A0002/audit_memory.py --run profile/results/optimization/A0002/R06-full-validation --output /tmp/a0002-memory-audit.json
```

The full audit recomputes raw-sample statistics and validates each imported
implementation, source/configuration, environment and sample count. Memory
checks cover all96 records and16 candidate offload/resident pairs, including
CPU tables, pinned allocations, GPU staging buffers and backing KV storage.
Host RSS includes setup and allocator effects; offload GPU savings do not imply
host-memory savings.

The four descriptive round-range regression signals are predeclared in
`A0002/R07-followup-plan/manifest.json`. R08 measures three fresh alternating
pairs per signal,24 processes total. Initial full-matrix points are retained
and excluded from the fresh paired analysis. Reproduce this followup only with
clean committed source, after all other GPU work has exited:

```sh
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/A0002/run_regression_followup.py --audit profile/results/optimization/A0002/reports/R06-final-audit/source.json --output /tmp/a0002-replay/R08-regression-followup
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/A0002/analyze_regression_followup.py /tmp/a0002-replay/R08-regression-followup
```

The followup analyzer writes its summary beside the run directory. The exact
child commands, alternating order, independent process identities and source
commits remain in the manifest. Its confidence interval is a95% Student-t
interval on three paired log ratios (df2); this small-sample assumption must
remain stated. A confidence interval containing1 is within_noise, not proof
of equivalence. Do not extend the run selectively until it looks favorable.

The new asynchronous correctness gate queues snapshots on GPU and synchronizes
only at trajectory completion; existing per-step CPU snapshots otherwise
introduce synchronization absent from generation timing. It checks128 steps,
batch1/2, bulk/pipeline, partial groups and cache-capacity boundary crossing:

```sh
/home/molly/miniconda3/envs/fla-bench/bin/python -m pytest -q tests/test_memory_offload_regressions.py -k async_generation
```

Do not run correctness tests or render figures concurrently with GPU timings.
All generation timings use predetermined device-resident token IDs and exclude
sampling. They measure prefill plus128 model calls, not end-to-end serving.

## Final figures

After timings exit, render into new output directories:

```sh
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/plot_final_comparison.py --audit profile/results/optimization/A0002/reports/R06-final-audit/source.json --output /tmp/a0002-final-scaling
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/plot_campaign.py --output /tmp/a0002-final-history
```

Both scripts export PNG/PDF/SVG and CSV/JSON. The full-comparison input is the
validated R06 report. Fresh regression-confirmation results remain separate;
no averaging selectively removes an unfavorable initial observation. See
`RESULTS.md` for the accepted verdict, exact implementation revision and limits.

## Matching baseline and final implementation timing

To reproduce independent primary confirmation with both sides and alternating
order (24 fresh processes), use a clean committed checkout and run:

```sh
mkdir -p /tmp/a0002-replay
cp profile/results/optimization/A0002/small-batch-summary.json /tmp/a0002-replay/
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/A0002/run_confirmation.py /tmp/a0002-replay/R03-confirmation
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/A0002/analyze_confirmation.py /tmp/a0002-replay/R03-confirmation
```

The full-validation controller prepares both frozen-baseline and candidate
commands and verifies every reuse against the shipped records. Its plan-only
mode records job counts and estimated cost before launching:

```sh
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/A0002/run_full_validation.py --output /tmp/a0002-replay/full-plan --plan-only
/home/molly/miniconda3/envs/fla-bench/bin/python profile/results/optimization/A0002/run_full_validation.py --output /tmp/a0002-replay/full-run
```

These controllers expect the accompanying campaign records and sibling frozen
checkout; preserve that layout. The already validated R06 evidence can be used
for the accepted identical implementation without rerunning it.
