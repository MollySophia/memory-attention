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
