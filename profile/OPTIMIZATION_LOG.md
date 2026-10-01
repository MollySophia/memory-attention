# Optimization log

Running record for Memory Attention experiments. One entry per optimization
step, newest last. Each entry must be reproducible from the stated commit.

## How to read an entry

- **commit** — the exact tree that produced the numbers. Re-check out and rerun.
- **env** — from `bench_fla.py`'s fingerprint. Numbers are not comparable across
  different `torch` / `flash_attn` / GPU rows.
- **prefill / decode** — median ms at 24 layers / hidden 2048 / 32 heads /
  batch 8 / seq 2048, BF16, against the `ma_gpu` reference (1.00x).
- **spread** — (max−min)/mean across the 5 per-round means. Descriptive
  variability only; acceptance requires independent paired runs, not a spread threshold.
- **correctness** — gate status from `profile/test_memory_offload.py`. Prefill,
  decode and left-padded batches, each against the resident folded path.

## Gate

```bash
python profile/test_memory_offload.py          # must print PASS, exit 0
```

Covers: prefill (bulk + pipeline), decode with a real KV cache, and a
left-padded batch that exercises the unpad/varlen attention branch.

## Benchmark

```bash
# single configuration
python profile/bench_fla.py --mode prefill --variants ma_gpu ma_offload \
    --batch-size 8 --seq-len 2048 --context-len 2048 --json out.json

# sweep (isolated process per variant/mode/shape)
python profile/sweep_bmk.py --bmk bench_fla.py --output sweep_results \
    --reference ma_gpu --batch-sizes 8 16 32 --lengths 2048 4096
```

`--reference ma_gpu` is required for this model: it has no `standard` variant
(no `v_proj`), so `ma_gpu` — the resident, norm-folded table — is the baseline
where the only difference is the transfer itself.

### Environment convention

Each sweep writes `<output>/env.json` recording the git commit, whether the
tree was dirty, source SHA-256, torch / torch_cuda / flash-attn / fla versions, python, GPU
name, capability and count. On a later run against the same directory:

- any difference in those fields is **rejected** with a diff of what changed,
  because a different GPU or torch build moves timings by more than any
  optimization under test;
- the environment is folded into every job signature, so `--resume` cannot
  reuse rows measured elsewhere;
- the source hash covers `fla/**/*.py`, profiling scripts and package metadata,
  including uncommitted changes; generated results are excluded;
- pass `--allow-env-change` to deliberately re-measure in a new environment.

**One sweep directory = one environment = one experiment.** When comparing
across optimization steps, compare the summary CSVs and check that their
`env.json` files agree; if the commit differs, the numbers are from a different
tree and are not directly comparable. Record the commit in the log entry.

## Baseline pending

Previous benchmark artifacts were removed on 2026-10-01 at the user's request
so the paper experiments can start with a clean dataset. The tracked historical
sweep remains recoverable from commit `544b760`; it is not the new baseline.

Follow `GOAL.md`: finish and freeze the measurement protocol, then record a
fresh baseline `A0000` before performance optimization. No current performance
claim or baseline measurement is established by this log.

## Entries

One per optimization step, newest last. Copy this template:

```markdown
### <short title> — <commit>

**Change.** What changed and why.
**Correctness.** `PASS` / failure, and anything the gate does not cover.
**Data.** `<output dir>` — prefill median vs baseline, decode median vs
baseline, spread, GPU parameters.
**Verdict.** Accepted / rejected / within_noise, with independent paired evidence.
```

### Decode timing boundary correction — 69d692d (2026-10-01)

**Change.** Move fixed-context KV-cache rollback and test-token selection before
the decode timer in `bench_fla.py`. Model execution, including its own KV-cache
update, remains timed. Warmup still resets the context before every call.
New decode rows record
`decode_timing_protocol=fixed_context_v2_rollback_excluded`.

**Correctness.** All 9 tests in `tests/test_decode_benchmark.py` and
`tests/test_memory_offload_regressions.py` pass in the `fla-bench` environment.
The timing regression assigns 100 ms to setup and 2 ms to model execution,
checks that only the 2 ms is reported, and verifies that every warmup/timed
decode begins at the same cache length across multiple rounds.

**Data.** No new performance comparison. Existing decode results include
rollback and token selection; they belong to the previous timing protocol.
Re-measure both baseline and candidate before comparing under the new protocol.

**Verdict.** Measurement correction, not an inference speedup.

### A0000 — baseline protocol preparation (2026-10-01)

Registered before harness edits on `attempts/paper-001`. Parent: `81684d3`.
No performance measurement yet; status pending, performance unavailable.
First preparation change separates cached last-token prefill from historical
full-logits/no-cache prefill and preserves raw samples. This is harness work,
not a model optimization. Protocol remains unfrozen until all GOAL.md gates
and reporting requirements are implemented. See `results/optimization/A0000/attempt.json`.

Preparation validation: 13 regression tests and the standalone offload gate
(commands recorded in `A0000/validation.json`). Initial new test used a
legacy cache field and failed twice after the logits checks passed; retained
that output, fixed the test to iterate the public cache interface, and reran.
No model arithmetic changed. Remaining protocol work is explicitly tracked;
these tests do not establish a baseline or full primary-shape correctness.

### A0000 — telemetry and growing-cache preparation (2026-10-01)

Source: `e13d2a86dbc98b42ada90ca007dd5edce9ebb7c7`. No model implementation changed.
Added untimed before/after GPU allocation/reservation and measured-region
peaks, process RSS/high-water (explicitly process-lifetime), raw table snapshots,
KV backing storage and all cached offloader capacities. Record source hash,
full commit, source patch, CPU/thread settings and before/after GPU telemetry.
Decode now preallocates the actual context length even if seq_len differs.
Outputs remain alive through synchronization; their release is outside timing.
Added generation timing including prefix plus 128 predetermined GPU-token
steps; no sampling, fresh cache for each repeated trajectory.

Validation: **23 passed** plus standalone **PASS**. Growing-cache comparisons
cover two seeds, bulk/pipeline, partial groups and slot reuse, non-unit norm,
QK normalization/gating, padded and unpadded prefixes. Exact logits, hidden
states and valid KV agree. Initial full padded-K comparison failed because
rotary uses empty_like and masks stores at negative padded positions; these
undefined masked entries are excluded. Unpadded KV remains fully compared.
Both the failure output and final results are preserved.

Small-model CLI smoke runs exercised all three modes and both placements;
KV capacities and all cached offloader sums were checked from emitted JSON.
These 1-warmup/2-sample/2-round smoke runs are **not baseline measurements**
and support no performance claim. Details: `A0000/telemetry-validation.json`.
Next: primary-shape correctness, explicit failure/OOM records and frozen
matrix orchestration, then freeze A0000 and perform formal measurements.

### A0000 — frozen paper_v1 baseline launched (2026-10-01)

Candidate SHA: `d949640ebf2f13f56021bd08c5c9f10e65d571c3`. Protocol: `A0000/protocol.json`.
Independent frozen checkout: `/home/molly/workspace-memory-attn/baseline-paper-001`.
26 regression tests pass, standalone gate passes, and committed-source primary
check passes: batch8/length2048, 24 layers, plus 3 growing decode steps,
296 exact tensor comparisons covering logits, all hidden states and complete KV.
Small-model 128-step multi-seed padded/unpadded coverage was retained.

R01 runs 48 isolated jobs (7 unique batch/length shapes x 2 modes x 3 placements,
plus 2 generation batches x 3 placements), 30 warmups, 30 samples, 5 rounds.
Each job records its PID, command, return code, raw results and log. Exceptions
produce structured oom/benchmark_failed status; missing process output is a
failure, never a zero measurement. The complete plan is preserved.

R01 is running; no baseline estimate or optimization claim yet. See
`A0000/R01/manifest.json` and `A0000/R01-controller.txt`; poll the controller
and current child PIDs before taking any restart action. This campaign will
profile then evaluate at most three focused candidates before full-matrix
confirmation; failure/noise never satisfies the optimization objective.

### A0000 — R01 primary observations and reporting (2026-10-01)

All six primary-shape jobs completed with 150 samples each. Median of round
means (ms), prefill / decode: ma_offload **210.7232 / 11.0559**;
ma_gpu **209.9147 / 10.9418**; ma_gpu_unfolded **212.9333 / 11.0427**.
Peak allocated GPU GiB: offload **6.448 / 5.747**, folded resident
**9.151 / 8.449**. This is one baseline run, not paired candidate evidence or
an optimization acceptance. The rest of R01 is still running.

Added `profile/report_paper_matrix.py`: validates frozen SHA, workload/model
configuration, 150 finite positive raw samples, recomputed round means and
latency/throughput before export. Missing/running/failed/OOM points remain
null, with annotations instead of zero-valued plot points. Error bars are
round-mean ranges, explicitly not confidence intervals. Two CPU-only report
regressions pass (corrupt aggregates/source rejected, missing values retained).

`reports/partial-02` is an explicitly PARTIAL snapshot with 5 completed jobs
at collection time, source CSV/JSON and six figures in PNG/SVG/PDF. Later
completed raw jobs are stored separately; final plots must be regenerated.
The current process manifest snapshot is `R01-checkpoint-01.json`.
No baseline source changes or additional GPU jobs were made during R01.

### A0000 — separate profiling harness prepared (2026-10-01)

R01 remains live, with no failures at this checkpoint. Authoritative controller
and active child processes were inspected; no restart performed. New completed
raw jobs are preserved with `R01-checkpoint-02.json`.

Added `profile/profile_paper.py` for one primary model call after 30 warmups,
separate from headline timings. It will export compressed Chrome CPU/CUDA
trace, operator counts/shapes/device and host durations, memory and source/env
metadata. It refuses to run while the baseline matrix has pending/running jobs;
that guard was exercised (expected exit1, no model built or output directory).
Syntax and CLI help pass; actual trace export remains unverified until R01 ends.
Four exact commands are stored in `profiling-plan.json` and must run serially.

Static observation: both supported KV cache implementations call torch.cat
for each decode layer. This is a profiling hypothesis, not a demonstrated
bottleneck or an accepted optimization. No algorithmic candidate was changed.

### A0000 — continued verified matrix wait (2026-10-01)

Controller PID 1432876 and active child verified live repeatedly; no restart.
Checkpoint counts: {"completed": 20, "running": 1, "pending": 27}. All completed raw records pass
frozen-protocol checks. Batch16/length2048 ma_offload prefill median of round
means is 427.7873 ms, peak allocated GPU 10.5274 GiB. This is baseline scaling
data, not a candidate gain. New raw data and R01-checkpoint-03.json preserved.
Profiling remains deferred until matrix completion.

### A0000 — complete batch sweep, length scan running (2026-10-01)

All 24 batch-sweep jobs (batch1/4/8/16, length2048, prefill/decode, three
placements) completed without failure and passed raw-data validation.
Summary: `batch-sweep-summary.json`; exact live-state snapshot: `R01-checkpoint-04.json`.
Controller PID 1432876 and current child verified live. Length512 scan has
started; length4096/8192 and generation still pending. No runtime restart.
No candidate speedup claim: these are frozen-baseline scaling measurements.

### A0000 — 512-length scan checkpoint (2026-10-01)

Counts: {"completed": 30, "running": 1, "pending": 17}. All completed records validate against frozen
source/configuration and recomputed raw-sample statistics. Controller and active
child verified live; no restart or concurrent GPU profiling. Details and process
observation: `R01-checkpoint-05.json`. Length4096/8192 and generation remain
required before baseline completion. This turn is continued verified waiting
plus preservation of newly completed measurement evidence.

### A0000 — 4096-length checkpoint (2026-10-01)

Counts: {"completed": 36, "running": 1, "pending": 11}. All completed measurements pass frozen-protocol
validation. Controller/current child verified live; original run continues
without restart or additional GPU work. New raw results and live process
observation are preserved in R01 and `R01-checkpoint-06.json`. The remaining
8192-length and growing-generation workloads are still required.

### A0000 — 8192-length checkpoint (2026-10-01)

Counts: {"completed": 42, "running": 1, "pending": 5}. Completed raw records validate against frozen
source/configuration and recomputed statistics. No completed measurement has
failed so far. Controller and active child verified live; no restart or
concurrent GPU profiling. Raw 8192-length measurements and process observation
preserved in `R01-checkpoint-07.json`. Generation remains required before
completion of the baseline and subsequent profiling/candidate experiments.

### A0000 — placement scaling audit while generation runs (2026-10-01)

All 42 fixed-workload records validate. `placement-scaling-summary.json`
compares each offload point with the folded resident point of identical shape.
Offload peak allocated GPU savings range from 1.952 to 2.828 GiB. Small
prefill shapes have larger offload overhead: batch1/length2048 latency ratio
1.514; batch8/length512 ratio1.560 (offload/resident). These single-process
ratios describe baseline placement, not paired optimization gains. Inspect
bulk gather/transfer behavior for these shapes in subsequent profiling.
Generation is still running; controller and current child verified live.
No model change or extra GPU job was introduced.

### A0000 — batch1 generation checkpoint (2026-10-01)

Counts: {"completed": 45, "running": 1, "pending": 2}. Completed generation raw samples validate;
KV storage matches prefix2048 + 128 decode tokens. Batch1 offload generation
median of round means is 657.4702 ms; folded resident is 598.0860 ms.
Both include prefix and exclude sampling; these are baseline placement data,
not paired optimization gains. Raw files and live process observation saved
in R01 and `R01-checkpoint-08.json`. Remaining jobs continue without restart.

### A0000 — batch8 offload generation complete (2026-10-01)

J46 completed: median of round means 1378.8977 ms for prefix2048 + 128
predetermined decode tokens, batch8. All 150 raw samples pass protocol
validation; final KV storage matches the 2176-token context. No sampling.
Folded-resident J47 is confirmed live; unfolded J48 remains pending.
This is baseline data only; full-matrix audit and profiling are still pending.

### A0000 — full frozen baseline complete and audited (2026-10-01)

All **48 jobs / 7200 raw samples** completed, with exit0 and no OOM, missing,
or failed point. Controller and final child exited. Audit confirms one frozen
source hash, one software environment, unchanged implementation, exact workload
configuration, recomputed estimates and final generation KV capacities.
`baseline-audit.json` records evidence; `attempt.json` accepts A0000 as step0
reference only. This is not an accepted optimization or task completion.

Final baseline figures: `reports/baseline-final/`, six PNG/SVG/PDF exports plus
source CSV/JSON; latency estimator and generation scope labels clarified.
GPU temperature/clocks vary over the long matrix and are retained per job.
Future gains still require alternating independent baseline/candidate pairs.

Started separate primary decode diagnostics for offload/resident. Both Kineto
captures returned but exported **CPU events only**, with no GPU kernel events
and zero operator device time; those zeros are not GPU latency measurements.
Raw traces preserved and limitation recorded in `diagnostics-audit.json`.
Committed external Nsight capture support, then launched single-call offload
decode capture (source d60be57); controller/child verified live. Actual Nsight
GPU event availability remains unverified. No model optimization made yet.

### Post-A0000 tooling — staged sampling requested by user (2026-10-01)

Updated GOAL.md and scripts without changing model arithmetic or paper_v1
measurement scope. `bench_fla.py` defaults to screening 3/5/1 for prefill/decode;
generation uses 2/5/3. Confirmation/full-validation single calls use 10/10/3.
Stage, plan ID, sampling unit and actual counts are recorded; explicit count
overrides produce custom plan IDs. `--stage legacy` preserves 30/30/5 sampling.
Exact A0000 source remains available in its frozen checkout.

Matrix driver defaults to four primary screening jobs (20 samples total).
`--stage full_validation` retains all 48 points (1350 samples total);
`--plan-only` writes exact commands and work counts without GPU execution.
Confirmation stage generates one side of a run, not an independent-pair claim.
Report validation uses the declared per-job plan; old metadata-free A0000
results remain validated against legacy counts. Screening figures are labeled
and absent off-scope shapes remain not_run.

Validation: 30 CPU tests passed for CLI defaults/overrides/errors, command-plan
agreement, timing boundaries, raw-sample integrity and old/new report schemas.
Re-audited all 48 A0000 jobs / 7200 samples unchanged. Evidence and plan previews:
`profile/results/protocol/staged-sampling/`. No performance rerun or A0001 model
candidate was started by this tooling update.

### A0000 diagnostics — event fallback resolves KV-copy hypothesis (2026-10-01)

Nsight child was blocked on pipe read with a zombie `file` child before model
capture; suspected numexpr platform.architecture import probe. Preserved process
evidence, intentionally terminated the failed diagnostic child, then killed
remaining own controller/agent after TERM failed. This was a diagnosed stalled
capture, not a benchmark timeout/restart. Kineto minimal matmul probes with
bundled and toolkit CUPTI also returned no GPU events (logs preserved).

Committed diagnostic CUDA-event backend: scoped torch.cat instrumentation,
restored afterward, no model arithmetic change. Primary offload decode reports
48 cats, 3,222,798,336 output bytes, 5.9039 ms summed cat stream spans versus
11.2209 ms total stream span. Intervals include instrumentation/CPU launch gaps;
not pure kernel times or headline performance. Resident diagnostic is running.

### A0001 — reusable KV append capacity registered (2026-10-01)

Parent A0000. Hypothesis: avoid full-history copies on each decode by appending
new K/V into reusable bounded capacity, lazily allocated on decode. Register
before implementation; candidate SHA and performance unavailable until code and
gates are ready. All numeric comparisons remain exact against an independent
concatenating reference; preserve training/sliding-window behavior and report
spare-capacity memory. Details: `A0001/attempt.json`. Start with staged screening,
then independent confirmation/full validation only if warranted.

Resident decode event diagnostic completed: 48 cats, same 3,222,798,336
output bytes, 5.9004 ms summed cat spans / 11.1230 ms total. This corroborates
a model-wide copy cost rather than offload-only transfer. All trace/event
artifacts and failures retained; no candidate timing claim yet.

A0001 implementation: modern inference cache owns 128-position capacity chunks,
lazily allocated on first append; prefill, training and window paths unchanged.
Legacy cache versions fall back to existing concatenation. Legacy exports clone
active values to prevent cross-branch overwrite; cache transforms invalidate
ownership and offload/prefetch release owned buffers. Independent reference
loads cache updates directly from the frozen baseline Git object.
39 regression tests passed, covering GQA, two seeds, left padding, both offload
policies, 128 growing steps, rollback/reorder, gradients and cache ownership.
Standalone gate PASS (worst difference zero). Initial large-rollback spare
capacity failure was fixed; original failure log retained. No timing yet.

A0001 full primary gate passed: 296 exact tensor comparisons across prefix2048
and three decode calls. R01 screening refused the uncommitted controller before
launch. R02 exposed an import-routing error: frozen baseline script imported
candidate model through the environment's editable install. Its decode KV
capacity matched the candidate (3.1875 GiB), confirmed by an import-path probe.
Stopped controller and active child; preserved R02 as invalid for comparisons.
Controller now sets checkout-specific PYTHONPATH and verifies actual module
path before each process. R03 will rerun the eight short measurements.

A0001 R03 screening completed with independently verified import paths: 8 jobs,
40 samples (3 warmups /5 samples /1 round). Matching scope/counts audited.
ma_offload prefill: 209.9285 -> 210.1527 ms; provisional ratio 0.999x; peak GPU 6.4485 -> 6.4485 GiB.
ma_offload decode: 11.0036 -> 5.9567 ms; provisional ratio 1.847x; peak GPU 5.7469 -> 5.8082 GiB.
ma_gpu prefill: 206.1965 -> 206.2933 ms; provisional ratio 1.000x; peak GPU 9.1508 -> 9.1508 GiB.
ma_gpu decode: 10.9145 -> 5.3349 ms; provisional ratio 2.046x; peak GPU 8.4494 -> 8.5106 GiB.
Screening is promising, not an accepted gain. No accepted_step advanced.
Next: >=3 independent alternating baseline/candidate confirmation pairs using
10/10/3, then full matrix/generation only if primary confirmation passes.
R02 remains invalid and excluded. Structured summary and CSV retained.

A0001 formal confirmation plan frozen before measurement: three independent
alternating process pairs per primary placement/mode (24 processes total),
10 warmups and 10 samples x3 rounds. Per-process estimator remains median of
round means. Report three paired ratios, geometric mean and 95% Student-t
interval on log ratios (df=2). Require decode interval above1 for repeatability;
a primary prefill interval wholly below1 is a resolved regression. Unresolved
intervals remain unresolved. Expected cost 8-12 minutes from R03 setup durations
and about103 seconds of prefill work. No full matrix/generation yet. Verify
checkout-specific imported modules and expected commits for every run.

A0001 R04 confirmation completed in 9.48 minutes: 24 independent
processes, 720 raw samples. All source/config/import/count/statistical audits passed.
ma_offload prefill: paired geometric speedup 0.9995x, 95% paired-log t interval [0.9876, 1.0114], within_noise.
ma_offload decode: paired geometric speedup 2.0252x, 95% paired-log t interval [1.9964, 2.0545], repeatable_improvement.
ma_gpu prefill: paired geometric speedup 0.9991x, 95% paired-log t interval [0.9932, 1.0050], within_noise.
ma_gpu decode: paired geometric speedup 2.0498x, 95% paired-log t interval [2.0246, 2.0753], repeatable_improvement.
Intervals use three process pairs (df=2), not within-process rounds; small-n
normal-log-ratio assumption is explicit. GPU peak offload saving stays above
2.7 GiB in every primary pair; decode peak increase about63 MiB. Raw environment
snapshots retain a separate trm-mcp process holding654 MiB; snapshots do not
provide continuous monitoring. Candidate nominated for full validation only;
status/accepted_step remain unset. Full validation plan:88 new processes plus
8 reused first-pair primary results, ~40.6 minutes from measured setup/model
costs; non-generation10/10/3 and generation2/5/3. All three primary pairs remain
available for uncertainty. No full matrix or generation performance started yet.

A0001 full-validation controller: 48 points per side, 96 total records; reuse
the eight first-pair primary records from R04 and launch88 new processes.
Verify source hashes identical to confirmation before launch, plus actual
checkout import paths and exact commits. Alternate baseline/candidate order
across points. Non-generation10/10/3, generation2/5/3. Keep OOM/unsupported
and failed points explicitly; investigate regressions before acceptance.
Estimated cost40.6 minutes, dominated by repeated full-model setup. No source
or candidate changes during timing. Conditional plan from prior stage retained.

A0001 R06 stopped deliberately after18 new completed jobs (plus8 reused),
with one active job interrupted and69 pending. Batch1/context2048 decode was
4.56296 ->4.77310 ms offload and4.57304 ->4.76848 ms resident, with nonoverlapping
round ranges in both. These are signals, not independent-process confidence.
Saved all raw data and interruption evidence before terminating own controller
and active child. No timeout/restart. Preserve candidate code for investigation.
Predeclare R07: reuse those first pairs and add two independent pairs per
placement in alternating order (8 new processes, estimated3 minutes),10/10/3.
Use paired-log95% t intervals (df2), as in R04; upper bound below1 confirms
regression. Do not complete the expensive matrix for a rejected candidate.
No accepted step advanced. Full-matrix audit script and environment checks
prepared; report scripts are outside the benchmark source fingerprint.
