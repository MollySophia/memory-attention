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
