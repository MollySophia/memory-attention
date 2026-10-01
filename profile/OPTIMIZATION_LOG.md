# Optimization log

Running record for Memory Attention experiments. One entry per optimization
step, newest last. Each entry must be reproducible from the stated commit.

## How to read an entry

- **commit** — the exact tree that produced the numbers. Re-check out and rerun.
- **env** — from `bench_fla.py`'s fingerprint. Numbers are not comparable across
  different `torch` / `flash_attn` / GPU rows.
- **prefill / decode** — median ms at 24 layers / hidden 2048 / 32 heads /
  batch 8 / seq 2048, BF16, against the `ma_gpu` reference (1.00x).
- **spread** — (max−min)/mean across the 5 per-round means. This is the noise
  floor. A change smaller than the spread is not a result.
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
**Verdict.** Better / worse / within noise, against the spread column.
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


## Campaign offload-gap-001 (new, independent of paper-001)

### A0000 — prepared baseline, accepted_step 0

Campaign ID: `offload-gap-001`. Parent restored revision:
`81684d34e44db04a0511698081cab336abf4a60d`. No archived model optimization
or archived timing has been imported. Only measurement infrastructure was
ported and reviewed; `fla/` remains identical to the restored revision.
New protocol: `offload_gap_v1`; initial plan: `screen_v1_w3_n5_r1`.

Prepared last-token cached prefill, fixed-context decode (rollback/token
selection excluded), 128-step generation, raw samples, memory/environment
snapshots, failure persistence, plan-only commands, and local-checkout import
routing. Screening defaults cover batches 1/8 in both placements; initial
baseline additionally covers 4/16 (16 jobs), pipeline candidates add 16.
Diagnostic profiling supports batches 1/8 and producer-thread CPU gather,
ID staging, H2D stream spans and acquire wait instrumentation. No inference
model source changed.

Verification: 42 tests pass; standalone correctness PASS (max difference 0).
The added 128-step trajectories cover two seeds, bulk/pipeline, GQA, non-unit
norm, gating/QK norm, group boundaries, single-slot reuse, repeated forwards,
left padding and shape switches. Initial diagnostic failures were restricted
to uninitialized left-padding K cells: rotary.py masks stores for negative
positions, and attention excludes those cells. Tests compare all valid KV
cells (all cells in unpadded cases), logits and hidden states bit-exactly.
The failed diagnostic log is preserved, not a model optimization attempt.

Initial screen planned work: 16 processes, 80 samples, 128 warmup/timed calls
plus decode prefixes. Initial estimate ~546 seconds assumes 30-second setup,
1000-ms prefill and 30-ms decode; these are provisional planning inputs until
measured per-process costs are available. Full validation/generation/confirmation
are intentionally not run at this stage. No gain claim is made.


A0000 primary screen and diagnostics: source frozen at
`62942a0` (full SHA and independent checkout in campaign manifest).
Batch-1 prefill: offload 43.603 ms, resident 28.813 ms, signed gap +14.790 ms
(+51.3%). Batch-8 prefill: 209.162 vs 205.555 ms, gap +3.607 ms (+1.8%).
Decode gaps: +0.054 ms at batch 1 and +0.092 ms at batch 8; short screens
cannot resolve these small differences as gains or regressions.

Paused the timing controller after the primary screen (one batch-4 child was
already launched and allowed to finish). Verified that child was terminal
before diagnostics; resumed the same controller after all 12 diagnostic jobs.
All eight PyTorch traces are CPU-only: unavailable device timing is not zero.
Working CUDA-event H2D instrumentation plus producer-thread CPU instrumentation
shows batch-1 prefill synchronous gather 11.166 ms and H2D span 3.497 ms;
ID staging is 0.028 ms. Batch-8 pipeline overlaps its 75.159-ms CPU gathers
and 41.842-ms transfer spans with model computation; those spans are not
additive. Decode gather spans are 0.024/0.044 ms (b1/b8), supporting retention
of the bulk path for tiny decode inputs. Raw traces, exact commands,
environment snapshots, and diagnostic limitations are preserved under A0000.

Prepared a balanced independent-process comparison driver under the campaign
results directory. Its plan-only audit verifies 48 jobs = 3 blocks x 4 primary
workloads x 2 placements x 2 implementations, each at 10/10/3. This is harness
work, not an optimization attempt. The measured-cost estimate is ~15 minutes;
no confirmation has been launched or claimed.


### A0001 — pipeline above 1024 tokens (campaign_id: offload-gap-001)

Parent implementation A0000, accepted_step null. Registered after all 16
initial baseline jobs and all primary profiles completed. Hypothesis:
batch-1 length-2048 prefill unnecessarily serializes ~11.17 ms CPU gather plus
~3.50 ms H2D with layer computation. Lower the auto-policy bulk cutoff from
4096 to 1024 tokens, keeping the existing bulk path for tiny decode requests.
The 1024 threshold is a predeclared conservative midpoint below the measured
2048-token bottleneck, not a fitted optimum. No arithmetic or resident-path
changes. This is a scheduling experiment, not a generic model speedup.

Screen predeclared: batches 1/8/16, prefill/decode length/context2048, both
placements, 12 isolated jobs, screening 3/5/1. Compare to matched A0000 R01.
Run all correctness gates plus automatic-policy transition checks first.
Only promising results advance to independent confirmation; no acceptance or
accepted-step increment from screening. Full validation/generation intentionally
not run yet. Exact candidate/parent SHAs are recorded in A0001/record.json.


A0001 screen complete: committed candidate `942a2a5`; 43 tests and standalone
gate pass. All 12 isolated jobs completed. Batch-1 prefill offload 29.378 ms,
resident 28.672 ms, gap +0.706 ms versus baseline +14.790 ms. This provisional
screen suggests a 1.484x offload speedup; it is not an accepted/headline gain.
Other primary and batch-16 points show no obvious screening regression;
all offloaded peaks remain at least 2510.5 MiB below matched resident peaks.
Full per-shape signed gaps, dispersion and memory are in screen-comparison.json.

The changed batch-1 prefill diagnostic completes in 30.238 ms (instrumented),
with 24 transfers instead of one. CPU gather spans sum to 3.481 ms and H2D
stream spans to 6.439 ms, overlapping model work. Spans are not additive.
Next: independent balanced baseline/candidate confirmation, 48 processes,
three process blocks at 10/10/3, both placements at all four primary workloads.
Use frozen source worktrees A0000=`62942a0`, A0001=`942a2a5`; this preserves
candidate commit identity across repeated runs while result commits advance.
No accepted step yet. Full matrix, growing generation and full-model frozen
reference fingerprints remain gates after confirmation.


A0001 validation preparation while R02 remains live: added strict reuse of
all three independent primary confirmation blocks for matching full-validation
points. Source commits/hashes, import routing, model dimensions, seed,
last-token/cache scope, raw counts, protocol/plan, package/GPU/driver identity,
CPU affinity and thread environment must match. Thirteen CPU-only harness
checks pass, including rejection of incompatible evidence. Reused raw files
retain their original confirmation stage and commands; they are referenced,
not relabeled or duplicated as fresh timings.

If confirmation nominates A0001, the complete 84-point paired prefill/decode
matrix needs 68 new processes and 16 reused primary points (48 existing
process records), followed by 12 new generation processes and eight independent
full-model correctness processes. Planning cost excludes the known controller
pause in baseline J09 setup. A sequential continuation controller waits for
R02 to terminate, checks the predeclared statistical/memory gate, then records
plans before launching these remaining gates. It stops for failed/inconclusive
confirmation or failed validation and never declares acceptance automatically.
