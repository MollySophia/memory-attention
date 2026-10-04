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


A0001 R02 confirmation design audit: all 48 jobs completed, but this run is
ineligible for formal acceptance. The predeclared gate stopped full validation:
batch-1 resident decode is 0.0614 ms slower for the candidate (95% paired
interval 0.0175–0.1053 ms), despite unchanged resident code, and offload decode
has large between/within-run variability. Batch-1 prefill has a consistent
~13.56 ms reduction, but no accepted claim is made from this flawed design.

Inspection found a concrete controller bug: reversing workload traversal
canceled the intended source and placement alternation. Each placement ran
its sources in the same order in all three blocks. Full order evidence and
all raw timings are retained in R02-design-audit.json and R02-confirmation.
This is a measurement-design failure, not a new optimization attempt.

Predeclare R03 under `formal_v1_w10_n10_r3__balanced_order_v2`: unchanged
source commits, shapes, warmup and sample counts, but stable canonical workload
and placement indices ensure source order and placement order alternate in
every block. Eighteen harness tests pass, including rejection of the original
R02 design and end-to-end validation of all eight corrected pair sequences.
R02 will not be combined with R03 or reused for full validation. Before R03,
collect one instrumented 10/10/3 batch-1 decode diagnostic per implementation
for CPU gather variability; preserve it separately from inference timing.


A0001 R03 corrected confirmation passes the predeclared primary gate. All 48
jobs completed with verified per-workload/per-placement source alternation.
Batch-1 prefill offload reduction: 13.7987 ms, paired 95% interval
[13.5270, 14.0705] ms; absolute-gap reduction: 13.7839 ms, interval
[13.4968, 14.0709] ms; paired offload speedup 1.4634x [1.4561, 1.4708].
Other primary workloads have no resolved offload regression, no resolved
resident slowdown, and retain GPU memory savings. Batch-1 decode remains
noisy: its estimated reduction is -0.1777 ms [-0.6276, +0.2723], so no decode
improvement is claimed. R02 remains excluded from this confirmation.

A0001 is nominated, not accepted. R04 full validation has started with 68 new
processes and 16 reused primary points referencing all 48 corrected R03 jobs.
The complete matrix still contains 84 source/placement/workload points.
R05 generation and R06 independent full-model exactness follow sequentially.
Prepared history/accepted-step plotting now has tests proving that the
incumbent is one whole model, failures are not zero latency, partial valid
measurements survive a failed resident partner, and accepted-step plots require
explicit eligible confirmation. Preview layout inspected; final publication
figures remain pending the completed campaign.


A0001 full-gate audit (2026-10-02): R04 completed all 68 new jobs plus 16
reused primary points; R05 completed all 12 generation jobs. R06 independently
compares frozen resident vs candidate offload at batches 1/8, seeds1234/4321:
all 129 logit checkpoints and selected complete hidden/KV checkpoints are
bit-exact and finite. Minimum matched peak allocated GPU saving is ~1999 MiB
in the full matrix. No OOM or unsupported points were omitted.

Acceptance remains pending: one-process full validation shows prefill increases
at b4/context2048 (+1.18%), b16/context2048 (+0.51%), and b8/context8192
(+0.88%). Those paths have the same policy on both sources, but the observation
needs investigation rather than omission. Batch-1 generation was +6.78% in R05.
A separate two-warmup generation diagnostic reverses that difference (baseline
664.04 ms vs candidate619.91 ms); GC contributes only ~0.02 ms. Aggregate CPU
time is much larger than wall time on both sources. This diagnostic does not
establish a unique cause and is not performance acceptance evidence.

Predeclare R07 generation regression audit: preserve the original b1 pair for
both placements and add exactly two independently alternating pairs per
placement under unchanged 2/5/3 whole-trajectory sampling. Eight new processes,
four reused process records, estimated218 seconds. No outliers removed and no
open-ended extension until favorable. If generation is not a resolved regression,
follow with the same bounded audit at the three flagged prefill points before
acceptance. This is validation of unchanged A0001, not a new optimization attempt.

### A0001 R07 outcome and R08 bounded prefill audit (offload-gap-001)

R07 retained the original generation pair and added exactly two alternating independent pairs per placement. Offload reduction -6.953 ms, 95% paired-block interval [-89.584, 75.678]; resident reduction 2.940 ms [-17.839, 23.719]. No resolved regression; no generation gain claim. All samples and the unfavorable original pair remain. GPU savings preserved in every block.

Per the previously declared conditional plan, R08 audits prefill b4/L2048, b16/L2048, b8/L8192. Each uses the original full-matrix pair plus exactly two alternating pairs per placement. 24 new processes, 12 reused; estimated 880.25 seconds including measured setup. Same formal 10/10/3 plan, no extra extension based on outcome. Original matrices remain immutable. Plan: A0001/R08-prefill-regression-plan/manifest.json.

### A0001 R08 continuation environment unavailable (offload-gap-001)

The continuation entered a restricted execution environment without /dev/nvidia devices; nvidia-smi cannot contact the driver. The previous exec session 54235 is unknown. The current PID namespace cannot establish whether the old host controller still runs, so no duplicate was launched and its manifest is unchanged. Seven of 24 new jobs have completed payloads (J013–J019), including J019 whose manifest still says running. Seventeen new jobs have no payload yet. R08-observation-unavailable.json records the exact observation. Do not treat this partial audit as passing, failing, or an independent three-block comparison. Reestablish GPU access and inspect the original controller before resuming only unfinished jobs under the unchanged plan. A0001 is still pending; the 5–10-attempt objective is unmet.

### A0001 R08 recovery (offload-gap-001)

Unrestricted environment restored. RTX 5090 is visible and idle; original controller PID 1990141 and all prior child PIDs are absent in host /proc. Source SHA/hash, GPU driver, torch/FlashAttention/Python, CPU affinity and thread environment match original data. Resume only 17 pending jobs under their unchanged commands/order, estimated 555.14 seconds. J019 already has a completed payload and is recovered without resampling. Original manifest is preserved as manifest-before-resume.json. No attempt ID, sampling effort, or acceptance gate changes.

### A0001 final verdict: accepted, accepted_step 1 (offload-gap-001)

R08 completed all 24 new processes plus 12 retained originals; interrupted jobs were resumed without changing their plan or resampling complete payloads. Prefill offload reductions (95% independent paired-block CI): b4/L2048 -0.289 ms [-3.049,2.471], b16/L2048 -3.393 [-9.406,2.619], b8/L8192 -3.902 [-18.370,10.567]. These are unresolved, not improvements. All three b16 pairs are slower; explicitly retain this limitation. The bounded plan is now closed, not extended until favorable.

Primary R03 confirms b1 prefill offload reduction 13.799 ms [13.527,14.071], absolute-gap reduction 13.784 [13.497,14.071], paired speedup 1.4634 [1.4561,1.4708]. No other primary or audited secondary point has a resolved offload regression under the predeclared criterion; resident controls do not supply the gain. Full matrix 84 points and generation 12 points complete with no OOM; minimum matched allocated GPU savings 1999.25 MiB and 2767.25 MiB respectively. Four full-model independent-reference comparisons at b1/b8, seeds1234/4321, including 128-step trajectories, are bit-exact. Small tests cover enabled normalization/gating, padding/GQA and slot reuse. Source audit verifies only auto-policy threshold changed.

Accept candidate 942a2a52d94cf6dbe6e753f7ac291daf8b0699fb as the first retained step. Baseline remains 62942a0387608fe21baaeb2dce9ef3b0947dde4d. Derived full-evidence-manifest.json and generation-evidence-manifest.json preserve all original and repeat raw sources. Do not claim decode/generation gains or equivalence of latency beyond the reported uncertainty. Next attempt starts from A0001; 5–10 attempt goal remains active.

### A0002 registration (offload-gap-001)

Parent accepted A0001 (source 942a2a5). Test serial byte copies for CPU bulk lookups of <=8 tokens. Existing full-model diagnostic gather cost is ~24/44 us at b1/b8 decode; new alternating CPU diagnostic is 4.675->2.702 us and 24.557->17.209 us for torch versus serial bytes. Hot microbenchmarks do not prove model speedup. No global thread change, no cached lookup/output, no extra table copy. Preserve original index_select fallback for noncontiguous tables and larger calls. Screen eight primary jobs, both placements, at b1/b8 after exactness tests; transfer volume and pipeline schedule unchanged. The separate layer-major diagnostic did not support that candidate and is not counted as an attempt.

A0002 candidate committed at ac6fbc2cbf534c804bb34227aa3ddee3e08728d9, frozen at /home/molly/workspace-memory-attn/offload-gap-001-A0002. Correctness: 51 pytest cases passed, standalone max error zero. Eight-job screening plan stored in A0002/R01-plan/manifest.json, approximately 207 seconds using ~25 second process setup and representative measured primary latencies (210 ms prefill, 11 ms decode). No formal improvement claim from this plan.

A0002 screen complete: b1 prefill offload/resident 29.437/28.654 ms; b1 decode 4.610/4.608; b8 prefill 209.882/206.334; b8 decode 10.991/10.919. Relative to A0001 screen, b8 decode offload reduction 0.0381 ms and gap reduction 0.0507 ms; b1 decode gap unchanged. Modest potential b8 gain warrants independent confirmation, not acceptance. R02 compares A0001 (baseline label) versus A0002, all four primary workloads and both placements, three balanced independent blocks, 48 processes, 10/10/3 formal samples. Estimated 896.22 seconds from A0002 screen process costs. If parent-relative gain is unresolved, classify within_noise and revert; do not let inherited A0001 prefill gains qualify A0002. If repeatable, frozen-A0000 comparison and full validation remain required.

### Host-memory accounting audit (offload-gap-001)

For audited A0000/A0001/A0002 source, ma_offload retains 3000 MiB folded CPU table, 3000 MiB raw snapshot, and an additional 3000 MiB CPU restore-embedding copy. The last capacity is inferred from unchanged source rather than a named raw telemetry counter; all three are already included in measured process RSS. Folded/unfolded resident placements also retain the 3000 MiB raw snapshot because benchmark setup restores the raw table before placement. Do not add these capacities to RSS. memory-accounting-audit.json preserves source SHA/snippets and the distinction between measured counters and inferred capacity. Existing RSS/memory-saving comparisons need no rerun or numerical correction.

### A0002 final verdict: within_noise, accepted_step null (offload-gap-001)

All 48 parent-relative confirmation jobs completed. b8 decode offload reduction 0.03623 ms, 95% CI [-0.00181,0.07426]; gap reduction 0.02709 [0.00211,0.05207]. Gap improved but the required latency criterion is unresolved. b1 decode offload reduction 0.07366 [-0.01219,0.15952], gap reduction -0.00972 [-0.24752,0.22808]. Prefill unchanged within uncertainty, as expected for a tiny-lookup-only change. No primary meets both required lower confidence bounds. Memory savings preserved.

Do not extend the three-pair plan until favorable. Record within_noise and revert model/test changes after this evidence commit. Source ac6fbc2 and frozen checkout preserve the candidate. Full matrix/generation/frozen-A0000 confirmation intentionally not run after failed parent-relative nomination. A0001 remains the verified incumbent; two optimization verdicts complete.

### A0003 registration (offload-gap-001)

Parent is verified A0001 after A0002 code-only revert. Hypothesis: two-layer automatic pipeline groups halve 24 gather/copy/event handoffs, reducing producer coordination seen in b1/b8 profiles. This is a scheduling tradeoff: first-transfer latency and per-slot capacity double, and microdiagnostic group2 gather can be more expensive. Test actual end-to-end effect; do not add overlapping profiler spans. Add auto minimum group size 2, preserving explicit pipeline group settings and bulk decode. Predeclare 12-job primary+batch16 screen, correctness including partial last group and slot reuse. No candidate acceptance from screen.

A0003 source 986e000b1d973a2a76155ff932aba2d367d03dc0 frozen at /home/molly/workspace-memory-attn/offload-gap-001-A0003. Correctness: 45 pytest cases and standalone exact-zero comparison passed. Twelve-job screen cost ~261 seconds, using rounded measured ~20-second setup, up to420-ms prefill/21-ms decode. The CLI requested group_size remains1; new auto_min_group_size=2 in complete model_config makes effective pipeline groups2. Bulk uses24 layers as before. Preserve this distinction in analysis.

### A0003 final verdict: rejected, accepted_step null (offload-gap-001)

All 12 screen jobs complete. Versus accepted A0001, prefill offload is slower by 2.465 ms at b1, 4.057 at b8, 9.968 at b16; signed gaps worsen 2.469/3.550/10.314 ms. GPU controls do not explain these regressions. Slot capacity also doubles. Small decode fluctuations are not an offsetting supported improvement. Reject at screen; no formal/full/generation escalation.

Separate b1 diagnostic finds 12 gathers total10.240 ms (first0.545), versus A0001 24 gathers total3.481 ms (first0.146). H2D submission CPU sum drops0.379->0.189 ms, but grouping increases gather cost and first readiness wait0.194->0.669 ms. Instrumented wall32.711 vs30.238 ms. Spans overlap and cannot be added; this is supporting diagnosis, not headline timing. Preserve complete profile and source986e000, commit results then restore only config/model/tests to A0001. Next insight: retaining one-layer gathers avoids the grouping penalty; test producer scheduling separately.

### A0004 registration (offload-gap-001)

Start from verified A0001 after A0003 code-only revert. Test caller-thread per-group production instead of a background producer. Keep one-layer gathers, same copy stream, slots, events, transfer bytes and arithmetic. Existing A0001 b1 ticket-acquire CPU sum10.55 ms and baseline b8 sum92.8 ms motivate removing worker/main coordination; do not interpret these overlapping spans as removable wall time. A0003 shows why group enlargement is not included. Cost risk: CPU gather may delay compute submission and each group switches stream context. Predeclare 12-job primary+batch16 screen after exactness tests.

A0004 implementation29a0d4ff761e862423e5cd22a8d2a475d9a99a78 frozen at /home/molly/workspace-memory-attn/offload-gap-001-A0004. All45 tests pass, including cross-stream partial-group/single-slot reuse and error cleanup, plus standalone exact-zero comparison. Twelve-job screening estimate261 seconds, same counts/representative measured costs as A0003.

A0004 screen: offload/resident b1 prefill29.027/28.714 ms, decode4.615/4.573; b8 prefill209.165/206.491, decode11.042/10.931; b16 prefill418.076/415.438, decode20.259/20.247. Incremental b1 prefill reduction0.351 ms and gap reduction0.394 versus A0001 screen. b8 prefill reduction0.298 and gap reduction0.982. b16 prefill difference-0.522 ms with essentially unchanged gap (-0.009 ms reduction). No accepted claim from these screens.

Separate b1 instrumented diagnostic: gather sum3.202 ms versus A0001 3.481; ticket acquisition CPU sum0.186 versus10.555 ms, H2D submit0.128 versus0.379 ms. Instrumented wall29.066 versus30.238 ms. Nonadditive and single-call, supporting only the mechanism.

Declare R02 parent confirmation: A0001 vs A0004, all four primary workloads, both placements, three balanced independent blocks, formal10/10/3, 48 processes. Estimated896.05 seconds from measured A0004 setup/call costs. No extension until favorable. If incremental gain does not pass, revert; otherwise independent frozen-A0000 comparison and full retention gates remain.

### A0004 final verdict: within_noise (gap confirmation insufficient), accepted_step null (offload-gap-001)

All 48 parent-relative processes completed. b1 prefill offload reduction 0.23508 ms, 95% CI [0.05842,0.41173], but gap reduction 0.29297 [-0.11176,0.69770]. The latency criterion passes; the mandatory absolute-gap criterion does not. b8 prefill offload reduction 0.79212 [-0.67606,2.26030], gap reduction 0.78384 [-0.87571,2.44339]. Decode changes unresolved. All estimates and positive latency evidence remain recorded.

No workload meets both lower confidence bounds, so classify the optimization objective as within_noise and close the predeclared plan without extending until favorable. No frozen-baseline confirmation/full matrix/generation escalation. Preserve source29a0d4f, all 48 raw results and profiles, then restore only offloader/test code from accepted A0001. A0001 remains incumbent; four verdicts complete.

### A0005 registration (offload-gap-001)

Start from verified A0001 after A0004 code-only revert. Test compute-stream H2D for bulk calls with <=8 tokens, avoiding copy-stream context switching and a redundant first-layer dependency wait. Baseline b1/b8 decode profiles: H2D submission CPU ~5.5/5.8 us, transfer event spans ~11/22 us. These are small costs; A0002 demonstrates why model-level confirmation matters. Tradeoff: copy now precedes Q/K execution instead of overlapping it. Preserve fresh gather, copied/consumed events, cross-call stream safety and all bytes/precision. Larger bulk and pipeline paths unchanged. Twelve-job screen includes b16 scheduling control, after exactness tests including changing streams and fresh IDs/weights.

A0005 final pre-timing source f33b0e383f2b3d9d7121ea3b92d1e45157aad1de, frozen at /home/molly/workspace-memory-attn/offload-gap-001-A0005. During implementation review, initial revision1f614db was refined to retain allocation-stream ordering on the first inline-copy call; no performance measurement existed at that revision. Its preliminary correctness log is preserved separately. Final revision passes46 tests (9.58s) plus standalone maximum error0. Twelve-job screen estimate261 seconds, same measured representative setup/call assumptions as A0003/A0004. All performance jobs use only final frozen f33b0e3.

A0005 screen complete: offload/resident b1 prefill29.243/28.745 ms, decode4.586/4.648; b8 prefill210.360/206.578, decode11.001/10.931; b16 prefill418.806/415.432, decode20.299/20.255. Relative to A0001, b1 decode offload reduction0.03426 ms and gap reduction0.06262; b8 reduction0.02790 and gap reduction0.05280. b1 resident is also slower, so this cannot support acceptance alone. Unchanged prefill fluctuations are not an A0005 gain. Signed negative b1 decode gap is preserved.

Separate instrumented b1 decode wall4.982 ms, ID staging0.0201, CPU gather0.0234 and H2D submission0.0058 ms; instrumentation does not establish a speedup and does not isolate all context-switch overhead. Preserve this unfavorable diagnostic as well.

R02 parent confirmation plan: A0001 vs A0005, all four primary workloads, both placements, three balanced independent process blocks, formal10/10/3, 48 jobs, estimated894.14 seconds from measured screen process costs. Apply the same dual latency/gap criterion; no extension until favorable. If no target decode workload qualifies, classify within_noise and revert.

### A0005 final verdict: within_noise, accepted_step null (offload-gap-001)

All 48 independent parent-confirmation processes completed. b1 decode offload reduction0.372980 ms, 95% CI[-0.275798,1.021758]; gap reduction0.436864 [-0.218786,1.092514]. b8 decode reduction0.021608 [-0.168654,0.211869]; gap reduction0.003542 [-0.183501,0.190584]. No targeted decode point meets both criteria. Unchanged b1 prefill shows latency reduction0.084529 [0.036308,0.132750], but gap remains unresolved and this unchanged path does not establish the proposed decode mechanism. No resolved primary regression or loss of measured GPU savings.

Close the predeclared plan without extension; no A0000 confirmation/full/generation escalation. Commit all source/evidence before restoring only offloader and regression-test files to accepted A0001. Five focused optimization attempts now have final verdicts, with one accepted cumulative step.

### Final campaign audit (offload-gap-001)

Five focused attempts complete: A0001 accepted; A0003 rejected; A0002/A0004/A0005 within_noise. Current model and tests match verified A0001. Final report and reproducibility commands are under results/optimization/offload-gap-001/final. Twelve figure families export PDF/SVG/PNG and CSV/JSON; accepted-step figures use eligible formal evidence, attempt history uses explicitly provisional screens. Core audit checks160 raw process references for84 full-matrix and12 generation points, exact source/environment/configuration, memory and four full-size exactness comparisons. All rendered figure families reviewed; shell syntax and48/68/12-process reproduction plans checked. No final performance rerun of unchanged source. Limitations include b16 adverse direction, noisy b4 decode, and no decode/generation gain. Final publication uses authorized origin campaign branch only.

### Continuation 01 (offload-gap-001)

User requests another5–10 optimization attempts. Preserve A0000–A0005 and the first final report. Start A0006 from verified A0001, keep frozen A0000 and unchanged protocols/gates. Current GPU/driver and idle654MiB TRM process match previous environment. New coordination diagnostics instrument worker release waits, DMA synchronization, tensor view creation and ticket setup, separately from model timing. Register new hypotheses only after evidence review.

### A0006 registration (offload-gap-001; continuation-01)

Parent verified A0001. Instrumented b1 pipeline release waits total1.176ms (overlapping, not removable wall time). Test refilling pinned host slot after DMA completion but before waiting for prior GPU consumer release. GPU overwrite still waits for consumed event; no extra buffers or cached lookups. Focused schedule change, twelve-job b1/b8/b16 screen, exactness and cancellation/slot-reuse gates first.

A0006 frozen source b58cdb8640fb2d6a57631e0c26fc6f9a72761f74;46 tests and standalone maxerror0 pass. New tests enforce early host refill with one slot and partial groups while retaining live GPU values, plus consumer-error cancellation. b8 diagnostic release waits total0.041ms, DMA waits60.918ms, view calls1.089ms; expectation is mostly b1, not blanket acceleration. Twelve-job screen estimate261sec from prior matching12-job costs (~20sec setup,420ms prefill,21ms decode).

A0006 twelve-job screen complete. b1 prefill offload29.145/resident28.624ms, reduction0.233ms and gap reduction0.186 vsA0001 screen. b8 prefill209.776/206.288, b16 418.220/415.029; offload point estimates slower0.314/0.666ms. Decode b1 4.658/4.626,b8 11.005/10.917,b16 20.295/20.249ms. No accepted gain from screening. Predeclare R02:48 independent balanced processes A0001 vsA0006, formal10/10/3, allfour primary workloads/bothplacements. Require both latency and gap lower95>0 at targetedprefill, no resolved primary regressions, fixed memory criteria. No plan extension until favorable. If passes, A0000 confirmation and fullretention gates still required. Exact cost estimate and order in R02-parent-plan/manifest.json.

### A0006 final verdict: within_noise (offload-gap-001; continuation-01)

48 parent-confirmation processes complete. b1 prefill offload reduction0.148526ms,95%CI[-0.160371,0.457424]; gap reduction0.159473[-0.193442,0.512389]. b8 prefill reduction-0.227075[-1.439697,0.985547], gap-0.045337[-1.801089,1.710414]. Bothdecode changes unresolved; no primary resolves a regression, and GPU savings preserved. No point meets both latency/gap requirements. Keep allsamples and frozen b58cdb8 source; no extension and nofull/generation escalation. Commit evidence then code-only restore A0001. First additional verdict complete.

### A0007 registration (offload-gap-001; continuation-01)

Parent A0001 after A0006 evidence/revert. Cache immutable zero-copy tensor view metadata, not token values. Instrumented96 buffer-view calls total0.316/1.089ms atb1/b8; source and layer slices add operations. Precompute source-group aliases, host/GPU transfer views and per-layer GPU aliases in setup. Keep worker, slot count, DMA/release order and allfreshlookups unchanged. Test alias visibility after weight andID changes, partialgroups and slots, plus existing128-step gates. Twelve-job screen predeclared, formaldualcriterion unchanged.

A0007 source2bcbe8d197ef3244af1f51beb1214e4f3ecf6ffc frozen.47 tests plusstandalone exactzero pass, including fourgroup/depth combinations observingfreshweightsandIDs throughmetadataaliases. Planned12-job screen~261sec.

Separate transfer-alternative microdiagnostics (after A0006 timing exited) pass exact output checks for every method. Single-layer gather/H2D vs mapped-host vsdedup/H2D/expand means ms: b1 .3386/.2356/.5027; b8 3.9410/1.6689/3.9105; b16 7.9514/3.3819/5.7616. FreshsameIDs pertrial,10warmups/10trials, rotatingmethodorder; no modelcomputeoverlap. Entire3000MiB sourcetable pinned for this diagnostic. Per-call dedup cost is included for this singlelayer (a fullmodel would amortize it acrosslayers). These are direction-selection diagnostics, not optimizationattempts or accepted gains. Rawdata andcompilerlogs retained.

### A0007 final verdict: rejected (offload-gap-001; continuation-01)

Twelve screenjobs complete. Offload/resident ms: b1prefill29.411/28.671,decode5.536/4.583; b8prefill209.997/206.480,decode11.051/10.919; b16prefill418.870/415.174,decode20.331/20.269. Parent-relative prefill offload reductions -0.0331/-0.5340/-1.3155ms atb1/b8/b16. b1gap worsens0.0336ms; b8gap improves0.1394ms onlywithslower offload, so no targetqualifies evenasapromisingscreen. Reject withoutformal/full/generation escalation. Unchangedbulk b1decode fluctuation retained withoutcausalclaim. Preserve2bcbe8d source andallresults, thenrestoreonlyoffloader/tests toA0001. Twoadditional verdicts complete.

### A0008 registration (offload-gap-001; continuation-01)

ParentA0001 afterA0007revert. DirectGPUgather frommappedpinnedCPUtable intoexistingboundedGPUslots, using64persistentCTAs onaseparatestream. Independentdiagnostic64-CTA meansb1/b8/b16 .233/1.711/3.374ms vsCPUgather+H2D .339/4.096/7.936; exactallmethods.64chosenbeforemodeltiming tolimitSMfootprint; no fullmodeloverlapclaim. Entire3000MiBfoldedtable pinned, freshreads everyforward, allCPUcopies andpinnedstorage accounted. Keepbulkdecodealgorithm andresidentpath unchanged. Finalreadstream syncbefore return preservesCPUtablemutation safety; cancellation/close/crossstream/partialgroups/CPUandGPU IDs requiregates. Twelve-jobscreen, standarddualprimaryconfirmation andfullretentiongates.

A0008 finalsourcefb17afb2a97b68ec61ef517622707f68ce33e051 frozen beforetiming.50 tests pass9.80sec plusstandalone maxerror0. Initial67395ae implementation differs only by a subsequent standalone print correction toshowactualCPUtablepinning; no earlierperformance. Originalstdout preserved. Newtests cover CPU/GPU IDs,crossstream repeatedforwards, CPUweightmutation immediatelyafterreturn, partialgroups,singleslot,consumererror andtotalpinnedtableaccounting. Twelve-job screen setupestimate30sec usesprior~20sec measuredsetup plus10secnewJIT/pin allowance; prefill420ms/decode21ms remainplanning estimates.

### A0008 final verdict: rejected (offload-gap-001; continuation-01)

All12screenjobs complete. Offload/resident ms: b1prefill29.423/28.758,decode5.354/4.629; b8prefill213.101/206.580,decode11.010/10.927; b16prefill428.715/415.070,decode20.372/20.245. RelativeA0001 prefillslower0.0447/3.6381/11.1612ms; b8/b16gapworse2.8646/11.0158ms. No targetprefill improvement; rejectwithoutformal/full/generation. Mappedread microgains do not establish end-to-end overlap; mechanismbreakdown remainsunmeasured. Whole3000MiBtable pinning plusbulk buffers correctlyrecorded; b1/b8totalpinned3000.09375/3000.75MiB. Commitallsource/evidence thenrevert sixchangedexecution/testfiles includingnewmodule. Threeadditionalverdicts complete.

### A0009 registration (offload-gap-001; continuation-01)

ParentA0001 afterA0008revert. Deduplicate/sortIDs oncewithineachforward, transferunique rowspergroup, reconstruct originalorderusingGPUinverseindex. Noacrosscall lookupreuse. Predeclare>=8192token threshold frommicrodiagnostics: b1perlayer compression costloses, b8nearlyneutral beforeamortizingIDwork, b16benefits. Targetprimaryb8prefill only; unchangedb1/decode cannotnominate anewstep. CountGPUinversebuffer andexpandedMtemporary. Test changingallunique/allidentical/unsortedIDs, freshweights, partialgroups, crossstreamslotreuse and128-stepfullsmallmodelgates. Twelve-jobscreen andoriginaldualcriteria.

A0009 source0ac06a0f7ce6c3ce77fd5a878c4a34c991527ec2 frozen.47tests pass9.57sec plusstandaloneexactzero. Existing128-step gated/QKnorm/GQA/padded tests nowforce dedupthreshold1 forsmallmodel, additionaltests varyunique counts andCPUweights acrossstreams. GPUinverseindexstorage includedinallcachedoffloadercapacities/totalbuffers. Twelve-jobscreen estimate261sec usingmatchingprior measuredsetup~20sec and420/21mscallestimates; no expensivefullmatrix unlessnominated.

### A0009 final verdict: rejected (offload-gap-001; continuation-01)

All12screenjobscomplete. Offload/resident ms:b1prefill29.681/28.734,decode4.716/4.582;b8prefill211.480/206.651,decode11.011/10.930;b16prefill419.581/415.231,decode20.256/20.246. Targetb8prefill slower2.01742ms vsA0001 andgapworse1.17298; b16slower2.02706/gapworse1.72058. Rejectatfixedscreenwithoutformal/full/generation. Within-calltransfercompression doesnottranslate toend-to-endgain; unique/expansionwork andoverlap requireconsideration, noisolatedcausalclaim. Fouradditionalverdictscomplete. Commitallevidence thenrestorefivechangedexecution/testfiles toA0001.

### A0010 registration (offload-gap-001; continuation-01)

ParentA0001 afterA0009revert. Alternate two pinnedhostbuffers foreachunchangedGPUslot. ProfileproducerDMAwait12.763/60.918ms b1/b8 motivatesmoreCPUlookahead. PreserveA0001 release-before-gather ordering, isolatinghostreusefromA0006earlyrefill. Separatehostcompletionevents, one eventrecordpercopy, consumerpointsselectedevent. Hostpinningdoubles;GPUcapacityandtransferbytes unchanged. No earlierrejectedoptimizations combined. Test inflightreusewithdelayedGPUconsumers,freshweights/IDs,partialgroups,crossstreams,errorcleanup andcompletehostmemoryaccounting. Twelve-jobscreen andunchangedretentiongates.

A0010 sourceb22f8f616ee933ddc4ad2dd09c3eb91e60ef477b frozen.47testspass9.62sec plusstandalone maxerror0; delayedGPUconsumer/crossstream/partialgroup hostreuse andpinnedcounter/errorcleanup verified. Twelve-jobscreen estimate261sec fromprior matchingprocess costs. Additionalhostpool correctlycounted; nochangeGPUslotcapacity.

### A0010 final verdict: rejected (offload-gap-001; continuation-01)

All12screenjobscomplete. Offload/resident ms:b1prefill31.109/28.719,decode4.694/4.619;b8prefill210.220/206.491,decode11.010/10.931;b16prefill418.863/415.229,decode20.264/20.258. Parent-relative prefillslower1.73113/0.75697/1.30852ms atb1/b8/b16;gapsworse1.68401/0.07212/1.00406. No targetgain; rejectwithoutformal/full/generation. Totalpinnedcachedbuffers64.09375/512.75/1025.5MiB, preservingGPUcapacitybutdoublingpipelinehostcapacity. Preserveallsource/evidence thenrestorethreechangedfiles toA0001. Fiveadditionalattempts nowhavefinalverdicts, noadditionalacceptedstep.

## Continuation 01 final audit

Completed five additional attempts A0006-A0010: one within_noise and four rejected; no new accepted step. The cumulative ledger contains ten optimization attempts (one accepted, four within_noise, five rejected), excluding baseline. Current model, benchmark and test source equals retained A0001. The additional raw audit passes 108 new model-process references; the core audit passes the explicit continuation count and 160 retained-validation references. All twelve final figure families were visually reviewed, reproduction shell syntax and CLI options checked, and fresh plan-only invocations produce 12 screen jobs and 48 parent-confirmation jobs. Reused A0001 validation and all unfavorable points are labeled. Original first-tranche final artifacts remain unchanged from their publication seal. Report: `results/optimization/offload-gap-001/continuation-01/final/REPORT.md`; publication is recorded separately after remote verification.

## Continuation 02 (offload-gap-001)

New user request: another ten optimization attempts, A0011-A0020. Current clean source equals verified A0001; prior reports and evidence are immutable historical tranches. Keep all original correctness, staged sampling, dual latency/gap, primary regression and CPU-offload memory gates. First investigate whether the token-major CPU table layout penalizes per-layer gathers, then ID staging and pipeline scheduling; register each concrete hypothesis before implementation. No numeric gain or deadline requirement.

### A0011 registration (offload-gap-001; continuation-02)

Test reusable pinned input-ID staging per cached offloader. Fresh-ID microdiagnostic shows b1/b8 prefill staging 0.00670/0.01341 ms with to(cpu), versus 0.00521/0.00690 ms using blocking copies into pinned storage; decode saves under 1 us. Preserve staging before embedding and CPU lookup; guard whole offloaded model call against concurrent buffer reuse. Count additional pinned IDs for every cached shape. All twelve b1/b8/b16 offload/resident screen jobs are predeclared; only independent dual latency/gap confirmation can establish a gain. Prior layout/rank diagnostics are not attempts.

A0011 source 812ac2d frozen; 45 tests pass in 9.95s and standalone maximum error zero. Twelve screen jobs complete: offload/resident b1 prefill 29.426/28.789 ms, decode 4.981/4.600; b8 prefill 209.075/205.463, decode 11.018/10.931; b16 prefill 419.032/415.479, decode 20.283/20.272. Relative to A0001, b8 prefill latency/gap reductions 0.387758/0.044000 ms, b1 decode slower 0.360538 ms. Predeclare fixed 48-process parent confirmation, all primary workloads/both placements/three balanced blocks, 10/10/3. Do not extend until favorable; no acceptance without full retention gates.

### A0011 final verdict: within_noise (offload-gap-001; continuation-02)

All 48 parent-confirmation jobs complete. b1 prefill offload reduction 0.179730 ms, 95% CI [-0.219170,0.578629]; gap reduction 0.186672 [-0.245018,0.618362]. b8 prefill reduction 0.319383 [-1.716318,2.355085]; gap 0.269243 [-2.234845,2.773330]. b1 decode reduction -0.178486 [-0.765859,0.408887], gap -0.192798 [-0.997163,0.611567]; b8 decode -0.035094 [-0.117475,0.047287], gap -0.045085 [-0.144919,0.054749]. None meets both lower-bound criteria; no resolved primary offload regression or lost GPU savings. Preserve all evidence, do not extend sampling, and intentionally skip frozen-baseline/full/generation escalation. Commit evidence then restore four changed source/test files to A0001. One of ten additional verdicts complete.

### A0012 registration (offload-gap-001; continuation-02)

Frozen A0001 instrumentation records 24 producer event waits consuming 13.1856/59.5677 ms thread CPU within 13.2182/59.8416 ms wait wall time at b1/b8. Test blocking=True for pipeline copied events only, reducing CPU wait consumption and potential CPU competition. No transfer, buffer, group or lookup changes; bulk decode unchanged. Sleeping wakeup latency may outweigh any benefit; lower CPU time alone is not a gain. Predeclare twelve-job screen and unchanged dual primary confirmation/retention gates. Separate mapped-bulk diagnostic shows b8 0.06145→0.03329 ms but b1 slightly slower; persistent native gather pools still lose to Torch and are not optimization attempts.

A0012 source 817cc6b frozen, 45 tests pass in 9.87s and standalone max error zero. Cross-stream delayed-consumer slot reuse, partial groups, fresh weights and cancellation covered. b8 mechanism diagnostic: wait wall 60.2107 ms, thread CPU 0.2281 ms, versus A0001 59.8416/59.5677 ms. CPU occupancy falls sharply; full-model latency/gap must still improve. Twelve-job screen estimate 261.17 seconds from prior setup and representative latency costs.

A0012 screen complete: offload/resident b1 prefill 29.237/28.679 ms, decode 4.880/4.625; b8 prefill 209.819/205.989, decode 10.999/10.913; b16 prefill 418.340/414.998, decode 20.276/20.258. b1 prefill latency/gap reductions 0.140985/0.148229 ms versus A0001, but b8 prefill is slower 0.356781 ms and gap worsens 0.174065 ms; b16 also slower. Predeclare fixed 48-process balanced parent confirmation, all four primary workloads, both placements, 10/10/3. Only a changed prefill point can nominate gain. No acceptance from reduced CPU usage or unchanged decode fluctuations; no sampling extension until favorable.

### A0013 registration and isolated draft (offload-gap-001; continuation-02)

Register mapped-host all-layer bulk gather for 8..16 tokens; target b8 decode, with b16 secondary. The component diagnostic reduces b8 0.061450 to 0.033291 ms and b16 0.105814 to 0.048239 ms; b1 loses and is excluded before timing. Pin and account for the 3000 MiB CPU table, keep pipeline and other bulk algorithms, and preserve fresh reads and CPU mutation safety with read-stream completion. This differs from rejected mapped-prefill A0008. Draft is prepared in an isolated temporary tree while A0012 confirmation runs; no GPU tests or timing yet. Activate only after A0012 verdict and reconcile with the best verified parent; a registration is not a completed attempt.

### A0012 final verdict: within_noise (offload-gap-001; continuation-02)

All 48 independent parent-confirmation jobs complete. b1 prefill offload reduction 0.044804 ms, 95% CI [-0.353767,0.443376]; gap reduction 0.079503 [-0.366290,0.525296]. b8 prefill reduction 0.132066 [-0.836286,1.100418], gap -0.066083 [-1.142762,1.010595]. b1 decode reduction -0.380249 [-1.310250,0.549752], gap -0.360232 [-1.415966,0.695501]; b8 decode 0.012882 [-0.044983,0.070746], gap -0.007754 [-0.103957,0.088448]. No point meets both latency/gap requirements. CPU occupancy benefit alone is outside the acceptance objective. No resolved primary offload regression or lost GPU savings; preserve all results, skip full/generation escalation, commit evidence then revert offloader/tests to A0001. Two of ten additional verdicts complete.

A0013 activated after A0012 evidence/revert; frozen source 5d5eb48.49 tests pass in 9.93s plus standalone maximum error zero. Growing-cache bulk tests force the mapped path; direct tests cover b8/b16 CPU/GPU IDs, cross-stream reuse, CPU mutation immediately after return, bounds/error cleanup and pinned-table counting. Twelve-job screen planned at 381.17 seconds, including extra setup/JIT/pinning allowance; actual commands persisted. Independent chunked gather/H2D diagnostics completed before this screen: b8 whole 3.901 ms versus chunk1024 6.661 and chunk512 6.607; b16 7.851 versus 12.774/12.639. All exact; this unpromising component variant is not registered as an optimization attempt.

### A0013 final verdict: rejected (offload-gap-001; continuation-02)

Twelve screen jobs complete. Offload/resident b1 prefill 30.033/28.681 ms, decode 4.632/4.660; b8 prefill 209.727/206.519, decode 11.054/10.925; b16 prefill 418.532/415.470, decode 20.269/20.255. Target b8 decode slower 0.025238 ms and gap worse 0.006473 ms versus A0001. Reject without confirmation/full/generation; b16 secondary improvement cannot substitute for a primary gain. Whole 3000 MiB CPU table pinning correctly counted; total cached pinned MiB b1/b8/b16 3032.09375/3256/3512. All twelve sources and memory capacities audited. Component mapped-read improvement did not translate to target model latency; no isolated causal attribution. Commit evidence then restore six execution/test files including removal of the new mapped-bulk module. Three of ten additional verdicts complete.

### A0014 registration (offload-gap-001; continuation-02)

Test one-slot automatic pipeline lookahead from verified A0001. Four one-layer slots retain 32/256/512 MiB pinned and device storage at b1/b8/b16. Since release occurs after k+m, before attention/MLP completes, a single slot may preserve useful next-layer overlap while reducing working set; it may instead starve compute. Existing wait/component profiles and the unsuccessful A0010 host expansion motivate the tradeoff, without claiming cache pressure caused earlier results. Preserve group size, wait policy/order, bulk decode, and explicit manual pipeline depths. Predeclare twelve-job screen and original dual primary prefill latency/gap gates; memory reduction alone is insufficient.

A0014 source b9938eb frozen.48 tests pass in 13.03s; standalone maximum error zero. New forced-auto 128-step cases exercise one-slot scheduling with two seeds, GQA, gating, QK normalization and padding; capacity test checks reduced host/GPU buffers and preservation of explicit manual depth. Twelve-job screen estimate 261.17 seconds, original sampling and all b1/b8/b16 placements.

A0014 screen complete: offload/resident b1 prefill 29.006/28.751 ms, decode 4.733/4.622; b8 prefill 207.372/205.592, decode 11.044/10.948; b16 prefill 417.130/415.593, decode 20.251/20.245. Relative to A0001, b1/b8 prefill offload reductions 0.372477/2.090906 ms and gap reductions 0.452067/1.876710 ms. Predeclare fixed 48-process independent parent confirmation across all primary workloads, both placements and three balanced blocks, 10/10/3. No acceptance from screen or memory savings alone. Pipeline buffers verified at one-layer capacity, total cached host/GPU buffers each 8.09375/64.75/129.5 MiB b1/b8/b16.

### A0015 registration and isolated draft (offload-gap-001; continuation-02)

Investigate compressed within-call transfer with fused inverse row lookup plus k+m. Frozen A0009 profile records 24 expansions totaling 1.50928 ms CUDA stream intervals and 1.55864 ms CPU spans; they overlap and are not guaranteed removable latency. A0009 is the existing unfused descriptive ablation. Keep >=8192-token threshold and b8 prefill target; preserve direct acquire API, fresh values, slot release and unfolded normalization fallback. Predeclare zero-error arithmetic contract: BF16 loads, FP32 elementwise addition, cast to original dtype, no reduction/reassociation. Count inverse storage and retained buffer capacity. Draft only while A0014 confirmation runs; reconcile against the best verified parent before activation, especially if A0014 is retained. No GPU tests or candidate timing yet; registration does not count as a verdict.

### A0014 final verdict: rejected (offload-gap-001; continuation-02)

All 48 independent parent-confirmation jobs complete. b1 prefill offload reduction 0.349586 ms CI[0.269728,0.429443], gap reduction 0.373642 CI[0.274240,0.473045]; b8 prefill offload 1.917756 CI[1.692958,2.142553], gap 1.698489 CI[0.306138,3.090839]. However b8 decode slows in all three blocks: reduction -0.040589 CI[-0.057702,-0.023475]. b1 decode remains unresolved. The resolved primary regression fails acceptance despite prefill gains and preserved memory savings. Bulk code is unchanged, but prefix pipeline capacity changes remain cached during decode; causal mechanism unproven. Reject without extending the fixed plan or running frozen-baseline/full/generation gates. Preserve evidence, then code-only restore A0001. Four of ten additional verdicts complete.

A0015 source 0436370 frozen after A0014 reversion.42 regressions pass in 10.07s, including exact fused BF16/FP16/FP32 arithmetic, grouped row strides, forced fused growing-cache GQA/QKnorm/gating/padding, direct acquire compatibility, cross-stream/fresh-weight/ID slot reuse and unfolded fallback; standalone maximum error zero. Twelve-job screen planned at 275.04 seconds (22 s/process setup including compilation allowance). No accepted gain or performance result yet.

A0015 twelve-point screen complete. Target b8 prefill offload/gap reduction 0.273381/0.382040 ms versus A0001; b16 secondary reductions 2.073716/2.145007 ms. b1 decode worsens 0.248058 ms and is preserved. Old A0009 unfused b8 screen is 2.290803 ms slower, only a descriptive ablation. All twelve sources, full four-slot capacity and inverse bytes audited. Predeclare fixed 48-process three-block parent confirmation, estimated 900.07 seconds, all primary workloads/placements under 10/10/3. Only changed b8 prefill may qualify; no acceptance from short screen.

### A0016 registration and isolated draft (offload-gap-001; continuation-02)

Restrict automatic one-slot pipeline to <=2048-token calls, preserving configured depth above that boundary and manual pipeline behavior. A0014 independently improved b1 prefill latency/gap but failed on b8 decode after prefix; this separately registered policy keeps the entire b8 primary path unchanged and tests the supported small-prefill regime. Fixed cutoff declared before results, not to be tuned after timing. All four primaries and b16 screen, zero-error placement/scheduling gates, independent confirmation and full regression requirements remain. Draft only while A0015 confirmation runs; reconcile with best verified parent after its final verdict.

### A0015 final verdict: within_noise (offload-gap-001; continuation-02)

All 48 parent-confirmation jobs completed. Target b8 prefill offload reduction 0.366669 ms CI[-0.645449,1.378787], gap reduction 0.158634 CI[-0.449015,0.766283]. Neither positive lower-bound criterion passes. All other primary comparisons remain unresolved, including unfavorable b1 decode; no claim from unchanged paths or old unfused screen. All 48 source hashes and offload memory counters verified. Close fixed plan without extension; frozen-baseline/full/generation intentionally not run. Commit evidence then restore A0001 implementation. Five of ten additional verdicts complete.

A0016 activated and frozen at 0ce6645 after A0015 evidence/revert.31 regressions pass in 14.58s, including forced-auto 128-step trajectories, padding/GQA/gating/QKnorm, token boundaries1024/1025/2048/2049, total-token grouping across batches, explicit manual depth and disabled-policy checks. Standalone maximum error zero. Twelve-job screen predeclared at263.04 seconds. Post-A0015 diagnostics completed sequentially before this screen: input ID CPU spans b1/b8 0.033814/0.047770 ms and embedding CUDA intervals0.041856/0.112032 ms (instrumented, overlapping); write-combined gather+H2D b8/b16 2.983973/5.945323 ms versus torch pinned3.869776/7.856003, all exact. Native default allocation control3.892142/7.923771 ms. These component results do not establish model gains or count as attempts.

### A0017 registration and isolated draft (offload-gap-001; continuation-02)

Test cudaHostAllocWriteCombined gather destinations only for pipeline inputs >=8192 tokens. Frozen A0001 component results support the large-input restriction: b8/b16 gather+H2D means2.983973/5.945323 ms versus torch pinned3.869776/7.856003 and native-default3.892142/7.923771; b1 shows no benefit. Keep ordinary CPU table, device/host capacities, fresh lookups and event order. External allocation ownership must follow all tensor views, and free must wait for DMA. Zero-error gate, full primary/secondary screen and independent retention gates apply. Draft only while A0016 is pending; reconcile with best verified parent before activation.

A0016 screen complete: b1 prefill offload/resident28.942/28.647 ms, target offload/gap reductions0.436107/0.411638 ms versus A0001. b1 decode slower0.451537 ms retained; b8 prefill slower0.381399 despite unchanged large-input pipeline policy. Predeclare original48-process parent confirmation, three balanced blocks across all four primary workloads/placements, estimated899.35 seconds. Only changed b1 prefill can qualify. Twelve-source and shape-specific memory-capacity audit passes.

### A0018 registration and isolated draft (offload-gap-001; continuation-02)

Test pipeline nonblocking ID D2H followed by embedding submission before waiting for CPU IDs. Input-startup diagnostics on frozen A0001 show ID CPU spans0.033814/0.047770 ms preceding embedding dispatch0.034976/0.046488 ms at b1/b8; instrumented spans are not additive removable latency. A0011 blocking reusable IDs was inconclusive, so overlap is a distinct hypothesis with that descriptive ablation. Keep same stream ordering and fresh CPU gather; serialize retained host IDs, account for pinned capacity, and wait on exceptional embedding paths before releasing ownership. Bulk/CPU/foreign-device ID fallback preserved. Draft only until A0017 verdict and best-parent reconciliation; all primary gates remain.

### A0019 registration and isolated draft (offload-gap-001; continuation-02)

Test producer submission before pipeline embedding dispatch: current pipeline entry event follows embedding, while startup diagnostics show instrumented embedding CUDA intervals0.041856/0.112032 ms at b1/b8. Defer embedding to first pipeline consumer to overlap initial CPU lookup/H2D, keeping bulk ordering and arithmetic. Earlier transfer may contend with embedding; no additive latency claim. Draft against A0001 only until preceding attempts finish; if A0018 is retained, preserve its asynchronous dispatch overlap with producer-side ID completion rather than reverting it. Fixed12-point screen and original independent/full gates apply.

### A0020 registration and isolated draft (offload-gap-001; continuation-02)

Test reusable GroupTicket/threading-event pool, motivated by frozen A0001 ticket setup CPU sums0.029966/0.032549 ms at b1/b8. Keep fresh CPU lookup and all CUDA copied/consumed dependencies; generation tags distinguish already-submitted prior-call CPU releases from current-generation waits after ticket reset. No A0007 tensor-view caching. Potential savings are small and may be erased by reset overhead; all independent gates remain. Test cross-stream repeated forwards, partial groups and failure poisoning. Draft only until A0019 verdict and best-parent merge. A0020 registration does not imply completion: only five of ten additional attempts currently have verdicts.

A0016 parent confirmation passes: b1 prefill offload reduction0.430878 ms CI[0.153321,0.708434], gap0.451613 CI[0.304395,0.598832]. All48 source hashes/memory checks pass. Other primaries have no resolved offload/resident regressions, but b8 decode is slower in all3 blocks (mean reduction-0.038070 CI[-0.100821,0.024682]), preserved as unresolved adverse direction. Not accepted: predeclare fresh48-process A0000/candidate confirmation, same three balanced blocks and sampling, estimate899.35 seconds. Full model correctness, matrix/generation and all regression gates remain. Best verified source remains A0001; no additional verdict counted.

A0016 conditional validation prepared while R03 runs. Reuse only four independently frozen A0000 resident correctness fingerprint files after exact source/environment/config/checkpoint validation; run four new candidate trajectories (historical estimate85.37 seconds), comparing all129 logits plus full hidden/KV at steps0/1/2/128.12 reference/controller tests pass, including rejection of mismatched or incomplete references and stopping before matrix on correctness failure. If R03 and correctness pass: full84-point matched matrix reuses16 primary points backed by all48 current confirmation processes, collecting68 new nonprimary paired jobs; generation12 new jobs. Controllers emit exact cost plans before launch and never mark acceptance. Full-matrix adverse points still require manual regression review and bounded investigation.

A0016 independent frozen-baseline confirmation passes; all48 source hashes and memory counters audited. Against A0000, b1 prefill cumulative offload reduction14.021136 ms CI[13.510024,14.532248], gap14.015407 CI[13.549947,14.480867]. This includes A0001; the new parent-relative improvement remains0.430878 ms. Other primary results unresolved without resolved regression. Gated controller has begun the four new full-model candidate trajectories with validated frozen resident references. Still not accepted; complete matrix/generation and manual regression/memory/source audits remain.

A0016 full-model exactness gate passes all four independent-source comparisons: b1/b8 x seeds1234/4321,129 logits per trajectory, all hidden/KV at prefix and steps1/2/128. Baseline references passed strict reuse checks; all four candidate runs were new. Full84-point matrix plan persisted before launch:68 new processes,16 reused points backed by48 current primary-confirmation processes, estimate1601.51 seconds. Full matrix is running; generation and manual regression/memory/source review remain, so no accepted status or incremented verdict count.

A0016 secondary regression review rule fixed before complete matrix/generation summaries: examine all ten nonprimary prefill/decode workloads and both generation workloads. Any original single-pair negative offload or folded-resident reduction triggers exactly two added balanced pairs, retaining the original. Primary confirmations are not extended. Zero is not adverse; failed/OOM/missing rows require review. Four selector tests pass, including refusal to reselect repeated results. This is a conservative investigation trigger, not a regression conclusion or permission to extend until favorable.

A0016 original full/generation matrices complete:84+12 successful points,128 referenced raw processes after valid primary reuse. All raw source hashes and candidate offloader capacities audited; conservative peak-GPU savings remain at least1999.25 MiB. Prior secondary trigger selects8/10 nonprimary full workloads and b8 generation (small resident slowdown); original observations retained. Plans fixed before launch:64 new full processes estimated1725.72 seconds, followed by8 new generation processes estimated316.23 seconds. Bounded controller runs sequentially after all original timing ended. No repeated primary confirmations, no acceptance yet, five of ten final verdicts remain complete.

### A0016 final verdict: accepted, cumulative step2 (offload-gap-001; continuation-02)

Retain source0ce6645 after all fixed gates: parent-relative b1 prefill offload reduction0.430878 ms CI[0.153321,0.708434], gap0.451613 CI[0.304395,0.598832]; independent A0000 cumulative confirmation also passes (14.021136 ms reduction includes A0001). All84 matrix and12 generation points complete; original observations plus exactly64 full and8 generation added processes retained. Both secondary regression gates pass with no resolved offload/resident slowdown or loss of memory savings. Recomputed full source/environment/sample/memory audit covers200 raw process results and four bit-exact full-model comparisons.

Preserve adverse directions: parent b8 decode all3 slower (reduction-0.038070 CI[-0.100821,0.024682]); secondary b16 decode all3 slower(-0.033485 CI[-0.082574,0.015604]); b8/4096 prefill all3 slower(-0.937146 CI[-2.972030,1.097739]); b8/8192 prefill unresolved(-4.475341 CI[-15.449950,6.499268]). b8 generation remains noise(0.438272 CI[-4.831173,5.707717]); b1 generation has one original pair. No additional extensions or broad decode/generation claims. Six of ten additional attempts now have final verdicts. A0017 must start from this verified source, preserving its selective-depth policy and tests.

A0017 activated from verified A0016 using the conditional merge, preserving the selective-depth configuration/model path and all A0016 tests. Frozen candidate7eb9906.37 regressions passed in13.24 seconds, including forced-WC128-step trajectories, storage lifetime after aliases and in-flight DMA, fresh rows with cross-stream slot reuse and8191/8192-token boundaries. Standalone maximum error zero. Fixed12-point screen (b1/b8/b16 both modes/placements), estimated263.04 seconds; only changed b8 prefill can nominate gain. Model timings begin after A0016's full validation and all secondary checks finish.

A0017 screen complete: b8 prefill offload207.824 ms/resident205.925/gap1.899 ms; target offload/gap reductions2.020062/1.692548 ms versus A0016. b1/b16 decode slower0.370803/0.431870 ms; retained without attributing mechanism. All12 source hashes and A0016-compatible memory capacities pass. Predeclare48-process parent confirmation across all four primaries, three balanced independent blocks, estimated901.01 seconds. Only changed b8 prefill can qualify; full gates remain conditional.

A0017 sequential conditional controller declared while parent confirmation still runs: wait on the recorded controller identity without restart; require original confirmation gate plus predeclared changed b8 prefill gain before fresh48-process A0000 confirmation. If that independent gate passes, four new candidate exactness trajectories with validated resident reuse precede84-point matrix and12-point generation. Each expensive stage emits its measured cost plan before launch. Secondary any-negative trigger fixed now, identical bounded original+two policy; no automatic acceptance. A0018 isolated conditional merges preserve A0016 depth and, only if retained, A0017 WC behavior/tests; no GPU testing or final verdict yet.

### A0017 final verdict: within_noise (offload-gap-001; continuation-02)

All48 parent-confirmation processes complete and source/environment/sampling/memory audits pass. Target b8 prefill offload reduction1.765735 ms CI[0.688570,2.842900] passes latency alone; gap reduction1.567970 ms CI[-0.678259,3.814199] fails the required positive lower bound despite all3 favorable gap signs. b1 decode slower in all3 blocks, reduction-0.393447 ms CI[-1.099139,0.312246], retained as unresolved. No qualifying primary; no extra pairs or threshold adjustment. Conditional controller stopped before A0000/full/generation; those stages intentionally not run. Commit evidence then restore A0016. Seven of ten additional verdicts complete; no WC retention or additive speedup claim.

A0018 activated after A0017 evidence and reversion, preserving A0016 selective-depth policy/tests. Frozen candidate6089489.34 regressions passed in13.06 seconds, covering128 growing steps, fresh IDs across streams, submission-before-wait, CPU fallback, embedding-exception cleanup and model-call serialization. Standalone maximum error zero. Predeclared12-point screen estimated263.04 seconds; primary eligibility remains b1/b8 prefill only. Retained ID staging memory is counted by telemetry; no lookup/output caching.

### A0018 final verdict: rejected at screening (offload-gap-001; continuation-02)

All12 screen processes complete. b1 prefill offload/gap reductions0.007509/0.015961 ms versus A0016 are tiny relative to descriptive offload sample SD0.019534 ms (parent0.010548); b8 prefill is slower0.160926 ms with gap worse0.227580 ms. b1/b16 decode slower0.249344/0.084073 ms. Screen deemed unpromising; no independent regression/significance claim or invented statistical threshold. All12 sources and pipeline-only pinned IDs audited;18 memory-audit tests pass including omitted IDs rejection. Stop before confirmation/full/generation, preserve evidence then restore A0016. Eight of ten additional verdicts complete.

A0019 activated from verified A0016 after A0018 evidence/reversion, preserving all selective-depth tests. Frozen candidatef59cdf0.34 regressions passed in13.05 seconds, including producer-before-embedding submission order, exact logits/hidden/KV, deferred-embedding exception cancellation/poisoning and128 growing steps. Standalone maximum error zero. Fixed12-point screen estimated263.04 seconds; b1/b8 prefill eligible, unchanged bulk decode cannot nominate gain. No buffer capacity or table representation change.

### A0019 final verdict: rejected at screening (offload-gap-001; continuation-02)

All12 screen processes complete and source/capacity audits pass;20 memory-audit tests pass. Both eligible prefill primaries are slower versus A0016: b1+0.042619 ms, b8+0.366468 ms. b8 gap worsens0.106847 ms; b1 apparent gap reduction0.024187 ms comes from slower resident reference, not an offload latency improvement. Unchanged bulk b1 decode reduction0.377981 ms cannot nominate this pipeline change. No independent regression claim or mechanism conclusion from the short screen. Reject without confirmation/full/generation, commit evidence then restore A0016. Nine of ten additional verdicts complete.

A0020 activated from A0016 after A0019 evidence/reversion. Frozen candidate4186955.36 regressions passed in12.96 seconds, including persistent ticket identities, fresh table rows over repeated cross-stream calls, generation-tagged slot reuse, partial groups, producer/consumer error poisoning and128 growing steps. Standalone maximum error zero. Fixed12-point screen estimated263.04 seconds; only pipeline prefill b1/b8 eligible. No added tensor buffers or lookup/output cache. This is the tenth additional registered experiment, still awaiting its actual timing verdict.

A0020 screen complete: eligible b1/b8 prefill offload reductions0.030696/0.048966 ms and gap reductions0.073350/0.276661 ms versus A0016. Small effects near within-process sample dispersion; both changed targets favorable, so test the profiling-supported coordination hypothesis under fixed independent confirmation. b1/b8 decode slower0.271334/0.035761 ms retained. All12 source/capacity checks pass;22 memory-audit tests pass. Predeclare48 balanced parent-confirmation processes, all four primary workloads and both placements, estimate900.71 seconds. No additional pairs until favorable and no accepted gain from the screen.

### A0020 final verdict: within_noise (offload-gap-001; continuation-02)

All48 independent parent-confirmation processes complete; sources, environment, sampling and memory audits pass. b1 prefill offload reduction0.030322 ms CI[-0.111244,0.171888], gap0.037519 CI[-0.153134,0.228172]; b8 prefill offload0.000883 CI[-0.402357,0.404122], gap-0.272545 CI[-1.340937,0.795846]. No eligible dual latency/gap gain; other primary comparisons unresolved without resolved regression. Close fixed plan without extension. Conditional controller stopped before A0000/full/generation. Commit all evidence then restore A0016. Ten of ten additional attempts now have verdicts: A0016 accepted; A0011/A0012/A0015/A0017/A0020 within_noise; A0013/A0014/A0018/A0019 rejected. Final report, figures and complete audit/publication remain.

### Continuation-02 final delivery

Exactly ten additional attempts A0011–A0020 finalized: one accepted, five within_noise, four rejected. Final execution source equals A0016 (0ce6645); incremental b1 prefill reduction0.430878 ms CI[0.153321,0.708434], gap reduction0.451613 CI[0.304395,0.598832]. Other workload gains are not claimed; adverse unresolved observations remain in the report. Twelve figure families exported and visually inspected, complete21-record ledger generated. Core audit passes20 attempts/two retained steps/200 final raw references; additional audit passes456 references;72 helper tests pass. Reproduction Bash syntax and fresh CPU-only plans verified. Both historical final directories unchanged. Final report in continuation-02/final/REPORT.md; report commit is followed by a separate verified-origin publication seal. No new GPU measurements for final artifacts.

### Supplemental-01: previously unscreened shapes

User authorized supplementary shape screening of the18 nonretained frozen candidates. Fixed115 workload comparisons/460 sequential processes, estimated10724seconds. Route-aware short/long/b4 prefill, tiny-bulk decode, high-context decode control and applicable128-step generation; eligible historicalb16 signs can nominate fresh confirmations. Positive offload+gap screens trigger exactly3 new independent balanced blocks, separate parent/current references and Holm correction across the frozen confirmation family. No new attempts, automatic retention, source edits or historical relabeling. Driver5tests passed; exact plans and sources committed before timing.

Supplemental coverage audit found A0011 ID staging affects bulk as well as pipeline. Before measuring the omitted cells and before independent-confirmation nomination, froze12-process addendum: b4/2048 and b8/512,4096 decode against actual parentA0001, both placements. Original460-process plan preserved; total118 new workload comparisons/472 processes. Addendum executes sequentially after original screens and before nomination. No statistical gate or sample-count changes;7 driver tests pass.

Supplemental-01 completed472 new screen processes plus17 eligible historicalb16 reviews. Preparation stopped on a historical module-path assertion (A0001 originally ran in main checkout at correct SHA/hash); fixed recorded-command/cwd path validation with8 passing tests and all historicalreferences audited. No GPU timing failed/repeated; failed preparation and unexecutedpartialplans archived. Frozen66 nominees,112 potential parent/current hypotheses:792 fresh parentprocesses estimated6.2494hours; conditionalcurrent atmost552. Explicit recovery starts only the unstarted independent confirmation; no threshold or sample-count changes.

## Supplemental shape survey completed

Completed the user-requested omitted-shape study for all18 nonretained frozen attempts:118 fresh screening workloads (472 processes),17 eligible historical b16 reviews,66 nominees and936 independent confirmation processes. Fixed112-comparison Holm family;34 conditional current slots remained unrun with p=1. Every measured comparison uses exactly3 fresh balanced blocks, independent of screening; no extension until favorable.

A0014 prefill b8/512 passes both parent and current-A0016 local gates. Against current:55.0926→52.3164ms, offload reduction2.7762ms [2.5098,3.0426], gap reduction2.7604ms [2.5246,2.9961], Holm p=0.02734. Historical primary decode regression remains, so it is not retained. A0013 generation b8 (prefix2048+128 predetermined steps) passes current comparison:29.0499ms reduction, Holm p=0.01302; parent comparison does not survive correction (p=0.17030), and the3000MiB pinned-table tradeoff remains. A0015 current b16 prefill fails the resident-slowdown guard despite corrected timing significance. No new accepted step or model source change.

All1408 new raw results passed provenance/sampling/memory audit;8 driver tests passed. Report, all performance/memory samples,19 figure sets (PNG/PDF/SVG), explicit coverage, failure/recovery record and reproduction commands are in [supplemental report](results/optimization/offload-gap-001/supplemental-01/final/REPORT.md). Original20 attempts,2 accepted steps, A0016 retained code and prior final reports remain unchanged.

## continuation-03 / A0021 registered

User authorized a cumulative integration of recovered local gains followed by optimization toward resident latency at every fixed workload. A0021 extends the selective one-slot auto-pipeline boundary from2048 to4096 total input tokens, recovering the A0014 b8/512 opportunity while preserving larger prefix capacities. A0013 mapped bulk is the next separate integration candidate after parent evaluation. Every candidate receives64 fresh full-screen processes, including generation; all source, correctness, independent confirmation and regression gates remain. Final near-GPU tolerance is provisional pending user preference. No historical verdict is rewritten.

A0021 complete-matrix screen:64/64 processes audited. Target b8/512 prefill offload/gap reductions2.4052/2.3635ms. Frozen independent confirmation:9 workloads,108 processes (three balanced blocks), estimated2374.64s. All primary controls, positive secondary signals and both generation points included; negative results preserved. Screening evidence does not establish retention or near-GPU completion.

A0021 finalized nonretained: target b8/512 prefill gain confirmed (offload2.31452ms [1.51509,3.11395], gap2.31708ms [1.59000,3.04416], Holm p=.02872), but b1 decode gap reduction−.12804ms [−.23351,−.02258] fails the declared regression guard. Offload slowdown alone is inconclusive. Preserve local gain and all adverse samples; no optional extension. Full validation intentionally not run. Restore A0016, recover A0013 first, then test cumulative short-prefill integration on a verified parent.
