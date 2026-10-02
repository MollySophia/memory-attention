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
