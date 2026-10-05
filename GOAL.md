# Memory Attention inference performance experiments

Status: original campaign `offload-gap-001` and supplemental survey completed:
20 attempts, two retained steps (A0001 and A0016). Current retained execution
source is A0028 (`6cc9354760ebde3bab9a90996a7713a936dfb439`), accepted
as step 4 after A0001, A0016 and A0023 in continuation-03. Editing this
document alone does not launch measurements. The user has now authorized
continuation-03: integrate the recovered local gains into a cumulative source
chain, then optimize until offload approaches resident latency at every matrix
and generation workload. This is an active objective with no fixed attempt cap.

Workflow revision (2026-10-04): future candidates require complete matrix
screening before a performance verdict. Coverage is mandatory; sampling effort
is staged. Apply this prospectively, preserving all original plans, verdicts,
raw results and supplemental findings. See
[completed survey](profile/results/optimization/offload-gap-001/supplemental-01/final/REPORT.md).
Continue any subsequently authorized work from the best verified implementation;
do not restart the completed campaign or reuse attempt IDs.

Historical campaign origin: restored `feat/offload` source
`81684d34e44db04a0511698081cab336abf4a60d`; verified harness/baseline A0000 is
`62942a0387608fe21baaeb2dce9ef3b0947dde4d`. Keep the old `attempts/paper-001`
branch and its archived results separate; do not import their accepted-step
counts, timings or model changes into this campaign.

## Objective and constraints

Optimize the Memory Attention architecture's CPU memory-table offload path.
The primary objective is to reduce the prefill and decode latency gap between
ma_offload and the equivalent all-GPU ma_gpu implementation, while preserving
model computation and the CPU offload memory saving. Use fixed seeded
random weights and input tokens; these experiments support performance and
numerical equivalence claims, not language model quality claims.

Final acceptance (user-confirmed): every one of the 16 fixed matrix and
generation workloads must satisfy
`T_offload - T_gpu <= max(0.01 * T_gpu, 0.1 ms)`.
For resident latency at or below 10 ms, the allowed extra latency is 0.1 ms;
above 10 ms, it is 1% of that workload's matched resident latency. Each
correctness-passing candidate must complete the full matrix screen. Final
acceptance additionally requires independent paired repeats and the simultaneous
confidence-bound criterion specified below; a screening mean is insufficient.

Primary target: ma_offload. Keep a frozen ma_offload implementation as the
optimization baseline, and ma_gpu as the folded resident placement reference.
Use ma_gpu_unfolded to measure the norm-folding contribution separately.
Focus hypotheses on offload overhead: token-ID staging, CPU table lookup,
pinned buffers, H2D transfer, transfer/compute overlap, prefetch scheduling,
synchronization, and their integration with Memory Attention. Profile these
costs before selecting a candidate. Generic MLP, RoPE, attention-kernel or KV
cache speedups are outside this campaign unless evidence identifies a specific
offload bottleneck they resolve; general model acceleration alone is not an
accepted offload optimization. Do not broaden scope to reach an iteration count.

For each matched workload, report T_offload and T_gpu, the absolute gap
`T_offload - T_gpu` in milliseconds, and the relative overhead
`T_offload / T_gpu - 1`. Preserve signed values and uncertainty. Measure both
placements for baseline and candidate under matched plans; keep the resident
reference behavior fixed. Under the newly authorized continuation, a retained
step must repeatably reduce offload latency and its absolute gap for at least
one of the fixed 16 matrix/generation workloads, without a resolved regression
elsewhere in the complete matrix or loss of memory savings. The four original
primary workloads remain mandatory regression/reporting points; a secondary
workload can now justify retention after the same full gates. This prospective
change does not relabel historical acceptance decisions.
Retention remains distinct from local effectiveness:
a confirmed gain at any secondary matrix or generation workload must be
recorded even if no primary workload improves or the candidate is not retained.
Do not discard such a candidate before completing its matrix screen and the
predeclared confirmation of nominated points. A shape-specific policy based on
these findings is a new candidate requiring its own validation, not an automatic
integration of the best observed points.
A slower resident reference cannot count as closing the gap. A shared speedup
without evidence of lower offload overhead does not satisfy the objective.
Do not change model dimensions, attention semantics, precision or output scope
to claim a speedup. Quantization and approximate algorithms require a separately
defined experiment. Do not cache outputs or token lookups across calls.

## Model and workloads

Current model: 24 layers, hidden size 2048, 32 query and 32 KV heads,
head dimension 64, vocabulary 32000, SwiGLU intermediate size 5632.
BF16 weights and computation, untied input embedding and LM head,
qk_norm=False, use_gate=False, fuse_norm=False. Seed 1234 for performance.
Record the complete configuration; correctness tests also exercise enabled
QK normalization and gating with non-unit memory norm affine weights.

Total parameters: 2,836,499,968. Memory tables: 1,572,864,000 parameters,
3000 MiB in BF16. Remaining parameters: 1,263,635,968, approximately
2410.19 MiB in BF16. Offload changes placement, not total model size.

Primary shapes: batches 1 and 8, prefill length 2048, decode prefix length
2048. The four primary workloads are prefill and decode at each batch size;
keep this set fixed during the campaign rather than selecting favorable points.
Prefill IDs [batch, 2048], hidden states [batch, 2048, 2048]; decode IDs [batch, 1].
Attention Q/K/V heads have dimension 64. Prefill transfers at most
batch * length * layers * kv_dim * sizeof(dtype): 192 MiB at batch 1 and
1536 MiB at batch 8. Decode transfers 0.09375 and 0.75 MiB per step, respectively.

Complete performance matrix (fixed scope for screening and final validation):

- Batch sweep: 1, 4, 8, 16 at prefill/context length 2048, both prefill and decode.
- Length sweep: 512, 2048, 4096, 8192 at batch 8, both prefill and decode.
- Deduplicate batch8/length2048: seven unique shapes, 14 prefill/decode workloads.
- Growing-cache generation: 128 decode steps after a 2048-token prefix,
  batches 1 and 8, fixed predetermined tokens, excluding sampling.
- Record OOM, unsupported and failed points explicitly; never silently drop
  them or substitute zero timings.

Every correctness-passing candidate must screen all 16 workloads with both
ma_offload and ma_gpu, regardless of its expected active branch or primary-shape
outcome. That is 32 isolated processes per implementation, or 64 for a fresh
matched parent/candidate screen. No extra Cartesian product of all batches and
lengths is implied. Use the current verified parent as the screen comparator;
retain frozen A0000 for cumulative validation. Do not stop a performance screen
because primary points are slow or unpromising. A correctness, OOM, unsupported
or benchmark failure may prevent completion; preserve it and record every
unmeasured cell and reason without claiming full coverage.

A new baseline, when required, covers the same 32-process matrix. Profiling may
start after the primary baseline points are ready, but profiling and timing
must run sequentially. Existing campaign evidence can supply screening points
only when source, scope, sampling plan and environment match; list reused and
fresh points explicitly. A source path believed unchanged is not by itself a
reason to omit a workload: setup, allocation and earlier prefix execution can
affect it. Confirmation always uses fresh independent pairs.

Full matrix coverage does not mean formal sampling at every point in every
round. Screen the whole matrix cheaply, then independently confirm nominated
gains and suspected regressions. A candidate proposed for retention still
requires full validation, generation, memory and correctness gates. Reuse valid
completed validation for an identical source/plan/environment rather than
rerunning it merely because that source is now called final.

## Measurement scope and staged sampling plans

Use profile/bench_fla.py for the actual model. profile/bmk.py implements a
separate experiment and must not supply this model's headline numbers.

Define and persist a protocol version in every result:

- Primary inference prefill: embedding, all layers, final norm, last-token LM
  head and KV cache construction. Return logits [batch, 1, vocab].
- Primary decode: one model call with real KV cache and logits
  [batch, 1, vocab]. Report latency per batch step and batch tokens/second.
- Retain the existing full-logits, use_cache=False prefill as a separate
  historical/ablation workload, with logits [batch, length, vocab].
- Apply logits_to_keep to prefill as well as decode. Do not compare different
  output/cache scopes in the same speedup.
- Keep benchmark-only cache rollback and test token selection outside the
  timed interval; verify this in the prepared harness before freezing A0000. Keep a
  fixed context for isolated decode timing, and measure growing context separately.
- Inputs begin on GPU for all placements. Include any runtime ID staging,
  CPU gather, synchronization and H2D needed by offload in wall time.
- Exclude model loading, norm folding, input generation, compilation, buffer
  allocation during setup and prefix setup for isolated decode.
- Synchronize around measured model calls, preserving asynchronous overlap
  within a call. CUDA-event/kernel timing is diagnostic, not the headline.
- Generation latency includes prefill and all 128 decode calls. State whether
  sampling/token selection is excluded; do not label it serving latency.

Separate measurement scope from sampling effort. Persist `protocol_version`,
`measurement_plan_id`, `stage`, warmup, repeats and rounds in each future run.
Keep the verified campaign protocol for unchanged measurement scope; use a
new sampling-plan ID for revised effort and a new protocol ID for changed
measurement semantics. Leave archived records and metadata unchanged. Freeze
each plan before collecting its data.
These are starting plans, not universal sample-size guarantees. Independence
comes from separate paired processes; many samples in one process do not replace
that requirement. Do not apply the old 30-warmup/150-sample plan to every point.

| Stage | Workloads and placements | Warmup / samples per round / rounds | Purpose |
| --- | --- | --- | --- |
| Screening | Complete 14-workload prefill/decode matrix, both ma_offload and ma_gpu, matched parent/candidate | 3 / 5 / 1 per point | Find local gains and suspected regressions across all shapes; no confirmed gain claims |
| Generation screening | Both generation workloads, matched parent/candidate and both placements | 2 full-trajectory warmups / 3 trajectories / 1 round | Complete each candidate screen without shortening the 128-step workload |
| Confirmation | All nominated gain/regression points, including secondary shapes and generation; all four primary workloads additionally required for retention | Prefill/decode: 10 / 10 / 3; generation: 2 / 5 / 3, in at least 3 fresh balanced independent process blocks | Confirm local effects and distinguish them from retention eligibility |
| Full validation | Complete batch/length matrix for a candidate nominated for retention, including placement/folding references | 10 / 10 / 3 per prefill/decode point | Scaling, memory and regression acceptance gates |
| Generation validation | Prefix2048 + 128 steps; batches1/8, placement/folding references | 2 full-trajectory warmups / 5 trajectories / 3 rounds | Growing-cache latency and memory |

The verified campaign harness already records stages and sampling plans.
Prefill/decode defaults to screening (3/5/1); generation currently defaults to
generation_validation (2/5/3). For the shorter generation screen, explicitly
pass warmup=2, repeats=3 and rounds=1, persist a distinct screening/custom plan
ID and verify the recorded effective counts. Do not silently use default
validation effort or label shorter screen data as validation. This document
revision does not itself modify the harness. Plan-only commands must expose
actual counts, job totals and estimated work before timing starts.
Diagnostic profiling should default to 10 warmup calls, allow an explicit
override, and run after a primary baseline without waiting for a full matrix.
Independent alternating baseline/candidate runs must cover both placements;
a single matrix run is not independent confirmation.

Screening still requires compilation and offload buffer setup before timing.
If the short warmup is insufficient, increase it for both baseline and candidate
and record the revised plan before comparing. Short-plan results are provisional
and must never supply headline or accepted-step speedups. These sampling plans
do not shorten correctness tests: retain multi-step/growing-cache gates.
Generation performance is also mandatory in each complete candidate screen.

A generation warmup/sample is an entire prefix plus 128-step trajectory, not
one decode call. Its warmup count is independent of the single-call benchmark.
Collect fresh baseline and candidate generation points under the same new
plan. For subsequent comparisons within this campaign, reuse valid measurements
only when source, scope, plan and environment match; collect fresh independent
runs for confirmation. Do not compare different sampling plans as matched
evidence or rerun unrelated matrix points merely to refresh one comparison.

Before screening, freeze the nomination rule: every workload with positive
mean offload-latency and absolute-gap reductions and preserved GPU memory
savings is a gain nominee. Also freeze a suspected-regression rule for offload
latency, gap, resident latency and memory guards, with thresholds justified by
baseline variability or a declared practical tolerance before candidate timing.
A negative screen mean alone is not a resolved regression. Preserve all signed
results, including those below the confirmation trigger. Do not select a
post-hoc top-K list, tune thresholds after results or omit secondary shapes.
Complete the screen before freezing the confirmation workload list, comparators,
block count, stopping rule and multiple-testing family. Confirm the union of
nominees once per comparator; screen observations do not enter confirmation
statistics. For retention, include all four primary workloads even if they
were not nominated by the screen.

Use at least three fresh balanced independent blocks per selected point; choose
the exact count before confirmation and do not extend it until favorable.
Correct gain claims across the frozen family of tested workloads/comparators
(e.g. Holm with a joint latency/gap test), while also reporting ordinary paired
95% intervals, resident behavior and memory guards. Predeclare regression
criteria separately and retain adverse evidence even when inconclusive.
If the historical parent differs from the current retained source, distinguish
parent-relative effects from current-source superiority; measure both before
claiming both. Any conditional comparator slots stay in the frozen family
(unrun slots receive p=1), not a smaller family selected after results.

Escalate measurement effort only for a concrete unresolved question: compilation
or allocation still occurring during measurement, warmup drift, run-order or
thermal effects, or paired uncertainty that cannot distinguish a gain from a
regression. First diagnose the cause. Then predeclare the revised plan and rerun
both sides at the affected points under new run IDs, retaining earlier results.
Prefer additional independent process pairs when between-run variability
dominates; add within-process samples only when request-level noise warrants it.
Do not keep extending a run until a favorable result appears. If uncertainty
remains, report within_noise; do not lower acceptance standards. Larger plans
such as 30 warmups and 150 samples are an escalation option, not the default.

Preserve every sample, round means and run-level estimates. The current
median_ms is a median of round means, not a median of individual request latencies; label it precisely.
Report throughput as batch * sequence_length / seconds for prefill and
batch / seconds for decode. Report p50/p95 of actual samples if used, separately
from statistics over rounds.

For a claimed gain, rerun frozen baseline and candidate in alternating order
on the same machine, with at least 3 independent process pairs for each
placement. Within each block, balance run order across baseline/candidate and
offload/resident. Report paired offload speedups, absolute-gap reductions and
relative overhead with uncertainty computed from these independent blocks. Round spread is descriptive,
not a confidence interval or a universal significance threshold. Classify
unresolved improvements as within noise. Profile separately from timing runs.

Profile as soon as the primary baseline is available; do not wait for a full
baseline matrix or generation sweep before investigating bottlenecks. Run GPU
profiling and GPU timing sequentially so they do not interfere. Use ma_gpu_unfolded
for the folding ablation and full validation, rather than automatically including
it in every screening run. After complete screening and the fixed confirmation
of nominated points, record local findings and the separate retention verdict.
A nonretained candidate need not run the more expensive final validation stage;
its full matrix and generation screening evidence must still be preserved.

## Memory and environment

Measure GPU allocated/reserved memory before execution and peak allocated/
reserved memory during prefill/decode. Include live KV cache, activations,
logits and all cached offload buffers. Report parameter placement separately.
Measure host process RSS and pinned buffer bytes across every cached offloader.
The current table is ordinary CPU memory; gathered transfer buffers are pinned.
Raw snapshots and restore copies also consume RAM and must be accounted for.
Record KV cache bytes and CPU/GPU buffer capacities.

Use /home/molly/miniconda3/envs/fla-bench/bin/python on the current RTX 5090.
Record GPU, driver, CUDA, torch, FlashAttention, Python, CPU, thread settings,
commit and source hash. Note GPU temperature/clocks and competing GPU work.
Use a new result directory per attempt/protocol/environment. Preserve source
or a patch for dirty attempts; a hash alone cannot reconstruct the code.
Preserve the frozen A0000 checkout and freeze each new candidate separately;
record checkout paths, source hashes and environment in the manifest. Neither
the historical tracked sweep nor the archived campaign's baseline substitutes
for this campaign's A0000. Claimed improvements require fresh matching paired runs; a stored
baseline point alone cannot replace independent confirmation.

## Correctness gates

Run existing tests before measuring every algorithmic candidate:

```bash
/home/molly/miniconda3/envs/fla-bench/bin/python -m pytest -q tests/test_memory_offload_regressions.py
/home/molly/miniconda3/envs/fla-bench/bin/python -m pytest -q tests/test_decode_benchmark.py
/home/molly/miniconda3/envs/fla-bench/bin/python profile/test_memory_offload.py
```

Extend coverage for each affected path. Compare against the frozen independent
reference using identical weights and tokens, including logits, hidden states
and KV state when applicable. Cover bulk/pipeline, group boundaries and slot
reuse, repeated forwards, multiple seeds, non-unit norm weights, prefill,
multi-step growing-cache decode, left padding, shape switches and GQA if the
change supports it. Include checks at both primary shapes for retained candidates.

Placement/scheduling-only changes must remain bit-exact against the folded
resident reference. For altered arithmetic/reduction order, define and justify
absolute/relative tolerances before inspecting candidate results; report max
absolute error, relative error with a stated near-zero rule, normalized RMS
error, finiteness and argmax agreement. Argmax agreement alone is insufficient.
Do not weaken gates to accept a candidate. The small-model gate's existing
0.05 absolute tolerance is not a universal tolerance for the full model.

## Required record for every attempt

Assign a unique monotonically increasing attempt ID before changing code.
Write an entry in profile/OPTIMIZATION_LOG.md and a structured record under
profile/results/optimization/offload-gap-001/<attempt_id>/, including failed
or reverted attempts. Include `campaign_id` in every record and log entry.

Within this campaign `A0000` identifies the frozen baseline; A0001–A0020
already exist. Continue monotonically with A0021 or the next unused ID. These IDs are scoped to `offload-gap-001`; they do not refer to
archived attempts. Never reuse an ID within this campaign after failure or
reversion. Separate an attempt from its repeated runs:
use run IDs such as `R01`, `R02` within the attempt directory. A code change
creates a new attempt; a repeat measurement of identical code does not.
Record `parent_attempt_id`, a short descriptive label, and `accepted_step`
(baseline 0; increment only for accepted cumulative improvements; null for
other attempts). Each accepted implementation must meet the offload latency,
absolute-gap reduction, regression and memory criteria in Objective and
constraints. Record tradeoffs
as separate candidates rather than silently replacing the best implementation.

Record:

- Hypothesis, expected bottleneck, change, parent baseline and exact source
  commit/hash or saved patch; original and candidate protocol versions.
- Stage and measurement plan, configuration, seed, environment, exact commands,
  raw timing samples, complete correctness output, memory data, logs and
  relevant profiles.
- Per-shape prefill/decode latency, throughput, memory and paired speedup
  against the new frozen baseline; matched resident latency, absolute gap,
  relative overhead and gap reduction with uncertainty.
- Retention status: accepted, rejected, within_noise, correctness_failed, oom,
  unsupported, benchmark_failed or interrupted. Include reason and next insight.
- Per-workload finding, separately from retention status: screening signal,
  confirmed local gain, resolved regression, within_noise or unavailable.
  Record comparator, effect size, uncertainty, corrected gain decision, memory
  tradeoffs and the specific reason preventing retention. "Not retained" does
  not mean "no useful gain". Preserve local gains even when another shape fails.
- Coverage ledger for all 16 workloads and both placements on each side:
  measured/reused/failed/not run, source and plan IDs, and reasons. Distinguish
  completed full screening from full validation and independent confirmation.
- For attempts blocked before timing, record performance as unavailable and
  explain why. Never substitute zero for failed/missing measurements.

Track workflow progress separately from the outcome status: screening,
confirmation, full_validation, then a final verdict. Passing screening or
primary confirmation is not `accepted` and does not advance `accepted_step`.
List intentionally unmeasured points as not run at this stage, with their
reason; do not confuse them with OOM, unsupported, or missing completed-run data.

Record failure evidence and a reproducible patch before reverting. Keep each
experiment focused on one hypothesis; combined optimizations need ablations.
Use the best verified implementation as the starting point for subsequent
attempts, while retaining the frozen baseline for final comparisons.

## Git audit trail and publishing

Every attempt, including a failed or rejected attempt, must have a committed
candidate implementation before measurement. Record the full candidate SHA
and parent SHA in the attempt metadata. Repeat runs of unchanged code share
that candidate commit. Commit the outcome, raw results and logs afterward;
include the attempt ID in both commit messages, for example
`experiment(A0001): overlap memory transfer` and
`results(A0001): rejected due to decode regression`. The results commit is
identified through Git history; do not try to embed its own SHA in itself.

Continue using campaign branch `exp-0/attempts/offload-gap-001`; preserve its
harness, baseline and attempt history. Do not recreate it from the restored
source or continue `attempts/paper-001`. Keep new attempt/result commits on
this campaign branch.
Any additional branches for this campaign must also use the `exp-0/` prefix.
For a rejected attempt, commit its evidence first, then revert the code change
with a new commit before the next attempt. Do not reset, squash or force-push
away attempt history. The final mainline can contain a separately curated set
of accepted changes, while the experiment branch preserves the full audit trail.

Remotes:

- `origin`: https://github.com/MollySophia/memory-attention.git (user's repository).
- `upstream`: https://github.com/Joluck/memory-attention.git (original repository).

The user authorizes pushing the campaign branch and its attempt/result commits
to `origin` during the experiment. Use explicit remote and branch names, e.g.
`git push -u origin exp-0/attempts/offload-gap-001`; do not rely on implicit push defaults.
Publish to `upstream` only after final results and the curated mainline are
ready. Routine intermediate attempt pushes belong to `origin`.

## Deliverables and stopping

Deliver a verified optimized implementation, regression tests, complete attempt
ledger and reproducible baseline/final commands. Plot from structured raw
results with a standalone script; export PDF/SVG and PNG plus source CSV/JSON.

Required figures: latency and throughput vs batch and context length for
prefill/decode; speedup per retained step; GPU/host memory vs workload; growing
generation latency; offload/resident absolute gap and relative overhead vs
workload and accepted step. Show measurement uncertainty, OOM/missing points and
failed attempts in the ledger. Label model size, precision, device, output/cache
scope and baselines in captions. Avoid implying quality validation from random
weights. Preserve historical results without relabeling their protocol.

For the optimization history, use attempt sequence (A0000, A0001, ...) on the
x-axis, with separate panels for prefill and decode at each primary batch
(1 and 8), keeping length/context fixed at 2048. Do not mix batch sizes in
a single history series.
Plot measured candidate latencies with status markers and uncertainty; show
failures without valid timings as annotations, not numerical points. Overlay
the incumbent accepted implementation as a step line using its stored measured
result; do not construct independent per-metric minima from different models.
This history shows the experiment process, not an additive decomposition.

For the paper's cumulative optimization figure, use accepted_step with short
labels (baseline, +change 1, +change 2, ...) on the x-axis. Use latency in ms
(lower is better), or speedup against the frozen baseline (higher is better,
baseline = 1). Keep the full attempt ledger, including failures, available.
Workload scaling figures use batch size or context length on the x-axis;
they answer a different question from optimization history. Do not connect
incomparable protocol/configuration points in a single series.

Show confirmed secondary-shape or generation gains in separate workload panels
against explicitly named comparators, with latency/gap reductions, intervals,
corrected decisions and unresolved regression/memory tradeoffs. Keep screening
heatmaps descriptive. Do not insert nonretained local gains into the cumulative
accepted-step figure or combine different candidates' best points into a
fictional final implementation.

## Execution order

1. For subsequently authorized work, verify the retained source, harness and
   evidence ledger. Continue monotonically after existing attempt IDs. If a
   new baseline is necessary, plan the complete matrix screen and profile once
   its primary points are ready, with timing/profiling sequential. Do not
   restart the completed campaign or rerun valid evidence without a reason.
2. Profile the relevant prefill/decode or generation path separately from
   timing. Identify a concrete offload bottleneck before registering a focused
   candidate. Record unavailable diagnostic data explicitly, never as zero.
3. Register and commit the candidate and full screening plan, run correctness
   gates, then complete all batch/length and generation screen points with
   matched parent/candidate offload and resident placements. Primary results
   cannot terminate the screen or exclude other shapes. Preserve failures and
   explicitly incomplete coverage when execution cannot finish.
4. Apply the predeclared nomination rule across the complete screen. Freeze
   and execute independent confirmation of gains and suspected regressions,
   including secondary/generation points, with paired uncertainty and fixed
   multiplicity accounting. Record local findings separately from retention.
   Inconclusive evidence remains within_noise, not equivalent or accepted.
5. For a candidate proposed for retention, confirm all four original primary
   workloads and the nominated gain, and meet the all-workload retention criterion. Complete the formal
   matrix, generation, applicable correctness and memory/regression gates.
   Reuse valid evidence where permitted; investigate regressions. A local gain
   with a regression elsewhere can motivate a separately registered
   shape-specific candidate, but cannot authorize automatic integration.
6. Commit the complete evidence and verdict, accept or revert, and continue
   from the best verified implementation only when further work is authorized.
   Plot retained cumulative improvements and nonretained local findings
   separately, preserving unfavorable and inconclusive results.

The original target was 5–10 focused, profiling-supported attempts; the completed
original campaign contains 20 attempts and two retained steps. Continuation-03
is now authorized without a fixed attempt cap; completing another bounded batch
does not satisfy its all-workload latency objective. The user-confirmed final
near-GPU criterion applies separately to all 16 workloads, including generation:
`T_offload - T_gpu <= max(0.01 * T_gpu, 0.1 ms)`.
Use matched resident latency for each workload, not a matrix-wide average.
Freeze the final independent paired-repeat plan before collecting its results;
do not use screening or candidate-selection observations as final confirmation.
For each independent pair, compute the signed residual
`T_offload - T_gpu - max(0.01 * T_gpu, 0.1 ms)` and assess its uncertainty.
Require the simultaneous one-sided 95% upper confidence bounds across all 16
workloads (Bonferroni correction over the fixed family of 16) to be at most zero.
A favorable mean or an inconclusive bound does not establish equivalence.
Report every workload's latency, gap, tolerance and residual bound; no missing
or failing workload can be offset by a gain elsewhere. Future attempt counts
are not a required number of accepted improvements. Preserve every
attempt, including failures, reversions and within_noise results, with its implementation, evidence and
verdict. Rejected, noisy or failed attempts count toward the attempt target
when they test a concrete offload hypothesis and record the outcome; they
do not advance accepted_step. Harness preparation and repeated runs of
unchanged code do not count as new optimization attempts. Each retained step requires correctness, independent gap/latency
confirmation, regression and memory gates. Do not substitute generic model
optimizations or superficial edits to reach the count. If offload-specific
opportunities are exhausted, report the evidence and the unmet target honestly.
No additional speedup target or total runtime budget has been set beyond the
per-workload near-GPU acceptance criterion above. Before launching
an expensive matrix, record the job count and estimated cost from measured
latencies, including warmup trajectories and per-process setup. These are
planning estimates, not permission gates or timeouts for live jobs.

Preserve per-job raw results automatically. Commit completed experiment stages
or meaningful groups of results, rather than every polling checkpoint. Keep
wait updates brief; verify the actual controller/child process before declaring
it stopped, and never restart solely because observation timed out. A failed or
unresolved campaign remains a valid record; do not declare success based on a
noisy point or omit failed attempts. Completing a baseline or bounded campaign
alone does not satisfy the optimization objective.
