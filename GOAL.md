# Memory Attention inference performance experiments

Status: new experiment specification; not started. Editing this document does
not resume the paused campaign or launch measurements.

Campaign ID: `offload-gap-001`. Start from the restored `feat/offload` code at
`81684d34e44db04a0511698081cab336abf4a60d`, not an accepted implementation from
the archived campaign. Prepare and verify the measurement harness, then freeze
its exact source commit as this campaign's new A0000. No baseline measurements
or accepted optimization steps exist yet for this campaign.

Keep the old `attempts/paper-001` branch and archived results as historical
records only. Do not import their accepted-step counts, baseline measurements
or source changes into the new experiment. Use the staged workflow below to
obtain a primary baseline before spending on full validation.

## Objective and constraints

Optimize the Memory Attention architecture's CPU memory-table offload path.
The primary objective is to reduce the prefill and decode latency gap between
ma_offload and the equivalent all-GPU ma_gpu implementation, while preserving
model computation and the CPU offload memory saving. Use fixed seeded
random weights and input tokens; these experiments support performance and
numerical equivalence claims, not language model quality claims.

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
reference behavior fixed. A retained step must repeatably reduce offload
latency and its absolute gap for at least one of the four predefined primary
workloads (prefill/decode at batches 1 and 8, length/context 2048), without a
resolved regression in any other primary workload or loss of memory savings.
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

Initial baseline screening: measure batches 1, 4, 8 and 16 at length/context
2048, both prefill and decode, with ma_offload and ma_gpu. This is 16 isolated
process jobs under the screening sampling plan. It establishes batch scaling
before candidate selection; it is not formal confirmation or full validation.
Do not infer small-batch overhead from batch 8 alone.

Final validation matrix (frozen for the current campaign):

- Batch sweep: 1, 4, 8, 16 at prefill/context length 2048.
- Length sweep: 512, 2048, 4096, 8192 at batch 8.
- Record OOM and unsupported points explicitly; do not silently drop them.
- A growing-cache generation run: 128 decode steps after a 2048-token prefix,
  batches 1 and 8, fixed predetermined tokens for equivalence across variants.

Every candidate screening covers both primary batches 1 and 8, prefill and
decode, and both placements (8 jobs per implementation). Candidates affecting
transfer volume or pipeline scheduling also screen batch 16 (4 additional jobs
per implementation), against matched frozen-baseline points. Declare these
additional points before collecting candidate timings. The full matrix is an acceptance gate
for a candidate nominated for retention, not a prerequisite for profiling or
screening every idea. Include regression checks against the frozen baseline at
all other shapes before accepting a retained candidate. If the final
implementation is the same verified source with the same measurement plan and
environment, reuse that full-matrix evidence rather than rerunning it simply
because the candidate is now called final.

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
Use a new protocol identifier for this campaign after verifying the harness.
Leave archived records and their metadata unchanged. Freeze each plan before
collecting its data.
These are starting plans, not universal sample-size guarantees. Independence
comes from separate paired processes; many samples in one process do not replace
that requirement. Do not apply the old 30-warmup/150-sample plan to every point.

| Stage | Workloads and placements | Warmup / samples per round / rounds | Purpose |
| --- | --- | --- | --- |
| Screening | Initial baseline: batches 1/4/8/16 (16 jobs); candidates: batches 1/8 (8 jobs per implementation), plus batch 16 for transfer-volume/pipeline changes; prefill/decode at length/context 2048, ma_offload and ma_gpu | 3 / 5 / 1 per point | Reject poor ideas quickly; no accepted gain claims |
| Confirmation | All four primary workloads: prefill/decode at batches 1/8, length/context 2048; independent baseline/candidate pairs for both ma_offload and ma_gpu in balanced blocks | 10 / 10 / 3 per point | Repeatable offload gap reduction and primary regression checks |
| Full validation | Complete batch/length matrix for a candidate nominated for retention, including placement/folding references | 10 / 10 / 3 per prefill/decode point | Scaling, memory and regression acceptance gates |
| Generation validation | Prefix2048 + 128 steps; batches1/8, placement/folding references | 2 full-trajectory warmups / 5 trajectories / 3 rounds | Growing-cache latency and memory |

Harness preparation is required before these plans can run. The restored
checkout's `profile/bench_fla.py` still defaults to 30/30/5; the archived
`--stage`, matrix driver and diagnostic profiler are not present in this
checkout. Do not describe those interfaces as already implemented or launch
the old defaults. Port only the necessary measurement infrastructure, review
it independently of archived model optimizations, and verify it before A0000.

The prepared single-configuration and matrix drivers must default to screening
(generation to generation validation), record actual counts and custom plan
IDs, and offer a plan-only mode with commands, job counts and estimated work.
Diagnostic profiling should default to 10 warmup calls, allow an explicit
override, and run after a primary baseline without waiting for a full matrix.
Independent alternating baseline/candidate runs must cover both placements;
a single matrix run is not independent confirmation.

Screening still requires compilation and offload buffer setup before timing.
If the short warmup is insufficient, increase it for both baseline and candidate
and record the revised plan before comparing. Short-plan results are provisional
and must never supply headline or accepted-step speedups. These sampling plans
do not shorten correctness tests: retain multi-step/growing-cache gates even
when generation performance is not measured during screening.

A generation warmup/sample is an entire prefix plus 128-step trajectory, not
one decode call. Its warmup count is independent of the single-call benchmark.
Collect fresh baseline and candidate generation points under the same new
plan. For subsequent comparisons within this campaign, reuse valid measurements
only when source, scope, plan and environment match; collect fresh independent
runs for confirmation. Do not compare different sampling plans as matched
evidence or rerun unrelated matrix points merely to refresh one comparison.

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
it in every screening run. For a rejected screen, record the evidence and move
on without running its full matrix or generation performance sweep.

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
Create a new frozen checkout from the prepared A0000 commit and record its
path, source hash and environment in this campaign's manifest. Neither the
historical tracked sweep nor the archived campaign's baseline is this new
baseline. Claimed improvements require fresh matching paired runs; a stored
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

Within this new campaign use `A0000` for the new frozen baseline, then `A0001`,
`A0002`, etc. These IDs are scoped to `offload-gap-001`; they do not refer to
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
- Status: accepted, rejected, within_noise, correctness_failed, oom,
  unsupported, benchmark_failed or interrupted. Include reason and next insight.
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

Use the new campaign branch `exp-0/attempts/offload-gap-001`, created from the
restored source revision when execution begins. Keep harness preparation and
the new baseline freeze on this branch; do not continue `attempts/paper-001`.
Keep all new attempts and result commits on the new branch.
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

## Execution order

1. Start a new campaign from the restored source. Prepare and verify the
   harness/correctness gates, freeze scope/configuration and the new A0000
   source, then measure the fresh 16-job initial baseline screen: batches
   1/4/8/16, prefill/decode at length/context 2048, ma_offload and ma_gpu.
   Record their absolute gaps and relative overheads. Do not reuse archived
   baseline timings. Profile as soon as the primary batch-1/8 results are
   available, with timing jobs paused or finished to avoid interference; the
   remaining baseline batch points must finish before selecting a candidate.
   A complete validation matrix is not a prerequisite for the first profile.
2. Profile prefill/decode at both primary batches 1 and 8 separately from timing. Identify a concrete
   offload bottleneck and its contribution to the resident/offload gap before
   registering and implementing a focused candidate. If a
   profiling backend lacks GPU events, record that limitation and use a working
   diagnostic path; never treat unavailable timings as zero.
3. Register and commit the candidate, run correctness gates, then screen primary
   offload prefill/decode at batches 1 and 8 with the short plan; add batch 16
   for transfer-volume or pipeline-scheduling changes. Compare against a matching frozen
   baseline run, including matched resident measurements to attribute gap
   changes. Reject clear regressions or unpromising ideas promptly.
4. For a promising candidate, run formal primary confirmation with at least
   three independent alternating baseline/candidate process pairs per placement
   for each of the four primary workloads. Confirm lower offload latency and
   absolute gap in at least one, preserved memory savings, and no resolved
   regression in any of the others. Record inconclusive results as within_noise,
   not accepted.
5. Nominate a candidate for retention only after confirmation. Run the full
   matrix, generation validation and all applicable correctness checks before
   acceptance. Reuse valid existing measurements where source, scope, sampling
   plan and environment match; rerun matching baseline points where required
   for comparisons. Investigate regressions rather than silently dropping them.
6. Commit the verdict/evidence, accept or revert, and continue from the best
   verified implementation. Generate final figures and reproducible commands
   from accepted evidence without duplicating an already completed identical
   validation run.

The target is 5–10 focused, profiling-supported offload optimization attempts
in this new campaign, starting with no accepted steps. This is an attempt
count, not a requirement for 5–10 accepted improvements. Register candidates
from A0001 after freezing A0000. Preserve every attempt, including failures,
reversions and within_noise results, with its implementation, evidence and
verdict. Rejected, noisy or failed attempts count toward the attempt target
when they test a concrete offload hypothesis and record the outcome; they
do not advance accepted_step. Harness preparation and repeated runs of
unchanged code do not count as new optimization attempts. Each retained step requires correctness, independent gap/latency
confirmation, regression and memory gates. Do not substitute generic model
optimizations or superficial edits to reach the count. If offload-specific
opportunities are exhausted, report the evidence and the unmet target honestly.
No numeric speedup target or total runtime budget has been set. Before launching
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
