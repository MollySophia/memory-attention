# Memory Attention inference performance experiments

Status: continued optimization campaign; A0002 is the accepted starting point.
The user now requests5–10 effective optimization iterations. Preserve A0000
and A0002 evidence; continue new attempts from A0003 onward.

Frozen baseline A0000 already completed under
paper_v1. Preserve its source, raw measurements and figures. This revision
changes the order and sampling plans for future work; it does not invalidate,
relabel or require rerunning the completed baseline. Use the staged workflow
below to reach useful optimization evidence before spending on full validation.

## Objective and constraints

Optimize MemoryForCausalLM prefill and decode end-to-end model latency while
preserving its computation and the CPU offload memory saving. Use fixed seeded
random weights and input tokens; these experiments support performance and
numerical equivalence claims, not language model quality claims.

Primary target: ma_offload. Keep a frozen ma_offload implementation as the
optimization baseline, and ma_gpu as the folded resident placement reference.
Use ma_gpu_unfolded to measure the norm-folding contribution separately.
Model-wide improvements may also benefit ma_gpu; record both placements.
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

Primary shape: batch 8, prefill length 2048, decode prefix length 2048.
Prefill IDs [8, 2048], hidden states [8, 2048, 2048]; decode IDs [8, 1].
Attention Q/K/V heads have dimension 64. Prefill transfers at most
batch * length * layers * kv_dim * sizeof(dtype) = 1536 MiB of gathered
memory values at this shape; decode transfers 0.75 MiB per step.

Final validation matrix (frozen for the current campaign):

- Batch sweep: 1, 4, 8, 16 at prefill/context length 2048.
- Length sweep: 512, 2048, 4096, 8192 at batch 8.
- Record OOM and unsupported points explicitly; do not silently drop them.
- A growing-cache generation run: 128 decode steps after a 2048-token prefix,
  batches 1 and 8, fixed predetermined tokens for equivalence across variants.

Use the primary shape during iteration. The full matrix is an acceptance gate
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
  timed interval (implemented as fixed_context_v2_rollback_excluded). Keep a
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
The existing A0000 paper_v1 records retain their original configuration; do not
retroactively edit their metadata. Freeze each plan before collecting its data.
These are starting plans, not universal sample-size guarantees. Independence
comes from separate paired processes; many samples in one process do not replace
that requirement. Do not apply the old 30-warmup/150-sample plan to every point.

| Stage | Workloads and placements | Warmup / samples per round / rounds | Purpose |
| --- | --- | --- | --- |
| Screening | Primary prefill and decode; ma_offload first, ma_gpu for attribution | 3 / 5 / 1 per point | Reject poor ideas quickly; no accepted gain claims |
| Confirmation | Primary prefill and decode; independent frozen baseline/candidate pairs for ma_offload, resident comparison separately | 10 / 10 / 3 per point | Repeatability and primary regression checks |
| Full validation | Complete batch/length matrix for a candidate nominated for retention, including placement/folding references | 10 / 10 / 3 per prefill/decode point | Scaling, memory and regression acceptance gates |
| Generation validation | Prefix2048 + 128 steps; batches1/8, placement/folding references | 2 full-trajectory warmups / 5 trajectories / 3 rounds | Growing-cache latency and memory |

Use `profile/bench_fla.py --stage <stage>` for a single configuration and
`profile/run_paper_matrix.py --stage <stage>` for a stage's workload set.
Both default to screening (single-config generation defaults to
generation_validation). `--stage legacy` selects the original 30/30/5 sampling;
use the frozen checkout to reproduce the exact A0000 implementation.
`--plan-only` on the matrix driver writes commands and planned sample/call counts
without launching measurements. Confirmation jobs from this driver represent
one side of a run; schedule independent alternating baseline/candidate pairs
separately. Explicit warmup/repeats/rounds overrides on bench_fla receive a
custom plan ID and retain their actual counts in the result.

Screening still requires compilation and offload buffer setup before timing.
If the short warmup is insufficient, increase it for both baseline and candidate
and record the revised plan before comparing. Short-plan results are provisional
and must never supply headline or accepted-step speedups. These sampling plans
do not shorten correctness tests: retain multi-step/growing-cache gates even
when generation performance is not measured during screening.

A generation warmup/sample is an entire prefix plus 128-step trajectory, not
one decode call. Its warmup count is independent of the single-call benchmark.
The completed A0000 generation data used 30 trajectory warmups and remain valid
under that recorded plan. For a future generation speedup comparison, either
use that original plan for both sides, or rerun only the matching baseline
and candidate generation points under the new plan. This also applies to the
reduced prefill/decode plans: collect matching baseline measurements for formal
comparisons, without rerunning unrelated baseline matrix points. Do not silently
compare different sampling plans as matched evidence.

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
on the same machine, with at least 3 independent process pairs. Report the
distribution of paired speedups and uncertainty. Round spread is descriptive,
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
The historical tracked sweep is recoverable from commit 544b760 and is not the
current baseline. A0000/R01 is the completed paper_v1 baseline (48 points,
7200 samples); its frozen candidate source is
`d949640ebf2f13f56021bd08c5c9f10e65d571c3`.
Use it for scaling/context and preserve the independent frozen checkout.
Claimed improvements still require fresh matching paired runs; a stored
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
change supports it. Include a primary-shape check for retained candidates.

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
profile/results/optimization/<attempt_id>/, including failed/reverted attempts.

Use `A0000` for the frozen baseline, then `A0001`, `A0002`, etc. Never reuse an
ID after failure or reversion. Separate an attempt from its repeated runs:
use run IDs such as `R01`, `R02` within the attempt directory. A code change
creates a new attempt; a repeat measurement of identical code does not.
Record `parent_attempt_id`, a short descriptive label, and `accepted_step`
(baseline 0; increment only for accepted cumulative improvements; null for
other attempts). One cumulative implementation must improve at least one
primary latency with repeatable evidence without a resolved regression in the
other primary latency or loss of the offload memory saving. Record tradeoffs
as separate candidates rather than silently replacing the best implementation.

Record:

- Hypothesis, expected bottleneck, change, parent baseline and exact source
  commit/hash or saved patch; original and candidate protocol versions.
- Stage and measurement plan, configuration, seed, environment, exact commands,
  raw timing samples, complete correctness output, memory data, logs and
  relevant profiles.
- Per-shape prefill/decode latency, throughput, memory and paired speedup
  against frozen baseline; resident-placement comparison separately.
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

Use a dedicated campaign branch such as `attempts/paper-001`, created from the
prepared source revision. Keep all attempts and result commits on that branch.
For a rejected attempt, commit its evidence first, then revert the code change
with a new commit before the next attempt. Do not reset, squash or force-push
away attempt history. The final mainline can contain a separately curated set
of accepted changes, while the experiment branch preserves the full audit trail.

Remotes:

- `origin`: https://github.com/MollySophia/memory-attention.git (user's repository).
- `upstream`: https://github.com/Joluck/memory-attention.git (original repository).

The user authorizes pushing the campaign branch and its attempt/result commits
to `origin` during the experiment. Use explicit remote and branch names, e.g.
`git push -u origin attempts/paper-001`; do not rely on implicit push defaults.
Publish to `upstream` only after final results and the curated mainline are
ready. Routine intermediate attempt pushes belong to `origin`.

## Deliverables and stopping

Deliver a verified optimized implementation, regression tests, complete attempt
ledger and reproducible baseline/final commands. Plot from structured raw
results with a standalone script; export PDF/SVG and PNG plus source CSV/JSON.

Required figures: latency and throughput vs batch and context length for
prefill/decode; speedup per retained step; GPU/host memory vs workload; growing
generation latency. Show measurement uncertainty, OOM/missing points and
failed attempts in the ledger. Label model size, precision, device, output/cache
scope and baselines in captions. Avoid implying quality validation from random
weights. Preserve historical results without relabeling their protocol.

For the optimization history, use attempt sequence (A0000, A0001, ...) on the
x-axis, with prefill and decode in separate panels at the same primary shape.
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

1. Reuse the existing frozen A0000 source and results. For a future campaign,
   verify the harness/correctness gates and freeze scope/configuration first,
   then establish only the primary offload/resident baseline needed to begin.
   A complete baseline matrix is not a prerequisite for the first profile.
2. Profile primary prefill/decode separately from timing. Identify a concrete
   bottleneck before registering and implementing a focused candidate. If a
   profiling backend lacks GPU events, record that limitation and use a working
   diagnostic path; never treat unavailable timings as zero.
3. Register and commit the candidate, run correctness gates, then screen primary
   offload prefill/decode with the short plan. Compare against a matching frozen
   baseline run. Reject clear regressions or unpromising ideas promptly; collect
   resident measurements where they help attribute a model-wide change.
4. For a promising candidate, run formal primary confirmation with at least
   three independent alternating baseline/candidate process pairs. Confirm
   memory savings and the absence of a resolved regression in the other primary
   latency. Record inconclusive results as within_noise, not accepted.
5. Nominate a candidate for retention only after confirmation. Run the full
   matrix, generation validation and all applicable correctness checks before
   acceptance. Reuse valid existing measurements where source, scope, sampling
   plan and environment match; rerun matching baseline points where required
   for comparisons. Investigate regressions rather than silently dropping them.
6. Commit the verdict/evidence, accept or revert, and continue from the best
   verified implementation. Generate final figures and reproducible commands
   from accepted evidence without duplicating an already completed identical
   validation run.

The original three-candidate campaign limit is superseded by the user’s
continuation request. Pursue5–10 effective optimization iterations starting
from accepted A0002, registering new focused attempts from A0003 onward.
Rejected/noisy ideas remain valuable evidence but do not count as effective
accepted improvements. Each retained step requires the existing correctness,
independent confirmation, regression and memory gates. Do not stop merely
because five superficial edits or failed screens have been recorded.
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
