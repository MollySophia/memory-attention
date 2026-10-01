# Memory Attention inference performance experiments

Status: proposed experiment protocol. This file describes future optimization
work; creating it does not start an optimization run. Freeze the measurement
protocol and establish a fresh baseline before evaluating optimizations.

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

Proposed validation matrix (freeze before optimization):

- Batch sweep: 1, 4, 8, 16 at prefill/context length 2048.
- Length sweep: 512, 2048, 4096, 8192 at batch 8.
- Record OOM and unsupported points explicitly; do not silently drop them.
- A growing-cache generation run: 128 decode steps after a 2048-token prefix,
  batches 1 and 8, fixed predetermined tokens for equivalence across variants.

Use the primary shape during iteration. Run the full matrix for a retained
candidate and the final implementation, including regression checks against
the frozen baseline at the other shapes.

## Measurement protocol to implement before optimization

Use profile/bench_fla.py for the actual model. profile/bmk.py implements a
separate experiment and must not supply this model's headline numbers.

Define and persist a protocol version in every result:

- Primary inference prefill: embedding, all layers, final norm, last-token LM
  head and KV cache construction. Return logits [batch, 1, vocab].
- Primary decode: one model call with real KV cache and logits
  [batch, 1, vocab]. Report latency per batch step and batch tokens/second.
- Retain the existing full-logits, use_cache=False prefill as a separate
  historical/ablation workload, with logits [batch, length, vocab].
- Apply logits_to_keep to prefill as well as decode; it currently only affects
  decode. Do not compare different output/cache scopes in the same speedup.
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

Start with 30 warmup calls, 30 samples per round, 5 rounds. Preserve every
sample, round means and run-level estimates. The current median_ms is a median
of round means, not a median of individual request latencies; label it precisely.
Report throughput as batch * sequence_length / seconds for prefill and
batch / seconds for decode. Report p50/p95 of actual samples if used, separately
from statistics over rounds.

For a claimed gain, rerun frozen baseline and candidate in alternating order
on the same machine, with at least 3 independent process pairs. Report the
distribution of paired speedups and uncertainty. Round spread is descriptive,
not a confidence interval or a universal significance threshold. Classify
unresolved improvements as within noise. Profile separately from timing runs.

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
Previous benchmark artifacts were removed to start with a clean dataset.
The historical tracked sweep is recoverable from commit 544b760; establish
a fresh A0000 under the frozen protocol and do not reuse old timings.

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
- Configuration, seed, environment, exact commands, raw timing samples,
  complete correctness output, memory data, logs and relevant profiles.
- Per-shape prefill/decode latency, throughput, memory and paired speedup
  against frozen baseline; resident-placement comparison separately.
- Status: accepted, rejected, within_noise, correctness_failed, oom,
  unsupported, benchmark_failed or interrupted. Include reason and next insight.
- For attempts blocked before timing, record performance as unavailable and
  explain why. Never substitute zero for failed/missing measurements.

Record failure evidence and a reproducible patch before reverting. Keep each
experiment focused on one hypothesis; combined optimizations need ablations.
Use the best verified implementation as the starting point for subsequent
attempts, while retaining the frozen baseline for final comparisons.

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

1. Preserve the current fixes/tests/protocol as a reproducible source revision.
2. Complete the measurement protocol, raw-sample/memory reporting and missing
   growing-cache correctness coverage. Verify the harness before kernel work.
3. Freeze the configuration/protocol and record fresh baseline A0000, including
   resident and offloaded placements. Keep the baseline code available for
   independent reruns; later protocol changes require a new baseline campaign.
4. Profile, register an attempt, change code, gate, measure, and record the
   outcome before accepting or reverting. Repeat in bounded campaigns.
5. Confirm retained candidates on the full matrix; generate final artifacts.

No numeric speedup target or runtime budget has been set. Work in explicitly
bounded experiment campaigns. A failed or unresolved campaign remains a valid
record; do not declare success based on a noisy point or omit failed attempts.
