# Ten additional offload optimization attempts

Completed exactly **A0011–A0020** in campaign `offload-gap-001`: **one accepted, five within noise, four rejected**. The retained implementation is **A0016**, source `0ce6645eb4c4bc3992258ad0878c72ea3c37a64e`, built on previously accepted A0001. Rejected and noisy implementations were reverted after their evidence commits. The [complete ledger](LEDGER.md) includes A0000 and all 20 attempts; earlier reports remain historical artifacts.

The new retained change uses one pipeline slot for automatic offload calls with **1025–2048 total input tokens**, preserving the existing 1024-token bulk cutoff and four-slot default for larger inputs. Explicit pipeline configuration remains unchanged. `memory_offload_single_slot_max_tokens=0` disables the new policy. The change reduces the small-prefill working set without changing precision, computation, table values, transfer volume or lookup freshness.

## Verified improvement

For batch 1 prefill at length 2048, the **incremental comparison against A0001** gives:

| Metric | Mean | 95% paired interval |
|---|---:|---:|
| Offload latency reduction | 0.430878 ms | [0.153321, 0.708434] |
| Absolute-gap reduction | 0.451613 ms | [0.304395, 0.598832] |
| Offload speedup | 1.014698× | [1.005171, 1.024225] |

The independently repeated **cumulative comparison against frozen A0000** gives 14.021136 ms offload reduction [13.510024, 14.532248], 14.015407 ms gap reduction [13.549947, 14.480867], and 1.474958× speedup [1.462204, 1.487712]. **Those cumulative numbers include A0001; they are not the gain from these ten attempts alone.** The parent comparison's gap moves from 0.720174 to 0.268560 ms; the separate A0000 comparison moves from 14.478608 to 0.463200 ms. Differences between these independent experiments are retained.

Sources: [incremental confirmation](../../A0016/parent-confirmation-analysis.json), [frozen-baseline confirmation](../../A0016/validation-v3-confirmation-analysis.json). Each uses three independent balanced process blocks for all four primary workloads and both offload/resident placements, with 10 warmups and 3 rounds of 10 measured calls per process. Intervals use Student-t with two degrees of freedom; symmetry is unverified and precision is limited. A slower resident reference cannot qualify as an improvement.

## Scope, results and limitations

The model has **2,836,499,968 parameters**, BF16, 24 layers, hidden size 2048, 32 query and KV heads, head dimension 64, SwiGLU size 5632, vocabulary 32000, untied embedding/head, qk_norm/use_gate/fuse_norm disabled for performance, seed 1234. Hardware/software: RTX 5090 32 GiB, driver 610.57.04, Ryzen 9950X, Python 3.12.2, torch 2.9.0+cu130, FlashAttention 2.8.3, torch threads 16/32 and 32-CPU affinity. Exact affinity, thread environment, competing processes and GPU telemetry are in every raw payload. The unchanged idle TRM process used approximately 654 MiB. GPU experiments ran sequentially.

Inputs start on GPU. Timed prefill includes embedding, all layers, norm, last-token logits and KV construction. Decode uses real fixed-context KV; benchmark cache rollback stays outside timing. Generation includes the 2048-token prefix and 128 predetermined GPU-token calls, excluding sampling. Seeded random weights establish performance/equivalence, **not language quality or serving latency**.

Final primary means from the fresh A0000/A0016 confirmation are below. Latencies are milliseconds; these means alone do not establish gains at other workloads.

| Workload | A0000 offload | A0000 resident | A0016 offload | A0016 resident | A0016 gap |
|---|---:|---:|---:|---:|---:|
| Prefill b1 | 43.542318 | 29.063710 | 29.521181 | 29.057981 | 0.463200 |
| Decode b1 | 4.910168 | 4.575402 | 5.014327 | 4.624927 | 0.389400 |
| Prefill b8 | 210.171380 | 206.748220 | 210.190545 | 206.765799 | 3.424746 |
| Decode b8 | 11.011948 | 10.940647 | 11.026391 | 10.926705 | 0.099686 |

No decode, batch-8 prefill or generation speedup is claimed. Important adverse observations remain:

- Parent-relative batch-8 decode is slower in all three blocks: reduction −0.038070 ms [−0.100821, 0.024682].
- A0000-relative batch-16 decode is slower in all three secondary pairs: −0.033485 ms [−0.082574, 0.015604]. Batch-8/4096 prefill is also slower in all three: −0.937146 ms [−2.972030, 1.097739].
- Batch-8/8192 prefill remains uncertain: −4.475341 ms [−15.449950, 6.499268].
- Batch-8 generation reduction is 0.438272 ms [−4.831173, 5.707717]. Batch-1 generation has only one original pair: offload 662.778028→645.514187 ms, with resident 595.323663→588.377195 ms. It does not establish an independent generation gain.

The predefined secondary rule reviewed all ten nonprimary matrix workloads and both generation workloads. Any negative original offload or resident reduction triggered exactly two further balanced pairs, retaining the original. Eight matrix workloads and batch-8 generation triggered this check: 64+8 new processes. All fixed regression/memory gates passed, but unresolved adverse directions above are **not proof of equivalence**. No outliers were discarded and no plan was extended until favorable. See [policy](../../A0016/secondary-regression-policy.json), [matrix analysis](../../A0016/full-regression-analysis.json) and [generation analysis](../../A0016/generation-regression-analysis.json).

## Correctness and memory

A0016 passed 31 regression tests and standalone maximum error zero. Four new full-model candidate trajectories at batches 1/8 and seeds 1234/4321 matched independently frozen resident references bit-for-bit: logits/argmax/finiteness at all 129 checkpoints, plus all hidden states and KV tensors at prefix and steps 1, 2 and 128. Reused resident fingerprints passed source, environment, configuration, tensor-shape and completeness checks. Small-model tests additionally cover padding, GQA, gating, non-unit norm weights, explicit policies, boundary shapes and 128 growing steps. [Source/correctness audit](../../A0016/source-correctness-audit.json).

All 84 full-matrix points and 12 generation points succeeded, including unfolded resident references; no OOM/unsupported point was dropped. Derived summaries preserve 200 raw process references, including primary confirmation and bounded secondary repeats. No additional identical GPU reruns were needed for final publication. GPU peak allocated savings remain positive at every matched point, with a conservative minimum of **1999.25 MiB**. The one-slot b1 prefill buffer is 8 MiB per host/device side; cached one-token bulk adds 0.09375 MiB. Larger b8/b16 capacities remain unchanged.

Host cost remains substantial: the ordinary CPU table is 3000 MiB, the raw table snapshot another 3000 MiB, and restore embedding copies approximately 3000 MiB (source-inferred capacity, included in measured RSS). Pinned transfer buffers, GPU allocated/reserved/peak memory, cached capacities and KV storage are separately recorded. KV rollback uses views rather than an independent full KV copy. A0016 adds no pinned ID buffer, mapped table, inverse map, write-combined allocation or lookup cache.

## The ten attempts

| Attempt | Hypothesis | Outcome and decisive evidence |
|---|---|---|
| A0011 | Reuse pinned input IDs with blocking staging | Within noise after 48 parent processes; no dual primary latency/gap gain. |
| A0012 | Blocking CUDA host-wait events | Within noise; CPU spin decreased, model gap gain unconfirmed. |
| A0013 | Mapped bulk gather for 8–16 tokens | Rejected screen; target b8 decode slower 0.025238 ms, with 3000 MiB table pinning disclosed. |
| A0014 | One slot for all automatic pipeline inputs | Rejected despite prefill gains: resolved b8 decode regression −0.040589 ms [−0.057702, −0.023475]. |
| A0015 | Deduplication with fused inverse gather/add | Within noise; target b8 gap reduction 0.158634 ms [−0.449015, 0.766283]. |
| A0016 | One slot restricted to small prefill | Accepted after independent parent/baseline confirmations, complete exactness, matrix, generation and regression gates. |
| A0017 | Write-combined buffers for large pipeline inputs | Within noise: b8 latency improved 1.765735 ms [0.688570, 2.842900], but gap reduction 1.567970 ms [−0.678259, 3.814199] failed the required criterion. |
| A0018 | Asynchronous ID staging overlapped with embedding | Rejected screen; b1 improvement only 0.007509 ms, b8 slower 0.160926 ms. |
| A0019 | Producer submission before embedding | Rejected screen; b1/b8 prefill slower 0.042619/0.366468 ms. |
| A0020 | Reuse coordination tickets/events | Within noise after 48 processes; b1 latency reduction 0.030322 ms [−0.111244, 0.171888], b8 0.000883 ms [−0.402357, 0.404122]. |

Component diagnostics and drafts do not count as attempts or accepted speedups. All ten have committed implementations, correctness output, actual model measurements and final verdicts. A0011–A0016 start from A0001; A0017–A0020 start from newly verified A0016. Full validation was intentionally not run for rejected/noisy candidates. [All records, exact source commits and links](LEDGER.md).

## Artifacts and reproduction

[Reproduction commands](REPRODUCE.md), [additional raw audit](additional-raw-audit.json), [core audit](core-audit.json), [manual review](manual-audit.json), and [publication record](publication.json) accompany the report. The additional audit checks 456 screen/parent-confirmation process references; the core audit also rechecks both retained implementations, including A0016's 200 final-validation references and raw correctness fingerprints.

Figures are exported as PNG/PDF/SVG with source CSV/JSON:

- [Batch scaling](validation/scaling-batch.png), [length scaling](validation/scaling-length.png), [batch memory](validation/memory-batch.png), [length memory](validation/memory-length.png).
- [Absolute gap](validation/absolute-gap.png), [relative overhead](validation/relative-overhead.png), [generation](validation/generation.png).
- [All-attempt history](history/attempt-history-screen_v1_w3_n5_r1.png), [accepted latency](history/accepted-steps-latency.png), [accepted speedup](history/accepted-steps-speedup.png), [accepted gap](history/accepted-steps-absolute-gap.png), [accepted overhead](history/accepted-steps-relative-overhead.png).

History uses provisional screens with sample SD and a single whole-model incumbent; its lines are never per-metric minima. Accepted-step figures use independent confirmations. Separate accepted steps were measured in separate balanced experiments: the plot is cumulative implementation history, not an additive decomposition. In particular, the new step's effect is established by its explicit A0001 parent confirmation, not by subtracting unrelated plotted means.
