# offload-gap-001 — five additional optimization attempts

**A0006–A0010 are complete. No new candidate is retained.** A0006 is within_noise after independent confirmation; A0007–A0010 are rejected at screening. The verified incumbent remains A0001, source `942a2a52d94cf6dbe6e753f7ac291daf8b0699fb`. Current model, benchmark and regression-test source has been restored to that implementation. The campaign now contains ten actual optimization attempts: one accepted, four within_noise and five rejected, plus frozen baseline A0000.

This continuation fulfills the requested five additional attempts. Repeated measurements, diagnostic prototypes and harness/audit work are not counted as attempts. Each candidate tests a distinct offload hypothesis, was registered and committed before model timing, passed its applicable correctness gates, and retains its exact source and raw evidence. Outcome commits precede code-only reverts. The [first-tranche report](../../final/REPORT.md) remains unchanged.

## Results of the additional attempts

All candidates start from accepted A0001. Screens are twelve isolated processes each: prefill/decode at batches1/8/16, length/context2048, ma_offload and folded ma_gpu; 3warmups/5samples/1round. Screen differences are descriptive, not confidence intervals or confirmed causal regressions. Only A0006 qualified for an independent parent confirmation. No candidate advanced to frozen-A0000 confirmation or full validation.

| Attempt | Change | Evidence relative to A0001 | Verdict |
|---|---|---|---|
| [A0006](../../A0006/record.json) | Refill a pinned host slot after its DMA finishes, before waiting for the old GPU consumer | Formal b1 prefill reduction0.1485ms,95%CI[-0.1604,0.4574]; gap reduction0.1595[-0.1934,0.5124]. Both unresolved. | within_noise |
| [A0007](../../A0007/record.json) | Precompute immutable source/buffer/layer tensor views | Screen prefill slower0.033/0.534/1.315ms at b1/b8/b16. No primary offload latency benefit. | rejected |
| [A0008](../../A0008/record.json) | GPU gather from mapped pinned CPU table, bounded64-CTA prefetch | Screen prefill slower0.045/3.638/11.161ms; b8/b16 gap worsens2.865/11.016ms. Whole3000MiB table becomes pinned. | rejected |
| [A0009](../../A0009/record.json) | Deduplicate IDs within each large pipeline call, transfer unique rows, restore order on GPU | Target b8 prefill slower2.017ms and gap worse1.173ms; b16 slower2.027ms. Threshold8192tokens fixed before timing from diagnostics. | rejected |
| [A0010](../../A0010/record.json) | Two alternating pinned host buffers per unchanged GPU slot | Screen prefill slower1.731/0.757/1.309ms; pinned pipeline capacity doubles. No primary prefill gain. | rejected |

A0006's formal plan contains48 separate processes: all four primary workloads × both sources × both placements × three balanced blocks. Each process has10warmups,10samples per round and three rounds. The fixed plan was closed without extending until favorable. Paired block differences use95% Student-t intervals with two degrees of freedom; three independent pairs give limited precision, normality is unverified and no multiple-comparison correction is applied.

| A0006 primary workload | Offload reduction ms,95%CI | Absolute-gap reduction ms,95%CI |
|---|---|---|
| b1 prefill | 0.1485[-0.1604,0.4574] | 0.1595[-0.1934,0.5124] |
| b1 decode | 0.0590[-0.0730,0.1910] | 0.0842[-0.1163,0.2846] |
| b8 prefill | -0.2271[-1.4397,0.9855] | -0.0453[-1.8011,1.7104] |
| b8 decode | -0.0035[-0.1409,0.1339] | -0.0141[-0.1548,0.1267] |

None meets the dual positive lower-bound criterion. No resolved primary regression or loss of GPU savings occurred in this confirmation. This does not prove equivalence. A0009's only changed primary target is b8 prefill; fluctuations on unchanged b1/decode paths cannot establish a new optimization step. Large b1 decode fluctuations in several short screens are preserved without attribution to unchanged bulk algorithms or sample removal.

## What the profiling established

Frozen A0001 coordination instrumentation observes96 buffer-view operations totaling0.316ms at b1 and1.089ms at b8; producer DMA synchronization totals12.763/60.918ms, and consumer-release waits1.176/0.041ms. These overlapping instrumented spans are not additive and are not removable wall time. They support the hypotheses tested by A0006, A0007 and A0010, while the model results show that reducing or rearranging those costs was insufficient.

Separate fresh-ID one-layer diagnostics compare CPU gather/H2D, direct mapped-host reads, and per-call deduplication/H2D/GPU expansion. All successful outputs are bit-exact. In the first diagnostic b8 means are3.941/1.669/3.911ms; b16 means7.951/3.382/5.762ms. A second bounded-CTA probe selects64CTAs before model timing: mapped b1/b8/b16 means0.233/1.711/3.374ms versus CPU gather/H2D0.339/4.096/7.936ms. It sacrifices a little isolated throughput to limit GPU resource use.

These component gains did not translate to the model. Mapped kernels use GPU resources and require host-read lifetime synchronization; deduplication adds sorting/inverse creation and GPU expansion. Their individual causal contributions to the observed model slowdowns were not isolated. The full pipeline already overlaps substantial work with model compute. No hardware-independent speedup or impossibility claim follows from these results.

The mapped-memory feasibility review used NVIDIA's [CUDA memory documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html) and [CUDA13.0 best practices](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-best-practices-guide/index.html), together with inspection of the installed Triton launcher. Actual capability and exactness were then tested locally. Component scripts do not run concurrently with model timing. Compiler warnings and all diagnostic outputs remain preserved in the continuation directory.

## Correctness and memory

A0006/A0007/A0008/A0009/A0010 pass46/47/50/47/47 combined regression and sampling tests, respectively, plus the standalone comparison with maximum error zero. Coverage includes128-step growing cache, two seeds, non-unit memory norms, enabled QK normalization/gating, GQA, padding and slot reuse. New path-specific coverage includes early host refill and cancellation; fresh weights/IDs through cached metadata; mapped reads across streams and immediate CPU-table mutation after return; changing unique-ID counts and restored order; and delayed GPU consumption with alternating host storage. A0009 forces the dedup path in the small-model growing-cache tests. These are correctness gates, not full-size validation claims for rejected candidates.

Every offload implementation keeps table values fresh between calls. A0007 caches only metadata aliases. A0009 shares duplicate rows only within one call. No quantization, dimensions, attention semantics or output scope changes are used. A0008's print-only correction before timing distinguishes pinned from ordinary CPU table storage; its preliminary correctness output is preserved, and only the final frozen source is timed.

Memory costs are explicit:

- A0008 pins the complete3000MiB CPU table. Total active pinned capacity including cached bulk decode buffers is3000.09375MiB at b1 and3000.75MiB at b8. GPU table residency is not introduced. It is rejected despite passing exactness tests.
- A0009 includes the retained GPU inverse-index buffer in offloader capacity counters; expanded row temporaries contribute to measured GPU peaks. The per-call inverse contains one int64 index per original token.
- A0010 doubles pipeline host capacity while keeping GPU slot capacity unchanged. Total cached pinned buffers at b1/b8/b16 are64.09375/512.75/1025.5MiB, including unchanged bulk decode buffers. The extra host memory does not produce a retained gain.

The retained A0001 remains unchanged:3000MiB CPU folded table,3000MiB raw snapshot and3000MiB restore embedding, already included in measured RSS. Restore capacity is source-inferred rather than a separate measured counter. KV rollback uses views, not a full independent KV snapshot. GPU peaks include real cache-update temporaries, live KV, activations, logits and cached buffers. Minimum matched peak allocated GPU savings remain1999.25MiB across its full matrix and2767.25MiB across generation. These are allocated-memory savings, not a statement that all reserved VRAM is released.

## Retained result, scope and limitations

A0001's previously confirmed b1 prefill result remains **43.573→29.774ms**, paired **1.463×** speedup,95%CI[1.456,1.471]. Absolute offload/resident gap decreases14.490→0.706ms; gap reduction13.784ms[13.497,14.071]. This is the original accepted improvement, not a new gain from this continuation. No decode, b8 prefill or generation gain is newly claimed.

The same verified source, protocol and environment permit reuse of A0001's84-point full matrix,12-point generation matrix and four full-size independent exactness comparisons. No identical final GPU sweep was repeated. Full-size checks cover batches1/8, seeds1234/4321, all129 logit checkpoints and selected full hidden/KV states through128 growing steps, with exact byte fingerprints, finiteness and argmax equality.

Original limitations remain visible: b16 prefill is slower in all three secondary audit pairs, with mean reduction-3.393ms and95%CI[-9.406,2.619]; b1 generation reduction-6.953ms[-89.584,75.678] is unresolved; b4 isolated decode has one-process spikes without a no-regression claim. Original invalid A0001/R02 pairing results remain excluded, and corrected R03 supplies retention evidence. See the [original full report](../../final/REPORT.md) for these investigations and the interrupted R08 recovery audit.

All model measurements use2,836,499,968 parameters, BF16,24layers, hidden2048,32query/KV heads, head dimension64, intermediate5632 and vocabulary32000; untied embedding/head, qk_norm/use_gate/fuse_norm false, performance seed1234. RTX5090, driver610.57.04, torch2.9.0+cu130, FlashAttention2.8.3, Python3.12.2, Ryzen9950X, intra-op16/inter-op32. Raw metadata records affinity, thread settings and GPU telemetry; an idle TRM process uses about654MiB GPU memory. Seeded random weights establish performance/equivalence, not language quality.

Inputs begin on GPU. Prefill includes embedding, layers, final norm, last-token head and KV construction; decode uses a real fixed-context cache. Runtime staging/gather/transfers/synchronization remain timed. Loading, folding, compilation/setup allocation, isolated-decode prefix setup, rollback and token selection are excluded. Generation includes prefix plus128 predetermined-token model calls; it excludes selection and is not serving latency. Primary throughput is batch×length/seconds for prefill and batch/seconds for decode. Signed gaps and uncertainty are preserved. Raw median_ms is the median of round means, not request p50.

## Audit trail and deliverables

- [Reproduction commands](REPRODUCE.md), [complete attempt ledger](../../../../../OPTIMIZATION_LOG.md), [continuation scope](../manifest.json).
- [Additional raw audit](additional-raw-audit.json):108 new model process results, exact frozen sources, twelve-job screen grids, A0006 balanced confirmation, sampling/configuration, correctness logs and added memory bookkeeping. Diagnostic runs are not included in the108 count.
- [Core retention audit](core-audit.json):ten total/five additional final verdicts, current source equals verified A0001,160 referenced retained-validation raw processes, exactness, memory and artifact checks.
- [Manual review](manual-audit.json):report, commands, figures, source/revert history and limitations. [Publication record](publication.json) records the report commit verified on origin.
- `validation/`:seven figure families for batch/context latency and throughput, GPU/host memory, generation, signed gap and relative overhead. PDF/SVG/PNG plus sourceCSV/JSON; reused A0001 evidence is labeled.
- `history/`:cumulative A0000–A0010 four-panel screening history with final status markers and a single whole-model incumbent line; accepted-step latency, speedup, gap and overhead use formal evidence. Five figure families, PDF/SVG/PNG and sourceCSV/JSON. No per-metric minima and no new accepted-step point for rejected/noisy candidates.

The campaign branch is published to origin as `exp-0/attempts/offload-gap-001`. No upstream publication or mainline merge is included.
