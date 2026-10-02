# offload-gap-001 final report

Five focused optimization attempts are complete. Retain A0001 only: change the automatic bulk cutoff from 4096 to 1024 tokens, routing batch-1 length-2048 prefill through the existing bounded transfer pipeline. The verified model source is `942a2a52d94cf6dbe6e753f7ac291daf8b0699fb`; the final branch restores that model implementation after preserving and reverting unsuccessful candidates. Frozen A0000 is `62942a0387608fe21baaeb2dce9ef3b0947dde4d`, prepared from restored `81684d34e44db04a0511698081cab336abf4a60d`. Archived campaign optimizations and measurements were not imported.

Batch-1 prefill offload latency decreases from **43.573 to 29.774 ms**, a paired **1.463× speedup** (95% CI 1.456–1.471). Offload latency reduction is **13.799 ms [13.527, 14.070]**; absolute offload/resident gap decreases from **14.490 to 0.706 ms**, reduction **13.784 ms [13.497, 14.071]**. Relative overhead decreases from 49.824% to 2.430%. Resident latency is essentially unchanged (29.083 to 29.068 ms); a slower resident reference does not explain the gain.

These claims use three independent balanced process blocks, each containing both sources and both placements for all four primary workloads. Within each process the formal plan is 10 warmups, 10 samples × three rounds. Intervals use paired block differences and Student-t with two degrees of freedom. Only three independent pairs support each interval; normality is unverified and no multiple-comparison correction is applied. All prespecified workloads are reported. Screening sample spread does not support accepted gain claims.

| Primary workload, context 2048 | Baseline offload ms | Final offload ms | Baseline resident ms | Final resident ms | Offload reduction ms, 95% CI | Gap reduction ms, 95% CI |
|---|---:|---:|---:|---:|---|---|
| b1 prefill | 43.5727 | 29.7740 | 29.0826 | 29.0677 | 13.7987 [13.5270,14.0705] | 13.7839 [13.4968,14.0709] |
| b1 decode | 4.6132 | 4.7908 | 4.6241 | 4.6302 | -0.1777 [-0.6276,0.2723] | -0.1716 [-0.7983,0.4550] |
| b8 prefill | 210.0996 | 209.6951 | 206.7666 | 206.4984 | 0.4045 [-0.5911,1.4000] | 0.1363 [-2.3080,2.5806] |
| b8 decode | 11.0882 | 11.0131 | 10.9341 | 10.9219 | 0.0750 [-0.0530,0.2030] | 0.0628 [-0.0600,0.1856] |

No decode, batch-8 prefill or generation improvement is claimed. No other primary workload has a statistically resolved offload regression under the declared criterion; this does not establish equivalence or exclude smaller regressions.

## Attempts and mechanism

| Attempt | Hypothesis tested | Verdict | Evidence and next insight |
|---|---|---|---|
| A0001 | Pipeline above 1024 tokens to overlap prefill CPU gather/H2D | accepted, step 1 | Baseline b1 gather 11.166 ms and H2D stream span 3.497 ms explain the serial overhead. Independent latency and gap reductions pass all retention gates. |
| A0002 | Serial NumPy byte-copy gather for ≤8-token bulk calls | within_noise | Microprobe improves; b8 model decode reduction 0.0362 ms [-0.0018,0.0743], while gap reduction is positive. Both criteria are required. |
| A0003 | Minimum two-layer automatic pipeline groups | rejected | Screen b1/b8/b16 prefill slower by 2.465/4.057/9.968 ms versus A0001. Larger CPU gathers outweigh fewer handoffs and double slot capacity. |
| A0004 | Produce pipeline groups on caller thread | within_noise | b1 prefill latency reduction 0.2351 ms [0.0584,0.4117] passes, but gap reduction 0.2930 [-0.1118,0.6977] does not. |
| A0005 | Submit ≤8-token bulk H2D on consuming stream | within_noise | b1 decode reduction 0.3730 [-0.2758,1.0218], gap 0.4369 [-0.2188,1.0925]; b8 reduction 0.0216 [-0.1687,0.2119], gap 0.0035 [-0.1835,0.1906]. Neither qualifies. |

A0002–A0005 each start from verified A0001, so their incremental confirmation compares against A0001. They are separate focused hypotheses, not cumulative accepted steps. A0003 stops after screening; A0002/A0004/A0005 stop after the predeclared 48-process parent confirmation. Full validation and generation were intentionally not run for these unqualified candidates. Every candidate was committed before timing, with exact source, profiles, tests and raw results retained; outcome commits precede code-only reverts. A0005's initial implementation received an allocation-order correction before any performance measurement; only its final frozen revision was timed.

Profiler spans overlap and are not additive. The operator profiler lacked GPU events; this limitation is preserved, with separate working CUDA-event transfer diagnostics and CPU gather/coordination instrumentation. Missing events were not interpreted as zero. Timing and profiling ran sequentially.

## Validation, memory and limitations

The retained candidate passed 43 combined regression/sampling tests plus standalone exact-zero comparison. Full-size independent folded-GPU reference checks cover batches 1/8, seeds 1234/4321, prefill and 128 growing-cache decode calls. Logits are fingerprinted every step, with hidden states and complete KV states at prefill and decode steps 1/2/128. All comparisons are bit-exact, maximum error zero and argmax agreement 1. Small-model coverage also exercises non-unit norms, QK normalization, gating, GQA, padding, group boundaries, slot reuse and repeated forwards. Later rejected candidates passed their own extended gates; their tests and source remain in committed history.

Full validation includes all **84 points**: baseline/final × prefill/decode × seven batch/context shapes × offload/folded-GPU/unfolded-GPU. All complete; no OOM or unsupported points. Primary evidence is reused from eligible confirmation. Generation includes all **12 points**, prefix 2048 plus 128 predetermined-token steps at batches 1/8 and all three placements, with two whole-trajectory warmups and five trajectories × three rounds. The final source, plan and environment are unchanged, so final reporting reuses verified measurements.

Secondary apparent regressions were investigated with predeclared bounded repetitions, retaining original measurements:

- Batch-1 generation reduction -6.953 ms, CI [-89.584,75.678]; no generation benefit established.
- Prefill b4/L2048 reduction -0.289 ms [-3.049,2.471]; b8/L8192 -3.902 [-18.370,10.567].
- Prefill b16/L2048 is slower in all three audit pairs by 2.121, 6.184 and 1.875 ms. Mean reduction -3.393 ms [-9.406,2.619] is unresolved under the declared criterion. This consistent adverse direction is a limitation and must not be described as uniform improvement.

Secondary b4 decode means are 7.903 versus 8.278 ms from one process each. Both contain isolated 24.56/25.88-ms spikes; other rounds are near 7.33/7.34 ms. Bulk decode source is unchanged, but the cause is not established. All samples remain; there is no independent gain, equivalence or no-regression claim at this point. See [descriptive investigation](secondary-decode-review.json).

A0001/R02 had a pairing-order design error and is explicitly benchmark_failed, preserved and excluded from acceptance/reuse. Corrected R03 supplies formal evidence. R08 was interrupted by loss of environment access; recovery verified absent old processes, preserved its original manifest and completed only pending jobs, recovering the already-written J019 payload without resampling. No unfavorable samples were dropped.

The 2,836,499,968-parameter BF16 model has 3000 MiB memory tables and 2410.19 MiB remaining weights. Offload puts the table in ordinary CPU memory and pins gathered buffers. Final peak **allocated** GPU savings relative to matched folded-resident points remain at least **1999.25 MiB** across the full matrix and **2767.25 MiB** across generation. This is measured allocated-memory saving, not a claim about all physically reserved VRAM. Allocated/reserved before and peak, host RSS, pinned capacities, cached GPU buffers and KV storage are retained per raw result.

CPU RAM includes the 3000 MiB active folded table, a 3000 MiB raw snapshot and an additional 3000 MiB retained restore embedding: 9000 MiB table storage for offload. The last copy's capacity is inferred from source and explicitly distinguished from raw counters. All copies are already in measured RSS; do not add capacities to RSS. Resident variants retain the raw snapshot too. Fixed-context rollback slices KV views, without an independent full KV snapshot. Actual cache-update concatenation temporaries, live KV, activations, logits and cached offload buffers contribute to GPU peaks. See [memory accounting](../memory-accounting-audit.json).

## Scope and artifacts

Measurements use RTX 5090 32 GiB, Ryzen 9 9950X, driver 610.57.04, torch 2.9.0+cu130, FlashAttention 2.8.3, Python 3.12.2; torch intra-op 16/inter-op 32, OMP/MKL/OpenBLAS settings unset. Raw payloads record affinity, temperatures/clocks and competing GPU processes, including an idle TRM process at about 654 MiB. Seeded random weights establish performance and equivalence, not language quality.

Model dimensions: 24 layers, hidden 2048, query/KV heads 32, head dimension 64, SwiGLU intermediate 5632, vocabulary 32000, untied embeddings/head, qk_norm/use_gate/fuse_norm false. Inputs start on GPU. Prefill includes embedding, layers, norm, last-token head and KV construction; decode uses real cache. Runtime staging/gather/H2D/synchronization are timed; loading, folding, compilation, setup allocation, prefix setup for isolated decode, rollback and token selection are excluded. Generation includes prefix plus 128 model calls and excludes token selection; it is not serving latency.

Point estimates are means of process means. Three-process points show Student-t intervals; single-process matrix points show descriptive round-mean ranges, explicitly not confidence intervals. Raw `median_ms` denotes median of round means, not individual requests. Throughput is batch × sequence length / seconds for prefill and batch / seconds for isolated decode. Signed gaps and overheads, including negative values, remain visible.

- [Reproduction commands](REPRODUCE.md), [complete ledger](../../../../OPTIMIZATION_LOG.md), and individual [A0001](../A0001/record.json), [A0002](../A0002/record.json), [A0003](../A0003/record.json), [A0004](../A0004/record.json), [A0005](../A0005/record.json) records.
- [Primary formal analysis](../A0001/validation-v2-confirmation-analysis.json), [full audited summary](../A0001/full-audited-summary.json), [generation audited summary](../A0001/generation-audited-summary.json), [source/correctness audit](../A0001/source-correctness-audit.json).
- `validation/`: batch/context latency and throughput, GPU/host memory, generation, signed absolute gap and relative overhead. Every figure has PDF/SVG/PNG and `source.csv/json`.
- `history/`: four-panel attempt sequence with status markers and whole-model incumbent line; accepted-step latency, speedup, gap and overhead from formal evidence. Includes CSV/JSON sources. No per-metric cherry-picked incumbent and no screen-derived headline speedup.
- [Machine audit](core-audit.json) checks source, all final raw samples/configurations/environments, memory savings, exactness, verdicts and artifact presence; [manual audit](manual-audit.json) records remaining review.

Campaign branch `exp-0/attempts/offload-gap-001` preserves all implementation/result/revert commits and is published to origin. No upstream publication or mainline merge is included.
