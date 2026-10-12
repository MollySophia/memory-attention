# R0002: fused memory recall

Initial baseline: `e5b52be0be5484a6f8d0c4391b158cb71e561759`. Candidate: `b7c33a6ef3679871d88871e528a35f7131ee2597`. Candidate fuses historical GPU embedding lookup, per-head RMSNorm and K+M into one Triton kernel during inference decode. It preserves BF16/FP16 normalization rounding, stores no historical V cache and creates no full-size M intermediates. Prefill, training and incompatible layouts/hooks use the existing path. No table normalization folding, CPU offload or RoPE is used.

## Fresh three-way latency comparison

Milliseconds, lower is better. Negative change means faster. All three paths were freshly measured in the same campaign, with matching weights and no-RoPE configuration. Three independent process blocks rotate the three placements through every ordering position for every workload. Paired two-sided 95% Student-t intervals (df2) are in summary.json/CSV. They describe these measurements, not a universal bound.

| Mode | Batch | Length/context | Full KV | Initial recall | Fused recall | Fused vs initial | Fused vs full KV |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 2048 | 28.969 | 28.961 | 28.972 | +0.04% | +0.01% |
| decode | 1 | 2048 | 3.999 | 4.367 | 3.521 | -19.38% | -11.96% |
| prefill | 4 | 2048 | 104.555 | 104.837 | 104.583 | -0.24% | +0.03% |
| decode | 4 | 2048 | 7.185 | 7.619 | 6.080 | -20.20% | -15.37% |
| prefill | 8 | 2048 | 204.265 | 204.187 | 203.942 | -0.12% | -0.16% |
| decode | 8 | 2048 | 10.909 | 13.232 | 10.227 | -22.71% | -6.25% |
| prefill | 16 | 2048 | 413.227 | 412.988 | 413.038 | +0.01% | -0.04% |
| decode | 16 | 2048 | 20.246 | 28.106 | 19.778 | -29.63% | -2.31% |
| prefill | 8 | 512 | 51.142 | 51.168 | 51.187 | +0.04% | +0.09% |
| decode | 8 | 512 | 5.572 | 5.480 | 4.993 | -8.89% | -10.40% |
| prefill | 8 | 4096 | 442.376 | 442.791 | 442.859 | +0.02% | +0.11% |
| decode | 8 | 4096 | 18.935 | 27.124 | 18.810 | -30.65% | -0.66% |
| prefill | 8 | 8192 | 999.331 | 1000.467 | 1000.261 | -0.02% | +0.10% |
| decode | 8 | 8192 | 34.648 | 51.268 | 34.681 | -32.35% | +0.10% |
| prefill | 32 | 2048 | 827.066 | 826.973 | 826.993 | +0.00% | -0.01% |
| decode | 32 | 2048 | 35.329 | 51.702 | 35.007 | -32.29% | -0.91% |
| prefill | 64 | 2048 | OOM | 1650.256 | 1655.915 | +0.35% | — |
| decode | 64 | 2048 | OOM | 99.878 | 66.607 | -33.31% | — |
| generation | 1 | 2048 | 513.495 | 563.160 | 461.507 | -18.04% | -10.12% |
| generation | 8 | 2048 | 1345.102 | 1783.306 | 1352.270 | -24.17% | +0.53% |

## Peak allocated GPU memory

GiB. These peaks exclude other processes, CUDA context, reserved-but-unused allocator storage and setup. Both recall modes retain the same raw K + int64 IDs, with 49.9023% less persistent cache than full KV when both run. They materialize V transiently.

| Mode | Batch | Length/context | Full KV peak | Initial recall peak | Fused recall peak |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 2048 | 5.795 | 5.608 | 5.608 |
| decode | 1 | 2048 | 5.708 | 5.528 | 5.513 |
| prefill | 4 | 2048 | 7.231 | 6.482 | 6.482 |
| decode | 4 | 2048 | 6.880 | 6.162 | 6.100 |
| prefill | 8 | 2048 | 9.145 | 7.648 | 7.648 |
| decode | 8 | 2048 | 8.443 | 7.008 | 6.884 |
| prefill | 16 | 2048 | 12.973 | 9.979 | 9.979 |
| decode | 16 | 2048 | 11.570 | 8.700 | 8.450 |
| prefill | 8 | 512 | 6.274 | 5.900 | 5.900 |
| decode | 8 | 512 | 6.100 | 5.740 | 5.709 |
| prefill | 8 | 4096 | 12.973 | 9.979 | 9.979 |
| decode | 8 | 4096 | 11.568 | 8.699 | 8.449 |
| prefill | 8 | 8192 | 20.630 | 14.641 | 14.641 |
| decode | 8 | 8192 | 17.818 | 12.080 | 11.580 |
| prefill | 32 | 2048 | 20.630 | 14.641 | 14.641 |
| decode | 32 | 2048 | 17.824 | 12.083 | 11.583 |
| prefill | 64 | 2048 | OOM | 23.966 | 23.966 |
| decode | 64 | 2048 | OOM | 18.857 | 17.857 |
| generation | 1 | 2048 | 5.795 | 5.608 | 5.608 |
| generation | 8 | 2048 | 9.145 | 7.648 | 7.648 |

## Protocol and evidence

RTX5090, 24 layers, hidden2048, 32 Q/KV heads, intermediate5632, vocab32000, 2.8365B BF16, QK norm/gates disabled, seed1234. No RoPE, GPU-resident unfolded memory tables throughout. Prefill/decode:10 warmups,10 samples ×3 rounds. Generation:2 warmups,5 trajectories ×3 rounds; prefix2048 +128 growing-cache decode steps, predetermined tokens, no sampling. Explicit calls bypass the existing Transformers generate compatibility issue. Prefill constructs cache and last-token logits; decode uses fixed context with rollback/token selection outside timing. Model cache updates and V reconstruction remain timed. No optimized static KV cache.

84 CUDA regression cases pass, including exact reconstruction for BF16/FP16 and head dimensions40/64/128, folded/unfolded tables and hook fallback. A compiled-kernel hash test verifies different history lengths reuse a single binary, avoiding per-step recompilation. Six full-model correctness processes validate exact logits at129 steps for batch1/8 and all hidden/raw-K states at prefix/steps1,2,128; initial/fused recall fingerprints match exactly, including IDs. Training and unsupported paths keep PyTorch autograd. No trained-checkpoint quality claim.

Environment correction: the 6994 MiB external allocation noted in the frozen manifest and baseline register was observed during earlier diagnostics. All 352 recorded before/after performance snapshots in R0002 show 654 MiB for the unrelated process (PID3625831), matching the original baseline background allocation. No unrelated process was stopped. These are snapshots, not continuous monitoring. All timing controls were freshly measured. Full-KV batch64 OOMs and skipped identical repeats remain in the matrix. See environment-audit.json.

Artifacts: manifest.json (all commands/statuses), correctness-audit.json, harness-audit.json, regression-tests.txt, summary.json/CSV and raw B*.json/TXT. Original baseline and microbench diagnostics are in the parent directory. Microbench stage latencies are diagnostic and not additive to model latency.
