# No-RoPE recall versus full KV caching

Both paths run frozen source `e5b52be0be5484a6f8d0c4391b158cb71e561759`, with identical seeded weights. Only `memory_recall` changes. RoPE is disabled in both. Tables stay on GPU and are not normalization-folded. This compares the first recall implementation against the corresponding no-RoPE model, not against the prior RoPE/offload study.

## Latency

Milliseconds; lower is better. Positive change means recall is slower. Means use three independent process blocks. Paired two-sided 95% Student-t intervals use those three block means (df=2), not individual samples as independent replicates. Neighboring workloads alternate placement order and block2 reverses the workload list. Audit shows each individual shape retains the same placement order across blocks; order bias is therefore not eliminated. Intervals describe block-to-block variation, not that systematic bias. Intervals crossing zero are inconclusive.

| Mode | Batch | Length/context | Full KV, ms | Recall, ms | Recall change | Recall − full KV, ms [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| prefill | 1 | 2048 | 28.853 | 28.872 | +0.07% | +0.019 [-0.040, +0.079] |
| decode | 1 | 2048 | 4.044 | 4.376 | +8.20% | +0.332 [+0.244, +0.419] |
| prefill | 4 | 2048 | 104.561 | 104.662 | +0.10% | +0.101 [-0.313, +0.515] |
| decode | 4 | 2048 | 7.181 | 7.626 | +6.20% | +0.445 [+0.440, +0.451] |
| prefill | 8 | 2048 | 204.194 | 204.246 | +0.03% | +0.052 [-0.748, +0.853] |
| decode | 8 | 2048 | 10.911 | 13.228 | +21.23% | +2.316 [+2.303, +2.330] |
| prefill | 16 | 2048 | 411.638 | 413.793 | +0.52% | +2.156 [+1.497, +2.815] |
| decode | 16 | 2048 | 20.255 | 28.119 | +38.82% | +7.864 [+7.854, +7.873] |
| prefill | 8 | 512 | 51.182 | 51.196 | +0.03% | +0.014 [-0.049, +0.076] |
| decode | 8 | 512 | 5.578 | 5.480 | -1.77% | -0.099 [-0.127, -0.070] |
| prefill | 8 | 4096 | 441.959 | 443.566 | +0.36% | +1.607 [+0.845, +2.369] |
| decode | 8 | 4096 | 18.937 | 27.135 | +43.30% | +8.199 [+8.182, +8.215] |
| prefill | 8 | 8192 | 995.198 | 1003.169 | +0.80% | +7.971 [+4.412, +11.529] |
| decode | 8 | 8192 | 34.659 | 51.305 | +48.03% | +16.647 [+16.534, +16.760] |
| prefill | 32 | 2048 | 824.616 | 829.693 | +0.62% | +5.077 [+4.334, +5.820] |
| decode | 32 | 2048 | 35.370 | 51.756 | +46.33% | +16.386 [+16.373, +16.399] |
| prefill | 64 | 2048 | OOM | 1649.966 | — | — |
| decode | 64 | 2048 | OOM | 99.950 | — | — |
| generation | 1 | 2048 | 517.292 | 568.604 | +9.93% | +51.312 [+35.073, +67.551] |
| generation | 8 | 2048 | 1346.041 | 1783.098 | +32.47% | +437.057 [+436.189, +437.926] |

## Memory

GiB (2^30 bytes). Cache counts retained tensor storage, including token IDs. Peak GPU is measured allocated memory during the timed workload, not reserved memory or nvidia-smi usage. Persistent cache savings do not imply the same reduction in peak GPU usage.

| Mode | Batch | Length/context | Full KV cache | Recall cache | Full KV GPU peak | Recall GPU peak |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 2048 | 0.375 | 0.188 | 5.795 | 5.608 |
| decode | 1 | 2048 | 0.375 | 0.188 | 5.708 | 5.528 |
| prefill | 4 | 2048 | 1.500 | 0.751 | 7.231 | 6.482 |
| decode | 4 | 2048 | 1.501 | 0.752 | 6.880 | 6.162 |
| prefill | 8 | 2048 | 3.000 | 1.503 | 9.145 | 7.648 |
| decode | 8 | 2048 | 3.001 | 1.504 | 8.443 | 7.008 |
| prefill | 16 | 2048 | 6.000 | 3.006 | 12.973 | 9.979 |
| decode | 16 | 2048 | 6.003 | 3.007 | 11.570 | 8.700 |
| prefill | 8 | 512 | 0.750 | 0.376 | 6.274 | 5.900 |
| decode | 8 | 512 | 0.751 | 0.376 | 6.100 | 5.740 |
| prefill | 8 | 4096 | 6.000 | 3.006 | 12.973 | 9.979 |
| decode | 8 | 4096 | 6.001 | 3.007 | 11.568 | 8.699 |
| prefill | 8 | 8192 | 12.000 | 6.012 | 20.630 | 14.641 |
| decode | 8 | 8192 | 12.001 | 6.012 | 17.818 | 12.080 |
| prefill | 32 | 2048 | 12.000 | 6.012 | 20.630 | 14.641 |
| decode | 32 | 2048 | 12.006 | 6.015 | 17.824 | 12.083 |
| prefill | 64 | 2048 | OOM | 12.023 | OOM | 23.966 |
| decode | 64 | 2048 | OOM | 12.029 | OOM | 18.857 |
| generation | 1 | 2048 | 0.398 | 0.200 | 5.795 | 5.608 |
| generation | 8 | 2048 | 3.188 | 1.597 | 9.145 | 7.648 |

## Scope and validation

RTX 5090; 24 layers, hidden 2048, 32 Q/KV heads, intermediate 5632, vocabulary 32000, 2.8365B BF16; QK norm and gates off; seed1234. Performance prefill/decode: 10 warmups, 10 samples × 3 rounds per process. Generation: 2 warmups, 5 trajectories × 3 rounds; prefix2048 +128 predetermined growing-cache decode steps, no sampling. Each process constructs fresh model weights. Construction and setup are excluded. Inputs start on GPU.

Prefill includes cache creation and last-token logits. Decode measures one fixed-context step with rollback/token selection excluded; the model’s cache update, historical table lookup, RMSNorm and V reconstruction remain timed. Generation uses explicit model calls, not the currently incompatible Transformers generate wrapper.

Full-model correctness passed at batch1/8: bit-exact logits at all129 prefix/decode steps; bit-exact hidden states and cached raw keys at prefix/steps1,2,128; recall stores int64 IDs in place of V. 69 smaller recall/offload regression tests passed before measurement. No trained-checkpoint language quality claim.

Recall retains raw K and per-layer token IDs, then materializes historical M and V each step. At key width2048 and BF16, persistent cache drops from4D to2D+8 bytes/token/layer: 49.9023% saved. QK norm is disabled here; enabling it would also require renormalizing recalled keys. No CPU offload or fused tiled reconstruction is used.

OOMs are preserved; only identical repeats are skipped. See manifest.json for every command/status, correctness-audit.json, harness-audit.json for unchanged timing functions, summary.json/CSV for numerical results, and B*.json/TXT for raw measurements.
