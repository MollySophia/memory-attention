# Recall baseline v1

Frozen code: `e5b52be0be5484a6f8d0c4391b158cb71e561759`, tag `recall-baseline-v1`, branch `exp-0/recall-no-rope`.
Optimization branch: `exp-0/recall`.
Frozen evidence commit: `24a7a7d2d9cfd66ff470c597c22fca2d0ad4775a`.

[Original full report](../recall-no-rope-001/REPORT.md) · [CSV](../recall-no-rope-001/summary.csv) · [Artifact hashes](BASELINE.json)

Both paths disable RoPE, use unfolded GPU tables and seeded 2.8365B BF16 weights. Values below are mean milliseconds over three independent processes. Context 2048:

| Batch | Full KV decode | Initial recall decode | Recall overhead | Full KV / recall peak GiB |
|---:|---:|---:|---:|---:|
| 1 | 4.044 | 4.376 | +8.20% | 5.708 / 5.528 |
| 4 | 7.181 | 7.626 | +6.20% | 6.880 / 6.162 |
| 8 | 10.911 | 13.228 | +21.23% | 8.443 / 7.008 |
| 16 | 20.255 | 28.119 | +38.82% | 11.570 / 8.700 |
| 32 | 35.370 | 51.756 | +46.33% | 17.824 / 12.083 |

Persistent cache saving: 49.9023%. Batch8/context8192 decode: 34.659 → 51.305 ms (+48.03%). Prefill overhead below 1%. Prefix2048 +128 decode steps: batch1 517.292 → 568.604 ms; batch8 1346.041 → 1783.098 ms. Full KV batch64 OOM; recall prefill1649.966 ms, decode99.950 ms. Full 20-workload matrix and correctness evidence are retained above.

## Rules for subsequent comparisons

Preserve this baseline. Report comparisons to both initial recall and no recall; do not substitute offload results or RoPE-enabled results. Keep the same cache semantics and memory accounting. Cover the entire matrix, retain regressions/OOMs, and record timings separately from correctness checks. A change in hardware/software/background GPU use requires fresh controls. The original per-shape placement order was fixed; new repeated comparisons should rotate all placements across ordering positions.

The 2026-10-12 GPU has an unrelated process holding about 6994 MiB, versus about 654 MiB during baseline collection. It must not be terminated or its memory attributed to recall. Current-run OOMs are not directly comparable to the frozen baseline capacity result.


## First optimization result

[Final R0002 comparison](R0002/REPORT.md) compares frozen baseline with `b7c33a6e` on `exp-0/recall`, using fresh full-KV and initial-recall controls, all20 workloads and three balanced-order blocks. The initial source and original evidence above are unchanged. [Environment audit](R0002/environment-audit.json) corrects the diagnostic-stage external allocation note: all recorded R0002 performance snapshots show654 MiB, not6994 MiB. R0001 is an incomplete, superseded diagnostic and must not be substituted for R0002.
