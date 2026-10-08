# Fresh upstream / PR performance comparison

Upstream main: `66bd7c6326a6a7badf01e7ef5a14ff967fb8000c`. PR head: `9925991f02502409c0bc838ee0d8c6e70ba878bc`.

RTX 5090, 24 layers, 2.8365B BF16, vocab 32000, hidden 2048, 32 Q/KV heads, intermediate 5632. Fixed random weights (seed 1234) and inputs; no language-quality claim. Inputs start on GPU. Last-token logits and real KV cache. Decode is one fixed-context step; generation includes prefix 2048 and all 128 growing-cache decode steps, excluding sampling. Setup, folding and allocation excluded.

Three independent blocks rotate the three placement orders. Prefill/decode: 10 warmups, 10 samples × 3 rounds/process. Generation: 2 warmups, 5 trajectories × 3 rounds/process. Intervals use paired process means (t, df=2). Timed functions are AST-identical to the prior harness; only fresh-model/source compatibility and untimed accounting were adapted.

## Latency

| Mode | Batch | Context | Main GPU ms | PR folded GPU ms | PR offload ms | Offload−main ms | Offload−folded ms (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| prefill | 1 | 2048 | 29.160 | 29.060 | 29.341 | +0.181 | +0.281 [+0.192, +0.370] |
| decode | 1 | 2048 | 5.477 | 4.612 | 4.478 | -1.000 | -0.134 [-0.328, +0.060] |
| prefill | 4 | 2048 | 106.289 | 106.396 | 106.753 | +0.464 | +0.357 [-0.202, +0.916] |
| decode | 4 | 2048 | 7.300 | 7.210 | 7.253 | -0.047 | +0.044 [+0.024, +0.063] |
| prefill | 8 | 2048 | 208.646 | 206.848 | 208.593 | -0.053 | +1.744 [+0.543, +2.946] |
| decode | 8 | 2048 | 11.036 | 10.932 | 10.934 | -0.102 | +0.002 [-0.028, +0.032] |
| prefill | 16 | 2048 | 421.808 | 417.562 | 420.686 | -1.122 | +3.124 [-4.393, +10.641] |
| decode | 16 | 2048 | 20.379 | 20.286 | 20.165 | -0.214 | -0.120 [-0.204, -0.037] |
| prefill | 8 | 512 | 52.151 | 51.974 | 52.482 | +0.331 | +0.508 [+0.347, +0.668] |
| decode | 8 | 512 | 5.706 | 5.600 | 5.605 | -0.101 | +0.005 [-0.020, +0.030] |
| prefill | 8 | 4096 | 451.487 | 447.421 | 450.454 | -1.033 | +3.034 [-3.130, +9.198] |
| decode | 8 | 4096 | 19.061 | 18.978 | 18.954 | -0.107 | -0.024 [-0.072, +0.023] |
| prefill | 8 | 8192 | 1018.112 | 1010.087 | 1014.939 | -3.172 | +4.852 [-22.693, +32.397] |
| decode | 8 | 8192 | 34.779 | 34.663 | 34.671 | -0.107 | +0.009 [-0.161, +0.179] |
| generation | 1 | 2048 | 699.735 | 588.439 | 594.210 | -105.525 | +5.771 [-3.562, +15.104] |
| generation | 8 | 2048 | 1356.926 | 1351.304 | 1343.710 | -13.216 | -7.594 [-9.746, -5.441] |
| prefill | 32 | 2048 | 845.996 | 836.460 | 842.126 | -3.869 | +5.667 [-9.069, +20.403] |
| decode | 32 | 2048 | 35.450 | 35.348 | 35.636 | +0.186 | +0.288 [+0.147, +0.428] |
| prefill | 64 | 2048 | OOM | OOM | OOM | unavailable | unavailable |
| decode | 64 | 2048 | OOM | OOM | OOM | unavailable | unavailable |

## Peak allocated GPU memory

| Mode | Batch | Context | Main GiB | PR folded GiB | PR offload GiB | Offload pinned host GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill | 1 | 2048 | 5.801 | 5.801 | 2.856 | 2.938 |
| decode | 1 | 2048 | 5.713 | 5.713 | 2.768 | 2.938 |
| prefill | 4 | 2048 | 7.237 | 7.237 | 4.409 | 2.961 |
| decode | 4 | 2048 | 6.886 | 6.886 | 4.058 | 2.961 |
| prefill | 8 | 2048 | 9.151 | 9.151 | 6.448 | 2.992 |
| decode | 8 | 2048 | 8.449 | 8.449 | 5.747 | 2.992 |
| prefill | 16 | 2048 | 12.979 | 12.979 | 10.527 | 3.430 |
| decode | 16 | 2048 | 11.576 | 11.576 | 9.124 | 3.430 |
| prefill | 8 | 512 | 6.280 | 6.280 | 3.390 | 2.945 |
| decode | 8 | 512 | 6.106 | 6.106 | 3.216 | 2.945 |
| prefill | 8 | 4096 | 12.985 | 12.985 | 10.533 | 3.430 |
| decode | 8 | 4096 | 11.580 | 11.580 | 9.128 | 3.430 |
| prefill | 8 | 8192 | 20.653 | 20.653 | 18.701 | 3.930 |
| decode | 8 | 8192 | 17.842 | 17.842 | 15.890 | 3.930 |
| generation | 1 | 2048 | 5.802 | 5.802 | 2.856 | 2.938 |
| generation | 8 | 2048 | 9.151 | 9.151 | 6.449 | 2.992 |
| prefill | 32 | 2048 | 20.636 | 20.636 | 18.686 | 3.933 |
| decode | 32 | 2048 | 17.830 | 17.830 | 15.880 | 3.933 |
| prefill | 64 | 2048 | OOM | OOM | OOM | OOM |
| decode | 64 | 2048 | OOM | OOM | OOM | OOM |

Host storage also includes raw restoration weights and process overhead. GPU allocated peaks are not nvidia-smi usage. Every raw JSON includes before/after environment and memory telemetry. Another ~654 MiB GPU process was present at planning time and is retained in telemetry.

[Comparison figure](comparison.png) · [PDF](comparison.pdf) · [SVG](comparison.svg)

## Audit and interpretation

Full-model main resident, PR folded resident and PR offload outputs match bit-for-bit at batch 1 and 8, prefix 2048, seed 1234: logits at all 129 steps; all 25 hidden states and 24 layer KV tensors at prefix and steps 1,2,128. Small-model regression suite separately covers49 cases.

The main-to-offload comparison includes normalization folding as well as table placement. The folded-GPU comparison isolates the incremental offload path more closely. Negative differences retain their sign; intervals crossing zero are inconclusive, not equivalence. No workload is silently omitted and no OOM timing is zero.

[Plan and execution](manifest.json) · [Audited statistics](summary.json) · [CSV](summary.csv) · [Correctness audit](correctness-audit.json) · [Harness compatibility audit](harness-audit.json)

Reproduce: use `/home/molly/miniconda3/envs/fla-bench/bin/python summarize.py` in this directory. Per-process exact commands, source roots, source SHAs and harness hashes are in the manifest.

An initial upstream process serialized complete valid timing data before an unsupported cleanup call failed. The post-measurement cleanup was guarded and the completed samples were audited and retained without rerunning. See the manifest recovery note and harness revision.
