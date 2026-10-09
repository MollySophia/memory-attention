# Original main benchmark offload vs PR offload

Direct execution of unmodified `main/profile/bmk.py` at `66bd7c6326a6a7badf01e7ef5a14ff967fb8000c`; PR reference is the completed `pr-offload-001` study at `9925991f02502409c0bc838ee0d8c6e70ba878bc`. No old offloader was transplanted into the PR model.

## Reading the table

Milliseconds, lower is better. Negative PR change means PR is faster. Each value is the mean of three independent process means; ± is the sample SD of those process means, not a confidence interval. The two studies were collected separately and are not paired. Small differences should not be read as confirmed improvements.

| Mode | Batch | Length/context | Original bmk offload (ms ± SD) | PR offload (ms ± SD) | PR change |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 2048 | 29.896 ± 0.072 | 29.341 ± 0.017 | -1.86% |
| decode | 1 | 2048 | 6.033 ± 0.329 | 4.478 ± 0.039 | -25.78% |
| prefill | 4 | 2048 | 112.661 ± 0.544 | 106.753 ± 0.354 | -5.24% |
| decode | 4 | 2048 | 6.136 ± 0.342 | 7.253 ± 0.008 | +18.20% |
| prefill | 8 | 2048 | 212.915 ± 0.301 | 208.593 ± 0.176 | -2.03% |
| decode | 8 | 2048 | 5.982 ± 0.302 | 10.934 ± 0.011 | +82.79% |
| prefill | 16 | 2048 | 427.605 ± 0.446 | 420.686 ± 1.756 | -1.62% |
| decode | 16 | 2048 | 8.538 ± 0.010 | 20.165 ± 0.029 | +136.20% |
| prefill | 8 | 512 | 55.858 ± 0.452 | 52.482 ± 0.117 | -6.04% |
| decode | 8 | 512 | 5.920 ± 0.378 | 5.605 ± 0.010 | -5.31% |
| prefill | 8 | 4096 | 457.069 ± 1.791 | 450.454 ± 1.909 | -1.45% |
| decode | 8 | 4096 | 7.402 ± 0.085 | 18.954 ± 0.007 | +156.07% |
| prefill | 8 | 8192 | 1025.115 ± 2.410 | 1014.939 ± 6.400 | -0.99% |
| decode | 8 | 8192 | 11.160 ± 0.024 | 34.671 ± 0.047 | +210.68% |
| prefill | 32 | 2048 | 853.286 ± 1.923 | 842.126 ± 3.671 | -1.31% |
| decode | 32 | 2048 | 11.993 ± 0.043 | 35.636 ± 0.058 | +197.13% |
| prefill | 64 | 2048 | OOM | OOM | — |
| decode | 64 | 2048 | OOM | OOM | — |

## Aligned settings

RTX 5090, BF16, 24 layers, hidden size 2048, 32 Q/KV heads, intermediate size 5632, vocabulary 32000, untied embedding/head, RMSNorm epsilon 1e-6, RoPE theta 10000, no QK normalization or gate, fused SwiGLU, last-token logits, seed 1234, 16 CPU threads, group size 1, prefetch depth 4. Per process: 10 warmups, 3 rounds of 10 repeats, synchronized wall-clock latency. Original bmk policy remains prefill pipeline / decode bulk. All 18 prefill/decode shapes are planned; identical OOM repeats are skipped.

## Preserved differences

- Original bmk runs its own `AttentionStack`, preallocates a static KV cache and overwrites slots. PR uses `MemoryForCausalLM` and its dynamic KV cache. Prefill in PR includes fresh cache creation; decode includes the model’s cache update.
- Original bmk has both CPU and GPU IDs available before timing and cycles a 16-input pool. PR starts with GPU IDs and includes any needed staging; its prefill repeats one input.
- Parameter accounting differs by 1,536 folded memory-normalization affine values (24 × 64): original bmk reports 2,836,498,432; PR reports 2,836,499,968. This is not a hidden-size/layer-count mismatch.
- Model dimensions match, but synthetic weight and input initialization differ. Outputs are not asserted equal across these two independent model implementations.
- Original bmk times each whole round with synchronization after every call; PR records each call separately. Both include embedding, transformer blocks, final normalization and last-token logits.
- Original bmk has no generation mode. The earlier PR generation results have no original-bmk counterpart.
- Original bmk reports parameter/staging sizes, not a comparable measured GPU allocation peak. No peak-memory comparison is inferred here.
- Results describe the complete implementations and must not be attributed solely to offload optimization.

## Evidence

`manifest.json` records exact commands, frozen-source hashes, job states and hardware. Raw JSON/TXT files preserve each process. `summary.json` and `summary.csv` contain the numerical comparison. PR raw data remain in `../pr-offload-001/`.
