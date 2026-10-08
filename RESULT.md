# A0030 result branch

This branch (`exp-0/result`) contains the verified A0030 implementation, organized
as five retained changes on the frozen A0000 baseline (`62942a0`). Model code,
regression tests, and benchmark code match A0030
(`a9e5bda62a020e4e1994037732b0dbb7606d5dda`). Rejected candidates and their
reverts remain on `exp-0/attempts/offload-gap-001`, together with full raw evidence.
This branch does not resume optimization or claim final acceptance.

## Retained changes

| Attempt | Change |
| --- | --- |
| A0001 | Lower the automatic bulk cutoff to 1024 tokens |
| A0016 | Use one pipeline GPU slot through 2048 input tokens |
| A0023 | Read mapped pinned host tables for inputs of 1–16 tokens |
| A0028 | Share host staging for 4096–16384 input tokens, retaining GPU lookahead |
| A0030 | Resolve offload dispatch metadata once per forward |

## Results and evidence

Measurements use a 24-layer, 2.8365B BF16 model on RTX 5090 with seeded random
weights. They establish performance and numerical behavior, not language quality.
The original 16 workloads have mean offload overheads within
`max(1% of matched GPU latency, 0.1 ms)`; only 10/16 diagnostic simultaneous
confidence bounds are within that tolerance. Final independent acceptance and a
fresh full A0000-to-A0030 cumulative comparison remain outstanding.

Supplemental batch-32 measurements at context 2048 found prefill mean overhead
+0.45% (paired 95% gap interval -5.995 to +13.510 ms) and decode +0.67%
(+0.181 to +0.296 ms). Batch 64 ran out of memory for both placements, including
decode prefix construction. These extra shapes do not alter the fixed matrix.

- [Complete A0030 report and figure index](https://github.com/MollySophia/memory-attention/blob/63eae96dbd5fdd96a1a768669ee77ba2cd89ab36/profile/results/optimization/offload-gap-001/continuation-03/A0030-RESULTS.md)
- [Retained chain and validation evidence](https://github.com/MollySophia/memory-attention/blob/63eae96dbd5fdd96a1a768669ee77ba2cd89ab36/profile/results/optimization/offload-gap-001/continuation-03/CHAIN.md)
- [All attempts, including rejected A0031](https://github.com/MollySophia/memory-attention/blob/63eae96dbd5fdd96a1a768669ee77ba2cd89ab36/profile/results/optimization/offload-gap-001/continuation-03/ledger-through-A0031/LEDGER.md)
- [Large-batch supplement](https://github.com/MollySophia/memory-attention/blob/63eae96dbd5fdd96a1a768669ee77ba2cd89ab36/profile/results/optimization/offload-gap-001/A0030/large-batch-01/REPORT.md)

Links are pinned to evidence commit `63eae96dbd5fdd96a1a768669ee77ba2cd89ab36`. Historical A0000 files inherited
from the baseline are not the current results; use the links above.

## Validation and local usage

The result branch was checked against the frozen A0030 source, including the
complete `fla/` and `tests/` trees and benchmark Python files. Numerical and
performance evidence belongs to that identical source; new result-branch commits
change the Git identity, not the measured implementation.

Use the existing environment `/home/molly/miniconda3/envs/fla-bench/bin/python`:

```sh
python -m pytest tests/test_memory_offload_regressions.py tests/test_decode_benchmark.py tests/test_sampling_plans.py -q
python profile/test_memory_offload.py
```

No new performance measurements were run while organizing this branch. Source
identity and verification results are recorded in `RESULT-VALIDATION.json`.
