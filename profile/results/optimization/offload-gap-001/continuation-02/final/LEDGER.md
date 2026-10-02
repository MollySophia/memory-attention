# Complete attempt ledger

Campaign `offload-gap-001`. A0000 is the frozen baseline; A0001–A0020 are actual finalized attempts. Only A0011–A0020 belong to this ten-attempt continuation.

| Attempt | Change | Parent | Verdict | Accepted step | Source |
|---|---|---|---|---:|---|
| [A0000](../../A0000/record.json) | restored model with verified measurement harness | — | accepted | 0 | `62942a0387608fe21baaeb2dce9ef3b0947dde4d` |
| [A0001](../../A0001/record.json) | pipeline above 1024 tokens | A0000 | accepted | 1 | `942a2a52d94cf6dbe6e753f7ac291daf8b0699fb` |
| [A0002](../../A0002/record.json) | serial byte copies for tiny CPU lookups | A0001 | within_noise | — | `ac6fbc2cbf534c804bb34227aa3ddee3e08728d9` |
| [A0003](../../A0003/record.json) | two-layer groups in automatic pipeline | A0001 | rejected | — | `986e000b1d973a2a76155ff932aba2d367d03dc0` |
| [A0004](../../A0004/record.json) | caller-thread pipeline producer | A0001 | within_noise | — | `29a0d4ff761e862423e5cd22a8d2a475d9a99a78` |
| [A0005](../../A0005/record.json) | compute-stream copies for tiny bulk transfers | A0001 | within_noise | — | `f33b0e383f2b3d9d7121ea3b92d1e45157aad1de` |
| [A0006](../../A0006/record.json) | refill host slot before consumer release | A0001 | within_noise | — | `b58cdb8640fb2d6a57631e0c26fc6f9a72761f74` |
| [A0007](../../A0007/record.json) | precompute immutable pipeline tensor views | A0001 | rejected | — | `2bcbe8d197ef3244af1f51beb1214e4f3ecf6ffc` |
| [A0008](../../A0008/record.json) | mapped-host GPU gather for pipeline | A0001 | rejected | — | `fb17afb2a97b68ec61ef517622707f68ce33e051` |
| [A0009](../../A0009/record.json) | within-call token dedup for large pipeline inputs | A0001 | rejected | — | `0ac06a0f7ce6c3ce77fd5a878c4a34c991527ec2` |
| [A0010](../../A0010/record.json) | alternate pinned host buffers per GPU slot | A0001 | rejected | — | `b22f8f616ee933ddc4ad2dd09c3eb91e60ef477b` |
| [A0011](../../A0011/record.json) | reuse pinned input-ID staging buffers | A0001 | within_noise | — | `812ac2decafaf2d9919028a7bbd0c740db07ba1a` |
| [A0012](../../A0012/record.json) | blocking producer DMA completion events | A0001 | within_noise | — | `817cc6b5081acdcc7d50935f98e0a4b8c0b71b3a` |
| [A0013](../../A0013/record.json) | mapped-host all-layer gather for small bulk inputs | A0001 | rejected | — | `5d5eb4858f5102b5e7c0c004e359304b13ac38c8` |
| [A0014](../../A0014/record.json) | one-slot automatic pipeline lookahead | A0001 | rejected | — | `b9938ebc46d7a92fc07898a09c72edc547f5f8f3` |
| [A0015](../../A0015/record.json) | fuse compressed-row expansion with memory addition | A0001 | within_noise | — | `0436370506e82e3a32d6ccbc6bb5217bc0e84637` |
| [A0016](../../A0016/record.json) | single-slot automatic pipeline for small prefill only | A0001 | accepted | 2 | `0ce6645eb4c4bc3992258ad0878c72ea3c37a64e` |
| [A0017](../../A0017/record.json) | write-combined pinned pipeline buffers for large inputs | A0016 | within_noise | — | `7eb99060e4ac0125bb93487abc85f139f1e73c8f` |
| [A0018](../../A0018/record.json) | submit embedding before waiting for pipeline input IDs | A0016 | rejected | — | `6089489be39a59a31360b8f89add2042cc0a7c8b` |
| [A0019](../../A0019/record.json) | start pipeline producer before embedding dispatch | A0016 | rejected | — | `f59cdf0c8d7fa32f98b6947c0ba784210f0e7437` |
| [A0020](../../A0020/record.json) | reuse per-layer pipeline coordination tickets | A0016 | within_noise | — | `41869553c1df91e5c8ae4cda16cd4c6b1c0ab44b` |

Full hypotheses, profiles, correctness logs, commands, raw results, uncertainties and reasons for unmeasured stages remain in each linked record. [Machine-readable ledger](attempt-ledger.json) preserves every field; [CSV](attempt-ledger.csv) provides the comparison columns.
