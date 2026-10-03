# Supplemental shape survey results

Evaluated all18 nonretained frozen attempts at118 newly selected workload points (472 fresh screening processes, including the predeclared12-process A0011 bulk-coverage addendum), plus eligible historical batch16 screens. 112 potential independent confirmation comparisons were fixed before confirmation; 78 were measured. 3 comparisons pass the local Holm-adjusted dual latency/gap gate; 1 workload cases pass against both the historical parent and current A0016. Historical verdicts and retained A0016 code remain unchanged.

Existing accepted-source evidence already shows a secondary-shape benefit: A0016 versus A0000 at b8/512 prefill reduces offload latency26.9542 ms [25.7087,28.1996] and gap26.9798 ms [25.6126,28.3469] in three retained regression pairs ([source](../../A0016/full-regression-analysis.json)). This is cumulative existing evidence, attributable to the A0001 cutoff path; it is not a new supplemental discovery. Earlier conversational claims of no b8 prefill gain apply only to the primary length2048.

The purpose is to test whether primary-shape screening missed workload-specific improvements. This survey does not prove that every unmeasured shape is equivalent. See [predeclared scope](../PLAN.md), [screen table](screen.csv), [complete confirmation table with intervals](confirmation.csv), and [raw structured source](source.json), [all process latency/throughput/memory rows](measurements.csv), and [explicit coverage map](../coverage.json).

## Interpretation

This survey distinguishes workload-specific gains from an implementation safe to retain across the campaign. A local pass is evidence for that comparator and workload only.

| Candidate / current-A0016 workload | Current → candidate offload ms | Speedup | Offload reduction ms [95% CI] | Gap reduction ms [95% CI] | Holm p | Local gate |
|---|---:|---:|---|---|---:|---|
| A0014 prefill b8/512 | 55.0926 → 52.3164 | 1.0531× | 2.7762 [2.5098, 3.0426] | 2.7604 [2.5246, 2.9961] | 0.02734 | pass |
| A0013 generation b8/2048 | 1373.8907 → 1344.8408 | 1.0216× | 29.0499 [27.4245, 30.6754] | 29.0387 [27.1335, 30.9439] | 0.01302 | pass |
| A0015 prefill b16/2048 | 424.3223 → 422.1576 | 1.0051× | 2.1647 [1.9570, 2.3725] | 2.2280 [1.9650, 2.4909] | 0.04095 | resident slowdown guard failed |

A0014 uses one automatic pipeline slot for all applicable prefill sizes. Its historical b8/2048 decode regression remains: offload reduction −0.040589 ms, 95% CI [−0.057702, −0.023475] ([original decision](../../A0014/record.json)). The current A0016 limits its accepted one-slot policy to at most2048 input tokens; b8/512 has4096 input tokens. A shape-specific policy is a possible follow-up hypothesis, not a validated implementation produced by this survey.

A0013 maps the complete pinned host table for small bulk inputs. Its current-A0016 generation result must be distinguished from its historical-parent result in the full table below; a current-only pass does not satisfy the predeclared both-comparator criterion. It also pins the entire3000MiB table. A0015 illustrates the separate resident guard: a corrected latency/gap p-value alone cannot establish reduced offload overhead when the matched resident reference has a resolved slowdown.

No candidate was integrated, and no new accepted attempt was declared. The original count remains20 attempts with2 retained steps, A0001 and A0016. The survey supplies additional local evidence, including negative and inconclusive results; it does not establish equivalence at unmeasured shapes.

The run completed472 new screening processes and936 independent confirmation processes:66 parent workload comparisons plus12 current-A0016 comparisons, each with3 balanced blocks. The other34 potential current comparisons remain unrun with p=1 in the fixed112-comparison family. These observed counts describe this run; fresh reproductions may nominate a different family under the same fixed rule.

A metadata preparation failure occurred after screening, when an old reference used its recorded original repository path rather than the later frozen-worktree path. Exact source SHA/hash were valid. The path audit was corrected and tested; the [failure record](../preparation-failure-01/workflow-controller.json) and [recovery record](../workflow-controller.json) are preserved. No GPU timing failed or was repeated for this recovery.

See [reproduction commands](REPRODUCE.md), [automated raw-data audit](audit.json), [driver tests](driver-tests.txt), and [manual review](manual-audit.md).

## Every measured confirmation

| Attempt | Workload | Comparator | Offload reduction ms [95% CI] | Gap reduction ms [95% CI] | Holm p | Local gate |
|---|---|---|---|---|---:|---|
| A0002 | decode b4/2048 | A0001 (parent) | 0.0202 [-0.0055, 0.0459] | 0.0197 [0.0115, 0.0279] | 1 | not established |
| A0002 | decode b8/512 | A0001 (parent) | 0.0201 [0.0058, 0.0344] | 0.0385 [0.0176, 0.0594] | 1 | not established |
| A0002 | decode b8/4096 | A0001 (parent) | 0.0302 [-0.0434, 0.1037] | 0.0325 [-0.0366, 0.1016] | 1 | not established |
| A0002 | generation b1/2048 | A0001 (parent) | 45.5351 [21.5815, 69.4888] | 43.2713 [33.8084, 52.7342] | 0.7384 | not established |
| A0002 | generation b8/2048 | A0001 (parent) | 4.9441 [1.6367, 8.2516] | 4.8717 [1.0217, 8.7217] | 1 | not established |
| A0002 | decode b8/512 | A0016 (current) | 0.0887 [-0.1577, 0.3351] | 0.0823 [-0.1435, 0.3082] | 1 | not established |
| A0002 | generation b1/2048 | A0016 (current) | 37.0409 [23.8994, 50.1823] | 34.4893 [4.7453, 64.2333] | 1 | not established |
| A0002 | generation b8/2048 | A0016 (current) | 5.6203 [-1.6375, 12.8781] | 5.4917 [-2.0855, 13.0688] | 1 | not established |
| A0003 | generation b1/2048 | A0001 (parent) | 7.4358 [-15.8057, 30.6774] | 6.6741 [-19.3550, 32.7031] | 1 | not established |
| A0004 | prefill b4/2048 | A0001 (parent) | 1.5834 [-0.0060, 3.1728] | 1.6462 [-0.1097, 3.4022] | 1 | not established |
| A0004 | prefill b8/512 | A0001 (parent) | 2.1953 [1.0677, 3.3229] | 2.2042 [1.0913, 3.3170] | 0.7116 | not established |
| A0004 | prefill b8/4096 | A0001 (parent) | -1.0420 [-2.8610, 0.7769] | -1.8705 [-3.5272, -0.2137] | 1 | not established |
| A0004 | prefill b8/8192 | A0001 (parent) | 0.6506 [-0.6911, 1.9924] | 3.8880 [-2.9597, 10.7357] | 1 | not established |
| A0004 | prefill b8/512 | A0016 (current) | 2.3245 [1.6765, 2.9724] | 2.3720 [1.2953, 3.4488] | 0.5693 | not established |
| A0005 | decode b4/2048 | A0001 (parent) | 0.0435 [-0.1407, 0.2277] | 0.0408 [-0.1489, 0.2306] | 1 | not established |
| A0005 | decode b8/8192 | A0001 (parent) | 0.0175 [-0.0643, 0.0994] | -0.0007 [-0.1103, 0.1088] | 1 | not established |
| A0005 | generation b1/2048 | A0001 (parent) | 0.4433 [-69.5377, 70.4242] | 6.0375 [-54.7027, 66.7777] | 1 | not established |
| A0005 | generation b8/2048 | A0001 (parent) | 1.8631 [-1.1028, 4.8289] | 2.0701 [-1.1854, 5.3256] | 1 | not established |
| A0006 | generation b1/2048 | A0001 (parent) | 20.8550 [-35.0298, 76.7399] | 27.2812 [-5.5048, 60.0672] | 1 | not established |
| A0006 | prefill b4/2048 | A0001 (parent) | 0.1814 [-1.7403, 2.1032] | 0.1641 [-1.7287, 2.0569] | 1 | not established |
| A0007 | decode b8/8192 | A0001 (parent) | 0.0451 [-0.0500, 0.1403] | 0.0228 [-0.0516, 0.0972] | 1 | not established |
| A0007 | prefill b4/2048 | A0001 (parent) | 0.3389 [-3.8584, 4.5362] | 0.3424 [-4.1649, 4.8497] | 1 | not established |
| A0007 | prefill b8/4096 | A0001 (parent) | -1.3689 [-4.3769, 1.6391] | -1.6262 [-5.9234, 2.6710] | 1 | not established |
| A0008 | generation b1/2048 | A0001 (parent) | 14.9299 [0.0929, 29.7669] | 19.2401 [2.8248, 35.6554] | 1 | not established |
| A0008 | prefill b4/2048 | A0001 (parent) | 0.8286 [-1.1855, 2.8428] | 0.8850 [-1.3385, 3.1085] | 1 | not established |
| A0008 | prefill b8/512 | A0001 (parent) | 2.1597 [1.6492, 2.6702] | 2.1804 [1.7793, 2.5815] | 0.1622 | not established |
| A0008 | generation b1/2048 | A0016 (current) | 5.3456 [-46.1229, 56.8142] | 11.0721 [-50.7068, 72.8510] | 1 | not established |
| A0008 | prefill b8/512 | A0016 (current) | 2.1328 [1.4200, 2.8457] | 2.1634 [1.3335, 2.9933] | 0.4124 | not established |
| A0010 | decode b8/8192 | A0001 (parent) | -0.0237 [-0.0930, 0.0455] | -0.0287 [-0.0921, 0.0347] | 1 | not established |
| A0010 | generation b1/2048 | A0001 (parent) | -20.9477 [-65.1355, 23.2402] | -21.6127 [-74.6863, 31.4608] | 1 | not established |
| A0010 | prefill b4/2048 | A0001 (parent) | -0.2162 [-1.2961, 0.8637] | -0.2197 [-1.3304, 0.8910] | 1 | not established |
| A0011 | decode b8/8192 | A0001 (parent) | 0.0327 [0.0120, 0.0533] | 0.0454 [0.0182, 0.0725] | 1 | not established |
| A0011 | prefill b4/2048 | A0001 (parent) | -0.3002 [-2.0183, 1.4179] | -0.3966 [-2.7356, 1.9425] | 1 | not established |
| A0011 | prefill b8/4096 | A0001 (parent) | -1.4608 [-3.7885, 0.8670] | -2.3490 [-4.1508, -0.5472] | 1 | not established |
| A0011 | prefill b8/8192 | A0001 (parent) | -2.3395 [-16.1954, 11.5164] | 0.1335 [-23.4536, 23.7205] | 1 | not established |
| A0011 | decode b4/2048 | A0001 (parent) | -0.0029 [-0.0877, 0.0818] | -0.0097 [-0.0907, 0.0712] | 1 | not established |
| A0011 | decode b8/4096 | A0001 (parent) | 0.0246 [-0.0402, 0.0895] | 0.0246 [-0.0513, 0.1005] | 1 | not established |
| A0011 | decode b16/2048 | A0001 (parent) | 0.0121 [-0.0878, 0.1121] | -0.0014 [-0.1091, 0.1064] | 1 | not established |
| A0011 | decode b8/8192 | A0016 (current) | -0.0286 [-0.1666, 0.1095] | -0.0340 [-0.1834, 0.1154] | 1 | not established |
| A0012 | generation b1/2048 | A0001 (parent) | 5.7687 [-31.5337, 43.0711] | 7.7170 [-18.4684, 33.9024] | 1 | not established |
| A0012 | prefill b8/8192 | A0001 (parent) | 0.5711 [-1.0267, 2.1688] | 6.1888 [-6.4712, 18.8488] | 1 | not established |
| A0013 | generation b8/2048 | A0001 (parent) | 30.1592 [23.8249, 36.4935] | 30.2061 [22.8560, 37.5563] | 0.1703 | not established |
| A0013 | decode b16/2048 | A0001 (parent) | 0.1429 [0.0998, 0.1859] | 0.1462 [0.0635, 0.2290] | 0.8432 | not established |
| A0013 | generation b8/2048 | A0016 (current) | 29.0499 [27.4245, 30.6754] | 29.0387 [27.1335, 30.9439] | 0.01302 | pass |
| A0013 | decode b16/2048 | A0016 (current) | 0.1405 [0.0003, 0.2807] | 0.1288 [-0.0007, 0.2583] | 1 | not established |
| A0014 | prefill b4/2048 | A0001 (parent) | 3.1493 [1.9658, 4.3327] | 3.1489 [1.9681, 4.3296] | 0.3997 | not established |
| A0014 | prefill b8/512 | A0001 (parent) | 2.3561 [2.1526, 2.5596] | 2.3619 [2.1857, 2.5382] | 0.02235 | pass |
| A0014 | prefill b8/4096 | A0001 (parent) | -0.1707 [-3.1519, 2.8106] | -1.0135 [-3.4072, 1.3803] | 1 | not established |
| A0014 | prefill b8/8192 | A0001 (parent) | 1.6700 [-3.9504, 7.2904] | 4.7311 [-7.8902, 17.3524] | 1 | not established |
| A0014 | prefill b16/2048 | A0001 (parent) | 2.0809 [-0.0136, 4.1753] | 2.1468 [-0.5332, 4.8269] | 1 | not established |
| A0014 | prefill b4/2048 | A0016 (current) | 2.6542 [2.3346, 2.9738] | 2.4993 [0.8476, 4.1509] | 1 | not established |
| A0014 | prefill b8/512 | A0016 (current) | 2.7762 [2.5098, 3.0426] | 2.7604 [2.5246, 2.9961] | 0.02734 | pass |
| A0015 | generation b8/2048 | A0001 (parent) | 1.2568 [-0.8200, 3.3336] | 1.2114 [-1.7564, 4.1792] | 1 | not established |
| A0015 | prefill b4/2048 | A0001 (parent) | -0.3152 [-2.4278, 1.7974] | -0.2804 [-2.9564, 2.3956] | 1 | not established |
| A0015 | prefill b8/4096 | A0001 (parent) | 1.1591 [-0.4933, 2.8114] | 0.5444 [-0.1220, 1.2107] | 1 | not established |
| A0015 | prefill b8/8192 | A0001 (parent) | 7.2948 [1.5664, 13.0231] | 9.7009 [-1.9290, 21.3307] | 1 | not established |
| A0015 | prefill b16/2048 | A0001 (parent) | 3.0587 [1.1542, 4.9633] | 3.1943 [0.5964, 5.7922] | 1 | not established |
| A0015 | prefill b16/2048 | A0016 (current) | 2.1647 [1.9570, 2.3725] | 2.2280 [1.9650, 2.4909] | 0.04095 | not established |
| A0017 | decode b8/8192 | A0016 (parent) | 0.0135 [-0.0808, 0.1077] | 0.0189 [-0.0848, 0.1226] | 1 | not established |
| A0017 | generation b8/2048 | A0016 (parent) | -1.5923 [-7.9147, 4.7301] | -1.1033 [-7.5441, 5.3376] | 1 | not established |
| A0017 | prefill b4/2048 | A0016 (parent) | 2.1387 [-0.8459, 5.1233] | 2.1704 [-0.8623, 5.2030] | 1 | not established |
| A0017 | prefill b8/4096 | A0016 (parent) | 1.7896 [0.9251, 2.6542] | 2.7299 [1.4031, 4.0567] | 0.6448 | not established |
| A0017 | prefill b8/8192 | A0016 (parent) | -0.3004 [-5.7668, 5.1660] | 0.4421 [-10.5700, 11.4541] | 1 | not established |
| A0017 | prefill b16/2048 | A0016 (parent) | 1.2012 [0.6018, 1.8006] | 0.5922 [-1.6036, 2.7879] | 1 | not established |
| A0018 | decode b8/8192 | A0016 (parent) | 0.0610 [-0.1033, 0.2253] | 0.0525 [-0.1495, 0.2545] | 1 | not established |
| A0018 | generation b1/2048 | A0016 (parent) | -12.9853 [-60.3765, 34.4060] | -14.3000 [-51.8675, 23.2676] | 1 | not established |
| A0018 | generation b8/2048 | A0016 (parent) | -0.2662 [-3.0466, 2.5142] | 0.0587 [-1.6954, 1.8128] | 1 | not established |
| A0018 | prefill b4/2048 | A0016 (parent) | -1.1913 [-3.3771, 0.9945] | -1.2239 [-3.3935, 0.9456] | 1 | not established |
| A0018 | prefill b8/8192 | A0016 (parent) | -5.1068 [-17.0077, 6.7941] | -4.3792 [-23.1153, 14.3568] | 1 | not established |
| A0018 | prefill b16/2048 | A0016 (parent) | -0.2701 [-1.1763, 0.6360] | -1.1118 [-3.8482, 1.6246] | 1 | not established |
| A0019 | prefill b8/512 | A0016 (parent) | -0.2973 [-2.1164, 1.5218] | -0.3138 [-2.1623, 1.5347] | 1 | not established |
| A0019 | prefill b8/4096 | A0016 (parent) | 0.1887 [-1.0957, 1.4730] | 1.5947 [-2.9793, 6.1687] | 1 | not established |
| A0020 | decode b8/8192 | A0016 (parent) | -0.0050 [-0.1631, 0.1532] | 0.0024 [-0.1343, 0.1391] | 1 | not established |
| A0020 | generation b1/2048 | A0016 (parent) | -15.8101 [-129.3754, 97.7552] | -18.0100 [-114.7661, 78.7460] | 1 | not established |
| A0020 | prefill b4/2048 | A0016 (parent) | -0.7350 [-1.6392, 0.1693] | -0.7091 [-1.7314, 0.3132] | 1 | not established |
| A0020 | prefill b8/512 | A0016 (parent) | -0.2109 [-1.3510, 0.9293] | -0.1914 [-1.3730, 0.9901] | 1 | not established |
| A0020 | prefill b8/8192 | A0016 (parent) | -5.2894 [-18.5653, 7.9865] | -5.5851 [-26.6484, 15.4783] | 1 | not established |
| A0020 | prefill b16/2048 | A0016 (parent) | -0.0168 [-0.5632, 0.5295] | -0.7321 [-3.3667, 1.9025] | 1 | not established |

Memory tradeoffs are retained in the process table: A0008/A0013 pin the entire3000MiB table; A0010 doubles pipeline host-buffer capacity; A0011/A0018 add pinned ID storage; A0009/A0015 retain inverse-index storage. A0017 changes host allocation type at the same byte capacity. Host process high-water marks include model loading. GPU savings gates use peak allocated bytes; reserved memory is reported separately. These local timing checks do not substitute for full applicable correctness and integration/regression validation.

All screens and every unfavorable confirmation remain available. Current-comparison slots skipped by the predeclared parent gate retain p=1 in the fixed multiplicity family; they do not shrink the correction denominator. Passing a parent comparison alone does not show superiority to current A0016. Passing a local comparison does not resolve regressions elsewhere or authorize retention. In particular, A0014’s prior primary regression remains part of its evidence.

Intervals use only three independent balanced process differences; within-process rounds are not independent replicates. The t model assumes sufficiently symmetric differences, not verified with n=3. Holm controls multiplicity conditional on valid p-values, but does not repair distribution or measurement assumptions. Inconclusive results are not equivalence claims. Seeded random weights support performance and equivalence experiments, not language quality.

Prefill/decode screen3 warmups/5 samples. Formal prefill/decode10 warmups/3×10 samples. Generation2 warmups/3×5 trajectories at prefix2048+128 predetermined steps, excluding sampling. Every source is frozen and both placements measured; GPU/host buffers, pinned tables, inverse maps and cached KV are audited from raw telemetry. All GPU jobs ran sequentially.

## Figures

[Screen offload reduction](offload_reduction_ms.png), [screen gap reduction](gap_reduction_ms.png). Colors are symmetric and saturated for readability; numeric labels retain full signed values. Blank cells are unmeasured, never zero.
- [A0002 all confirmation intervals](A0002-confirmation.png)
- [A0003 all confirmation intervals](A0003-confirmation.png)
- [A0004 all confirmation intervals](A0004-confirmation.png)
- [A0005 all confirmation intervals](A0005-confirmation.png)
- [A0006 all confirmation intervals](A0006-confirmation.png)
- [A0007 all confirmation intervals](A0007-confirmation.png)
- [A0008 all confirmation intervals](A0008-confirmation.png)
- [A0010 all confirmation intervals](A0010-confirmation.png)
- [A0011 all confirmation intervals](A0011-confirmation.png)
- [A0012 all confirmation intervals](A0012-confirmation.png)
- [A0013 all confirmation intervals](A0013-confirmation.png)
- [A0014 all confirmation intervals](A0014-confirmation.png)
- [A0015 all confirmation intervals](A0015-confirmation.png)
- [A0017 all confirmation intervals](A0017-confirmation.png)
- [A0018 all confirmation intervals](A0018-confirmation.png)
- [A0019 all confirmation intervals](A0019-confirmation.png)
- [A0020 all confirmation intervals](A0020-confirmation.png)
