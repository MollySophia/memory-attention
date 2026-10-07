# A0030 阶段结果整理

截至 2026-10-07，按用户要求停留在 A0030，暂停后续优化与新增性能测试。执行源码已从 A0031 单独 revert 回 A0030；实验提交、原始数据和历史失败记录完整保留。

当前保留源码：`a9e5bda62a020e4e1994037732b0dbb7606d5dda`。累计链：**A0000 → A0001 → A0016 → A0023 → A0028 → A0030**，共 5 个保留优化步骤。

## 当前结论

- A0030 全部 16 个场景的平均额外耗时有 **16/16** 在容差内；复用保留验证数据计算的同时置信上界有 **10/16** 在容差内。
- 尚未执行最终独立验收，不能宣布所有场景达标。筛选、保留验证与最终验收是不同证据。
- A0030 的 480 份性能结果和六组完整模型正确性比较通过审计；完整比较逐位一致。模型为 2.8365B BF16、RTX 5090、24 层、hidden 2048，使用随机权重与固定 token，仅支持性能与数值一致性结论。
- 16 场景由七组 batch/context 的 prefill/decode 加两种 generation 构成，不是 batch×length 的笛卡尔积。每个正式场景六组独立进程配对；prefill/decode 每进程 10 warmup、10 samples×3 rounds，generation 2 warmup、5 trajectories×3 rounds。

## 保留优化链

| 步骤 | Attempt | 改动 | 已有收益依据 |
| --- | --- | --- | --- |
| 0 | A0000 | 冻结原始基线 | 累计比较基准 |
| 1 | A0001 | bulk 阈值降到 1024 token | b1/2048 prefill 相对 A0000 约减少 13.80 ms |
| 2 | A0016 | ≤2048 token 使用单 GPU pipeline slot | b1/2048 prefill 相对 A0001 约减少 0.431 ms |
| 3 | A0023 | 1–16 token 从 mapped host 表读取 | 相对 A0016：b1/b8 generation 减少 27.99/30.89 ms；b16 decode 减少 0.101 ms |
| 4 | A0028 | 4096–16384 token 共享 host staging，保持多 GPU slot | 相对 A0023：b8/512 prefill 减少 2.460 ms |
| 5 | A0030 | 每次 forward 只解析一次 offload 元数据 | 相对 A0028：b1 generation 减少 10.477 ms；b8/2048 decode 减少 0.00999 ms |

这些是各自匹配父版本的增量效果，不能跨场景相加，也不能当作同一轮 A0000→A0030 累计测量。详见 [优化链与原始证据](CHAIN.md)。

## A0030 各场景与全 GPU 的差距

单位 ms。差值 = offload − GPU，负数表示 offload 更快；相对开销为每组配对比值的均值。容差 = max(1%×GPU, 0.1 ms)。Generation 是 prefix 2048 + 128 decode 步的总耗时。

“余量上界”是每对 `offload−GPU−max(1%×GPU,0.1 ms)` 的单侧 95% 上界，固定 16 场景 Bonferroni 校正；≤0 表示该诊断上界在容差内，仍不是最终独立验收。

| 场景 | batch | 长度/context | GPU | offload | 差值 | 相对开销 | 容差 | 余量上界 | 诊断区间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| prefill | 1 | 2048 | 29.000 | 29.289 | +0.289 | +1.00% | 0.290 | +0.059 | 未证明 |
| prefill | 4 | 2048 | 106.061 | 106.427 | +0.366 | +0.35% | 1.061 | -0.364 | 容差内 |
| prefill | 8 | 512 | 52.039 | 52.418 | +0.379 | +0.73% | 0.520 | +0.164 | 未证明 |
| prefill | 8 | 2048 | 206.737 | 208.009 | +1.272 | +0.62% | 2.067 | +0.607 | 未证明 |
| prefill | 8 | 4096 | 448.253 | 451.233 | +2.980 | +0.67% | 4.483 | +0.265 | 未证明 |
| prefill | 8 | 8192 | 1013.239 | 1019.732 | +6.493 | +0.64% | 10.132 | +0.359 | 未证明 |
| prefill | 16 | 2048 | 419.000 | 422.025 | +3.025 | +0.72% | 4.190 | +0.641 | 未证明 |
| decode | 1 | 2048 | 4.526 | 4.452 | -0.074 | -1.62% | 0.100 | -0.065 | 容差内 |
| decode | 4 | 2048 | 7.202 | 7.250 | +0.047 | +0.66% | 0.100 | -0.043 | 容差内 |
| decode | 8 | 512 | 5.592 | 5.590 | -0.002 | -0.04% | 0.100 | -0.093 | 容差内 |
| decode | 8 | 2048 | 10.922 | 10.927 | +0.005 | +0.05% | 0.109 | -0.082 | 容差内 |
| decode | 8 | 4096 | 19.011 | 18.947 | -0.064 | -0.33% | 0.190 | -0.029 | 容差内 |
| decode | 8 | 8192 | 34.620 | 34.608 | -0.011 | -0.03% | 0.346 | -0.342 | 容差内 |
| decode | 16 | 2048 | 20.236 | 20.121 | -0.115 | -0.57% | 0.202 | -0.306 | 容差内 |
| generation | 1 | 2048 | 592.604 | 585.560 | -7.043 | -1.18% | 5.926 | -0.535 | 容差内 |
| generation | 8 | 2048 | 1349.742 | 1343.163 | -6.579 | -0.49% | 13.497 | -19.184 | 容差内 |

[完整 CSV](../A0030/gap-diagnostics/gaps.csv) 保留配对区间、原始 manifest 路径和样本数；[完整 JSON](../A0030/gap-diagnostics/gaps.json) 保留统计说明。

## 内存与正确性

CPU 表为 3000 MiB BF16，mapped 路径使整表锁页；这不是额外一份 GPU 常驻表。需同时计入 host staging、GPU 临时缓冲、KV、激活与输出。RSS 包含运行时和模型加载状态，不能等同于表内存。

| 场景 | batch/context | GPU 常驻峰值 GiB | offload 峰值 GiB | 节省 GiB | offload pinned GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| prefill | 1/2048 | 5.801 | 2.856 | 2.945 | 2.938 |
| prefill | 4/2048 | 7.237 | 4.409 | 2.828 | 2.961 |
| prefill | 8/512 | 6.280 | 3.390 | 2.890 | 2.945 |
| prefill | 8/2048 | 9.151 | 6.448 | 2.702 | 2.992 |
| prefill | 8/4096 | 12.985 | 10.533 | 2.452 | 3.430 |
| prefill | 8/8192 | 20.653 | 18.701 | 1.952 | 3.930 |
| prefill | 16/2048 | 12.979 | 10.527 | 2.452 | 3.430 |
| decode | 1/2048 | 5.713 | 2.768 | 2.945 | 2.938 |
| decode | 4/2048 | 6.886 | 4.058 | 2.828 | 2.961 |
| decode | 8/512 | 6.106 | 3.216 | 2.890 | 2.945 |
| decode | 8/2048 | 8.449 | 5.747 | 2.702 | 2.992 |
| decode | 8/4096 | 11.580 | 9.128 | 2.452 | 3.430 |
| decode | 8/8192 | 17.842 | 15.890 | 1.952 | 3.930 |
| decode | 16/2048 | 11.576 | 9.124 | 2.452 | 3.430 |
| generation | 1/2048 | 5.802 | 2.856 | 2.945 | 2.938 |
| generation | 8/2048 | 9.151 | 6.449 | 2.702 | 2.992 |

[A0030 完整审计](../A0030/final-integrity-audit.json)：64 screen + 240 selected confirmation + 144 remaining confirmation + 32 folding reference。六组全模型比较覆盖 b1/2048、b8/2048、b8/512 × seeds 1234/4321；每条轨迹 129 次 logits，且在 prefix、decode 1/2/128 检查全部 hidden/KV。

## A0031 归档结果

A0031 尝试只将第一组查表改为 mapped。64 个筛选进程和 192 个独立确认进程均完成；固定六组、Holm8 结果复算通过。**未保留，执行代码已回退。**

- 确认局部收益：b8/2048 prefill 减少 0.659 ms（95% CI 0.285–1.033）；b16/2048 减少 2.115 ms（1.498–2.732）。
- 阻止保留：b1 generation 增加 6.336 ms（2.108–10.565），gap 增加 5.981 ms；b8/4096 的 resident 对照触发变慢保护判据，不能将其 gap 改善全部归因于 offload 优化。
- 未执行剩余正式矩阵、folding 与全模型最终比较，因为已不符合保留条件。b8 generation 的负向筛选信号仍留在记录中，未作为独立确认回归宣称。
- [Attempt 记录](../A0031/record.json)、[确认审计](../A0031/confirmation-audit.json)、[全部 Attempt 台账](ledger-through-A0031/LEDGER.md)。台账含 A0000 和 31 个尝试；continuation 的局部发现表含 102 个提名比较、13 个确认收益点，这不等于 13 个有效或保留版本。

## 图表索引

下列图均同时提供 PNG、PDF、SVG，邻近 CSV/JSON 为作图源数据。误差条与比较基准写在图注中。

| 图 | 用途 |
| --- | --- |
| [延迟随 batch/context 变化](../A0030/workload-diagnostics/latency_ms-scaling.pdf) | A0030 offload 与 GPU prefill/decode |
| [吞吐随 batch/context 变化](../A0030/workload-diagnostics/tokens_per_second-scaling.pdf) | 同一测量范围下的吞吐 |
| [生成耗时](../A0030/workload-diagnostics/generation.pdf) | prefix +128 步总耗时 |
| [GPU/host 内存](../A0030/workload-diagnostics/memory.pdf) | 显存节省与 host 代价 |
| [绝对差距](../A0030/gap-diagnostics/gaps.pdf)、[相对开销](../A0030/gap-diagnostics/relative-overhead.pdf) | 全 16 场景容差与区间 |
| [尝试历史](history-through-A0031/attempt-history-screen_v1_w3_n5_r1.pdf) | 含失败尝试及保留版本阶梯线；是筛选历史，不是累计正式收益 |
| [局部收益](local-findings-through-A0031/local-gains.pdf)、[阻止保留的回归](local-findings-through-A0031/blocking-guards.pdf) | 保留与未保留方案的发现均留档 |

## 尚未完成的证据

最终独立的 16 场景容差验收、fresh A0000→最终版本累计比较和对应累计性能图尚未完成。本次仅整理既有结果，没有启动新性能测量。重新开始优化或补测需用户后续指示。

## 复现整理

从仓库根目录使用 `/home/molly/miniconda3/envs/fla-bench/bin/python`：

```sh
python profile/results/optimization/offload-gap-001/continuation-03/build_ledger.py --through A0031
python profile/results/optimization/offload-gap-001/continuation-03/plot_local_findings.py --through A0031
python profile/results/optimization/offload-gap-001/continuation-03/plot_workload_scaling.py --attempt A0030
python profile/results/optimization/offload-gap-001/continuation-03/plot_candidate_gaps.py --attempt A0030
python profile/results/optimization/offload-gap-001/plot_history.py --output profile/results/optimization/offload-gap-001/continuation-03/history-through-A0031 --through A0031 --history-only
python profile/results/optimization/offload-gap-001/continuation-03/summarize_a0030.py
```
