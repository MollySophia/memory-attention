# Memory Attention：paper-001 最终结果

保留 A0002（accepted_step=1）：多样本 batch 使用按128位置扩容的 KV 缓冲区，避免每一步重新拼接完整历史；batch1 保留原拼接路径。仅在 inference mode 且无滑动窗口时启用。训练、普通 no_grad、滑动窗口及 legacy 路径保留原行为。预填充输出和模型计算不变。

冻结基线：`d949640ebf2f13f56021bd08c5c9f10e65d571c3`；保留实现：`7647da8c431fa4e7bf5800d7556ef2f8f6c171ed`。后续提交为结果、测试和文档。模型为2,836,499,968参数、BF16、RTX5090，主测 batch8/context2048，最后一个 token 的 logits 和真实 KV cache。

## 独立配对主测

每项三个独立交替基线／候选进程对，每进程10 warmup、10样本×3轮。表中延迟为三个进程估计的几何均值；每进程估计是各轮均值的中位数。加速比区间为配对 log 比值的95% Student-t区间（df2）。

| 模式 | 放置 | 基线 ms | A0002 ms | 加速比及95%区间 |
|---|---|---:|---:|---|
| Prefill | CPU offload | 210.103 | 210.025 | 1.0004 [0.9898,1.0111]，within_noise |
| Decode | CPU offload | 11.029 | 5.541 | **1.9905 [1.8969,2.0886]** |
| Prefill | GPU resident | 206.877 | 206.819 | 1.0003 [0.9916,1.0091]，within_noise |
| Decode | GPU resident | 10.933 | 5.326 | 2.0527 [2.0424,2.0631] |

完整配对结果见 [confirmation-summary.json](A0002/confirmation-summary.json)。Prefill未确认提升，也未确认回退。CUDA-event诊断显示 prefill 主要耗时在未被本次 KV 复用改变的 MLP；事件跨度包含CPU发射间隙和测量开销，不能当作纯内核耗时。

## 完整验证与限制

完整矩阵覆盖7种 batch/length组合×prefill/decode×3种放置，加上batch1/8的128步生成，共48点/实现、96条结果、2700样本。19条经源码、协议、采样及环境审计后复用，其余77条新测；无OOM、失败或缺失。原始结果和复用来源保存在 [R06 manifest](A0002/R06-full-validation/manifest.json)。

全矩阵中4个单次比较的变慢信号另用24个新进程、720样本复核，全部 within_noise，初始点未合并进新分析。详见 [复核报告](A0002/regression-followup-summary.json)。这表示未确认回退，不是等价性证明。batch1 offload decode的早期独立配对区间尤其宽：[0.7224,1.4155]，必须保留该不确定性。

batch8的生成轨迹（2048-token prefill +128步固定token decode，不含sampling）offload耗时1375.86→940.08ms，单次配对估计1.464×。这不是独立确认的加速比，也不是服务端端到端延迟。

所有16组候选 offload/resident工作负载对均保留GPU peak allocated节省，范围1.952–2.828GiB。CPU表仍为3000MiB，pinned传输缓冲区、GPU staging、KV backing storage和host RSS均有记录。GPU reserved包含allocator缓存，部分大工作负载offload reserved并不更低；host RSS也不因此更低。详见 [内存审计](A0002/R06-memory-final.json)。

41项既有候选正确性测试通过；新增4项128步、只在轨迹末尾同步的异步生成测试通过；主形状296项 logits/hidden/KV比对精确相等；独立offload检查最大差值为0。参考cache从冻结Git对象加载，避免两条路径共享新实现造成假通过。随机权重仅支持数值等价与性能结论，不支持文本质量结论。

## 历史与交付物

A0001为所有batch启用复用，虽然主测约2×，但独立确认batch1 resident回退，已拒绝并回退实现；失败、污染的导入测量、被中止矩阵和候选patch全部保留。A0002针对该证据保留batch1拼接。原A0000的30/30/5历史数据未改写。

- [实验账本](../../OPTIMIZATION_LOG.md)
- [复现说明和命令](REPRODUCE.md)
- [延迟缩放图](final-figures/scaling/median_ms.png)
- [吞吐缩放图](final-figures/scaling/tokens_per_second.png)
- [GPU allocated](final-figures/scaling/gpu_peak_allocated_gib.png)、[GPU reserved](final-figures/scaling/gpu_peak_reserved_gib.png)、[host RSS](final-figures/scaling/host_rss_gib.png)
- [生成耗时与显存](final-figures/scaling/growing_generation.png)
- [包含失败的优化历史](final-figures/history/attempt_history.png)
- [累计接受步骤](final-figures/history/accepted_steps.png)

每张图均有同名PDF/SVG；各目录含CSV/JSON源数据。缩放图误差条为进程内轮均值范围，不是置信区间。历史图baseline采用相同正式采样计划下的冻结基线复测，并在JSON注明来源；历史延迟误差条为三个进程范围。累计图只含accepted步骤并使用匹配基线的配对区间，不拼接不同模型的单项最优值。
