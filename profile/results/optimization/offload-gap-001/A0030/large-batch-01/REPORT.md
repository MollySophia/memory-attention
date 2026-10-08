# A0030 大 batch 补测

固定 A0030；24 层、2.8365B BF16，RTX 5090，长度/context 2048。Prefill 包含 KV 构建与 last-token logits；decode 为已构建 KV 后的单步。使用随机权重与固定输入，不评估输出质量。

每点四组独立进程配对，GPU/offload 顺序交替。每进程 10 warmup、10 samples × 3 rounds。差值区间为配对进程均值的双侧 95% Student-t 区间（df=3）；这不是最终 16 场景验收。

| 场景 | batch | GPU ms | offload ms | 差值 ms（95% CI） | 相对开销 |
| --- | ---: | ---: | ---: | --- | ---: |
| prefill | 32 | 834.679 | 838.436 | +3.757 [-5.995, +13.510] | +0.45% |
| prefill | 64 | OOM | OOM | 无法配对 | — |
| decode | 32 | 35.349 | 35.587 | +0.238 [+0.181, +0.296] | +0.67% |
| decode | 64 | OOM | OOM | 无法配对 | — |

## 显存峰值

GPU allocated 峰值的进程均值（GiB）；不等同于 nvidia-smi 或 reserved 显存。

| 场景 | batch | GPU GiB | offload GiB |
| --- | ---: | ---: | ---: |
| prefill | 32 | 20.636 | 18.686 |
| prefill | 64 | OOM | OOM |
| decode | 32 | 17.830 | 15.880 |
| decode | 64 | OOM | OOM |

OOM 保留原始错误；同一配置的后续重复跳过。未改变长度、精度或模型以规避 OOM。当前设备另有约 654 MiB 常驻进程，其状态记录于各次原始 telemetry。

[计划与执行记录](manifest.json) · [完整统计与内存数据](summary.json)

复现：使用 fla-bench Python 执行本目录 `summarize.py`；测量命令逐条保存在 manifest 中。
