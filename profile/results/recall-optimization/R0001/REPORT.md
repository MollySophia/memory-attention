# R0001: superseded warm-kernel diagnostic

This study was stopped at a process boundary after code review found `ROWS` was a Triton compile-time constant. The growing history could trigger per-length compilation, which warmup-based timings conceal. Completed measurements and correctness checks are retained, but this is not an accepted complete performance result. See ../R0002/ for the runtime-length fix and final three-way comparison.

No completed sample was discarded. The final child finished successfully before its controller was terminated. Pending jobs were not executed. Source candidate: `00f111f1`.
