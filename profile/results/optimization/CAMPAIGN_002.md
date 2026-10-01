# Continued optimization from accepted A0002

User request: continue GOAL.md iterations toward5–10 effective optimizations.
A0000 remains the frozen historical baseline; A0002 is the accepted incumbent.
Use fresh incumbent/candidate pairs to isolate each incremental effect, plus
matched A0000/final evidence for cumulative claims. Failure/noise does not count
as an accepted effective step. IDs continue at A0003. Preserve all raw results.

Initial hypotheses (roadmap, not registered implementations or claimed gains):

1. Delay memory-transfer acquisition until Q/K normalization and rotary work is queued, increasing overlap without changing arithmetic.
2. Coalesce K/V buffer append copies into one launch while preserving exact bytes, capacity and cache ownership.
3. Combine Q/K rotary launches with exact existing arithmetic, including GQA and offset handling.
4. Reduce CPU/stream coordination in small-input table transfer without caching lookups.
5. Improve pipeline gather layout/grouping where measured transfer stalls warrant it.
6. Remove repeated decode-only shape/view/cache dispatch overhead, preserving public cache semantics.
7. Specialize native RMSNorm inference while preserving BF16 rounding; define tolerances before any arithmetic change.
8. Improve small-token projection/MLP execution based on post-A0002 diagnostics; separate arithmetic-changing candidates and ablations.

Profile or inspect concrete costs before each registration. Screen3/5/1; confirm
promising primary gains with at least3 fresh alternating pairs. Full validation
only for retention candidates. Never run diagnostics/tests/plots with timing.
The roadmap may change when evidence rejects a mechanism. Do not combine these
hypotheses in one candidate to manufacture a win.
