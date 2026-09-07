# Experimental ZeRO-3 pass simulation

Set `compile.pass_mode="auto"` to compare ZeRO-3 with ZeRO-3 + prefetch after
mandatory baseline profiling. `compile.simulation_mode="overlap"` is the search
default; `"serial"` disables modeled overlap using the same costs for new profiles.

The portable API is `simulate(graphs, profile, mode="overlap",
initial_memory_bytes=..., memory_limit_bytes=...)` in `core.py`. Its default is
`mode="serial"` for compatibility with saved v0 inputs. It never executes tensor
operations or communication benchmarks. Missing profile values are errors.

The overlap model schedules compute, all-gather, gradient-copy and reduction streams.
All-gather and reduce-scatter share a serialized NCCL communicator frontier.
Gathers start after preceding compute and queued gathers; `wait` blocks compute
on the requested parameter. A fused prefetch sums per-parameter communication
costs and makes every member ready at the **end of the group**, matching the
runtime's completion events. An ordinary gather after prefetch does not repeat
the transfer. Multiple groups share the same communication stream.

Compute uses the initial isolated operator profile. `reduce_grad` appends a
gradient to a dtype-specific bucket. If the next gradient would exceed capacity,
it first flushes the existing bucket; an exact fill does not trigger a flush.
Each flush waits for copies, pre-divides each gradient, submits a group of
per-parameter reduce-scatter calls, and casts/copies receive shards when their
storage dtype differs. Compute waits for copies before dropping gradient
references and for pending reduction before reusing a bucket. Single and double
buffering, oversized-gradient buffer growth, and final `end_backward` flush,
completion wait and bucket release are modeled. The actual FX graph is unchanged.

All ranks prepare representative-size tables for all-gather, D2D copy,
pre-division, reduce-scatter and required receive conversion/copy before rank 0
searches. Sizes are source bytes except for all-gather (padded output bytes).
Lookups use the smallest measured size at least as large as the request. Both
serial and overlap use the same isolated costs for new reduction profiles;
measured baseline reduction barriers are not added again. Old saved profiles
retain their original serial or compute/gather-only behavior.

Supported search conditions are fixed shapes, one forward/backward pair, one DP
group, eager execution, matching gather/gradient dtypes, a single gradient dtype,
`gradient_accumulation_steps=1`, gradient sizes divisible by world size, and no
offload or symmetric memory. All-gather/reduce debug synchronization must be off.
These initial restrictions are checked rather than silently approximated.

Noncontiguous gradient materialization uses the contiguous D2D-copy cost as an
approximation, with separate compute work and temporary storage. Host dispatch,
SM/bandwidth contention, collective grouping efficiency, allocator rounding and
fragmentation are excluded. Costs from the benchmark's process-group communicator
approximate the runtime's custom NCCL communicator. This models potential overlap
and does not establish wall-clock throughput or prediction accuracy.

Results include:

- `estimated_time_ms`, `peak_memory_bytes`, `feasible` and `reason`;
- `timeline`: each node's enqueue frontier, resource start/end and stream;
- `memory_timeline`: ordered allocation/free changes in device time;
- `memory_trace`: per-node views used by pass planning;
- `serial_work_ms`, `overlapped_ms` and `exposed_gather_wait_ms`;
- `reduction_timeline`: copy, pre-division, communication groups, receive stores
  and waits, with parent graph node and bucket/parameter identity;
- `exposed_reduce_wait_ms` and `reduction_waits_ms`: copy-cleanup, buffer-reuse,
  buffer-resize and final-backward waits on the compute path.

Gather buffers are allocated at the modeled launch frontier and remain live
until the last release and completion of queued GPU use. Storage aliases share
one lifetime, saved forward values survive backward consumers, and workspace
and reduction buffers contribute to the peak. Allocations already queued by the
host can differ from this device-time model; reserved memory is not estimated.
Search records `simulation_mode` so CPU replay uses the same model. Old saved
results without that field use serial replay.
