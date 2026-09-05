# Experimental ZeRO-3 pass simulation

Set `compile.pass_mode="auto"` to compare ZeRO-3 with ZeRO-3 + prefetch after
mandatory baseline profiling. `compile.simulation_mode="overlap"` is the search
default; `"serial"` retains the initial serial model.

The portable API is `simulate(graphs, profile, mode="overlap",
initial_memory_bytes=..., memory_limit_bytes=...)` in `core.py`. Its default is
`mode="serial"` for compatibility with saved v0 inputs. It never executes tensor
operations or communication benchmarks. Missing profile values are errors.

The overlap model schedules one compute stream and one all-gather stream.
Gathers start after preceding compute and queued gathers; `wait` blocks compute
on the requested parameter. A fused prefetch sums per-parameter communication
costs and makes every member ready at the **end of the group**, matching the
runtime's completion events. An ordinary gather after prefetch does not repeat
the transfer. Multiple groups share the same communication stream.

Compute uses the initial isolated operator profile. Gradient-copy/reduce/flush
costs from the actual baseline are synchronized intervals: reduction overlap is
not modeled. Host dispatch overhead, SM/bandwidth contention, allocator rounding
and fragmentation are also excluded. This is a model of potential compute/gather
overlap, not a prediction of wall-clock training throughput. Supported search
conditions remain fixed shapes, one forward/backward pair, one DP group, eager
execution, matching gather/gradient dtypes and no offload or symmetric memory.
All-gather debug synchronization flags must be disabled in overlap search.

Results include:

- `estimated_time_ms`, `peak_memory_bytes`, `feasible` and `reason`;
- `timeline`: each node's enqueue frontier, resource start/end and stream;
- `memory_timeline`: ordered allocation/free changes in device time;
- `memory_trace`: per-node views used by pass planning;
- `serial_work_ms`, `overlapped_ms` and `exposed_gather_wait_ms`.

Gather buffers are allocated at the modeled launch frontier and remain live
until the last release and completion of queued GPU use. Storage aliases share
one lifetime, saved forward values survive backward consumers, and workspace
and reduction buffers contribute to the peak. Allocations already queued by the
host can differ from this device-time model; reserved memory is not estimated.
Search records `simulation_mode` so CPU replay uses the same model. Old saved
results without that field use serial replay.
