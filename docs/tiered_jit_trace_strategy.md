# Optional tiered JIT and trace compilation

Status: implemented as an optional, default-off runtime feature.

Implemented in the shared dynamic-translation source and therefore built
independently into both PS4 and isolated PS5 runtimes. Device validation on
Xiaomi with Dreaming Sarah covered OFF and ON menu/gameplay paths. Bloodborne
(`CUSA03173`) exposed an unsafe conditional side exit: a newly published
guarded trace chose the wrong successor, entered a null-base memory access
and stopped the guest thread while audio continued. Conditional guarded
traces therefore remain fail-closed; Tier 2 currently publishes only
unconditional/fallthrough multi-block traces and preserves Tier 1 as the
fallback. Trace formation now validates a successor against the sampled
dominant target rather than incorrectly requiring the physical fallthrough
address, so non-contiguous direct-jump traces can actually be promoted.

Latest Xiaomi device validation (both checkboxes enabled):

- Bloodborne `CUSA03173`: menu 29–30 FPS; saved gameplay reached and remained
  responsive at 5–6 FPS; live `guards=0` two-block traces were published;
  sound produced non-zero samples; no guest exception, signal failure or
  KGSL kill occurred.
- Dreaming Sarah `PPSA02929`: menu 59 FPS and gameplay 35 FPS; picture,
  movement and non-zero audio samples were confirmed; conditional candidates
  were rejected without disturbing Tier 1.

Installed validation artifacts were
`e1033bd26c518cbc289e57f9e9ba00ecff4ecef47c3c394b3c4d5b2a3117ea34`
for PS4 and
`dc5aff2210f7d9e03026423776ea0689eb92907487e99f357a3d6721bbd76b5a`
for PS5.

## Objective

Add an optional optimizing tier above the current AArch64 JIT without
replacing or weakening the current compiler:

- Tier 0: the existing semantic/interpreter bridge.
- Tier 1: the existing `InitialNative` basic-block compiler.
- Tier 2: hot multi-block traces compiled in the background.

Tier 1 remains the source of truth, the immediate fallback and the only
active path when the options are disabled.

## Existing contracts to preserve

The current compiler already provides most of the publication and safety
mechanisms that Tier 2 needs:

- `RetiringTranslationRuntime` owns decoded blocks and native artifacts.
- `GuestRegionDirectory` and the thread-local 4-way execution cache resolve
  normal entries.
- `EdgeDestinationDirectory` and `OutboundEdgeState` atomically connect
  direct edges.
- The shared-chain ABI keeps one resident native frame across chained Tier 1
  blocks.
- `PublishedCodeWitness` revisions, sparse inspections and byte comparison
  retire stale self-modifying code.
- Indirect jumps and returns use a bounded PIC.
- Persistent IR/native caches have title, executable and native-ABI identity.
- Native faults publish the exact guest instruction.

Tier 2 must reuse these contracts. It must not introduce a parallel
dispatcher, a second guest-state format or a separate memory model.

## Why traces can still help

Shared-chain Tier 1 removes the central dispatcher on many edges, but each
basic-block boundary still normally:

- chooses and validates an edge;
- flushes cached guest GPR/YMM state;
- stores the successor RIP;
- branches through an indirect host pointer;
- reloads the next block's register cache;
- recomputes liveness and materializes flags within each block independently.

A trace can keep guest values and lazy flags resident across several blocks,
turn the common conditional successor into fallthrough, remove redundant
loads/stores and eliminate dead flag work across block boundaries.

## Managed UI options

Expose two restart-required checkboxes, both off by default:

1. `Tiered JIT`
   - enables low-overhead hot-edge profiling and the background promotion
     worker;
   - Tier 0 and Tier 1 behavior is otherwise unchanged.
2. `JIT trace compilation`
   - enables multi-block Tier 2 compilation;
   - depends on `Tiered JIT`; enabling it implicitly enables the master
     option, while disabling the master suspends all Tier 2 publication.

Suggested IDs:

```text
MANAGED_OPTIMIZATION_TIERED_JIT = 4
MANAGED_OPTIMIZATION_JIT_TRACE_COMPILATION = 5
```

Suggested preference keys:

```text
managed_tiered_jit
managed_jit_trace_compilation
```

The existing options 1–3 are resolved through the ordinary runtime setter.
The implementation now provides:

- `executor_lsx4_ps5_runtime_set_managed_optimization`;
- a separate resolved function pointer in `runtime_bridge.cpp`;
- routing to the active PS4 or PS5 library;
- storage of the JIT switches in common dynamic-translation code, which is
  compiled independently into each runtime `.so`.

This prevents a PS5 checkbox from accidentally changing the dormant PS4
runtime instance.

## Profiling without slowing the current JIT

When `Tiered JIT` is off, generated Tier 1 code must contain no additional
hot-path counter or callback.

When it is on:

1. Reuse the existing native-edge safepoint cadence. Tier 1 already reaches a
   validation path periodically (`kNativeEdgeSchedulePeriod`, currently
   4096 transitions).
2. At that safepoint, record `(source RIP, destination RIP)` in a per-thread
   fixed-size sketch with non-atomic saturating counters.
3. Merge only candidate hot edges into a process-wide profile at an existing
   dispatcher/safepoint boundary.
4. Queue a seed after a minimum number of sampled occurrences and a stable
   successor ratio.

Initial policy:

```text
minimum sampled edge count: 32
minimum dominant-successor ratio: 90%
maximum trace blocks: 8
maximum guest instructions: 256
maximum side exits: 8
one low-priority compiler worker
maximum active Tier-2 code: 32 MiB
```

With one sample per 4096 transitions, a four-sample seed represents roughly
16K hot-edge executions. Thresholds should be adaptive: lower during loading
and higher after the code budget is under pressure.

RIP-only sampling is insufficient for trace formation. The existing
`SampleJitGuestRip` and interpreter heat table remain useful diagnostics but
must not be treated as an edge profile.

## Trace plan

Introduce a `TracePlan` separate from `LsxDecodedRegion`:

```text
seed RIP
ordered member blocks
dominant internal edges
side-exit RIPs
member witness revisions and byte hashes
instruction/fault descriptors
code and compile budgets
```

Formation rules for the first implementation:

- start only from an already published native Tier 1 block;
- follow the dominant successor while it remains stable;
- stop at HLE/syscall/dispatcher boundaries, indirect control flow,
  unsupported operations, conflicting loop headers or the size budget;
- allow a conditional branch only when its dominant side is known;
- use a guard and exact state spill for the non-dominant side;
- allow a loop-closing backedge only when all loop members are inside the
  trace and a periodic safepoint remains reachable;
- never speculate guest memory values or HLE results in version 1.

Non-contiguous guest blocks remain explicit members with individual witness
revisions and byte validation. The emitter receives a temporary combined
instruction stream plus explicit guarded exits; the combined stream is never
used as the SMC identity.

Tier-1 releases decoded instructions after native publication to control
memory usage. The Tier-2 worker therefore re-decodes only queued hot members
from current guest bytes and checks their spans against the retained Tier-1
metadata before compilation.

Only native blocks with a shared-chain-compatible `DirectEntry` are eligible.
Dispatcher-boundary and synchronous-fault-resume blocks are excluded. This
restriction is required by the existing Tier-1 ABI and prevented faults seen
when early prototypes admitted every `NativeCode` block.

## Compiler reuse

Refactor the current native emitter into:

1. block analysis and register-cache selection;
2. instruction emission;
3. control-tail emission;
4. entry/exit publication.

Tier 1 continues using exactly the same calls and output as before. Tier 2
feeds several analyzed blocks into one emitter session:

- one shared native frame;
- one cross-block register allocation/liveness pass;
- internal hot edges become labels/fallthrough;
- side exits flush only live dirty state and jump to the preserved Tier 1
  entry;
- helper calls and faultable operations use the existing state/fault
  publication rules.

This avoids two subtly different x86 semantics implementations.

## Publication and rollback

Do not overwrite or delete the Tier 1 artifact.

For every promoted seed keep:

```text
baseline external entry
baseline resident entry
trace external entry
trace resident entry
composite witness
state: queued / compiling / active / retired / rejected
```

Compilation occurs outside `block_cache_mutex_`. Publication is a short
critical section:

1. recheck every member witness revision and guest byte hash;
2. reject the result if any member changed;
3. atomically republish the seed destination to the trace entry;
4. preserve the original Tier 1 destination for all side exits and rollback.

Any member mutation retires the whole trace and republishes Tier 1. A compile
failure, code-budget rejection or worker shutdown leaves Tier 1 untouched.

The checkboxes are applied before game launch. Runtime disabling can be added
later; the first version may require a restart, matching the current managed
optimization behavior.

## Faults, SMC and safepoints

- Each faultable instruction retains its exact `NativeFaultDescriptor`.
- Side exits set the exact guest RIP before transferring to Tier 1.
- A composite witness references every member block.
- The trace loop keeps periodic validation/cancellation safepoints; it may
  not create an unbounded native loop that bypasses SMC checks.
- Direct and indirect trace exits reuse `OutboundEdgeState` and the existing
  PIC.
- Tier 2 is initially excluded when live checked-native diagnostics or store
  watchpoints are enabled.

## Persistent data

Version 1 should not persist Tier-2 machine code. Persist only a compact
edge profile in a new, separately versioned file keyed by:

```text
title + executable fingerprint + JIT semantic ABI + trace-profile ABI
```

On the next launch, the profile may seed background compilation only after
all constituent Tier 1 blocks have been decoded and validated.

Native trace persistence can be considered later with a separate artifact
kind and ABI. It must not reuse the existing single-block native record
without explicitly representing all member witnesses and side exits.

## Metrics and acceptance gates

Add counters to the existing JIT JSON/HUD:

```text
profiled edges
queued / compiled / active / retired / rejected traces
compile time and generated bytes
average blocks and instructions per trace
trace entries and internal edges
side exits and side-exit ratio
SMC retirements
Tier-2 code budget
estimated Tier-1 boundary operations avoided
```

Required gates before enabling by default:

1. Options off: byte-for-byte equivalent Tier 1 emission for representative
   blocks and no profiler worker/counters on the execution path.
2. Options on, publication disabled: no gameplay or state difference.
3. Exact differential tests for arithmetic flags, SIMD, calls/returns,
   conditional side exits, faults and SMC invalidation.
4. Long device runs with automatic rollback on any composite-witness
   mismatch.
5. PS4 and PS5 A/B runs because the source is shared but each `.so` owns an
   isolated runtime instance.

## Rollout order

1. UI keys, active-runtime routing and common configuration; defaults off.
2. Edge profiler and telemetry only.
3. Background trace planning with publication disabled.
4. Publish straight-line traces with no calls, indirect edges or memory
   writes.
5. Add guarded conditional side exits.
6. Add internal loops and broader memory operations with precise faults.
7. Persist profiles; consider native trace persistence only after stability.

At every stage, disabling the options must return immediately to unchanged
Tier 1 behavior.

## Expected performance

These are engineering estimates until an edge profile is collected:

| Workload | Expected guest-CPU throughput | Expected end-to-end FPS |
|---|---:|---:|
| Long native loops already contained in one Tier 1 block | 0–8% | 0–5% |
| Short branch-heavy native blocks with stable successors | 15–35% | 8–25% |
| Helper-heavy hot traces after safe helper elimination/inlining | 25–60% | 15–40% |
| GPU/present-bound frame | 0–10% | 0–5% |

The upper range comes from retaining guest registers/flags and removing
multiple block boundaries, not merely from eliminating the central
dispatcher, because shared-chain Tier 1 already avoids much of that cost.

For the current `PPSA01342` first-frame problem, the observed bottleneck is
the synchronous Vulkan path: repeated 3840x2160 resource creation,
upload/readback, fence waits and roughly four graphics draws per second.
Tiered traces will not fix the black surface contract and are unlikely to
produce a large wall-clock improvement at this stage. A realistic immediate
gain is 0–10%; after the GPU path stops dominating, 10–30% overall is
plausible for CPU-heavy gameplay.

An order-of-magnitude FPS increase should not be promised from trace JIT
alone. The first implementation decision should be based on measured
`guest CPU time / frame`, hot-edge concentration and side-exit ratio.
