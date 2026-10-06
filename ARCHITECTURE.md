# LSX4 implementation architecture

## Scope

This document describes the primary PS4 runtime used by the Android application.
It identifies implementation ownership, data flow, and source evidence. It does
not describe experimental PS5 code or compatibility-only translator paths as part
of the primary PS4 architecture.

## Architecture identity

LSX4 uses a project-owned, in-process x86-64 to AArch64 dynamic binary translator
for its primary PS4 execution path. The translator is compiled into the LSX4 native
runtime and executes guest code in the same runtime boundary as the loader and HLE
integration.

The primary guest CPU pipeline is:

1. The PS4 loader maps the guest executable and resolves imported modules.
2. The iced-x86 FFI decodes x86-64 guest instructions.
3. LSX4 converts decoded instructions into its instruction model and operation IR.
4. The baseline compiler emits native AArch64 blocks into an LSX4-owned executable
   arena.
5. Published blocks are linked through direct edges, guarded exits, indirect target
   caches, and dispatcher fallbacks.
6. Hot blocks may be promoted through local recompilation and guarded trace
   compilation.
7. HLE calls cross an LSX4-owned invocation frame into the Funnel ARM service layer.

The primary path does not delegate guest CPU execution to an external whole-process
translator. Compatibility and diagnostic code for alternative translators remains
in the repository, but it is outside the architecture identity defined here.

## Guest CPU translator

### Decode and IR

`src/executor/dynamic_translation/decode_frontend.cpp` uses the iced-x86 FFI to
decode guest instructions. The result is normalized into LSX4 instruction records.
`operation_ir.*`, `scalar_ir_builder.*`, and the semantic modules define the
project-owned intermediate representation and guest behavior.

### Native code generation

`native_compiler.*`, `native_encoder.*`, `native_address_emitter.*`, and
`jit_aarch64_stubs.S` generate and enter AArch64 code. `code_memory.*` owns executable
memory. `region_registry.*`, `edge_portal.*`, and the execution core publish and
connect translated regions.

### Tiering and traces

The execution core supports three native tiers:

- Tier 0 provides synchronous baseline block compilation.
- Tier 1 recompiles hot individual blocks while retaining the Tier 0 fallback.
- Tier 2 compiles guarded multi-block paths and closed loops from observed edges.

Promotion preserves the previous entry for rollback. Guard failure, changed guest
code, compilation failure, or runtime disablement returns execution to a retained
lower tier.

### Code identity and invalidation

`code_witness.*` records guest byte fingerprints and page revisions. Published code
is validated against current guest bytes. Mutation retires affected translations
and prevents stale native code from remaining authoritative.

`artifact_repository.*` stores versioned decoded IR and relocatable native artifacts.
Records are tied to title identity, executable fingerprint, guest bytes, and the JIT
ABI. A mismatch causes rejection and fresh translation.

## Loader and HLE boundary

Funnel ARM owns the PS4 loader, module linker, kernel HLE, system libraries, process
lifecycle, guest memory services, audio services, and native input adaptation.

LSX4 owns the guest CPU to HLE transition. `hle_invocation_frame.*` captures guest
arguments and preserved state, invokes the resolved HLE target, and commits the
result back to the LSX4 machine image. Guest callbacks return through the same
runtime contract.

This boundary is important: the HLE and guest CPU implementations are connected,
but they are not the same component and do not have the same ownership or lineage.

## Graphics implementation

The graphics path is maintained in Funnel ARM and follows a shadPS4-derived design:

1. Guest GNM and PM4 commands enter the AMDGPU command processor.
2. Guest GCN shaders are decoded into the shader recompiler IR.
3. Shader passes produce SPIR-V for the host device.
4. The Vulkan renderer manages resources, synchronization, pipelines, presentation,
   and the Android Vulkan driver boundary.

LSX4-specific Android work includes mobile GPU policy, readback batching, Vulkan call
reduction, asynchronous pipeline handling, optional fragment shading rate, and
configurable robustness behavior. These adaptations do not change the documented
shadPS4 lineage of the HLE and GPU layers.

## Android integration

`android-app` owns activities, lifecycle, game selection, settings, touch controls,
controller input, the presentation surface, and JNI loading.

The JNI bridge loads `liblsx4_executor_android.so` and resolves the runtime API. The
normal game launch calls the AArch64 JIT entry point. The native runtime combines the
LSX4 translator with the Funnel ARM emulation layer in one final Android target.

## Implementation ownership

| Area | Owner | Source evidence |
| --- | --- | --- |
| Android application and lifecycle | LSX4 | `android-app/` |
| JNI and runtime loading | LSX4 | `android-app/app/src/main/cpp/runtime_bridge.cpp` |
| x86-64 decode and instruction model | LSX4 | `src/executor/dynamic_translation/decode_frontend.cpp`, `instruction_model.h` |
| LSX4 operation IR and semantics | LSX4 | `src/executor/dynamic_translation/operation_ir.*`, `*_semantics.*` |
| AArch64 compiler and executable cache | LSX4 | `native_compiler.*`, `native_encoder.*`, `code_memory.*` |
| Tiering, traces, linking, and rollback | LSX4 | `retiring_execution_core.*`, `edge_portal.*` |
| Guest code witnesses and invalidation | LSX4 | `code_witness.*`, `region_registry.*` |
| Persistent IR and native artifacts | LSX4 | `artifact_repository.*` |
| Guest CPU to HLE ABI | LSX4 | `hle_invocation_frame.*`, `runtime_gateway.h` |
| Loader, linker, kernel, and system HLE | Funnel ARM | `funnel-arm/src/core/` |
| Guest shader translation | Funnel ARM | `funnel-arm/src/shader_recompiler/` |
| AMDGPU and Vulkan emulation | Funnel ARM | `funnel-arm/src/video_core/` |
| Native input adaptation | Funnel ARM | `funnel-arm/src/input/` |

## Lineage statement

The Funnel ARM HLE, shader, and GPU implementation is derived from shadPS4, with
additional Orbis-derived foundations identified by Funnel ARM. LSX4 does not claim a
clean-room origin for those layers.

The LSX4 guest CPU translator, its operation IR, AArch64 compiler, tiering model,
code-witness system, artifact cache, and HLE invocation ABI are maintained as a
separate project-owned runtime implementation.

## Machine-readable record

`docs/architecture.yaml` records the same facts in a stable structure for tooling,
repository indexing, and automated analysis. The Markdown document remains the
human-readable authority. Any change to the primary backend or layer ownership must
update both files in the same commit.
