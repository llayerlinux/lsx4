# PS5 desktop isolation

The current device-test checkpoint and exact continuation procedure are in
[`PS5_RUNTIME_HANDOFF.md`](../PS5_RUNTIME_HANDOFF.md).

## Decision

The PS5 desktop implementation is owned by Funnel ARM under
`funnel-arm/src/ps5_desktop`. It is activated only by
`LSX4_ENABLE_PS5_DESKTOP_PATH=ON` and follows an LSX4 executor path under
`src/executor/ps5_desktop`.

The option is off by default. With the option off:

- the PS5 manifest is not included;
- no PS5 source is configured or compiled;
- no PS5 object enters `lsx4_executor_android`;
- PS4 initialization, HLE lookup, JIT caches, fault handlers, and GPU state
  remain unchanged.

With the option on, Funnel PS5 sources and the common translation sources are
compiled into the separate `lsx4_executor_ps5_android` shared library. They
are not linked into the PS4 runtime. The two libraries compile their own copy
of the translation core, so their global state and native artifact caches are
independent even when both libraries are present in one application.

The PS5 library exports only the `executor_lsx4_ps5_*` ABI. Its internal JIT
and bridge symbols are hidden by a dedicated version script. Runtime platform
selection belongs above the native libraries; PS5 conditionals must not enter
the PS4 process entry.

## Source ownership

| Owner | Directory | Responsibility |
| --- | --- | --- |
| Funnel ARM | `src/ps5_desktop` | PS5 loader, memory model, HLE, Prospero/AGC GPU path, and platform contract |
| LSX4 | `src/executor/ps5_desktop` | PS5-to-JIT adapter, public C ABI, mapping registry, HLE callbacks, entry path, cache namespace, and fault bridge |
| LSX4 | `src/executor/dynamic_translation` | Candidate platform-neutral translation core only |

PS5 code may not include a PS4 implementation as a shortcut. Code becomes
shared only after it has a platform-neutral interface and PS4 regression
coverage.

## JIT reuse decision gate

The first choice is one translation core with injected platform contracts:

- guest page size and mapping policy;
- HLE resolver and ABI metadata;
- TLS/FS/GS setup;
- fault delivery;
- CPU capability profile;
- native artifact-cache namespace and ABI epoch.

JIT source duplication is allowed only if extracting one of these contracts
would change observable PS4 behavior or make PS4 verification impractical.
If duplication is required, it lives below `src/executor/ps5_desktop/jit`,
uses the `Lsx4::Ps5Desktop` namespace, has a distinct artifact-cache ABI
epoch, and receives PS5-only fixes. It must not replace or patch PS4 files.

## Current compatibility gate

The isolated backend contract now reports `ready == true` for the currently
declared CPU-state and instruction requirements:

- PS4 code witnesses retain 4 KiB guest pages;
- the separately compiled PS5 JIT uses 16 KiB guest pages for invalidation;
- PS5-only semantic dispatch implements immediate and register
  `EXTRQ`/`INSERTQ`;
- `MONITORX` is a no-op and `MWAITX` yields the host thread.

These features are compiled only when `LSX4_PS5_DESKTOP_PATH` is defined.
They do not enter the PS4 semantic dispatch.

## Runtime ABI

`src/executor/ps5_desktop/runtime_api.h` exposes the isolated loader-facing
interface:

- initialize the PS5 runtime and its separate artifact store;
- register and unregister 16 KiB identity-mapped guest ranges;
- install PS5 HLE resolution, invocation, FP-result, and signal callbacks;
- bind ELF imports to isolated PS5 HLE thunks;
- load, relocate, unload, and launch a NextGen ELF or uncompressed SELF eboot;
- load a PRX as an owner-bound module and run its initializer/finalizer
  lifecycle independently;
- load a complete game directory, scanning `sce_module`, `sce_modules`,
  `Media/Modules`, and `Media/Plugins`;
- resolve full symbol names and NID prefixes between images, rebind deferred
  function/data relocations after later PRX loads, and start boot modules in
  `DT_NEEDED` order;
- emit a machine-readable and textual compatibility report before launch;
- create and destroy a PS5 thread context with a FreeBSD/AMD64 Variant II
  static TLS block, TCB self pointers, stack canary, and guest-visible DTV;
- execute a guest entry with PS5 arguments, FS base, and supplied stack;
- query status and reset PS5-only state.

Runtime ABI version 7 includes `bind_import`, eboot/PRX lifecycle calls,
game-directory loading, and the compatibility probe report,
and explicit per-thread TLS/TCB lifecycle calls. The TLS layout follows the
model used by SharpEmu: module 1 occupies the first aligned negative offset
from FS, later PRX images receive increasing module IDs/static offsets, and
the TCB contains one guest-visible DTV. The runtime reserves `0x20000` bytes
for static TLS, so a PRX loaded after a thread context already exists can add
its `tdata`/`tbss`, advance the DTV generation and maximum module ID, install
its DTV slot, and update the thread's maximum static offset without moving FS.
This is guest-visible late-DTV behavior rather than an unbounded allocator;
modules which exceed the reservation are rejected.
The loader supports `PT_LOAD`, SCE RELRO, TLS/proc-param discovery, standard
and SCE dynamic metadata, relative relocations, local symbols, and imported
`R_X86_64_64`/`PC32`/`PLT32`/`GLOB_DAT`/`JUMP_SLOT`/`32`/`32S`/`PC64`/
`SIZE32`/`SIZE64`/`RELATIVE64` bindings. TLS relocations include `DTPMOD64`,
`DTPOFF64`, and `TPOFF64`. It discovers preinit/init/fini arrays, runs the
eboot startup sequence before entry, and exposes explicit PRX start/stop
calls. `DT_NEEDED` strings and dynamic exports are retained, including full
names, NID prefixes before `#`, and aliases without a leading underscore.
Deferred imports are rebound through temporarily writable host pages and their
original RELRO protection is restored immediately. Encrypted or compressed
SELF segments remain rejected instead of
being mapped incorrectly.

The executable milestone is covered by `tests/ps5_runtime_smoke.cpp`. On
Android/AArch64 it creates an uncompressed NextGen SELF, applies a
`JUMP_SLOT` import, executes `EXTRQ`, `INSERTQ`, `MONITORX`, and `MWAITX`,
reads initialized data through FS-relative Variant II TLS, validates the TCB
self pointer in an HLE callback, validates extended relocation values, runs
eboot `DT_INIT`, loads a synthetic PRX with module ID 2, reads that module's
TLS through `FS:[-0x40]`, exports a NID-qualified function, rebinds a late
eboot import to that function, validates `DT_NEEDED`, runs `DT_FINI`, and
returns `0x2468`. A second scenario builds the SharpEmu-compatible directory
layout, runs `load_game`, validates the probe report, starts its module
automatically, and launches the eboot without initializing the PS4 runtime.
It also loads a third TLS-bearing PRX after the thread context exists,
validates the late TLS bytes and DTV slot, exercises callback/mapping/thread
invalid-argument gates, and performs a complete reset/reinitialize/relaunch in
one process. `tests/ps5_loader_matrix.cpp` covers 38 accepted/rejected loader
cases, including stripped fSELF with stale optional section headers.

The ARM64 sanitizer configuration uses ASan and UBSan with frame pointers.
Only UBSan `vptr` is disabled: the dedicated version script intentionally
localizes all C++ RTTI in the plugin, which makes that check report correct
libc++ and Xbyak objects as cross-DSO type mismatches. All other enabled
checks run with `halt_on_error=1`.

For a legally obtained decrypted game dump, build
`tools/ps5_game_probe.cpp` with:

```powershell
.\scripts\build-ps5-game-probe-android.ps1
```

On the ARM64 target, invoke it with the isolated runtime library and either the
game directory or `eboot.bin`. Add `--launch` only after reviewing the printed
unresolved data/TLS import count:

```text
./ps5_game_probe ./liblsx4_executor_ps5_android.so /data/local/tmp/game
./ps5_game_probe ./liblsx4_executor_ps5_android.so /data/local/tmp/game --launch
```

The probe installs a diagnostic zero-return HLE policy so execution can expose
the first semantic blocker. It is intentionally not presented as the
production Prospero HLE implementation.

Source-profile isolation can be checked without configuring the full Android
dependency graph:

```powershell
cmake -P cmake/verify_ps5_source_isolation.cmake
```
