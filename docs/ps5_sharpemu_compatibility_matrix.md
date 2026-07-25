# PS5 SharpEmu compatibility ledger

This ledger compares the isolated LSX4 PS5 loader/runtime semantics with the
local SharpEmu reference. It does not claim compatibility with protected
retail input or a complete PS5 system.

## Implemented and synthetically verified

| Area | LSX4 status |
| --- | --- |
| Containers | Raw NextGen ELF and decrypted/uncompressed fSELF; both known SELF magic values |
| Memory | Independent 16 KiB PS5 guest pages and low guest mapping registry |
| Dynamic metadata | Standard and SCE string, symbol, RELA, PLT-RELA, dependency, initializer and finalizer metadata |
| Relocations | Types 0, 1, 2, 4, 6, 7, 8, 10, 11, 16, 17, 18, 24, 32, 33 and 38, including signed/unsigned 32-bit overflow gates |
| Section fallback | `SHT_SYMTAB` exports and `SHT_RELA` when dynamic relocations are absent; stale optional fSELF sections do not reject an otherwise loadable stripped image |
| Linking | `DT_NEEDED`, full-name/NID/underscore aliases, deferred function/data/TLS imports and late intermodule rebinding |
| TLS | `PT_TLS`, FreeBSD/AMD64 Variant II TCB, DTV, module IDs/static offsets and late module installation into existing threads using a `0x20000` reservation |
| Lifecycle | preinit/init/fini and arrays, dependency-ordered startup, explicit PRX start/stop and process reset/relaunch |
| Discovery | `sce_module`, `sce_modules`, `Media/Modules`, `Media/Plugins`; `.prx`/`.sprx` case normalization; libkernel skip; FMOD plugin startup exception |
| Runtime boundary | Isolated ABI `lsx4-ps5-runtime/7`, version node `LSX4_PS5_RUNTIME_7.0`, PS5 artifact namespace and PS4/PS5 source overlap check |
| Validation | ARM64 smoke, 38-case loader matrix, ABI invalid-argument gates, ASan and UBSan excluding the cross-DSO `vptr` false-positive category |

## Implemented but not yet verified with a real dump

| Area | Remaining evidence |
| --- | --- |
| Game-directory loading | `load_game`, compatibility gate, module scan/order and first eboot launch need validation against legally obtained decrypted game layouts |
| Import behavior | Real NID sets, data/TLS bindings and module graphs may expose missing aliases or ordering constraints |
| Late TLS capacity | Guest-visible behavior is covered synthetically; real module count and aggregate TLS size must remain inside the reservation |
| Initializer/finalizer ABI | Synthetic functions run; real module calling conventions and side effects need observation |
| proc-param | Address is exposed by the isolated runtime, but the downstream metadata/HLE contract is not complete |
| Frontend consumer | The CLI probe is wired; production Android catalog/import/launch integration remains to be verified |

## Deliberately not implemented

| Area | Boundary |
| --- | --- |
| Retail protection | No SELF decryption, key handling, fake-signing bypass or protected retail unpacking |
| Compression/encryption | Compressed or encrypted SELF segments are rejected |
| System software | No complete Prospero system-module or production HLE graph |
| Diagnostic HLE | The CLI probe's zero-return policy is diagnostic, not a production implementation |
| Exception metadata | SharpEmu-style EH-frame/module registry is not exposed through the isolated public ABI; any future parser must be bounded and requires an intentional ABI decision |
| Metadata/content | `param.json` is not used for executable mapping; it belongs to metadata and app-content HLE |
| Graphics | No production GPU/AGC implementation |
| End-to-end claim | A real game workload, menus, audio, input and gameplay must be demonstrated on device before calling a title compatible |

SharpEmu itself expects decrypted/fake-signed input and does not provide retail
SELF decryption. Its 4 KiB page choice is not copied: the isolated LSX4 PS5
contract remains 16 KiB.
