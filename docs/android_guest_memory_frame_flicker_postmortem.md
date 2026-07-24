# Android Guest-Memory Frame Flicker Postmortem

## Scope and result

This document records the investigation of recurring previous-frame and partially constructed-frame flashes in Android builds. Nidhogg 2 (`CUSA07640`) was the primary reproducer. The same class of defect could affect other games that update GPU-visible resources through frequent CPU writes.

The confirmed fix is a strict page-watcher contract: logical watcher permissions and the host operating-system page protection must be updated synchronously, and an ordinary write fault may invalidate only the faulting 4 KiB page. Explicit bulk writes retain a separate whole-range path.

The final build was verified on the Vivo device at 60 FPS with no detected ABA reversals or severe texture-strip frames during a 30-second capture. The fix adds no polling loop, timeout, or global CPU-side submit serialization.

Recorded against:

- repository HEAD: `b082c63c35f9` plus the current uncommitted working tree;
- `src/video_core/page_manager.cpp` SHA-256: `2E0E9005D565CA0E45DEF9310F6C8730165FC03DB564FCC6A441B369B8A33B94`;
- tested Android library SHA-256: `45BF090B2544590BBC88CE34F72F32CE0759103949F265B07B643A99FA0DF4F6`;
- verification date: 2026-07-24.

Line numbers below refer to that snapshot. Function names are the durable references if later edits move the lines.

## Observable symptoms

The defect had two visually distinct layers:

1. A severe texture-atlas or horizontal-strip corruption appeared periodically, commonly once every six captured frames.
2. After the severe strip was removed, the game could still alternate between a complete frame and a partially constructed frame. Missing UI, fragments of text, diagonal geometry, or a previous frame appeared for a single refresh and then reverted.

Important diagnostic characteristics:

- The flashes occurred approximately two or three times per second during ordinary observation, but the severe form could have a precise six-frame cadence in a 60 FPS recording.
- A static screenshot was not a reliable test because it usually captured a valid frame.
- The defect was content-dependent. A scene or character selection could temporarily look correct while another scene still reproduced it.
- Stable reported FPS and submit/present rates did not rule it out. Nidhogg 2 continued to report approximately `60/60/60` while individual presented frames were wrong.
- Limbo was not a useful negative or positive reproducer for this specific contract. Nidhogg 2 and Hyper Light Drifter exercised the relevant update pattern more aggressively.

## Reliable detection method

Use a continuous 12-30 second, 60 FPS screen recording rather than isolated screenshots. Inspect both visual contact sheets and frame-to-frame metrics.

Two useful detectors are:

- **Severe-frame detector:** identifies the large texture-atlas or strip discontinuity seen in the first failure mode.
- **ABA detector:** identifies a short reversal in which frame `N` returns toward frame `N-2` after a different or incomplete frame `N-1`. This catches alternating complete/incomplete frames that do not look like the severe atlas strip.

The recordings used during this investigation were temporary diagnostic artifacts and were not added to the repository. Their identifying names and results are retained here so later work can reproduce the same progression.

| Experiment | Capture | Result |
| --- | --- | --- |
| Initial clean baseline | `lsx4_nidhogg_clean.mp4`, 714 frames / 12 s | 120 severe frames and 118 ABA events; severe corruption was nearly every sixth frame |
| CPU-side submit serialization | `lsx4_nidhogg_serial.mp4`, 588 frames | 98 severe frames; still exactly periodic, so CPU queue serialization was not the root fix |
| GPU timeline chaining | `lsx4_nidhogg_gpu_chain.mp4` | Removed the severe detector result in one red scene, but did not eliminate the content-dependent defect |
| Different character/scene | `lsx4_nidhogg_char.mp4`, 719 frames | 120 severe frames, again every six frames |
| Exact ordinary fault page | `nidhogg_exact_page.mp4`, 722 frames | Severe atlas-strip frames fell to zero, but 58 ABA/partial-frame reversals remained |
| Exact ordinary fault page, alternate content | `nidhogg_exact_alt.mp4`, 721 frames | 41 ABA reversals remained |
| Strict synchronous watcher protection | `nidhogg_strict_watchers.mp4`, 717 frames / 12 s | Severe frames: 0; ABA events: 0; frame delta mean 1.520, p95 2.017, max 2.134 |
| Strict watcher long run | `nidhogg_strict_watchers_30.mp4`, 1801 frames / 30 s | Severe frames: 0; ABA events: 0; frame delta mean 1.516, p95 2.002, max 2.662 |

## Unsuccessful or incomplete approaches

### Static screenshots

Screenshots frequently showed a correct frame even while the defect flashed several times per second. They are suitable for checking persistent texture damage, but not for accepting a temporal-ordering fix.

### Global CPU-side GNM submit serialization

The diagnostic mode waits for the submitted master-timeline value after each submit:

- [`src/video_core/renderer_vulkan/vk_scheduler.cpp`](../src/video_core/renderer_vulkan/vk_scheduler.cpp), `Scheduler::SubmitExecution`, lines 1277-1286;
- [`src/executor/native_runtime_api.cpp`](../src/executor/native_runtime_api.cpp), sentinel `run-android-serialize-gnm-submits`, lines 22360-22367.

This did not remove the six-frame corruption cadence. It also places a host wait in a throughput-critical path, so it must not become the production solution.

### Vulkan master-timeline chaining

Android submissions were chained to the preceding master-timeline value:

- [`src/video_core/renderer_vulkan/vk_scheduler.cpp`](../src/video_core/renderer_vulkan/vk_scheduler.cpp), `Scheduler::SubmitExecution`, lines 1192-1211.

This made queue ordering explicit and improved some scenes, but later content still exposed partial-frame flashes. It was therefore useful ordering hardening, not the complete root-cause fix.

### Presenter and render-target source selection

The investigation traced the live color target and the VideoOut source:

- [`src/video_core/renderer_vulkan/vk_presenter.cpp`](../src/video_core/renderer_vulkan/vk_presenter.cpp), `Presenter::PrepareFrame`, lines 438-468;
- [`src/video_core/texture_cache/texture_cache.cpp`](../src/video_core/texture_cache/texture_cache.cpp), `ExecutorTrackLiveDrawnColorTarget`, lines 889-907;
- [`src/video_core/texture_cache/texture_cache.cpp`](../src/video_core/texture_cache/texture_cache.cpp), `ExecutorTakeLiveFullResolutionTarget`, lines 909 onward;
- [`src/video_core/renderer_vulkan/vk_presenter.cpp`](../src/video_core/renderer_vulkan/vk_presenter.cpp), final present waits and submit, lines 1084-1089.

The trace established that Nidhogg 2 legitimately alternated between two full-resolution 1920x1080 surfaces: one VideoOut/sRGB surface and one UNORM render-target alias. The presenter was not randomly selecting an unrelated target. Source-selection changes could alter how the symptom looked, but did not remove its memory-coherency cause.

The bounded trace switch is retained at [`src/executor/native_runtime_api.cpp`](../src/executor/native_runtime_api.cpp), lines 22736-22740. The `run-trace-videoout-source` sentinel must be absent during normal performance testing.

### Treating every `AcquireMem` as a full barrier

PM4 tracing showed a high `AcquireMem` rate, approximately 17,849 occurrences in 18 seconds. Converting every occurrence into a full host/GPU barrier was rejected because:

- the frequency would make the cost substantial;
- it did not explain the exact page-update corruption signature;
- it would mask the broken watcher contract with global synchronization.

The `run-live-pm4-trace` sentinel was diagnostic only and must be absent during normal runs.

### Expanding one write fault to neighboring pages

An earlier optimization speculatively expanded a single write fault to a wider neighboring range. This reduced fault traffic but invalidated pages that had not actually faulted. It caused the severe texture-atlas strip and its precise six-frame cadence.

Removing that expansion and invalidating only the aligned faulting page eliminated the severe strip, but did not by itself fix the remaining ABA flashes. It was necessary, but not sufficient.

## Root cause

The problem was guest-memory coherency, not raw presenter ordering.

There were two coupled violations:

1. **Speculative range invalidation:** an ordinary page fault could be expanded beyond the page that actually faulted. Neighboring GPU-visible data was invalidated at the wrong point in the producer sequence.
2. **Logical/physical protection divergence:** watcher counters could say that a page required write protection while the host page remained writable because protection application was deferred or skipped through a permissive bypass state. CPU writes could then occur without a fault, so the texture cache did not receive the event that made the new data visible at the correct time.

This explains why queue serialization and presenter changes were incomplete. They could order already-recorded events, but they could not manufacture a missing memory-write notification.

## Final fix

### 1. Apply watcher protection synchronously

[`src/video_core/page_manager.cpp`](../src/video_core/page_manager.cpp), `ApplyWatcherProtection`, lines 445-449, now always calls `Protect(address, size, perms)`.

Do not reintroduce a condition that skips this call because an `ActualWriteProtected` bit, deferred state, or historical bypass appears to match. The watcher transition and operating-system protection transition form one contract.

The two watcher update paths that invoke it are:

- `UpdatePageWatchers`, lines 663-724, with the protection application at 679-686;
- `UpdatePageWatchersForRegion`, lines 727 onward, with the protection application at 747-754.

### 2. Do not persist an ordinary write-fault bypass

[`src/video_core/page_manager.cpp`](../src/video_core/page_manager.cpp), `CommitWriteFault`, lines 465-480, ignores the former persistent-fault request and only restores writable host permissions if the current cached page permissions actually contain `Write`.

The critical behavior is:

- if the page is already physically writable, the fault is committed;
- if the current logical permission does not allow writes, the function refuses to relax protection;
- otherwise it applies the current page permissions to exactly that page.

Do not allow a normal fault to leave the page permanently writable while a watcher remains active.

### 3. Invalidate exactly one page for an ordinary fault

[`src/video_core/page_manager.cpp`](../src/video_core/page_manager.cpp), `GuestFaultSignalHandler`, lines 587-624, initializes the invalidation range to the aligned 4 KiB faulting page at lines 595-596 and submits that range at lines 608-610.

Only an explicitly registered bulk write may replace that range with its known full extent, at lines 598-605. There must be no heuristic neighboring-page burst based solely on a recent fault address or cadence.

### 4. Preserve the explicit bulk-write fast path

Large known copies must not pay one signal fault per page. Their full range is invalidated before the write:

- [`src/video_core/page_manager.cpp`](../src/video_core/page_manager.cpp), `PrepareBulkGuestWrite`, lines 519-540;
- `SetBulkGuestWriteActivity`, lines 558-584;
- public entry points, lines 876-891.

The libc guest bridges explicitly bracket their destination range:

- [`src/core/aerolib/stubs.cpp`](../src/core/aerolib/stubs.cpp), `ExecutorLibcMemcpy`, lines 3937-3944;
- `ExecutorLibcMemmove`, lines 3954-3961;
- `ExecutorLibcMemset`, lines 4014-4019.

This separation is essential:

- unknown ordinary writes use the exact faulting page;
- known bulk writes use the exact caller-provided range;
- neither path guesses neighboring pages.

## Required invariants

Future changes to page tracking must preserve all of the following:

1. A watcher-count transition that changes logical permissions synchronously applies the equivalent host page protection before returning.
2. A write fault relaxes protection only when the current logical permissions permit writing.
3. An ordinary fault invalidates exactly the aligned faulting page.
4. A range invalidation larger than one page requires an explicit, active bulk-write range from the producer.
5. A bypass used for an active bulk write is removed when the bulk operation ends or a new watcher becomes active.
6. Presenter or Vulkan queue barriers are not substitutes for a missing CPU-write notification.
7. The production path remains event-driven: no polling, fixed delay, per-submit host wait, or game-specific address/signature rule.

## Regression procedure

For a page-watcher or Android presentation change:

1. Build and install the current `.so`, then verify its host and device SHA-256 values match.
2. Remove diagnostic sentinels, especially `run-android-serialize-gnm-submits`, `run-live-pm4-trace`, and `run-trace-videoout-source`.
3. Run Nidhogg 2 through both menu and character/gameplay scenes for at least 30 seconds.
4. Record at 60 FPS and inspect the sequence, not only a screenshot.
5. Require zero severe-strip frames and zero ABA reversals for acceptance.
6. Confirm FPS, submit rate, and present rate did not regress; the confirmed run remained at approximately `60/60/60`.
7. Recheck Hyper Light Drifter because it exercised a related resource-update pattern, and retain a game such as Limbo as a broader non-regression check rather than as the primary reproducer.

If a future change improves one scene but fails another, treat it as incomplete. The successful fix must satisfy the memory-protection invariants independently of engine, game, surface address, or frame cadence.
