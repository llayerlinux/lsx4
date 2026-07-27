# PS5 runtime: контрольная точка PPSA01342

Обновлено: 2026-07-27 07:xx (Europe/Moscow). Работа намеренно
поставлена пользователем на паузу. Телефон после команды `стоп` не
трогать, пока пользователь явно не попросит продолжить.

Этот файл — самодостаточная точка продолжения. Он описывает наблюдаемые
контракты и состояние эксперимента, а не только текущие номера строк.
Если реализация к моменту продолжения изменилась, сначала сопоставить
описанные ниже контракты с актуальным кодом и `git diff`, не переносить
старые хаки вслепую.

## Цель и ограничения

- Довести третью, большую PS5-игру `PPSA01342` до корректного меню,
  затем самостоятельно пройти в геймплей и визуально подтвердить его.
- Тестировать только на Xiaomi `24117RK2CG` (`zorn`), ADB serial:
  `adb-65d16af6-M4anBX._adb-tls-connect._tcp`.
- Vivo не трогать ни одной командой: на нём параллельно тестируются
  другие задачи. Любая ADB-команда обязана содержать точный `-s`.
- PS4-слой и уже работающие PS5-игры не регрессить. Все новые пути
  должны оставаться внутри изолированного PS5 runtime.
- Не тратить время на синтетические тесты, пока проблема находится
  между запуском и первым корректным кадром. Основной цикл:
  точный запуск тайтла -> узкие логи/профиль -> скрин/скринкаст ->
  общий контрактный фикс -> повтор.
- Не считать задачу завершённой по splash/одному кадру. Нужны меню,
  переход из меню и устойчивый геймплей с картинкой, звуком и вводом.

## Репозиторий и воспроизводимость

- Worktree: `D:\LProj\PetProj\lsx4-jit-independent-wt`
- Ветка: `main`
- Базовый commit на паузе:
  `901f4cbab4bc094678a85a52088ad9d2dcab3a54`
- Worktree сильно dirty и содержит ценную совместную работу. Нельзя
  делать `reset`, `checkout`, `clean`, `stash` или удалять чужие
  изменения.
- Основные изменённые/новые PS5-файлы:
  - `src/executor/ps5_desktop/runtime_api.cpp`
  - `src/executor/ps5_desktop/vulkan_presenter.cpp`
  - `src/executor/ps5_desktop/vulkan_presenter.h`
  - `src/executor/ps5_desktop/gen5_compute_recompiler.{h,cpp}`
  - `src/executor/ps5_desktop/vulkan_fullscreen_triangle.{vert,spv.inc}`
  - `src/executor/ps5_desktop/vulkan_guest_*`
- SHA-256 исходников в момент паузы:
  - `runtime_api.cpp`:
    `164F793D2DC2A2ADA1BBDA29C7147169A85ECE828E58A794A988386EC918184E`
  - `vulkan_presenter.cpp`:
    `69444D599269D5FDF0DC993695FFE8D052E965F535F4529974025B253CCA36F1`
  - `gen5_compute_recompiler.cpp`:
    `0B17B2C9E6FF8D7D2E1EED07127658056D16B8C7497392EF1345F05122BE682D`

Перед продолжением выполнить только read-only сверку:

```powershell
git status --short
git rev-parse HEAD
git diff -- src/executor/ps5_desktop/runtime_api.cpp `
  src/executor/ps5_desktop/vulkan_presenter.cpp `
  src/executor/ps5_desktop/gen5_compute_recompiler.cpp
```

Если commit/hash отличаются, сохранить актуальные изменения и
переоценить контракт по разделу «Что уже реализовано».

## Последний фактический результат

- Игра запускалась точным параметром тайтла; корректный полноэкранный
  splash был виден.
- После splash гостевой кадр оставался чёрным. Последняя локальная
  фиксация:
  `build/ps5-import/PPSA01342/uipass.png`.
- На ней: чёрный кадр, виртуальные органы управления, HUD
  `time 07:36`, `blocks 249,891`, `FPS 0` и системный ANR-диалог
  приложения «Безопасность».
- Последний процесс игры (PID `13240`) упал примерно через 7:36 с
  `SIGSEGV`, fault address `0`, thread `ps4run-intent-a`. Основной
  процесс приложения PID `9478` оставался жив. Это PID только того
  запуска; не использовать его как константу.
- При паузе новый запуск не выполнялся. Последний диагностический
  source-патч, описанный ниже, не собран и на устройство не ставился.
- Установленная на телефоне `.so` соответствует предыдущей сборке:
  - локальный файл:
    `build/runtime-isolation-on/liblsx4_executor_ps5_android.so`
  - размер: `43,796,536` bytes
  - время: `2026-07-27 07:13:44`
  - SHA-256:
    `CC8926201A5E46870667B971A3568050AE66DB0E4A64B528DC952CEB5D63FD73`

## Что уже реализовано

Искать по именам функций и лог-маркерам, а не по номерам строк.

1. В PS5 Gen5 пути добавлены:
   - компиляция гостевых compute/vertex/pixel шейдеров в SPIR-V;
   - Vulkan compute и graphics execution;
   - BC1–BC7 и block detiling;
   - 3D texture type `10`;
   - generic fullscreen-triangle bridge;
   - HLE-контракты, pthread priority NID и диагностический fault probe.
2. `ExecuteVulkanGen5Graphics` теперь принимает `kind=0` storage
   buffer descriptors в vertex и pixel stages:
   - создаёт host-coherent storage buffers;
   - загружает guest memory для read bindings;
   - формирует descriptor sets для VS set 0 и PS set 1;
   - привязывает их перед draw;
   - после fence записывает written/atomic buffers обратно гостю.
3. Ранняя проверка descriptor kinds допускает:
   - VS: `0`, `16`, `17`;
   - PS: `0`, `1..13`, `15`, `16`, `17`.
   Благодаря этому исчез прежний отказ `reason=vertex-descriptor` для
   важного 3840x2160 UI-pass.
4. До паузы важный pass имел:
   - VS `0x2048881700`, buffers `2`, output parameter 0;
   - PS `0x204ef7f800`, buffers `1`, input parameter 0, MRT1;
   - target `0x205d650000`, `3840x2160`.
   После допуска `kind=0` stage был принят без
   `vertex-descriptor`, `pixel-descriptor`, `guest-buffer`,
   descriptor-update или Vulkan failure, но визуальный ненулевой
   output ещё не подтверждён.
5. Перед паузой в `vulkan_presenter.cpp` добавлена узкая диагностика
   buffer-backed graphics passes:
   - лог `buffer pass=... stage=vs|ps ... nonzero=...`;
   - `target output ... buffer_pass=...`;
   - `storage output buffer_pass=...`.
   Этот патч находится только в source, ещё не собран и не запускался.

## Наблюдаемые GPU-контракты

- Зарегистрированные video-out scanout buffers:
  `0x2057d00000` и `0x2059d00000`, логически `1920x1080`.
- Последние их выборки были полностью нулевыми:
  `nonzero=0/4096`.
- Основная промежуточная 3840x2160 поверхность
  `0x205d650000` также была нулевой до выполнения позднего UI-pass.
- Декодированный Bink/video target около `0x20fa100000` был
  содержательным: примерно `4008/4096`, max `123`.
- У Bink pixel shader `0x2048a18b00` MRT=0 и имеются read/write
  image aliases. Несколько fullscreen bridge passes выполнялись, но
  содержимое ещё не было корректно доведено до scanout.
- Единственный устойчивый оставшийся graphics reject относился к
  targetless read-only pass (`target=0`, images=2, samplers=2,
  pin=1). Его нельзя считать причиной чёрного scanout без новых
  доказательств.

## Узкое место производительности

- Game RSS достигал примерно `3.18 GB`.
- Swapchain создавался `3840x2160`, 5 images, около 33 MB на image.
- `ExecuteVulkanGen5Graphics` сейчас на каждый draw заново создаёт и
  уничтожает staging/guest buffers, images, samplers, descriptor
  layouts/pool/sets, shader modules, render pass, framebuffer,
  pipeline, command buffer и fence.
- На каждом draw также выполняются guest read, detile, upload,
  синхронный `vkWaitForFences`, readback, retile и guest write.
  Наблюдаемая скорость была около 4 draw/s. Телефон целиком уходил в
  ANR, что затрудняло проверку и, вероятно, усиливало memory pressure.
- `TryAgcDriverSubmitDcb` не держит `g_runtime.mutex` на всём Vulkan
  исполнении. Он держит отдельный `g_agc_submit_mutex`; короткие
  участки `g_runtime.mutex` находятся до/после. Поэтому нельзя
  механически «исправлять долгий runtime mutex»: это не подтверждено.
- `ExecuteVulkanGen5Graphics` держит глобальный presenter
  `g_mutex` на всём синхронном draw. Снимать его без новой
  синхронизации нельзя: queue, command pool и presenter state имеют
  Vulkan external-synchronization requirements.

## Диагностический флаг

На устройстве файл
`files/lsx4-home/diag-jit-rip-probe` был переименован в
`files/lsx4-home/diag-jit-rip-probe.disabled`.

Это убрало огромный поток fiber/APR trace и заметно разгрузило запуск,
но одновременно отключило полезный fault sentinel, поэтому последний
null crash не получил guest RIP. При необходимости короткой
диагностики лучше сначала разделить флаги в коде:

- fault handler/sentinel — отдельный тихий флаг;
- verbose fiber/APR traces — отдельный флаг с жёстким лимитом.

Не включать старый общий флаг на длительный прогон.

## Следующий минимальный цикл

1. Убедиться, что пользователь явно снял паузу и Xiaomi доступен.
2. Собрать текущий source с уже добавленной узкой buffer-pass
   диагностикой. Синтетические тесты не запускать.
3. Поставить `.so` в оба runtime location и запустить строго
   `PPSA01342`, не выбирать игру пролистыванием.
4. Фильтровать logcat только по `LSX4-PS5-GFX`, crash и ANR.
5. Дождаться buffer-backed pass для target `0x205d650000` и проверить:
   - ненулевые ли VS/PS guest buffers до draw;
   - ненулевой ли target сразу после draw;
   - есть ли written storage output.
6. Ветвление по факту:
   - buffers нулевые -> исправлять получение/обновление descriptor
     user data и guest address/size;
   - buffers содержательны, target нулевой -> сравнить SPIR-V ABI,
     vertex fetch/interpolation/export контракт с SharpEmu;
   - target содержателен, scanout нулевой -> исправлять общую цепочку
     surface ownership/copy/flip до registered video-out buffer;
   - target и scanout содержательны, экран чёрный -> проверять
     present selection, format/tile/scale и stale-frame policy.
7. После первого стабильного кадра оптимизировать общим GPU cache:
   pipeline/layout/module cache и persistent guest-image resources.
   Не вводить title-specific адреса или шейдеры.
8. Когда появится меню, самостоятельно нажать нужный пункт через
   виртуальный геймпад и проверить устойчивый геймплей, звук и ввод.
9. После общего фикса коротко перепроверить две ранее работающие
   PS5-игры и PS4 smoke path, не меняя PS4 реализацию.

## Команды продолжения

ADB:

```powershell
$adb = 'C:\Users\Litesav\AppData\Local\Android\Sdk\platform-tools\adb.exe'
$xiaomi = 'adb-65d16af6-M4anBX._adb-tls-connect._tcp'
& $adb -s $xiaomi get-state
```

Сборка:

```powershell
& 'C:\Users\Litesav\AppData\Local\Android\Sdk\cmake\3.22.1\bin\cmake.exe' `
  --build build/runtime-isolation-on `
  --target lsx4_executor_ps5_android -j 8
```

Runtime нужно обновить в обоих местах sandbox приложения:

```text
files/runtime/liblsx4_executor_ps5_android.so
files/liblsx4_executor_ps5_android.so
```

Использовать промежуточный `/data/local/tmp` и `run-as
app.lsx4.android`; после копирования сверить размер/hash в обоих
местах. Не выполнять команды без `-s $xiaomi`.

Точный запуск:

```powershell
& $adb -s $xiaomi shell am start `
  -n app.lsx4.android/.MainActivity `
  --ez autoload_existing_runtime true `
  --ez autoinit_runtime true `
  --es scan_game_path /data/user/0/app.lsx4.android/files/lsx4-home/games/PPSA01342/eboot.bin `
  --es launch_game_path /data/user/0/app.lsx4.android/files/lsx4-home/games/PPSA01342/eboot.bin `
  --es game_platform ps5 `
  --ez fullscreen_render true `
  --ez embedded_aarch64_jit_backend true
```

Скриншот с Xiaomi делать в два шага:

```powershell
& $adb -s $xiaomi shell 'screencap -p > /data/local/tmp/ppsa01342.png'
& $adb -s $xiaomi pull /data/local/tmp/ppsa01342.png `
  build/ps5-import/PPSA01342/
```

Полезные локальные референсы:

- SharpEmu: `D:\LProj\PetProj\_tmp_sharpemu_ps5_analysis`
- Kyty: `D:\LProj\PetProj\_tmp_kytyps5_analysis`
- сохранённые SharpEmu-сравнения:
  `build/ps5-import/PPSA01342/sharpemu-compare.*.log`
- последний сохранённый AGC лог:
  `build/ps5-import/PPSA01342/demons-current-agc.log`

## Запрещённые быстрые обходы

- Не подменять scanout «последней ненулевой поверхностью» без
  доказанного surface/flip контракта: это может показать видео, но
  скрыть проблему меню и сломать другие игры.
- Не хардкодить `PPSA01342`, shader addresses или guest addresses в
  runtime-логику.
- Не переносить PS5 GPU обходы в PS4 слой.
- Не снимать глобальный Vulkan mutex и не распараллеливать queue /
  command pool без корректной external synchronization.
- Не объявлять успех по splash или единичному промежуточному кадру.
