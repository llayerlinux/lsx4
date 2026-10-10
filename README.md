<p align="center">
  <img src="assets/lsx4-banner.png" width="640" alt="LSX4, open-source PlayStation 4 emulator for Android">
</p>

<div align="center">
  <img src="https://img.shields.io/badge/APRIL-2026-02?style=for-the-badge" alt="April 2026">
  <a href="https://discord.gg/p5KBs5cHsX"><img src="https://img.shields.io/discord/1459510161512333519?style=for-the-badge&amp;logo=discord&amp;logoColor=white&amp;label=Discord&amp;color=5865F2" alt="Discord"></a>
  <br>
  <a href="https://ko-fi.com/llayer"><img src="https://img.shields.io/badge/Support%20me%20on-Ko--fi-%23FF5E5B?style=for-the-badge&amp;logo=ko-fi&amp;logoColor=white" alt="Support LSX4 on Ko-fi"></a>
</div>

# LSX4

LSX4 is an open-source PlayStation 4 emulator application for Android.

LSX4 does not include games, firmware, or other copyrighted system content.

Development also makes use of current-generation large language models (LLMs).

## Download

<p align="center">
  <a href="https://play.google.com/store/apps/details?id=com.jetcalls.lsx4"><img src="https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png" height="80" alt="Get it on Google Play"></a>
</p>

## Implementation architecture

LSX4 is split into three main parts:

- The Android application handles the game library, settings, input, lifecycle,
  and the rendering surface.
- The native execution runtime runs PS4 x86-64 code on AArch64 Android devices.
- Funnel ARM provides the loader, HLE services, shader translation, and Vulkan GPU
  emulation used by the runtime.

### Guest CPU execution

The standard Android runtime uses LSX4's own in-process x86-64 to AArch64 JIT
compiler. It does not boot a Linux userland, run the game in a Linux container, or
hand guest execution to an external whole-process translator. The PS4 executable is
mapped directly by the native runtime, while the loader and HLE layer provide the
kernel and system services expected by the game.

For each guest code region, iced-x86 is used only to decode x86-64 instructions.
LSX4 converts the decoded instructions into its own instruction model and operation
IR, applies the guest semantics, and emits executable AArch64 blocks into an
LSX4-managed code arena. Calls from compiled guest code into emulated system
services use the LSX4 invocation ABI rather than a Linux syscall layer.

The instruction model, operation IR, AArch64 code generator, executable code cache,
block linker, tiering system, and invalidation logic are implemented in LSX4. The
baseline tier compiles native blocks synchronously. Hot blocks are recompiled at the
next tier, and stable hot paths can be combined into guarded traces. Direct edges
connect known block targets, indirect-target caches handle dynamic branches, and a
dispatcher fallback resolves targets that have not been compiled yet.

Guest-byte fingerprints and page revisions are checked before published code is
reused. If self-modifying guest code changes a mapped page, LSX4 retires the stale
translation and compiles a new version. Versioned artifacts retain a safe rollback
target when a trace guard fails or a newer translation is invalidated.

### Runtime documentation

[`ARCHITECTURE.md`](ARCHITECTURE.md) describes the runtime boundaries and links each
component to its source files. [`docs/architecture.yaml`](docs/architecture.yaml)
contains the same information in a machine-readable format.

## Current development priorities

- Improving performance in playable games.
- Expanding title compatibility by resolving frontiers and invariants.
- Implementing Mali support and improving performance on Mali devices.
- Increasing the number of playable titles by testing currently untested games.

## Playable games

Tested on devices powered by Snapdragon 8 Gen 3 and Snapdragon 8 Gen 5.

### Video demonstrations

<p align="center">
  <a href="https://www.youtube.com/watch?v=CQpTIQlKarM"><img src="https://img.youtube.com/vi/CQpTIQlKarM/hqdefault.jpg" width="49%" alt="Watch LSX4 video demonstration 1 on YouTube"></a>
  <a href="https://www.youtube.com/watch?v=4UmC0u8DJmY"><img src="https://img.youtube.com/vi/4UmC0u8DJmY/hqdefault.jpg" width="49%" alt="Watch LSX4 video demonstration 2 on YouTube"></a>
</p>

### Device previews (102 playable games)

<table>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/bloodborne.jpg" alt="Bloodborne running in LSX4"><br><sub><b>Bloodborne</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/the-last-of-us-remastered.png" alt="The Last of Us Remastered gameplay running in LSX4, verified by the user"><br><sub><b>The Last of Us Remastered</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/fear-the-spotlight-poco.png" alt="Fear The Spotlight gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Fear The Spotlight</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/inside.jpg" alt="INSIDE"><br><sub><b>INSIDE</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/wizordum-poco.png" alt="Wizordum gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Wizordum</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/hollow-knight-poco.png" alt="Hollow Knight gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Hollow Knight</b> · Poco Snapdragon 8 Gen 3</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/mortisomem-poco.png" alt="Mortisomem gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Mortisomem</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/cat-from-hell-2-poco.png" alt="Cat From Hell 2 gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Cat From Hell 2</b> · Poco Snapdragon 8 Gen 3</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/megaton-rainfall-poco.png" alt="Megaton Rainfall gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Megaton Rainfall</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/trine.png" alt="Trine running in LSX4 on Vivo"><br><sub><b>Trine</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/risk-of-rain-2.png" alt="Risk of Rain 2 gameplay running in LSX4 on Vivo"><br><sub><b>Risk of Rain 2</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/rollerdrome.png" alt="Rollerdrome training arena running in LSX4 on Vivo"><br><sub><b>Rollerdrome</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/deaths-door.png" alt="Death's Door gameplay running in LSX4 on Vivo"><br><sub><b>Death's Door</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/dredge.png" alt="DREDGE running in LSX4 on Vivo"><br><sub><b>DREDGE</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/marble-it-up-ultra-poco.png" alt="Marble It Up! Ultra gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Marble It Up! Ultra</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/the-gardens-between.png" alt="The Gardens Between running in LSX4 on Vivo"><br><sub><b>The Gardens Between</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/the-pedestrian.png" alt="The Pedestrian running in LSX4 on Vivo"><br><sub><b>The Pedestrian</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/going-under.png" alt="Going Under running in LSX4 on Vivo"><br><sub><b>Going Under</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/far-lone-sails.png" alt="FAR: Lone Sails running in LSX4 on Vivo"><br><sub><b>FAR: Lone Sails</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/orc-slayer-poco.png" alt="Orc Slayer gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Orc Slayer</b> · Poco Snapdragon 8 Gen 3</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/jazzpunk-directors-cut.jpg" alt="Jazzpunk: Director's Cut running in LSX4 on Vivo"><br><sub><b>Jazzpunk: Director's Cut</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/medievil.jpg" alt="MediEvil running in LSX4 on Vivo"><br><sub><b>MediEvil</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/fate-reawakened-poco.png" alt="FATE Reawakened gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>FATE Reawakened</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/buildest-poco.png" alt="Buildest gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Buildest</b> · Poco Snapdragon 8 Gen 3</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/besiege-poco.png" alt="Besiege gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Besiege</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/blueys-quest-for-the-gold-pen-poco.png" alt="Bluey&#x27;s Quest for the Gold Pen gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Bluey&#x27;s Quest for the Gold Pen</b> · Poco Snapdragon 8 Gen 3</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/fluster-cluck-poco.png" alt="Fluster Cluck gameplay running in LSX4"><br><sub><b>Fluster Cluck</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/super-monkey-ball-banana-blitz.png" alt="Super Monkey Ball: Banana Blitz gameplay running in LSX4 on Vivo"><br><sub><b>Super Monkey Ball: Banana Blitz</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/cant-drive-this.png" alt="Can&#39;t Drive This gameplay running in LSX4 on Vivo"><br><sub><b>Can&#39;t Drive This</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/a-short-hike.jpg" alt="A Short Hike running in LSX4 on Vivo"><br><sub><b>A Short Hike</b></sub></td>
  </tr>
</table>

<details>
<summary>Show more gameplay screenshots</summary>

<table>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/old-mans-journey.png" alt="Old Man's Journey running in LSX4 on Vivo"><br><sub><b>Old Man's Journey</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/nubla.jpg" alt="Nubla running in LSX4 on Vivo"><br><sub><b>Nubla</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/bonfire-peaks.png" alt="Bonfire Peaks running in LSX4 on Vivo"><br><sub><b>Bonfire Peaks</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/a-monsters-expedition-poco.png" alt="A Monster&#x27;s Expedition gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>A Monster&#x27;s Expedition</b> · Poco Snapdragon 8 Gen 3</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/megalan-11.png" alt="MEGALAN 11 running in LSX4 on Vivo"><br><sub><b>MEGALAN 11</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/grand-prix-rock-n-racing.png" alt="Grand Prix RockN Racing running in LSX4 on Vivo"><br><sub><b>Grand Prix RockN Racing</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/rock-n-racing-off-road-dx.png" alt="Rock 'N Racing Off Road DX running in LSX4 on Vivo"><br><sub><b>Rock 'N Racing Off Road DX</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/absolute-drift-poco.png" alt="Absolute Drift gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Absolute Drift</b> · Poco Snapdragon 8 Gen 3</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/what-the-golf-poco.png" alt="WHAT THE GOLF? gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>WHAT THE GOLF?</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/lethal-league-blaze.png" alt="Lethal League Blaze running in LSX4 on Vivo"><br><sub><b>Lethal League Blaze</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/resogun.png" alt="RESOGUN running in LSX4 on Vivo"><br><sub><b>RESOGUN</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/i-ai.jpg" alt="I, AI gameplay running in LSX4 on Vivo"><br><sub><b>I, AI</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/spider-rite-of-the-shrouded-moon.jpg" alt="Spider: Rite of the Shrouded Moon running in LSX4 on Vivo"><br><sub><b>Spider: Rite of the Shrouded Moon</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/dead-cells.jpg" alt="Dead Cells"><br><sub><b>Dead Cells</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/rogue-legacy-2.png" alt="Rogue Legacy 2 running in LSX4 on Vivo"><br><sub><b>Rogue Legacy 2</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/hyper-light-drifter.jpg" alt="Hyper Light Drifter running in LSX4"><br><sub><b>Hyper Light Drifter</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/grim-guardians-demon-purge-poco.png" alt="Grim Guardians: Demon Purge gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Grim Guardians: Demon Purge</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/dark-devotion.jpg" alt="Dark Devotion running in LSX4 on Vivo"><br><sub><b>Dark Devotion</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/rain-world.jpg" alt="Rain World running in LSX4"><br><sub><b>Rain World</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/owlboy.jpg" alt="Owlboy running in LSX4"><br><sub><b>Owlboy</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/islets.png" alt="Islets running in LSX4 on Vivo"><br><sub><b>Islets</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/thornkin-poco.png" alt="Thornkin gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Thornkin</b> · Poco Snapdragon 8 Gen 3</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/axiom-verge-2.jpg" alt="Axiom Verge 2 running in LSX4 on Vivo"><br><sub><b>Axiom Verge 2</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/carto.png" alt="Carto running in LSX4 on Vivo"><br><sub><b>Carto</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/dandara.png" alt="Dandara running in LSX4 on Vivo"><br><sub><b>Dandara</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/punch-club-2-fast-forward-poco.png" alt="Punch Club 2: Fast Forward gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Punch Club 2: Fast Forward</b> · Poco Snapdragon 8 Gen 3</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/coffee-talk-poco.png" alt="Coffee Talk gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Coffee Talk</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/chained-echoes.png" alt="Chained Echoes running in LSX4 on Vivo"><br><sub><b>Chained Echoes</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/the-messenger.png" alt="The Messenger running in LSX4 on Vivo"><br><sub><b>The Messenger</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/cyber-shadow.jpg" alt="Cyber Shadow running in LSX4 on Vivo"><br><sub><b>Cyber Shadow</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/broforce.png" alt="Broforce gameplay running in LSX4 on Vivo"><br><sub><b>Broforce</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/fight-n-rage.jpg" alt="Fight'N Rage running in LSX4 on Vivo"><br><sub><b>Fight'N Rage</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/farlands-journey.png" alt="Farlands Journey running in LSX4 on Vivo"><br><sub><b>Farlands Journey</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/dont-bite-me-bro.png" alt="Don&#39;t Bite Me Bro! gameplay running in LSX4 on Vivo"><br><sub><b>Don&#39;t Bite Me Bro!</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/void-terrarium-plus-plus.jpg" alt="void tRrLM();++ //Void Terrarium++ running in LSX4 on Vivo"><br><sub><b>void tRrLM();++ //Void Terrarium++</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/might-and-magic-clash-of-heroes.jpg" alt="Might &amp; Magic: Clash of Heroes running in LSX4 on Vivo"><br><sub><b>Might &amp; Magic: Clash of Heroes</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/despots-game.png" alt="Despot's Game running in LSX4 on Vivo"><br><sub><b>Despot's Game</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/lucy-dreaming.jpg" alt="Lucy Dreaming running in LSX4 on Vivo"><br><sub><b>Lucy Dreaming</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/turnip-boy-commits-tax-evasion.png" alt="Turnip Boy Commits Tax Evasion running in LSX4 on Vivo"><br><sub><b>Turnip Boy Commits Tax Evasion</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/sonic-mania.jpg" alt="Sonic Mania running in LSX4"><br><sub><b>Sonic Mania</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/shovel-knight.jpg" alt="Shovel Knight running in LSX4"><br><sub><b>Shovel Knight</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/hillbilly-doomsday.jpg" alt="Hillbilly Doomsday running in LSX4 on Vivo"><br><sub><b>Hillbilly Doomsday</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/super-rude-bear-resurrection-poco.png" alt="Super Rude Bear Resurrection gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Super Rude Bear Resurrection</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/super-meat-boy-poco.png" alt="Super Meat Boy! gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>Super Meat Boy!</b> · Poco Snapdragon 8 Gen 3</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/ultimate-chicken-horse.png" alt="Ultimate Chicken Horse running in LSX4 on Vivo"><br><sub><b>Ultimate Chicken Horse</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/toto-temple-deluxe.jpg" alt="Toto Temple Deluxe running in LSX4 on Vivo"><br><sub><b>Toto Temple Deluxe</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/perfect-universe.jpg" alt="Perfect Universe running in LSX4 on Vivo"><br><sub><b>Perfect Universe</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/limbo.jpg" alt="Limbo running in LSX4"><br><sub><b>Limbo</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/thomas-was-alone.png" alt="Thomas Was Alone gameplay running in LSX4 on Vivo"><br><sub><b>Thomas Was Alone</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/gonner.png" alt="GONNER running in LSX4 on Vivo"><br><sub><b>GONNER</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/cosmophony.jpg" alt="Cosmophony running in LSX4 on Vivo"><br><sub><b>Cosmophony</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/just-shapes-and-beats.jpg" alt="Just Shapes &amp; Beats running in LSX4"><br><sub><b>Just Shapes &amp; Beats</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/the-binding-of-isaac-rebirth.jpg" alt="The Binding of Isaac: Rebirth running in LSX4"><br><sub><b>The Binding of Isaac: Rebirth</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/hotline-miami-2.jpg" alt="Hotline Miami 2: Wrong Number running in LSX4"><br><sub><b>Hotline Miami 2: Wrong Number</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/not-a-hero.jpg" alt="NOT A HERO running in LSX4 on Vivo"><br><sub><b>NOT A HERO</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/nuclear-throne.jpg" alt="Nuclear Throne running in LSX4 on Vivo"><br><sub><b>Nuclear Throne</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/level-22-poco.png" alt="LEVEL 22 gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>LEVEL 22</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/nidhogg-2.jpg" alt="Nidhogg 2 running in LSX4"><br><sub><b>Nidhogg 2</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/mega-shoot.png" alt="Mega Shoot running in LSX4 on Vivo"><br><sub><b>Mega Shoot</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/replay-vhs-is-not-dead.jpg" alt="REPLAY: VHS is not dead running in LSX4 on Vivo"><br><sub><b>REPLAY: VHS is not dead</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/retrace-memories-of-death.png" alt="Retrace: Memories of Death running in LSX4 on Vivo"><br><sub><b>Retrace: Memories of Death</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/dreaming-sarah.jpg" alt="Dreaming Sarah running in LSX4 on Vivo"><br><sub><b>Dreaming Sarah</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/needy-girl-overdose-poco.png" alt="NEEDY GIRL OVERDOSE gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>NEEDY GIRL OVERDOSE</b> · Poco Snapdragon 8 Gen 3</sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/deltarune.jpg" alt="Deltarune running in LSX4"><br><sub><b>Deltarune Chapters 1 &amp; 2</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/undertale.jpg" alt="Undertale running in LSX4"><br><sub><b>Undertale</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/final-fantasy-ii-poco.png" alt="FINAL FANTASY II gameplay running in LSX4 on Poco Snapdragon 8 Gen 3"><br><sub><b>FINAL FANTASY II</b> · Poco Snapdragon 8 Gen 3</sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/final-fantasy.png" alt="FINAL FANTASY running in LSX4 on Vivo"><br><sub><b>FINAL FANTASY</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/another-world.jpg" alt="Another World running in LSX4 on Vivo"><br><sub><b>Another World</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/ninja-jajamaru-collection.png" alt="Ninja JaJaMaru Collection running in LSX4 on Vivo"><br><sub><b>Ninja JaJaMaru Collection</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/downwell.jpg" alt="Downwell running in LSX4"><br><sub><b>Downwell</b></sub></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/screenshots/nidhogg.jpg" alt="Nidhogg running in LSX4"><br><sub><b>Nidhogg</b></sub></td>
    <td align="center" width="50%"><img src="assets/screenshots/minit.jpg" alt="Minit running in LSX4"><br><sub><b>Minit</b></sub></td>
  </tr>
</table>

</details>

The table mirrors the games marked **Playable** in the Vivo test-device library through September 5, 2026.

### Playable compatibility list (102 games)

This list includes only games that have been tested. The broader set of supported games is larger and still requires testing.

<details>
<summary>Show complete compatibility list</summary>

| Game | Playable | Notes |
| --- | :---: | --- |
| Bloodborne | ✅ | |
| The Last of Us Remastered | ✅ | |
| Fear The Spotlight | ✅ | |
| INSIDE | ✅ | |
| Wizordum | ✅ | |
| Hollow Knight | ✅ | |
| Mortisomem | ✅ | |
| Cat From Hell 2 | ✅ | |
| Megaton Rainfall | ✅ | |
| Trine | ✅ | |
| Risk of Rain 2 | ✅ | |
| Rollerdrome | ✅ | |
| Death's Door | ✅ | |
| DREDGE | ✅ | |
| Marble It Up! Ultra | ✅ | |
| The Gardens Between | ✅ | |
| The Pedestrian | ✅ | |
| Going Under | ✅ | |
| FAR: Lone Sails | ✅ | |
| Orc Slayer | ✅ | |
| Jazzpunk: Director's Cut | ✅ | |
| MediEvil | ✅ | |
| FATE Reawakened | ✅ | |
| Buildest | ✅ | |
| Besiege | ✅ | |
| Bluey's Quest for the Gold Pen | ✅ | |
| Fluster Cluck | ✅ | |
| Super Monkey Ball: Banana Blitz | ✅ | |
| Can't Drive This | ✅ | |
| A Short Hike | ✅ | |
| Old Man's Journey | ✅ | |
| Nubla | ✅ | |
| Bonfire Peaks | ✅ | |
| A Monster's Expedition | ✅ | |
| MEGALAN 11 | ✅ | |
| Grand Prix RockN Racing | ✅ | |
| Rock 'N Racing Off Road DX | ✅ | |
| Absolute Drift | ✅ | |
| WHAT THE GOLF? | ✅ | |
| Lethal League Blaze | ✅ | |
| RESOGUN | ✅ | |
| I, AI | ✅ | |
| Spider: Rite of the Shrouded Moon | ✅ | |
| Dead Cells | ✅ | |
| Rogue Legacy 2 | ✅ | |
| Hyper Light Drifter | ✅ | |
| Grim Guardians: Demon Purge | ✅ | |
| Dark Devotion | ✅ | |
| Rain World | ✅ | |
| Owlboy | ✅ | |
| Islets | ✅ | |
| Thornkin | ✅ | |
| Axiom Verge 2 | ✅ | |
| Carto | ✅ | |
| Dandara | ✅ | |
| Punch Club 2: Fast Forward | ✅ | |
| Coffee Talk | ✅ | |
| Chained Echoes | ✅ | |
| The Messenger | ✅ | |
| Cyber Shadow | ✅ | |
| Broforce | ✅ | |
| Fight'N Rage | ✅ | |
| Farlands Journey | ✅ | |
| Don't Bite Me Bro! | ✅ | |
| void tRrLM();++ //Void Terrarium++ | ✅ | |
| Might & Magic: Clash of Heroes | ✅ | |
| Despot's Game | ✅ | |
| Lucy Dreaming | ✅ | |
| Turnip Boy Commits Tax Evasion | ✅ | |
| Sonic Mania | ✅ | |
| Shovel Knight | ✅ | |
| Hillbilly Doomsday | ✅ | |
| Super Rude Bear Resurrection | ✅ | |
| Super Meat Boy! | ✅ | |
| Ultimate Chicken Horse | ✅ | |
| Toto Temple Deluxe | ✅ | |
| Perfect Universe | ✅ | |
| Limbo | ✅ | |
| Thomas Was Alone | ✅ | |
| GONNER | ✅ | |
| Cosmophony | ✅ | |
| Just Shapes & Beats | ✅ | |
| The Binding of Isaac: Rebirth | ✅ | |
| Hotline Miami 2: Wrong Number | ✅ | |
| NOT A HERO | ✅ | |
| Nuclear Throne | ✅ | |
| LEVEL 22 | ✅ | |
| Nidhogg 2 | ✅ | |
| Mega Shoot | ✅ | |
| REPLAY: VHS is not dead | ✅ | |
| Retrace: Memories of Death | ✅ | |
| Dreaming Sarah | ✅ | |
| NEEDY GIRL OVERDOSE | ✅ | |
| Deltarune Chapters 1 & 2 | ✅ | |
| Undertale | ✅ | |
| FINAL FANTASY II | ✅ |  |
| FINAL FANTASY | ✅ | |
| Another World | ✅ | |
| Ninja JaJaMaru Collection | ✅ | |
| Downwell | ✅ | |
| Nidhogg | ✅ | |
| Minit | ✅ | |

</details>

## Contributing

Contributions are welcome. Bug reports, compatibility results, performance traces, and focused pull requests are especially useful. Please open an issue before starting a substantial architectural change.

## Repository layout

| Path | Purpose |
| --- | --- |
| `android-app/` | LSX4 Android application |
| `src/` | LSX4 client-owned native translation and Android runtime sources |
| `externals/` | Native build dependencies |
| `cmake/` | Android native build support |
| `scripts/` | Android dependency build helpers |
| `funnel-arm/` | Android/ARM-adapted emulation layer and authoritative native source profile; desktop distributions and duplicate externals are intentionally absent |
| `assets/` | Project artwork |

## Android application

```sh
cd android-app
./gradlew assembleDebug
```

The debug-only test mode can package a locally owned, decrypted NGS2 module from a
shad4pc `sys_modules` directory:

```sh
./gradlew assembleDebug -Pshad4pcNgs2Module=/path/to/shad4pc/user/sys_modules/libSceNgs2.sprx
```

`SHAD4PC_NGS2_MODULE` provides the same path through the environment. The module is
never included in release assets. LSX4 records the hash of a module installed by test
mode and removes only that managed copy when test mode is disabled or a release build
is launched; a module imported by the user is left untouched.

## Native runtime

```sh
git submodule update --init --recursive
powershell -File scripts/build-ffmpeg-android-pic.ps1
cmake --preset android-arm64-release
cmake --build --preset android-arm64-release
```

The native ownership boundary and optimization strategy are documented in
[`docs/arm_desktop_client_layer_architecture.md`](docs/arm_desktop_client_layer_architecture.md).
