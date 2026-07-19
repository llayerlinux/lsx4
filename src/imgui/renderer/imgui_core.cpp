// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <SDL3/SDL_events.h>
#include <imgui.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_set>

#include "common/config.h"
#include "common/path_util.h"
#include "core/debug_state.h"
#include "core/devtools/layer.h"
#include "imgui/imgui_layer.h"
#include "imgui_core.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include "sdl_window.h"
#include "texture_manager.h"
#include "video_core/renderer_vulkan/vk_presenter.h"

#include "imgui_fonts/notosansjp_regular.ttf.g.cpp"
#include "imgui_fonts/proggyvector_regular.ttf.g.cpp"

static void CheckVkResult(const vk::Result err) {
    LOG_ERROR(ImGui, "Vulkan error {}", vk::to_string(err));
}

static std::vector<ImGui::Layer*> layers;

// Update layers before rendering to allow layer changes to be applied during rendering.
// Using deque to keep the order of changes in case a Layer is removed then added again between
// frames.
static std::deque<std::pair<bool, ImGui::Layer*>> change_layers{};
static std::mutex change_layers_mutex{};
// Updated synchronously by AddLayer/RemoveLayer while the object is still alive. The host input
// thread queries only this set, never a Layer pointer, so a destructor queued for the next frame
// cannot leave a dangling pointer for modal detection.
static std::unordered_set<ImGui::Layer*> capturing_layers{};

struct HostGamepadEvent {
    ImGuiKey key;
    bool pressed;
};

struct HostGamepadMapping {
    std::uint32_t mask;
    ImGuiKey key;
};

// Same digital mapping used by the SDL backend on PC. Square/touchpad are intentionally omitted:
// the PC backend does not feed Square into ImGui navigation either, and touchpad is not SDL Back.
static constexpr std::array host_gamepad_mappings{
    HostGamepadMapping{0x000008, ImGuiKey_GamepadStart},
    HostGamepadMapping{0x000002, ImGuiKey_GamepadL3},
    HostGamepadMapping{0x000004, ImGuiKey_GamepadR3},
    HostGamepadMapping{0x000010, ImGuiKey_GamepadDpadUp},
    HostGamepadMapping{0x000020, ImGuiKey_GamepadDpadRight},
    HostGamepadMapping{0x000040, ImGuiKey_GamepadDpadDown},
    HostGamepadMapping{0x000080, ImGuiKey_GamepadDpadLeft},
    HostGamepadMapping{0x000100, ImGuiKey_GamepadL2},
    HostGamepadMapping{0x000200, ImGuiKey_GamepadR2},
    HostGamepadMapping{0x000400, ImGuiKey_GamepadL1},
    HostGamepadMapping{0x000800, ImGuiKey_GamepadR1},
    HostGamepadMapping{0x001000, ImGuiKey_GamepadFaceUp},
    HostGamepadMapping{0x002000, ImGuiKey_GamepadFaceRight},
    HostGamepadMapping{0x004000, ImGuiKey_GamepadFaceDown},
};

static std::deque<HostGamepadEvent> host_gamepad_events{};
static std::mutex host_gamepad_mutex{};
static std::uint32_t host_gamepad_captured_buttons{};

struct HostTextInputEvent {
    std::string text{};
    ImGuiKey key{ImGuiKey_None};
    std::uint32_t repeat_count{};
};

// Text, deletion and submit must share one FIFO. Keeping text on one queue and Backspace/Enter on
// the gamepad queue reorders a quick "type, then delete" sequence whenever both arrive between two
// renderer frames: Backspace would run first and the just-deleted character would then be inserted.
static std::deque<HostTextInputEvent> host_text_input_events{};
static std::mutex host_text_input_mutex{};
// Written only by the renderer thread after the modal layers draw, read lock-free by Android's UI
// thread. Unlike HasModalLayer(), this says an actual InputText owns the ImGui active ID.
static std::atomic_bool host_text_input_active{};

// Large enough for the complete PS4 IME text buffer while still preventing an accidental Android
// Intent from growing the renderer-thread queue without bound.
static constexpr std::size_t MaxHostTextInputBytes = 16 * 1024;

static ImGuiID dock_id;

namespace ImGui {

namespace Core {

static bool HasModalLayer() {
    std::scoped_lock lock{change_layers_mutex};
    return !capturing_layers.empty();
}

static void DrainHostGamepadEvents() {
    std::deque<HostGamepadEvent> pending;
    {
        std::scoped_lock lock{host_gamepad_mutex};
        pending.swap(host_gamepad_events);
    }
    auto& io = GetIO();
    for (const auto& event : pending) {
        io.AddKeyEvent(event.key, event.pressed);
    }
}

static void DrainHostTextInputEvents() {
    std::deque<HostTextInputEvent> pending;
    {
        std::scoped_lock lock{host_text_input_mutex};
        pending.swap(host_text_input_events);
    }
    auto& io = GetIO();
    for (const auto& event : pending) {
        if (event.key == ImGuiKey_None) {
            io.AddInputCharactersUTF8(event.text.c_str());
            continue;
        }
        for (std::uint32_t i = 0; i < event.repeat_count; ++i) {
            io.AddKeyEvent(event.key, true);
            io.AddKeyEvent(event.key, false);
        }
    }
}

static bool QueueHostTextKey(ImGuiKey key, std::uint32_t repeat_count = 1) {
    if (!host_text_input_active.load(std::memory_order_acquire) || !HasModalLayer() ||
        repeat_count == 0 || repeat_count > MaxHostTextInputBytes) {
        return false;
    }
    std::scoped_lock lock{host_text_input_mutex};
    host_text_input_events.push_back({.key = key, .repeat_count = repeat_count});
    return true;
}

bool QueueHostTextInput(const char* utf8_text) {
    if (utf8_text == nullptr || utf8_text[0] == '\0' || !HasModalLayer()) {
        return false;
    }

    const std::size_t byte_count = std::strlen(utf8_text);
    if (byte_count == 0 || byte_count > MaxHostTextInputBytes) {
        return false;
    }

    std::scoped_lock lock{host_text_input_mutex};
    host_text_input_events.push_back({.text = std::string{utf8_text, byte_count}});
    return true;
}

std::uint32_t QueueHostGamepadButtons(std::uint32_t button_mask, bool pressed) {
    std::uint32_t mapped_mask = 0;
    for (const auto& mapping : host_gamepad_mappings) {
        mapped_mask |= mapping.mask;
    }
    mapped_mask &= button_mask;
    if (mapped_mask == 0) {
        return 0;
    }

    // A down event belongs to ImGui only while a modal layer is really present. A corresponding up
    // event remains captured after that layer closes, otherwise the closing Cross/Circle press leaks
    // into the first guest frame and leaves ImGui's key state stuck down.
    if (pressed && !HasModalLayer()) {
        return 0;
    }

    std::scoped_lock lock{host_gamepad_mutex};
    const std::uint32_t consumed = pressed ? mapped_mask
                                           : (mapped_mask & host_gamepad_captured_buttons);
    if (consumed == 0) {
        return 0;
    }
    if (pressed) {
        host_gamepad_captured_buttons |= consumed;
    } else {
        host_gamepad_captured_buttons &= ~consumed;
    }
    for (const auto& mapping : host_gamepad_mappings) {
        if ((consumed & mapping.mask) != 0) {
            host_gamepad_events.push_back({mapping.key, pressed});
        }
    }
    return consumed;
}

void Initialize(const ::Vulkan::Instance& instance, const Frontend::WindowSDL& window,
                const u32 image_count, vk::Format surface_format,
                const vk::AllocationCallbacks* allocator) {

    const auto config_path = GetUserPath(Common::FS::PathType::UserDir) / "imgui.ini";
    const auto log_path = GetUserPath(Common::FS::PathType::LogDir) / "imgui_log.txt";

    CreateContext();
    ImGuiIO& io = GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.DisplaySize = ImVec2((float)window.GetWidth(), (float)window.GetHeight());
    PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f); // Makes the window edges rounded

    auto path = config_path.u8string();
    char* config_file_buf = new char[path.size() + 1]();
    std::memcpy(config_file_buf, path.c_str(), path.size());
    io.IniFilename = config_file_buf;

    path = log_path.u8string();
    char* log_file_buf = new char[path.size() + 1]();
    std::memcpy(log_file_buf, path.c_str(), path.size());
    io.LogFilename = log_file_buf;

    // The embedded TTF fonts are generated as empty 1-byte stubs in the Android build
    // (the font-embedding step does not produce real data), so stbtt would assert on
    // invalid font data and abort. ImGui here is only the frame compositor — the PS4 menu
    // text is drawn by the guest, not ImGui — so any valid font works. Fall back to ImGui's
    // built-in default font on Android (or whenever the embedded data is a stub).
    if (imgui_font_notosansjp_regular_compressed_size <= 1) {
        io.FontDefault = io.Fonts->AddFontDefault();
    } else {
        ImFontGlyphRangesBuilder rb{};
        rb.AddRanges(io.Fonts->GetGlyphRangesDefault());
        rb.AddRanges(io.Fonts->GetGlyphRangesGreek());
        rb.AddRanges(io.Fonts->GetGlyphRangesKorean());
        rb.AddRanges(io.Fonts->GetGlyphRangesJapanese());
        rb.AddRanges(io.Fonts->GetGlyphRangesCyrillic());
        // Must persist until the font atlas is actually built (later), or ImGui reads freed memory and
        // asserts "is your glyph range array persistent?". On Android the stub-font path above is taken,
        // so this desktop (PC writer) path is where the dangling local bit. Make it static.
        static ImVector<ImWchar> ranges{};
        ranges.clear();
        rb.BuildRanges(&ranges);
        ImFontConfig font_cfg{};
        font_cfg.OversampleH = 2;
        font_cfg.OversampleV = 1;
        io.FontDefault = io.Fonts->AddFontFromMemoryCompressedTTF(
            imgui_font_notosansjp_regular_compressed_data,
            imgui_font_notosansjp_regular_compressed_size, 32.0f, &font_cfg, ranges.Data);
        io.Fonts->AddFontFromMemoryCompressedTTF(imgui_font_proggyvector_regular_compressed_data,
                                                 imgui_font_proggyvector_regular_compressed_size,
                                                 32.0f);
        io.Fonts->AddFontFromMemoryCompressedTTF(imgui_font_notosansjp_regular_compressed_data,
                                                 imgui_font_notosansjp_regular_compressed_size,
                                                 128.0f, &font_cfg, ranges.Data);
    }

    io.FontGlobalScale = 0.5f;

    StyleColorsDark();

    ::Core::Devtools::Layer::SetupSettings();
#if !defined(ANDROID)
    // Android has no real SDL_Window (GetSDLWindow() is null — the surface is a bridged
    // ANativeWindow). The ImGui SDL platform backend crashes on a null window, so we run
    // ImGui renderer-only on Android; input comes from the Android overlay instead.
    Sdl::Init(window.GetSDLWindow());
#endif

#if defined(ANDROID)
    // The Android overlay is a real gamepad input backend even though there is no SDL_Window.
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
#endif

    const Vulkan::InitInfo vk_info{
        .instance = instance.GetInstance(),
        .physical_device = instance.GetPhysicalDevice(),
        .device = instance.GetDevice(),
        .queue_family = instance.GetPresentQueueFamilyIndex(),
        .queue = instance.GetPresentQueue(),
        .image_count = image_count,
        .min_allocation_size = 1024 * 1024,
        .pipeline_rendering_create_info{
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &surface_format,
        },
        .allocator = allocator,
        .check_vk_result_fn = &CheckVkResult,
    };
    Vulkan::Init(vk_info);

    TextureManager::StartWorker();

    char label[32];
    ImFormatString(label, IM_ARRAYSIZE(label), "WindowOverViewport_%08X", GetMainViewport()->ID);
    dock_id = ImHashStr(label);

#if !defined(ANDROID)
    if (const auto dpi = SDL_GetWindowDisplayScale(window.GetSDLWindow()); dpi > 0.0f) {
        GetIO().FontGlobalScale *= dpi;
    }
#endif

    std::at_quick_exit([] { SaveIniSettingsToDisk(GetIO().IniFilename); });
}

void OnResize() {
#if !defined(ANDROID)
    Sdl::OnResize();
#endif
}

void OnSurfaceFormatChange(vk::Format surface_format) {
    Vulkan::OnSurfaceFormatChange(surface_format);
}

void Shutdown(const vk::Device& device) {
    host_text_input_active.store(false, std::memory_order_release);
    auto result = device.waitIdle();
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(ImGui, "Failed to wait for Vulkan device idle on shutdown: {}",
                    vk::to_string(result));
    }

    TextureManager::StopWorker();

    const ImGuiIO& io = GetIO();
    const auto ini_filename = (void*)io.IniFilename;
    const auto log_filename = (void*)io.LogFilename;

    Vulkan::Shutdown();
#if !defined(ANDROID)
    Sdl::Shutdown();
#endif
    DestroyContext();

    delete[] (char*)ini_filename;
    delete[] (char*)log_filename;
}

bool ProcessEvent(SDL_Event* event) {
#if !defined(ANDROID)
    Sdl::ProcessEvent(event);
#endif
    switch (event->type) {
    // Don't block release/up events
    case SDL_EVENT_MOUSE_MOTION:
    case SDL_EVENT_MOUSE_WHEEL:
    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
        const auto& io = GetIO();
        return io.WantCaptureMouse && io.Ctx->NavWindow != nullptr &&
               (io.Ctx->NavWindow->Flags & ImGuiWindowFlags_NoNav) == 0;
    }
    case SDL_EVENT_TEXT_INPUT:
    case SDL_EVENT_KEY_DOWN: {
        const auto& io = GetIO();
        return io.WantCaptureKeyboard && io.Ctx->NavWindow != nullptr &&
               io.Ctx->NavWindow->ID != dock_id;
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION: {
        const auto& io = GetIO();
        return io.NavActive && io.Ctx->NavWindow != nullptr && io.Ctx->NavWindow->ID != dock_id;
    }
    default:
        return false;
    }
}

ImGuiID NewFrame(bool is_reusing_frame) {
    {
        std::scoped_lock lock{change_layers_mutex};
        while (!change_layers.empty()) {
            const auto [to_be_added, layer] = change_layers.front();
            if (to_be_added) {
                layers.push_back(layer);
            } else {
                const auto [begin, end] = std::ranges::remove(layers, layer);
                layers.erase(begin, end);
            }
            change_layers.pop_front();
        }
    }

#if defined(ANDROID)
    // No SDL platform backend on Android — feed ImGui the minimum it needs (valid display
    // size + delta time) so ImGui::NewFrame() does not assert/crash.
    {
        ImGuiIO& io = GetIO();
        if (io.DisplaySize.x <= 0.0f || io.DisplaySize.y <= 0.0f) {
            io.DisplaySize = ImVec2(1920.0f, 1080.0f);
        }
        io.DeltaTime = 1.0f / 60.0f;
    }
#else
    Sdl::NewFrame(is_reusing_frame);
#endif
    DrainHostGamepadEvents();
    DrainHostTextInputEvents();
    ImGui::NewFrame();

    ImGuiWindowFlags flags =
        ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar;
    if (!DebugState.IsShowingDebugMenuBar()) {
        flags |= ImGuiDockNodeFlags_NoTabBar;
    }
    ImGuiID dockId = DockSpaceOverViewport(0, GetMainViewport(), flags);

    for (auto* layer : layers) {
        layer->Draw();
    }

    const ImGuiContext* context = GetCurrentContext();
    // ActiveId can survive until the frame after a dialog removes its layer. Gate it with the
    // synchronously maintained modal set so Android never reopens its keyboard for that stale ID.
    const bool input_text_active = context != nullptr && context->ActiveId != 0 &&
                                   context->InputTextState.ID == context->ActiveId && HasModalLayer();
    host_text_input_active.store(input_text_active, std::memory_order_release);

    return dockId;
}

void Render(const vk::CommandBuffer& cmdbuf, const vk::ImageView& image_view,
            const vk::Extent2D& extent) {
    ImGui::Render();
    ImDrawData* draw_data = GetDrawData();
    if (draw_data->CmdListsCount == 0) {
        return;
    }

    if (Config::getVkHostMarkersEnabled()) {
        cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
            .pLabelName = "ImGui Render",
        });
    }

    vk::RenderingAttachmentInfo color_attachments[1]{
        {
            .imageView = image_view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
        },
    };
    vk::RenderingInfo render_info{};
    render_info.renderArea = vk::Rect2D{
        .offset = {0, 0},
        .extent = extent,
    };
    render_info.layerCount = 1;
    render_info.colorAttachmentCount = 1;
    render_info.pColorAttachments = color_attachments;
    cmdbuf.beginRendering(render_info);
    Vulkan::RenderDrawData(*draw_data, cmdbuf);
    cmdbuf.endRendering();
    if (Config::getVkHostMarkersEnabled()) {
        cmdbuf.endDebugUtilsLabelEXT();
    }
}

bool MustKeepDrawing() {
    bool modal_or_pending = false;
    {
        std::scoped_lock lock{change_layers_mutex};
        // A single pending Add/Remove is enough to require another ImGui frame. The previous
        // size()>1 check missed the common Devtools+one-dialog transition and also raced AddLayer.
        modal_or_pending = !capturing_layers.empty() || !change_layers.empty();
    }
    return modal_or_pending || DebugState.IsShowingDebugMenuBar();
}

} // namespace Core

void Layer::AddLayer(Layer* layer) {
    std::scoped_lock lock{change_layers_mutex};
    if (layer != nullptr && layer->CapturesGamepadInput()) {
        capturing_layers.insert(layer);
    }
    change_layers.emplace_back(true, layer);
}

void Layer::RemoveLayer(Layer* layer) {
    std::scoped_lock lock{change_layers_mutex};
    capturing_layers.erase(layer);
    if (capturing_layers.empty()) {
        // Do not leave Android's keyboard visible while waiting for the next renderer frame.
        host_text_input_active.store(false, std::memory_order_release);
    }
    change_layers.emplace_back(false, layer);
}

} // namespace ImGui

// Optional frontend ABI. It returns the exact subset of the Orbis button mask captured by an
// active ImGui modal; callers forward every unconsumed bit to the guest pad path unchanged.
extern "C" std::uint32_t executor_lsx4_runtime_system_ui_gamepad_button(
    std::uint32_t button_mask, int pressed) {
    return ImGui::Core::QueueHostGamepadButtons(button_mask, pressed != 0);
}

// Optional Android/frontend ABI. Text is queued, never applied directly from the Java UI thread;
// Dear ImGui consumes it on the renderer thread at the next NewFrame. Returning zero tells the
// frontend that no modal system UI owns the text and normal Android dispatch should continue.
extern "C" int executor_lsx4_runtime_system_ui_text_input(const char* utf8_text) {
    return ImGui::Core::QueueHostTextInput(utf8_text) ? 1 : 0;
}

// Lock-free frontend query used to drive Android's soft keyboard. This deliberately reports an
// active InputText, not merely an open modal, so message dialogs never summon the keyboard.
extern "C" int executor_lsx4_runtime_system_ui_text_active() {
    return host_text_input_active.load(std::memory_order_acquire) ? 1 : 0;
}

extern "C" int executor_lsx4_runtime_system_ui_text_submit() {
    return ImGui::Core::QueueHostTextKey(ImGuiKey_Enter) ? 1 : 0;
}

extern "C" int executor_lsx4_runtime_system_ui_text_backspace(std::uint32_t count) {
    return ImGui::Core::QueueHostTextKey(ImGuiKey_Backspace, count) ? 1 : 0;
}
