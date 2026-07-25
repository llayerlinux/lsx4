# SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

set(BUILD_SHARED_LIBS OFF)
set(BUILD_TESTING OFF)
set_directory_properties(PROPERTIES
    EXCLUDE_FROM_ALL ON
    SYSTEM ON
)

set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
endif()

if (NOT TARGET Boost::headers)
    add_subdirectory(ext-boost)
endif()

if (NOT TARGET fmt::fmt)
    add_subdirectory(fmt)
endif()

if (EXECUTOR_ANDROID_NATIVE_CORE_PROBE_ENABLE_AVPLAYER AND
    NOT TARGET FFmpeg::ffmpeg)
    set(LSX4_ANDROID_FFMPEG_ROOT
        "${CMAKE_SOURCE_DIR}/build/lsx4-ffmpeg-android-arm64-pic"
        CACHE PATH "Android arm64 PIC FFmpeg installation")
    set(LSX4_ANDROID_FFMPEG_LIBRARIES
        avformat avcodec swscale avutil avfilter swresample)
    if (NOT EXISTS "${LSX4_ANDROID_FFMPEG_ROOT}/include/libavcodec/avcodec.h")
        message(FATAL_ERROR
            "Android PIC FFmpeg is missing at ${LSX4_ANDROID_FFMPEG_ROOT}. "
            "Run scripts/build-ffmpeg-android-pic.ps1 before configuring.")
    endif()
    add_library(ffmpeg INTERFACE)
    target_include_directories(ffmpeg INTERFACE "${LSX4_ANDROID_FFMPEG_ROOT}/include")
    foreach (library IN LISTS LSX4_ANDROID_FFMPEG_LIBRARIES)
        set(archive "${LSX4_ANDROID_FFMPEG_ROOT}/lib/lib${library}.a")
        if (NOT EXISTS "${archive}")
            message(FATAL_ERROR "Android PIC FFmpeg archive is missing: ${archive}")
        endif()
        target_link_libraries(ffmpeg INTERFACE "${archive}")
    endforeach()
    add_library(FFmpeg::ffmpeg ALIAS ffmpeg)
endif()

file(GLOB LIBATRAC9_SOURCES LibAtrac9/C/src/*.c)
list(FILTER LIBATRAC9_SOURCES EXCLUDE REGEX "LibAtrac9/C/src/bit_reader\\.c$")
list(APPEND LIBATRAC9_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/lsx4-overrides/LibAtrac9/bit_reader.c")
add_library(LibAtrac9 STATIC ${LIBATRAC9_SOURCES})
target_include_directories(LibAtrac9 PUBLIC LibAtrac9/C/src)

if (NOT TARGET ZLIB::ZLIB)
    set(ZLIB_ENABLE_TESTS OFF)
    set(WITH_GTEST OFF)
    set(WITH_NEW_STRATEGIES ON)
    set(WITH_NATIVE_INSTRUCTIONS ON)
    set(ZLIB_COMPAT ON CACHE BOOL "" FORCE)
    include(FetchContent)
    FetchContent_Declare(
        ZLIB
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/zlib-ng"
        OVERRIDE_FIND_PACKAGE
    )
    FetchContent_MakeAvailable(ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlib)
    set(ZLIB_INCLUDE_DIRS "${FETCHCONTENT_BASE_DIR}/zlib-build")
endif()

if (NOT TARGET SDL3::SDL3)
    set(SDL_TEST_LIBRARY OFF)
    set(SDL_PIPEWIRE OFF)
    add_subdirectory(sdl3)
endif()

if (NOT TARGET SDL3_mixer::SDL3_mixer)
    set(SDLMIXER_FLAC_LIBFLAC OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_FLAC_DRFLAC OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_GME OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_MOD_XMP OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_MP3_DRMP3 OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_MP3_MPG123 OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_MIDI_FLUIDSYNTH OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_MIDI_TIMIDITY OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_OPUS OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_VORBIS_STB OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_VORBIS_VORBISFILE OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_VORBIS_TREMOR OFF CACHE BOOL "" FORCE)
    set(SDLMIXER_WAVPACK OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory(sdl3_mixer)
endif()

if (NOT TARGET Vulkan::Headers)
    set(VULKAN_HEADERS_ENABLE_MODULE OFF)
    add_subdirectory(vulkan-headers)
endif()

if (NOT TARGET GPUOpen::VulkanMemoryAllocator)
    add_subdirectory(vma)
endif()

if (NOT TARGET RenderDoc::API)
    add_library(renderdoc INTERFACE)
    target_include_directories(renderdoc INTERFACE ./renderdoc)
    add_library(RenderDoc::API ALIAS renderdoc)
endif()

if (NOT TARGET glslang::glslang)
    set(SKIP_GLSLANG_INSTALL ON CACHE BOOL "")
    set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "")
    set(ENABLE_SPVREMAPPER OFF CACHE BOOL "")
    set(ENABLE_CTEST OFF CACHE BOOL "")
    set(ENABLE_HLSL OFF CACHE BOOL "")
    set(BUILD_EXTERNAL OFF CACHE BOOL "")
    set(ENABLE_OPT OFF CACHE BOOL "")
    add_subdirectory(glslang)
    file(COPY glslang/SPIRV DESTINATION glslang/glslang FILES_MATCHING PATTERN "*.h")
    target_include_directories(glslang INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/glslang")
endif()

if (NOT TARGET tsl::robin_map)
    add_subdirectory(robin-map)
endif()

if (NOT TARGET xbyak::xbyak)
    add_subdirectory(xbyak)
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/xbyak_aarch64/CMakeLists.txt" AND
    NOT TARGET xbyak_aarch64::xbyak_aarch64)
    set(SAMPLE OFF CACHE BOOL "" FORCE)
    set(TEST OFF CACHE BOOL "" FORCE)
    set(SHARED OFF CACHE BOOL "" FORCE)
    add_subdirectory(xbyak_aarch64)
endif()

if (NOT TARGET magic_enum::magic_enum)
    add_subdirectory(magic_enum)
endif()

if (NOT TARGET toml11::toml11)
    add_subdirectory(toml11)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        get_target_property(_toml11_compile_options toml11 INTERFACE_COMPILE_OPTIONS)
        list(REMOVE_ITEM _toml11_compile_options "/Zc:preprocessor")
        set_target_properties(toml11 PROPERTIES INTERFACE_COMPILE_OPTIONS ${_toml11_compile_options})
      endif()
    endif()
endif()

if (NOT TARGET xxHash::xxhash)
    add_library(xxhash xxhash/xxhash.h xxhash/xxhash.c)
    target_include_directories(xxhash PUBLIC xxhash)
    add_library(xxHash::xxhash ALIAS xxhash)
endif()

add_subdirectory(sirit)
if (WIN32)
    target_compile_options(sirit PRIVATE "-Wno-error=unused-command-line-argument")
endif()

if (NOT TARGET half::half)
    add_library(half INTERFACE)
    target_include_directories(half INTERFACE half/include)
    add_library(half::half ALIAS half)
endif()

if (NOT TARGET PNG::PNG)
    set(PNG_SHARED OFF CACHE BOOL "" FORCE)
    set(PNG_STATIC ON CACHE BOOL "" FORCE)
    set(PNG_TESTS OFF CACHE BOOL "" FORCE)
    set(PNG_TOOLS OFF CACHE BOOL "" FORCE)
    set(SKIP_INSTALL_ALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(libpng)
    add_library(PNG::PNG ALIAS png_static)
endif()

add_library(Dear_ImGui
        dear_imgui/imgui.cpp
        dear_imgui/imgui_demo.cpp
        dear_imgui/imgui_draw.cpp
        dear_imgui/imgui_internal.h
        dear_imgui/imgui_tables.cpp
        dear_imgui/imgui_widgets.cpp
)
target_include_directories(Dear_ImGui INTERFACE dear_imgui/)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    option(TRACY_ENABLE "" OFF)
else()
    option(TRACY_ENABLE "" ON)
endif()
option(TRACY_NO_CRASH_HANDLER "" ON) # Otherwise texture cache exceptions will be treaten as a crash
option(TRACY_ON_DEMAND "" ON)
option(TRACY_NO_FRAME_IMAGE "" ON)
option(TRACY_FIBERS "" OFF) # For AmdGpu frontend profiling, disabled due to instability
option(TRACY_NO_SYSTEM_TRACING "" ON)
option(TRACY_NO_CALLSTACK "" ON)
option(TRACY_NO_CODE_TRANSFER "" ON)
option(TRACY_NO_SAMPLING "" ON)
option(TRACY_ONLY_LOCALHOST "" ON)
option(TRACY_NO_CONTEXT_SWITCH "" ON)
add_subdirectory(tracy)

if (NOT TARGET pugixml::pugixml)
    option(PUGIXML_NO_EXCEPTIONS "" ON)
    add_subdirectory(pugixml)
endif()

add_subdirectory(gcn)

if (NOT TARGET stb::headers)
    add_library(stb INTERFACE)
    target_include_directories(stb INTERFACE stb)
    add_library(stb::headers ALIAS stb)
endif()

add_subdirectory(aacdec)

if (NOT TARGET nlohmann_json::nlohmann_json)
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(json)
endif()

if (NOT TARGET miniz::miniz)
add_subdirectory(miniz)
add_library(miniz::miniz ALIAS miniz)
endif()

if (NOT TARGET CLI11::CLI11)
    set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(CLI11)
endif()


if (NOT TARGET OpenAL::OpenAL)
set(ALSOFT_ENABLE_MODULES OFF CACHE BOOL "" FORCE)
set(LIBTYPE "STATIC" CACHE STRING "" FORCE)
set(ALSOFT_UTILS OFF CACHE BOOL "" FORCE)
set(ALSOFT_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ALSOFT_TESTS OFF CACHE BOOL "" FORCE)
set(ALSOFT_INSTALL OFF CACHE BOOL "" FORCE)
set(ALSOFT_CONFIG OFF CACHE BOOL "" FORCE)

if (WIN32)
    set(ALSOFT_BACKEND_WASAPI ON CACHE BOOL "" FORCE)
    set(ALSOFT_BACKEND_DSOUND OFF CACHE BOOL "" FORCE)
    set(ALSOFT_BACKEND_WINMM OFF CACHE BOOL "" FORCE)
elseif (APPLE)
    set(ALSOFT_BACKEND_COREAUDIO ON CACHE BOOL "" FORCE)
else()
    set(ALSOFT_BACKEND_ALSA ON CACHE BOOL "" FORCE)
    set(ALSOFT_BACKEND_PULSEAUDIO ON CACHE BOOL "" FORCE)
    set(ALSOFT_BACKEND_PIPEWIRE OFF CACHE BOOL "" FORCE)
endif()
set(ALSOFT_BACKEND_NULL ON CACHE BOOL "" FORCE)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(openal-soft EXCLUDE_FROM_ALL)
endif()

add_library(Cpp_Httplib INTERFACE)
target_include_directories(Cpp_Httplib INTERFACE cpp-httplib/)

