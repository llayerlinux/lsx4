# SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(lsx4_android_runtime CXX C ASM)

if (NOT ANDROID)
    message(FATAL_ERROR "LSX4 native runtime must be configured with the Android NDK toolchain")
endif()

set(LSX4_ARM_DESKTOP_LAYER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/funnel-arm" CACHE PATH
    "Funnel ARM desktop layer source directory")
option(LSX4_EXTERNAL_FUNNEL_BASELINE_COMPAT
       "Build the runtime against a historical Funnel tree for regression bisection"
       OFF)
set(LSX4_ARM_DESKTOP_LAYER_MANIFEST
    "${LSX4_ARM_DESKTOP_LAYER_DIR}/cmake/lsx4_arm_desktop_layer.cmake")
if(NOT EXISTS "${LSX4_ARM_DESKTOP_LAYER_MANIFEST}")
    message(FATAL_ERROR
        "Funnel ARM desktop layer is unavailable; initialize the funnel-arm submodule")
endif()
include("${LSX4_ARM_DESKTOP_LAYER_MANIFEST}")

if(UNIX AND NOT APPLE)
    set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

    include(CheckPIESupported)
    check_pie_supported(OUTPUT_VARIABLE pie_check LANGUAGES C CXX)
    if(NOT CMAKE_C_LINK_PIE_SUPPORTED OR NOT CMAKE_CXX_LINK_PIE_SUPPORTED)
        message(WARNING "PIE is not supported at link time: ${pie_check}")
    endif()
endif()

set(ENABLE_DISCORD_RPC OFF)
set(ENABLE_UPDATER OFF)
option(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB "Disable desktop FFmpeg/libusb fronts for the Android native-core probe" ON)
option(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_ENABLE_AVPLAYER "Enable the real FFmpeg-backed AvPlayer in the Android native-core probe" ON)
option(LSX4_ENABLE_PS5_DESKTOP_PATH
       "Build the isolated Funnel PS5 desktop boundary (never part of the PS4 runtime)"
       OFF)

if(LSX4_ENABLE_PS5_DESKTOP_PATH)
    set(LSX4_PS5_DESKTOP_LAYER_MANIFEST
        "${LSX4_ARM_DESKTOP_LAYER_DIR}/cmake/lsx4_ps5_desktop_layer.cmake")
    if(NOT EXISTS "${LSX4_PS5_DESKTOP_LAYER_MANIFEST}")
        message(FATAL_ERROR
            "Funnel PS5 desktop layer is unavailable: ${LSX4_PS5_DESKTOP_LAYER_MANIFEST}")
    endif()
    include("${LSX4_PS5_DESKTOP_LAYER_MANIFEST}")
endif()

if (APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(BASE_ARCHITECTURE "${CMAKE_OSX_ARCHITECTURES}")
elseif (CMAKE_SYSTEM_PROCESSOR)
    set(BASE_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}")
else()
    set(BASE_ARCHITECTURE "${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()

if (BASE_ARCHITECTURE MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(ARCHITECTURE "x86_64")
elseif (BASE_ARCHITECTURE MATCHES "(aarch64)|(AARCH64)|(arm64)|(ARM64)")
    set(ARCHITECTURE "arm64")
else()
    message(FATAL_ERROR "Unsupported CPU architecture: ${BASE_ARCHITECTURE}")
endif()

if (ARCHITECTURE STREQUAL "x86_64")
    add_compile_options(-march=x86-64-v3)
endif()

if (APPLE AND ARCHITECTURE STREQUAL "x86_64" AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64")
    list(APPEND CMAKE_IGNORE_PREFIX_PATH "/opt/homebrew")

    set(ENV{PKG_CONFIG_DIR} "")
    set(ENV{PKG_CONFIG_LIBDIR} "${CMAKE_SYSROOT}/usr/lib/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig:${CMAKE_SYSROOT}/usr/local/lib/pkgconfig:${CMAKE_SYSROOT}/usr/local/share/pkgconfig")
    set(ENV{PKG_CONFIG_SYSROOT_DIR} ${CMAKE_SYSROOT})
endif()

function(create_target_directory_groups target_name)

    source_group("Other Files" REGULAR_EXPRESSION ".")

    get_target_property(target_sources "${target_name}" SOURCES)

    foreach(file_name IN LISTS target_sources)
        get_filename_component(dir_name "${file_name}" PATH)
        string(REPLACE "/" "\\" group_name "${dir_name}")
        source_group("${group_name}" FILES "${file_name}")
    endforeach()
endfunction()

if (CLANG_FORMAT)
    set(SRCS ${PROJECT_SOURCE_DIR}/src)
    set(CCOMMENT "Running clang format against all the .h and .cpp files in src/")
    if (WIN32)
        add_custom_target(clang-format
            COMMAND powershell.exe -Command "Get-ChildItem '${SRCS}/*' -Include *.cpp,*.h,*.mm -Recurse | Foreach {&'${CLANG_FORMAT}' -i $_.fullname}"
            COMMENT ${CCOMMENT})
    else()
        add_custom_target(clang-format
            COMMAND find ${SRCS} -iname *.h -o -iname *.cpp -o -iname *.mm | xargs ${CLANG_FORMAT} -i
            COMMENT ${CCOMMENT})
    endif()
    unset(SRCS)
    unset(CCOMMENT)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/externals/cmake-modules/GetGitRevisionDescription.cmake")
get_git_head_revision(GIT_REF_SPEC GIT_REV)
git_describe(GIT_DESC --always --long --dirty)
git_branch_name(GIT_BRANCH)
string(TIMESTAMP BUILD_DATE "%Y-%m-%d %H:%M:%S")

message("start git things")

message("check for remote and branch")
execute_process(
  COMMAND git rev-parse --abbrev-ref --symbolic-full-name @{u}
  OUTPUT_VARIABLE GIT_REMOTE_NAME
  RESULT_VARIABLE GIT_REMOTE_RESULT
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (GIT_REMOTE_RESULT OR GIT_REMOTE_NAME STREQUAL "")
  message(STATUS "check default push")
  execute_process(
    COMMAND git config --get remote.pushDefault
    OUTPUT_VARIABLE GIT_REMOTE_NAME
    RESULT_VARIABLE GIT_REMOTE_RESULT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "got remote: ${GIT_REMOTE_NAME}")
endif()

if (GIT_REMOTE_RESULT OR GIT_REMOTE_NAME STREQUAL "")
  message(STATUS "check github")
  set(GIT_REMOTE_NAME "origin")

  if (DEFINED ENV{GITHUB_HEAD_REF} AND NOT "$ENV{GITHUB_HEAD_REF}" STREQUAL "")
    message(STATUS "github head ref: $ENV{GITHUB_HEAD_REF}")
    set(GITHUB_HEAD_REF "$ENV{GITHUB_HEAD_REF}")
  else()
    set(GITHUB_HEAD_REF "")
  endif()

  if (DEFINED ENV{GITHUB_REF} AND NOT "$ENV{GITHUB_REF}" STREQUAL "")
    message(STATUS "github ref: $ENV{GITHUB_REF}")
    string(REGEX REPLACE "^refs/[^/]*/" "" GITHUB_BRANCH "$ENV{GITHUB_REF}")
    string(REGEX MATCH "refs/pull/([0-9]+)/merge" MATCHED_REF "$ENV{GITHUB_REF}")
    if (MATCHED_REF)
      set(PR_NUMBER "${CMAKE_MATCH_1}")
      set(GITHUB_BRANCH "")
      message(STATUS "PR number: ${PR_NUMBER}")
    else()
      set(PR_NUMBER "")
    endif()
  else()
    set(GITHUB_BRANCH "")
    set(PR_NUMBER "")
  endif()

  if (NOT "${PR_NUMBER}" STREQUAL "" AND NOT "${GITHUB_HEAD_REF}" STREQUAL "")
    set(GIT_BRANCH "pr-${PR_NUMBER}-${GITHUB_HEAD_REF}")
  elseif (NOT "${PR_NUMBER}" STREQUAL "" AND NOT "${GITHUB_BRANCH}" STREQUAL "")
    set(GIT_BRANCH "pr-${PR_NUMBER}-${GITHUB_BRANCH}")
  elseif (NOT "${PR_NUMBER}" STREQUAL "")
    set(GIT_BRANCH "pr-${PR_NUMBER}")
  elseif ("${PR_NUMBER}" STREQUAL "" AND NOT "${GITHUB_HEAD_REF}" STREQUAL "")
    set(GIT_BRANCH "${GITHUB_HEAD_REF}")
  elseif ("${PR_NUMBER}" STREQUAL "" AND NOT "${GITHUB_BRANCH}" STREQUAL "")
    set(GIT_BRANCH "${GITHUB_BRANCH}")
  elseif ("${PR_NUMBER}" STREQUAL "" AND NOT "${GITHUB_REF}" STREQUAL "")
    set(GIT_BRANCH "${GITHUB_REF}")
  elseif("${GIT_BRANCH}" STREQUAL "")
    message(STATUS "couldn't find branch")
    set(GIT_BRANCH "detached-head")
  endif()
else()
  string(FIND "${GIT_REMOTE_NAME}" "/" INDEX)
  if (INDEX GREATER -1)
    string(SUBSTRING "${GIT_REMOTE_NAME}" 0 "${INDEX}" GIT_REMOTE_NAME)
  elseif("${GIT_REMOTE_NAME}" STREQUAL "")
    message(STATUS "reset to origin")
    set(GIT_REMOTE_NAME "origin")
  endif()
endif()

message(STATUS "getting remote link")
execute_process(
  COMMAND git config --get remote.${GIT_REMOTE_NAME}.url
  OUTPUT_VARIABLE GIT_REMOTE_URL
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(EMULATOR_VERSION_MAJOR "0")
set(EMULATOR_VERSION_MINOR "15")
set(EMULATOR_VERSION_PATCH "0")

set(APP_VERSION "${EMULATOR_VERSION_MAJOR}.${EMULATOR_VERSION_MINOR}.${EMULATOR_VERSION_PATCH}")
set(APP_IS_RELEASE true)
configure_file("${LSX4_ARM_DESKTOP_SOURCE_DIR}/common/scm_rev.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/src/common/scm_rev.cpp" @ONLY)

message("-- end git things, remote: ${GIT_REMOTE_NAME}, branch: ${GIT_BRANCH}, link: ${GIT_REMOTE_URL}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Boost 1.84.0 CONFIG)
find_package(CLI11 2.6.1 CONFIG)
find_package(FFmpeg 5.1.2 MODULE)
find_package(fmt 10.2.0 CONFIG)
find_package(glslang 15 CONFIG)
find_package(half 1.12.0 MODULE)
find_package(magic_enum 0.9.7 CONFIG)
find_package(miniz 3.1 CONFIG)
find_package(nlohmann_json 3.12 CONFIG)
find_package(PNG 1.6 MODULE)
find_package(OpenAL CONFIG)
find_package(RenderDoc 1.6.0 MODULE)
find_package(SDL3_mixer 2.8.1 CONFIG)
if (SDL3_mixer_FOUND)
    find_package(SDL3 3.1.2 CONFIG)
endif()
find_package(stb MODULE)
find_package(toml11 4.2.0 CONFIG)
find_package(tsl-robin-map 1.3.0 CONFIG)
find_package(VulkanHeaders 1.4.329 CONFIG)
find_package(VulkanMemoryAllocator 3.1.0 CONFIG)
find_package(xbyak 7.07 CONFIG)
find_package(xxHash 0.8.2 MODULE)
find_package(ZLIB 1.3 MODULE)
find_package(pugixml 1.14 CONFIG)
list(POP_BACK CMAKE_MODULE_PATH)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(_LIBCPP_VERSION version LIBCPP)
    if(LIBCPP)
        add_compile_options(-fexperimental-library)
    endif()
endif()

add_subdirectory(externals)
include_directories(BEFORE
    "${LSX4_ARM_DESKTOP_SOURCE_DIR}"
    "${LSX4_ARM_DESKTOP_LAYER_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(LSX_ICED_MANIFEST
    "${LSX4_ARM_DESKTOP_SOURCE_DIR}/common/iced_x86_ffi/Cargo.toml")
set(LSX_ICED_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo")
file(GLOB_RECURSE LSX_ICED_UPSTREAM_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/externals/iced-x86/src/rust/iced-x86/src/*.rs")
if(ANDROID)
    if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
        message(FATAL_ERROR "The iced-x86 decoder bridge currently supports Android arm64-v8a")
    endif()
    set(LSX_ICED_RUST_TARGET "aarch64-linux-android")
    set(LSX_ICED_TARGET_ARGUMENTS --target "${LSX_ICED_RUST_TARGET}")
    set(LSX_ICED_LIBRARY
        "${LSX_ICED_TARGET_DIR}/${LSX_ICED_RUST_TARGET}/release/liblsx_iced_decoder.a")
elseif(WIN32)
    set(LSX_ICED_TARGET_ARGUMENTS)
    set(LSX_ICED_LIBRARY "${LSX_ICED_TARGET_DIR}/release/lsx_iced_decoder.lib")
else()
    set(LSX_ICED_TARGET_ARGUMENTS)
    set(LSX_ICED_LIBRARY "${LSX_ICED_TARGET_DIR}/release/liblsx_iced_decoder.a")
endif()
add_custom_command(
    OUTPUT "${LSX_ICED_LIBRARY}"
    COMMAND "${CMAKE_COMMAND}" -E env
            "CARGO_TARGET_DIR=${LSX_ICED_TARGET_DIR}"
            "RUSTFLAGS=-A unexpected_cfgs -A unused_attributes"
            "${CARGO_EXECUTABLE}" build
            --manifest-path "${LSX_ICED_MANIFEST}"
            --locked --release ${LSX_ICED_TARGET_ARGUMENTS}
    DEPENDS
        "${LSX_ICED_MANIFEST}"
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/common/iced_x86_ffi/Cargo.lock"
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/common/iced_x86_ffi/src/lib.rs"
        "${CMAKE_CURRENT_SOURCE_DIR}/externals/iced-x86/src/rust/iced-x86/Cargo.toml"
        ${LSX_ICED_UPSTREAM_SOURCES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building the iced-x86 decoder bridge"
    VERBATIM
)
add_custom_target(lsx_iced_decoder_build DEPENDS "${LSX_ICED_LIBRARY}")
add_library(lsx_iced_decoder STATIC IMPORTED GLOBAL)
set_target_properties(lsx_iced_decoder PROPERTIES IMPORTED_LOCATION "${LSX_ICED_LIBRARY}")
add_dependencies(lsx_iced_decoder lsx_iced_decoder_build)


lsx4_define_android_desktop_profile()
list(APPEND LSX4_ARM_DESKTOP_SOURCES
    "${CMAKE_CURRENT_BINARY_DIR}/src/common/scm_rev.cpp")
set(LSX4_NATIVE_CORE_SOURCES ${LSX4_ARM_DESKTOP_SOURCES})
lsx4_assert_arm_desktop_ownership(${LSX4_NATIVE_CORE_SOURCES})






set(LSX4_JIT_COMMON_SOURCES
    src/executor/dynamic_translation/address_resolver.cpp
    src/executor/dynamic_translation/native_address_emitter.cpp
    src/executor/dynamic_translation/atomic_memory.cpp
    src/executor/dynamic_translation/code_witness.cpp
    src/executor/dynamic_translation/condition_materializer.cpp
    src/executor/dynamic_translation/code_memory.cpp
    src/executor/dynamic_translation/decode_frontend.cpp
    src/executor/dynamic_translation/diagnostic_probe_policy.cpp
    src/executor/dynamic_translation/edge_portal.cpp
    src/executor/dynamic_translation/execution_context.cpp
    src/executor/dynamic_translation/execution_driver.cpp
    src/executor/dynamic_translation/live_state_port.cpp
    src/executor/dynamic_translation/process_memory.cpp
    src/executor/dynamic_translation/trace_memory.cpp
    src/executor/dynamic_translation/stack_windows.cpp
    src/executor/dynamic_translation/foundation_checks.cpp
    src/executor/dynamic_translation/floating_relation.cpp
    src/executor/dynamic_translation/guest_capability_profile.cpp
    src/executor/dynamic_translation/guest_mapping_policy.cpp
    src/executor/dynamic_translation/host_isa_profile.cpp
    src/executor/dynamic_translation/hle_invocation_frame.cpp
    src/executor/dynamic_translation/instruction_diagnostics.cpp
    src/executor/dynamic_translation/integer_math.cpp
    src/executor/dynamic_translation/json_text.cpp
    src/executor/dynamic_translation/machine_state.cpp
    src/executor/dynamic_translation/native_encoder.cpp
    src/executor/dynamic_translation/native_status_emitter.cpp
    src/executor/dynamic_translation/native_compiler.cpp
    src/executor/dynamic_translation/operation_ir.cpp
    src/executor/dynamic_translation/region_plan.cpp
    src/executor/dynamic_translation/register_access.cpp
    src/executor/dynamic_translation/region_registry.cpp
    src/executor/dynamic_translation/scalar_ir_builder.cpp
    src/executor/dynamic_translation/scalar_semantics.cpp
    src/executor/dynamic_translation/translation_service.cpp
    src/executor/dynamic_translation/value_residency.cpp
    src/executor/dynamic_translation/vector_semantics.cpp
    src/executor/dynamic_translation/x87_semantics.cpp
    src/executor/dynamic_translation/x87_state.cpp
    src/executor/dynamic_translation/retiring_execution_core.cpp
    src/executor/dynamic_translation/jit_aarch64_stubs.S
    src/executor/dynamic_translation/artifact_repository.cpp
    src/executor/dynamic_translation/decode_bridge.cpp
    src/executor/dynamic_translation/vector_lowering_policy.cpp
    src/executor/dynamic_translation/packed_lane_ops.cpp
    src/executor/dynamic_translation/partial_move_plan.cpp
    src/executor/dynamic_translation/partial_move_emitter.cpp
)

add_library(lsx4_executor_android SHARED
    ${LSX4_NATIVE_CORE_SOURCES}
    ${LSX4_JIT_COMMON_SOURCES}
    src/executor/android_performance_hint.cpp
    src/executor/native_runtime_api.cpp
    src/executor/ps4_jit_platform.cpp
    src/executor/runtime_death_catcher.cpp
)

if(LSX4_EXTERNAL_FUNNEL_BASELINE_COMPAT)
    target_compile_definitions(lsx4_executor_android PRIVATE
        LSX4_EXTERNAL_FUNNEL_BASELINE_COMPAT=1)
endif()

if(LSX4_ENABLE_PS5_DESKTOP_PATH)
    lsx4_define_ps5_desktop_profile()
    lsx4_assert_ps5_desktop_ownership(${LSX4_PS5_DESKTOP_SOURCES})

    set(LSX4_KYTY_GEN5_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../_tmp_kytyps5_analysis"
        CACHE PATH "Kyty source tree used by the isolated PS5 Gen5 shader translator")
    if(EXISTS "${LSX4_KYTY_GEN5_DIR}/src/graphics/shader/recompiler/ShaderRecompiler.cpp")
        file(GLOB LSX4_GEN5_RECOMPILER_SOURCES CONFIGURE_DEPENDS
            "${LSX4_KYTY_GEN5_DIR}/src/graphics/shader/recompiler/*.cpp"
            "${LSX4_KYTY_GEN5_DIR}/src/graphics/shader/recompiler/shaderIR/*.cpp"
            "${LSX4_KYTY_GEN5_DIR}/src/graphics/shader/recompiler/spirvEmitter/*.cpp")
        list(APPEND LSX4_GEN5_RECOMPILER_SOURCES
            "${LSX4_KYTY_GEN5_DIR}/src/graphics/guest_gpu/gpu_format.cpp")
    else()
        message(FATAL_ERROR
            "The isolated PS5 Gen5 translator source tree is missing: ${LSX4_KYTY_GEN5_DIR}")
    endif()

    set(LSX4_KYTY_GENERATED_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated/kyty")
    file(MAKE_DIRECTORY "${LSX4_KYTY_GENERATED_DIR}")
    set(KYTY_VERSION "0.2.2")
    set(KYTY_ENDIAN KYTY_ENDIAN_LITTLE)
    set(KYTY_PLATFORM KYTY_PLATFORM_LINUX)
    set(KYTY_COMPILER CLANG)
    set(KYTY_LINKER LLD)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(KYTY_BUILD KYTY_BUILD_DEBUG)
    else()
        set(KYTY_BUILD KYTY_BUILD_RELEASE)
    endif()
    set(KYTY_FINAL OFF)
    configure_file(
        "${LSX4_KYTY_GEN5_DIR}/src/cmake_config.h.in"
        "${LSX4_KYTY_GENERATED_DIR}/cmake_config.h"
        @ONLY)

    add_library(lsx4_executor_ps5_android SHARED
        ${LSX4_PS5_DESKTOP_SOURCES}
        ${LSX4_JIT_COMMON_SOURCES}
        ${LSX4_GEN5_RECOMPILER_SOURCES}
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/common/x86_decoder.cpp"
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/common/thread.cpp"
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}/core/libraries/audio/audio_mix.cpp"
        src/executor/ps5_desktop/backend_contract.cpp
        src/executor/ps5_desktop/bc7decomp.cpp
        src/executor/ps5_desktop/gen5_compute_recompiler.cpp
        src/executor/ps5_desktop/ps5_audio_scheduler.cpp
        src/executor/ps5_desktop/vulkan_presenter.cpp
        src/executor/ps5_desktop/runtime_api.cpp
        src/executor/android_performance_hint.cpp)
    set_property(
        SOURCE "${LSX4_ARM_DESKTOP_SOURCE_DIR}/common/thread.cpp"
        APPEND PROPERTY COMPILE_OPTIONS
        "-iquote${LSX4_ARM_DESKTOP_SOURCE_DIR}")
    target_include_directories(lsx4_executor_ps5_android BEFORE PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/executor/ps5_desktop/kyty_compat"
        "${LSX4_KYTY_GENERATED_DIR}"
        "${LSX4_KYTY_GEN5_DIR}/src")
    target_include_directories(lsx4_executor_ps5_android PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${LSX4_ARM_DESKTOP_SOURCE_DIR}")
    target_compile_definitions(lsx4_executor_ps5_android PRIVATE
        LSX4_PS5_DESKTOP_PATH=1)
    set_target_properties(lsx4_executor_ps5_android PROPERTIES
        OUTPUT_NAME lsx4_executor_ps5_android)
endif()

option(LSX4_FAST_TRANSLATION_ITERATION
       "Compile the large translation core with reduced optimization for local iteration"
       OFF)
option(LSX4_TRANSLATION_THIN_LTO
       "Allow cross-file inlining across the split dynamic-translation implementation"
       ON)
if(ANDROID AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(LSX4_TRANSLATION_CORE_OPTIONS -mbranch-protection=bti -g0)
    if(LSX4_FAST_TRANSLATION_ITERATION)
        list(APPEND LSX4_TRANSLATION_CORE_OPTIONS -O1)
    endif()
    set_source_files_properties(src/executor/dynamic_translation/retiring_execution_core.cpp PROPERTIES
        COMPILE_OPTIONS "${LSX4_TRANSLATION_CORE_OPTIONS}")
    if(LSX4_TRANSLATION_THIN_LTO AND
       CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo|MinSizeRel)$")
        set(LSX4_TRANSLATION_LTO_SOURCES
            src/executor/dynamic_translation/retiring_execution_core.cpp
            src/executor/dynamic_translation/condition_materializer.cpp
            src/executor/dynamic_translation/floating_relation.cpp
            src/executor/dynamic_translation/guest_mapping_policy.cpp
            src/executor/dynamic_translation/instruction_diagnostics.cpp
            src/executor/dynamic_translation/integer_math.cpp
            src/executor/dynamic_translation/native_status_emitter.cpp
            src/executor/dynamic_translation/packed_lane_ops.cpp
            src/executor/dynamic_translation/partial_move_emitter.cpp
            src/executor/dynamic_translation/process_memory.cpp
            src/executor/dynamic_translation/vector_lowering_policy.cpp)
        set_property(SOURCE ${LSX4_TRANSLATION_LTO_SOURCES} APPEND PROPERTY
            COMPILE_OPTIONS -flto=thin)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/thinlto-cache")
        target_link_options(lsx4_executor_android PRIVATE
            -flto=thin
            "LINKER:--thinlto-cache-dir=${CMAKE_CURRENT_BINARY_DIR}/thinlto-cache")
        if(TARGET lsx4_executor_ps5_android)
            file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/thinlto-cache-ps5")
            target_link_options(lsx4_executor_ps5_android PRIVATE
                -flto=thin
                "LINKER:--thinlto-cache-dir=${CMAKE_CURRENT_BINARY_DIR}/thinlto-cache-ps5")
        endif()
    endif()
endif()

file(GLOB_RECURSE LSX4_JIT_NATIVE_ARTIFACT_ABI_SOURCES
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    "${CMAKE_CURRENT_SOURCE_DIR}/src/executor/dynamic_translation/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/executor/dynamic_translation/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/executor/dynamic_translation/*.in")
list(SORT LSX4_JIT_NATIVE_ARTIFACT_ABI_SOURCES)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${LSX4_JIT_NATIVE_ARTIFACT_ABI_SOURCES})
set(LSX4_JIT_NATIVE_ARTIFACT_SOURCE_MATERIAL "")
foreach(LSX4_JIT_NATIVE_ARTIFACT_ABI_SOURCE
        IN LISTS LSX4_JIT_NATIVE_ARTIFACT_ABI_SOURCES)
    file(SHA256 "${LSX4_JIT_NATIVE_ARTIFACT_ABI_SOURCE}"
         LSX4_JIT_NATIVE_ARTIFACT_SOURCE_FILE_HASH)
    file(RELATIVE_PATH LSX4_JIT_NATIVE_ARTIFACT_SOURCE_RELATIVE
         "${CMAKE_CURRENT_SOURCE_DIR}"
         "${LSX4_JIT_NATIVE_ARTIFACT_ABI_SOURCE}")
    string(APPEND LSX4_JIT_NATIVE_ARTIFACT_SOURCE_MATERIAL
        "${LSX4_JIT_NATIVE_ARTIFACT_SOURCE_RELATIVE}:"
        "${LSX4_JIT_NATIVE_ARTIFACT_SOURCE_FILE_HASH}\n")
endforeach()
string(SHA256 LSX4_JIT_NATIVE_ARTIFACT_SOURCE_HASH
    "${LSX4_JIT_NATIVE_ARTIFACT_SOURCE_MATERIAL}")

set(LSX4_JIT_NATIVE_ARTIFACT_ABI_EPOCH "4b739d10c6e58a21"
    CACHE STRING "Explicit compatibility salt for persisted AArch64 translations")
string(LENGTH "${LSX4_JIT_NATIVE_ARTIFACT_ABI_EPOCH}"
       LSX4_JIT_NATIVE_ARTIFACT_ABI_EPOCH_LENGTH)
if(NOT LSX4_JIT_NATIVE_ARTIFACT_ABI_EPOCH_LENGTH EQUAL 16 OR
   LSX4_JIT_NATIVE_ARTIFACT_ABI_EPOCH MATCHES "[^0-9a-fA-F]")
    message(FATAL_ERROR
        "LSX4_JIT_NATIVE_ARTIFACT_ABI_EPOCH must contain exactly 16 hexadecimal digits")
endif()
string(TOLOWER "${LSX4_JIT_NATIVE_ARTIFACT_ABI_EPOCH}"
       LSX4_JIT_NATIVE_ARTIFACT_ABI_EPOCH_NORMALIZED)
set(LSX4_JIT_NATIVE_ARTIFACT_ABI_MATERIAL
    "ps4:${LSX4_JIT_NATIVE_ARTIFACT_ABI_EPOCH_NORMALIZED}:${LSX4_JIT_NATIVE_ARTIFACT_SOURCE_HASH}")
string(SHA256 LSX4_JIT_NATIVE_ARTIFACT_ABI_HASH
    "${LSX4_JIT_NATIVE_ARTIFACT_ABI_MATERIAL}")
string(SUBSTRING "${LSX4_JIT_NATIVE_ARTIFACT_ABI_HASH}" 0 16
       EXECUTOR_JIT_NATIVE_CACHE_ABI_HASH_64)
set(EXECUTOR_JIT_NATIVE_CACHE_ABI_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated/executor/dynamic_translation")
file(MAKE_DIRECTORY "${EXECUTOR_JIT_NATIVE_CACHE_ABI_INCLUDE_DIR}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/executor/dynamic_translation/native_artifact_layout.h.in"
    "${EXECUTOR_JIT_NATIVE_CACHE_ABI_INCLUDE_DIR}/native_artifact_layout.h"
    @ONLY
)

if(TARGET lsx4_executor_ps5_android)
    set(LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_EPOCH "5053354a49543031"
        CACHE STRING "PS5-only compatibility epoch for persisted AArch64 translations")
    string(LENGTH "${LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_EPOCH}"
           LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_EPOCH_LENGTH)
    if(NOT LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_EPOCH_LENGTH EQUAL 16 OR
       LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_EPOCH MATCHES "[^0-9a-fA-F]")
        message(FATAL_ERROR
            "LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_EPOCH must contain exactly 16 hexadecimal digits")
    endif()
    string(TOLOWER "${LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_EPOCH}"
           LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_EPOCH_NORMALIZED)
    set(LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_MATERIAL
        "ps5:${LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_EPOCH_NORMALIZED}:${LSX4_JIT_NATIVE_ARTIFACT_SOURCE_HASH}")
    string(SHA256 LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_HASH
        "${LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_MATERIAL}")
    string(SUBSTRING "${LSX4_PS5_JIT_NATIVE_ARTIFACT_ABI_HASH}" 0 16
           EXECUTOR_JIT_NATIVE_CACHE_ABI_HASH_64)
    set(LSX4_PS5_JIT_NATIVE_CACHE_ABI_INCLUDE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated/ps5/executor/dynamic_translation")
    file(MAKE_DIRECTORY "${LSX4_PS5_JIT_NATIVE_CACHE_ABI_INCLUDE_DIR}")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/executor/dynamic_translation/native_artifact_layout.h.in"
        "${LSX4_PS5_JIT_NATIVE_CACHE_ABI_INCLUDE_DIR}/native_artifact_layout.h"
        @ONLY
    )
endif()

target_include_directories(lsx4_executor_android PRIVATE
    "${EXECUTOR_JIT_NATIVE_CACHE_ABI_INCLUDE_DIR}"
    "${LSX4_ARM_DESKTOP_SOURCE_DIR}"
    "${LSX4_ARM_DESKTOP_LAYER_DIR}")
set_target_properties(lsx4_executor_android PROPERTIES
    OUTPUT_NAME lsx4_executor_android
)

set(LSX4_EXECUTOR_ANDROID_EXPORT_MAP
    "${CMAKE_CURRENT_SOURCE_DIR}/src/executor/lsx4_executor_android.map")
target_link_options(lsx4_executor_android PRIVATE
    "LINKER:--version-script=${LSX4_EXECUTOR_ANDROID_EXPORT_MAP}")
set_property(TARGET lsx4_executor_android APPEND PROPERTY
    LINK_DEPENDS "${LSX4_EXECUTOR_ANDROID_EXPORT_MAP}")

if(TARGET lsx4_executor_ps5_android)
    target_include_directories(lsx4_executor_ps5_android PRIVATE
        "${LSX4_PS5_JIT_NATIVE_CACHE_ABI_INCLUDE_DIR}")
    set(LSX4_EXECUTOR_PS5_ANDROID_EXPORT_MAP
        "${CMAKE_CURRENT_SOURCE_DIR}/src/executor/ps5_desktop/lsx4_executor_ps5_android.map")
    target_link_options(lsx4_executor_ps5_android PRIVATE
        "LINKER:--version-script=${LSX4_EXECUTOR_PS5_ANDROID_EXPORT_MAP}"
        "LINKER:--no-undefined")
    set_property(TARGET lsx4_executor_ps5_android APPEND PROPERTY
        LINK_DEPENDS "${LSX4_EXECUTOR_PS5_ANDROID_EXPORT_MAP}")
endif()

if(CMAKE_STRIP AND CMAKE_BUILD_TYPE MATCHES "^(Release|MinSizeRel)$")
    add_custom_command(TARGET lsx4_executor_android POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy
            "$<TARGET_FILE:lsx4_executor_android>"
            "$<TARGET_FILE:lsx4_executor_android>.symbols"
        COMMAND "${CMAKE_STRIP}" --strip-unneeded "$<TARGET_FILE:lsx4_executor_android>"
        COMMENT "Stripping private symbols from liblsx4_executor_android.so"
        VERBATIM)
endif()

create_target_directory_groups(lsx4_executor_android)
add_dependencies(lsx4_executor_android lsx_iced_decoder_build)
if(TARGET lsx4_executor_ps5_android)
    create_target_directory_groups(lsx4_executor_ps5_android)
    add_dependencies(lsx4_executor_ps5_android lsx_iced_decoder_build)
endif()

set(LSX4_ANDROID_MEDIA_LIBS)
if (EXECUTOR_ANDROID_NATIVE_CORE_PROBE_ENABLE_AVPLAYER)
    set(LSX4_ANDROID_MEDIA_LIBS FFmpeg::ffmpeg)
endif()
if (EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
    target_compile_definitions(lsx4_executor_android PRIVATE EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
endif()
if (EXECUTOR_ANDROID_NATIVE_CORE_PROBE_ENABLE_AVPLAYER)
    target_compile_definitions(lsx4_executor_android PRIVATE EXECUTOR_ANDROID_NATIVE_CORE_PROBE_ENABLE_AVPLAYER)
endif()

target_link_libraries(lsx4_executor_android PRIVATE magic_enum::magic_enum fmt::fmt toml11::toml11 tsl::robin_map xbyak::xbyak RenderDoc::API ${LSX4_ANDROID_MEDIA_LIBS} Dear_ImGui gcn half::half ZLIB::ZLIB PNG::PNG)
target_link_libraries(lsx4_executor_android PRIVATE Boost::headers GPUOpen::VulkanMemoryAllocator LibAtrac9 sirit Vulkan::Headers xxHash::xxhash lsx_iced_decoder glslang::glslang SDL3::SDL3 SDL3_mixer::SDL3_mixer pugixml::pugixml)
target_link_libraries(lsx4_executor_android PRIVATE stb::headers nlohmann_json::nlohmann_json miniz::miniz fdk-aac CLI11::CLI11 OpenAL::OpenAL Cpp_Httplib)
if(TARGET xbyak_aarch64::xbyak_aarch64)
    target_link_libraries(lsx4_executor_android PRIVATE xbyak_aarch64::xbyak_aarch64)
endif()
if(TARGET lsx4_executor_ps5_android)
    target_link_libraries(lsx4_executor_ps5_android PRIVATE
        fmt::fmt
        tsl::robin_map
        nlohmann_json::nlohmann_json
        PNG::PNG
        Vulkan::Headers
        xxHash::xxhash
        lsx_iced_decoder)
    if(TARGET xbyak_aarch64::xbyak_aarch64)
        target_link_libraries(lsx4_executor_ps5_android PRIVATE
            xbyak_aarch64::xbyak_aarch64)
    endif()
endif()
if(ANDROID)
    target_link_libraries(lsx4_executor_android PRIVATE android aaudio mediandk dl log EGL GLESv3)
    if(TARGET lsx4_executor_ps5_android)
        target_link_libraries(lsx4_executor_ps5_android PRIVATE android aaudio dl log vulkan)
    endif()
    target_link_options(lsx4_executor_android PRIVATE
        -Wl,--wrap=exit
        -Wl,--wrap=_exit
        -Wl,--wrap=_Exit
        -Wl,--wrap=exit_group
        -Wl,--wrap=abort
        -Wl,--wrap=tgkill
        -Wl,--wrap=__stack_chk_fail
        -Wl,--wrap=__assert2
    )
endif()

target_compile_definitions(lsx4_executor_android PRIVATE IMGUI_USER_CONFIG="imgui/imgui_config.h")
target_compile_definitions(Dear_ImGui PRIVATE IMGUI_USER_CONFIG="${LSX4_ARM_DESKTOP_SOURCE_DIR}/imgui/imgui_config.h")

if (ANDROID AND EXECUTOR_GPU_CORRUPTION_DIAG)
    set_source_files_properties(${LSX4_ARM_VIDEO_SOURCES} ${LSX4_ARM_SHADER_SOURCES}
        PROPERTIES COMPILE_DEFINITIONS "_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG;BOOST_ENABLE_ASSERT_HANDLER")
endif()

if (ANDROID AND EXECUTOR_GPU_CORRUPTION_ASAN)
    set_source_files_properties(${LSX4_ARM_VIDEO_SOURCES} ${LSX4_ARM_SHADER_SOURCES}
        PROPERTIES COMPILE_OPTIONS "-fsanitize=address;-fno-omit-frame-pointer")
    target_link_options(lsx4_executor_android PRIVATE -fsanitize=address -shared-libasan)
endif()

add_compile_definitions(BOOST_ASIO_STANDALONE)

target_include_directories(lsx4_executor_android PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lsx4_executor_android PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/externals/tracy/public)

set(HOST_SHADERS_INCLUDE ${LSX4_ARM_DESKTOP_SOURCE_DIR}/video_core/host_shaders)

add_subdirectory(${HOST_SHADERS_INCLUDE}
    "${CMAKE_CURRENT_BINARY_DIR}/lsx4-arm-host-shaders")
add_dependencies(lsx4_executor_android host_shaders)
target_include_directories(lsx4_executor_android PRIVATE ${HOST_SHADERS_INCLUDE})

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeRC.cmake")
cmrc_add_resource_library(embedded-resources
        ALIAS res::embedded
        NAMESPACE res
        WHENCE "${LSX4_ARM_DESKTOP_LAYER_DIR}"
        ${LSX4_ARM_DESKTOP_SOURCE_DIR}/runtime_resources/trophy.wav
        ${LSX4_ARM_DESKTOP_SOURCE_DIR}/runtime_resources/bronze.png
        ${LSX4_ARM_DESKTOP_SOURCE_DIR}/runtime_resources/gold.png
        ${LSX4_ARM_DESKTOP_SOURCE_DIR}/runtime_resources/platinum.png
        ${LSX4_ARM_DESKTOP_SOURCE_DIR}/runtime_resources/silver.png)
target_link_libraries(lsx4_executor_android PRIVATE res::embedded)

add_subdirectory(${LSX4_ARM_DESKTOP_SOURCE_DIR}/imgui/renderer
    "${CMAKE_CURRENT_BINARY_DIR}/lsx4-arm-imgui-renderer")
add_dependencies(lsx4_executor_android ImGui_Resources)
target_include_directories(lsx4_executor_android PRIVATE ${IMGUI_RESOURCES_INCLUDE})

install(TARGETS lsx4_executor_android LIBRARY DESTINATION lib)
if(TARGET lsx4_executor_ps5_android)
    install(TARGETS lsx4_executor_ps5_android LIBRARY DESTINATION lib)
endif()
