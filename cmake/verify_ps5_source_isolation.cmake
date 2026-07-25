# SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
# SPDX-License-Identifier: GPL-2.0-or-later

cmake_minimum_required(VERSION 3.24)

cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH LSX4_ROOT)
set(LSX4_ARM_DESKTOP_LAYER_DIR "${LSX4_ROOT}/funnel-arm")

include("${LSX4_ARM_DESKTOP_LAYER_DIR}/cmake/lsx4_arm_desktop_layer.cmake")
include("${LSX4_ARM_DESKTOP_LAYER_DIR}/cmake/lsx4_ps5_desktop_layer.cmake")

file(GLOB_RECURSE ps4_common_sources
    "${LSX4_ARM_DESKTOP_SOURCE_DIR}/common/*.cpp")
file(GLOB_RECURSE ps4_core_sources
    "${LSX4_ARM_DESKTOP_SOURCE_DIR}/core/*.cpp")
file(GLOB_RECURSE ps4_imgui_sources
    "${LSX4_ARM_DESKTOP_SOURCE_DIR}/imgui/*.cpp")
file(GLOB_RECURSE ps4_input_sources
    "${LSX4_ARM_DESKTOP_SOURCE_DIR}/input/*.cpp")
file(GLOB_RECURSE ps4_shader_sources
    "${LSX4_ARM_DESKTOP_SOURCE_DIR}/shader_recompiler/*.cpp")
file(GLOB_RECURSE ps4_video_sources
    "${LSX4_ARM_DESKTOP_SOURCE_DIR}/video_core/*.cpp")
list(FILTER ps4_imgui_sources EXCLUDE REGEX "/renderer/generated_fonts/")
set(LSX4_ARM_DESKTOP_SOURCES
    ${ps4_common_sources}
    ${ps4_core_sources}
    ${ps4_imgui_sources}
    ${ps4_input_sources}
    ${ps4_shader_sources}
    ${ps4_video_sources}
    "${LSX4_ARM_DESKTOP_SOURCE_DIR}/emulator.cpp"
    "${LSX4_ARM_DESKTOP_SOURCE_DIR}/sdl_window.cpp")

lsx4_define_ps5_desktop_profile()
lsx4_assert_arm_desktop_ownership(${LSX4_ARM_DESKTOP_SOURCES})
lsx4_assert_ps5_desktop_ownership(${LSX4_PS5_DESKTOP_SOURCES})

foreach(ps4_source IN LISTS LSX4_ARM_DESKTOP_SOURCES)
    if(ps4_source MATCHES "[/\\\\]ps5_desktop[/\\\\]")
        message(FATAL_ERROR
            "PS5 source entered the PS4 desktop profile: ${ps4_source}")
    endif()
endforeach()

foreach(ps5_source IN LISTS LSX4_PS5_DESKTOP_SOURCES)
    list(FIND LSX4_ARM_DESKTOP_SOURCES "${ps5_source}" ps4_source_index)
    if(NOT ps4_source_index EQUAL -1)
        message(FATAL_ERROR
            "Source belongs to both PS4 and PS5 profiles: ${ps5_source}")
    endif()
endforeach()

list(LENGTH LSX4_ARM_DESKTOP_SOURCES ps4_source_count)
list(LENGTH LSX4_PS5_DESKTOP_SOURCES ps5_source_count)
if(ps5_source_count EQUAL 0)
    message(FATAL_ERROR "The isolated PS5 source profile is empty")
endif()

message(STATUS
    "PS5 source isolation verified: PS4=${ps4_source_count}, PS5=${ps5_source_count}, overlap=0")
