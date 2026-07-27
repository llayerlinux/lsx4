#pragma once

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#define KYTY_CLASS_NO_COPY(name) \
public:                          \
    name(const name&) = delete;  \
    name& operator=(const name&) = delete; \
    name(name&&) noexcept = delete; \
    name& operator=(name&&) noexcept = delete;

#define KYTY_CLASS_DEFAULT_COPY(name) \
public:                               \
    name(const name&) = default;      \
    name& operator=(const name&) = default; \
    name(name&&) noexcept = default;  \
    name& operator=(name&&) noexcept = default;
