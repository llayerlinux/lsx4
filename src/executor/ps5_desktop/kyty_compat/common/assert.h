#pragma once

#include "common/common.h"

#include <cstdlib>

#define EXIT_IF(condition) \
    do { if (condition) std::abort(); } while (false)
#define EXIT(...) std::abort()
#define EXIT_COLOR(...) std::abort()
#define EXIT_NOT_IMPLEMENTED(condition) \
    do { if (condition) std::abort(); } while (false)
#define KYTY_NOT_IMPLEMENTED std::abort()
