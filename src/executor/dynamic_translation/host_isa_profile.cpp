// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/host_isa_profile.h"

#if defined(__ANDROID__) && defined(__aarch64__)
#include <android/log.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#endif

namespace Executor::Jit {
namespace {

constexpr std::uint64_t Bit(const HostIsaFeature feature) noexcept {
    return static_cast<std::uint64_t>(feature);
}

HostIsaProfile DetectHostIsaProfile() noexcept {
    HostIsaProfile profile{};
#if defined(__ANDROID__) && defined(__aarch64__)
    profile.hwcap = getauxval(AT_HWCAP);
#if defined(AT_HWCAP2)
    profile.hwcap2 = getauxval(AT_HWCAP2);
#endif

#if defined(HWCAP_ATOMICS)
    if ((profile.hwcap & HWCAP_ATOMICS) != 0) {
        profile.features |= Bit(HostIsaFeature::Lse);
    }
#endif
#if defined(HWCAP_FLAGM)
    if ((profile.hwcap & HWCAP_FLAGM) != 0) {
        profile.features |= Bit(HostIsaFeature::FlagM);
    }
#endif
#if defined(HWCAP_LRCPC)
    if ((profile.hwcap & HWCAP_LRCPC) != 0) {
        profile.features |= Bit(HostIsaFeature::Lrcpc);
    }
#endif
#if defined(HWCAP_ILRCPC)
    if ((profile.hwcap & HWCAP_ILRCPC) != 0) {
        profile.features |= Bit(HostIsaFeature::Ilrcpc);
    }
#endif
#if defined(HWCAP_SVE)
    if ((profile.hwcap & HWCAP_SVE) != 0) {
        profile.features |= Bit(HostIsaFeature::Sve);
    }
#endif
#if defined(HWCAP2_FLAGM2)
    if ((profile.hwcap2 & HWCAP2_FLAGM2) != 0) {
        profile.features |= Bit(HostIsaFeature::FlagM2);
    }
#endif
#if defined(HWCAP2_LRCPC3)
    if ((profile.hwcap2 & HWCAP2_LRCPC3) != 0) {
        profile.features |= Bit(HostIsaFeature::Lrcpc3);
    }
#endif
#if defined(HWCAP2_SVE2)
    if ((profile.hwcap2 & HWCAP2_SVE2) != 0 &&
        profile.Has(HostIsaFeature::Sve)) {
        profile.features |= Bit(HostIsaFeature::Sve2);
    }
#endif
    profile.sve_vector_bytes = CurrentThreadSveVectorBytes();
    if (profile.sve_vector_bytes == 0) {
        profile.features &= ~(Bit(HostIsaFeature::Sve) |
                              Bit(HostIsaFeature::Sve2));
    }
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4-JIT",
        "[EXECUTOR_HOST_ISA] hwcap=0x%llx hwcap2=0x%llx features=0x%llx "
        "sveVectorBytes=%u flagm=%d flagm2=%d lrcpc=%d ilrcpc=%d "
        "lrcpc3=%d sve2=%d",
        static_cast<unsigned long long>(profile.hwcap),
        static_cast<unsigned long long>(profile.hwcap2),
        static_cast<unsigned long long>(profile.features),
        profile.sve_vector_bytes,
        profile.Has(HostIsaFeature::FlagM) ? 1 : 0,
        profile.Has(HostIsaFeature::FlagM2) ? 1 : 0,
        profile.Has(HostIsaFeature::Lrcpc) ? 1 : 0,
        profile.Has(HostIsaFeature::Ilrcpc) ? 1 : 0,
        profile.Has(HostIsaFeature::Lrcpc3) ? 1 : 0,
        profile.Has(HostIsaFeature::Sve2) ? 1 : 0);
#endif
    return profile;
}

}

std::uint32_t CurrentThreadSveVectorBytes() noexcept {
#if defined(__ANDROID__) && defined(__aarch64__) && defined(PR_SVE_GET_VL) && \
    defined(PR_SVE_VL_LEN_MASK)
    const int vector_length = prctl(PR_SVE_GET_VL);
    if (vector_length > 0) {
        return static_cast<std::uint32_t>(vector_length) &
               static_cast<std::uint32_t>(PR_SVE_VL_LEN_MASK);
    }
#endif
    return 0;
}

const HostIsaProfile& GetHostIsaProfile() noexcept {
    static const HostIsaProfile profile = DetectHostIsaProfile();
    return profile;
}

}
