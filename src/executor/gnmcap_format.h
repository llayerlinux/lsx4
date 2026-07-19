// .gnmcap (v1) -- GNM frame capture/replay container. Header-only writer + reader.
// Replay-oriented; v1 defines and round-trips the
// container byte-exact (NO replay claim -- there is no GNM frame-capture source yet). See memory
// project_shadps4_layered_roadmap (Codex verdict-B follow-up).
#pragma once

#include "common/content_fingerprint.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace Executor::GnmCap {

inline constexpr char kMagic[8] = {'G', 'N', 'M', 'C', 'A', 'P', '\0', '\0'};
inline constexpr uint32_t kVersionV1 = 0x00010000;  // major<<16 | minor

enum class SectionType : uint32_t {
    Submit = 1,
    QueueBlob = 2,
    Draw = 3,
    Regs = 4,
    Shader = 5,
    MemoryRange = 6,
    Reloc = 7,
};

enum class QueueType : uint32_t { Dcb = 0, Ccb = 1 };

// --- POD records (tightly packed; all scalars little-endian) ---------------------------------------
#pragma pack(push, 1)
struct Header {
    char magic[8];
    uint32_t version;
    uint8_t endian;  // 0 = little
    uint8_t reserved[3];
    uint32_t section_count;
    uint64_t content_hash;  // Domain-separated v1 fingerprint of all bytes after this field.
};
struct SectionEntry {
    uint32_t type;
    uint32_t flags;
    uint64_t length;
    uint64_t hash;  // Domain-separated v1 fingerprint of payload.
};
struct SubmitRec {
    uint32_t submit_id;
    uint32_t queue_count;
    uint32_t dcb_section_idx;
    uint32_t ccb_section_idx;
};
struct QueueBlobHdr {
    uint32_t queue_type;  // QueueType
    uint64_t base_addr;
    uint32_t dword_count;
    uint32_t reserved;
};
struct DrawRec {
    uint32_t submit_id;
    uint32_t queue_type;
    uint64_t pm4_offset;
    uint64_t regs_hash;
    uint64_t vs_shader_hash;
    uint64_t ps_shader_hash;
};
struct RegsHdr {
    uint64_t regs_hash;
    uint32_t regs_bytes;
    uint32_t reserved;
};
struct ShaderHdr {
    uint64_t shader_hash;
    uint32_t stage;  // 0=PS 1=VS 2=GS 3=ES 4=HS 5=LS 6=CS
    uint32_t code_bytes;
    uint32_t crc32;
    uint32_t reserved;
};
struct MemoryRangeHdr {
    uint64_t guest_base;
    uint64_t size;
    uint32_t flags;
    uint32_t reserved;
    uint64_t sha256_lo;  // Reserved digest lane; zero unless a section contract defines it.
    uint64_t sha256_hi;
};
struct RelocRec {
    uint64_t old_base;
    uint64_t new_base;
    uint64_t size;
    uint32_t reason;
    uint32_t reserved;
};
#pragma pack(pop)

inline uint64_t FingerprintCaptureBytes(const void* data, const size_t len,
                                        const uint64_t domain) {
    return Common::FingerprintBytes(
        {static_cast<const uint8_t*>(data), len}, domain);
}

// --- In-memory model -------------------------------------------------------------------------------
struct Section {
    SectionType type;
    uint32_t flags{0};
    std::vector<uint8_t> payload;
};

struct Capture {
    std::vector<Section> sections;

    template <typename T>
    static void AppendPod(std::vector<uint8_t>& v, const T& rec) {
        const auto* p = reinterpret_cast<const uint8_t*>(&rec);
        v.insert(v.end(), p, p + sizeof(T));
    }
    static void AppendBytes(std::vector<uint8_t>& v, const void* data, size_t len) {
        const auto* p = static_cast<const uint8_t*>(data);
        v.insert(v.end(), p, p + len);
    }

    Section& AddSection(SectionType t) {
        sections.push_back(Section{t, 0, {}});
        return sections.back();
    }
};

// --- Serialize -------------------------------------------------------------------------------------
inline std::vector<uint8_t> Serialize(const Capture& cap) {
    std::vector<uint8_t> body;  // everything after content_hash

    // section table
    std::vector<SectionEntry> entries;
    entries.reserve(cap.sections.size());
    for (const auto& s : cap.sections) {
        SectionEntry e{};
        e.type = static_cast<uint32_t>(s.type);
        e.flags = s.flags;
        e.length = s.payload.size();
        e.hash = FingerprintCaptureBytes(s.payload.data(), s.payload.size(),
                                         Common::FingerprintDomain::CaptureSection);
        entries.push_back(e);
    }
    for (const auto& e : entries) {
        Capture::AppendPod(body, e);
    }
    for (const auto& s : cap.sections) {
        Capture::AppendBytes(body, s.payload.data(), s.payload.size());
    }

    Header h{};
    std::memcpy(h.magic, kMagic, sizeof(kMagic));
    h.version = kVersionV1;
    h.endian = 0;
    h.section_count = static_cast<uint32_t>(cap.sections.size());
    h.content_hash = FingerprintCaptureBytes(body.data(), body.size(),
                                             Common::FingerprintDomain::CaptureBody);

    std::vector<uint8_t> out;
    Capture::AppendPod(out, h);
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

// --- Deserialize -----------------------------------------------------------------------------------
enum class ParseResult : uint32_t {
    Ok = 0,
    BadMagic,
    BadVersion,
    BadContentHash,
    Truncated,
    BadSectionHash,
};

inline ParseResult Deserialize(const std::vector<uint8_t>& bytes, Capture& out) {
    out.sections.clear();
    if (bytes.size() < sizeof(Header)) {
        return ParseResult::Truncated;
    }
    Header h{};
    std::memcpy(&h, bytes.data(), sizeof(Header));
    if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0) {
        return ParseResult::BadMagic;
    }
    if ((h.version >> 16) != (kVersionV1 >> 16)) {
        return ParseResult::BadVersion;
    }
    const uint8_t* body = bytes.data() + sizeof(Header);
    const size_t body_len = bytes.size() - sizeof(Header);
    if (FingerprintCaptureBytes(body, body_len, Common::FingerprintDomain::CaptureBody) !=
        h.content_hash) {
        return ParseResult::BadContentHash;
    }
    const size_t table_bytes = static_cast<size_t>(h.section_count) * sizeof(SectionEntry);
    if (body_len < table_bytes) {
        return ParseResult::Truncated;
    }
    std::vector<SectionEntry> entries(h.section_count);
    if (h.section_count) {
        std::memcpy(entries.data(), body, table_bytes);
    }
    size_t off = table_bytes;
    for (const auto& e : entries) {
        if (off + e.length > body_len) {
            return ParseResult::Truncated;
        }
        if (FingerprintCaptureBytes(body + off, e.length,
                                    Common::FingerprintDomain::CaptureSection) != e.hash) {
            return ParseResult::BadSectionHash;
        }
        Section s{};
        s.type = static_cast<SectionType>(e.type);
        s.flags = e.flags;
        s.payload.assign(body + off, body + off + e.length);
        out.sections.push_back(std::move(s));
        off += e.length;
    }
    return ParseResult::Ok;
}

inline const char* ParseResultName(ParseResult r) {
    switch (r) {
    case ParseResult::Ok: return "Ok";
    case ParseResult::BadMagic: return "BadMagic";
    case ParseResult::BadVersion: return "BadVersion";
    case ParseResult::BadContentHash: return "BadContentHash";
    case ParseResult::Truncated: return "Truncated";
    case ParseResult::BadSectionHash: return "BadSectionHash";
    }
    return "Unknown";
}

}  // namespace Executor::GnmCap
