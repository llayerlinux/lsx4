// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/ps5_desktop/runtime_api.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t SegmentFileOffset = 0x4000;
constexpr std::size_t SegmentFileSize = 0x1000;
constexpr std::size_t DynamicFileOffset = SegmentFileOffset + 0x200;
constexpr std::size_t StringFileOffset = SegmentFileOffset + 0x400;
constexpr std::size_t SymbolFileOffset = SegmentFileOffset + 0x500;
constexpr std::size_t RelaFileOffset = SegmentFileOffset + 0x800;
constexpr std::uint64_t DynamicVirtualAddress = 0x200;
constexpr std::uint64_t StringVirtualAddress = 0x400;
constexpr std::uint64_t SymbolVirtualAddress = 0x500;
constexpr std::uint64_t RelaVirtualAddress = 0x800;
constexpr std::int64_t DtSceStringTable = 0x61000035;
constexpr std::int64_t DtSceStringSize = 0x61000037;
constexpr std::int64_t DtSceSymbolTable = 0x61000039;
constexpr std::int64_t DtSceSymbolEntry = 0x6100003b;
constexpr std::int64_t DtSceSymbolTableSize = 0x6100003f;
constexpr std::int64_t DtSceRela = 0x6100002f;
constexpr std::int64_t DtSceRelaSize = 0x61000031;
constexpr std::int64_t DtSceRelaEntry = 0x61000033;
constexpr std::uint32_t PtSceDynlibData = 0x61000000;

#pragma pack(push, 1)
struct SelfHeader {
    std::uint8_t ident[12]{};
    std::uint16_t size1{};
    std::uint16_t size2{};
    std::uint64_t file_size{};
    std::uint16_t segment_count{};
    std::uint16_t unknown{};
    std::uint32_t padding{};
};

struct SelfSegment {
    std::uint64_t type{};
    std::uint64_t offset{};
    std::uint64_t compressed_size{};
    std::uint64_t decompressed_size{};
};
#pragma pack(pop)

static_assert(sizeof(SelfHeader) == 32);
static_assert(sizeof(SelfSegment) == 32);

template <typename Function>
Function Resolve(void* const library, const char* const name) {
    return reinterpret_cast<Function>(dlsym(library, name));
}

template <typename T>
T ReadAt(const std::vector<std::uint8_t>& bytes,
         const std::size_t offset) {
    T value{};
    if (offset <= bytes.size() && sizeof(T) <= bytes.size() - offset) {
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
    }
    return value;
}

template <typename T>
void WriteAt(std::vector<std::uint8_t>& bytes,
             const std::size_t offset, const T& value) {
    if (offset + sizeof(T) > bytes.size()) {
        bytes.resize(offset + sizeof(T));
    }
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

bool WriteFile(const std::string& path,
               const std::vector<std::uint8_t>& bytes) {
    const int descriptor =
        open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (descriptor < 0) {
        return false;
    }
    std::size_t written{};
    while (written < bytes.size()) {
        const auto result =
            write(descriptor, bytes.data() + written,
                  bytes.size() - written);
        if (result <= 0) {
            close(descriptor);
            return false;
        }
        written += static_cast<std::size_t>(result);
    }
    return close(descriptor) == 0;
}

Elf64_Ehdr Header(const std::vector<std::uint8_t>& bytes) {
    return ReadAt<Elf64_Ehdr>(bytes, 0);
}

void SetHeader(std::vector<std::uint8_t>& bytes,
               const Elf64_Ehdr& header) {
    WriteAt(bytes, 0, header);
}

Elf64_Phdr Program(const std::vector<std::uint8_t>& bytes,
                   const std::size_t index) {
    const auto header = Header(bytes);
    return ReadAt<Elf64_Phdr>(
        bytes, header.e_phoff + index * sizeof(Elf64_Phdr));
}

void SetProgram(std::vector<std::uint8_t>& bytes,
                const std::size_t index,
                const Elf64_Phdr& program) {
    const auto header = Header(bytes);
    WriteAt(
        bytes, header.e_phoff + index * sizeof(Elf64_Phdr), program);
}

std::vector<std::uint8_t> BasicElf(const std::uint16_t program_count = 1) {
    std::vector<std::uint8_t> bytes(
        SegmentFileOffset + SegmentFileSize);
    Elf64_Ehdr header{};
    std::memcpy(header.e_ident, ELFMAG, SELFMAG);
    header.e_ident[EI_CLASS] = ELFCLASS64;
    header.e_ident[EI_DATA] = ELFDATA2LSB;
    header.e_ident[EI_VERSION] = EV_CURRENT;
    header.e_ident[EI_OSABI] = ELFOSABI_FREEBSD;
    header.e_ident[EI_ABIVERSION] = 2;
    header.e_type = ET_DYN;
    header.e_machine = EM_X86_64;
    header.e_version = EV_CURRENT;
    header.e_entry = 0;
    header.e_phoff = sizeof(Elf64_Ehdr);
    header.e_ehsize = sizeof(Elf64_Ehdr);
    header.e_phentsize = sizeof(Elf64_Phdr);
    header.e_phnum = program_count;
    SetHeader(bytes, header);

    Elf64_Phdr load{};
    load.p_type = PT_LOAD;
    load.p_flags = PF_R | PF_W | PF_X;
    load.p_offset = SegmentFileOffset;
    load.p_vaddr = 0;
    load.p_filesz = SegmentFileSize;
    load.p_memsz = 0x4000;
    load.p_align = 0x4000;
    SetProgram(bytes, 0, load);
    bytes[SegmentFileOffset] = 0xc3;
    return bytes;
}

void AddDynamic(std::vector<std::uint8_t>& bytes,
                const std::vector<Elf64_Dyn>& dynamic) {
    auto header = Header(bytes);
    if (header.e_phnum < 2) {
        header.e_phnum = 2;
        SetHeader(bytes, header);
    }
    Elf64_Phdr program{};
    program.p_type = PT_DYNAMIC;
    program.p_flags = PF_R;
    program.p_offset = DynamicFileOffset;
    program.p_vaddr = DynamicVirtualAddress;
    program.p_filesz = dynamic.size() * sizeof(Elf64_Dyn);
    program.p_memsz = program.p_filesz;
    program.p_align = 8;
    SetProgram(bytes, 1, program);
    std::memcpy(
        bytes.data() + DynamicFileOffset, dynamic.data(),
        dynamic.size() * sizeof(Elf64_Dyn));
}

void AddTls(std::vector<std::uint8_t>& bytes,
            const std::size_t index = 2,
            const std::uint64_t alignment = 0x20,
            const std::uint64_t file_size = 8,
            const std::uint64_t memory_size = 0x20) {
    auto header = Header(bytes);
    header.e_phnum = static_cast<std::uint16_t>(
        std::max<std::size_t>(header.e_phnum, index + 1));
    SetHeader(bytes, header);
    Elf64_Phdr tls{};
    tls.p_type = PT_TLS;
    tls.p_flags = PF_R;
    tls.p_offset = SegmentFileOffset + 0x100;
    tls.p_vaddr = 0x100;
    tls.p_filesz = file_size;
    tls.p_memsz = memory_size;
    tls.p_align = alignment;
    SetProgram(bytes, index, tls);
}

std::vector<std::uint8_t> WrapSelf(
    const std::vector<std::uint8_t>& elf, const bool ps5_magic) {
    constexpr std::size_t embedded_elf_offset =
        sizeof(SelfHeader) + sizeof(SelfSegment);
    constexpr std::size_t self_payload_offset = 0x1000;
    const auto header = Header(elf);
    const auto header_bytes =
        header.e_phoff +
        static_cast<std::size_t>(header.e_phnum) * sizeof(Elf64_Phdr);
    std::vector<std::uint8_t> self(
        self_payload_offset + SegmentFileSize);
    SelfHeader self_header{};
    if (ps5_magic) {
        self_header.ident[0] = 0x54;
        self_header.ident[1] = 0x14;
        self_header.ident[2] = 0xf5;
        self_header.ident[3] = 0xee;
    } else {
        self_header.ident[0] = 0x4f;
        self_header.ident[1] = 0x15;
        self_header.ident[2] = 0x3d;
        self_header.ident[3] = 0x1d;
    }
    self_header.segment_count = 1;
    self_header.file_size = self.size();
    SelfSegment segment{};
    segment.type = 0x800;
    segment.offset = self_payload_offset;
    segment.compressed_size = SegmentFileSize;
    segment.decompressed_size = SegmentFileSize;
    WriteAt(self, 0, self_header);
    WriteAt(self, sizeof(SelfHeader), segment);
    std::memcpy(
        self.data() + embedded_elf_offset, elf.data(), header_bytes);
    std::memcpy(
        self.data() + self_payload_offset,
        elf.data() + SegmentFileOffset, SegmentFileSize);
    return self;
}

std::vector<std::uint8_t> WrapSelfWithStaleSections(
    const std::vector<std::uint8_t>& elf) {
    constexpr std::size_t embedded_elf_offset =
        sizeof(SelfHeader) + sizeof(SelfSegment);
    auto self = WrapSelf(elf, true);
    auto header = Header(elf);
    header.e_shoff = 0x200;
    header.e_shentsize = sizeof(Elf64_Shdr);
    header.e_shnum = 1;
    WriteAt(self, embedded_elf_offset, header);
    Elf64_Shdr stale_relocations{};
    stale_relocations.sh_type = SHT_RELA;
    stale_relocations.sh_size = sizeof(Elf64_Rela);
    stale_relocations.sh_link = 7;
    stale_relocations.sh_entsize = sizeof(Elf64_Rela) - 1;
    WriteAt(
        self, embedded_elf_offset + header.e_shoff,
        stale_relocations);
    return self;
}

std::vector<std::uint8_t> SectionFallbackElf() {
    auto bytes = BasicElf();
    bytes.resize(0x5800);
    auto header = Header(bytes);
    header.e_shoff = 0x5000;
    header.e_shentsize = sizeof(Elf64_Shdr);
    header.e_shnum = 4;
    SetHeader(bytes, header);

    constexpr std::size_t strings_offset = 0x5200;
    constexpr std::size_t symbols_offset = 0x5300;
    constexpr std::size_t relocations_offset = 0x5400;
    constexpr char names[] = "\0_section_export#mod#lib\0";
    std::memcpy(bytes.data() + strings_offset, names, sizeof(names));

    Elf64_Sym symbol{};
    symbol.st_name = 1;
    symbol.st_info = ELF64_ST_INFO(STB_GLOBAL, STT_OBJECT);
    symbol.st_shndx = 1;
    symbol.st_value = 0x100;
    symbol.st_size = 8;
    WriteAt(bytes, symbols_offset + sizeof(Elf64_Sym), symbol);

    Elf64_Rela relocation{};
    relocation.r_offset = 0x180;
    relocation.r_info = ELF64_R_INFO(1, R_X86_64_64);
    relocation.r_addend = 4;
    WriteAt(bytes, relocations_offset, relocation);

    Elf64_Shdr strings{};
    strings.sh_type = SHT_STRTAB;
    strings.sh_offset = strings_offset;
    strings.sh_size = sizeof(names);
    Elf64_Shdr symbols{};
    symbols.sh_type = SHT_SYMTAB;
    symbols.sh_offset = symbols_offset;
    symbols.sh_size = 2 * sizeof(Elf64_Sym);
    symbols.sh_link = 1;
    symbols.sh_entsize = sizeof(Elf64_Sym);
    Elf64_Shdr relocations{};
    relocations.sh_type = SHT_RELA;
    relocations.sh_offset = relocations_offset;
    relocations.sh_size = sizeof(Elf64_Rela);
    relocations.sh_link = 2;
    relocations.sh_entsize = sizeof(Elf64_Rela);
    WriteAt(bytes, header.e_shoff + sizeof(Elf64_Shdr), strings);
    WriteAt(bytes, header.e_shoff + 2 * sizeof(Elf64_Shdr), symbols);
    WriteAt(bytes, header.e_shoff + 3 * sizeof(Elf64_Shdr), relocations);
    return bytes;
}

std::vector<std::uint8_t> RelocationMatrixElf() {
    auto bytes = BasicElf(3);
    AddTls(bytes);
    constexpr char names[] = "\0matrix_symbol#mod#lib\0";
    std::memcpy(bytes.data() + StringFileOffset, names, sizeof(names));
    Elf64_Sym symbol{};
    symbol.st_name = 1;
    symbol.st_info = ELF64_ST_INFO(STB_GLOBAL, STT_TLS);
    symbol.st_shndx = 1;
    symbol.st_value = 0x108;
    symbol.st_size = 0x20;
    WriteAt(bytes, SymbolFileOffset + sizeof(Elf64_Sym), symbol);

    const std::array<std::uint32_t, 15> types{
        R_X86_64_64,       R_X86_64_PC32,     R_X86_64_PLT32,
        R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT, R_X86_64_RELATIVE,
        R_X86_64_32,       R_X86_64_32S,      R_X86_64_DTPMOD64,
        R_X86_64_DTPOFF64, R_X86_64_TPOFF64,  R_X86_64_PC64,
        R_X86_64_SIZE32,   R_X86_64_SIZE64,   R_X86_64_RELATIVE64,
    };
    std::array<Elf64_Rela, types.size()> relocations{};
    for (std::size_t index = 0; index < relocations.size(); ++index) {
        auto& relocation = relocations[index];
        relocation.r_offset = 0xa00 + index * 8;
        const auto type = types[index];
        const bool no_symbol =
            type == R_X86_64_RELATIVE ||
            type == R_X86_64_RELATIVE64 ||
            type == R_X86_64_DTPMOD64 ||
            type == R_X86_64_32 || type == R_X86_64_32S;
        relocation.r_info =
            (static_cast<std::uint64_t>(no_symbol ? 0 : 1) << 32u) |
            type;
        relocation.r_addend =
            type == R_X86_64_32S ? -7 :
            type == R_X86_64_32 ? 0x1234 :
            type == R_X86_64_RELATIVE ||
                    type == R_X86_64_RELATIVE64
                ? 0x321
                : 4;
    }
    std::memcpy(
        bytes.data() + RelaFileOffset, relocations.data(),
        sizeof(relocations));
    std::vector<Elf64_Dyn> dynamic(9);
    dynamic[0].d_tag = DT_STRTAB;
    dynamic[0].d_un.d_ptr = StringVirtualAddress;
    dynamic[1].d_tag = DT_STRSZ;
    dynamic[1].d_un.d_val = sizeof(names);
    dynamic[2].d_tag = DT_SYMTAB;
    dynamic[2].d_un.d_ptr = SymbolVirtualAddress;
    dynamic[3].d_tag = DT_SYMENT;
    dynamic[3].d_un.d_val = sizeof(Elf64_Sym);
    dynamic[4].d_tag = DtSceSymbolTableSize;
    dynamic[4].d_un.d_val = 2 * sizeof(Elf64_Sym);
    dynamic[5].d_tag = DT_RELA;
    dynamic[5].d_un.d_ptr = RelaVirtualAddress;
    dynamic[6].d_tag = DT_RELASZ;
    dynamic[6].d_un.d_val = sizeof(relocations);
    dynamic[7].d_tag = DT_RELAENT;
    dynamic[7].d_un.d_val = sizeof(Elf64_Rela);
    dynamic[8].d_tag = DT_NULL;
    AddDynamic(bytes, dynamic);
    return bytes;
}

std::vector<std::uint8_t> SceDynamicMetadataElf() {
    auto bytes = BasicElf(3);
    constexpr std::size_t metadata_file_offset =
        SegmentFileOffset + 0x400;
    constexpr std::size_t metadata_size = 0x300;
    Elf64_Phdr metadata{};
    metadata.p_type = PtSceDynlibData;
    metadata.p_flags = PF_R;
    metadata.p_offset = metadata_file_offset;
    metadata.p_filesz = metadata_size;
    metadata.p_memsz = metadata_size;
    metadata.p_align = 8;
    SetProgram(bytes, 2, metadata);

    constexpr char names[] = "\0sce_export#mod#lib\0libSynthetic.prx\0";
    std::memcpy(bytes.data() + metadata_file_offset, names, sizeof(names));
    Elf64_Sym symbol{};
    symbol.st_name = 1;
    symbol.st_info = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC);
    symbol.st_shndx = 1;
    symbol.st_value = 0x120;
    symbol.st_size = 8;
    WriteAt(
        bytes, metadata_file_offset + 0x40 + sizeof(Elf64_Sym),
        symbol);
    Elf64_Rela relocation{};
    relocation.r_offset = 0x180;
    relocation.r_info = ELF64_R_INFO(0, R_X86_64_RELATIVE);
    relocation.r_addend = 0x120;
    WriteAt(bytes, metadata_file_offset + 0x100, relocation);

    std::vector<Elf64_Dyn> dynamic(12);
    dynamic[0].d_tag = DtSceStringTable;
    dynamic[0].d_un.d_ptr = 0;
    dynamic[1].d_tag = DtSceStringSize;
    dynamic[1].d_un.d_val = sizeof(names);
    dynamic[2].d_tag = DtSceSymbolTable;
    dynamic[2].d_un.d_ptr = 0x40;
    dynamic[3].d_tag = DtSceSymbolEntry;
    dynamic[3].d_un.d_val = sizeof(Elf64_Sym);
    dynamic[4].d_tag = DtSceSymbolTableSize;
    dynamic[4].d_un.d_val = 2 * sizeof(Elf64_Sym);
    dynamic[5].d_tag = DtSceRela;
    dynamic[5].d_un.d_ptr = 0x100;
    dynamic[6].d_tag = DtSceRelaSize;
    dynamic[6].d_un.d_val = sizeof(Elf64_Rela);
    dynamic[7].d_tag = DtSceRelaEntry;
    dynamic[7].d_un.d_val = sizeof(Elf64_Rela);
    dynamic[8].d_tag = DT_NEEDED;
    dynamic[8].d_un.d_val =
        std::string_view{names, sizeof(names)}.find("libSynthetic");
    dynamic[9].d_tag = DT_NULL;
    dynamic.resize(10);
    AddDynamic(bytes, dynamic);
    return bytes;
}

struct Api {
    decltype(&executor_lsx4_ps5_runtime_initialize) initialize{};
    decltype(&executor_lsx4_ps5_runtime_load_eboot) load{};
    decltype(&executor_lsx4_ps5_runtime_unload_eboot) unload{};
    decltype(&executor_lsx4_ps5_runtime_probe_report) probe{};
    decltype(&executor_lsx4_ps5_runtime_status) status{};
    decltype(&executor_lsx4_ps5_runtime_reset) reset{};
};

struct Matrix {
    Api api;
    std::string root;
    int total{};
    int passed{};

    bool Run(
        const std::string& name, const std::vector<std::uint8_t>& bytes,
        const bool expect_success, const std::string_view status_fragment = {},
        const std::function<bool(const Lsx4Ps5LoadedExecutable&)>& inspect = {}) {
        ++total;
        const auto path = root + "/" + name + ".bin";
        if (!WriteFile(path, bytes)) {
            std::fprintf(stderr, "FAIL %-30s write errno=%d\n",
                         name.c_str(), errno);
            return false;
        }
        Lsx4Ps5LoadedExecutable executable{};
        const auto result = api.load(path.c_str(), &executable);
        const std::string runtime_status =
            api.status() != nullptr ? api.status() : "";
        bool ok = expect_success ? result == 0 : result != 0;
        if (!status_fragment.empty() &&
            runtime_status.find(status_fragment) == std::string::npos) {
            ok = false;
        }
        if (ok && result == 0 && inspect) {
            ok = inspect(executable);
        }
        if (result == 0) {
            if (api.unload(executable.handle) != 0) {
                ok = false;
            }
        }
        unlink(path.c_str());
        if (!ok) {
            std::fprintf(
                stderr, "FAIL %-30s rc=%d status=%s\n",
                name.c_str(), result, runtime_status.c_str());
            return false;
        }
        ++passed;
        std::printf("PASS %-30s\n", name.c_str());
        return true;
    }
};

std::vector<Elf64_Dyn> BasicDynamic() {
    std::vector<Elf64_Dyn> dynamic(2);
    dynamic[0].d_tag = DT_INIT;
    dynamic[0].d_un.d_ptr = 0;
    dynamic[1].d_tag = DT_NULL;
    return dynamic;
}

}

int main(const int argc, char** const argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s liblsx4_executor_ps5_android.so\n",
                     argv[0]);
        return 2;
    }
    void* const library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 3;
    }
    Api api{
        .initialize = Resolve<decltype(Api::initialize)>(
            library, "executor_lsx4_ps5_runtime_initialize"),
        .load = Resolve<decltype(Api::load)>(
            library, "executor_lsx4_ps5_runtime_load_eboot"),
        .unload = Resolve<decltype(Api::unload)>(
            library, "executor_lsx4_ps5_runtime_unload_eboot"),
        .probe = Resolve<decltype(Api::probe)>(
            library, "executor_lsx4_ps5_runtime_probe_report"),
        .status = Resolve<decltype(Api::status)>(
            library, "executor_lsx4_ps5_runtime_status"),
        .reset = Resolve<decltype(Api::reset)>(
            library, "executor_lsx4_ps5_runtime_reset"),
    };
    if (api.initialize == nullptr || api.load == nullptr ||
        api.unload == nullptr || api.probe == nullptr ||
        api.status == nullptr || api.reset == nullptr) {
        std::fprintf(stderr, "missing PS5 runtime ABI export\n");
        return 4;
    }

    api.reset();
    Lsx4Ps5LoadedExecutable before_initialize{};
    if (api.load("/does/not/exist", &before_initialize) !=
        LSX4_PS5_EXECUTE_NOT_INITIALIZED) {
        std::fprintf(stderr, "load-before-initialize policy failed\n");
        return 5;
    }
    Lsx4Ps5RuntimeConfig invalid_config{};
    invalid_config.guest_page_size = 0x1000;
    if (api.initialize(&invalid_config) !=
        LSX4_PS5_EXECUTE_INVALID_ARGUMENT) {
        std::fprintf(stderr, "guest-page ABI gate failed\n");
        return 6;
    }
    Lsx4Ps5RuntimeConfig config{};
    config.title_id = "MATRIX00001";
    if (api.initialize(&config) != 0) {
        std::fprintf(stderr, "initialize: %s\n", api.status());
        return 7;
    }

    const std::string root =
        "/data/local/tmp/lsx4-ps5-loader-matrix-" +
        std::to_string(static_cast<long long>(getpid()));
    if (mkdir(root.c_str(), 0700) != 0) {
        std::fprintf(stderr, "mkdir %s: errno=%d\n", root.c_str(), errno);
        return 8;
    }
    Matrix matrix{.api = api, .root = root};
    bool all = true;

    const auto baseline = BasicElf();
    all &= matrix.Run("elf-baseline", baseline, true);
    all &= matrix.Run("self-ps4-magic", WrapSelf(baseline, false), true);
    all &= matrix.Run("self-ps5-magic", WrapSelf(baseline, true), true);
    all &= matrix.Run(
        "self-stale-section-fallback",
        WrapSelfWithStaleSections(baseline), true);

    all &= matrix.Run("empty-file", {}, false, "truncated ELF header");
    auto value = baseline;
    value[0] = 0;
    all &= matrix.Run("bad-magic", value, false, "unsupported PS5 ELF identity");
    value = baseline;
    value[EI_CLASS] = ELFCLASS32;
    all &= matrix.Run("bad-class", value, false, "unsupported PS5 ELF identity");
    value = baseline;
    value[EI_DATA] = ELFDATA2MSB;
    all &= matrix.Run("bad-endian", value, false, "unsupported PS5 ELF identity");
    value = baseline;
    auto header = Header(value);
    header.e_machine = EM_AARCH64;
    SetHeader(value, header);
    all &= matrix.Run("bad-machine", value, false, "unsupported PS5 ELF identity");
    value = baseline;
    header = Header(value);
    header.e_type = ET_EXEC;
    SetHeader(value, header);
    all &= matrix.Run("bad-type", value, false, "unsupported PS5 ELF identity");
    value = baseline;
    header = Header(value);
    header.e_phentsize = sizeof(Elf64_Phdr) - 1;
    SetHeader(value, header);
    all &= matrix.Run("bad-phentsize", value, false, "unsupported PS5 ELF identity");
    value = baseline;
    header = Header(value);
    header.e_phnum = 0;
    SetHeader(value, header);
    all &= matrix.Run("zero-phnum", value, false, "unsupported PS5 ELF identity");
    value = baseline;
    header = Header(value);
    header.e_phoff = value.size() - 4;
    SetHeader(value, header);
    all &= matrix.Run("truncated-phdr", value, false, "truncated ELF program table");
    value = baseline;
    auto program = Program(value, 0);
    program.p_type = PT_NOTE;
    SetProgram(value, 0, program);
    all &= matrix.Run("no-load-segment", value, false, "no loadable image");
    value = baseline;
    program = Program(value, 0);
    program.p_filesz = program.p_memsz + 1;
    SetProgram(value, 0, program);
    all &= matrix.Run("filesz-over-memsz", value, false, "file size exceeds");
    value = baseline;
    program = Program(value, 0);
    program.p_vaddr = std::numeric_limits<std::uint64_t>::max() - 0x100;
    SetProgram(value, 0, program);
    all &= matrix.Run("vaddr-overflow", value, false, "size overflow");
    value = baseline;
    header = Header(value);
    header.e_entry = 0x8000;
    SetHeader(value, header);
    all &= matrix.Run("entry-outside", value, false);

    value = BasicElf(3);
    AddTls(value, 2, 3);
    all &= matrix.Run("tls-bad-alignment", value, false, "invalid or duplicate");
    value = BasicElf(3);
    AddTls(value, 2, 0x20, 0x30, 0x20);
    all &= matrix.Run("tls-filesz-over-memsz", value, false, "invalid or duplicate");
    value = BasicElf(4);
    AddTls(value, 2);
    AddTls(value, 3);
    all &= matrix.Run("tls-duplicate", value, false, "invalid or duplicate");

    value = baseline;
    auto self = WrapSelf(value, true);
    auto self_segment = ReadAt<SelfSegment>(self, sizeof(SelfHeader));
    self_segment.type = 0;
    WriteAt(self, sizeof(SelfHeader), self_segment);
    all &= matrix.Run("self-missing-plain-segment", self, false, "no unencrypted");
    self = WrapSelf(value, true);
    self_segment = ReadAt<SelfSegment>(self, sizeof(SelfHeader));
    self_segment.compressed_size -= 1;
    WriteAt(self, sizeof(SelfHeader), self_segment);
    all &= matrix.Run("self-compressed-segment", self, false, "compressed or encrypted");
    self = WrapSelf(value, true);
    self.resize(self.size() - 1);
    all &= matrix.Run("self-truncated-segment", self, false, "truncated PT_LOAD");
    value = BasicElf(2);
    AddDynamic(value, BasicDynamic());
    self = WrapSelf(value, true);
    all &= matrix.Run("self-dynamic-from-mapped-load", self, true);

    value = BasicElf(2);
    AddDynamic(value, BasicDynamic());
    program = Program(value, 1);
    program.p_filesz = 17;
    SetProgram(value, 1, program);
    all &= matrix.Run("dynamic-truncated", value, false, "truncated PT_DYNAMIC");

    value = BasicElf(2);
    std::vector<Elf64_Dyn> dynamic(4);
    dynamic[0].d_tag = DT_RELASZ;
    dynamic[0].d_un.d_val = sizeof(Elf64_Rela);
    dynamic[1].d_tag = DT_SYMENT;
    dynamic[1].d_un.d_val = 1;
    dynamic[2].d_tag = DT_RELAENT;
    dynamic[2].d_un.d_val = sizeof(Elf64_Rela);
    dynamic[3].d_tag = DT_NULL;
    AddDynamic(value, dynamic);
    all &= matrix.Run("dynamic-bad-syment", value, false, "symbol table entry size");
    value = BasicElf(2);
    dynamic[1].d_tag = DT_SYMENT;
    dynamic[1].d_un.d_val = sizeof(Elf64_Sym);
    dynamic[2].d_un.d_val = 1;
    AddDynamic(value, dynamic);
    all &= matrix.Run("dynamic-bad-relaent", value, false, "relocation table entry size");

    value = BasicElf(2);
    dynamic.assign(5, {});
    dynamic[0].d_tag = DT_STRTAB;
    dynamic[0].d_un.d_ptr = StringVirtualAddress;
    dynamic[1].d_tag = DT_STRSZ;
    dynamic[1].d_un.d_val = 4;
    dynamic[2].d_tag = DT_NEEDED;
    dynamic[2].d_un.d_val = 5;
    dynamic[3].d_tag = DT_NULL;
    AddDynamic(value, dynamic);
    all &= matrix.Run("needed-bad-offset", value, false, "dynamic string offset");
    value = BasicElf(2);
    std::memcpy(value.data() + StringFileOffset, "abcd", 4);
    dynamic[2].d_un.d_val = 0;
    AddDynamic(value, dynamic);
    all &= matrix.Run("needed-unterminated", value, false, "unterminated");

    value = BasicElf(2);
    dynamic.assign(5, {});
    dynamic[0].d_tag = DT_SYMTAB;
    dynamic[0].d_un.d_ptr = SymbolVirtualAddress;
    dynamic[1].d_tag = DT_SYMENT;
    dynamic[1].d_un.d_val = sizeof(Elf64_Sym);
    dynamic[2].d_tag = DtSceSymbolTableSize;
    dynamic[2].d_un.d_val = sizeof(Elf64_Sym) + 1;
    dynamic[3].d_tag = DT_NULL;
    AddDynamic(value, dynamic);
    all &= matrix.Run("symbol-table-misaligned", value, false, "dynamic symbol table");

    value = BasicElf(2);
    Elf64_Rela relocation{};
    relocation.r_offset = 0x5000;
    relocation.r_info = ELF64_R_INFO(0, R_X86_64_RELATIVE);
    WriteAt(value, RelaFileOffset, relocation);
    dynamic.assign(4, {});
    dynamic[0].d_tag = DT_RELA;
    dynamic[0].d_un.d_ptr = RelaVirtualAddress;
    dynamic[1].d_tag = DT_RELASZ;
    dynamic[1].d_un.d_val = sizeof(Elf64_Rela);
    dynamic[2].d_tag = DT_RELAENT;
    dynamic[2].d_un.d_val = sizeof(Elf64_Rela);
    dynamic[3].d_tag = DT_NULL;
    AddDynamic(value, dynamic);
    all &= matrix.Run("relocation-target-outside", value, false, "target is outside");
    value = BasicElf(2);
    relocation.r_offset = 0x180;
    relocation.r_info = ELF64_R_INFO(0, R_X86_64_IRELATIVE);
    WriteAt(value, RelaFileOffset, relocation);
    AddDynamic(value, dynamic);
    all &= matrix.Run("relocation-unsupported", value, false, "unsupported PS5 relocation type");
    value = BasicElf(2);
    relocation.r_info = ELF64_R_INFO(0, R_X86_64_32);
    relocation.r_addend = 0x100000000ll;
    WriteAt(value, RelaFileOffset, relocation);
    AddDynamic(value, dynamic);
    all &= matrix.Run("relocation-u32-overflow", value, false, "unsigned 32-bit");
    value = BasicElf(2);
    relocation.r_info = ELF64_R_INFO(0, R_X86_64_32S);
    relocation.r_addend = 0x80000000ll;
    WriteAt(value, RelaFileOffset, relocation);
    AddDynamic(value, dynamic);
    all &= matrix.Run("relocation-s32-overflow", value, false, "signed 32-bit");

    value = BasicElf(2);
    dynamic = BasicDynamic();
    dynamic[0].d_un.d_ptr = 0x5000;
    AddDynamic(value, dynamic);
    all &= matrix.Run("initializer-outside", value, false, "initializer points outside");

    value = SectionFallbackElf();
    all &= matrix.Run(
        "section-symbol-rela-fallback", value, true, {},
        [&](const Lsx4Ps5LoadedExecutable& executable) {
            std::uint64_t patched{};
            std::memcpy(
                &patched,
                reinterpret_cast<const void*>(executable.base + 0x180),
                sizeof(patched));
            Lsx4Ps5GameProbeReport report{};
            return patched == executable.base + 0x104 &&
                   api.probe(
                       executable.handle, &report, nullptr, 0) == 0 &&
                   report.exported_symbols >= 3;
        });
    value = SectionFallbackElf();
    auto section_header = Header(value);
    auto rela_section = ReadAt<Elf64_Shdr>(
        value, section_header.e_shoff + 3 * sizeof(Elf64_Shdr));
    rela_section.sh_entsize = sizeof(Elf64_Rela) - 1;
    WriteAt(
        value, section_header.e_shoff + 3 * sizeof(Elf64_Shdr),
        rela_section);
    all &= matrix.Run("section-rela-bad-entsize", value, false, "section relocation table");

    value = RelocationMatrixElf();
    all &= matrix.Run(
        "all-supported-relocations", value, true, {},
        [](const Lsx4Ps5LoadedExecutable& executable) {
            const auto read64 = [&](const std::size_t index) {
                std::uint64_t result{};
                std::memcpy(
                    &result,
                    reinterpret_cast<const void*>(
                        executable.base + 0xa00 + index * 8),
                    sizeof(result));
                return result;
            };
            const auto read32 = [&](const std::size_t index) {
                std::uint32_t result{};
                std::memcpy(
                    &result,
                    reinterpret_cast<const void*>(
                        executable.base + 0xa00 + index * 8),
                    sizeof(result));
                return result;
            };
            const auto patch = [&](const std::size_t index) {
                return executable.base + 0xa00 + index * 8;
            };
            return read64(0) == executable.base + 0x10c &&
                   static_cast<std::int32_t>(read32(1)) ==
                       static_cast<std::int32_t>(
                           executable.base + 0x10c - patch(1)) &&
                   static_cast<std::int32_t>(read32(2)) ==
                       static_cast<std::int32_t>(
                           executable.base + 0x10c - patch(2)) &&
                   read64(3) == executable.base + 0x108 &&
                   read64(4) == executable.base + 0x108 &&
                   read64(5) == executable.base + 0x321 &&
                   read32(6) == 0x1234 &&
                   static_cast<std::int32_t>(read32(7)) == -7 &&
                   read64(8) == 1 &&
                   read64(9) == 0x10c &&
                   read64(10) ==
                       static_cast<std::uint64_t>(0x10c - 0x20) &&
                   read64(11) ==
                       executable.base + 0x10c - patch(11) &&
                   read32(12) == 0x24 &&
                   read64(13) == 0x24 &&
                   read64(14) == executable.base + 0x321;
        });

    value = SceDynamicMetadataElf();
    all &= matrix.Run(
        "sce-dynamic-metadata", value, true, {},
        [&](const Lsx4Ps5LoadedExecutable& executable) {
            std::uint64_t patched{};
            std::memcpy(
                &patched,
                reinterpret_cast<const void*>(executable.base + 0x180),
                sizeof(patched));
            Lsx4Ps5GameProbeReport report{};
            return patched == executable.base + 0x120 &&
                   api.probe(
                       executable.handle, &report, nullptr, 0) == 0 &&
                   report.exported_symbols >= 2 &&
                   report.dependencies == 1;
        });

    api.reset();
    rmdir(root.c_str());
    dlclose(library);
    std::printf(
        "PS5_LOADER_MATRIX_%s passed=%d total=%d\n",
        all && matrix.passed == matrix.total ? "OK" : "FAILED",
        matrix.passed, matrix.total);
    return all && matrix.passed == matrix.total ? 0 : 9;
}
