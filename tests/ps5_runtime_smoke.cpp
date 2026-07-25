// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/ps5_desktop/runtime_api.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

#pragma pack(push, 1)
struct SmokeSelfHeader {
    std::uint8_t ident[12]{};
    std::uint16_t size1{};
    std::uint16_t size2{};
    std::uint64_t file_size{};
    std::uint16_t segment_count{};
    std::uint16_t unknown{};
    std::uint32_t padding{};
};

struct SmokeSelfSegment {
    std::uint64_t type{};
    std::uint64_t offset{};
    std::uint64_t compressed_size{};
    std::uint64_t decompressed_size{};
};
#pragma pack(pop)

bool WriteSyntheticNextGenEboot(const char* const path) {
    constexpr std::size_t segment_offset = 0x4000;
    constexpr std::size_t container_data_offset = 0x5000;
    constexpr std::size_t segment_size = 0x600;
    constexpr std::size_t elf_offset =
        sizeof(SmokeSelfHeader) + 2 * sizeof(SmokeSelfSegment);
    std::vector<std::uint8_t> image(
        container_data_offset + segment_size);
    SmokeSelfHeader self{};
    self.ident[0] = 0x4f;
    self.ident[1] = 0x15;
    self.ident[2] = 0x3d;
    self.ident[3] = 0x1d;
    self.segment_count = 2;
    self.unknown = 0x22;
    self.file_size = image.size();
    std::array<SmokeSelfSegment, 2> self_segments{};
    self_segments[0].type = 0x800;
    self_segments[0].offset = container_data_offset;
    self_segments[0].compressed_size = segment_size;
    self_segments[0].decompressed_size = segment_size;
    self_segments[1].type = 0x800 | (UINT64_C(1) << 20u);
    self_segments[1].offset = container_data_offset + 0x200;
    self_segments[1].compressed_size = 11 * sizeof(Elf64_Dyn);
    self_segments[1].decompressed_size = self_segments[1].compressed_size;
    std::memcpy(image.data(), &self, sizeof(self));
    std::memcpy(image.data() + sizeof(self), self_segments.data(),
                sizeof(self_segments));

    Elf64_Ehdr header{};
    std::memcpy(header.e_ident, ELFMAG, SELFMAG);
    header.e_ident[EI_CLASS] = ELFCLASS64;
    header.e_ident[EI_DATA] = ELFDATA2LSB;
    header.e_ident[EI_VERSION] = EV_CURRENT;
    header.e_ident[EI_OSABI] = ELFOSABI_FREEBSD;
    header.e_ident[EI_ABIVERSION] = 2;
    header.e_type = 0xfe10;
    header.e_machine = EM_X86_64;
    header.e_version = EV_CURRENT;
    header.e_entry = 0;
    header.e_phoff = sizeof(Elf64_Ehdr);
    header.e_ehsize = sizeof(Elf64_Ehdr);
    header.e_phentsize = sizeof(Elf64_Phdr);
    header.e_phnum = 3;

    std::array<Elf64_Phdr, 3> programs{};
    programs[0].p_type = PT_LOAD;
    programs[0].p_flags = PF_R | PF_W | PF_X;
    programs[0].p_offset = segment_offset;
    programs[0].p_vaddr = 0;
    programs[0].p_filesz = segment_size;
    programs[0].p_memsz = 0x4000;
    programs[0].p_align = 0x4000;
    programs[1].p_type = PT_DYNAMIC;
    programs[1].p_flags = PF_R | PF_W;
    programs[1].p_offset = segment_offset + 0x200;
    programs[1].p_vaddr = 0x200;
    programs[1].p_filesz = 11 * sizeof(Elf64_Dyn);
    programs[1].p_memsz = programs[1].p_filesz;
    programs[1].p_align = 8;
    programs[2].p_type = PT_TLS;
    programs[2].p_flags = PF_R;
    programs[2].p_offset = segment_offset + 0x580;
    programs[2].p_vaddr = 0x580;
    programs[2].p_filesz = sizeof(std::uint64_t);
    programs[2].p_memsz = 0x20;
    programs[2].p_align = 0x20;
    std::memcpy(image.data() + elf_offset, &header, sizeof(header));
    std::memcpy(image.data() + elf_offset + header.e_phoff, programs.data(),
                sizeof(programs));

    std::vector<std::uint8_t> code;
    const auto emit = [&](const std::initializer_list<std::uint8_t> bytes) {
        code.insert(code.end(), bytes.begin(), bytes.end());
    };
    const auto emit_u64 = [&](const std::uint64_t value) {
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
        code.insert(code.end(), bytes, bytes + sizeof(value));
    };
    emit({0x48, 0xb8});
    emit_u64(0x1122334455667788ull);
    emit({0x66, 0x48, 0x0f, 0x6e, 0xc0});
    emit({0x66, 0x0f, 0x78, 0xc0, 0x08, 0x08});
    emit({0x66, 0x0f, 0xef, 0xc9});
    emit({0xf2, 0x0f, 0x78, 0xc8, 0x08, 0x10});
    emit({0x66, 0x48, 0x0f, 0x7e, 0xcf});
    emit({0x0f, 0x01, 0xfa});
    emit({0x0f, 0x01, 0xfb});
    // mov rdi, qword ptr fs:[-0x20] -- main-module Variant II TLS.
    emit({0x64, 0x48, 0x8b, 0x3c, 0x25, 0xe0, 0xff, 0xff, 0xff});
    // mov rsi, qword ptr fs:[0] -- TCB self pointer.
    emit({0x64, 0x48, 0x8b, 0x34, 0x25, 0x00, 0x00, 0x00, 0x00});
    const auto call_offset = code.size();
    emit({0xff, 0x15, 0x00, 0x00, 0x00, 0x00});
    const auto module_call_offset = code.size();
    emit({0xff, 0x15, 0x00, 0x00, 0x00, 0x00});
    emit({0xc3});
    const auto displacement = static_cast<std::int32_t>(
        0x100 - (call_offset + 6));
    std::memcpy(code.data() + call_offset + 2,
                 &displacement, sizeof(displacement));
    const auto module_displacement = static_cast<std::int32_t>(
        0x130 - (module_call_offset + 6));
    std::memcpy(code.data() + module_call_offset + 2,
                &module_displacement, sizeof(module_displacement));
    std::memcpy(image.data() + container_data_offset,
                code.data(), code.size());
    const std::array<std::uint8_t, 12> initializer{
        0x48, 0xc7, 0x05, 0x9d, 0x00, 0x00,
        0x00, 0x34, 0x12, 0x00, 0x00, 0xc3};
    std::memcpy(image.data() + container_data_offset + 0x80,
                initializer.data(), initializer.size());

    std::array<Elf64_Dyn, 11> dynamic{};
    dynamic[0].d_tag = DT_STRTAB;
    dynamic[0].d_un.d_ptr = 0x500;
    dynamic[1].d_tag = DT_STRSZ;
    dynamic[1].d_un.d_val = 32;
    dynamic[2].d_tag = DT_SYMTAB;
    dynamic[2].d_un.d_ptr = 0x400;
    dynamic[3].d_tag = DT_SYMENT;
    dynamic[3].d_un.d_val = sizeof(Elf64_Sym);
    dynamic[4].d_tag = DT_RELA;
    dynamic[4].d_un.d_ptr = 0x340;
    dynamic[5].d_tag = DT_RELASZ;
    dynamic[5].d_un.d_val = 4 * sizeof(Elf64_Rela);
    dynamic[6].d_tag = DT_RELAENT;
    dynamic[6].d_un.d_val = sizeof(Elf64_Rela);
    dynamic[7].d_tag = DT_JMPREL;
    dynamic[7].d_un.d_ptr = 0x300;
    dynamic[8].d_tag = DT_PLTRELSZ;
    dynamic[8].d_un.d_val = 2 * sizeof(Elf64_Rela);
    dynamic[9].d_tag = DT_INIT;
    dynamic[9].d_un.d_ptr = 0x80;
    dynamic[10].d_tag = DT_NULL;
    std::memcpy(image.data() + container_data_offset + 0x200,
                dynamic.data(), sizeof(dynamic));

    std::array<Elf64_Rela, 2> import_relocations{};
    import_relocations[0].r_offset = 0x100;
    import_relocations[0].r_info =
        ELF64_R_INFO(1, R_X86_64_JUMP_SLOT);
    import_relocations[1].r_offset = 0x130;
    import_relocations[1].r_info =
        ELF64_R_INFO(3, R_X86_64_JUMP_SLOT);
    std::memcpy(image.data() + container_data_offset + 0x300,
                import_relocations.data(), sizeof(import_relocations));
    std::array<Elf64_Rela, 4> general_relocations{};
    general_relocations[0].r_offset = 0x108;
    general_relocations[0].r_info = ELF64_R_INFO(2, R_X86_64_64);
    general_relocations[0].r_addend = 4;
    general_relocations[1].r_offset = 0x110;
    general_relocations[1].r_info = ELF64_R_INFO(0, R_X86_64_RELATIVE64);
    general_relocations[1].r_addend = 0x580;
    general_relocations[2].r_offset = 0x118;
    general_relocations[2].r_info = ELF64_R_INFO(2, R_X86_64_SIZE32);
    general_relocations[3].r_offset = 0x11c;
    general_relocations[3].r_info = ELF64_R_INFO(2, R_X86_64_PC32);
    general_relocations[3].r_addend = -4;
    std::memcpy(image.data() + container_data_offset + 0x340,
                 general_relocations.data(), sizeof(general_relocations));
    std::array<Elf64_Sym, 4> symbols{};
    symbols[1].st_name = 1;
    symbols[1].st_info = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC);
    symbols[2].st_info = ELF64_ST_INFO(STB_LOCAL, STT_OBJECT);
    symbols[2].st_shndx = 1;
    symbols[2].st_value = 0x500;
    symbols[2].st_size = 0x1234;
    symbols[3].st_name = 13;
    symbols[3].st_info = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC);
    std::memcpy(image.data() + container_data_offset + 0x400,
                symbols.data(), sizeof(symbols));
    constexpr std::array<char, 32> strings{
        '\0', 's', 'm', 'o', 'k', 'e', 'I', 'm',
        'p', 'o', 'r', 't', '\0',
        'm', 'o', 'd', 'u', 'l', 'e', 'E', 'x',
        'p', 'o', 'r', 't', '\0'};
    std::memcpy(image.data() + container_data_offset + 0x500,
                strings.data(), strings.size());
    constexpr std::uint64_t tls_initial_value = 0x770000;
    std::memcpy(image.data() + container_data_offset + 0x580,
                &tls_initial_value, sizeof(tls_initial_value));

    const int descriptor =
        open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (descriptor < 0) {
        return false;
    }
    const auto written = write(descriptor, image.data(), image.size());
    close(descriptor);
    return written == static_cast<ssize_t>(image.size());
}

bool WriteSyntheticNextGenModule(const char* const path) {
    constexpr std::size_t segment_offset = 0x4000;
    constexpr std::size_t segment_size = 0x400;
    std::vector<std::uint8_t> image(segment_offset + segment_size);

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
    header.e_phnum = 3;

    std::array<Elf64_Phdr, 3> programs{};
    programs[0].p_type = PT_LOAD;
    // Real PS5 modules commonly mark text PF_X without an explicit PF_R.
    programs[0].p_flags = PF_X;
    programs[0].p_offset = segment_offset;
    programs[0].p_vaddr = 0;
    programs[0].p_filesz = segment_size;
    programs[0].p_memsz = 0x4000;
    programs[0].p_align = 0x4000;
    programs[1].p_type = PT_DYNAMIC;
    programs[1].p_flags = PF_R;
    programs[1].p_offset = segment_offset + 0x100;
    programs[1].p_vaddr = 0x100;
    programs[1].p_filesz = 11 * sizeof(Elf64_Dyn);
    programs[1].p_memsz = programs[1].p_filesz;
    programs[1].p_align = 8;
    programs[2].p_type = PT_TLS;
    programs[2].p_flags = PF_R;
    programs[2].p_offset = segment_offset + 0x300;
    programs[2].p_vaddr = 0x300;
    programs[2].p_filesz = sizeof(std::uint64_t);
    programs[2].p_memsz = 0x20;
    programs[2].p_align = 0x20;
    std::memcpy(image.data(), &header, sizeof(header));
    std::memcpy(image.data() + header.e_phoff,
                programs.data(), sizeof(programs));

    const std::array<std::uint8_t, 10> initializer{
        0x64, 0x48, 0x8b, 0x04, 0x25,
        0xc0, 0xff, 0xff, 0xff, 0xc3};
    std::memcpy(image.data() + segment_offset,
                initializer.data(), initializer.size());
    const std::array<std::uint8_t, 11> finalizer{
        0x48, 0xb8, 0xbc, 0x9a, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xc3};
    std::memcpy(image.data() + segment_offset + 0x20,
                 finalizer.data(), finalizer.size());
    const std::array<std::uint8_t, 11> exported_function{
        0x48, 0xb8, 0x68, 0x24, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xc3};
    std::memcpy(image.data() + segment_offset + 0x40,
                exported_function.data(), exported_function.size());
    std::array<Elf64_Dyn, 11> dynamic{};
    dynamic[0].d_tag = DT_RELA;
    dynamic[0].d_un.d_ptr = 0x1e0;
    dynamic[1].d_tag = DT_RELASZ;
    dynamic[1].d_un.d_val = sizeof(Elf64_Rela);
    dynamic[2].d_tag = DT_RELAENT;
    dynamic[2].d_un.d_val = sizeof(Elf64_Rela);
    dynamic[3].d_tag = DT_INIT;
    dynamic[3].d_un.d_ptr = 0;
    dynamic[4].d_tag = DT_FINI;
    dynamic[4].d_un.d_ptr = 0x20;
    dynamic[5].d_tag = DT_SYMTAB;
    dynamic[5].d_un.d_ptr = 0x220;
    dynamic[6].d_tag = DT_STRTAB;
    dynamic[6].d_un.d_ptr = 0x250;
    dynamic[7].d_tag = DT_STRSZ;
    dynamic[7].d_un.d_val = 34;
    dynamic[8].d_tag = DT_SYMENT;
    dynamic[8].d_un.d_val = sizeof(Elf64_Sym);
    dynamic[9].d_tag = DT_NEEDED;
    dynamic[9].d_un.d_val = 20;
    dynamic[10].d_tag = DT_NULL;
    std::memcpy(image.data() + segment_offset + 0x100,
                dynamic.data(), sizeof(dynamic));
    Elf64_Rela tls_module_relocation{};
    tls_module_relocation.r_offset = 0x1a0;
    tls_module_relocation.r_info =
        ELF64_R_INFO(0, R_X86_64_DTPMOD64);
    tls_module_relocation.r_offset = 0x320;
    std::memcpy(image.data() + segment_offset + 0x1e0,
                 &tls_module_relocation,
                 sizeof(tls_module_relocation));
    std::array<Elf64_Sym, 2> symbols{};
    symbols[1].st_name = 1;
    symbols[1].st_info = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC);
    symbols[1].st_shndx = 1;
    symbols[1].st_value = 0x40;
    symbols[1].st_size = exported_function.size();
    std::memcpy(image.data() + segment_offset + 0x220,
                symbols.data(), sizeof(symbols));
    constexpr std::array<char, 34> strings{
        '\0', 'm', 'o', 'd', 'u', 'l', 'e', 'E', 'x', 'p',
        'o', 'r', 't', '#', 'N', 'I', 'D', '0', '0', '\0',
        'l', 'i', 'b', 'k', 'e', 'r', 'n', 'e', 'l', '.',
        'p', 'r', 'x', '\0'};
    std::memcpy(image.data() + segment_offset + 0x250,
                strings.data(), strings.size());
    constexpr std::uint64_t module_tls_initial_value = 0x5678;
    std::memcpy(image.data() + segment_offset + 0x300,
                 &module_tls_initial_value,
                sizeof(module_tls_initial_value));

    const int descriptor =
        open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (descriptor < 0) {
        return false;
    }
    const auto written = write(descriptor, image.data(), image.size());
    close(descriptor);
    return written == static_cast<ssize_t>(image.size());
}

struct HleSmokeState {
    int binds{};
    int invokes{};
    bool argument_ok{};
    std::uint64_t expected_fs_base{};
};

int BindSmokeImport(void* const context, const char* const symbol,
                     const std::uint64_t, std::uint64_t* const native) {
    auto& state = *static_cast<HleSmokeState*>(context);
    if (symbol == nullptr || native == nullptr) {
        return -1;
    }
    if (std::strcmp(symbol, "moduleExport") == 0) {
        *native = 0;
        return 0;
    }
    if (std::strcmp(symbol, "smokeImport") != 0) {
        return -1;
    }
    ++state.binds;
    *native = 0;
    return 0;
}

std::uint64_t InvokeSmokeImport(
    void* const context, const Lsx4Ps5HleCall* const call) {
    auto& state = *static_cast<HleSmokeState*>(context);
    if (call == nullptr || call->function == 0) {
        return 0;
    }
    ++state.invokes;
    state.argument_ok =
        call->integer_arguments[0] == 0x770000 &&
        (state.expected_fs_base == 0
             ? call->integer_arguments[1] != 0
             : call->integer_arguments[1] == state.expected_fs_base);
    return 0x1234;
}

template <typename Function>
Function Resolve(void* const library, const char* const name) {
    return reinterpret_cast<Function>(dlsym(library, name));
}

} // namespace

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

    const auto initialize = Resolve<
        decltype(&executor_lsx4_ps5_runtime_initialize)>(
        library, "executor_lsx4_ps5_runtime_initialize");
    const auto load = Resolve<
        decltype(&executor_lsx4_ps5_runtime_load_eboot)>(
        library, "executor_lsx4_ps5_runtime_load_eboot");
    const auto load_module = Resolve<
        decltype(&executor_lsx4_ps5_runtime_load_module)>(
        library, "executor_lsx4_ps5_runtime_load_module");
    const auto load_game = Resolve<
        decltype(&executor_lsx4_ps5_runtime_load_game)>(
        library, "executor_lsx4_ps5_runtime_load_game");
    const auto probe_report = Resolve<
        decltype(&executor_lsx4_ps5_runtime_probe_report)>(
        library, "executor_lsx4_ps5_runtime_probe_report");
    const auto start_module = Resolve<
        decltype(&executor_lsx4_ps5_runtime_start_module)>(
        library, "executor_lsx4_ps5_runtime_start_module");
    const auto stop_module = Resolve<
        decltype(&executor_lsx4_ps5_runtime_stop_module)>(
        library, "executor_lsx4_ps5_runtime_stop_module");
    const auto set_callbacks = Resolve<
        decltype(&executor_lsx4_ps5_runtime_set_callbacks)>(
        library, "executor_lsx4_ps5_runtime_set_callbacks");
    const auto register_mapping = Resolve<
        decltype(&executor_lsx4_ps5_runtime_register_mapping)>(
        library, "executor_lsx4_ps5_runtime_register_mapping");
    const auto unregister_mapping = Resolve<
        decltype(&executor_lsx4_ps5_runtime_unregister_mapping)>(
        library, "executor_lsx4_ps5_runtime_unregister_mapping");
    const auto launch = Resolve<
        decltype(&executor_lsx4_ps5_runtime_launch_eboot)>(
        library, "executor_lsx4_ps5_runtime_launch_eboot");
    const auto create_thread = Resolve<
        decltype(&executor_lsx4_ps5_runtime_create_thread_context)>(
        library, "executor_lsx4_ps5_runtime_create_thread_context");
    const auto destroy_thread = Resolve<
        decltype(&executor_lsx4_ps5_runtime_destroy_thread_context)>(
        library, "executor_lsx4_ps5_runtime_destroy_thread_context");
    const auto unload = Resolve<
        decltype(&executor_lsx4_ps5_runtime_unload_eboot)>(
        library, "executor_lsx4_ps5_runtime_unload_eboot");
    const auto status = Resolve<
        decltype(&executor_lsx4_ps5_runtime_status)>(
        library, "executor_lsx4_ps5_runtime_status");
    const auto reset = Resolve<
        decltype(&executor_lsx4_ps5_runtime_reset)>(
        library, "executor_lsx4_ps5_runtime_reset");
    if (initialize == nullptr || set_callbacks == nullptr ||
        register_mapping == nullptr || unregister_mapping == nullptr ||
        load == nullptr || load_module == nullptr ||
        load_game == nullptr || probe_report == nullptr ||
        start_module == nullptr || stop_module == nullptr ||
        launch == nullptr ||
        create_thread == nullptr || destroy_thread == nullptr ||
        unload == nullptr || status == nullptr || reset == nullptr) {
        std::fprintf(stderr, "missing PS5 runtime ABI export\n");
        return 4;
    }

    const std::string eboot = "/data/local/tmp/lsx4-ps5-smoke-eboot.bin";
    const std::string module_path =
        "/data/local/tmp/lsx4-ps5-smoke-module.prx";
    const std::string late_module_path =
        "/data/local/tmp/lsx4-ps5-smoke-late-module.prx";
    if (!WriteSyntheticNextGenEboot(eboot.c_str()) ||
        !WriteSyntheticNextGenModule(module_path.c_str()) ||
        !WriteSyntheticNextGenModule(late_module_path.c_str())) {
        std::fprintf(stderr, "cannot create synthetic eboot\n");
        return 5;
    }

    Lsx4Ps5RuntimeConfig config{};
    config.title_id = "SMOKE00001";
    if (initialize(&config) != 0) {
        std::fprintf(stderr, "initialize: %s\n", status());
        return 6;
    }
    Lsx4Ps5RuntimeCallbacks invalid_callbacks{};
    invalid_callbacks.size =
        sizeof(Lsx4Ps5RuntimeCallbacks) - 1;
    if (set_callbacks(nullptr) != LSX4_PS5_EXECUTE_INVALID_ARGUMENT ||
        set_callbacks(&invalid_callbacks) !=
            LSX4_PS5_EXECUTE_INVALID_ARGUMENT) {
        std::fprintf(stderr, "callback invalid-argument gate failed\n");
        reset();
        return 7;
    }
    invalid_callbacks = {};
    ++invalid_callbacks.abi_version;
    if (set_callbacks(&invalid_callbacks) !=
        LSX4_PS5_EXECUTE_INVALID_ARGUMENT) {
        std::fprintf(stderr, "callback ABI gate failed\n");
        reset();
        return 7;
    }
    Lsx4Ps5GuestMapping invalid_mapping{};
    invalid_mapping.protection = LSX4_PS5_GUEST_READ;
    invalid_mapping.guest_address = 0x700000001ull;
    invalid_mapping.byte_count = LSX4_PS5_GUEST_PAGE_SIZE;
    if (register_mapping(nullptr) !=
            LSX4_PS5_EXECUTE_INVALID_ARGUMENT ||
        register_mapping(&invalid_mapping) !=
            LSX4_PS5_EXECUTE_INVALID_ARGUMENT) {
        std::fprintf(stderr, "mapping invalid-argument gate failed\n");
        reset();
        return 7;
    }
    Lsx4Ps5GuestMapping mapping{};
    mapping.protection = LSX4_PS5_GUEST_READ;
    mapping.guest_address = 0x700000000ull;
    mapping.byte_count = LSX4_PS5_GUEST_PAGE_SIZE;
    if (register_mapping(&mapping) != 0 ||
        unregister_mapping(
            mapping.guest_address, mapping.byte_count) != 0 ||
        unregister_mapping(
            mapping.guest_address, mapping.byte_count) !=
            LSX4_PS5_EXECUTE_INVALID_ARGUMENT) {
        std::fprintf(stderr, "mapping register/unregister gate failed\n");
        reset();
        return 7;
    }
    Lsx4Ps5GuestThreadContext invalid_thread{};
    if (create_thread(0, nullptr) !=
            LSX4_PS5_EXECUTE_INVALID_ARGUMENT ||
        create_thread(0xfeed, &invalid_thread) !=
            LSX4_PS5_EXECUTE_INVALID_ARGUMENT ||
        destroy_thread(0xfeed) !=
            LSX4_PS5_EXECUTE_INVALID_ARGUMENT) {
        std::fprintf(stderr, "thread-context invalid-argument gate failed\n");
        reset();
        return 7;
    }
    HleSmokeState hle_state{};
    Lsx4Ps5RuntimeCallbacks callbacks{};
    callbacks.context = &hle_state;
    callbacks.bind_import = &BindSmokeImport;
    callbacks.invoke_hle = &InvokeSmokeImport;
    if (set_callbacks(&callbacks) != 0) {
        std::fprintf(stderr, "set callbacks: %s\n", status());
        reset();
        return 7;
    }
    Lsx4Ps5LoadedExecutable executable{};
    if (load(eboot.c_str(), &executable) != 0) {
        std::fprintf(stderr, "load: %s\n", status());
        reset();
        return 8;
    }
    std::uint64_t absolute_value{};
    std::uint64_t relative64_value{};
    std::uint32_t size32_value{};
    std::int32_t pc32_value{};
    std::memcpy(&absolute_value,
                reinterpret_cast<const void*>(executable.base + 0x108),
                sizeof(absolute_value));
    std::memcpy(&relative64_value,
                reinterpret_cast<const void*>(executable.base + 0x110),
                sizeof(relative64_value));
    std::memcpy(&size32_value,
                reinterpret_cast<const void*>(executable.base + 0x118),
                sizeof(size32_value));
    std::memcpy(&pc32_value,
                reinterpret_cast<const void*>(executable.base + 0x11c),
                sizeof(pc32_value));
    if (absolute_value != executable.base + 0x504 ||
        relative64_value != executable.base + 0x580 ||
        size32_value != 0x1234 || pc32_value != 0x3e0) {
        std::fprintf(stderr, "unexpected extended relocation values\n");
        unload(executable.handle);
        reset();
        return 9;
    }
    if (executable.tls_memory_size != 0x20 ||
        executable.tls_alignment != 0x20 ||
        executable.tls_static_offset != 0x20) {
        std::fprintf(stderr, "unexpected PT_TLS layout\n");
        unload(executable.handle);
        reset();
        return 10;
    }
    Lsx4Ps5LoadedExecutable module{};
    if (load_module(
            executable.handle, module_path.c_str(), &module) != 0 ||
        (module.flags & LSX4_PS5_EXECUTABLE_MODULE) == 0 ||
        module.owner_handle != executable.handle ||
        module.initializer_count != 1 ||
        module.tls_module_id != 2 ||
        module.tls_static_offset != 0x40) {
        std::fprintf(stderr, "load module: %s\n", status());
        unload(executable.handle);
        reset();
        return 11;
    }
    std::uint64_t module_id_relocation{};
    std::memcpy(
        &module_id_relocation,
        reinterpret_cast<const void*>(module.base + 0x320),
        sizeof(module_id_relocation));
    if (module_id_relocation != 2) {
        std::fprintf(stderr, "unexpected module DTPMOD64 value\n");
        unload(executable.handle);
        reset();
        return 12;
    }
    std::uint64_t rebound_module_export{};
    std::memcpy(
        &rebound_module_export,
        reinterpret_cast<const void*>(executable.base + 0x130),
        sizeof(rebound_module_export));
    if (rebound_module_export != module.base + 0x40) {
        std::fprintf(stderr, "late intermodule import was not rebound\n");
        unload(executable.handle);
        reset();
        return 12;
    }
    Lsx4Ps5GuestResult module_result{};
    if (start_module(module.handle, &module_result) != 0 ||
        module_result.status != LSX4_PS5_EXECUTE_OK ||
        module_result.value != 0x5678) {
        std::fprintf(stderr, "start module: %s\n", status());
        unload(executable.handle);
        reset();
        return 13;
    }
    Lsx4Ps5GuestResult module_stop_result{};
    if (stop_module(module.handle, &module_stop_result) != 0 ||
        module_stop_result.status != LSX4_PS5_EXECUTE_OK ||
        module_stop_result.value != 0x9abc) {
        std::fprintf(stderr, "stop module: %s\n", status());
        unload(executable.handle);
        reset();
        return 14;
    }
    Lsx4Ps5GuestThreadContext thread_context{};
    if (create_thread(executable.handle, &thread_context) != 0 ||
        thread_context.fs_base == 0 ||
        thread_context.tls_static_offset != 0x40) {
        std::fprintf(stderr, "create thread context: %s\n", status());
        unload(executable.handle);
        reset();
        return 16;
    }
    hle_state.expected_fs_base = thread_context.fs_base;
    Lsx4Ps5LoadedExecutable late_module{};
    if (load_module(
            executable.handle, late_module_path.c_str(),
            &late_module) != 0 ||
        late_module.tls_module_id != 3 ||
        late_module.tls_static_offset != 0x60) {
        std::fprintf(stderr, "late TLS module load: %s\n", status());
        destroy_thread(thread_context.handle);
        unload(executable.handle);
        reset();
        return 17;
    }
    std::uint64_t late_tls_value{};
    std::uint64_t late_dtv_pointer{};
    std::uint64_t late_dtv_maximum{};
    std::memcpy(
        &late_tls_value,
        reinterpret_cast<const void*>(
            thread_context.fs_base - late_module.tls_static_offset),
        sizeof(late_tls_value));
    std::memcpy(
        &late_dtv_maximum,
        reinterpret_cast<const void*>(thread_context.fs_base + 0x108),
        sizeof(late_dtv_maximum));
    std::memcpy(
        &late_dtv_pointer,
        reinterpret_cast<const void*>(
            thread_context.fs_base + 0x110 +
            (late_module.tls_module_id - 1) * sizeof(std::uint64_t)),
        sizeof(late_dtv_pointer));
    if (late_tls_value != 0x5678 ||
        late_dtv_maximum != late_module.tls_module_id ||
        late_dtv_pointer !=
            thread_context.fs_base - late_module.tls_static_offset) {
        std::fprintf(stderr, "late TLS DTV installation failed\n");
        destroy_thread(thread_context.handle);
        unload(executable.handle);
        reset();
        return 17;
    }
    Lsx4Ps5GuestEntry entry{};
    entry.fs_base = thread_context.fs_base;
    Lsx4Ps5GuestResult result{};
    if (launch(executable.handle, &entry, &result) != 0 ||
        result.status != LSX4_PS5_EXECUTE_OK ||
        result.value != 0x2468 || hle_state.binds != 1 ||
        hle_state.invokes != 1 || !hle_state.argument_ok) {
        std::fprintf(
            stderr, "launch: status=%d runtime=%s\n",
            result.status, status());
        destroy_thread(thread_context.handle);
        unload(executable.handle);
        reset();
        return 17;
    }
    std::uint64_t initializer_witness{};
    std::memcpy(
        &initializer_witness,
        reinterpret_cast<const void*>(executable.base + 0x128),
        sizeof(initializer_witness));
    if (initializer_witness != 0x1234) {
        std::fprintf(stderr, "eboot DT_INIT did not run\n");
        destroy_thread(thread_context.handle);
        unload(executable.handle);
        reset();
        return 18;
    }
    if (destroy_thread(thread_context.handle) != 0) {
        std::fprintf(stderr, "destroy thread context: %s\n", status());
        unload(executable.handle);
        reset();
        return 19;
    }
    if (unload(module.handle) != 0) {
        std::fprintf(stderr, "unload module: %s\n", status());
        unload(executable.handle);
        reset();
        return 20;
    }
    std::printf(
        "PS5_SMOKE_OK base=0x%llx entry=0x%llx value=0x%llx flags=0x%x\n",
        static_cast<unsigned long long>(executable.base),
        static_cast<unsigned long long>(executable.entry),
        static_cast<unsigned long long>(result.value),
        executable.flags);
    const std::string game_directory =
        "/data/local/tmp/lsx4-ps5-game-smoke";
    const std::string game_module_directory =
        game_directory + "/sce_module";
    const std::string game_modules_directory =
        game_directory + "/sce_modules";
    const std::string media_directory =
        game_directory + "/Media";
    const std::string media_modules_directory =
        media_directory + "/Modules";
    const std::string media_plugins_directory =
        media_directory + "/Plugins";
    const std::string game_eboot = game_directory + "/eboot.bin";
    const std::string game_module =
        game_module_directory + "/smoke_module.prx";
    const std::string game_second_module =
        game_modules_directory + "/second_module.sprx";
    const std::string game_media_module =
        media_modules_directory + "/media_module.PRX";
    const std::string game_plugin =
        media_plugins_directory + "/optional_plugin.prx";
    const std::string game_fmod_plugin =
        media_plugins_directory + "/libfmod.prx";
    const std::string game_kernel_module =
        game_module_directory + "/libkernel.prx";
    const std::string game_broken_module =
        game_module_directory + "/broken.prx";
    (void)unlink(game_broken_module.c_str());
    (void)mkdir(game_directory.c_str(), 0700);
    (void)mkdir(game_module_directory.c_str(), 0700);
    (void)mkdir(game_modules_directory.c_str(), 0700);
    (void)mkdir(media_directory.c_str(), 0700);
    (void)mkdir(media_modules_directory.c_str(), 0700);
    (void)mkdir(media_plugins_directory.c_str(), 0700);
    if (!WriteSyntheticNextGenEboot(game_eboot.c_str()) ||
        !WriteSyntheticNextGenModule(game_module.c_str()) ||
        !WriteSyntheticNextGenModule(game_second_module.c_str()) ||
        !WriteSyntheticNextGenModule(game_media_module.c_str()) ||
        !WriteSyntheticNextGenModule(game_plugin.c_str()) ||
        !WriteSyntheticNextGenModule(game_fmod_plugin.c_str()) ||
        !WriteSyntheticNextGenModule(game_kernel_module.c_str())) {
        std::fprintf(stderr, "auto-game setup failed: %s\n", status());
        return 21;
    }
    HleSmokeState auto_hle_state{};
    callbacks.context = &auto_hle_state;
    if (set_callbacks(&callbacks) != 0) {
        std::fprintf(stderr, "auto-game callbacks failed: %s\n", status());
        reset();
        return 22;
    }
    Lsx4Ps5LoadedExecutable auto_executable{};
    Lsx4Ps5GameProbeReport game_report{};
    if (load_game(
             game_directory.c_str(), &auto_executable,
             &game_report) != 0 ||
        game_report.discovered_modules != 6 ||
        game_report.loaded_modules != 5 ||
        game_report.failed_modules != 0 ||
        game_report.deferred_function_imports != 1 ||
        game_report.unresolved_data_imports != 0 ||
        game_report.missing_dependencies != 1 ||
        (game_report.flags &
         LSX4_PS5_GAME_READY_TO_LAUNCH) == 0) {
        std::fprintf(stderr, "auto-game load/probe failed: %s\n", status());
        reset();
        return 23;
    }
    std::array<char, 1024> probe_text{};
    Lsx4Ps5GameProbeReport repeated_report{};
    if (probe_report(
            auto_executable.handle, &repeated_report,
            probe_text.data(), probe_text.size()) != 0 ||
        std::strstr(probe_text.data(), "ready=yes") == nullptr) {
        std::fprintf(stderr, "probe report export failed: %s\n", status());
        unload(auto_executable.handle);
        reset();
        return 24;
    }
    Lsx4Ps5GuestResult auto_result{};
    if (launch(auto_executable.handle, nullptr, &auto_result) != 0 ||
        auto_result.status != LSX4_PS5_EXECUTE_OK ||
        auto_result.value != 0x2468 ||
        auto_hle_state.binds != 1 ||
        auto_hle_state.invokes != 1 ||
        !auto_hle_state.argument_ok) {
        std::fprintf(
            stderr,
            "auto-game launch failed: base=0x%llx status=%d "
            "value=0x%llx runtime=%s\n",
            static_cast<unsigned long long>(auto_executable.base),
            auto_result.status,
            static_cast<unsigned long long>(auto_result.value), status());
        unload(auto_executable.handle);
        reset();
        return 25;
    }
    unload(auto_executable.handle);
    const std::array<std::uint8_t, 4> broken_module{
        0x7f, 'B', 'A', 'D'};
    const int broken_descriptor =
        open(game_broken_module.c_str(),
             O_CREAT | O_TRUNC | O_WRONLY, 0600);
    const auto broken_written =
        broken_descriptor >= 0
        ? write(
              broken_descriptor, broken_module.data(),
              broken_module.size())
        : -1;
    if (broken_descriptor >= 0) {
        close(broken_descriptor);
    }
    Lsx4Ps5LoadedExecutable gated_executable{};
    Lsx4Ps5GameProbeReport gated_report{};
    Lsx4Ps5GuestResult gated_result{};
    if (broken_written !=
            static_cast<ssize_t>(broken_module.size()) ||
        load_game(
            game_directory.c_str(), &gated_executable,
            &gated_report) != 0 ||
        gated_report.discovered_modules != 7 ||
        gated_report.loaded_modules != 5 ||
        gated_report.failed_modules != 1 ||
        (gated_report.flags &
         LSX4_PS5_GAME_READY_TO_LAUNCH) != 0 ||
        launch(gated_executable.handle, nullptr, &gated_result) !=
            LSX4_PS5_EXECUTE_BACKEND_NOT_READY) {
        std::fprintf(
            stderr, "failed-module compatibility gate failed: %s\n",
            status());
        if (gated_executable.handle != 0) {
            unload(gated_executable.handle);
        }
        reset();
        return 26;
    }
    unload(gated_executable.handle);
    unload(executable.handle);
    reset();
    if (initialize(&config) != 0) {
        std::fprintf(stderr, "second-session initialize: %s\n", status());
        return 27;
    }
    HleSmokeState second_session_hle{};
    callbacks.context = &second_session_hle;
    if (set_callbacks(&callbacks) != 0) {
        std::fprintf(stderr, "second-session callbacks: %s\n", status());
        reset();
        return 28;
    }
    Lsx4Ps5LoadedExecutable second_session_executable{};
    Lsx4Ps5LoadedExecutable second_session_module{};
    Lsx4Ps5GuestResult second_session_result{};
    if (load(
            eboot.c_str(), &second_session_executable) != 0 ||
        load_module(
            second_session_executable.handle, module_path.c_str(),
            &second_session_module) != 0 ||
        launch(
            second_session_executable.handle, nullptr,
            &second_session_result) != 0 ||
        second_session_result.status != LSX4_PS5_EXECUTE_OK ||
        second_session_result.value != 0x2468 ||
        second_session_hle.binds != 1 ||
        second_session_hle.invokes != 1) {
        std::fprintf(
            stderr,
            "second-session relaunch failed: status=%d runtime=%s\n",
            second_session_result.status, status());
        if (second_session_executable.handle != 0) {
            unload(second_session_executable.handle);
        }
        reset();
        return 29;
    }
    unload(second_session_executable.handle);
    reset();
    dlclose(library);
    unlink(eboot.c_str());
    unlink(module_path.c_str());
    unlink(late_module_path.c_str());
    unlink(game_eboot.c_str());
    unlink(game_module.c_str());
    unlink(game_second_module.c_str());
    unlink(game_media_module.c_str());
    unlink(game_plugin.c_str());
    unlink(game_fmod_plugin.c_str());
    unlink(game_kernel_module.c_str());
    unlink(game_broken_module.c_str());
    rmdir(game_module_directory.c_str());
    rmdir(game_modules_directory.c_str());
    rmdir(media_modules_directory.c_str());
    rmdir(media_plugins_directory.c_str());
    rmdir(media_directory.c_str());
    rmdir(game_directory.c_str());
    return 0;
}
