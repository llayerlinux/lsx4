#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) {
        return 2;
    }
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 3;
    }
    using SelfTest = const char* (*)();
    auto selftest = reinterpret_cast<SelfTest>(
        dlsym(library, "executor_lsx4_runtime_jit_selftest"));
    if (selftest == nullptr) {
        std::fprintf(stderr, "dlsym: %s\n", dlerror());
        return 4;
    }
    const char* result = selftest();
    if (result == nullptr) {
        return 5;
    }
    std::fputs(result, stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    std::_Exit(0);
}
