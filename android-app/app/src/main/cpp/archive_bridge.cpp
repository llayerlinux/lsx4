/*
 * UnRAR source code may be used in any software to handle RAR archives
 * without limitations free of charge, but cannot be used to develop a RAR
 * (WinRAR) compatible archiver and to re-create RAR compression algorithm,
 * which is proprietary.
 */

#include <jni.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "dll.hpp"

namespace {

class UtfChars {
public:
    UtfChars(JNIEnv* env, jstring value) : env_(env), value_(value) {
        chars_ = value == nullptr ? nullptr : env_->GetStringUTFChars(value, nullptr);
    }

    ~UtfChars() {
        if (chars_ != nullptr) {
            env_->ReleaseStringUTFChars(value_, chars_);
        }
    }

    const char* get() const {
        return chars_;
    }

private:
    JNIEnv* env_;
    jstring value_;
    const char* chars_ = nullptr;
};

bool IsSafeArchivePath(const char* raw_name) {
    if (raw_name == nullptr) {
        return false;
    }
    std::string name(raw_name);
    if (name.empty() || name.size() > 4096 || name.front() == '/' ||
        name.front() == '\\') {
        return false;
    }
    if (name.size() >= 2 &&
        ((name[0] >= 'A' && name[0] <= 'Z') ||
         (name[0] >= 'a' && name[0] <= 'z')) &&
        name[1] == ':') {
        return false;
    }

    size_t begin = 0;
    while (begin <= name.size()) {
        const size_t end = name.find_first_of("/\\", begin);
        const size_t length =
            (end == std::string::npos ? name.size() : end) - begin;
        if (length == 2 && name[begin] == '.' && name[begin + 1] == '.') {
            return false;
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return true;
}

int CALLBACK UnrarCallback(UINT message, LPARAM, LPARAM, LPARAM) {
    switch (message) {
    case UCM_PROCESSDATA:
        return 1;
    case UCM_LARGEDICT:
        return 1;
    default:
        return -1;
    }
}

}

extern "C" JNIEXPORT jint JNICALL
Java_app_lsx4_android_RuntimeBridge_extractRarArchive(
    JNIEnv* env, jclass, jstring archive_path, jstring destination_path,
    jlong maximum_bytes, jint maximum_files) {
    UtfChars archive_chars(env, archive_path);
    UtfChars destination_chars(env, destination_path);
    if (archive_chars.get() == nullptr || destination_chars.get() == nullptr ||
        maximum_bytes <= 0 || maximum_files <= 0) {
        return ERAR_BAD_DATA;
    }

    RAROpenArchiveDataEx open_data{};
    open_data.ArcName = const_cast<char*>(archive_chars.get());
    open_data.OpenMode = RAR_OM_EXTRACT;
    HANDLE archive = RAROpenArchiveEx(&open_data);
    if (archive == nullptr) {
        return open_data.OpenResult == 0 ? ERAR_BAD_ARCHIVE :
                                           static_cast<jint>(open_data.OpenResult);
    }
    RARSetCallback(archive, UnrarCallback, 0);

    std::uint64_t total_bytes = 0;
    int files = 0;
    int result = ERAR_SUCCESS;
    for (;;) {
        RARHeaderDataEx header{};
        result = RARReadHeaderEx(archive, &header);
        if (result == ERAR_END_ARCHIVE) {
            result = ERAR_SUCCESS;
            break;
        }
        if (result != ERAR_SUCCESS) {
            break;
        }
        if (!IsSafeArchivePath(header.FileName) || header.RedirType != 0) {
            result = ERAR_BAD_DATA;
            break;
        }
        if ((header.Flags & RHDF_DIRECTORY) == 0) {
            if (++files > maximum_files) {
                result = ERAR_BAD_DATA;
                break;
            }
            const std::uint64_t size =
                static_cast<std::uint64_t>(header.UnpSize) |
                (static_cast<std::uint64_t>(header.UnpSizeHigh) << 32U);
            const std::uint64_t limit = static_cast<std::uint64_t>(maximum_bytes);
            if (size > limit || total_bytes > limit - size) {
                result = ERAR_EWRITE;
                break;
            }
            total_bytes += size;
        }

        result = RARProcessFile(
            archive, RAR_EXTRACT, const_cast<char*>(destination_chars.get()), nullptr);
        if (result != ERAR_SUCCESS) {
            break;
        }
    }

    const int close_result = RARCloseArchive(archive);
    if (result == ERAR_SUCCESS && close_result != ERAR_SUCCESS) {
        result = close_result;
    }
    return result;
}
