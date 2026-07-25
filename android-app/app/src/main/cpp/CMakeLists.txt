cmake_minimum_required(VERSION 3.22.1)
project(lsx_runtime_bridge LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
    unrar_source
    URL https://www.rarlab.com/rar/unrarsrc-7.2.7.tar.gz
    URL_HASH SHA256=01D903A7DCF413CB2925696D7796E48E38D471F79BFE7EF3AD2AEBF6C12DBEFD
)
FetchContent_MakeAvailable(unrar_source)

set(UNRAR_SOURCE_DIR "${unrar_source_SOURCE_DIR}")
if(EXISTS "${UNRAR_SOURCE_DIR}/unrar/dll.cpp")
    set(UNRAR_SOURCE_DIR "${UNRAR_SOURCE_DIR}/unrar")
endif()

set(UNRAR_SOURCES
    archive.cpp arcread.cpp blake2s.cpp cmddata.cpp consio.cpp crc.cpp crypt.cpp
    dll.cpp encname.cpp errhnd.cpp extinfo.cpp extract.cpp file.cpp filefn.cpp
    filcreat.cpp filestr.cpp find.cpp getbits.cpp global.cpp hash.cpp headers.cpp
    largepage.cpp list.cpp match.cpp options.cpp pathfn.cpp qopen.cpp rawread.cpp
    rdwrfn.cpp resource.cpp rijndael.cpp rarvm.cpp scantree.cpp secpassword.cpp
    sha1.cpp sha256.cpp smallfn.cpp strfn.cpp strlist.cpp system.cpp threadpool.cpp
    timefn.cpp ui.cpp unicode.cpp unpack.cpp volume.cpp rs16.cpp
)
list(TRANSFORM UNRAR_SOURCES PREPEND "${UNRAR_SOURCE_DIR}/")

add_library(
    lsx_runtime_bridge
    SHARED
    runtime_bridge.cpp
    archive_bridge.cpp
    ${UNRAR_SOURCES}
)
target_compile_features(lsx_runtime_bridge PRIVATE cxx_std_17)
target_compile_definitions(
    lsx_runtime_bridge
    PRIVATE
    _UNIX
    RARDLL
    RAR_SMP
    _FILE_OFFSET_BITS=64
    _LARGEFILE_SOURCE
)
target_compile_options(
    lsx_runtime_bridge
    PRIVATE
    -Wno-logical-op-parentheses
    -Wno-switch
    -Wno-dangling-else
)
target_include_directories(lsx_runtime_bridge PRIVATE "${UNRAR_SOURCE_DIR}")
target_link_libraries(lsx_runtime_bridge PRIVATE android log dl)
