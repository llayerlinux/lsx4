cmake_minimum_required(VERSION 3.22.1)
project(lsx_runtime_bridge LANGUAGES CXX)

add_library(lsx_runtime_bridge SHARED runtime_bridge.cpp)
target_compile_features(lsx_runtime_bridge PRIVATE cxx_std_17)
target_link_libraries(lsx_runtime_bridge PRIVATE android log dl)
