cmake_minimum_required(VERSION 3.24)

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
project(cpp_torch_dev
        VERSION 0.1.0
        DESCRIPTION "C++/CUDA inference and training pipeline for SmolLM2"
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 5060 Ti is Blackwell sm_120. Allow override for other dev machines.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# Export compile_commands.json for clangd / editors.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release if user didn't pick a build type.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Put all binaries in build/bin, libs in build/lib.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# -----------------------------------------------------------------------------
# ccache (optional but huge win for CUDA rebuilds)
# -----------------------------------------------------------------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
  set(CMAKE_CUDA_COMPILER_LAUNCHER  ${CCACHE_PROGRAM})
  message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
endif()

# -----------------------------------------------------------------------------
# Warnings
# -----------------------------------------------------------------------------
add_library(ctd_warnings INTERFACE)
target_compile_options(ctd_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion>
  # nvcc silently ignores most host warnings unless we forward them.
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# -----------------------------------------------------------------------------
# CUDA libraries (cuBLAS etc.)
# -----------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------------
# Dependencies via FetchContent
# -----------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# GoogleTest for unit tests.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# nlohmann/json — used by the safetensors loader to parse the header.
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

# -----------------------------------------------------------------------------
# Subdirs
# -----------------------------------------------------------------------------
add_subdirectory(src)

enable_testing()
add_subdirectory(tests)

add_subdirectory(tools)
