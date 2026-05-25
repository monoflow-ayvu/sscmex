# SSCMA-Micro library configuration for SG2002 (reCamera)
#
# This file configures the SSCMA-Micro library for use with the
# CVitek TPU engine on the SG2002 chip.

set(SSCMA_MICRO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/SSCMA-Micro")

if(NOT EXISTS "${SSCMA_MICRO_DIR}/sscma")
    message(FATAL_ERROR "SSCMA-Micro not found at: ${SSCMA_MICRO_DIR}")
endif()

# Fail fast when the SG2002 SDK isn't available. SSCMA-Micro's CVI engine
# header unconditionally `#include <cviruntime.h>` whenever
# MA_USE_ENGINE_CVI=1 is set, and that header only ships with the CVITEK
# TPU SDK (under $SG200X_SDK_PATH/install/.../cvitek_tpu_sdk/include).
# Without this guard the build dies later with a confusing
# "cviruntime.h: No such file or directory" deep inside SSCMA-Micro.
#
# Consumers building from source are expected to run inside the sscmex
# devenv (which downloads reCameraOS_sdk_v0.2.0.tar.gz) or to point
# SG200X_SDK_PATH at an existing extraction. In package consumers that
# don't have the SDK on the host, the recommended path is to take the
# precompiled NIF from a tagged GitHub release (cc_precompiler will fetch
# it automatically when checksum.exs is present).
if(NOT DEFINED SG200X_SDK_PATH OR "${SG200X_SDK_PATH}" STREQUAL "")
    message(FATAL_ERROR
        "SG200X_SDK_PATH is not set, but SSCMA-Micro's CVI engine "
        "(MA_USE_ENGINE_CVI=1) requires <cviruntime.h> from the CVITEK "
        "TPU SDK.\n"
        "  Fix one of:\n"
        "    1. Use the sscmex devenv (provides the SDK automatically):\n"
        "         direnv allow   # or: devenv shell\n"
        "    2. Point SG200X_SDK_PATH at an extracted reCameraOS SDK,\n"
        "       e.g. download reCameraOS_sdk_v0.2.0.tar.gz from\n"
        "       https://github.com/Seeed-Studio/reCamera-OS/releases\n"
        "       and set:\n"
        "         export SG200X_SDK_PATH=/path/to/sg2002_recamera_emmc\n"
        "    3. Use the precompiled NIF from a GitHub release. In your\n"
        "       parent project, ensure a checksum.exs is present (Hex\n"
        "       packaging includes one automatically; for git deps, add\n"
        "       the checksum.exs from the matching release as a dep\n"
        "       overlay) so cc_precompiler can fetch the precompiled\n"
        "       artefact instead of building from source."
    )
endif()

message(STATUS "SSCMA-Micro found at: ${SSCMA_MICRO_DIR}")

# Include directories for SSCMA-Micro
include_directories("${SSCMA_MICRO_DIR}")
include_directories("${SSCMA_MICRO_DIR}/sscma")  # For extension files to find "core/ma_types.h"
include_directories("${SSCMA_MICRO_DIR}/3rdparty")
include_directories("${SSCMA_MICRO_DIR}/3rdparty/eigen")
include_directories("${SSCMA_MICRO_DIR}/3rdparty/json")

# Compiler definitions for CVitek (SG2002) TPU engine
# MA_USE_ENGINE_CVI - Enable CVitek TPU backend
# MA_USE_FILESYSTEM - Enable filesystem support for model loading
# MA_USE_ENGINE_TENSOR_NAME - Enable tensor name support (required by CVI engine)
add_definitions(-DMA_USE_ENGINE_CVI=1)
add_definitions(-DMA_USE_FILESYSTEM=1)
add_definitions(-DMA_USE_ENGINE_TENSOR_NAME=1)

# Include all SSCMA-Micro sources (core, porting, client, server, extensions).
# MA_USE_ENGINE_CVI and ma_config_board.h (MA_OSAL_PTHREAD) control which code paths are built.
file(GLOB_RECURSE SSCMA_MICRO_SOURCES
    "${SSCMA_MICRO_DIR}/sscma/**/*.cpp"
)
# Exclude FreeRTOS OSAL - we use pthread only (MA_OSAL_PTHREAD in ma_config_board.h)
list(FILTER SSCMA_MICRO_SOURCES EXCLUDE REGEX "ma_osal_freertos\\.cpp$")
# Exclude bytetrack extension - requires Eigen (fetched by SSCMA-Micro's fetch_eigen.sh, not in vendor)
list(FILTER SSCMA_MICRO_SOURCES EXCLUDE REGEX "bytetrack/")
# Exclude server/ and client/ - require cJSON (fetched by fetch_cjson.sh, not in vendor)
list(FILTER SSCMA_MICRO_SOURCES EXCLUDE REGEX "server/")
list(FILTER SSCMA_MICRO_SOURCES EXCLUDE REGEX "client/")

# Create static library for SSCMA-Micro components
add_library(sscma_micro STATIC ${SSCMA_MICRO_SOURCES})

# Set Position Independent Code for the static library (required for linking into shared library)
set_target_properties(sscma_micro PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -ffast-math: allow reordering/fusing float ops (FMA, reciprocal approx) in
#   NMS IOU, cv::convert, sigmoid — safe because detection scores don't need
#   IEEE precision.
# -ftree-vectorize: already implied by -O3 in GCC, but explicit here so it
#   applies even when CMAKE_BUILD_TYPE != Release. Together with the
#   -march=rv64gcv_zfh from the toolchain, GCC emits RISC-V V (vector)
#   instructions for tight byte/float loops (e.g. the S8 subtract-128 in
#   Detector::preprocess, NMS sort, pixel conversion).
target_compile_options(sscma_micro PRIVATE -ffast-math -ftree-vectorize)

# Set include directories for the library
target_include_directories(sscma_micro PUBLIC
    "${SSCMA_MICRO_DIR}"
    "${SSCMA_MICRO_DIR}/3rdparty"
    "${SSCMA_MICRO_DIR}/3rdparty/eigen"
    "${SSCMA_MICRO_DIR}/3rdparty/json"
)

# Link against TPU SDK libraries (provided by SG2002 SDK)
if(DEFINED SG200X_SDK_PATH AND NOT "${SG200X_SDK_PATH}" STREQUAL "")
    # The TPU SDK libraries are already set up in cmake/project.cmake
    # We just need to link against them
    target_link_libraries(sscma_micro PUBLIC
        cviruntime
    )
endif()

# Statically link libstdc++ and libgcc to avoid version mismatches with target device
# This bundles the C++ runtime with the NIF
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(sscma_micro PUBLIC -static-libstdc++ -static-libgcc)
endif()

message(STATUS "SSCMA-Micro configured with CVI engine support")
