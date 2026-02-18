# SSCMA-Micro library configuration for SG2002 (reCamera)
#
# This file configures the SSCMA-Micro library for use with the
# CVitek TPU engine on the SG2002 chip.

set(SSCMA_MICRO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/SSCMA-Micro")

if(NOT EXISTS "${SSCMA_MICRO_DIR}/sscma")
    message(FATAL_ERROR "SSCMA-Micro not found at: ${SSCMA_MICRO_DIR}. Run: git submodule update --init")
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
# Temporarily exclude YOLO26 pose implementation due upstream type mismatch compile failure.
list(FILTER SSCMA_MICRO_SOURCES EXCLUDE REGEX "ma_model_yolo26_pose\\.cpp$")

# Create static library for SSCMA-Micro components
add_library(sscma_micro STATIC ${SSCMA_MICRO_SOURCES})

# Set Position Independent Code for the static library (required for linking into shared library)
set_target_properties(sscma_micro PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
