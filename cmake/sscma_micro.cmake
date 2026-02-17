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

# Collect ALL SSCMA-Micro source files using glob patterns
# Conditional compilation (MA_USE_ENGINE_CVI) will handle excluding unused engine code
file(GLOB_RECURSE SSCMA_MICRO_SOURCES
    "${SSCMA_MICRO_DIR}/sscma/core/**/*.cpp"
)
# Note: extensions excluded due to external dependencies (e.g., Eigen for bytetrack)
# that require additional setup scripts from SSCMA-Micro

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
