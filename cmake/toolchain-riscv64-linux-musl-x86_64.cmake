# CMake toolchain file for RISCV cross-compilation for SG2002

# Set system information
set(CMAKE_SYSTEM_NAME          Linux)
set(CMAKE_SYSTEM_PROCESSOR     riscv)
set(ARCH riscv)

# Get absolute path to SDK
if(DEFINED ENV{SG200X_SDK_PATH})
    set(SG200X_SDK_PATH $ENV{SG200X_SDK_PATH})

    # Convert relative path to absolute path
    get_filename_component(SG200X_SDK_PATH_ABS "${SG200X_SDK_PATH}" ABSOLUTE)

    # Construct compiler paths
    set(COMPILER_PATH "${SG200X_SDK_PATH_ABS}/../host-tools/gcc/riscv64-linux-musl-x86_64/bin")

    # Convert compiler paths to absolute paths
    get_filename_component(CMAKE_C_COMPILER "${COMPILER_PATH}/riscv64-unknown-linux-musl-gcc" ABSOLUTE)
    get_filename_component(CMAKE_CXX_COMPILER "${COMPILER_PATH}/riscv64-unknown-linux-musl-g++" ABSOLUTE)
    get_filename_component(CMAKE_OBJCOPY "${COMPILER_PATH}/riscv64-unknown-linux-musl-objcopy" ABSOLUTE)

    message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
    message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
else()
    message(WARNING "SG200X_SDK_PATH environment variable not set")
endif()

# Set sysroot path
if(NOT DEFINED RISCV_SYSROOT_PATH AND DEFINED SG200X_SDK_PATH)
    set(RISCV_SYSROOT_PATH "${SG200X_SDK_PATH}/buildroot-2021.05/output/cvitek_CV181X_musl_riscv64/host/riscv64-buildroot-linux-musl/sysroot")
endif()

if(DEFINED RISCV_SYSROOT_PATH)
    set(CMAKE_FIND_ROOT_PATH ${RISCV_SYSROOT_PATH})
    # search for programs in the build host directories
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    # for libraries and headers in the target directories
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endif()

# Cache the OBJCOPY setting
set(CMAKE_OBJCOPY ${CMAKE_OBJCOPY} CACHE FILEPATH "The toolchain objcopy command" FORCE)

# Set RISCV flags for SG2002 (with vector and floating point extensions)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv_zfh -mabi=lp64d" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv_zfh -mabi=lp64d" CACHE STRING "" FORCE)
