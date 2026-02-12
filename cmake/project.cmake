# SG2002 SDK integration for SSCMEx

set(SOPHGO_PLATFORM ON)
set(CMAKE_CXX_STANDARD 17)

if(NOT "${SG200X_SDK_PATH}" STREQUAL "")
    message(STATUS "SG200X_SDK_PATH: ${SG200X_SDK_PATH}")

    # Get the parent directory of SG200X_SDK_PATH
    get_filename_component(SG200X_ROOT "${SG200X_SDK_PATH}" DIRECTORY)

    # Set sysroot path (go up two levels from sg2002_recamera_emmc)
    if("${SYSROOT}" STREQUAL "")
        set(SYSROOT ${SG200X_ROOT}/buildroot-2021.05/output/cvitek_CV181X_musl_riscv64/host/riscv64-buildroot-linux-musl/sysroot)
    endif()

    message(STATUS "SYSROOT: ${SYSROOT}")

    # Include system libraries from sysroot (if it exists)
    if(EXISTS "${SYSROOT}/usr/include")
        include_directories(SYSTEM "${SYSROOT}/usr/include")
    endif()
    if(EXISTS "${SYSROOT}/usr/lib")
        link_directories("${SYSROOT}/usr/lib")
    endif()

    # TPU SDK includes and libraries
    if(EXISTS "${SG200X_SDK_PATH}/tpu_musl_riscv64/cvitek_tpu_sdk/include")
        include_directories(SYSTEM "${SG200X_SDK_PATH}/tpu_musl_riscv64/cvitek_tpu_sdk/include")
    endif()
    if(EXISTS "${SG200X_SDK_PATH}/tpu_musl_riscv64/cvitek_tpu_sdk/lib")
        link_directories("${SG200X_SDK_PATH}/tpu_musl_riscv64/cvitek_tpu_sdk/lib")
    endif()
    if(EXISTS "${SG200X_SDK_PATH}/rootfs/mnt/system/lib")
        link_directories("${SG200X_SDK_PATH}/rootfs/mnt/system/lib")
    endif()
else()
    message(WARNING "SG200X_SDK_PATH is not set - TPU SDK will not be available")
endif()
