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

    # CVI MPI headers for video subsystem
    set(CVI_MPI_INCLUDE "${SG200X_SDK_PATH}/osdrv/interdrv/v2/include/common")
    if(EXISTS "${CVI_MPI_INCLUDE}")
        include_directories(SYSTEM "${CVI_MPI_INCLUDE}")
        include_directories(SYSTEM "${CVI_MPI_INCLUDE}/uapi")
        include_directories(SYSTEM "${CVI_MPI_INCLUDE}/uapi/linux")
        message(STATUS "CVI MPI includes: ${CVI_MPI_INCLUDE}")
    endif()

    # CVI chip-specific headers (cv181x for SG2002)
    set(CVI_CHIP_INCLUDE "${SG200X_SDK_PATH}/osdrv/interdrv/v2/include/chip/cv181x")
    if(EXISTS "${CVI_CHIP_INCLUDE}")
        include_directories(SYSTEM "${CVI_CHIP_INCLUDE}")
        include_directories(SYSTEM "${CVI_CHIP_INCLUDE}/uapi")
        message(STATUS "CVI chip includes: ${CVI_CHIP_INCLUDE}")
    endif()

    # CVI MPI main headers (cvi_venc.h, cvi_vb.h, cvi_vpss.h, etc.)
    set(CVI_MPI_MAIN_INCLUDE "${SG200X_SDK_PATH}/cvi_mpi/include")
    if(EXISTS "${CVI_MPI_MAIN_INCLUDE}")
        include_directories(SYSTEM "${CVI_MPI_MAIN_INCLUDE}")
        message(STATUS "CVI MPI main includes: ${CVI_MPI_MAIN_INCLUDE}")
    endif()

    # CVI ISP headers (cvi_isp.h, cvi_ae.h, cvi_awb.h, etc.)
    set(CVI_ISP_INCLUDE "${SG200X_SDK_PATH}/cvi_mpi/include/isp/cv181x")
    if(EXISTS "${CVI_ISP_INCLUDE}")
        include_directories(SYSTEM "${CVI_ISP_INCLUDE}")
        message(STATUS "CVI ISP includes: ${CVI_ISP_INCLUDE}")
    endif()

    # TPU SDK - the actual path includes install/soc_sg2002_recamera_emmc/
    set(TPU_SDK_PATH "${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk")

    if(EXISTS "${TPU_SDK_PATH}/include")
        include_directories(SYSTEM "${TPU_SDK_PATH}/include")
        message(STATUS "TPU SDK includes: ${TPU_SDK_PATH}/include")
    endif()
    if(EXISTS "${TPU_SDK_PATH}/lib")
        link_directories("${TPU_SDK_PATH}/lib")
        message(STATUS "TPU SDK libs: ${TPU_SDK_PATH}/lib")
    endif()

    # Also check old path for backwards compatibility
    if(EXISTS "${SG200X_SDK_PATH}/tpu_musl_riscv64/cvitek_tpu_sdk/include")
        include_directories(SYSTEM "${SG200X_SDK_PATH}/tpu_musl_riscv64/cvitek_tpu_sdk/include")
    endif()
    if(EXISTS "${SG200X_SDK_PATH}/tpu_musl_riscv64/cvitek_tpu_sdk/lib")
        link_directories("${SG200X_SDK_PATH}/tpu_musl_riscv64/cvitek_tpu_sdk/lib")
    endif()
    if(EXISTS "${SG200X_SDK_PATH}/rootfs/mnt/system/lib")
        link_directories("${SG200X_SDK_PATH}/rootfs/mnt/system/lib")
    endif()

    # CVI MPI libraries (libvenc.so, libvpss.so, libvi.so, libsys.so)
    set(CVI_MPI_LIB_DIR "${SG200X_SDK_PATH}/cvi_mpi/lib")
    if(EXISTS "${CVI_MPI_LIB_DIR}")
        link_directories("${CVI_MPI_LIB_DIR}")
        message(STATUS "CVI MPI libs: ${CVI_MPI_LIB_DIR}")
    endif()
else()
    message(WARNING "SG200X_SDK_PATH is not set - TPU SDK will not be available")
endif()
