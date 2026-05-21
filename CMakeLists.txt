cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(sscmex_nif C CXX)

# Get directories
if(NOT DEFINED PRIV_DIR)
    if(DEFINED MIX_APP_PATH AND NOT "${MIX_APP_PATH}" STREQUAL "")
        set(PRIV_DIR "${MIX_APP_PATH}/priv")
    else()
        set(PRIV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/priv")
    endif()
endif()

# Set C_SRC if not defined (for direct CMake invocation)
if(NOT DEFINED C_SRC OR "${C_SRC}" STREQUAL "")
    set(C_SRC "${CMAKE_CURRENT_SOURCE_DIR}/c_src")
endif()

# Get ERTS include directory
if(DEFINED ERTS_INCLUDE_DIR AND NOT "${ERTS_INCLUDE_DIR}" STREQUAL "")
    set(ERTS_INCLUDE_DIR "${ERTS_INCLUDE_DIR}")
else()
    find_program(ERL_EXECUTABLE erl)
    if(ERL_EXECUTABLE)
        execute_process(
            COMMAND ${ERL_EXECUTABLE} -noshell -eval "io:format('~s', [code:root_dir() ++ \"/erts-\" ++ erlang:system_info(version) ++ \"/include\"]), halt()."
            OUTPUT_VARIABLE ERTS_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    else()
        message(FATAL_ERROR "Erlang executable 'erl' not found")
    endif()
endif()

message(STATUS "PRIV_DIR: ${PRIV_DIR}")
message(STATUS "ERTS_INCLUDE_DIR: ${ERTS_INCLUDE_DIR}")

# Safety check for ERTS_INCLUDE_DIR
if(NOT EXISTS "${ERTS_INCLUDE_DIR}")
    message(FATAL_ERROR "ERTS_INCLUDE_DIR does not exist or is empty: '${ERTS_INCLUDE_DIR}'. Please ensure Erlang/OTP is installed and ERTS_INCLUDE_DIR is set correctly.")
endif()

# Platform-specific linker flags
if(UNIX AND APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)

# Use -O3 for all targets in this project (overrides Release default of -O2)
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Enable LTO for cross-file inlining (NIF templates, SSCMA-Micro vtable devirtualization)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Include SG200X SDK toolchain if CMAKE_TOOLCHAIN_FILE is set
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# Include SG200X SDK if SG200X_SDK_PATH is set
if(DEFINED SG200X_SDK_PATH AND NOT "${SG200X_SDK_PATH}" STREQUAL "")
    message(STATUS "SG200X_SDK_PATH: ${SG200X_SDK_PATH}")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/project.cmake)
endif()

# Include SSCMA-Micro configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/sscma_micro.cmake)

# ERTS includes
include_directories("${ERTS_INCLUDE_DIR}")

# Include c_src directory for ma_config_board.h and other porting headers
include_directories("${C_SRC}")

# Include sophgo video subsystem headers
include_directories("${C_SRC}/sophgo/video")
include_directories("${C_SRC}/sophgo/video/include")
include_directories("${C_SRC}/sophgo/common")

# Source files
set(SOURCE_FILES
    "${C_SRC}/sscmex_nif.cpp"
    "${C_SRC}/sscma_porting.cpp"
    "${C_SRC}/sscma_yolov7.cpp"
    "${C_SRC}/ma_device.cpp"
    "${C_SRC}/ma_camera_sg200x.cpp"
)

# Sophgo video subsystem sources (glob all .c files)
file(GLOB_RECURSE SOPHGO_VIDEO_SOURCES
    "${C_SRC}/sophgo/video/*.c"
    "${C_SRC}/sophgo/common/*.c"
)

# Build NIF shared library
add_library(sscmex_nif SHARED ${SOURCE_FILES} ${SOPHGO_VIDEO_SOURCES})

# Promote `venc` threads from `SCHED_OTHER` to `SCHED_RR` priority 80 to
# prevent starving when other functions (ex: WiFi chip) consume several ms
# to run, making the CSI line overflow.
# Nerves is ran as `root`, but if otherwise you'd need to run this then
# grant `CAP_SYS_NICE` and also enable `CONFIG_RT` in your kernel.
target_compile_definitions(sscmex_nif PRIVATE APP_IPCAM_RT_SCHED_ENABLE=1)

# Link TPU runtime library if SDK is available
if(DEFINED SG200X_SDK_PATH AND NOT "${SG200X_SDK_PATH}" STREQUAL "")
    # TPU SDK library path
    set(TPU_LIB_DIR "${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/lib")

    # Link against shared TPU libraries, CVI MPI (video pipeline), and SSCMA-Micro
    # CVI MPI libs use short names; see cmake/project.cmake for CVI_MPI_LIB_DIR
    target_link_libraries(sscmex_nif PRIVATE
        sscma_micro           # SSCMA-Micro library
        cviruntime            # Main TPU runtime (shared)
        venc                  # Video encoder (CVI_VENC_*)
        vpss                  # VPSS (CVI_VPSS_*)
        vi                    # Video input (CVI_VI_*)
        sys                   # System / VB / bind (CVI_SYS_*, CVI_VB_*)
        isp                   # ISP (CVI_ISP_*)
        ae                    # Auto exposure (CVI_AE_Register, etc.)
        awb                   # Auto white balance
        af                    # Auto focus
        gdc                   # Geometric distortion correction
        isp_algo             # ISP algorithms
        cvi_bin_isp          # CVI bin ISP
        cvi_bin              # CVI bin
        sns_sc530ai_2l       # SC530AI_2L sensor (stSnsSC530AI_2L_Obj)
        sns_ov5647           # OV5647 sensor (if used)
        vo                   # Video output (get_vo_bin_guardmagic_code for libcvi_bin)
        z                     # zlib
        dl                    # dynamic loading
        pthread               # threads
        json-c                # JSON library for video subsystem
    )

    # Statically link libstdc++ and libgcc to avoid version mismatches with target device
    # This bundles the C++ runtime with the NIF
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_options(sscmex_nif PRIVATE -static-libstdc++ -static-libgcc)
    endif()

    message(STATUS "Linking with TPU runtime library (shared): cviruntime")

    # Copy TPU libraries to priv/lib for bundling with the NIF
    set(PRIV_LIB_DIR "${PRIV_DIR}/lib")
    add_custom_command(TARGET sscmex_nif POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PRIV_LIB_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TPU_LIB_DIR}/libcviruntime.so" "${PRIV_LIB_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TPU_LIB_DIR}/libcvikernel.so" "${PRIV_LIB_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TPU_LIB_DIR}/libcvimath.so" "${PRIV_LIB_DIR}/"
        COMMENT "Copying TPU libraries to priv/lib"
    )
    message(STATUS "TPU libraries will be bundled in: ${PRIV_LIB_DIR}")

    # OpenCV 3.2.0 for image processing NIF functions
    set(OPENCV_SDK_PATH "${TPU_SDK_PATH}/opencv")
    if(EXISTS "${OPENCV_SDK_PATH}/include")
        include_directories(SYSTEM "${OPENCV_SDK_PATH}/include")
        include_directories(SYSTEM "${OPENCV_SDK_PATH}/include/opencv")
        message(STATUS "OpenCV includes: ${OPENCV_SDK_PATH}/include")
    endif()
    if(EXISTS "${OPENCV_SDK_PATH}/lib")
        link_directories("${OPENCV_SDK_PATH}/lib")
        message(STATUS "OpenCV libs: ${OPENCV_SDK_PATH}/lib")
    endif()

    target_link_libraries(sscmex_nif PRIVATE
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
    )

    # Copy OpenCV shared libraries to priv/lib for bundling with the NIF
    add_custom_command(TARGET sscmex_nif POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_SDK_PATH}/lib/libopencv_core.so.3.2.0" "${PRIV_LIB_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_SDK_PATH}/lib/libopencv_imgproc.so.3.2.0" "${PRIV_LIB_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OPENCV_SDK_PATH}/lib/libopencv_imgcodecs.so.3.2.0" "${PRIV_LIB_DIR}/"
        COMMENT "Copying OpenCV libraries to priv/lib"
    )
    message(STATUS "OpenCV libraries will be bundled in: ${PRIV_LIB_DIR}")
endif()

# Install to priv directory
install(TARGETS sscmex_nif RUNTIME DESTINATION "${PRIV_DIR}")

# Set properties
set_target_properties(sscmex_nif PROPERTIES PREFIX "")
if(NOT WIN32)
    set_target_properties(sscmex_nif PROPERTIES SUFFIX ".so")
endif()

# RPATH settings - include /usr/lib where TPU libraries are on target
if(UNIX AND NOT APPLE)
    set_target_properties(sscmex_nif PROPERTIES
        INSTALL_RPATH "$ORIGIN/lib:/usr/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX AND APPLE)
    set_target_properties(sscmex_nif PROPERTIES
        INSTALL_RPATH "@loader_path/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
