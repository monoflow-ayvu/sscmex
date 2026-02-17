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

# Platform-specific linker flags
if(UNIX AND APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)

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

# Source files
set(SOURCE_FILES
    "${C_SRC}/sscmex_nif.cpp"
    "${C_SRC}/sscma_porting.cpp"
    "${C_SRC}/ma_device.cpp"
)

# Build NIF shared library
add_library(sscmex_nif SHARED ${SOURCE_FILES})

# Link TPU runtime library if SDK is available
if(DEFINED SG200X_SDK_PATH AND NOT "${SG200X_SDK_PATH}" STREQUAL "")
    # TPU SDK library path
    set(TPU_LIB_DIR "${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/lib")

    # Link against shared TPU libraries and SSCMA-Micro
    target_link_libraries(sscmex_nif PRIVATE
        sscma_micro           # SSCMA-Micro library
        cviruntime            # Main TPU runtime (shared)
        z                     # zlib
        dl                    # dynamic loading
        pthread               # threads
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
