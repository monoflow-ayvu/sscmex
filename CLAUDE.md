# SSCMEx - AI Assistant Context

## Project Overview

SSCMEx is a Native Implemented Function (NIF) based Elixir library for the SG2002 chip (reCamera) that provides bindings to SSCMA (Smart Sensor and Control Module for AI). This project converts the existing Port-based approach from `sscma-elixir` to a NIF-based implementation for improved performance when handling binary data from camera and TPU operations.

### Goals

1. Convert SSCMA Port implementation to NIF-based implementation
2. Use the same RISCV cross-compilation toolchain as sscma-elixir
3. Integrate with Nx for efficient binary data handling (images, TPU tensors)
4. Target SG2002 chip in RISCV mode with TPU support

### Why NIF vs Port?

| Aspect | Port | NIF |
|--------|------|-----|
| Performance | Inter-process communication overhead | In-process, direct function calls |
| Data Transfer | Serialization/deserialization required | Direct memory access |
| Binary Data | High overhead for large binaries | Zero-copy possible with Nx |
| Crash Isolation | Process crash doesn't crash BEAM | NIF crash crashes entire BEAM |
| Complexity | Simpler C code | More complex resource management |

For camera/TPU workloads with large binary data (images, tensors), NIF provides significant performance benefits.

---

## Key Requirements

1. **Target Hardware**: SG2002 chip (RISCV) with TPU (reCamera)
2. **Toolchain**: Same as sscma-elixir (CMake-only with RISCV cross-compilation)
3. **NIF Approach**: Replace Port with NIF for better binary data performance
4. **Nx Integration**: Use Nx for tensor operations (camera frames, TPU input/output)
5. **Build System**: Use elixir_make with CMake wrapper to integrate with SG2002 SDK

---

## Reference Projects

The project draws inspiration from and analyzes four reference projects:

| Project | Description | GitHub URL |
|---------|-------------|------------|
| sscma-elixir | Port-based, provides the SG2002 toolchain and SDK setup | https://github.com/monoflow-ayvu/sscma-elixir |
| rayex | NIF using Bundlex/Unifex with flake.nix | https://github.com/sonscallon/rayex |
| emlx | NIF using elixir_make with Nx backend for ML operations | https://github.com/mlx-platform/emlx |
| adbc | NIF using elixir_make with CMake for complex C++ projects | https://github.com/elixir-explorer/adbc |

---

## Reference Project Analysis

### 1. SSCMA-elixir (Port-based, Toolchain Reference)

**GitHub:** https://github.com/monoflow-ayvu/sscma-elixir

This project provides the reference toolchain and SDK setup for SG2002 cross-compilation.

#### devenv.nix - Development Environment Setup

```nix
{
  env = {
    SG200X_FHS = "1";
    CMAKE_TOOLCHAIN_FILE = "${config.env.DEVENV_ROOT}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake";
    SG200X_SDK_PATH = "${config.env.DEVENV_ROOT}/.devenv/state/sg200x-sdk/sg2002_recamera_emmc";
  };
}
```

**Key Points:**
- Sets `CMAKE_TOOLCHAIN_FILE` for cross-compilation
- `SG200X_SDK_PATH` points to SG2002 SDK installation
- `SG200X_FHS` flag enables FHS (Filesystem Hierarchy Standard) compatibility

#### toolchain-riscv64-linux-musl-x86_64.cmake - Cross-Compilation Toolchain

```cmake
set( CMAKE_SYSTEM_NAME          Linux )
set( CMAKE_SYSTEM_PROCESSOR     riscv )
set( ARCH riscv )

set(COMPILER_PATH "${SG200X_SDK_PATH_ABS}/../host-tools/gcc/riscv64-linux-musl-x86_64/bin")

set(CMAKE_C_COMPILER "${COMPILER_PATH}/riscv64-unknown-linux-musl-gcc")
set(CMAKE_CXX_COMPILER "${COMPILER_PATH}/riscv64-unknown-linux-musl-g++")

# RISCV flags for SG2002 (with vector and floating point extensions)
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv_zfh -mabi=lp64d" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv_zfh -mabi=lp64d" )
```

**Key Points:**
- Uses `riscv64-unknown-linux-musl-gcc` cross-compiler from SDK host-tools
- Target architecture flags: `rv64gcv_zfh` (RV64 with Vector, Float, Half-float)
- ABI: `lp64d` (Long/Pointer 64-bit with double float)
- Sets sysroot path for libraries

#### project.cmake - SDK Integration

```cmake
set(SOPHGO_PLATFORM ON)
set(CMAKE_CXX_STANDARD 17)

set(SYSROOT ${SG200X_SDK_PATH}/buildroot-2021.05/output/cvitek_CV181X_musl_riscv64/host/riscv64-buildroot-linux-musl/sysroot)

include_directories(SYSTEM "${SYSROOT}/usr/include")
link_directories("${SYSROOT}/usr/lib")

# TPU SDK includes
include_directories(SYSTEM "${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/include")
link_directories("${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk/lib")
link_directories("${SG200X_SDK_PATH}/install/soc_sg2002_recamera_emmc/rootfs/mnt/system/lib")
```

**Key Points:**
- C++17 standard
- Buildroot sysroot for system libraries
- TPU SDK includes and libraries for SG2002
- System libraries from rootfs

---

### 2. Rayex (Bundlex/Unifex NIF)

**GitHub:** https://github.com/sonscallon/rayex

This project demonstrates a DSL-based NIF approach using Bundlex and Unifex.

#### bundlex.exs - Native Library Configuration

```elixir
defmodule Rayex.BundlexProject do
  use Bundlex.Project

  def project() do
    [
      natives: natives(Bundlex.get_target())
    ]
  end

  def natives(_platform) do
    [
      rayex: [
        sources: ["rayex.c"],
        interface: [:nif, :cnode],
        os_deps: [
          raylib: {:pkg_config, "raylib"}
        ],
        preprocessor: Unifex
      ]
    ]
  end
end
```

#### rayex.spec.exs - Unifex Interface Specification

```elixir
module Rayex.Unifex.Raylib

interface [NIF, CNode]

# Simple function
spec init_window(width :: int, height :: int, title :: string) :: :ok :: label

# Returning values
spec get_screen_width() :: result :: int

# Complex struct
type vector2 :: %Rayex.Structs.Vector2{
       x: float,
       y: float
     }

# Using payload for C structs
spec load_sound(file_name :: string) :: sound :: payload
dirty(:io, load_sound: 1)
```

#### Advantages
- **DSL-based**: No manual NIF boilerplate
- **Type-safe**: Unifex generates type-safe bindings
- **Multi-interface**: Supports NIF and CNode from same spec

#### Disadvantages for SSCMEx
- **Nix dependency**: Not using devenv.nix
- **Limited C++**: Unifex designed primarily for C
- **Cross-compilation**: Not designed for complex cross-compilation scenarios

---

### 3. EMLX (Nx-based NIF with elixir_make)

**GitHub:** https://github.com/mlx-platform/emlx

This project demonstrates Nx integration with NIF for ML operations.

#### mix.exs - MLX Configuration

```elixir
def project do
  libmlx_config = libmlx_config()

  [
    app: :emlx,
    version: @version,
    description: "MLX bindings and backend for Nx",
    elixir: "~> 1.17",
    make_env: %{
      "MLX_DIR" => libmlx_config.dir,
      "MLX_VERSION" => libmlx_config.version,
      "MLX_INCLUDE_DIR" => Path.join(libmlx_config.dir, "include"),
      "MLX_LIB_DIR" => Path.join(libmlx_config.dir, "lib"),
      "EMLX_CACHE_DIR" => libmlx_config.cache_dir,
      "EMLX_VERSION" => @version,
    },
    compilers: compilers(libmlx_config),
    aliases: aliases()
  ]
end
```

#### Nx Backend Approach

```elixir
# Nx backend implementation
defmodule Emlx.Backend do
  @behaviour Nx.Backend

  # Convert Nx tensor to binary for C++
  defn from_binary(tensor, opts) do
    # Nx.Tensor -> binary
  end

  # Convert C++ binary to Nx tensor
  defn to_binary(tensor) do
    # Nx.Tensor -> binary
  end
end
```

#### Advantages for SSCMEx
- **Nx integration**: Perfect for camera/TPU tensor data
- **elixir_make**: Simple, well-tested build system
- **Binary handling**: Demonstrates efficient binary transfer patterns

---

### 4. ADBC (CMake-based NIF with elixir_make)

**GitHub:** https://github.com/elixir-explorer/adbc

This project demonstrates CMake integration for complex C++ NIFs.

#### mix.exs - CMake Integration

```elixir
defmodule Adbc.MixProject do
  use Mix.Project

  @version "0.8.0-dev"
  @github_url "https://github.com/elixir-explorer/adbc"

  def project do
    [
      app: :adbc,
      version: @version,
      elixir: "~> 1.14",
      compilers: [:elixir_make] ++ Mix.compilers()
    ] ++ precompiled()
  end

  defp precompiled do
    if System.get_env("ADBC_BUILD") in ["1", "true"] do
      []
    else
      [
        make_precompiler: {:nif, CCPrecompiler},
        make_precompiler_url:
          "#{@github_url}/releases/download/v#{@version}/@{artefact_filename}",
        make_precompiler_filename: "adbc_nif",
        cc_precompiler: [
          cleanup: "clean",
          compilers: %{
            {:unix, :linux} => %{
              "x86_64-linux-gnu" => {
                "x86_64-linux-gnu-gcc",
                "x86_64-linux-gnu-g++"
              },
              "aarch64-linux-gnu" => {
                "aarch64-linux-gnu-gcc",
                "aarch64-linux-gnu-g++"
              }
            }
          }
        ]
      ]
    end
  end
end
```

#### Makefile - CMake Wrapper

```makefile
ifndef MIX_APP_PATH
	MIX_APP_PATH=$(shell pwd)
endif

PRIV_DIR = $(MIX_APP_PATH)/priv
NIF_SO = $(PRIV_DIR)/adbc_nif.so
ADBC_SRC = $(THIRD_PARTY_DIR)/apache-arrow-adbc

ifdef CMAKE_TOOLCHAIN_FILE
	CMAKE_CONFIGURE_FLAGS=-D CMAKE_TOOLCHAIN_FILE="$(CMAKE_TOOLCHAIN_FILE)"
endif

# Build NIF
$(NIF_SO): priv_dir adbc
	@ mkdir -p "$(CMAKE_ADBC_NIF_BUILD_DIR)" && \
	cd "$(CMAKE_ADBC_NIF_BUILD_DIR)" && \
	cmake --no-warn-unused-cli \
		-D CMAKE_BUILD_TYPE="$(CMAKE_BUILD_TYPE)" \
		-D C_SRC="$(C_SRC)" \
		-D ADBC_SRC="$(ADBC_SRC)" \
		-D MIX_APP_PATH="$(MIX_APP_PATH)" \
		-D PRIV_DIR="$(PRIV_DIR)" \
		-D ERTS_INCLUDE_DIR="$(ERTS_INCLUDE_DIR)" \
		$(CMAKE_CONFIGURE_FLAGS) "$(shell pwd)" && \
	make && \
	cp "$(CMAKE_ADBC_NIF_BUILD_DIR)/adbc_nif.so" "$(NIF_SO)"
```

#### CMakeLists.txt - NIF Configuration

```cmake
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(adbc_nif C CXX)

if(NOT DEFINED PRIV_DIR)
    if(DEFINED MIX_APP_PATH AND NOT "${MIX_APP_PATH}" STREQUAL "")
        set(PRIV_DIR "${MIX_APP_PATH}/priv")
    else()
        set(PRIV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/priv")
    endif()
endif()

if(DEFINED ERTS_INCLUDE_DIR AND NOT "${ERTS_INCLUDE_DIR}" STREQUAL "")
    set(ERTS_INCLUDE_DIR "${ERTS_INCLUDE_DIR}")
else()
    set(ERTS_INCLUDE_DIR_ONE_LINER "erl -noshell -eval \"io:format('~ts/erts-~ts/include/', [code:root_dir(), erlang:system_info(version)]), halt().\"")
    execute_process(COMMAND bash -c "${ERTS_INCLUDE_DIR_ONE_LINER}" OUTPUT_VARIABLE ERTS_INCLUDE_DIR)
endif()

if(UNIX AND APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
endif()

set(CMAKE_CXX_STANDARD 17)
```

#### adbc_nif.cpp - Resource Management Pattern

```cpp
#include <erl_nif.h>
#include "nif_utils.hpp"

// Template-based resource types
template<> ErlNifResourceType * NifRes<struct AdbcDatabase>::type = nullptr;
template<> ErlNifResourceType * NifRes<struct AdbcConnection>::type = nullptr;
template<> ErlNifResourceType * NifRes<struct AdbcStatement>::type = nullptr;

static ERL_NIF_TERM adbc_database_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    using res_type = NifRes<struct AdbcDatabase>;
    ERL_NIF_TERM error{};
    auto database = res_type::allocate_resource(env, error);
    if (database == nullptr) {
        return error;
    }
    struct AdbcError adbc_error{};
    AdbcStatusCode code = AdbcDatabaseNew(&database->val, &adbc_error);
    if (code != ADBC_STATUS_OK) {
        return nif_error_from_adbc_error(env, &adbc_error);
    }
    ERL_NIF_TERM ret = database->make_resource(env);
    return erlang::nif::ok(env, ret);
}

// NIF function table with dirty job flags
static ErlNifFunc nif_functions[] = {
    {"adbc_database_new", 0, adbc_database_new, 0},
    {"adbc_database_init", 1, adbc_database_init, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"adbc_connection_init", 2, adbc_connection_init, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"adbc_statement_execute_query", 1, adbc_statement_execute_query, ERL_NIF_DIRTY_JOB_IO_BOUND},
};

ERL_NIF_INIT(Elixir.Adbc.Nif, nif_functions, on_load, on_reload, on_upgrade, NULL);
```

#### Advantages for SSCMEx
- **CMake-based**: Matches SSCMA SDK's build system
- **Resource management**: Clean pattern for TPU handles
- **Cross-compilation**: Supports toolchain files
- **C++**: Full C++17 support

---

## Build System Comparison

| Feature | Bundlex/Unifex | elixir_make | elixir_cmake |
|---------|----------------|-------------|--------------|
| DSL Support | Yes | No | No |
| C++ Support | Limited | Full | Full |
| Cross-compilation | Limited | Good | Excellent |
| CMake Integration | No | Manual | Native |
| Nx Integration | Manual | Manual | Manual |
| Resource Management | Auto-generated | Manual | Manual |
| Learning Curve | Moderate | Low | Moderate |

---

## Recommended Architecture for SSCMEx

### Build System: elixir_make with CMake Wrapper

Based on the analysis, **elixir_make with a CMake wrapper** is the recommended build system because:

1. **Matches SSCMA SDK**: The SG2002 toolchain uses CMake
2. **Cross-compilation**: First-class support for toolchain files
3. **C++ Support**: Full C++17 for TPU SDK integration
4. **Precedent**: ADBC demonstrates this pattern successfully

### Project Structure

```
sscmex/
├── devenv.nix              # Dev environment setup (from sscma-elixir)
├── mix.exs                 # Mix project with elixir_make
├── Makefile                # CMake wrapper
├── CMakeLists.txt          # NIF CMake configuration
├── cmake/
│   ├── toolchain-riscv64-linux-musl-x86_64.cmake  # RISCV toolchain
│   └── project.cmake       # SG2002 SDK integration
├── c_src/
│   ├── sscmex_nif.cpp      # Main NIF implementation
│   ├── nif_utils.hpp       # Resource management utilities
│   ├── tpu_wrapper.cpp     # TPU SDK wrapper
│   └── camera_wrapper.cpp  # Camera wrapper
├── lib/
│   ├── sscmex.ex           # Main Elixir module
│   ├── sscmex_nif.ex       # NIF loader
│   └── backend/
│       └── sscmex.ex       # Nx backend implementation
├── priv/
│   └── lib/                # TPU SDK libraries
└── 3rd_party/
    └── sscma/              # SSCMA source code
```

### Nx Integration Strategy

```elixir
defmodule Sscmex.Backend do
  @behaviour Nx.Backend

  # Camera frame to tensor
  def from_binary(frame, opts) do
    # Convert raw camera frame to Nx tensor
    # frame format depends on camera (YUV, RGB, etc.)
  end

  # TPU inference
  defn run_inference(tensor, model) do
    # Send tensor to TPU
    # Receive result as binary
    # Convert to Nx tensor
  end
end
```

### NIF Resource Types

```cpp
// Resource types for TPU and Camera handles
template<> ErlNifResourceType * NifRes<struct SscmaTpuContext>::type = nullptr;
template<> ErlNifResourceType * NifRes<struct SscmaCameraContext>::type = nullptr;
template<> ErlNifResourceType * NifRes<struct SscmaModel>::type = nullptr;
```

---

## Implementation Examples

### devenv.nix Setup

```nix
{
  env = {
    SG200X_FHS = "1";
    CMAKE_TOOLCHAIN_FILE = "${config.env.DEVENV_ROOT}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake";
    SG200X_SDK_PATH = "${config.env.DEVENV_ROOT}/.devenv/state/sg200x-sdk/sg2002_recamera_emmc";
  };
}
```

### mix.exs Configuration

```elixir
defmodule Sscmex.MixProject do
  use Mix.Project

  @version "0.1.0"
  @github_url "https://github.com/your-org/sscmex"

  def project do
    [
      app: :sscmex,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      compilers: [:elixir_make] ++ Mix.compilers(),
      deps: deps(),
      package: package(),
      description: "SSCMA NIF bindings for SG2002 chip"
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.9"},
      {:elixir_make, "~> 0.8", runtime: false}
    ]
  end
end
```

### Makefile (CMake Wrapper)

```makefile
ifndef MIX_APP_PATH
	MIX_APP_PATH=$(shell pwd)
endif

PRIV_DIR = $(MIX_APP_PATH)/priv
SSCMEX_SO = $(PRIV_DIR)/sscmex_nif.so

ifdef CMAKE_TOOLCHAIN_FILE
	CMAKE_CONFIGURE_FLAGS=-D CMAKE_TOOLCHAIN_FILE="$(CMAKE_TOOLCHAIN_FILE)"
endif

build: $(SSCMEX_SO)

$(SSCMEX_SO): $(PRIV_DIR)
	@ mkdir -p "$(MIX_APP_PATH)/cmake_build" && \
	cd "$(MIX_APP_PATH)/cmake_build" && \
	cmake --no-warn-unused-cli \
		-D CMAKE_BUILD_TYPE=Release \
		-D C_SRC="$(shell pwd)/c_src" \
		-D MIX_APP_PATH="$(MIX_APP_PATH)" \
		-D PRIV_DIR="$(PRIV_DIR)" \
		-D SG200X_SDK_PATH="$(SG200X_SDK_PATH)" \
		$(CMAKE_CONFIGURE_FLAGS) "$(shell pwd)" && \
	cmake --build . --config Release && \
	cp "sscmex_nif.so" "$(SSCMEX_SO)"

clean:
	@rm -rf "$(MIX_APP_PATH)/cmake_build"
	@rm -rf "$(SSCMEX_SO)"
```

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(sscmex_nif C CXX)

# Get directories
if(NOT DEFINED PRIV_DIR)
    if(DEFINED MIX_APP_PATH)
        set(PRIV_DIR "${MIX_APP_PATH}/priv")
    else()
        set(PRIV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/priv")
    endif()
endif()

# Get ERTS include directory
if(DEFINED ERTS_INCLUDE_DIR)
    set(ERTS_INCLUDE_DIR "${ERTS_INCLUDE_DIR}")
else()
    execute_process(COMMAND bash -c "erl -noshell -eval 'io:format(\"~ts\", [code:root_dir() ++ \"/erts-\" ++ erlang:system_info(version) ++ \"/include\"]), halt().'" OUTPUT_VARIABLE ERTS_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Include SG200X toolchain if defined
if(DEFINED SG200X_SDK_PATH AND NOT "${SG200X_SDK_PATH}" STREQUAL "")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/project.cmake)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)

# ERTS includes
include_directories("${ERTS_INCLUDE_DIR}")

# Source files
set(SOURCE_FILES
    "${C_SRC}/nif_utils.cpp"
    "${C_SRC}/sscmex_nif.cpp"
)

# Build NIF
add_library(sscmex_nif SHARED ${SOURCE_FILES})

# Link TPU SDK libraries
if(DEFINED SG200X_SDK_PATH)
    target_link_libraries(sscmex_nif PUBLIC cvitek_tpu)
endif()

# Install to priv
install(TARGETS sscmex_nif RUNTIME DESTINATION "${PRIV_DIR}")

# Set properties
set_target_properties(sscmex_nif PROPERTIES PREFIX "")
set_target_properties(sscmex_nif PROPERTIES SUFFIX ".so")

# Platform-specific settings
if(UNIX AND APPLE)
    set_target_properties(sscmex_nif PROPERTIES
        INSTALL_RPATH "@loader_path/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX AND NOT APPLE)
    set_target_properties(sscmex_nif PROPERTIES
        INSTALL_RPATH "$ORIGIN/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
```

---

## Development Setup

Use devenv to set up the development environment with SG2002 SDK:

```bash
devenv shell
```

This will set up:
- SG2002 SDK path
- RISCV cross-compilation toolchain
- CMAKE_TOOLCHAIN_FILE

---

## Key Files to Create

Based on the analysis:
- `devenv.nix` - Dev environment setup
- `mix.exs` - Mix project with elixir_make
- `Makefile` - CMake wrapper
- `CMakeLists.txt` - NIF CMake configuration
- `cmake/toolchain-riscv64-linux-musl-x86_64.cmake` - RISCV toolchain (from sscma-elixir)
- `cmake/project.cmake` - SG2002 SDK integration (from sscma-elixir)
- `c_src/sscmex_nif.cpp` - Main NIF implementation
- `c_src/nif_utils.hpp` - Resource management utilities
- `c_src/tpu_wrapper.cpp` - TPU SDK wrapper
- `c_src/camera_wrapper.cpp` - Camera wrapper
- `lib/sscmex.ex` - Main Elixir module
- `lib/sscmex/backend.ex` - Nx backend implementation

---

## Implementation Checklist

1. [ ] Set up devenv.nix for SG2002 SDK
2. [ ] Create mix.exs with elixir_make
3. [ ] Create Makefile wrapper for CMake
4. [ ] Copy toolchain files from sscma-elixir
5. [ ] Set up CMakeLists.txt for NIF
6. [ ] Implement resource management (nif_utils.hpp)
7. [ ] Implement TPU wrapper (tpu_wrapper.cpp)
8. [ ] Implement camera wrapper (camera_wrapper.cpp)
9. [ ] Implement main NIF (sscmex_nif.cpp)
10. [ ] Create Elixir API (sscmex.ex)
11. [ ] Implement Nx backend (sscmex/backend.ex)
12. [ ] Test cross-compilation for RISCV
13. [ ] Test on actual SG2002 hardware

---

## References

- [devenv.sh Reference](https://devenv.sh/reference/options/)
- [elixir_make Documentation](https://hexdocs.pm/elixir_make/Mix.Tasks.Compile.ElixirMake.html)
- [SSCMA Repository](https://github.com/seeed-studio/SSCMA)
- [SG2002 Documentation](https://developer.crowpi.com.cn/docs/SG2002)
- [Nx Documentation](https://hexdocs.pm/nx)
