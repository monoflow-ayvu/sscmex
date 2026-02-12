# SSCMEx - AI Assistant Context

## Project Overview

SSCMEx is a Native Implemented Function (NIF) based Elixir library for the SG2002 chip (reCamera) that provides bindings to SSCMA (Smart Sensor and Control Module for AI).

## Context

This project converts the existing Port-based approach from `sscma-elixir` to a NIF-based implementation for improved performance when handling binary data from camera and TPU operations.

## Key Requirements

1. **Target Hardware**: SG2002 chip (RISCV) with TPU (reCamera)
2. **Toolchain**: Same as sscma-elixir (CMake-only with RISCV cross-compilation)
3. **NIF Approach**: Replace Port with NIF for better binary data performance
4. **Nx Integration**: Use Nx for tensor operations (camera frames, TPU input/output)
5. **Build System**: Use elixir_cmake to integrate with CMake-based SG2002 SDK

## Reference Projects

The project draws inspiration from and analyzes four reference projects:

- **sscma-elixir** - Port-based, provides the SG2002 toolchain and SDK setup
- **rayex** - NIF using Bundlex/Unifex with flake.nix
- **emlx** - NIF using elixir_make with Nx backend for ML operations
- **adbc** - NIF using elixir_cmake with CMake for complex C++ projects

## Documentation

All detailed analysis, architecture decisions, and implementation patterns are documented in:

**[PROJECT.md](./PROJECT.md)**

Please read PROJECT.md for:
- Deep analysis of each reference project
- Build system comparison and recommendations
- Recommended architecture for SSCMEx
- Implementation checklist
- Code examples for all major components

## Development Setup

Use devenv to set up the development environment with SG2002 SDK:

```bash
devenv shell
```

This will set up:
- SG2002 SDK path
- RISCV cross-compilation toolchain
- CMAKE_TOOLCHAIN_FILE

## Next Steps

When working on this project:
1. Start by reviewing [PROJECT.md](./PROJECT.md) for full context
2. Follow the implementation checklist
3. Refer to the code examples in PROJECT.md for each component

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
