{ pkgs, config, ... }:

{
  # Basic packages needed for development environment
  packages = with pkgs; [
    # Build essentials
    cmake
    gnumake
    pkg-config
    ninja

    # Additional tools
    git
    wget
    curl
    unzip

    # Tools for SDK extraction
    gnutar
    gzip
    findutils

    # Basic system tools
    coreutils
    bashInteractive
    which

    # Libraries needed by cross-compiler
    zlib
    ncurses
    glibc

    # Elixir development
    inotify-tools

    # Used by the example
    autoconf
    automake
    fwup
    squashfsTools
    libmnl
    inotify-tools
  ];

  # Environment variables with absolute paths
  env = {
    SG200X_FHS = "1";
    CMAKE_TOOLCHAIN_FILE = "${config.env.DEVENV_ROOT}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake";
    SG200X_SDK_PATH = "${config.env.DEVENV_ROOT}/.devenv/state/sg200x-sdk/sg2002_recamera_emmc";
    # Default Nerves target for the example project
    MIX_TARGET = "nerves_system_sg2002";
  };

  languages = {
    elixir = {
      enable = true;
      package = pkgs.beam28Packages.elixir_1_18;
    };
    erlang = {
      enable = true;
      package = pkgs.beam28Packages.erlang;
    };
  };

  # Task to setup SDK before entering shell
  tasks."sg200x:setup-sdk" = {
    exec = ''
      # Setup SDK and host-tools if not present
      RECAMERA_ROOT="${config.env.DEVENV_ROOT}/.devenv/state/sg200x-sdk"

      if [ ! -d "$RECAMERA_ROOT/sg2002_recamera_emmc" ]; then
        echo "📦 Setting up SDK and host-tools..."
        mkdir -p "$RECAMERA_ROOT"

        # Download and extract SDK
        echo "📁 Downloading and extracting SDK..."
        if [ ! -f "$RECAMERA_ROOT/reCameraOS_sdk_v0.2.0.tar.gz" ]; then
          wget -O "$RECAMERA_ROOT/reCameraOS_sdk_v0.2.0.tar.gz" \
            "https://github.com/Seeed-Studio/reCamera-OS/releases/download/0.2.0/reCameraOS_sdk_v0.2.0.tar.gz"
        fi

        tar -xzf "$RECAMERA_ROOT/reCameraOS_sdk_v0.2.0.tar.gz" -C "$RECAMERA_ROOT" --strip-components=0

        # Remove broken symlinks
        echo "Cleaning up broken symlinks..."
        find "$RECAMERA_ROOT" -type l -exec test ! -e {} \; -print | while read link; do
          echo "Removing broken symlink: $link"
          rm -f "$link"
        done

        # Clone host-tools
        echo "🔧 Cloning host-tools..."
        git clone --verbose --progress https://github.com/sophgo/host-tools.git "$RECAMERA_ROOT/host-tools"

        echo "✅ SDK setup complete!"
      else
        echo "✅ SDK already set up in $RECAMERA_ROOT"
      fi
    '';
    before = [ "devenv:enterShell" ];
  };

  # Shell initialization script (runs after SDK setup)
  enterShell = ''
    # Set up environment variables with absolute paths
    RECAMERA_ROOT="${config.env.DEVENV_ROOT}/.devenv/state/sg200x-sdk"
    export SG200X_SDK_PATH="$RECAMERA_ROOT/sg2002_recamera_emmc"

    # Add cross-compiler to PATH with absolute path
    # This allows CMake's find_program() to locate the compiler
    # NOTE: We do NOT set CC/CXX environment variables globally because:
    # 1. CMake uses the toolchain file (CMAKE_TOOLCHAIN_FILE) for cross-compilation
    # 2. Other build tools (like Nerves' port Makefile) need to use the host compiler
    COMPILER_DIR="$RECAMERA_ROOT/host-tools/gcc/riscv64-linux-musl-x86_64/bin"
    export PATH="$COMPILER_DIR:$PATH"

    # Create symlinks for toolchain naming convention
    if [ -d "$COMPILER_DIR" ] && [ ! -f "$COMPILER_DIR/riscv64-unknown-linux-musl-gcc" ]; then
      echo "🔗 Creating compiler symlinks for toolchain compatibility..."
      for tool in gcc g++ objcopy objdump ar as ld nm ranlib strip; do
        if [ -f "$COMPILER_DIR/riscv64-linux-musl-$tool" ]; then
          ln -sf "riscv64-linux-musl-$tool" "$COMPILER_DIR/riscv64-unknown-linux-musl-$tool"
        fi
      done
    fi

    # Enable shell history for erlang/elixir
    export ERL_AFLAGS="-kernel shell_history enabled"

    # Configure UTF-8 locale for proper Unicode support
    export LC_ALL="en_US.UTF-8"
    export LANG="en_US.UTF-8"

    echo "🚀 SG200X devenv Development Environment Ready!"
    echo "📁 SDK Path: $SG200X_SDK_PATH"
    echo "🔧 Toolchain File: ${config.env.CMAKE_TOOLCHAIN_FILE}"
    echo "🔧 Compiler Dir: $COMPILER_DIR"
    echo "🎯 Nerves Target: $MIX_TARGET"
    echo ""
    echo "ℹ️  Cross-compilation is configured via CMAKE_TOOLCHAIN_FILE"
    echo "   Use: cmake -DCMAKE_TOOLCHAIN_FILE=\$CMAKE_TOOLCHAIN_FILE ."
    echo ""
  '';

  # Enable Cachix
  cachix = {
    enable = true;
    pull = [ "pre-commit-hooks" "fermuch" ];
  };

  # Enable devcontainer to generate a devcontainer.json file
  devcontainer.enable = true;

  # Publish libraries to environment
  env.LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [ zlib ncurses glibc libmnl ]);
  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [ zlib ncurses glibc libmnl ]);
  env.PKG_CONFIG_PATH = pkgs.lib.makeSearchPath "lib/pkgconfig" (with pkgs; [ zlib ncurses glibc libmnl ]);
}
