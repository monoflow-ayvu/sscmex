{ pkgs, config, ... }:

let
  # Nix derivation: RISCV SDK tarball (replaces wget + tar in the old task)
  sg200x-sdk = pkgs.stdenv.mkDerivation {
    pname = "sg200x-sdk";
    version = "0.2.0";

    src = pkgs.fetchurl {
      url = "https://github.com/Seeed-Studio/reCamera-OS/releases/download/0.2.0/reCameraOS_sdk_v0.2.0.tar.gz";
      sha256 = "023g8i888pl6gv24s9yglfb3dy1mbl78gdjmjwqbyr7idigamxkm";
    };

    nativeBuildInputs = [ pkgs.findutils ];
    dontBuild = true;
    dontConfigure = true;

    installPhase = ''
      runHook preInstall
      find . -type l | while IFS= read -r link; do
        [ -e "$link" ] || rm -f "$link"
      done
      mkdir -p "$out"
      cp -r . "$out/"
      runHook postInstall
    '';
  };

  # Nix derivation: sophgo/host-tools RISCV cross-compiler
  # autoPatchelfHook replaces the manual NixOS ELF-patching logic in the old enterShell
  sg200x-host-tools = pkgs.stdenv.mkDerivation {
    pname = "sg200x-host-tools";
    version = "unstable-2024-12-19";

    src = pkgs.fetchFromGitHub {
      owner = "sophgo";
      repo = "host-tools";
      rev = "103c66f126fa98fcaa8b54f37fa06f6b293fd074";
      hash = "sha256-OARUHjWRIcsKo0LVm1T4/CBaf2Lis3YKO9ZXfC5KD8E=";
    };

    nativeBuildInputs = [ pkgs.autoPatchelfHook ];

    # Skip unresolvable deps from GDB binaries (gdb is not needed for cross-compilation)
    autoPatchelfIgnoreMissingDeps = [
      "libncursesw.so.5"
      "libncurses.so.5"
      "libtinfo.so.5"
      "libpython2.7.so.1.0"
      "liblzma.so.5"
      "libcrypt.so.1"
      "libdb-5.3.so"
      "libgdbm.so.5"
      "libuuid.so.1"
      "libreadline.so.7"
      "libpython3.9.so.1.0"
    ];

    # Runtime libs needed by the pre-built x86_64 GCC host binaries
    buildInputs = [
      pkgs.zlib
      pkgs.ncurses
      pkgs.stdenv.cc.cc.lib  # libstdc++.so.6, libgcc_s.so.1
      pkgs.libmpc
      pkgs.mpfr
      pkgs.gmp
    ];

    dontBuild = true;
    dontConfigure = true;

    installPhase = ''
      runHook preInstall

      # Remove dangling symlinks (sysroot pseudo-fs links like /dev/fd -> /proc/self/fd
      # that can't exist in the Nix store)
      find . -type l | while IFS= read -r link; do
        [ -e "$link" ] || rm -f "$link"
      done

      mkdir -p "$out"
      cp -r gcc "$out/"

      # Expose compilers in $out/bin — Nix adds this to PATH automatically.
      # Also create riscv64-unknown-linux-musl-* aliases expected by CMake.
      mkdir -p "$out/bin"
      bin_dir="$out/gcc/riscv64-linux-musl-x86_64/bin"
      for f in "$bin_dir"/*; do
        base=$(basename "$f")
        ln -sf "$f" "$out/bin/$base"
        alt=$(echo "$base" | sed 's/riscv64-linux-musl-/riscv64-unknown-linux-musl-/')
        if [ "$alt" != "$base" ]; then
          ln -sf "$f" "$out/bin/$alt"
        fi
      done

      runHook postInstall
    '';
  };

in {
  packages = with pkgs; [
    cmake gnumake pkg-config ninja clang
    git wget curl unzip
    gnutar gzip findutils
    coreutils bashInteractive which
    zlib ncurses
    inotify-tools
    autoconf automake fwup squashfsTools libmnl
    sg200x-host-tools
  ];

  env = {
    SG200X_FHS = "1";
    CMAKE_TOOLCHAIN_FILE = "${config.env.DEVENV_ROOT}/cmake/toolchain-riscv64-linux-musl-x86_64.cmake";
    SG200X_SDK_PATH = "${sg200x-sdk}";
    SG200X_HOST_TOOLS_PATH = "${sg200x-host-tools}";
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

  enterShell = ''
    export ERL_AFLAGS="-kernel shell_history enabled"
    export LC_ALL="en_US.UTF-8"
    export LANG="en_US.UTF-8"

    echo "SG200X devenv Development Environment Ready!"
    echo "SDK Path: $SG200X_SDK_PATH"
    echo "Toolchain File: $CMAKE_TOOLCHAIN_FILE"
    echo "Host Tools: $SG200X_HOST_TOOLS_PATH"
    echo "Nerves Target: $MIX_TARGET"
  '';

  cachix = {
    enable = true;
    pull = [ "pre-commit-hooks" "fermuch" ];
  };

  devcontainer.enable = true;
}
