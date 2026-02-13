{ pkgs, config, ... }:

{
  # Basic packages needed for Nerves firmware building
  packages = with pkgs; [
    # Build essentials
    cmake
    gnumake
    pkg-config
    ninja
    git
    wget
    curl
    unzip

    # Basic system tools
    coreutils
    bashInteractive
    which
    findutils

    # Elixir development
    inotify-tools

    # Firmware tools
    xz

    # Library dependencies
    libmnl
  ];

  # Nerves target for this project
  env = {
    MIX_TARGET = "nerves_system_sg2002";
    C_INCLUDE_PATH = pkgs.lib.makeIncludePath (with pkgs; [ libmnl ]);
    LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [ libmnl ]);
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [ libmnl ]);
  };

  # Shell initialization
  enterShell = ''
    # Enable shell history for erlang/elixir
    export ERL_AFLAGS="-kernel shell_history enabled"

    # Configure UTF-8 locale for proper Unicode support
    export LC_ALL="en_US.UTF-8"
    export LANG="en_US.UTF-8"

    echo "🚀 Nerves Firmware Build Environment Ready!"
    echo "🎯 MIX_TARGET: $MIX_TARGET"
    echo ""
  '';

  # Enable Cachix
  cachix = {
    enable = true;
    pull = [ "pre-commit-hooks" "fermuch" ];
  };
}
