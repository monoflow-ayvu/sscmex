defmodule Sscmex.MixProject do
  use Mix.Project

  @version "0.1.5"
  @github_url "https://github.com/monoflow-ayvu/sscmex"

  def project do
    [
      app: :sscmex,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      description: "SSCMA NIF bindings for SG2002 chip",
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_precompiler: {:nif, CCPrecompiler},
      make_precompiler_url: "#{@github_url}/releases/download/v#{@version}/@{artefact_filename}",
      make_precompiler_filename: "sscmex_nif",
      make_precompiler_priv_paths: ["sscmex_nif.so", "lib"],
      cc_precompiler: [
        compilers: %{
          {:unix, :linux} => %{
            "riscv64-buildroot-linux-musl" => {
              "riscv64-unknown-linux-musl-gcc",
              "riscv64-unknown-linux-musl-g++"
            }
          }
        }
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:elixir_make, "~> 0.8", runtime: false},
      {:cc_precompiler, "~> 0.1", runtime: false},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end

  defp package do
    files =
      ~w(
        c_src
        cmake
        lib
        vendor
        CMakeLists.txt
        Makefile
        README.md
        mix.exs
      )
      |> maybe_add_checksum()

    [
      name: "sscmex",
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @github_url},
      files: files
    ]
  end

  defp maybe_add_checksum(files) do
    if File.exists?("checksum.exs"), do: ["checksum.exs" | files], else: files
  end
end
