defmodule SscmexExample.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/fermuch/sscmex"

  def project do
    [
      app: :sscmex_example,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      source_url: @source_url,
      compilers: Mix.compilers(),
      aliases: [loadconfig: [&bootstrap/1]]
    ]
  end

  defp bootstrap(args) do
    Application.start(:nerves_bootstrap)
    Mix.Task.run("loadconfig", args)
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      mod: {SscmexExample.Application, []},
      extra_applications: [:logger, :runtime_tools]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:sscmex, path: ".."},
      {:nerves_bootstrap, "~> 1.13", runtime: false},
      {:nerves, "~> 1.10", runtime: false},
      {:nerves_runtime, "~> 0.13"},
      {:vintage_net, "~> 0.3"},
      # SG2002 Nerves system - specify as application dependency
      {:nerves_system_sg2002,
       github: "fermuch/nerves_system_sg2002",
       runtime: false,
       # Tell Nerves which targets this supports
       app: :sscmex_example,
       targets: [:nerves_system_sg2002],
       nerves: [compile: true]}
    ]
  end
end
