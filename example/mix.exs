defmodule SscmexExample.MixProject do
  use Mix.Project

  @app :sscmex_example
  @version "0.1.0"
  @source_url "https://github.com/fermuch/sscmex"
  @all_targets [:nerves_system_sg2002]

  def project do
    [
      app: @app,
      version: @version,
      elixir: "~> 1.18",
      archives: [nerves_bootstrap: "~> 1.14"],
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      source_url: @source_url,
      compilers: Mix.compilers(),
      aliases: [loadconfig: [&bootstrap/1]],
      releases: [{@app, release()}],
      preferred_cli_target: [run: :host, test: :host]
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
      {:nerves_pack, "~> 0.7.1", targets: @all_targets},
      # SG2002 Nerves system - specify as application dependency
      {:nerves_system_sg2002,
       github: "fermuch/nerves_system_sg2002",
       runtime: false,
       targets: :nerves_system_sg2002,
       nerves: [compile: true]},
    ]
  end

  def release do
    [
      overwrite: true,
      cookie: "#{@app}_cookie",
      include_erts: &Nerves.Release.erts/0,
      steps: [&Nerves.Release.init/1, :assemble],
      strip_beams: Mix.env() == :prod or [keep: ["Docs"]]
    ]
  end
end
