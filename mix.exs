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
      deps: deps(),
      package: package(),
      description: "SSCMA NIF bindings for SG2002 chip",
      compilers: [:elixir_make] ++ Mix.compilers()
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
      {:elixir_make, "~> 0.8", runtime: false}
    ]
  end

  defp package do
    [
      name: "sscmex",
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @github_url}
    ]
  end
end
