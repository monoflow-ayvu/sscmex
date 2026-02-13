defmodule SscmexExample do
  @moduledoc """
  Sscmex Example Application

  A minimal Nerves application to test SSCMEx NIF on real SG2002 hardware
  with VintageNet network configuration for WiFi.
  """

  use Application
  require Logger

  def start(_type, _args) do
    Logger.info("Starting Sscmex Example Application...")
    Logger.info("Application: #{Application.spec(:sscmex_example)}")
    Logger.info("Target: #{Mix.target()}")

    # Test loading NIF
    case Sscmex.Nif.load_nif() do
      :ok ->
        Logger.info("SSCMEx NIF loaded successfully!")
        test_nif()
      {:error, reason} ->
        Logger.error("Failed to load SSCMEx NIF: #{inspect(reason)}")
    end

    {:ok, self}
  end

  defp test_nif do
    case Sscmex.hello() do
      {:ok, message} ->
        Logger.info("NIF test passed: #{message}")
      {:error, reason} ->
        Logger.error("NIF test failed: #{inspect(reason)}")
    end
end
