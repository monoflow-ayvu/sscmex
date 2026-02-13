defmodule SSCMEx do
  @moduledoc """
  SSCMA NIF bindings for SG2002 chip.

  Provides high-level Elixir interface to SSCMA (Smart Sensor and Control Module for AI)
  running on the reCamera SG2002 chip.
  """

  @doc """
  Returns a greeting from the NIF.

  ## Examples

      iex> SSCMEx.hello()
      {:ok, "Hello from SSCMEx NIF!"}
  """
  defdelegate hello, to: SSCMEx.Nif

  @doc """
  Loads the NIF library. Called automatically on application start.

  Returns `:ok` on success, `{:error, reason}` on failure.
  """
  defdelegate load_nif, to: SSCMEx.Nif

  @doc """
  Checks if the NIF is loaded and functional.

  ## Examples

      iex> SSCMEx.nif_loaded?()
      true
  """
  def nif_loaded? do
    case hello() do
      {:ok, _} -> true
      _ -> false
    end
  end

  @doc """
  Tests the TPU by initializing the runtime and getting device info.

  Returns `{:ok, info}` with TPU information on success,
  `{:error, reason}` on failure.

  ## Examples

      iex> SSCMEx.tpu_test()
      {:ok, %{chip: "cv181x", version: "1.0.0", status: :ready}}
  """
  defdelegate tpu_test, to: SSCMEx.Nif

  @doc """
  Gets the TPU SDK version.

  ## Examples

      iex> SSCMEx.tpu_version()
      {:ok, "1.0.0"}
  """
  defdelegate tpu_version, to: SSCMEx.Nif
end
