defmodule SSCMEx do
  @moduledoc """
  SSCMA NIF bindings for SG2002 chip (reCamera).

  Provides high-level Elixir interface to SSCMA (Smart Sensor and Control Module for AI)
  using SSCMA-Micro as the backend for TPU operations.

  The NIF is automatically loaded when the module is first used.
  """

  @doc """
  Returns a greeting from the NIF.

  ## Examples

      iex> SSCMEx.hello()
      {:ok, "Hello from SSCMEx NIF!"}
  """
  defdelegate hello, to: SSCMEx.Nif

  @doc """
  Loads the NIF library.

  This is called automatically via `@on_load` when the module is first referenced.
  You only need to call this manually if you want to reload the NIF.

  Returns `:ok` on success, `{:error, reason}` on failure.
  """
  defdelegate load_nif, to: SSCMEx.Nif

  @doc """
  Initializes the TPU via SSCMA-Micro.

  This function creates an EngineCVI instance and initializes it to verify
  the SSCMA-Micro library is properly integrated.

  Returns `{:ok, info}` on success, `{:error, reason}` on failure.

  ## Examples

      iex> SSCMEx.sscma_init()
      {:ok, %{success: true, engine: :cvi, library: :sscma_micro, status: :ok, sscma_version: "2.0.0", chip: :sg2002}}
  """
  defdelegate sscma_init, to: SSCMEx.Nif

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
end
