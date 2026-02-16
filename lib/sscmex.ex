defmodule SSCMEx do
  @moduledoc """
  SSCMA NIF bindings for SG2002 chip (reCamera).

  Provides high-level Elixir interface to SSCMA (Smart Sensor and Control Module for AI)
  using SSCMA-Micro as the backend for TPU operations.

  The NIF is automatically loaded when the module is first used.

  ## Quick Start

      {:ok, engine} = SSCMEx.Engine.new()
      :ok = SSCMEx.Engine.load(engine, "/path/to/model.cvimodel")
      {:ok, input_count} = SSCMEx.Engine.get_input_size(engine)
  """

  @doc """
  Loads the NIF library manually (usually not needed - auto-loaded via @on_load).
  """
  defdelegate load_nif, to: SSCMEx.Nif
end
