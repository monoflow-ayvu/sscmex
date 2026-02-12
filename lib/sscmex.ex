defmodule Sscmex do
  @moduledoc """
  SSCMA NIF bindings for SG2002 chip.

  Provides high-level Elixir interface to SSCMA (Smart Sensor and Control Module for AI)
  running on the reCamera SG2002 chip.
  """

  @doc """
  Returns a greeting from the NIF.

  ## Examples

      iex> Sscmex.hello()
      {:ok, "Hello from SSCMEx NIF!"}
  """
  defdelegate hello, to: Sscmex.Nif
end
