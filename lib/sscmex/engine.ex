defmodule SSCMEx.Engine do
  @moduledoc """
  SSCMA-Micro EngineCVI wrapper.

  The engine is automatically garbage collected when no longer referenced.
  """

  @type resource :: reference()
  @type t :: %__MODULE__{resource: resource()}

  defstruct [:resource]

  @doc """
  Creates and initializes a new CVI engine.

  ## Examples

      {:ok, engine} = SSCMEx.Engine.new()
  """
  @spec new() :: {:ok, t()} | {:error, term()}
  def new do
    with {:ok, resource} <- SSCMEx.Nif.engine_cvi_new(),
         {:ok, :initialized} <- SSCMEx.Nif.engine_cvi_init(resource) do
      {:ok, %__MODULE__{resource: resource}}
    end
  end

  @doc """
  Loads a model from file path.

  ## Examples

      :ok = SSCMEx.Engine.load(engine, "/path/to/model.cvimodel")
  """
  @spec load(t(), String.t()) :: :ok | {:error, term()}
  def load(%__MODULE__{resource: resource}, path) do
    case SSCMEx.Nif.engine_cvi_load(resource, path) do
      {:ok, :loaded} -> :ok
      error -> error
    end
  end

  @doc """
  Returns the number of input tensors.
  """
  @spec get_input_size(t()) :: {:ok, non_neg_integer()} | {:error, term()}
  def get_input_size(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.engine_cvi_get_input_size(resource)
  end

  @doc """
  Returns the number of output tensors.
  """
  @spec get_output_size(t()) :: {:ok, non_neg_integer()} | {:error, term()}
  def get_output_size(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.engine_cvi_get_output_size(resource)
  end
end
