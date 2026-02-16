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

  @doc """
  Get input tensor as map with binary data.

  Returns a map with keys:
  - `:shape` - tensor shape as list of integers
  - `:type` - tensor type atom (:u8, :s8, :f32, etc.)
  - `:size` - tensor size in bytes
  - `:quant_param` - quantization parameters map with `:scale` and `:zero_point`
  - `:data` - tensor data as binary
  - `:name` - tensor name string
  - `:is_physical` - boolean atom (:true or :false)
  """
  @spec get_input(t(), non_neg_integer()) :: {:ok, map()} | {:error, term()}
  def get_input(%__MODULE__{resource: resource}, index) do
    SSCMEx.Nif.engine_cvi_get_input(resource, index)
  end

  @doc """
  Get output tensor as map with binary data.

  See `get_input/2` for return format.
  """
  @spec get_output(t(), non_neg_integer()) :: {:ok, map()} | {:error, term()}
  def get_output(%__MODULE__{resource: resource}, index) do
    SSCMEx.Nif.engine_cvi_get_output(resource, index)
  end

  @doc """
  Get input tensor shape as list of integers.
  """
  @spec get_input_shape(t(), non_neg_integer()) :: {:ok, [integer()]} | {:error, term()}
  def get_input_shape(%__MODULE__{resource: resource}, index) do
    SSCMEx.Nif.engine_cvi_get_input_shape(resource, index)
  end

  @doc """
  Get output tensor shape as list of integers.
  """
  @spec get_output_shape(t(), non_neg_integer()) :: {:ok, [integer()]} | {:error, term()}
  def get_output_shape(%__MODULE__{resource: resource}, index) do
    SSCMEx.Nif.engine_cvi_get_output_shape(resource, index)
  end

  @doc """
  Get input quantization parameters.

  Returns a map with `:scale` (float) and `:zero_point` (integer).
  """
  @spec get_input_quant_param(t(), non_neg_integer()) :: {:ok, map()} | {:error, term()}
  def get_input_quant_param(%__MODULE__{resource: resource}, index) do
    SSCMEx.Nif.engine_cvi_get_input_quant_param(resource, index)
  end

  @doc """
  Get output quantization parameters.

  Returns a map with `:scale` (float) and `:zero_point` (integer).
  """
  @spec get_output_quant_param(t(), non_neg_integer()) :: {:ok, map()} | {:error, term()}
  def get_output_quant_param(%__MODULE__{resource: resource}, index) do
    SSCMEx.Nif.engine_cvi_get_output_quant_param(resource, index)
  end

  @doc """
  Set input tensor data from binary.

  The binary size must match the expected tensor size exactly.
  """
  @spec set_input(t(), non_neg_integer(), binary()) :: :ok | {:error, term()}
  def set_input(%__MODULE__{resource: resource}, index, data) when is_binary(data) do
    case SSCMEx.Nif.engine_cvi_set_input(resource, index, data) do
      {:ok, _} -> :ok
      error -> error
    end
  end

  @doc """
  Get input tensor index by name.

  Returns the index of the input tensor with the given name.
  """
  @spec get_input_num(t(), String.t()) :: {:ok, integer()} | {:error, term()}
  def get_input_num(%__MODULE__{resource: resource}, name) do
    SSCMEx.Nif.engine_cvi_get_input_num(resource, name)
  end

  @doc """
  Get output tensor index by name.

  Returns the index of the output tensor with the given name.
  """
  @spec get_output_num(t(), String.t()) :: {:ok, integer()} | {:error, term()}
  def get_output_num(%__MODULE__{resource: resource}, name) do
    SSCMEx.Nif.engine_cvi_get_output_num(resource, name)
  end

  @doc """
  Run inference (blocking, uses dirty CPU scheduler).

  After calling this function, output tensors will contain the inference results.
  """
  @spec run(t()) :: :ok | {:error, term()}
  def run(%__MODULE__{resource: resource}) do
    case SSCMEx.Nif.engine_cvi_run(resource) do
      {:ok, _} -> :ok
      error -> error
    end
  end
end
