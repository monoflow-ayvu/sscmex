defmodule SSCMEx.Nif do
  @moduledoc false
  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:sscmex), ~c"sscmex_nif")
    :erlang.load_nif(path, 0)
  end

  # EngineCVI resource functions
  defdelegate engine_cvi_new, to: SSCMEx.Nif.Loader
  defdelegate engine_cvi_init(resource), to: SSCMEx.Nif.Loader
  defdelegate engine_cvi_load(resource, path), to: SSCMEx.Nif.Loader
  defdelegate engine_cvi_get_input_size(resource), to: SSCMEx.Nif.Loader
  defdelegate engine_cvi_get_output_size(resource), to: SSCMEx.Nif.Loader
end

defmodule SSCMEx.Nif.Loader do
  @moduledoc false

  def engine_cvi_new, do: :erlang.nif_error(:undefined_function)
  def engine_cvi_init(_resource), do: :erlang.nif_error(:undefined_function)
  def engine_cvi_load(_resource, _path), do: :erlang.nif_error(:undefined_function)
  def engine_cvi_get_input_size(_resource), do: :erlang.nif_error(:undefined_function)
  def engine_cvi_get_output_size(_resource), do: :erlang.nif_error(:undefined_function)
end
