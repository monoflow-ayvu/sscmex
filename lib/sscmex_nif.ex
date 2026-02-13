defmodule SSCMEx.Nif do
  @moduledoc false

  @doc """
  Loads the NIF library.
  """
  def load_nif do
    path = :filename.join(:code.priv_dir(:sscmex), ~c"sscmex_nif")
    :erlang.load_nif(path, 0)
  end

  defdelegate hello, to: SSCMEx.Nif.Loader
  defdelegate tpu_test, to: SSCMEx.Nif.Loader
  defdelegate tpu_version, to: SSCMEx.Nif.Loader
end

defmodule SSCMEx.Nif.Loader do
  @moduledoc false

  def hello, do: :erlang.nif_error(:undefined_function)
  def tpu_test, do: :erlang.nif_error(:undefined_function)
  def tpu_version, do: :erlang.nif_error(:undefined_function)
end
