defmodule SSCMEx.Nif do
  @moduledoc false

  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:sscmex), ~c"sscmex_nif")
    :erlang.load_nif(path, 0)
  end

  defdelegate hello, to: SSCMEx.Nif.Loader
  defdelegate sscma_init, to: SSCMEx.Nif.Loader
end

defmodule SSCMEx.Nif.Loader do
  @moduledoc false

  def hello, do: :erlang.nif_error(:undefined_function)
  def sscma_init, do: :erlang.nif_error(:undefined_function)
end
