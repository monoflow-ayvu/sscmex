defmodule Sscmex.Nif do
  @moduledoc false

  def load_nif do
    path = :filename.join(:code.priv_dir(:sscmex), ~c"sscmex_nif")
    :erlang.load_nif(path, 0)
  end

  defdelegate hello, to: Sscmex.Nif.Loader
end

defmodule Sscmex.Nif.Loader do
  def hello, do: :erlang.nif_error(:undefined_function)
end
