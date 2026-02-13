defmodule SscmexExample.Application do
  @moduledoc false

  use Application

  def start(_type, _args) do
    SscmexExample.start(nil, nil)
  end
end
