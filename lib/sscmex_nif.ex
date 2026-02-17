defmodule SSCMEx.Nif do
  @moduledoc false
  # @on_load :load_nif

  require Logger

  def load_nif do
    path = :filename.join(:code.priv_dir(:sscmex), ~c"sscmex_nif")

    case :erlang.load_nif(path, 0) do
      :ok ->
        :ok

      {:error, {:reload, _}} ->
        :ok

      err ->
        case :erlang.system_info(:system_architecture) do
          ~c"riscv64-buildroot-linux-musl" ->
            raise "SSCMEx load_nif: #{inspect(err)}"
          _ ->
            Logger.error("SSCMEx: System not supported")
            :ok
        end
    end
  end

  # EngineCVI resource functions
  def engine_cvi_new, do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_init(_resource), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_load(_resource, _path), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_run(_resource), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_get_input_size(_resource), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_get_output_size(_resource), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_get_input(_resource, _index), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_get_output(_resource, _index), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_get_input_shape(_resource, _index), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_get_output_shape(_resource, _index), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_get_input_quant_param(_resource, _index), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_get_output_quant_param(_resource, _index), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_set_input(_resource, _index, _data), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_get_input_num(_resource, _name), do: :erlang.nif_error(:nif_not_loaded)
  def engine_cvi_get_output_num(_resource, _name), do: :erlang.nif_error(:nif_not_loaded)

  # Model resource functions
  def model_create(_engine_resource), do: :erlang.nif_error(:nif_not_loaded)
  def model_create(_engine_resource, _algorithm_id), do: :erlang.nif_error(:nif_not_loaded)
  def model_get_type(_model_resource), do: :erlang.nif_error(:nif_not_loaded)
  def model_get_name(_model_resource), do: :erlang.nif_error(:nif_not_loaded)
  def model_get_input_type(_model_resource), do: :erlang.nif_error(:nif_not_loaded)
  def model_get_output_type(_model_resource), do: :erlang.nif_error(:nif_not_loaded)
  def model_run(_model_resource, _image_struct), do: :erlang.nif_error(:nif_not_loaded)
  def model_set_config(_model_resource, _option, _value), do: :erlang.nif_error(:nif_not_loaded)
  def model_get_perf(_model_resource), do: :erlang.nif_error(:nif_not_loaded)
end
