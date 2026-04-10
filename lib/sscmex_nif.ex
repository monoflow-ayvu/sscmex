defmodule SSCMEx.Nif do
  @moduledoc false
  @on_load :load_nif

  require Logger
  @supported_nerves_targets ["nerves_system_sg2002"]

  def load_nif do
    path = :filename.join(:code.priv_dir(:sscmex), ~c"sscmex_nif")

    case :erlang.load_nif(path, 0) do
      :ok ->
        :ok

      {:error, {:reload, _}} ->
        :ok

      err ->
        handle_load_error(err)
    end
  end

  defp handle_load_error(err) do
    if :erlang.system_info(:system_architecture) == ~c"riscv64-buildroot-linux-musl" do
      raise "SSCMEx load_nif: #{inspect(err)}"
    end

    case System.get_env("MIX_TARGET") do
      <<"nerves_system_", _::binary>> = mix_target ->
        if mix_target in @supported_nerves_targets do
          :ok
        else
          Logger.error("SSCMEx: Nerves target not supported: #{mix_target}")
          :ok
        end

      _ ->
        :ok
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
  def model_get_config(_model_resource, _option), do: :erlang.nif_error(:nif_not_loaded)
  def model_get_perf(_model_resource), do: :erlang.nif_error(:nif_not_loaded)

  # Device resource functions
  def device_get_instance, do: :erlang.nif_error(:nif_not_loaded)
  def device_get_info(_device_resource), do: :erlang.nif_error(:nif_not_loaded)
  def device_get_sensors(_device_resource), do: :erlang.nif_error(:nif_not_loaded)
  def device_get_models(_device_resource), do: :erlang.nif_error(:nif_not_loaded)

  # Camera resource functions
  def camera_get(_device_resource, _index), do: :erlang.nif_error(:nif_not_loaded)
  def camera_count(_device_resource), do: :erlang.nif_error(:nif_not_loaded)
  def camera_init(_camera_resource, _preset_idx), do: :erlang.nif_error(:nif_not_loaded)
  def camera_deinit(_camera_resource), do: :erlang.nif_error(:nif_not_loaded)
  def camera_get_presets(_camera_resource), do: :erlang.nif_error(:nif_not_loaded)
  def camera_get_preset_idx(_camera_resource), do: :erlang.nif_error(:nif_not_loaded)
  def camera_is_initialized(_camera_resource), do: :erlang.nif_error(:nif_not_loaded)
  def camera_start_stream(_camera_resource, _mode), do: :erlang.nif_error(:nif_not_loaded)
  def camera_stop_stream(_camera_resource), do: :erlang.nif_error(:nif_not_loaded)
  def camera_is_streaming(_camera_resource), do: :erlang.nif_error(:nif_not_loaded)
  def camera_retrieve_frame(_camera_resource, _format), do: :erlang.nif_error(:nif_not_loaded)
  def camera_set_ctrl(_camera_resource, _ctrl, _value), do: :erlang.nif_error(:nif_not_loaded)
  def camera_get_ctrl(_camera_resource, _ctrl), do: :erlang.nif_error(:nif_not_loaded)
  def camera_get_id(_camera_resource), do: :erlang.nif_error(:nif_not_loaded)
  def camera_request_keyframe(_camera_resource, _channel), do: :erlang.nif_error(:nif_not_loaded)

  # Image processing functions
  def image_convert(_image, _format, _opts), do: :erlang.nif_error(:nif_not_loaded)
  def image_resize(_image, _dimensions, _opts), do: :erlang.nif_error(:nif_not_loaded)

  # ISP functions
  def isp_available, do: :erlang.nif_error(:nif_not_loaded)
end
