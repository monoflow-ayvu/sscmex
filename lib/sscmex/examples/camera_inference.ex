defmodule SSCMEx.Examples.CameraInference do
  @moduledoc """
  Full example: load model, configure camera to model input size, capture one frame,
  then run inference.

  Run from IEx (from the sscmex or example project):

      SSCMEx.load_nif()

      # run_once returns {:ok, result} or {:error, reason} — handle both
      case SSCMEx.Examples.CameraInference.run_once("/data/yolo11n_detection_cv181x_int8.cvimodel") do
        {:ok, result} -> IO.inspect(result.detections)
        {:error, reason} -> IO.puts("Failed: \#{inspect(reason)}")
      end

  If you get `{:error, ~c"retrieve_frame_failed"}`: check dmesg. "vpss_get_chn_frame fail" or
  "vi err" means the camera/VI pipeline is not delivering frames (sensor or timing). Try
  `preset_idx: 3` for a smaller resolution, or a longer warm-up. "ion allocated failed" means
  out of ION memory. This example reduces peak usage by configuring the camera output
  to the model input resolution before streaming.
  """

  defmodule Error do
    @moduledoc false
    defexception [:message]
  end

  @doc """
  One-shot example: load model, configure camera to model input size, then run inference.

  ## Options

  - `:model_path` (required) - Path to the .cvimodel file
  - `:preset_idx` - Sensor preset index (default 3). Output size is still set from model input shape.
  - `:camera_fps` - Camera FPS for the RAW channel (default 3)
  - `:threshold_score` - Detection score threshold 0.0–1.0 (default 0.5)
  - `:threshold_nms` - NMS threshold 0.0–1.0 (default 0.45)
  - `:warm_up_ms` - Delay after start_stream before first retrieve (default 1500). Increase if dmesg shows "vi err".

  ## Returns

  - `{:ok, %{frame: image, detections: detections, perf: perf}}` on success
  - `{:error, reason}` on failure

  ## Example

      case SSCMEx.Examples.CameraInference.run_once(
             "/data/yolo11n_detection_cv181x_int8.cvimodel",
             preset_idx: 3,
             threshold_score: 0.5
           ) do
        {:ok, result} ->
          IO.inspect(result.detections)
          IO.inspect(result.perf)
        {:error, reason} ->
          IO.puts("Error: \#{inspect(reason)}")
      end

  Or use `run_once!/2` to raise on failure (so you get a clear error instead of MatchError):

      result = SSCMEx.Examples.CameraInference.run_once!(path, preset_idx: 3)
  """
  def run_once(model_path, opts \\ []) do
    preset_idx = Keyword.get(opts, :preset_idx, 3)
    camera_fps = Keyword.get(opts, :camera_fps, 3)
    threshold_score = Keyword.get(opts, :threshold_score, 0.5)
    threshold_nms = Keyword.get(opts, :threshold_nms, 0.45)
    warm_up_ms = Keyword.get(opts, :warm_up_ms, 1500)

    with {:ok, device} <- SSCMEx.Device.get_instance(),
         {:ok, 1} <- SSCMEx.Camera.count(device),
         {:ok, camera} <- SSCMEx.Camera.get(device, 0),
         {:ok, engine} <- SSCMEx.Engine.new(),
         :ok <- SSCMEx.Engine.load(engine, model_path),
         {:ok, {input_w, input_h}} <- model_input_window(engine),
         {:ok, model} <- SSCMEx.Model.create(engine),
         :ok <- SSCMEx.Model.set_config(model, :threshold_score, threshold_score),
         :ok <- SSCMEx.Model.set_config(model, :threshold_nms, threshold_nms) do
      run_once_with_camera(
        camera,
        model,
        preset_idx,
        camera_fps,
        input_w,
        input_h,
        warm_up_ms
      )
    end
  end

  @doc """
  Same as `run_once/2` but raises on error so you get a clear exception instead of MatchError.

  Use when you want to pattern-match only on success; on failure you get e.g.:
  `raise SSCMEx.Examples.CameraInference.Error, "retrieve_frame_failed"`

      result = SSCMEx.Examples.CameraInference.run_once!(path, preset_idx: 3)
  """
  def run_once!(model_path, opts \\ []) do
    case run_once(model_path, opts) do
      {:ok, result} -> result
      {:error, reason} -> raise Error, message: format_error(reason)
    end
  end

  defp format_error(reason) when is_list(reason), do: to_string(reason)
  defp format_error(reason), do: inspect(reason)

  @doc """
  Step-by-step flow you can copy into your app.

  This flow mirrors `sscma-elixir`: load model first, then configure camera output to
  model input dimensions before starting the stream.
  """
  def steps do
    """
    # 1. Load NIF (once per VM)
    :ok = SSCMEx.load_nif()

    # 2. Get device and camera
    {:ok, device} = SSCMEx.Device.get_instance()
    {:ok, camera} = SSCMEx.Camera.get(device, 0)

    # 3. Start model (engine + load + create)
    {:ok, engine} = SSCMEx.Engine.new()
    :ok = SSCMEx.Engine.load(engine, "/path/to/model.cvimodel")
    {:ok, [_, input_h, input_w, _]} = SSCMEx.Engine.get_input_shape(engine, 0)
    {:ok, model} = SSCMEx.Model.create(engine)
    :ok = SSCMEx.Model.set_config(model, :threshold_score, 0.5)
    :ok = SSCMEx.Model.set_config(model, :threshold_nms, 0.45)

    # 4. Init and configure camera RAW channel to model input size
    {:ok, :initialized} = SSCMEx.Camera.init(camera, 3)
    {:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :channel, 0)
    {:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :window, {input_w, input_h})
    {:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :format, :rgb888)
    {:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :fps, 3)
    {:ok, :streaming} = SSCMEx.Camera.start_stream(camera, :refresh_on_return)

    # 5. Grab a frame and run through model
    {:ok, image} = SSCMEx.Camera.retrieve_frame(camera, :rgb888)
    {:ok, detections} = SSCMEx.Model.run(model, image)

    # 6. Get outputs (for a detector: list of %{x, y, w, h, score, target})
    IO.inspect(detections)

    # Optional: timing
    {:ok, perf} = SSCMEx.Model.get_perf(model)
    # perf = %{preprocess: ms, inference: ms, postprocess: ms}
    """
  end

  defp run_once_with_camera(
         camera,
         model,
         preset_idx,
         camera_fps,
         input_w,
         input_h,
         warm_up_ms
       ) do
    try do
      with {:ok, :initialized} <- SSCMEx.Camera.init(camera, preset_idx),
           :ok <- configure_raw_channel(camera, input_w, input_h, camera_fps),
           {:ok, :streaming} <- SSCMEx.Camera.start_stream(camera, :refresh_on_return),
           # Give VI/VPSS time to stabilize; "vi err" in dmesg often needs a longer warm-up
           :ok <- wait_ms(warm_up_ms),
           {:ok, frame} <- retrieve_frame_with_retry(camera, 10, 200),
           # Build image and run inference
           {:ok, detections} <- run_frame(model, frame),
           {:ok, perf} <- SSCMEx.Model.get_perf(model) do
        {:ok,
         %{
           frame: frame,
           detections: detections,
           perf: perf
         }}
      end
    after
      safe_camera_shutdown(camera)
    end
  end

  defp model_input_window(engine) do
    with {:ok, shape} <- SSCMEx.Engine.get_input_shape(engine, 0) do
      infer_hw_from_shape(shape)
    end
  end

  defp infer_hw_from_shape([_n, h, w, c])
       when is_integer(h) and is_integer(w) and c in [1, 3] and h > 0 and w > 0,
       do: {:ok, {w, h}}

  defp infer_hw_from_shape([_n, c, h, w])
       when is_integer(h) and is_integer(w) and c in [1, 3] and h > 0 and w > 0,
       do: {:ok, {w, h}}

  defp infer_hw_from_shape([h, w, c])
       when is_integer(h) and is_integer(w) and c in [1, 3] and h > 0 and w > 0,
       do: {:ok, {w, h}}

  defp infer_hw_from_shape([c, h, w])
       when is_integer(h) and is_integer(w) and c in [1, 3] and h > 0 and w > 0,
       do: {:ok, {w, h}}

  defp infer_hw_from_shape(_shape), do: {:error, ~c"unsupported_input_shape"}

  defp configure_raw_channel(camera, width, height, fps) do
    with {:ok, :ok} <- SSCMEx.Camera.set_ctrl(camera, :channel, 0),
         {:ok, :ok} <- SSCMEx.Camera.set_ctrl(camera, :window, {width, height}),
         {:ok, :ok} <- SSCMEx.Camera.set_ctrl(camera, :format, :rgb888),
         {:ok, :ok} <- SSCMEx.Camera.set_ctrl(camera, :fps, fps) do
      :ok
    end
  end

  defp safe_camera_shutdown(camera) do
    _ = safe_call(fn -> SSCMEx.Camera.stop_stream(camera) end)
    _ = safe_call(fn -> SSCMEx.Camera.deinit(camera) end)
    :ok
  end

  defp safe_call(fun) do
    fun.()
  rescue
    _ -> :ok
  catch
    _, _ -> :ok
  end

  defp run_frame(model, %SSCMEx.Image{} = image), do: SSCMEx.Model.run(model, image)

  defp run_frame(model, %{width: w, height: h, format: f, data: data}) do
    SSCMEx.Model.run(model, SSCMEx.Image.new(w, h, f, data))
  end

  defp retrieve_frame_with_retry(camera, attempts, delay_ms) when attempts > 0 do
    case SSCMEx.Camera.retrieve_frame(camera, :rgb888) do
      {:ok, frame} ->
        {:ok, frame}

      _ ->
        Process.sleep(delay_ms)
        retrieve_frame_with_retry(camera, attempts - 1, delay_ms)
    end
  end

  defp retrieve_frame_with_retry(_camera, 0, _delay_ms), do: {:error, ~c"retrieve_frame_failed"}

  defp wait_ms(ms) do
    Process.sleep(ms)
    :ok
  end
end
