defmodule SSCMEx.Camera do
  @moduledoc """
  Camera interface for SG2002 (reCamera).

  Provides access to camera hardware for capturing frames and video streams.
  Cameras are obtained from the Device singleton.

  ## Camera and TPU together

  Camera and TPU share ION carveout memory. If you use both and see
  `retrieve_frame_failed` with dmesg "ion allocated failed" or "sys_ion_alloc fail":
  use a lower-resolution preset (for example index 3: 1280x720 @ 30fps), then choose
  the flow that matches your use-case:
  - streaming inference: start camera first, then load engine/model
  - one-shot capture: capture one frame, stop/deinit camera, then load engine/model

  ## Example

      {:ok, device} = SSCMEx.Device.get_instance()
      {:ok, count} = SSCMEx.Camera.count(device)

      if count > 0 do
        {:ok, camera} = SSCMEx.Camera.get(device, 0)

        # Initialize with preset (resolution/fps)
        {:ok, presets} = SSCMEx.Camera.get_presets(camera)
        {:ok, :initialized} = SSCMEx.Camera.init(camera, 3)

        # Start streaming
        {:ok, :streaming} = SSCMEx.Camera.start_stream(camera, :refresh_on_return)

        # Capture frame from channel 0
        {:ok, frame} = SSCMEx.Camera.retrieve_frame(camera, 0)

        # Stop streaming
        {:ok, :stopped} = SSCMEx.Camera.stop_stream(camera)
        {:ok, :deinitialized} = SSCMEx.Camera.deinit(camera)
      end
  """

  @type resource :: reference()
  @type t :: %__MODULE__{resource: resource()}

  defstruct [:resource]

  @type preset :: %{
          description: String.t()
        }

  @type frame :: SSCMEx.Image.t()

  @type stream_mode :: :refresh_on_return | :refresh_on_retrieve
  @type pixel_format ::
          :rgb888
          | :rgb565
          | :yuv422
          | :gray
          | :jpeg
          | :h264
          | :h265
          | :rgb888_planar

  @type ctrl_type ::
          :window
          | :channel
          | :format
          | :fps
          | :quality
          | :ae_mode
          | :max_iso
          | :exposure_us
          | :gain
          | :exposure_range
          | :tnr_enable
          | :tnr_strength
          | :brightness
          | :contrast
          | :saturation
          | :sharpness
          | :nr_strength
          | :ynr_strength
          | :cnr_strength

  @doc """
  Get a camera from the device by index.

  ## Parameters
  - `device` - Device obtained from `SSCMEx.Device.get_instance/0`
  - `index` - Camera index (0-based)

  ## Examples

      {:ok, camera} = SSCMEx.Camera.get(device, 0)
  """
  @spec get(SSCMEx.Device.t(), non_neg_integer()) :: {:ok, t()} | {:error, term()}
  def get(%SSCMEx.Device{resource: device_res}, index) do
    case SSCMEx.Nif.camera_get(device_res, index) do
      {:ok, resource} -> {:ok, %__MODULE__{resource: resource}}
      error -> error
    end
  end

  @doc """
  Get the number of cameras available on the device.

  ## Examples

      {:ok, count} = SSCMEx.Camera.count(device)
  """
  @spec count(SSCMEx.Device.t()) :: {:ok, non_neg_integer()} | {:error, term()}
  def count(%SSCMEx.Device{resource: device_res}) do
    SSCMEx.Nif.camera_count(device_res)
  end

  @doc """
  Initialize the camera with a preset.

  Presets define resolution and framerate combinations.

  ## Parameters
  - `camera` - Camera resource
  - `preset_idx` - Index of preset (see `get_presets/1`)

  ## Examples

      {:ok, :initialized} = SSCMEx.Camera.init(camera, 0)
  """
  @spec init(t(), non_neg_integer()) :: {:ok, :initialized} | {:error, term()}
  def init(%__MODULE__{resource: resource}, preset_idx) do
    SSCMEx.Nif.camera_init(resource, preset_idx)
  end

  @doc """
  Deinitialize the camera.

  ## Examples

      {:ok, :deinitialized} = SSCMEx.Camera.deinit(camera)
  """
  @spec deinit(t()) :: {:ok, :deinitialized} | {:error, term()}
  def deinit(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.camera_deinit(resource)
  end

  @doc """
  Get available presets for the camera.

  Each preset describes a resolution/framerate combination.

  ## Examples

      {:ok, presets} = SSCMEx.Camera.get_presets(camera)
      # => [%{description: "1920x1080 @ 30fps"}, ...]
  """
  @spec get_presets(t()) :: {:ok, [preset()]} | {:error, term()}
  def get_presets(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.camera_get_presets(resource)
  end

  @doc """
  Get the current preset index.

  ## Examples

      {:ok, idx} = SSCMEx.Camera.get_preset_idx(camera)
  """
  @spec get_preset_idx(t()) :: {:ok, non_neg_integer()} | {:error, term()}
  def get_preset_idx(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.camera_get_preset_idx(resource)
  end

  @doc """
  Check if the camera is initialized.

  ## Examples

      {:ok, true} = SSCMEx.Camera.is_initialized(camera)
  """
  @spec is_initialized(t()) :: {:ok, boolean()} | {:error, term()}
  def is_initialized(%__MODULE__{resource: resource}) do
    case SSCMEx.Nif.camera_is_initialized(resource) do
      {:ok, true} -> {:ok, true}
      {:ok, false} -> {:ok, false}
      error -> error
    end
  end

  @doc """
  Start the camera stream.

  ## Stream Modes
  - `:refresh_on_return` - Frame buffer is refreshed when frame is returned
  - `:refresh_on_retrieve` - Frame buffer is refreshed when frame is retrieved

  ## Examples

      {:ok, :streaming} = SSCMEx.Camera.start_stream(camera, :refresh_on_return)
  """
  @spec start_stream(t(), stream_mode()) :: {:ok, :streaming} | {:error, term()}
  def start_stream(%__MODULE__{resource: resource}, mode) do
    SSCMEx.Nif.camera_start_stream(resource, mode)
  end

  @doc """
  Stop the camera stream.

  ## Examples

      {:ok, :stopped} = SSCMEx.Camera.stop_stream(camera)
  """
  @spec stop_stream(t()) :: {:ok, :stopped} | {:error, term()}
  def stop_stream(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.camera_stop_stream(resource)
  end

  @doc """
  Check if the camera is currently streaming.

  ## Examples

      {:ok, true} = SSCMEx.Camera.is_streaming(camera)
  """
  @spec is_streaming(t()) :: {:ok, boolean()} | {:error, term()}
  def is_streaming(%__MODULE__{resource: resource}) do
    case SSCMEx.Nif.camera_is_streaming(resource) do
      {:ok, true} -> {:ok, true}
      {:ok, false} -> {:ok, false}
      error -> error
    end
  end

  @doc """
  Retrieve a frame from the camera.

  ## Pixel Formats
  - `:rgb888` - RGB888 format (raw pixels)
  - `:yuv422` - YUV422 format
  - `:jpeg` - JPEG encoded
  - `:h264` - H.264 encoded video frame
  - `:h265` - H.265 encoded video frame

  ## Channel selection

  Use `configure_channel/3` before `start_stream/2` to assign a format and resolution
  to each channel. Retrieve frames by channel index:

      {:ok, frame} = SSCMEx.Camera.retrieve_frame(camera, 0)

  ## Returns
  Returns `%SSCMEx.Image{}`.

  The image includes:
  - `width`, `height`, `format`, `data`
  - metadata fields when available from the camera pipeline (`size`, `timestamp`, `key`)

  ## Examples

      {:ok, image} = SSCMEx.Camera.retrieve_frame(camera, 0)
      # image.data contains pixels in the format configured for channel 0
  """
  @spec retrieve_frame(t(), 0..2) :: {:ok, frame()} | {:error, term()}
  def retrieve_frame(%__MODULE__{resource: resource}, channel) do
    case SSCMEx.Nif.camera_retrieve_frame(resource, channel) do
      {:ok, %{width: width, height: height, format: image_format, data: data} = frame_map} ->
        {:ok,
         %SSCMEx.Image{
           width: width,
           height: height,
           format: image_format,
           data: data,
           size: Map.get(frame_map, :size, byte_size(data)),
           timestamp: Map.get(frame_map, :timestamp),
           key: Map.get(frame_map, :key)
         }}

      error ->
        error
    end
  end

  @doc """
  Configure a camera channel before `start_stream/2`.

  Must be called after `init/2`. Attempting to change the format after the stream
  has started returns `{:error, _}`.

  ## Options

  - `:format` - pixel format atom. Supported: `:rgb888`, `:yuv422`, `:gray`,
    `:jpeg`, `:h264`, `:h265`, `:nv12`, `:nv21`. Note: `:rgb565` and
    `:rgb888_planar` are not supported as native camera channel formats.
  - `:width` and `:height` - frame dimensions (must both be provided together).
  - `:fps` - target frame rate.

  ## JPEG quality

  Quality control (`set_ctrl(cam, :quality, value)`) applies to the channel that
  is currently selected. Select the channel with `configure_channel/3` (which leaves
  it as current) or `set_ctrl(cam, :channel, n)` before setting quality.

  ## Examples

      # Two RGB888 channels at different resolutions
      :ok = Camera.configure_channel(cam, 0, format: :rgb888, width: 1920, height: 1080)
      :ok = Camera.configure_channel(cam, 1, format: :rgb888, width: 640, height: 480)

      # JPEG on channel 2
      :ok = Camera.configure_channel(cam, 2, format: :jpeg, width: 1280, height: 720)

      # Set JPEG quality for channel 2 (channel 2 is still selected after configure_channel above)
      {:ok, :ok} = Camera.set_ctrl(cam, :quality, 75)
  """
  @spec configure_channel(t(), 0..2, keyword()) :: :ok | {:error, term()}
  def configure_channel(%__MODULE__{} = cam, channel, opts \\ []) when channel in 0..2 do
    with {:ok, :ok} <- set_ctrl(cam, :channel, channel),
         :ok <- maybe_set_ctrl(cam, :format, opts[:format]),
         :ok <- maybe_set_window(cam, opts[:width], opts[:height]),
         :ok <- maybe_set_ctrl(cam, :fps, opts[:fps]) do
      :ok
    end
  end

  defp maybe_set_ctrl(_cam, _ctrl, nil), do: :ok

  defp maybe_set_ctrl(cam, ctrl, value) do
    case set_ctrl(cam, ctrl, value) do
      {:ok, :ok} -> :ok
      error -> error
    end
  end

  defp maybe_set_window(_cam, nil, nil), do: :ok
  defp maybe_set_window(_cam, nil, _), do: {:error, :missing_width}
  defp maybe_set_window(_cam, _, nil), do: {:error, :missing_height}

  defp maybe_set_window(cam, w, h) do
    case set_ctrl(cam, :window, {w, h}) do
      {:ok, :ok} -> :ok
      error -> error
    end
  end

  @doc """
  Set a camera control value.

  ## Standard Controls
  - `:window` - Set resolution `{width, height}`
  - `:channel` - Set channel index
  - `:format` - Set pixel format
  - `:fps` - Set frames per second
  - `:quality` - Set JPEG encoding quality (1-99, value 50 is reserved)

  ## ISP Controls (requires `isp_available?/0` to return `true`)

  ### AE (Auto-Exposure) Controls
  - `:ae_mode` - Set AE mode (`:auto` or `:manual`)
  - `:max_iso` - Set maximum ISO for auto-exposure (100-12800)
  - `:exposure_us` - Set manual exposure time in microseconds (1-1000000)
  - `:gain` - Set manual analog gain, 10-bit precision (1024=1x, 2048=2x, range 1024-65536)
  - `:exposure_range` - Set auto-exposure time limits `{min_us, max_us}` (1-1000000)

  ### TNR (Temporal Noise Reduction) Controls
  - `:tnr_enable` - Enable/disable 3D noise reduction (`true` or `false`)
  - `:tnr_strength` - Set TNR intensity (0-255, manual mode)

  ### Image Tuning Controls
  - `:brightness` - Set image brightness via YContrast CenterLuma (0-255)
  - `:contrast` - Set image contrast via YContrast ContrastHigh (0-255)
  - `:saturation` - Set color saturation (0-255)
  - `:sharpness` - Set edge sharpness via Sharpen GlobalGain (0-255)

  ### Noise Reduction Controls
  - `:nr_strength` - Raw/Bayer spatial NR strength (0-255); higher = less grain
  - `:ynr_strength` - Luma NR strength post-demosaic (0-255); reduces luma noise
  - `:cnr_strength` - Chroma NR strength (0-255); reduces color noise/fringing

  ## Examples

      # Set resolution
      {:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :window, {1280, 720})

      # Set FPS
      {:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :fps, 15)

      # Set JPEG quality (1-99, higher = less compression)
      {:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :quality, 75)

      # Limit max ISO to reduce noise in low light
      {:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :max_iso, 800)

      # Enable 3D noise reduction
      {:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :tnr_enable, true)

      # Increase brightness slightly
      {:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :brightness, 140)
  """
  @spec set_ctrl(t(), ctrl_type(), term()) :: {:ok, :ok} | {:error, term()}
  def set_ctrl(%__MODULE__{resource: resource}, ctrl, value) do
    SSCMEx.Nif.camera_set_ctrl(resource, ctrl, value)
  end

  @doc """
  Get a camera control value.

  ## Control Types and return values

  ### Standard Controls
  - `:window` -> `{width, height}`
  - `:channel` -> channel index integer
  - `:format` -> pixel format atom
  - `:fps` -> fps integer
  - `:quality` -> quality integer (1-99, value 50 is reserved)

  ### ISP Controls
  - `:ae_mode` -> `:auto` or `:manual`
  - `:max_iso` -> maximum ISO integer (100-12800)
  - `:exposure_us` -> current exposure time in microseconds
  - `:gain` -> current analog gain (10-bit: 1024=1x)
  - `:exposure_range` -> `{min_us, max_us}` tuple
  - `:tnr_enable` -> `true` or `false`
  - `:tnr_strength` -> TNR strength integer (0-255)
  - `:brightness` -> brightness integer (0-255)
  - `:contrast` -> contrast integer (0-255)
  - `:saturation` -> saturation integer (0-255)
  - `:sharpness` -> sharpness integer (0-255)
  - `:nr_strength` -> raw/Bayer NR strength integer (0-255)
  - `:ynr_strength` -> luma NR strength integer (0-255)
  - `:cnr_strength` -> chroma NR strength integer (0-255)

  For channel-specific controls (`:window`, `:format`, `:fps`), this reads the
  currently selected channel. Use `set_ctrl(camera, :channel, idx)` first.

  ## Examples

      {:ok, quality} = SSCMEx.Camera.get_ctrl(camera, :quality)
      {:ok, :auto} = SSCMEx.Camera.get_ctrl(camera, :ae_mode)
      {:ok, {min_us, max_us}} = SSCMEx.Camera.get_ctrl(camera, :exposure_range)
      {:ok, 800} = SSCMEx.Camera.get_ctrl(camera, :max_iso)
  """
  @spec get_ctrl(t(), ctrl_type()) :: {:ok, term()} | {:error, term()}
  def get_ctrl(%__MODULE__{resource: resource}, ctrl) do
    SSCMEx.Nif.camera_get_ctrl(resource, ctrl)
  end

  @doc """
  Get the camera ID.

  ## Examples

      {:ok, id} = SSCMEx.Camera.get_id(camera)
  """
  @spec get_id(t()) :: {:ok, non_neg_integer()} | {:error, term()}
  def get_id(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.camera_get_id(resource)
  end

  @doc """
  Check if the ISP (Image Signal Processor) is available.

  Probes the ISP by attempting to read exposure attributes on pipe 0.

  ## Examples

      {:ok, true} = SSCMEx.Camera.isp_available()
  """
  @spec isp_available() :: {:ok, boolean()} | {:error, term()}
  def isp_available do
    case SSCMEx.Nif.isp_available() do
      {:ok, true} -> {:ok, true}
      {:ok, false} -> {:ok, false}
      error -> error
    end
  end
end
