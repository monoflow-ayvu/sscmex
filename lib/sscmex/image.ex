defmodule SSCMEx.Image do
  @moduledoc """
  Image structure for model inference.

  This is a zero-copy implementation - the `data` field holds a reference
  to the existing binary, not a copy. Elixir binaries are reference-counted
  and immutable, so no data is copied when creating or passing images.

  ## Supported Formats

  - `:rgb888` - 24-bit RGB (3 bytes per pixel)
  - `:rgb565` - 16-bit RGB (2 bytes per pixel)
  - `:yuv422` - YUV 4:2:2 (2 bytes per pixel)
  - `:gray` - 8-bit grayscale (1 byte per pixel)

  ## Example

      # Create image from camera frame (zero-copy)
      frame = capture_camera_frame()  # Returns binary
      image = SSCMEx.Image.new(640, 480, :rgb888, frame)

      # Run inference
      {:ok, results} = SSCMEx.Model.run(model, image)
  """

  @type format :: :rgb888 | :rgb565 | :yuv422 | :gray

  @type t :: %__MODULE__{
          width: non_neg_integer(),
          height: non_neg_integer(),
          format: format(),
          data: binary()
        }

  defstruct [:width, :height, :format, :data]

  @doc """
  Create a new image struct.

  This is zero-copy - `data` binary is referenced, not copied.

  ## Parameters

  - `width` - Image width in pixels
  - `height` - Image height in pixels
  - `format` - Pixel format (`:rgb888`, `:rgb565`, `:yuv422`, or `:gray`)
  - `data` - Binary image data (not copied)

  ## Example

      image = SSCMEx.Image.new(640, 480, :rgb888, frame_data)
  """
  @spec new(non_neg_integer(), non_neg_integer(), format(), binary()) :: t()
  def new(width, height, format, data) do
    %__MODULE__{width: width, height: height, format: format, data: data}
  end

  @doc """
  Returns the expected byte size for the image data based on dimensions and format.

  ## Example

      iex> SSCMEx.Image.data_size(%SSCMEx.Image{width: 640, height: 480, format: :rgb888, data: <<>>})
      921600
  """
  @spec data_size(t()) :: non_neg_integer()
  def data_size(%__MODULE__{width: w, height: h, format: format}) do
    bytes_per_pixel(format) * w * h
  end

  @doc """
  Validates that the image data size matches the expected size for the given dimensions and format.

  ## Example

      iex> image = SSCMEx.Image.new(2, 2, :rgb888, <<255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255>>)
      iex> SSCMEx.Image.valid?(image)
      true
  """
  @spec valid?(t()) :: boolean()
  def valid?(%__MODULE__{data: data} = image) do
    Kernel.byte_size(data) == data_size(image)
  end

  # Returns bytes per pixel for each format
  @spec bytes_per_pixel(format()) :: non_neg_integer()
  defp bytes_per_pixel(:rgb888), do: 3
  defp bytes_per_pixel(:rgb565), do: 2
  defp bytes_per_pixel(:yuv422), do: 2
  defp bytes_per_pixel(:gray), do: 1
end
