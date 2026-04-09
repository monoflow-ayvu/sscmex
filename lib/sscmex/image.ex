defmodule SSCMEx.Image do
  @moduledoc """
  Image structure used by camera capture and model inference.

  This is a zero-copy wrapper over an Elixir binary. The `data` field holds a
  reference to the original binary.

  ## Supported Formats

  Raw (fixed bytes-per-pixel):
  - `:rgb888`
  - `:rgb888_planar`
  - `:rgb565`
  - `:yuv422`
  - `:gray`

  Encoded (variable size):
  - `:jpeg`
  - `:h264`
  - `:h265`

  ## Example

      # Create image from camera frame (zero-copy)
      frame = capture_camera_frame()  # Returns binary
      image = SSCMEx.Image.new(640, 480, :rgb888, frame)

      # Run inference
      {:ok, results} = SSCMEx.Model.run(model, image)
  """

  @typedoc """
  Supported pixel formats for convert operations.
  Note: YUV422 can only be used as input (YUV422→RGB/Grayscale),
  not as output (RGB→YUV422 is not supported by OpenCV 3.2.0).
  """
  @type convert_format :: :rgb888 | :rgb565 | :yuv422 | :gray | :jpeg | :webp

  @typedoc """
  Interpolation methods for resize operations.
  """
  @type interpolation :: :nearest | :bilinear | :bicubic | :area | :lanczos4

  @type raw_format :: :rgb888 | :rgb888_planar | :rgb565 | :yuv422 | :gray
  @type encoded_format :: :jpeg | :h264 | :h265
  @type format :: raw_format() | encoded_format()

  @type t :: %__MODULE__{
          width: non_neg_integer(),
          height: non_neg_integer(),
          format: format(),
          data: binary(),
          size: non_neg_integer() | nil,
          timestamp: integer() | nil,
          key: boolean() | nil
        }

  defstruct [:width, :height, :format, :data, :size, :timestamp, :key]

  @doc """
  Create a new image struct.

  This is zero-copy - `data` binary is referenced, not copied.

  ## Parameters

  - `width` - Image width in pixels
  - `height` - Image height in pixels
  - `format` - Pixel format
  - `data` - Binary image data (not copied)

  ## Example

      image = SSCMEx.Image.new(640, 480, :rgb888, frame_data)
  """
  @spec new(non_neg_integer(), non_neg_integer(), format(), binary()) :: t()
  def new(width, height, format, data) do
    %__MODULE__{
      width: width,
      height: height,
      format: format,
      data: data,
      size: byte_size(data),
      timestamp: nil,
      key: nil
    }
  end

  @doc """
  Convert image to a different format.

  ## Supported conversions

  Raw formats (RGB888, RGB565, YUV422, Grayscale):
  - RGB888 ↔ Grayscale
  - RGB888 ↔ RGB565
  - YUV422 → RGB888/Grayscale (note: reverse not supported)

  Compressed formats:
  - Any raw format → JPEG
  - Any raw format → WebP (requires libwebp support in SDK)

  ## Options

    * `:quality` - JPEG/WebP quality 0-100 (default: 85)

  ## Examples

      {:ok, webp} = SSCMEx.Image.convert(image, :webp, quality: 90)
      {:ok, gray} = SSCMEx.Image.convert(image, :gray)
      {:ok, rgb} = SSCMEx.Image.convert(yuv_image, :rgb888)
  """
  @spec convert(t(), convert_format(), keyword()) :: {:ok, t()} | {:error, atom()}
  def convert(%__MODULE__{} = image, format, opts \\ []) do
    quality = Keyword.get(opts, :quality, 85)
    SSCMEx.Nif.image_convert(image, format, quality)
  end

  @doc """
  Resize image to new dimensions.

  ## Options

    * `:interpolation` - `:nearest`, `:bilinear` (default), `:bicubic`, `:area`, `:lanczos4`

  ## Examples

      {:ok, small} = SSCMEx.Image.resize(image, {320, 240})
      {:ok, thumb} = SSCMEx.Image.resize(image, {160, 120}, interpolation: :lanczos4)
  """
  @spec resize(t(), {pos_integer(), pos_integer()}, keyword()) ::
          {:ok, t()} | {:error, atom()}
  def resize(%__MODULE__{} = image, {w, h}, opts \\ [])
      when is_integer(w) and is_integer(h) and w > 0 and h > 0 do
    interp = opts |> Keyword.get(:interpolation, :linear) |> map_interpolation()
    SSCMEx.Nif.image_resize(image, {w, h}, interp)
  end

  @doc """
  Returns the expected byte size for the image data based on dimensions and format.

  ## Example

      iex> SSCMEx.Image.data_size(%SSCMEx.Image{width: 640, height: 480, format: :rgb888, data: <<>>})
      921600
  """
  @spec data_size(t()) :: non_neg_integer()
  def data_size(%__MODULE__{width: w, height: h, format: format, data: data}) do
    case bytes_per_pixel(format) do
      {:ok, bpp} -> bpp * w * h
      :encoded -> byte_size(data)
      :unsupported -> 0
    end
  end

  @doc """
  Validates that the image data size matches the expected size for the given dimensions and format.

  ## Example

      iex> image = SSCMEx.Image.new(2, 2, :rgb888, <<255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255>>)
      iex> SSCMEx.Image.valid?(image)
      true
  """
  @spec valid?(t()) :: boolean()
  def valid?(%__MODULE__{width: w, height: h, format: format, data: data}) do
    case bytes_per_pixel(format) do
      {:ok, bpp} -> byte_size(data) == bpp * w * h
      :encoded -> byte_size(data) > 0
      :unsupported -> false
    end
  end

  # Maps Elixir interpolation atoms to the atoms the C NIF recognises.
  defp map_interpolation(:bilinear), do: :linear
  defp map_interpolation(:bicubic), do: :cubic
  defp map_interpolation(other), do: other

  # Returns bytes per pixel for raw formats, :encoded for compressed formats.
  @spec bytes_per_pixel(format()) :: {:ok, non_neg_integer()} | :encoded | :unsupported
  defp bytes_per_pixel(:rgb888), do: {:ok, 3}
  defp bytes_per_pixel(:rgb888_planar), do: {:ok, 3}
  defp bytes_per_pixel(:rgb565), do: {:ok, 2}
  defp bytes_per_pixel(:yuv422), do: {:ok, 2}
  defp bytes_per_pixel(:gray), do: {:ok, 1}
  defp bytes_per_pixel(:jpeg), do: :encoded
  defp bytes_per_pixel(:h264), do: :encoded
  defp bytes_per_pixel(:h265), do: :encoded
  defp bytes_per_pixel(_), do: :unsupported
end
