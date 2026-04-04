defmodule SSCMEx.ImageTest do
  @moduledoc """
  Tests for SSCMEx.Image module.

  Note: These tests verify the Elixir API layer. The actual NIF functions
  require the compiled shared library and SG2002 hardware to work properly.
  Tests that call NIF functions will fail with :nif_not_loaded error in the
  test environment.
  """

  use ExUnit.Case
  doctest SSCMEx.Image

  describe "new/4" do
    test "creates image struct with correct fields" do
      data = <<255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255>>
      image = SSCMEx.Image.new(2, 2, :rgb888, data)

      assert image.width == 2
      assert image.height == 2
      assert image.format == :rgb888
      assert image.data == data
      assert image.size == 12
    end

    test "creates image with encoded format" do
      jpeg_data = <<0xFF, 0xD8, 0xFF, 0xE0>>
      image = SSCMEx.Image.new(640, 480, :jpeg, jpeg_data)

      assert image.width == 640
      assert image.height == 480
      assert image.format == :jpeg
      assert image.data == jpeg_data
    end
  end

  describe "data_size/1" do
    test "calculates size for raw formats" do
      image = %SSCMEx.Image{width: 640, height: 480, format: :rgb888, data: <<>>}
      assert SSCMEx.Image.data_size(image) == 921_600

      image = %SSCMEx.Image{width: 640, height: 480, format: :gray, data: <<>>}
      assert SSCMEx.Image.data_size(image) == 307_200

      image = %SSCMEx.Image{width: 640, height: 480, format: :rgb565, data: <<>>}
      assert SSCMEx.Image.data_size(image) == 614_400

      image = %SSCMEx.Image{width: 640, height: 480, format: :yuv422, data: <<>>}
      assert SSCMEx.Image.data_size(image) == 614_400
    end

    test "returns data size for encoded formats" do
      jpeg_data = :binary.copy(<<0xFF>>, 1000)
      image = %SSCMEx.Image{width: 640, height: 480, format: :jpeg, data: jpeg_data}
      assert SSCMEx.Image.data_size(image) == 1000
    end
  end

  describe "valid?/1" do
    test "returns true for valid raw image" do
      data = :binary.copy(<<0>>, 640 * 480 * 3)
      image = SSCMEx.Image.new(640, 480, :rgb888, data)
      assert SSCMEx.Image.valid?(image)
    end

    test "returns false for incorrect data size" do
      data = :binary.copy(<<0>>, 100)
      image = SSCMEx.Image.new(640, 480, :rgb888, data)
      refute SSCMEx.Image.valid?(image)
    end

    test "returns true for valid encoded image" do
      jpeg_data = <<0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10>>
      image = SSCMEx.Image.new(640, 480, :jpeg, jpeg_data)
      assert SSCMEx.Image.valid?(image)
    end

    test "returns false for empty encoded image" do
      image = SSCMEx.Image.new(640, 480, :jpeg, <<>>)
      refute SSCMEx.Image.valid?(image)
    end
  end

  describe "convert/3" do
    @tag :skip
    test "converts RGB888 to grayscale" do
      rgb_data = :binary.copy(<<255, 0, 0>>, 640 * 480)
      image = SSCMEx.Image.new(640, 480, :rgb888, rgb_data)

      assert {:ok, gray_image} = SSCMEx.Image.convert(image, :gray)
      assert gray_image.format == :gray
      assert gray_image.width == 640
      assert gray_image.height == 480
      assert byte_size(gray_image.data) == 640 * 480
    end

    @tag :skip
    test "converts RGB888 to RGB565" do
      rgb_data = :binary.copy(<<255, 0, 0>>, 640 * 480)
      image = SSCMEx.Image.new(640, 480, :rgb888, rgb_data)

      assert {:ok, rgb565_image} = SSCMEx.Image.convert(image, :rgb565)
      assert rgb565_image.format == :rgb565
      assert byte_size(rgb565_image.data) == 640 * 480 * 2
    end

    @tag :skip
    test "converts RGB888 to JPEG with quality option" do
      rgb_data = :binary.copy(<<255, 128, 64>>, 640 * 480)
      image = SSCMEx.Image.new(640, 480, :rgb888, rgb_data)

      assert {:ok, jpeg_image} = SSCMEx.Image.convert(image, :jpeg, quality: 85)
      assert jpeg_image.format == :jpeg
      assert byte_size(jpeg_image.data) < 640 * 480 * 3
    end

    @tag :skip
    test "converts RGB888 to WebP with quality option" do
      rgb_data = :binary.copy(<<255, 128, 64>>, 640 * 480)
      image = SSCMEx.Image.new(640, 480, :rgb888, rgb_data)

      assert {:ok, webp_image} = SSCMEx.Image.convert(image, :webp, quality: 90)
      assert webp_image.format == :webp
      assert byte_size(webp_image.data) < 640 * 480 * 3
    end

    @tag :skip
    test "converts JPEG to WebP (optimized workflow)" do
      # Create a minimal valid JPEG (this is a 2x2 gray JPEG)
      # In real usage, this would be actual JPEG data from camera/file
      jpeg_data =
        <<0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00,
          0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06,
          0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D,
          0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D,
          0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28,
          0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
          0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x02, 0x00, 0x02,
          0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
          0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
          0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10,
          0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
          0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
          0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42,
          0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
          0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37,
          0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
          0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73,
          0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
          0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
          0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA,
          0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
          0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
          0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08,
          0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0x34, 0xA0, 0x28, 0xA0,
          0x0A, 0x28, 0xA2, 0x80, 0xFF, 0xD9>>

      image = SSCMEx.Image.new(2, 2, :jpeg, jpeg_data)

      assert {:ok, webp_image} = SSCMEx.Image.convert(image, :webp, quality: 85)
      assert webp_image.format == :webp
      assert byte_size(webp_image.data) > 0
    end

    @tag :skip
    test "converts YUV422 to RGB888" do
      yuv_data = :binary.copy(<<128, 64>>, 640 * 480)
      image = SSCMEx.Image.new(640, 480, :yuv422, yuv_data)

      assert {:ok, rgb_image} = SSCMEx.Image.convert(image, :rgb888)
      assert rgb_image.format == :rgb888
      assert byte_size(rgb_image.data) == 640 * 480 * 3
    end

    @tag :skip
    test "returns error for unsupported conversion (RGB888 to YUV422)" do
      rgb_data = :binary.copy(<<255, 0, 0>>, 640 * 480)
      image = SSCMEx.Image.new(640, 480, :rgb888, rgb_data)

      assert {:error, :unsupported_conversion} = SSCMEx.Image.convert(image, :yuv422)
    end

    @tag :skip
    test "returns error for invalid image" do
      image = %SSCMEx.Image{width: 640, height: 480, format: :rgb888, data: <<0, 0>>}

      assert {:error, _} = SSCMEx.Image.convert(image, :gray)
    end
  end

  describe "resize/3" do
    @tag :skip
    test "resizes image with default bilinear interpolation" do
      rgb_data = :binary.copy(<<255, 128, 64>>, 640 * 480)
      image = SSCMEx.Image.new(640, 480, :rgb888, rgb_data)

      assert {:ok, resized} = SSCMEx.Image.resize(image, {320, 240})
      assert resized.width == 320
      assert resized.height == 240
      assert resized.format == :rgb888
      assert byte_size(resized.data) == 320 * 240 * 3
    end

    @tag :skip
    test "resizes with nearest neighbor interpolation" do
      rgb_data = :binary.copy(<<255, 128, 64>>, 640 * 480)
      image = SSCMEx.Image.new(640, 480, :rgb888, rgb_data)

      assert {:ok, resized} = SSCMEx.Image.resize(image, {160, 120}, interpolation: :nearest)
      assert resized.width == 160
      assert resized.height == 120
    end

    @tag :skip
    test "resizes with lanczos4 interpolation" do
      rgb_data = :binary.copy(<<255, 128, 64>>, 640 * 480)
      image = SSCMEx.Image.new(640, 480, :rgb888, rgb_data)

      assert {:ok, resized} = SSCMEx.Image.resize(image, {320, 240}, interpolation: :lanczos4)
      assert resized.width == 320
      assert resized.height == 240
    end

    @tag :skip
    test "returns error for invalid dimensions" do
      rgb_data = :binary.copy(<<255, 128, 64>>, 640 * 480)
      image = SSCMEx.Image.new(640, 480, :rgb888, rgb_data)

      assert {:error, :invalid_width} = SSCMEx.Image.resize(image, {0, 240})
      assert {:error, :invalid_height} = SSCMEx.Image.resize(image, {320, -1})
    end

    @tag :skip
    test "returns error for compressed input" do
      jpeg_data = <<0xFF, 0xD8, 0xFF, 0xE0>>
      image = SSCMEx.Image.new(640, 480, :jpeg, jpeg_data)

      assert {:error, :unsupported_format_for_resize} = SSCMEx.Image.resize(image, {320, 240})
    end
  end

  describe "bytes_per_pixel/1 (private)" do
    test "correctly calculates bytes per pixel for all formats" do
      rgb888_image = %SSCMEx.Image{width: 10, height: 10, format: :rgb888, data: <<>>}
      assert SSCMEx.Image.data_size(rgb888_image) == 10 * 10 * 3

      rgb565_image = %SSCMEx.Image{width: 10, height: 10, format: :rgb565, data: <<>>}
      assert SSCMEx.Image.data_size(rgb565_image) == 10 * 10 * 2

      gray_image = %SSCMEx.Image{width: 10, height: 10, format: :gray, data: <<>>}
      assert SSCMEx.Image.data_size(gray_image) == 10 * 10 * 1

      yuv422_image = %SSCMEx.Image{width: 10, height: 10, format: :yuv422, data: <<>>}
      assert SSCMEx.Image.data_size(yuv422_image) == 10 * 10 * 2
    end
  end
end
