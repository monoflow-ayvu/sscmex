defmodule SSCMEx.Model do
  @moduledoc """
  High-level API for ML model inference using SSCMA-Micro.

  This module provides a convenient interface for creating and using
  ML models (detectors, classifiers, etc.) with the TPU.

  ## Supported Model Types

  - `:fomo` - Fast Object Mobile Object detection
  - `:yolov5` - YOLOv5 object detection
  - `:yolov7` - YOLOv7 object detection (anchor-based, decoded on-device)
  - `:yolov8` - YOLOv8 object detection
  - `:yolo11` - YOLO11 object detection
  - `:classifier` - Image classification
  - `:yolov8_pose` - YOLOv8 pose estimation
  - `:yolo11_pose` - YOLO11 pose estimation
  - `:yolo11_seg` - YOLO11 instance segmentation
  - `:yolo26` - YOLO26 object detection

  Note: With the currently pinned SSCMA-Micro commit, YOLO26 is detection-only.

  YOLOv7 cvimodels must be produced from a *cleaned* ONNX whose embedded
  post-processing has been stripped (see `scripts/yolov7_to_clean_onnx.py`
  and `scripts/build_yolov7_cvimodel.sh`). The on-device decoder lives in
  `c_src/sscma_yolov7.cpp` and is selected automatically when the engine's
  outputs match the 3-raw-head layout `[1, 3, H, W, 5+nc]` at strides
  8/16/32.

  ## Example

      # Create engine and load model
      {:ok, engine} = SSCMEx.Engine.new()
      :ok = SSCMEx.Engine.load(engine, "/data/yolo11n.cvimodel")

      # Create detector model
      {:ok, model} = SSCMEx.Model.create(engine)

      # Configure thresholds
      :ok = SSCMEx.Model.set_config(model, :threshold_score, 0.5)

      # Prepare image (zero-copy)
      image = SSCMEx.Image.new(640, 480, :rgb888, frame_data)

      # Run inference
      {:ok, detections} = SSCMEx.Model.run(model, image)

      # Process results
      for det <- detections do
        IO.puts("Class \#{det.target}: \#{det.score}")
      end
  """

  @type resource :: reference()
  @type t :: %__MODULE__{resource: resource()}

  defstruct [:resource]

  @type model_type ::
          :fomo
          | :yolov5
          | :yolov7
          | :yolov8
          | :yolo11
          | :classifier
          | :yolov8_pose
          | :yolo11_pose
          | :yolo11_seg
          | :yolo26
          | :unknown
  @type input_type :: :image | :audio | :text | :unknown
  @type output_type :: :tensor | :boxes | :classes | :points | :keypoints | :segments | :unknown
  @type config_option :: :threshold_score | :threshold_nms

  @type detection :: %{
          x: float(),
          y: float(),
          w: float(),
          h: float(),
          score: float(),
          target: integer()
        }

  @type classification :: %{
          score: float(),
          target: integer()
        }

  @type point :: %{
          x: float(),
          y: float(),
          score: float(),
          target: integer()
        }

  @type keypoint :: %{
          x: float(),
          y: float(),
          z: float()
        }

  @type keypoint_result :: %{
          box: detection(),
          points: [keypoint()]
        }

  @type segment_mask :: %{
          width: non_neg_integer(),
          height: non_neg_integer(),
          data: binary()
        }

  @type segment_result :: %{
          box: detection(),
          mask: segment_mask()
        }

  @type inference_result ::
          detection() | classification() | point() | keypoint_result() | segment_result()

  @type perf :: %{
          preprocess: integer(),
          inference: integer(),
          postprocess: integer()
        }

  @doc """
  Create a new model from an engine.

  The model is automatically created based on the loaded model type.
  You must load a model with `SSCMEx.Engine.load/2` before calling this;
  otherwise you get `{:error, ~c"no_model_loaded"}`.

  ## Parameters

  - `engine` - An initialized engine with a loaded model
  - `opts` - Options (optional)
    - `:algorithm_id` - Algorithm ID for model creation (default: 0)

  ## Examples

      {:ok, engine} = SSCMEx.Engine.new()
      :ok = SSCMEx.Engine.load(engine, "/path/to/model.cvimodel")
      {:ok, model} = SSCMEx.Model.create(engine)
      {:ok, model} = SSCMEx.Model.create(engine, algorithm_id: 0)
  """
  @spec create(SSCMEx.Engine.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def create(%SSCMEx.Engine{resource: engine_resource}, opts \\ []) do
    algorithm_id = Keyword.get(opts, :algorithm_id, 0)

    case SSCMEx.Nif.model_create(engine_resource, algorithm_id) do
      {:ok, resource} -> {:ok, %__MODULE__{resource: resource}}
      error -> error
    end
  end

  @doc """
  Get the model type.

  ## Examples

      {:ok, type} = SSCMEx.Model.get_type(model)
      # => :yolo11
  """
  @spec get_type(t()) :: {:ok, model_type()} | {:error, term()}
  def get_type(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.model_get_type(resource)
  end

  @doc """
  Get the model name.

  ## Examples

      {:ok, name} = SSCMEx.Model.get_name(model)
      # => "yolo11n"
  """
  @spec get_name(t()) :: {:ok, String.t()} | {:error, term()}
  def get_name(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.model_get_name(resource)
  end

  @doc """
  Get the model input type.

  ## Examples

      {:ok, input_type} = SSCMEx.Model.get_input_type(model)
      # => :image
  """
  @spec get_input_type(t()) :: {:ok, input_type()} | {:error, term()}
  def get_input_type(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.model_get_input_type(resource)
  end

  @doc """
  Get the model output type.

  ## Examples

      {:ok, output_type} = SSCMEx.Model.get_output_type(model)
      # => :boxes
  """
  @spec get_output_type(t()) :: {:ok, output_type()} | {:error, term()}
  def get_output_type(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.model_get_output_type(resource)
  end

  @doc """
  Run inference on an image.

  This is a zero-copy operation - the image data is not copied.

  ## Parameters

  - `model` - The model to run inference on
  - `image` - An `%SSCMEx.Image{}` struct with the image data

  ## Returns

  Returns a list of decoded results based on the model output type:

  - `:boxes` -> `[%{x, y, w, h, score, target}, ...]`
  - `:classes` -> `[%{score, target}, ...]`
  - `:points` -> `[%{x, y, score, target}, ...]`
  - `:keypoints` -> `[%{box: %{...}, points: [%{x, y, z}, ...]}, ...]`
  - `:segments` -> `[%{box: %{...}, mask: %{width, height, data}}, ...]`

  You can inspect the output type with `get_output_type/1`.

  ## Examples

      image = SSCMEx.Image.new(640, 480, :rgb888, frame_data)
      {:ok, results} = SSCMEx.Model.run(model, image)
  """
  @spec run(t(), SSCMEx.Image.t()) :: {:ok, [inference_result()]} | {:error, term()}
  def run(%__MODULE__{resource: resource}, %SSCMEx.Image{} = image) do
    SSCMEx.Nif.model_run(resource, image)
  end

  @doc """
  Set a model configuration option.

  ## Options

  - `:threshold_score` - Minimum confidence score threshold (0.0 - 1.0)
  - `:threshold_nms` - Non-maximum suppression threshold (0.0 - 1.0)

  ## Examples

      :ok = SSCMEx.Model.set_config(model, :threshold_score, 0.5)
      :ok = SSCMEx.Model.set_config(model, :threshold_nms, 0.45)
  """
  @spec set_config(t(), config_option(), float()) :: :ok | {:error, term()}
  def set_config(%__MODULE__{resource: resource}, option, value) when is_float(value) do
    case SSCMEx.Nif.model_set_config(resource, option, value) do
      {:ok, _} -> :ok
      error -> error
    end
  end

  @doc """
  Get a model configuration option.

  ## Options

  - `:threshold_score` - Minimum confidence score threshold
  - `:threshold_nms` - Non-maximum suppression threshold

  ## Examples

      {:ok, threshold} = SSCMEx.Model.get_config(model, :threshold_score)
  """
  @spec get_config(t(), config_option()) :: {:ok, float()} | {:error, term()}
  def get_config(%__MODULE__{resource: resource}, option) do
    SSCMEx.Nif.model_get_config(resource, option)
  end

  @doc """
  Get performance metrics from the last inference.

  Returns a map with timing information in milliseconds:

      %{preprocess: 10, inference: 50, postprocess: 5}

  ## Examples

      {:ok, perf} = SSCMEx.Model.get_perf(model)
      IO.puts("Inference took \#{perf.inference}ms")
  """
  @spec get_perf(t()) :: {:ok, perf()} | {:error, term()}
  def get_perf(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.model_get_perf(resource)
  end
end
