defmodule SSCMEx.Model do
  @moduledoc """
  High-level API for ML model inference using SSCMA-Micro.

  This module provides a convenient interface for creating and using
  ML models (detectors, classifiers, etc.) with the TPU.

  ## Supported Model Types

  - `:fomo` - Fast Object Mobile Object detection
  - `:yolov5` - YOLOv5 object detection
  - `:yolov8` - YOLOv8 object detection
  - `:yolo11` - YOLO11 object detection
  - `:classifier` - Image classification
  - `:yolov8_pose` - YOLOv8 pose estimation
  - `:yolo11_pose` - YOLO11 pose estimation
  - `:yolo11_seg` - YOLO11 instance segmentation

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

  @type model_type :: :fomo | :yolov5 | :yolov8 | :yolo11 | :classifier | :yolov8_pose | :yolo11_pose | :yolo11_seg | :unknown
  @type input_type :: :image | :audio | :text | :unknown
  @type output_type :: :boxes | :classes | :keypoints | :segments | :unknown
  @type config_option :: :threshold_score | :threshold_nms

  @type detection :: %{
          x: float(),
          y: float(),
          w: float(),
          h: float(),
          score: float(),
          target: integer()
        }

  @type perf :: %{
          preprocess: integer(),
          inference: integer(),
          postprocess: integer()
        }

  @doc """
  Create a new model from an engine.

  The model is automatically created based on the loaded model type.

  ## Parameters

  - `engine` - An initialized engine with a loaded model
  - `opts` - Options (optional)
    - `:algorithm_id` - Algorithm ID for model creation (default: 0)

  ## Examples

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

  For detector models, returns a list of detections:

      [%{x: 0.5, y: 0.3, w: 0.1, h: 0.2, score: 0.95, target: 0}, ...]

  Where:
  - `x`, `y` - Center coordinates (normalized 0-1)
  - `w`, `h` - Width and height (normalized 0-1)
  - `score` - Confidence score
  - `target` - Class ID

  ## Examples

      image = SSCMEx.Image.new(640, 480, :rgb888, frame_data)
      {:ok, detections} = SSCMEx.Model.run(model, image)
  """
  @spec run(t(), SSCMEx.Image.t()) :: {:ok, [detection()]} | {:error, term()}
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
