defmodule SSCMEx.Device do
  @moduledoc """
  Device singleton wrapper for SG2002 chip.

  The Device provides access to system information, sensors (including cameras),
  and model metadata. This is a singleton - only one device instance exists.

  ## Example

      {:ok, device} = SSCMEx.Device.get_instance()
      {:ok, info} = SSCMEx.Device.get_info(device)
      IO.puts("Device: \#{info.name}")

      {:ok, sensors} = SSCMEx.Device.get_sensors(device)
      for sensor <- sensors do
        IO.puts("Sensor: \#{sensor.id} (\#{sensor.type})")
      end
  """

  @type resource :: reference()
  @type t :: %__MODULE__{resource: resource()}

  defstruct [:resource]

  @type sensor :: %{
          id: non_neg_integer(),
          type: :camera | :microphone | :imu | :unknown
        }

  @type model_info :: %{
          id: non_neg_integer(),
          name: String.t(),
          type: atom(),
          size: non_neg_integer(),
          addr: non_neg_integer()
        }

  @type device_info :: %{
          name: String.t(),
          id: String.t(),
          version: String.t(),
          boot_count: non_neg_integer()
        }

  @doc """
  Get the Device singleton instance.

  This returns the same device instance on every call.

  ## Examples

      {:ok, device} = SSCMEx.Device.get_instance()
  """
  @spec get_instance() :: {:ok, t()} | {:error, term()}
  def get_instance do
    case SSCMEx.Nif.device_get_instance() do
      {:ok, resource} -> {:ok, %__MODULE__{resource: resource}}
      error -> error
    end
  end

  @doc """
  Get device information (name, id, version, boot count).

  Returns a map with:
  - `:name` - Device name
  - `:id` - Device unique identifier
  - `:version` - Firmware/software version
  - `:boot_count` - Number of times the device has booted

  ## Examples

      {:ok, info} = SSCMEx.Device.get_info(device)
      # => %{name: "reCamera", id: "abc123", version: "1.0.0", boot_count: 42}
  """
  @spec get_info(t()) :: {:ok, device_info()} | {:error, term()}
  def get_info(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.device_get_info(resource)
  end

  @doc """
  Get list of available sensors.

  Returns a list of sensor maps, each with:
  - `:id` - Sensor ID
  - `:type` - Sensor type (`:camera`, `:microphone`, `:imu`, or `:unknown`)

  ## Examples

      {:ok, sensors} = SSCMEx.Device.get_sensors(device)
      # => [%{id: 0, type: :camera}, ...]

      # Find camera sensor
      camera = Enum.find(sensors, &(&1.type == :camera))
  """
  @spec get_sensors(t()) :: {:ok, [sensor()]} | {:error, term()}
  def get_sensors(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.device_get_sensors(resource)
  end

  @doc """
  Get list of available model metadata.

  This returns information about models stored on the device, not
  currently loaded models. Use `SSCMEx.Engine` to load and run models.

  Returns a list of model info maps, each with:
  - `:id` - Model ID
  - `:name` - Model name
  - `:type` - Model type atom
  - `:size` - Model size in bytes
  - `:addr` - Memory address (for embedded models)

  ## Examples

      {:ok, models} = SSCMEx.Device.get_models(device)
      for model <- models do
        IO.puts("Model: \#{model.name} (id=\#{model.id})")
      end
  """
  @spec get_models(t()) :: {:ok, [model_info()]} | {:error, term()}
  def get_models(%__MODULE__{resource: resource}) do
    SSCMEx.Nif.device_get_models(resource)
  end
end
