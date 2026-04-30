defmodule SSCMEx.Tick do
  @moduledoc """
  Same-clock helpers for `SSCMEx.Image.timestamp`.

  `SSCMEx.Image.timestamp` is set inside the NIF from
  `clock_gettime(CLOCK_MONOTONIC)` in nanoseconds. Erlang's
  `System.monotonic_time/1` is also CLOCK_MONOTONIC-based on Linux but
  uses a different epoch (VM-start vs system-boot), so subtracting one
  from the other gives nonsense. Use `now/0` here to read the same
  clock that stamps frames, then arithmetic on `image.timestamp` is
  meaningful.

  ## Example — measuring chip-to-Elixir frame dwell

      {:ok, image} = SSCMEx.Camera.retrieve_frame(cam, 0)
      dwell_ns = SSCMEx.Tick.now() - image.timestamp
      dwell_ms = div(dwell_ns, 1_000_000)
  """

  @doc """
  Returns the current value of `CLOCK_MONOTONIC` in nanoseconds —
  the same clock and unit that `SSCMEx.Image.timestamp` is stamped
  with at frame capture time.
  """
  @spec now() :: integer()
  def now, do: SSCMEx.Nif.tick_now()
end
