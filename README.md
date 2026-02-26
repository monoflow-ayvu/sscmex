# SSCMEx

`SSCMEx` is an Elixir NIF wrapper for SSCMA-Micro on SG2002/reCamera.

It provides:
- camera access (`SSCMEx.Camera`)
- model loading/inference (`SSCMEx.Engine`, `SSCMEx.Model`)
- zero-copy image structs (`SSCMEx.Image`)

## Installation

Add `sscmex` to your dependencies:

```elixir
def deps do
  [
    {:sscmex, "~> 0.1.0"}
  ]
end
```

## Precompiled NIF target selection

`SSCMEx` uses `elixir_make` + `cc_precompiler` for precompiled NIFs.

- In host builds, target detection follows the current host triplet.
- In Nerves cross-builds for `MIX_TARGET=nerves_system_sg2002`, `SSCMEx` maps the target
  to `riscv64-buildroot-linux-musl` automatically.
- For this known Nerves target, `SSCMEx` normalizes `TARGET_ARCH`, `TARGET_OS`, and
  `TARGET_ABI` to match published precompiled artifacts, even if those env vars are
  already set by the host toolchain.
- To preserve externally provided `TARGET_*` values for custom experiments, set
  `SSCMEX_KEEP_TARGET_TRIPLET=1`.

Example manual override:

```bash
export SSCMEX_KEEP_TARGET_TRIPLET=1
export TARGET_ARCH=riscv64
export TARGET_OS=nerves
export TARGET_ABI=musl
MIX_TARGET=nerves_system_sg2002 mix compile
```

## Manual IEx flow (camera -> TPU -> results)

Use this when you want to test manually without `SSCMEx.Examples.*`.

```elixir
# 1) Load NIF
:ok = SSCMEx.load_nif()

# 2) Device + camera
{:ok, device} = SSCMEx.Device.get_instance()
{:ok, camera} = SSCMEx.Camera.get(device, 0)

# 3) TPU engine + model
model_path = "/data/yolov8n_cv181x_int8.cvimodel"
{:ok, engine} = SSCMEx.Engine.new()
:ok = SSCMEx.Engine.load(engine, model_path)
{:ok, model} = SSCMEx.Model.create(engine)
:ok = SSCMEx.Model.set_config(model, :threshold_score, 0.5)
:ok = SSCMEx.Model.set_config(model, :threshold_nms, 0.45)

# 4) Camera config and stream
{:ok, :initialized} = SSCMEx.Camera.init(camera, 0)
{:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :channel, 0)
{:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :window, {640, 640})
{:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :format, :rgb888)
{:ok, :ok} = SSCMEx.Camera.set_ctrl(camera, :fps, 3)
{:ok, :streaming} = SSCMEx.Camera.start_stream(camera, :refresh_on_return)

Process.sleep(1500)

# 5) Grab frame and run inference
# retrieve_frame already returns %SSCMEx.Image{}
{:ok, image} = SSCMEx.Camera.retrieve_frame(camera, :rgb888)
{:ok, results} = SSCMEx.Model.run(model, image)
{:ok, perf} = SSCMEx.Model.get_perf(model)

IO.inspect(results, label: "results")
IO.inspect(perf, label: "perf")

# 6) Cleanup
{:ok, :stopped} = SSCMEx.Camera.stop_stream(camera)
{:ok, :deinitialized} = SSCMEx.Camera.deinit(camera)
```

`SSCMEx.Model.run/2` returns different result maps depending on model output type:

- `:boxes` (detection): `%{x, y, w, h, score, target}`
- `:classes` (classification): `%{score, target}`
- `:points`: `%{x, y, score, target}`
- `:keypoints` (pose): `%{box: %{x, y, w, h, score, target}, points: [%{x, y, z}, ...]}`
- `:segments` (segmentation): `%{box: %{x, y, w, h, score, target}, mask: %{width, height, data}}`

For backward compatibility, detection models still return the same bbox fields (`x`, `y`, `w`, `h`, `score`, `target`).

## Notes

- `SSCMEx.Camera.retrieve_frame/2` now returns `%SSCMEx.Image{}` directly.
- For one-shot tests, keep `fps` low (for example `3`) to reduce memory pressure.

## Runtime observability (SG2002)

When debugging camera/TPU contention, these runtime stats are useful:

- VB/ION pool status (best signal for camera-side memory pressure):
  - `cat /proc/cvitek/vb`
  - Look at per-pool `BlkSz`, `BlkCnt`, `Free`, and especially `MinFree`.
  - If `MinFree` drops near `0`, the pipeline is close to buffer exhaustion.

- TPU usage profiling:
  - Enable: `echo 1 > /proc/tpu/usage_profiling`
  - Read: `cat /proc/tpu/usage_profiling`
  - Disable: `echo 0 > /proc/tpu/usage_profiling`

- Per-inference model timings (already exposed by SSCMEx):
  - `{:ok, perf} = SSCMEx.Model.get_perf(model)`
  - Returns `%{preprocess: ms, inference: ms, postprocess: ms}` from the last run.

- Optional bandwidth monitor (if your image enables it):
  - `echo 1 > /proc/mon/bw_profiling`
  - `cat /proc/mon/profiling_window_ms`

