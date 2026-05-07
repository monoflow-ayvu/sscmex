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
    {:sscmex, "~> 0.4.0"}
  ]
end
```

`SSCMEx` ships precompiled NIFs for `riscv64-linux-musl` (Nerves target
`nerves_system_sg2002`) on every tagged GitHub release. Hex consumers
get the precompiled artefact automatically; nothing else needs to be on
the build host.

### Building from source

Source builds require the SG2002 SDK, because SSCMA-Micro's CVI engine
header `#include <cviruntime.h>` only ships with the CVITEK TPU SDK.

If `mix deps.compile sscmex` ever falls back to a source build (typical
trigger: pinning to a git branch instead of a Hex version), you'll get a
clear `FATAL_ERROR` from cmake telling you to either:

- run inside the `sscmex` devenv (provides the SDK automatically), or
- set `SG200X_SDK_PATH` to an extracted [reCameraOS SDK
  release](https://github.com/Seeed-Studio/reCamera-OS/releases), or
- pin to a tagged Hex/Git release that has matching precompiled NIFs in
  its GitHub release assets, so `cc_precompiler` can skip the build.

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
# (second arg is the channel index 0..2, not a format — format was set
#  by set_ctrl(:format, :rgb888) above.)
{:ok, image} = SSCMEx.Camera.retrieve_frame(camera, 0)
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

## YOLOv7 cvimodel workflow

YOLOv7 is supported by an in-tree native decoder (`c_src/sscma_yolov7.cpp`)
because upstream SSCMA-Micro doesn't ship one. The runtime side is
automatic — `SSCMEx.Model.create/1` will return a `:yolov7` model whenever
the loaded cvimodel exposes the three raw Conv heads at `[1, 3, H, W,
5+nc]`. Building that cvimodel is a 3-step offline pipeline:

1. **Re-export the .pt to a clean ONNX.** YOLOv7's default `--grid` ONNX
   embeds anchor decoding via `ScatterND`, which TPU-MLIR can't compile
   well for cv181x. The `scripts/yolov7_pt_to_clean_onnx.py` script
   re-exports the model with `Detect.export = True`, so the graph stops
   at the three permuted Conv outputs (raw heads, pre-sigmoid):

   ```bash
   git clone --depth 1 https://github.com/WongKinYiu/yolov7.git /tmp/yolov7
   pip install torch onnx
   python scripts/yolov7_pt_to_clean_onnx.py \
       --pt   safety.pt \
       --out  safety_clean.onnx \
       --yolov7-repo /tmp/yolov7 \
       --classes-json safety_map.json
   ```

   The result is a self-contained ONNX (weights inlined) at IR v8 / opset
   17 — exactly what `tpu_mlir==1.7` accepts, so no `downgrade_onnx.py`
   step is needed. Outputs are named `head_p3`, `head_p4`, `head_p5`
   (strides 8, 16, 32).

2. **Run TPU-MLIR inside the official Docker container.** The Sophgo
   `tpuc_dev:v3.1` image is required (newer glibc than most hosts; the
   pip install bundle's `libc.so.6` only works in that image). Drop a
   directory of representative calibration images alongside the ONNX
   (the [PPE-YOLOv8 dataset on
   Kaggle](https://www.kaggle.com/datasets/shlokraval/ppe-dataset-yolov8)
   is a good source for the safety model — pick ~100 images covering
   lighting and pose variety), then:

   ```bash
   docker run --privileged --rm -it \
       -v "$PWD":/workspace -w /workspace \
       sophgo/tpuc_dev:v3.1 \
       bash -lc 'pip install tpu_mlir[all]==1.7 && \
           bash scripts/build_yolov7_cvimodel.sh \
               --onnx safety_clean.onnx \
               --calib-dir ./calib_imgs \
               --name safety'
   ```

   The script wraps `model_transform.py` → `run_calibration.py` →
   `model_deploy.py` with the Seeed-recommended INT8 flags (`--quant_input
   --customization_format RGB_PACKED --fuse_preprocess --aligned_input
   --processor cv181x`). It produces `safety_int8.cvimodel`.

3. **Flash and use.** Copy `safety_int8.cvimodel` to `/data` on the
   reCamera, then in IEx the standard `SSCMEx.Model` API works
   unmodified:

   ```elixir
   {:ok, engine} = SSCMEx.Engine.new()
   :ok = SSCMEx.Engine.load(engine, "/data/safety_int8.cvimodel")
   {:ok, model} = SSCMEx.Model.create(engine)
   {:ok, :yolov7} = SSCMEx.Model.get_type(model)

   :ok = SSCMEx.Model.set_config(model, :threshold_score, 0.45)
   :ok = SSCMEx.Model.set_config(model, :threshold_nms,   0.45)
   {:ok, detections} = SSCMEx.Model.run(model, image)
   ```

   Detections come back in the same `%{x, y, w, h, score, target}` shape
   as `:yolov5`, so any downstream code that handles YOLOv5 detections
   handles these. The class index `target` follows the order in the
   `.pt`'s `model.names` (which may differ from `safety_map.json` —
   verify with the metadata file produced in step 1).

If you only have the ONNX (no `.pt`), `scripts/yolov7_to_clean_onnx.py`
performs the equivalent surgery on an existing ONNX. It needs the original
`.onnx.data` external-weights file alongside; the `.pt` route is simpler.

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

