#!/usr/bin/env bash
#
# build_yolov7_cvimodel.sh — Convert a YOLOv7 cleaned ONNX to a cv181x
# cvimodel using the SOPHGO TPU-MLIR toolchain.
#
# This mirrors the official reCamera flow documented at
# https://wiki.seeedstudio.com/recamera_model_conversion (step-by-step
# guide for YOLO11n; the parameters below are the YOLO5/v7 variant).
#
# Run inside the official docker container (libc/libm requirements):
#
#   git clone --depth 1 https://github.com/WongKinYiu/yolov7.git /tmp/yolov7
#   python scripts/yolov7_pt_to_clean_onnx.py \
#       --pt model.pt --out model_clean.onnx \
#       --yolov7-repo /tmp/yolov7
#
#   docker run --privileged --rm -it \
#       -v "$PWD":/workspace -w /workspace \
#       sophgo/tpuc_dev:v3.1 \
#       bash -lc 'pip install tpu_mlir[all]==1.7 && \
#           bash scripts/build_yolov7_cvimodel.sh \
#               --onnx model_clean.onnx \
#               --calib-dir ./calib_imgs \
#               --name model'
#
# Inputs
# ------
#   --onnx       Cleaned ONNX (from scripts/yolov7_pt_to_clean_onnx.py)
#   --calib-dir  Directory of ~100 representative JPEGs/PNGs from the
#                deployment scene.
#   --name       Output basename (e.g. "model" -> model_int8.cvimodel)
#   --shape      Optional, defaults to "[[1,3,640,640]]"
#   --processor  cv181x | cv182x | cv183x  (cv181x = SG2002)  [default cv181x]
#   --quantize   INT8 | F16  [default INT8 — INT8 is required for SG2002 perf]
#
# Output
# ------
#   ${name}_int8.cvimodel       Flash to /data on the device.
#   ${name}_calib_table         Reusable calibration table.
#   ${name}.mlir                Intermediate, kept for debugging.
#
# Pre-processing baked into the cvimodel:
#   mean=0,0,0   scale=1/255   pixel_format=rgb
#   --customization_format RGB_PACKED + --fuse_preprocess + --aligned_input
#   means the model accepts raw aligned RGB888 bytes from the camera and
#   does normalisation+quantisation internally — no Elixir-side preproc
#   needed beyond resizing the camera frame to 640x640.

set -euo pipefail

ONNX=""
CALIB_DIR=""
NAME=""
SHAPE='[[1,3,640,640]]'
PROCESSOR="cv181x"
QUANTIZE="INT8"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx)      ONNX="$2";      shift 2 ;;
        --calib-dir) CALIB_DIR="$2"; shift 2 ;;
        --name)      NAME="$2";      shift 2 ;;
        --shape)     SHAPE="$2";     shift 2 ;;
        --processor) PROCESSOR="$2"; shift 2 ;;
        --quantize)  QUANTIZE="$2";  shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

[[ -z "$ONNX" || -z "$NAME" ]] && {
    echo "Missing required argument. Use --help." >&2; exit 2; }

if [[ "$QUANTIZE" == "INT8" && -z "$CALIB_DIR" ]]; then
    echo "INT8 quantization requires --calib-dir." >&2; exit 2
fi

# 1) ONNX -> top-mlir (FP32 reference)
#
# Output names are head_p3,head_p4,head_p5 — produced by the export script.
# `--keep_aspect_ratio --pixel_format rgb` matches Seeed's YOLO recipe.
model_transform.py \
    --model_name "$NAME" \
    --model_def "$ONNX" \
    --input_shapes "$SHAPE" \
    --mean "0.0,0.0,0.0" \
    --scale "0.0039216,0.0039216,0.0039216" \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names head_p3,head_p4,head_p5 \
    --mlir "${NAME}.mlir"

if [[ "$QUANTIZE" == "INT8" ]]; then
    # 2) Calibrate -> calibration table (INT8 scales for activations)
    run_calibration.py "${NAME}.mlir" \
        --dataset "$CALIB_DIR" \
        --input_num 100 \
        -o "${NAME}_calib_table"

    # 3) Deploy -> cvimodel (INT8 symmetric)
    model_deploy.py \
        --mlir "${NAME}.mlir" \
        --quantize INT8 \
        --quant_input \
        --processor "$PROCESSOR" \
        --calibration_table "${NAME}_calib_table" \
        --customization_format RGB_PACKED \
        --fuse_preprocess \
        --aligned_input \
        --model "${NAME}_int8.cvimodel"

    OUTFILE="${NAME}_int8.cvimodel"
else
    # F16 fallback (used while iterating; still supported on cv181x but slower)
    model_deploy.py \
        --mlir "${NAME}.mlir" \
        --quantize F16 \
        --quant_input \
        --processor "$PROCESSOR" \
        --customization_format RGB_PACKED \
        --fuse_preprocess \
        --aligned_input \
        --tolerance 0.99,0.9 \
        --model "${NAME}_f16.cvimodel"

    OUTFILE="${NAME}_f16.cvimodel"
fi

echo
echo "[ok] wrote $OUTFILE"
echo
echo "Flash $OUTFILE to /data on the SG2002 device, then in IEx:"
echo
echo "    {:ok, engine} = SSCMEx.Engine.new()"
echo "    :ok = SSCMEx.Engine.load(engine, \"/data/$OUTFILE\")"
echo "    {:ok, model} = SSCMEx.Model.create(engine)"
echo "    {:ok, :yolov7} = SSCMEx.Model.get_type(model)"
echo
echo "    :ok = SSCMEx.Model.set_config(model, :threshold_score, 0.45)"
echo "    :ok = SSCMEx.Model.set_config(model, :threshold_nms,   0.45)"
echo "    {:ok, dets} = SSCMEx.Model.run(model, image)"
