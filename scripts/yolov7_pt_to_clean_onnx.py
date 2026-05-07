#!/usr/bin/env python3
"""
yolov7_pt_to_clean_onnx.py — Re-export a YOLOv7 .pt to a TPU-MLIR-friendly,
self-contained ONNX whose graph stops at the three raw Conv heads.

This is the recommended path (over scripts/yolov7_to_clean_onnx.py, which
operates on an existing ONNX) because it produces an opset-17 / IR-8 ONNX
with all weights inlined — exactly what `tpu_mlir==1.7` (and the
`sophgo/tpuc_dev:v3.1` image) accepts directly, with no downgrade step.

Why we strip post-processing
----------------------------
The default `python export.py --weights safety.pt --grid` graph embeds
sigmoid + anchor decoding into ONNX using `ScatterND` and dynamic slices.
TPU-MLIR's cv181x backend either rejects those or compiles them into very
slow CPU fallbacks. By setting `Detect.export = True` (or `IDetect.export
= True` on heavy YOLOv7 variants), `forward()` short-circuits and just
returns the three permuted Conv outputs `[1, 3, H, W, 5+nc]`. The runtime
side (sscmex/c_src/sscma_yolov7.cpp) handles the remaining math.

Usage
-----
First clone the YOLOv7 repo somewhere; we need its `models` package on
sys.path to deserialize the checkpoint:

    git clone --depth 1 https://github.com/WongKinYiu/yolov7.git /tmp/yolov7

Then:

    python scripts/yolov7_pt_to_clean_onnx.py \\
        --pt   /path/to/safety.pt \\
        --out  /path/to/safety_clean.onnx \\
        --yolov7-repo /tmp/yolov7 \\
        --classes-json /path/to/safety_map.json

Notes
-----
* The script writes a metadata JSON next to the ONNX (`<out>.metadata.json`)
  containing strides, anchor priors and class names — purely informational
  for downstream tools; the runtime decoder hardcodes the standard YOLOv7
  P3/P4/P5 anchors.
* If your training used custom anchors, edit `kAnchors` in
  `c_src/sscma_yolov7.cpp` to match — they're verifiable from the .pt via
  `ckpt['model'].model[-1].anchor_grid`.
"""

import argparse
import json
import os
import sys


def export(pt_path: str, out_path: str, yolov7_repo: str, classes):
    if not os.path.isdir(yolov7_repo):
        raise SystemExit(
            f"--yolov7-repo {yolov7_repo!r} does not exist. Clone with: "
            "git clone --depth 1 https://github.com/WongKinYiu/yolov7.git")
    sys.path.insert(0, yolov7_repo)

    import warnings; warnings.filterwarnings("ignore")
    import torch
    from models.yolo import Detect, IDetect

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = ckpt["model"].float().eval()
    nc = int(model.nc)

    if classes is None:
        classes = list(getattr(model, "names", [str(i) for i in range(nc)]))
    if len(classes) != nc:
        raise SystemExit(
            f"class count mismatch: model has {nc} classes, got {len(classes)}.")

    # Set export=True on every detection head so forward() skips
    # sigmoid + anchor decoding and returns the raw permuted heads.
    head = None
    for m in model.modules():
        if isinstance(m, (Detect, IDetect)):
            m.export = True
            head = m
        if hasattr(m, "inplace"):
            m.inplace = False
    if head is None:
        raise SystemExit("No Detect/IDetect head found in the model.")

    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out = model(dummy)
    if not (isinstance(out, list) and len(out) == 3):
        raise SystemExit(
            f"Unexpected forward output {type(out).__name__} (len="
            f"{len(out) if hasattr(out, '__len__') else '?'}). "
            "Detect.export = True did not take effect.")

    # Confirm raw-head shapes look like [1, 3, H, W, 5+nc] in P3/P4/P5 order
    shapes = [tuple(t.shape) for t in out]
    print(f"raw heads: {shapes}")
    for sh in shapes:
        assert sh[0] == 1 and sh[1] == 3 and sh[4] == 5 + nc, sh

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["images"],
        output_names=["head_p3", "head_p4", "head_p5"],
        opset_version=17,        # TPU-MLIR 1.7 accepts opset 17
        do_constant_folding=True,
        dynamo=False,            # use the legacy TorchScript exporter
    )

    # Side metadata file — anchors recovered straight from the head buffers
    anchor_grid = head.anchor_grid.detach().reshape(3, 3, 2).tolist()
    strides     = (head.stride.detach().tolist()
                   if hasattr(head, "stride") and head.stride is not None
                   else [8, 16, 32])
    meta = {
        "input": {"name": "images", "shape": [1, 3, 640, 640], "layout": "NCHW"},
        "outputs": [
            {"name": "head_p3", "stride": int(strides[0]),
             "shape": list(shapes[0]), "anchors": anchor_grid[0]},
            {"name": "head_p4", "stride": int(strides[1]),
             "shape": list(shapes[1]), "anchors": anchor_grid[1]},
            {"name": "head_p5", "stride": int(strides[2]),
             "shape": list(shapes[2]), "anchors": anchor_grid[2]},
        ],
        "num_classes": nc,
        "classes": list(classes),
        "preprocess": {"mean": [0.0, 0.0, 0.0],
                       "scale": [1.0 / 255.0] * 3, "color_order": "RGB"},
    }
    meta_path = os.path.splitext(out_path)[0] + ".metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    print(f"[ok] wrote {meta_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Re-export YOLOv7 .pt -> clean self-contained ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--pt", dest="pt_path", required=True,
                    help="Path to YOLOv7 .pt checkpoint")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Output cleaned ONNX path")
    ap.add_argument("--yolov7-repo", required=True,
                    help="Path to a checkout of github.com/WongKinYiu/yolov7")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="Class names (training order). Defaults to "
                         "model.names if present.")
    ap.add_argument("--classes-json",
                    help="JSON file whose keys (in order) are class names")
    args = ap.parse_args()

    classes = args.classes
    if args.classes_json:
        with open(args.classes_json) as f:
            classes = list(json.load(f).keys())

    export(args.pt_path, args.out_path, args.yolov7_repo, classes)


if __name__ == "__main__":
    main()
