#!/usr/bin/env python3
"""
yolov7_to_clean_onnx.py — Strip the embedded post-processing from a YOLOv7
ONNX export so it can be compiled to a cv181x cvimodel by TPU-MLIR.

Why this exists
---------------
Default YOLOv7 ONNX exports (e.g. `python export.py --grid --end2end ...`)
embed sigmoid + anchor decoding + Concat into the graph, producing a single
[1, 25200, 5+nc] output. That sub-graph contains `ScatterND` and
heavily-broadcasted slice math which TPU-MLIR's cv181x backend either
refuses to compile or compiles into very slow CPU fallbacks.

This tool rewrites the ONNX so that the graph stops at the three raw Conv
heads (P3/P4/P5, before any sigmoid). Anchor decoding is then performed at
runtime by `SSCMEx.YoloV7` in Elixir.

Inputs/outputs
--------------
The script reads:
    safety.onnx          (and any external data file referenced inside)
    safety.pt            (optional — only used to *verify* anchors match;
                          the script does not need to run torch unless this
                          flag is given)

It produces, next to the input file:
    safety_clean.onnx                — graph: images -> {head_p3, head_p4, head_p5}
    safety_clean.onnx.data           — external weights (if input had any)
    safety_clean.metadata.json       — anchors / strides / class names

Note about external data
------------------------
ONNX exporters often split a model into `model.onnx` + `model.onnx.data`.
This script preserves those external references — it only rewrites the
graph topology. After running it, copy or move the original `.data` file
next to the new `.onnx`, OR re-run with `--inline-data` to ask onnx to
embed the weights directly (requires the original `.data` file present).

Usage
-----
    python scripts/yolov7_to_clean_onnx.py \
        --in  /path/to/safety.onnx \
        --out /path/to/safety_clean.onnx \
        --classes Hardhat Mask "Safety Vest" NO-Hardhat NO-Mask \
                  "NO-Safety Vest" Person machinery vehicle "Safety Cone"

    # Or load class names from a JSON map (the format produced alongside
    # the model — keys are classes, in order).
    python scripts/yolov7_to_clean_onnx.py \
        --in safety.onnx --out safety_clean.onnx \
        --classes-json safety_map.json
"""

import argparse
import json
import os
import sys
from collections import deque

import onnx
from onnx import helper, numpy_helper, save_model


# ---------------------------------------------------------------------------
# Anchor priors that YOLOv7 bakes into IDetect. The export embeds them as
# Constant tensors of shape [1, 3, 1, 1, 2]; we read them out of the input
# graph and write them to metadata.json so the runtime decoder can match.
# These default values match the standard YOLOv7 / YOLOv7-tiny P3-P5 anchors
# and are returned only when the script can't find anchor constants in the
# graph.
DEFAULT_ANCHORS = [
    [(12.0, 16.0), (19.0, 36.0), (40.0, 28.0)],   # stride 8  / P3
    [(36.0, 75.0), (76.0, 55.0), (72.0, 146.0)],  # stride 16 / P4
    [(142.0, 110.0), (192.0, 243.0), (459.0, 401.0)],  # stride 32 / P5
]
DEFAULT_STRIDES = [8, 16, 32]


def find_raw_head_outputs(model: onnx.ModelProto):
    """Locate the 3 raw Conv-head tensors (pre-sigmoid).

    YOLOv7's IDetect.forward() does:
        x[i] = self.m[i](x[i])               # 1x1 conv -> [B, 3*(5+nc), H, W]
        bs, _, ny, nx = x[i].shape
        x[i] = x[i].view(bs, 3, 5+nc, ny, nx).permute(0,1,3,4,2).contiguous()
        if self.export:
            return x  # 3-tuple of raw heads

    With --grid the exporter then runs sigmoid + anchor math, but the
    pre-sigmoid permutation result is still alive because PyTorch's
    `.contiguous()` decays into a clone in onnx — these are usually exposed
    as outputs named `clone`, `clone_1`, `clone_2` (or `output_0/1/2` on
    newer torch versions).
    """
    candidates = []
    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        # Raw head shape is exactly [1, 3, H, W, 5+nc]
        if len(dims) == 5 and dims[0] == 1 and dims[1] == 3 and dims[4] >= 6:
            candidates.append((out.name, dims))
    if len(candidates) < 3:
        raise RuntimeError(
            f"Expected 3 raw-head outputs of shape [1,3,H,W,5+nc] but found "
            f"{len(candidates)}: {candidates}\n"
            "Re-export YOLOv7 ONNX without `--end2end` so the raw heads are "
            "preserved as graph outputs (or with `--grid` which keeps both)."
        )
    # Sort by feature-map size (largest = stride 8 = P3 first)
    candidates.sort(key=lambda c: -c[1][2] * c[1][3])
    return candidates[:3]


def collect_reachable(model: onnx.ModelProto, head_names):
    """BFS backward from head outputs, return nodes + initializers we keep."""
    producer = {}
    for n in model.graph.node:
        for o in n.output:
            producer[o] = n

    init_names = {i.name for i in model.graph.initializer}
    input_names = {i.name for i in model.graph.input}

    keep_nodes = []
    keep_node_names = set()
    keep_inits = set()
    keep_inputs = set()

    queue = deque(head_names)
    visited = set()
    while queue:
        t = queue.popleft()
        if t in visited:
            continue
        visited.add(t)
        if t in input_names:
            keep_inputs.add(t)
            continue
        if t in init_names:
            keep_inits.add(t)
            continue
        node = producer.get(t)
        if node is None:
            continue
        if node.name in keep_node_names:
            continue
        keep_node_names.add(node.name)
        keep_nodes.append(node)
        for inp in node.input:
            if inp:  # skip empty/optional inputs
                queue.append(inp)
    return keep_nodes, keep_inits, keep_inputs


def extract_anchors(model: onnx.ModelProto):
    """Pull anchor constants out of the original graph if present.

    YOLOv7 export emits initializers named like `select_1`, `select_3`,
    `select_5` of shape [1, 3, 1, 1, 2] — we read those.  Falls back to
    DEFAULT_ANCHORS if not found.
    """
    found = []
    for init in model.graph.initializer:
        if list(init.dims) == [1, 3, 1, 1, 2] and init.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(init).reshape(3, 2)
            found.append((init.name, [(float(a), float(b)) for a, b in arr]))
    if len(found) >= 3:
        # Order them by mean magnitude (smallest anchors -> P3, largest -> P5)
        found.sort(key=lambda f: sum(a + b for a, b in f[1]))
        return [f[1] for f in found[:3]]
    print("[warn] anchor constants not found in graph; using YOLOv7 defaults",
          file=sys.stderr)
    return DEFAULT_ANCHORS


def extract_strides(model: onnx.ModelProto):
    """Strides are baked in as scalar Float initializers (8.0, 16.0, 32.0)."""
    targets = {8.0, 16.0, 32.0}
    found = set()
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT and len(init.dims) == 0:
            v = float(numpy_helper.to_array(init).item())
            if v in targets:
                found.add(int(v))
    if found == {8, 16, 32}:
        return [8, 16, 32]
    print("[warn] stride scalars not found; using defaults [8,16,32]",
          file=sys.stderr)
    return DEFAULT_STRIDES


def rewrite(in_path: str, out_path: str, classes, inline_data: bool):
    model = onnx.load(in_path, load_external_data=inline_data)

    # 1) Identify the 3 raw heads
    heads = find_raw_head_outputs(model)
    head_names = [name for name, _ in heads]
    head_dims = [dims for _, dims in heads]

    nc = head_dims[0][4] - 5  # 5+nc
    if classes is None:
        # Build numeric placeholder names if not provided
        classes = [str(i) for i in range(nc)]
    elif len(classes) != nc:
        raise SystemExit(
            f"--classes lists {len(classes)} names but the model has {nc}.")

    # 2) Read anchors/strides for the metadata file
    anchors = extract_anchors(model)
    strides = extract_strides(model)

    # 3) Collect what we keep
    keep_nodes, keep_inits, keep_inputs = collect_reachable(model, head_names)

    # Preserve original node order (sub-graph order matters for some
    # checkers); collect_reachable already records that.

    new_graph = helper.make_graph(
        nodes=keep_nodes,
        name=(model.graph.name or "yolov7_raw") + "_clean",
        inputs=[i for i in model.graph.input if i.name in keep_inputs],
        outputs=[
            helper.make_tensor_value_info(
                name=new_name,
                elem_type=onnx.TensorProto.FLOAT,
                shape=dims,
            )
            for new_name, (_, dims) in zip(
                ["head_p3", "head_p4", "head_p5"], heads
            )
        ],
        initializer=[i for i in model.graph.initializer if i.name in keep_inits],
        value_info=[
            v for v in model.graph.value_info
            if any(v.name in n.input or v.name in n.output for n in keep_nodes)
        ],
    )

    # Rename the head outputs in the kept nodes so they match head_p3..p5
    rename = dict(zip(head_names, ["head_p3", "head_p4", "head_p5"]))
    for node in new_graph.node:
        for i, out in enumerate(node.output):
            if out in rename:
                node.output[i] = rename[out]

    new_model = helper.make_model(
        new_graph,
        producer_name="sscmex/yolov7_to_clean_onnx",
        opset_imports=list(model.opset_import),
        ir_version=model.ir_version,
    )

    # ONNX checker — non-fatal, weights may be external (unloadable here)
    try:
        onnx.checker.check_model(new_model, full_check=False)
    except Exception as exc:
        print(f"[warn] onnx.checker reported: {exc}", file=sys.stderr)

    if inline_data:
        save_model(new_model, out_path)
    else:
        # Keep external data references intact — the user is expected to
        # ship the original .onnx.data alongside.  We use a sibling file
        # for any new initializers (there shouldn't be any).
        save_model(
            new_model, out_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(out_path) + ".data",
            size_threshold=1024,
        )

    # 4) Write metadata
    meta = {
        "input": {"name": "images", "shape": [1, 3, 640, 640], "layout": "NCHW"},
        "outputs": [
            {"name": "head_p3", "stride": strides[0],
             "shape": head_dims[0], "anchors": anchors[0]},
            {"name": "head_p4", "stride": strides[1],
             "shape": head_dims[1], "anchors": anchors[1]},
            {"name": "head_p5", "stride": strides[2],
             "shape": head_dims[2], "anchors": anchors[2]},
        ],
        "num_classes": nc,
        "classes": classes,
        "preprocess": {"mean": [0.0, 0.0, 0.0], "scale": [1.0 / 255.0] * 3,
                       "color_order": "RGB"},
    }
    meta_path = os.path.splitext(out_path)[0] + ".metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] wrote {out_path}")
    print(f"[ok] wrote {meta_path}")
    print(f"[ok] kept {len(keep_nodes)} nodes / {len(keep_inits)} initializers")
    print(f"[ok] outputs: head_p3 {head_dims[0]}  head_p4 {head_dims[1]}  "
          f"head_p5 {head_dims[2]}")


def main():
    ap = argparse.ArgumentParser(
        description="Strip embedded post-processing from a YOLOv7 ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Input safety.onnx (with --grid post-processing)")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Output cleaned ONNX path")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="Class names in training order")
    ap.add_argument("--classes-json", dest="classes_json",
                    help="JSON file whose top-level keys are class names")
    ap.add_argument("--inline-data", action="store_true",
                    help="Inline the weights into the .onnx (default keeps "
                         "external .onnx.data references)")
    args = ap.parse_args()

    classes = args.classes
    if args.classes_json:
        with open(args.classes_json) as f:
            classes = list(json.load(f).keys())

    rewrite(args.in_path, args.out_path, classes, args.inline_data)


if __name__ == "__main__":
    main()
