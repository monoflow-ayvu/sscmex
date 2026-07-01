/*
 * c_src/sscma_yolov7.h — YOLOv7 anchor-based decoder for SSCMA-Micro
 *
 * SSCMA-Micro upstream ships YoloV5/YoloV8/Yolo11 decoders but not YoloV7.
 * This class plugs into the same `ma::model::Detector` base class, so it
 * lives behind the existing SSCMEx.Model / SSCMEx.Engine APIs without
 * modifying the vendored submodule.
 *
 * It expects a cvimodel produced from a YOLOv7 ONNX whose post-processing
 * (anchor decoding + sigmoid + Concat) has been stripped, leaving three
 * raw Conv heads as outputs. See scripts/yolov7_to_clean_onnx.py for the
 * exact graph surgery and scripts/build_yolov7_cvimodel.sh for the
 * TPU-MLIR pipeline that compiles it to .cvimodel.
 *
 * Expected outputs (any order — looked up by name first, falls back to
 * sorting by feature-map size):
 *
 *   head_p3 : [1, 3, H/8 , W/8 , 5+nc]   stride 8
 *   head_p4 : [1, 3, H/16, W/16, 5+nc]   stride 16
 *   head_p5 : [1, 3, H/32, W/32, 5+nc]   stride 32
 *
 * Each anchor cell stores [tx, ty, tw, th, t_obj, t_cls0..t_cls(nc-1)]
 * pre-sigmoid. The standard YOLOv7 decoding rule is:
 *
 *   bx     = (sigmoid(tx) * 2 - 0.5 + grid_x) * stride
 *   by     = (sigmoid(ty) * 2 - 0.5 + grid_y) * stride
 *   bw     = (sigmoid(tw) * 2)^2 * anchor_w
 *   bh     = (sigmoid(th) * 2)^2 * anchor_h
 *   score  = sigmoid(t_obj) * sigmoid(max_t_cls)
 *   target = argmax(t_cls)
 *
 * The class supports both s8 (quantized) and f32 outputs — the common
 * cv181x output types.
 */

#ifndef _SSCMEX_YOLOV7_H
#define _SSCMEX_YOLOV7_H

#include <cstdint>

#include "sscma/core/engine/ma_engine_base.h"
#include "sscma/core/model/ma_model_detector.h"

namespace sscmex {

class YoloV7 : public ma::model::Detector {
public:
    // Reported via getType(); must not alias any ma_model_type_t or
    // model_type_to_atom() will mislabel a real vendor type. 15u collided with
    // MA_MODEL_TYPE_YOLO11_POSE_SH; 16u is past the vendor enum's end.
    static constexpr uint16_t kModelType = 16u;

    explicit YoloV7(ma::engine::Engine* engine);
    ~YoloV7() override;

    // Returns true iff the engine's outputs match the YOLOv7 raw-head
    // pattern (3 outputs, each [1, 3, H, W, 5+nc] with H,W matching the
    // input strides 8/16/32 and 1 <= nc <= 80).
    static bool isValid(ma::engine::Engine* engine);

protected:
    ma_err_t postprocess() override;

private:
    static constexpr int kNumScales      = 3;
    static constexpr int kAnchorsPerCell = 3;
    static constexpr int kBoxFields      = 5;  // tx, ty, tw, th, t_obj

    struct Scale {
        ma_tensor_t tensor;
        int32_t grid_h;
        int32_t grid_w;
        int32_t stride;
        // Per-anchor (w,h) priors in input-image pixel space.
        float anchors[kAnchorsPerCell][2];
    };

    Scale scales_[kNumScales];
    int32_t num_class_;
    int32_t input_w_;
    int32_t input_h_;

    void postprocessScaleS8(const Scale& s);
    void postprocessScaleF32(const Scale& s);
};

}  // namespace sscmex

#endif  // _SSCMEX_YOLOV7_H
