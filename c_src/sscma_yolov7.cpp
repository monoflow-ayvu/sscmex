#include "sscma_yolov7.h"

#include <algorithm>
#include <cmath>
#include <forward_list>
#include <limits>

#include "sscma/core/ma_compiler.h"  // MA_CLIP
#include "sscma/core/ma_types.h"
#include "sscma/core/utils/ma_nms.h"

namespace sscmex {

using ma::engine::Engine;
using ma::model::Detector;

namespace {

// Standard YOLOv7 P3/P4/P5 anchor priors (input-image pixel space).
// Edit to match if you train/finetune YOLOv7 with custom anchors — verify
// against `model.model[-1].anchor_grid` from your .pt.
constexpr float kAnchors[3][3][2] = {
    {{ 12.f,  16.f}, { 19.f,  36.f}, { 40.f,  28.f}},  // stride 8  (P3)
    {{ 36.f,  75.f}, { 76.f,  55.f}, { 72.f, 146.f}},  // stride 16 (P4)
    {{142.f, 110.f}, {192.f, 243.f}, {459.f, 401.f}},  // stride 32 (P5)
};

inline float sigmoidf(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

// In logit space the score gate is `t_obj > logit(threshold_score)`; this
// short-circuits the per-cell sigmoid+class-search for boxes whose
// objectness can't possibly clear the threshold even before multiplying by
// a class probability <= 1.
inline float score_logit_threshold(float p) noexcept {
    if (p <= 0.0f) return -std::numeric_limits<float>::infinity();
    if (p >= 1.0f) return  std::numeric_limits<float>::infinity();
    return std::log(p / (1.0f - p));
}

// Extract real (H, W) from a 4D input shape, regardless of NHWC vs NCHW.
// Returns false if the shape doesn't look like a single image batch.
inline bool extract_input_hw(const ma_shape_t& in_sh, int32_t& h, int32_t& w) {
    if (in_sh.size != 4)      return false;
    if (in_sh.dims[0] != 1)   return false;
    const bool nhwc = (in_sh.dims[3] == 3 || in_sh.dims[3] == 1);
    const bool nchw = (in_sh.dims[1] == 3 || in_sh.dims[1] == 1);
    if (nhwc) {
        h = in_sh.dims[1];
        w = in_sh.dims[2];
    } else if (nchw) {
        h = in_sh.dims[2];
        w = in_sh.dims[3];
    } else {
        return false;
    }
    return true;
}

constexpr int32_t kStrides[3] = {8, 16, 32};

}  // namespace


YoloV7::YoloV7(Engine* engine)
    : Detector(engine, "yolov7", static_cast<ma_model_type_t>(kModelType)),
      num_class_(0),
      input_w_(0),
      input_h_(0) {
    input_w_ = static_cast<int32_t>(img_.width);
    input_h_ = static_cast<int32_t>(img_.height);

    // Map each engine output to its scale by matching the output's grid_h
    // against input_h / stride. Works for both square and rectangular
    // inputs and does not depend on output ordering.
    int32_t order[kNumScales] = {-1, -1, -1};
    const int32_t out_count = engine->getOutputSize();
    const int n = (out_count < kNumScales) ? out_count : kNumScales;
    for (int32_t i = 0; i < n; ++i) {
        const auto sh = engine->getOutputShape(i);
        if (sh.size != 4 && sh.size != 5) continue;
        const int32_t grid_h = sh.dims[2];
        for (int32_t s = 0; s < kNumScales; ++s) {
            if (order[s] < 0 && kStrides[s] > 0 &&
                grid_h == input_h_ / kStrides[s]) {
                order[s] = i;
                break;
            }
        }
    }

    // Derive num_class from any successfully-mapped output. Once we know
    // the stride, grid_w is determined by input_w / stride and so is nf.
    for (int32_t s = 0; s < kNumScales; ++s) {
        if (order[s] < 0) continue;
        const auto sh = engine->getOutputShape(order[s]);
        const int32_t grid_w = input_w_ / kStrides[s];
        int32_t nf = 0;
        if (sh.size == 5) {
            nf = sh.dims[4];
        } else if (sh.size == 4 && grid_w > 0) {
            // rank-4 cvimodel layout: dims[3] = grid_w * nf
            nf = sh.dims[3] / grid_w;
        }
        if (nf > kBoxFields) {
            num_class_ = nf - kBoxFields;
            break;
        }
    }

    for (int32_t s = 0; s < kNumScales; ++s) {
        Scale& sc = scales_[s];
        if (order[s] < 0) continue;  // shouldn't happen if isValid() passed
        sc.tensor = engine->getOutput(order[s]);
        sc.stride = kStrides[s];
        sc.grid_h = input_h_ / kStrides[s];
        sc.grid_w = input_w_ / kStrides[s];
        for (int a = 0; a < kAnchorsPerCell; ++a) {
            sc.anchors[a][0] = kAnchors[s][a][0];
            sc.anchors[a][1] = kAnchors[s][a][1];
        }
    }
}

YoloV7::~YoloV7() = default;

bool YoloV7::isValid(Engine* engine) {
    if (engine == nullptr) return false;
    if (engine->getOutputSize() != kNumScales) return false;
    if (engine->getInputSize()  != 1)          return false;

    int32_t in_h = 0, in_w = 0;
    if (!extract_input_hw(engine->getInputShape(0), in_h, in_w)) return false;
    if (in_h < 32 || in_h % 32 != 0 || in_w < 32 || in_w % 32 != 0) return false;

    // Each output must:
    //   * carry a 4D or 5D shape (rank-5 = raw ONNX layout, rank-4 =
    //     cvimodel collapses the last two dims into one)
    //   * have dims[1] == 3 anchors per cell
    //   * match exactly one of the strides {8, 16, 32}, identified by
    //     grid_h == input_h / stride
    //   * have a feature count nf in [kBoxFields+1, kBoxFields+80]
    //     consistent across all three outputs.
    bool stride_seen[kNumScales] = {false, false, false};
    int32_t last_nf = -1;

    for (int32_t i = 0; i < kNumScales; ++i) {
        const auto sh = engine->getOutputShape(i);
        if (sh.size != 4 && sh.size != 5)             return false;
        if (sh.dims[0] != 1)                          return false;
        if (sh.dims[1] != kAnchorsPerCell)            return false;
        if (sh.dims[2] <= 0 || sh.dims[3] <= 0)       return false;

        const int32_t grid_h = sh.dims[2];
        int32_t stride_idx = -1;
        for (int32_t s = 0; s < kNumScales; ++s) {
            if (!stride_seen[s] && grid_h == in_h / kStrides[s]) {
                stride_idx = s;
                break;
            }
        }
        if (stride_idx < 0) return false;
        stride_seen[stride_idx] = true;

        const int32_t expected_grid_w = in_w / kStrides[stride_idx];
        int32_t nf = 0;
        if (sh.size == 5) {
            if (sh.dims[3] != expected_grid_w) return false;
            nf = sh.dims[4];
        } else {
            // rank-4: dims[3] = grid_w * nf — divide by the grid_w that
            // the input shape implies for this stride, no guessing.
            if (expected_grid_w <= 0) return false;
            if (sh.dims[3] % expected_grid_w != 0) return false;
            nf = sh.dims[3] / expected_grid_w;
        }
        if (nf < (kBoxFields + 1) || nf > (kBoxFields + 80)) return false;
        if (last_nf < 0) last_nf = nf;
        else if (last_nf != nf) return false;
    }

    return stride_seen[0] && stride_seen[1] && stride_seen[2];
}

void YoloV7::postprocessScaleS8(const Scale& s) {
    const float scale = s.tensor.quant_param.scale;
    const int32_t zp  = s.tensor.quant_param.zero_point;
    const int8_t* data = s.tensor.data.s8;
    if (data == nullptr || scale == 0.0f) return;

    const int   record_stride = kBoxFields + num_class_;
    const float img_w = static_cast<float>(input_w_);
    const float img_h = static_cast<float>(input_h_);
    const float thr_score = static_cast<float>(threshold_score_);
    const float logit_thr = score_logit_threshold(thr_score);
    const float stride_f  = static_cast<float>(s.stride);

    for (int a = 0; a < kAnchorsPerCell; ++a) {
        for (int gy = 0; gy < s.grid_h; ++gy) {
            for (int gx = 0; gx < s.grid_w; ++gx) {
                const int idx = ((a * s.grid_h + gy) * s.grid_w + gx) * record_stride;

                // Early gate on raw objectness — saves the per-cell
                // sigmoid + class scan when t_obj is too low to ever
                // clear thr_score (since cls_p <= 1).
                const float t_obj = (data[idx + 4] - zp) * scale;
                if (t_obj <= logit_thr) continue;
                const float obj_p = sigmoidf(t_obj);

                int     best_t   = 0;
                int8_t  best_raw = data[idx + kBoxFields];
                for (int t = 1; t < num_class_; ++t) {
                    const int8_t v = data[idx + kBoxFields + t];
                    if (v > best_raw) { best_raw = v; best_t = t; }
                }
                const float t_cls = (best_raw - zp) * scale;
                const float cls_p = sigmoidf(t_cls);
                const float score = obj_p * cls_p;
                if (score <= thr_score) continue;

                const float tx = (data[idx + 0] - zp) * scale;
                const float ty = (data[idx + 1] - zp) * scale;
                const float tw = (data[idx + 2] - zp) * scale;
                const float th = (data[idx + 3] - zp) * scale;

                const float bx = (sigmoidf(tx) * 2.0f - 0.5f + static_cast<float>(gx)) * stride_f;
                const float by = (sigmoidf(ty) * 2.0f - 0.5f + static_cast<float>(gy)) * stride_f;
                const float bw_factor = sigmoidf(tw) * 2.0f;
                const float bh_factor = sigmoidf(th) * 2.0f;
                const float bw = bw_factor * bw_factor * s.anchors[a][0];
                const float bh = bh_factor * bh_factor * s.anchors[a][1];

                ma_bbox_t box;
                box.x      = MA_CLIP(bx / img_w, 0.0f, 1.0f);
                box.y      = MA_CLIP(by / img_h, 0.0f, 1.0f);
                box.w      = MA_CLIP(bw / img_w, 0.0f, 1.0f);
                box.h      = MA_CLIP(bh / img_h, 0.0f, 1.0f);
                box.score  = score;
                box.target = best_t;
                results_.emplace_front(box);
            }
        }
    }
}

void YoloV7::postprocessScaleF32(const Scale& s) {
    const float* data = s.tensor.data.f32;
    if (data == nullptr) return;

    const int   record_stride = kBoxFields + num_class_;
    const float img_w = static_cast<float>(input_w_);
    const float img_h = static_cast<float>(input_h_);
    const float thr_score = static_cast<float>(threshold_score_);
    const float logit_thr = score_logit_threshold(thr_score);
    const float stride_f  = static_cast<float>(s.stride);

    for (int a = 0; a < kAnchorsPerCell; ++a) {
        for (int gy = 0; gy < s.grid_h; ++gy) {
            for (int gx = 0; gx < s.grid_w; ++gx) {
                const int idx = ((a * s.grid_h + gy) * s.grid_w + gx) * record_stride;

                const float t_obj = data[idx + 4];
                if (t_obj <= logit_thr) continue;
                const float obj_p = sigmoidf(t_obj);

                int   best_t   = 0;
                float best_raw = data[idx + kBoxFields];
                for (int t = 1; t < num_class_; ++t) {
                    const float v = data[idx + kBoxFields + t];
                    if (v > best_raw) { best_raw = v; best_t = t; }
                }
                const float cls_p = sigmoidf(best_raw);
                const float score = obj_p * cls_p;
                if (score <= thr_score) continue;

                const float tx = data[idx + 0];
                const float ty = data[idx + 1];
                const float tw = data[idx + 2];
                const float th = data[idx + 3];

                const float bx = (sigmoidf(tx) * 2.0f - 0.5f + static_cast<float>(gx)) * stride_f;
                const float by = (sigmoidf(ty) * 2.0f - 0.5f + static_cast<float>(gy)) * stride_f;
                const float bw_factor = sigmoidf(tw) * 2.0f;
                const float bh_factor = sigmoidf(th) * 2.0f;
                const float bw = bw_factor * bw_factor * s.anchors[a][0];
                const float bh = bh_factor * bh_factor * s.anchors[a][1];

                ma_bbox_t box;
                box.x      = MA_CLIP(bx / img_w, 0.0f, 1.0f);
                box.y      = MA_CLIP(by / img_h, 0.0f, 1.0f);
                box.w      = MA_CLIP(bw / img_w, 0.0f, 1.0f);
                box.h      = MA_CLIP(bh / img_h, 0.0f, 1.0f);
                box.score  = score;
                box.target = best_t;
                results_.emplace_front(box);
            }
        }
    }
}

ma_err_t YoloV7::postprocess() {
    results_.clear();

    for (int s = 0; s < kNumScales; ++s) {
        const Scale& sc = scales_[s];
        switch (sc.tensor.type) {
            case MA_TENSOR_TYPE_S8:
                postprocessScaleS8(sc);
                break;
            case MA_TENSOR_TYPE_F32:
                postprocessScaleF32(sc);
                break;
            default:
                return MA_ENOTSUP;
        }
    }

    ma::utils::nms(results_, threshold_nms_, threshold_score_, false, false);
    results_.sort([](const ma_bbox_t& a, const ma_bbox_t& b) { return a.x < b.x; });
    return MA_OK;
}

}  // namespace sscmex
