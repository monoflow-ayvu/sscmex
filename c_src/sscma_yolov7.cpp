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

// Standard YOLOv7 P3/P4/P5 anchor priors (input-image pixel space). They
// were extracted from the original safety.onnx initializers (`select_1`,
// `select_3`, `select_5`); see scripts/yolov7_to_clean_onnx.py for how
// they're written to the metadata.json. If you train/finetune YOLOv7 with
// custom anchors, edit the values here to match.
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

}  // namespace


YoloV7::YoloV7(Engine* engine)
    : Detector(engine, "yolov7", static_cast<ma_model_type_t>(kModelType)),
      num_class_(0),
      input_w_(0),
      input_h_(0) {
    input_w_ = static_cast<int32_t>(img_.width);
    input_h_ = static_cast<int32_t>(img_.height);

    // Try to look the heads up by name first (matches the names emitted by
    // scripts/yolov7_to_clean_onnx.py: head_p3, head_p4, head_p5). The
    // cvimodel runtime appends suffixes like `_Transpose_f32`, so a strict
    // name match will usually miss; we fall back to sorting outputs by
    // grid-cell count (largest -> P3 -> stride 8).
    int32_t order[kNumScales] = {-1, -1, -1};
    order[0] = engine->getOutputNum("head_p3");
    order[1] = engine->getOutputNum("head_p4");
    order[2] = engine->getOutputNum("head_p5");

    bool by_name_ok = (order[0] >= 0 && order[1] >= 0 && order[2] >= 0 &&
                       order[0] != order[1] && order[1] != order[2] &&
                       order[0] != order[2]);

    auto grid_h_of = [](const ma_shape_t& sh) -> int32_t {
        // rank-5: [1, 3, H, W, 5+nc]   (raw ONNX layout)
        // rank-4: [1, 3, H, W*(5+nc)]  (cvimodel collapses the last two)
        if (sh.size == 5) return sh.dims[2];
        if (sh.size == 4) return sh.dims[2];
        return 0;
    };
    auto grid_w_of = [](const ma_shape_t& sh, int32_t num_features) -> int32_t {
        if (sh.size == 5) return sh.dims[3];
        if (sh.size == 4 && num_features > 0) return sh.dims[3] / num_features;
        return 0;
    };

    // Determine num_class from the largest output: square grid, so
    //   nf == dims[last] / grid_h   (rank-4)  or
    //   nf == dims[4]               (rank-5)
    // Pick the output with the largest dims[2]*(dims[last]/...) to be safe.
    {
        const int32_t out_count = engine->getOutputSize();
        int32_t pick = 0;
        int32_t max_cells = -1;
        for (int32_t i = 0; i < out_count && i < kNumScales; ++i) {
            const auto sh = engine->getOutputShape(i);
            const int32_t h = grid_h_of(sh);
            int32_t cells = h * h;  // assume square
            if (cells > max_cells) {
                max_cells = cells;
                pick = i;
            }
        }
        const auto big = engine->getOutputShape(pick);
        const int32_t h = grid_h_of(big);
        int32_t nf = 0;
        if (big.size == 5) {
            nf = big.dims[4];
        } else if (big.size == 4 && h > 0) {
            // Square grid: dims[3] = W * nf = h * nf  =>  nf = dims[3] / h
            nf = big.dims[3] / h;
        }
        num_class_ = nf - kBoxFields;
    }

    if (!by_name_ok) {
        struct Info { int32_t idx; int32_t cells; };
        Info infos[kNumScales];
        const int32_t out_count = engine->getOutputSize();
        const int n = (out_count < kNumScales) ? out_count : kNumScales;
        for (int32_t i = 0; i < n; ++i) {
            const auto sh = engine->getOutputShape(i);
            const int32_t H = grid_h_of(sh);
            infos[i] = {i, H * H};  // square grid
        }
        std::sort(infos, infos + n,
                  [](const Info& a, const Info& b) { return a.cells > b.cells; });
        for (int32_t i = 0; i < n; ++i) order[i] = infos[i].idx;
    }

    for (int32_t s = 0; s < kNumScales; ++s) {
        Scale& sc = scales_[s];
        sc.tensor   = engine->getOutput(order[s]);
        const auto& sh = sc.tensor.shape;
        sc.grid_h = grid_h_of(sh);
        sc.grid_w = grid_w_of(sh, kBoxFields + num_class_);

        // Stride is input_w / grid_w when feature map is in normal NCHW
        // order. Square inputs are standard for YOLOv7.
        sc.stride = (sc.grid_w > 0)
                        ? (input_w_ / sc.grid_w)
                        : (s == 0 ? 8 : (s == 1 ? 16 : 32));

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

    const auto in_sh = engine->getInputShape(0);
    if (in_sh.size != 4) return false;

    int32_t n = in_sh.dims[0];
    int32_t h = in_sh.dims[1];
    int32_t w = in_sh.dims[2];
    int32_t c = in_sh.dims[3];
    const bool nhwc = (c == 3 || c == 1);
    if (!nhwc) std::swap(h, c);
    if (n != 1 || h < 32 || h % 32 != 0 || (c != 3 && c != 1)) return false;

    // Output may be reported as either:
    //   rank-5: [1, 3, H, W, 5+nc]   (raw ONNX layout)
    //   rank-4: [1, 3, H, W*(5+nc)]  (cvimodel collapses last two dims)
    // We accept both. For the rank-4 case we assume square grids
    // (W == H), which is the normal YOLOv7 layout for square inputs.
    int32_t total_cells = 0;
    int32_t last_nf     = -1;
    for (int32_t i = 0; i < kNumScales; ++i) {
        const auto sh = engine->getOutputShape(i);
        if (sh.size != 4 && sh.size != 5)            return false;
        if (sh.dims[0] != 1)                          return false;
        if (sh.dims[1] != kAnchorsPerCell)            return false;
        if (sh.dims[2] <= 0 || sh.dims[3] <= 0)       return false;

        int32_t grid_h = sh.dims[2];
        int32_t grid_w = 0;
        int32_t nf     = 0;
        if (sh.size == 5) {
            grid_w = sh.dims[3];
            nf     = sh.dims[4];
        } else {
            // rank-4: dims[3] = grid_w * nf. Assume square grid.
            grid_w = grid_h;
            if (grid_w == 0 || sh.dims[3] % grid_w != 0) return false;
            nf = sh.dims[3] / grid_w;
        }
        if (nf < (kBoxFields + 1) || nf > (kBoxFields + 80)) return false;
        if (last_nf < 0) last_nf = nf;
        else if (last_nf != nf)                   return false;

        total_cells += grid_h * grid_w;
    }

    const int32_t hs8  = h / 8,  ws8  = w / 8;
    const int32_t hs16 = h / 16, ws16 = w / 16;
    const int32_t hs32 = h / 32, ws32 = w / 32;
    const int32_t expected_cells = hs8 * ws8 + hs16 * ws16 + hs32 * ws32;
    return total_cells == expected_cells;
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
