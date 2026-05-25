#include "ma_model_yolo11_pose_sh.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <math.h>
#include <vector>

#include "../math/ma_math.h"
#include "../utils/ma_nms.h"

namespace ma::model {

static void compute_dfl(float* tensor, int dfl_len, float* box) {
    for (int b = 0; b < 4; b++) {
        float exp_t[dfl_len];
        float exp_sum = 0;
        float acc_sum = 0;
        for (int i = 0; i < dfl_len; i++) {
            exp_t[i] = exp(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }
        for (int i = 0; i < dfl_len; i++) {
            acc_sum += exp_t[i] / exp_sum * i;
        }
        box[b] = acc_sum;
    }
}

struct TensorInfo {
    int index;
    int channels;
    int grid_h;
    int grid_w;
};

static bool classifyOutputs(Engine* engine, int box_idx[3], int cls_idx[3], int kpt_idx[3], int& cls_ch, int& kpt_ch) {
    std::vector<TensorInfo> box_tensors, other_tensors;
    for (int i = 0; i < 9; i++) {
        auto shape = engine->getOutputShape(i);
        if (shape.size != 4 || shape.dims[0] != 1)
            return false;
        TensorInfo info = {i, shape.dims[1], shape.dims[2], shape.dims[3]};
        if (info.channels == 64) {
            box_tensors.push_back(info);
        } else {
            other_tensors.push_back(info);
        }
    }

    if (box_tensors.size() != 3 || other_tensors.size() != 6)
        return false;

    std::sort(other_tensors.begin(), other_tensors.end(),
              [](const TensorInfo& a, const TensorInfo& b) { return a.channels < b.channels; });

    int small_ch = other_tensors[0].channels;
    int large_ch = other_tensors[3].channels;
    if (small_ch == large_ch || small_ch < 1 || large_ch % 3 != 0 || large_ch < 3)
        return false;

    for (int i = 0; i < 3; i++) {
        if (other_tensors[i].channels != small_ch)
            return false;
        if (other_tensors[i + 3].channels != large_ch)
            return false;
    }

    cls_ch = small_ch;
    kpt_ch = large_ch;

    std::vector<TensorInfo*> cls_tensors, kpt_tensors;
    for (auto& t : other_tensors) {
        if (t.channels == cls_ch)
            cls_tensors.push_back(&t);
        else
            kpt_tensors.push_back(&t);
    }

    auto cmp = [](const TensorInfo* a, const TensorInfo* b) { return a->grid_h * a->grid_w > b->grid_h * b->grid_w; };
    std::sort(box_tensors.begin(), box_tensors.end(),
              [](const TensorInfo& a, const TensorInfo& b) { return a.grid_h * a.grid_w > b.grid_h * b.grid_w; });
    std::sort(cls_tensors.begin(), cls_tensors.end(), cmp);
    std::sort(kpt_tensors.begin(), kpt_tensors.end(), cmp);

    for (int s = 0; s < 3; s++) {
        if (box_tensors[s].grid_h != cls_tensors[s]->grid_h || box_tensors[s].grid_w != cls_tensors[s]->grid_w)
            return false;
        if (box_tensors[s].grid_h != kpt_tensors[s]->grid_h || box_tensors[s].grid_w != kpt_tensors[s]->grid_w)
            return false;
    }

    for (int s = 0; s < 3; s++) {
        box_idx[s] = box_tensors[s].index;
        cls_idx[s] = cls_tensors[s]->index;
        kpt_idx[s] = kpt_tensors[s]->index;
    }

    return true;
}

Yolo11PoseSH::Yolo11PoseSH(Engine* p_engine_) : PoseDetector(p_engine_, "yolo11_pose_sh", MA_MODEL_TYPE_YOLO11_POSE_SH) {
    MA_ASSERT(p_engine_ != nullptr);

    int box_idx[3], cls_idx[3], kpt_idx[3];
    int cls_ch, kpt_ch;
    classifyOutputs(p_engine_, box_idx, cls_idx, kpt_idx, cls_ch, kpt_ch);

    for (int s = 0; s < 3; s++) {
        box_outputs_[s] = p_engine_->getOutput(box_idx[s]);
        cls_outputs_[s] = p_engine_->getOutput(cls_idx[s]);
        kpt_outputs_[s] = p_engine_->getOutput(kpt_idx[s]);
    }

    num_class_     = cls_ch;
    num_keypoints_ = kpt_ch / 3;
    num_record_    = 0;
    for (int s = 0; s < 3; s++)
        num_record_ += box_outputs_[s].shape.dims[2] * box_outputs_[s].shape.dims[3];
}

Yolo11PoseSH::~Yolo11PoseSH() {}

bool Yolo11PoseSH::isValid(Engine* engine) {
    const auto inputs_count  = engine->getInputSize();
    const auto outputs_count = engine->getOutputSize();

    if (inputs_count != 1 || outputs_count != 9)
        return false;

    const auto& input_shape = engine->getInputShape(0);
    if (input_shape.size != 4)
        return false;

    int n = input_shape.dims[0], h = input_shape.dims[1], w = input_shape.dims[2], c = input_shape.dims[3];
    bool is_nhwc = c == 3 || c == 1;
    if (!is_nhwc)
        std::swap(h, c);

    if (n != 1 || h < 32 || h % 32 != 0 || w < 32 || w % 32 != 0 || (c != 3 && c != 1))
        return false;

    int box_idx[3], cls_idx[3], kpt_idx[3];
    int cls_ch, kpt_ch;
    if (!classifyOutputs(engine, box_idx, cls_idx, kpt_idx, cls_ch, kpt_ch))
        return false;

    for (int s = 0; s < 3; s++) {
        auto box_shape = engine->getOutputShape(box_idx[s]);
        int expected_h = h >> (s + 3);
        int expected_w = w >> (s + 3);
        if (box_shape.dims[2] != expected_h || box_shape.dims[3] != expected_w)
            return false;
    }

    return true;
}

ma_err_t Yolo11PoseSH::postprocess() {
    results_.clear();
    if (box_outputs_[0].type == MA_TENSOR_TYPE_F32) {
        return postProcessF32();
    } else if (box_outputs_[0].type == MA_TENSOR_TYPE_S8) {
        return postProcessI8();
    }
    return MA_ENOTSUP;
}

ma_err_t Yolo11PoseSH::postProcessF32() {
    int dfl_len                             = box_outputs_[0].shape.dims[1] / 4;
    const float score_threshold_non_sigmoid = ma::math::inverseSigmoid(threshold_score_);

    std::forward_list<ma_bbox_ext_t> multi_level_bboxes;

    for (int i = 0; i < 3; i++) {
        int grid_h          = box_outputs_[i].shape.dims[2];
        int grid_w          = box_outputs_[i].shape.dims[3];
        int grid_l          = grid_h * grid_w;
        int stride          = img_.height / grid_h;
        float* output_score = cls_outputs_[i].data.f32;
        float* output_box   = box_outputs_[i].data.f32;

        for (int j = 0; j < grid_h; j++) {
            for (int k = 0; k < grid_w; k++) {
                int offset = j * grid_w + k;
                int target = -1;
                float max  = score_threshold_non_sigmoid;
                for (int c = 0; c < num_class_; c++) {
                    float score = output_score[offset + c * grid_l];
                    if (score < max) [[likely]]
                        continue;
                    max    = score;
                    target = c;
                }

                if (target < 0)
                    continue;

                float rect[4];
                float before_dfl[dfl_len * 4];
                int box_offset = j * grid_w + k;
                for (int b = 0; b < dfl_len * 4; b++) {
                    before_dfl[b] = output_box[box_offset];
                    box_offset += grid_l;
                }
                compute_dfl(before_dfl, dfl_len, rect);

                float x1 = (-rect[0] + k + 0.5) * stride;
                float y1 = (-rect[1] + j + 0.5) * stride;
                float x2 = (rect[2] + k + 0.5) * stride;
                float y2 = (rect[3] + j + 0.5) * stride;
                float bw = x2 - x1;
                float bh = y2 - y1;

                ma_bbox_ext_t bbox;
                bbox.level  = i;
                bbox.index  = j * grid_w + k;
                bbox.x      = (x1 + bw / 2.0) / img_.width;
                bbox.y      = (y1 + bh / 2.0) / img_.height;
                bbox.w      = bw / img_.width;
                bbox.h      = bh / img_.height;
                bbox.score  = ma::math::sigmoid(max);
                bbox.target = target;
                multi_level_bboxes.emplace_front(std::move(bbox));
            }
        }
    }

    ma::utils::nms(multi_level_bboxes, threshold_nms_, threshold_score_, false, true);

    if (multi_level_bboxes.empty())
        return MA_OK;

    std::vector<ma_pt3f_t> n_keypoint(num_keypoints_);

    for (auto& bbox : multi_level_bboxes) {
        int level   = bbox.level;
        int offset  = bbox.index;
        int grid_h  = kpt_outputs_[level].shape.dims[2];
        int grid_w  = kpt_outputs_[level].shape.dims[3];
        int grid_l  = grid_h * grid_w;
        int stride  = img_.height / grid_h;
        int j       = offset / grid_w;
        int k       = offset % grid_w;
        float* kpt_data = kpt_outputs_[level].data.f32;

        for (int p = 0; p < num_keypoints_; p++) {
            float raw_x = kpt_data[offset + (p * 3) * grid_l];
            float raw_y = kpt_data[offset + (p * 3 + 1) * grid_l];
            float raw_v = kpt_data[offset + (p * 3 + 2) * grid_l];
            n_keypoint[p].x = (raw_x * 2.0f + k) * stride / img_.width;
            n_keypoint[p].y = (raw_y * 2.0f + j) * stride / img_.height;
            n_keypoint[p].z = ma::math::sigmoid(raw_v);
        }

        ma_keypoint3f_t keypoint;
        keypoint.box = {.x = bbox.x, .y = bbox.y, .w = bbox.w, .h = bbox.h, .score = bbox.score, .target = bbox.target};
        keypoint.pts = n_keypoint;
        results_.emplace_front(std::move(keypoint));
    }

    return MA_OK;
}

ma_err_t Yolo11PoseSH::postProcessI8() {
    int dfl_len                             = box_outputs_[0].shape.dims[1] / 4;
    const float score_threshold_non_sigmoid = ma::math::inverseSigmoid(threshold_score_);

    std::forward_list<ma_bbox_ext_t> multi_level_bboxes;

    for (int i = 0; i < 3; i++) {
        int grid_h           = box_outputs_[i].shape.dims[2];
        int grid_w           = box_outputs_[i].shape.dims[3];
        int grid_l           = grid_h * grid_w;
        int stride           = img_.height / grid_h;
        int8_t* output_score = cls_outputs_[i].data.s8;
        int8_t* output_box   = box_outputs_[i].data.s8;

        for (int j = 0; j < grid_h; j++) {
            for (int k = 0; k < grid_w; k++) {
                int offset = j * grid_w + k;
                int target = -1;
                int8_t max_raw = -128;
                for (int c = 0; c < num_class_; c++) {
                    int8_t score = output_score[offset + c * grid_l];
                    if (score < max_raw) [[likely]]
                        continue;
                    max_raw = score;
                    target  = c;
                }

                if (target < 0)
                    continue;

                float score = ma::math::dequantizeValue(max_raw, cls_outputs_[i].quant_param.scale, cls_outputs_[i].quant_param.zero_point);

                if (score > score_threshold_non_sigmoid) {
                    float rect[4];
                    float before_dfl[dfl_len * 4];
                    int box_offset = j * grid_w + k;
                    for (int b = 0; b < dfl_len * 4; b++) {
                        before_dfl[b] = ma::math::dequantizeValue(output_box[box_offset], box_outputs_[i].quant_param.scale, box_outputs_[i].quant_param.zero_point);
                        box_offset += grid_l;
                    }
                    compute_dfl(before_dfl, dfl_len, rect);

                    float x1 = (-rect[0] + k + 0.5) * stride;
                    float y1 = (-rect[1] + j + 0.5) * stride;
                    float x2 = (rect[2] + k + 0.5) * stride;
                    float y2 = (rect[3] + j + 0.5) * stride;
                    float bw = x2 - x1;
                    float bh = y2 - y1;

                    ma_bbox_ext_t bbox;
                    bbox.level  = i;
                    bbox.index  = j * grid_w + k;
                    bbox.x      = (x1 + bw / 2.0) / img_.width;
                    bbox.y      = (y1 + bh / 2.0) / img_.height;
                    bbox.w      = bw / img_.width;
                    bbox.h      = bh / img_.height;
                    bbox.score  = ma::math::sigmoid(score);
                    bbox.target = target;
                    multi_level_bboxes.emplace_front(std::move(bbox));
                }
            }
        }
    }

    ma::utils::nms(multi_level_bboxes, threshold_nms_, threshold_score_, false, true);

    if (multi_level_bboxes.empty())
        return MA_OK;

    std::vector<ma_pt3f_t> n_keypoint(num_keypoints_);

    for (auto& bbox : multi_level_bboxes) {
        int level   = bbox.level;
        int offset  = bbox.index;
        int grid_h  = kpt_outputs_[level].shape.dims[2];
        int grid_w  = kpt_outputs_[level].shape.dims[3];
        int grid_l  = grid_h * grid_w;
        int stride  = img_.height / grid_h;
        int j       = offset / grid_w;
        int k       = offset % grid_w;
        int8_t* kpt_data = kpt_outputs_[level].data.s8;
        float kpt_scale  = kpt_outputs_[level].quant_param.scale;
        int kpt_zp       = kpt_outputs_[level].quant_param.zero_point;

        for (int p = 0; p < num_keypoints_; p++) {
            float raw_x = ma::math::dequantizeValue(kpt_data[offset + (p * 3) * grid_l], kpt_scale, kpt_zp);
            float raw_y = ma::math::dequantizeValue(kpt_data[offset + (p * 3 + 1) * grid_l], kpt_scale, kpt_zp);
            float raw_v = ma::math::dequantizeValue(kpt_data[offset + (p * 3 + 2) * grid_l], kpt_scale, kpt_zp);
            n_keypoint[p].x = (raw_x * 2.0f + k) * stride / img_.width;
            n_keypoint[p].y = (raw_y * 2.0f + j) * stride / img_.height;
            n_keypoint[p].z = ma::math::sigmoid(raw_v);
        }

        ma_keypoint3f_t keypoint;
        keypoint.box = {.x = bbox.x, .y = bbox.y, .w = bbox.w, .h = bbox.h, .score = bbox.score, .target = bbox.target};
        keypoint.pts = n_keypoint;
        results_.emplace_front(std::move(keypoint));
    }

    return MA_OK;
}

}  // namespace ma::model
