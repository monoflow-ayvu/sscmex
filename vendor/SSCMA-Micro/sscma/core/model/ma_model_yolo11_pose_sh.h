#ifndef _MA_MODEL_YOLO11_POSE_SH_H_
#define _MA_MODEL_YOLO11_POSE_SH_H_

#include <cstdint>
#include <vector>

#include "ma_model_pose_detector.h"

namespace ma::model {

class Yolo11PoseSH : public PoseDetector {
private:
    ma_tensor_t box_outputs_[3];
    ma_tensor_t cls_outputs_[3];
    ma_tensor_t kpt_outputs_[3];
    int32_t num_record_;
    int32_t num_class_;
    int32_t num_keypoints_;

protected:
    ma_err_t postprocess() override;

    ma_err_t postProcessI8();
    ma_err_t postProcessF32();

public:
    Yolo11PoseSH(Engine* engine);
    ~Yolo11PoseSH();

    static bool isValid(Engine* engine);
};

}  // namespace ma::model

#endif  // _MA_MODEL_YOLO11_POSE_SH_H_
