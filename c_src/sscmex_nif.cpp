#include <erl_nif.h>
#include <string>
#include <cstring>
#include <forward_list>
#include <limits>
#include <vector>
#include <time.h>

#include "nif_utils.hpp"
#include "sscma/core/engine/ma_engine_cvi.h"
#include "sscma/core/model/ma_model_factory.h"
#include "sscma/core/model/ma_model_classifier.h"
#include "sscma/core/model/ma_model_detector.h"
#include "sscma/core/model/ma_model_point_detector.h"
#include "sscma/core/model/ma_model_pose_detector.h"
#include "sscma/core/model/ma_model_segmentor.h"
#include "sscma/porting/ma_device.h"
#include "sscma/porting/ma_camera.h"
#include "ma_camera_sg200x.h"
#include "sscma_yolov7.h"
#include "app_ipcam_venc.h"
#include <cvi_isp.h>
#include <cvi_ae.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#define SSCMEX_FORMAT_WEBP 100

using namespace ma::engine;
using namespace ma::model;
using namespace ma;

// Forward declarations
struct TensorDataRes;
struct ModelRes;
struct DeviceRes;

// Instantiate template for EngineCVI
template<> ErlNifResourceType* NifRes<EngineCVI>::type = nullptr;

// Tensor data resource for zero-copy binary transfer
struct TensorDataRes {
    NifRes<EngineCVI>* engine_res;  // Reference to engine (keeps it alive)
    uint8_t* data;                   // Pointer to tensor data
    size_t size;                     // Size in bytes
};

// Instantiate template for TensorDataRes
template<> ErlNifResourceType* NifRes<TensorDataRes>::type = nullptr;

// Model resource - wraps ma::Model and keeps engine alive
struct ModelRes {
    Model* model;                        // The model instance
    NifRes<EngineCVI>* engine_res;       // Reference to engine (keeps it alive)
};

// Instantiate template for ModelRes
template<> ErlNifResourceType* NifRes<ModelRes>::type = nullptr;

// Device resource - wraps ma::Device singleton
struct DeviceRes {
    Device* device;  // Pointer to Device singleton
};

// Instantiate template for DeviceRes
template<> ErlNifResourceType* NifRes<DeviceRes>::type = nullptr;

// Camera resource - wraps ma::Camera
struct CameraRes {
    Camera* camera;  // Pointer to Camera instance
};

// Instantiate template for CameraRes
template<> ErlNifResourceType* NifRes<CameraRes>::type = nullptr;

// Helper to make atoms
static ERL_NIF_TERM make_atom(ErlNifEnv *env, const char *name) {
    return enif_make_atom(env, name);
}

// Helper to make ok tuple
static ERL_NIF_TERM make_ok(ErlNifEnv *env, ERL_NIF_TERM term) {
    return enif_make_tuple2(env, make_atom(env, "ok"), term);
}

// Helper to make error tuple
static ERL_NIF_TERM make_error(ErlNifEnv *env, const char *message) {
    return enif_make_tuple2(env, make_atom(env, "error"), enif_make_string(env, message, ERL_NIF_LATIN1));
}

// Helper to get string from binary (Elixir "string") or char list
static bool get_string_or_binary(ErlNifEnv *env, ERL_NIF_TERM term, char* buf, size_t buf_size) {
    // First try as binary (Elixir string literal)
    ErlNifBinary bin;
    if (enif_inspect_iolist_as_binary(env, term, &bin)) {
        if (bin.size >= buf_size) return false;
        memcpy(buf, bin.data, bin.size);
        buf[bin.size] = '\0';
        return true;
    }

    // Fallback to char list
    if (enif_get_string(env, term, buf, buf_size, ERL_NIF_LATIN1)) {
        return true;
    }

    return false;
}

// Helper: tensor type enum to atom
static ERL_NIF_TERM tensor_type_to_atom(ErlNifEnv* env, ma_tensor_type_t type) {
    switch (type) {
        case MA_TENSOR_TYPE_U8:   return make_atom(env, "u8");
        case MA_TENSOR_TYPE_S8:   return make_atom(env, "s8");
        case MA_TENSOR_TYPE_U16:  return make_atom(env, "u16");
        case MA_TENSOR_TYPE_S16:  return make_atom(env, "s16");
        case MA_TENSOR_TYPE_U32:  return make_atom(env, "u32");
        case MA_TENSOR_TYPE_S32:  return make_atom(env, "s32");
        case MA_TENSOR_TYPE_U64:  return make_atom(env, "u64");
        case MA_TENSOR_TYPE_S64:  return make_atom(env, "s64");
        case MA_TENSOR_TYPE_F16:  return make_atom(env, "f16");
        case MA_TENSOR_TYPE_F32:  return make_atom(env, "f32");
        case MA_TENSOR_TYPE_F64:  return make_atom(env, "f64");
        default:                  return make_atom(env, "none");
    }
}

// Helper: shape to list
static ERL_NIF_TERM shape_to_list(ErlNifEnv* env, const ma_shape_t& shape) {
    ERL_NIF_TERM* terms = (ERL_NIF_TERM*)enif_alloc(sizeof(ERL_NIF_TERM) * shape.size);
    for (uint32_t i = 0; i < shape.size; i++) {
        terms[i] = enif_make_int(env, shape.dims[i]);
    }
    ERL_NIF_TERM list = enif_make_list_from_array(env, terms, shape.size);
    enif_free(terms);
    return list;
}

// Helper: quant_param to map
static ERL_NIF_TERM quant_param_to_map(ErlNifEnv* env, const ma_quant_param_t& qp) {
    ERL_NIF_TERM map = enif_make_new_map(env);
    ERL_NIF_TERM scale_key = make_atom(env, "scale");
    ERL_NIF_TERM zp_key = make_atom(env, "zero_point");
    enif_make_map_put(env, map, scale_key, enif_make_double(env, qp.scale), &map);
    enif_make_map_put(env, map, zp_key, enif_make_int(env, qp.zero_point), &map);
    return map;
}

// Resource destructor - called when Elixir GC collects the resource
static void engine_cvi_dtor(ErlNifEnv* env, void* obj) {
    auto* res = static_cast<NifRes<EngineCVI>*>(obj);
    if (res->val) {
        delete res->val;
        res->val = nullptr;
    }
}

// Tensor data resource destructor - releases engine reference
static void tensor_data_dtor(ErlNifEnv* env, void* obj) {
    auto* res = static_cast<NifRes<TensorDataRes>*>(obj);
    if (res->val) {
        if (res->val->engine_res) {
            // Release our reference to the engine
            enif_release_resource(res->val->engine_res);
            res->val->engine_res = nullptr;
        }
        delete res->val;
        res->val = nullptr;
    }
}

// Model resource destructor - removes model and releases engine reference
static void model_dtor(ErlNifEnv* env, void* obj) {
    auto* res = static_cast<NifRes<ModelRes>*>(obj);
    if (res->val) {
        if (res->val->model) {
            ModelFactory::remove(res->val->model);
            res->val->model = nullptr;
        }
        if (res->val->engine_res) {
            enif_release_resource(res->val->engine_res);
            res->val->engine_res = nullptr;
        }
        delete res->val;
        res->val = nullptr;
    }
}

// Device resource destructor - Device is a singleton, no explicit cleanup needed
static void device_dtor(ErlNifEnv* env, void* obj) {
    auto* res = static_cast<NifRes<DeviceRes>*>(obj);
    if (res->val) {
        // Device is a singleton managed by the system, don't delete it
        res->val->device = nullptr;
        delete res->val;
        res->val = nullptr;
    }
}

// Camera resource destructor - Camera is managed by Device, no explicit cleanup
static void camera_dtor(ErlNifEnv* env, void* obj) {
    auto* res = static_cast<NifRes<CameraRes>*>(obj);
    if (res->val) {
        // Camera is managed externally (registered with Device)
        res->val->camera = nullptr;
        delete res->val;
        res->val = nullptr;
    }
}

// Helper: atom to pixel format
static ma_pixel_format_t atom_to_pixel_format(ErlNifEnv* env, ERL_NIF_TERM term) {
    char atom[32];
    if (!enif_get_atom(env, term, atom, sizeof(atom), ERL_NIF_LATIN1)) {
        return MA_PIXEL_FORMAT_UNKNOWN;
    }
    if (strcmp(atom, "rgb888") == 0) return MA_PIXEL_FORMAT_RGB888;
    if (strcmp(atom, "rgb565") == 0) return MA_PIXEL_FORMAT_RGB565;
    if (strcmp(atom, "yuv422") == 0) return MA_PIXEL_FORMAT_YUV422;
    if (strcmp(atom, "gray") == 0) return MA_PIXEL_FORMAT_GRAYSCALE;
    if (strcmp(atom, "jpeg") == 0) return MA_PIXEL_FORMAT_JPEG;
    if (strcmp(atom, "h264") == 0) return MA_PIXEL_FORMAT_H264;
    if (strcmp(atom, "h265") == 0) return MA_PIXEL_FORMAT_H265;
    if (strcmp(atom, "rgb888_planar") == 0) return MA_PIXEL_FORMAT_RGB888_PLANAR;
    return MA_PIXEL_FORMAT_UNKNOWN;
}

// Helper: pixel format to atom
static ERL_NIF_TERM pixel_format_to_atom(ErlNifEnv* env, ma_pixel_format_t format) {
    switch (format) {
        case MA_PIXEL_FORMAT_RGB888: return make_atom(env, "rgb888");
        case MA_PIXEL_FORMAT_RGB565: return make_atom(env, "rgb565");
        case MA_PIXEL_FORMAT_YUV422: return make_atom(env, "yuv422");
        case MA_PIXEL_FORMAT_GRAYSCALE: return make_atom(env, "gray");
        case MA_PIXEL_FORMAT_JPEG: return make_atom(env, "jpeg");
        case MA_PIXEL_FORMAT_H264: return make_atom(env, "h264");
        case MA_PIXEL_FORMAT_H265: return make_atom(env, "h265");
        case MA_PIXEL_FORMAT_RGB888_PLANAR: return make_atom(env, "rgb888_planar");
        default:                     return make_atom(env, "unknown");
    }
}

// Helper: model type to atom
static ERL_NIF_TERM model_type_to_atom(ErlNifEnv* env, ma_model_type_t type) {
    // YOLOv7 lives outside MA_MODEL_TYPE_* (kept out of the vendored enum so
    // we don't fork the submodule). It's the same numeric value the ctor
    // passes through to Detector — matched by uint here.
    if (static_cast<uint16_t>(type) == sscmex::YoloV7::kModelType) {
        return make_atom(env, "yolov7");
    }
    switch (type) {
        case MA_MODEL_TYPE_FOMO:       return make_atom(env, "fomo");
        case MA_MODEL_TYPE_YOLOV5:     return make_atom(env, "yolov5");
        case MA_MODEL_TYPE_YOLOV8:     return make_atom(env, "yolov8");
        case MA_MODEL_TYPE_YOLO11:     return make_atom(env, "yolo11");
        case MA_MODEL_TYPE_IMCLS:      return make_atom(env, "classifier");
        case MA_MODEL_TYPE_YOLOV8_POSE: return make_atom(env, "yolov8_pose");
        case MA_MODEL_TYPE_YOLO11_POSE: return make_atom(env, "yolo11_pose");
        case MA_MODEL_TYPE_YOLO11_SEG: return make_atom(env, "yolo11_seg");
        case MA_MODEL_TYPE_YOLO26:     return make_atom(env, "yolo26");
        default:                       return make_atom(env, "unknown");
    }
}

// Helper: input type to atom
static ERL_NIF_TERM input_type_to_atom(ErlNifEnv* env, ma_input_type_t type) {
    switch (type) {
        case MA_INPUT_TYPE_IMAGE:  return make_atom(env, "image");
        case MA_INPUT_TYPE_AUDIO:  return make_atom(env, "audio");
        case MA_INPUT_TYPE_TENSOR: return make_atom(env, "tensor");
        default:                   return make_atom(env, "unknown");
    }
}

// Helper: output type to atom
static ERL_NIF_TERM output_type_to_atom(ErlNifEnv* env, ma_output_type_t type) {
    switch (type) {
        case MA_OUTPUT_TYPE_TENSOR:    return make_atom(env, "tensor");
        case MA_OUTPUT_TYPE_BBOX:      return make_atom(env, "boxes");
        case MA_OUTPUT_TYPE_CLASS:     return make_atom(env, "classes");
        case MA_OUTPUT_TYPE_POINT:     return make_atom(env, "points");
        case MA_OUTPUT_TYPE_KEYPOINT:  return make_atom(env, "keypoints");
        case MA_OUTPUT_TYPE_SEGMENT:   return make_atom(env, "segments");
        default:                       return make_atom(env, "unknown");
    }
}

static bool is_compressed_format(ma_pixel_format_t format) {
    return format == MA_PIXEL_FORMAT_JPEG ||
           format == MA_PIXEL_FORMAT_H264 ||
           format == MA_PIXEL_FORMAT_H265;
}

static bool expected_raw_image_size(uint16_t width, uint16_t height, ma_pixel_format_t format, size_t* out_size) {
    if (!out_size || width == 0 || height == 0) {
        return false;
    }

    size_t bytes_per_pixel = 0;
    switch (format) {
        case MA_PIXEL_FORMAT_RGB888:
        case MA_PIXEL_FORMAT_RGB888_PLANAR:
            bytes_per_pixel = 3;
            break;
        case MA_PIXEL_FORMAT_RGB565:
        case MA_PIXEL_FORMAT_YUV422:
            bytes_per_pixel = 2;
            break;
        case MA_PIXEL_FORMAT_GRAYSCALE:
            bytes_per_pixel = 1;
            break;
        default:
            return false;
    }

    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (pixel_count > (std::numeric_limits<size_t>::max() / bytes_per_pixel)) {
        return false;
    }

    *out_size = pixel_count * bytes_per_pixel;
    return true;
}

static const char* validate_image_for_model_run(const ma_img_t& img, size_t binary_size) {
    if (img.width == 0 || img.height == 0) {
        return "image_dimensions_invalid";
    }

    if (img.data == nullptr) {
        return "image_data_missing";
    }

    if (img.size == 0) {
        return "image_size_invalid";
    }

    if (static_cast<size_t>(img.size) > binary_size) {
        return "image_size_exceeds_binary";
    }

    if (is_compressed_format(img.format)) {
        return "image_format_not_supported_for_inference";
    }

    size_t expected_size = 0;
    if (!expected_raw_image_size(img.width, img.height, img.format, &expected_size)) {
        return "image_format_not_supported";
    }

    if (expected_size != static_cast<size_t>(img.size)) {
        return "image_size_mismatch";
    }

    return nullptr;
}

// Helper: parse SSCMEx.Image struct from Elixir (zero-copy)
static bool get_image_struct(ErlNifEnv* env, ERL_NIF_TERM term, ma_img_t* img, ErlNifBinary* bin) {
    if (!img || !bin || !enif_is_map(env, term)) return false;

    *img = ma_img_t{};

    ERL_NIF_TERM width_term, height_term, format_term, data_term, size_term;

    // Get width
    if (!enif_get_map_value(env, term, make_atom(env, "width"), &width_term)) return false;
    int width_int;
    if (!enif_get_int(env, width_term, &width_int)) return false;
    if (width_int <= 0 || width_int > static_cast<int>(std::numeric_limits<uint16_t>::max())) return false;
    img->width = (uint16_t)width_int;

    // Get height
    if (!enif_get_map_value(env, term, make_atom(env, "height"), &height_term)) return false;
    int height_int;
    if (!enif_get_int(env, height_term, &height_int)) return false;
    if (height_int <= 0 || height_int > static_cast<int>(std::numeric_limits<uint16_t>::max())) return false;
    img->height = (uint16_t)height_int;

    // Get format
    if (!enif_get_map_value(env, term, make_atom(env, "format"), &format_term)) return false;
    img->format = atom_to_pixel_format(env, format_term);
    if (img->format == MA_PIXEL_FORMAT_UNKNOWN) return false;

    // Get data binary (zero-copy - just gets pointer to existing data)
    if (!enif_get_map_value(env, term, make_atom(env, "data"), &data_term)) return false;
    if (!enif_inspect_iolist_as_binary(env, data_term, bin)) return false;

    size_t declared_size = bin->size;
    if (enif_get_map_value(env, term, make_atom(env, "size"), &size_term)) {
        ErlNifUInt64 size_u64;
        if (!enif_get_uint64(env, size_term, &size_u64)) return false;
        if (size_u64 > static_cast<ErlNifUInt64>(std::numeric_limits<size_t>::max())) return false;
        declared_size = static_cast<size_t>(size_u64);
    }

    if (declared_size == 0 ||
        declared_size > bin->size ||
        declared_size > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }

    size_t expected_size = 0;
    if (expected_raw_image_size(img->width, img->height, img->format, &expected_size)) {
        if (expected_size > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) return false;
        if (declared_size != expected_size || bin->size != expected_size) return false;
    } else if (is_compressed_format(img->format)) {
        if (declared_size == 0 || declared_size > bin->size) return false;
    }

    img->size = static_cast<uint32_t>(declared_size);
    img->data = bin->data;

    // Set defaults for other fields
    img->rotate = MA_PIXEL_ROTATE_0;
    img->timestamp = 0;
    img->key = false;
    img->index = 0;
    img->count = 1;
    img->physical = false;

    return true;
}

// Helper: bbox to map
static ERL_NIF_TERM bbox_to_map(ErlNifEnv* env, const ma_bbox_t& bbox) {
    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "x"), enif_make_double(env, bbox.x), &map);
    enif_make_map_put(env, map, make_atom(env, "y"), enif_make_double(env, bbox.y), &map);
    enif_make_map_put(env, map, make_atom(env, "w"), enif_make_double(env, bbox.w), &map);
    enif_make_map_put(env, map, make_atom(env, "h"), enif_make_double(env, bbox.h), &map);
    enif_make_map_put(env, map, make_atom(env, "score"), enif_make_double(env, bbox.score), &map);
    enif_make_map_put(env, map, make_atom(env, "target"), enif_make_int(env, bbox.target), &map);
    return map;
}

// Helper: class result to map
static ERL_NIF_TERM class_to_map(ErlNifEnv* env, const ma_class_t& cls) {
    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "score"), enif_make_double(env, cls.score), &map);
    enif_make_map_put(env, map, make_atom(env, "target"), enif_make_int(env, cls.target), &map);
    return map;
}

// Helper: point result to map
static ERL_NIF_TERM point_to_map(ErlNifEnv* env, const ma_point_t& point) {
    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "x"), enif_make_double(env, point.x), &map);
    enif_make_map_put(env, map, make_atom(env, "y"), enif_make_double(env, point.y), &map);
    enif_make_map_put(env, map, make_atom(env, "score"), enif_make_double(env, point.score), &map);
    enif_make_map_put(env, map, make_atom(env, "target"), enif_make_int(env, point.target), &map);
    return map;
}

// Helper: 3D point to map
static ERL_NIF_TERM pt3f_to_map(ErlNifEnv* env, const ma_pt3f_t& point) {
    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "x"), enif_make_double(env, point.x), &map);
    enif_make_map_put(env, map, make_atom(env, "y"), enif_make_double(env, point.y), &map);
    enif_make_map_put(env, map, make_atom(env, "z"), enif_make_double(env, point.z), &map);
    return map;
}

// Helper: keypoint result to map
static ERL_NIF_TERM keypoint_to_map(ErlNifEnv* env, const ma_keypoint3f_t& keypoint) {
    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "box"), bbox_to_map(env, keypoint.box), &map);

    std::vector<ERL_NIF_TERM> point_terms;
    point_terms.reserve(keypoint.pts.size());
    for (const auto& point : keypoint.pts) {
        point_terms.push_back(pt3f_to_map(env, point));
    }

    ERL_NIF_TERM points_list =
        point_terms.empty() ? enif_make_list(env, 0) : enif_make_list_from_array(env, point_terms.data(), point_terms.size());
    enif_make_map_put(env, map, make_atom(env, "points"), points_list, &map);
    return map;
}

// Helper: segmentation result to map
static bool segment_to_map(ErlNifEnv* env, const ma_segm2f_t& segment, ERL_NIF_TERM* out_term) {
    if (!out_term) return false;

    ErlNifBinary mask_bin;
    if (!enif_alloc_binary(segment.mask.data.size(), &mask_bin)) {
        return false;
    }

    if (!segment.mask.data.empty()) {
        memcpy(mask_bin.data, segment.mask.data.data(), segment.mask.data.size());
    }

    ERL_NIF_TERM mask_map = enif_make_new_map(env);
    enif_make_map_put(env, mask_map, make_atom(env, "width"), enif_make_int(env, segment.mask.width), &mask_map);
    enif_make_map_put(env, mask_map, make_atom(env, "height"), enif_make_int(env, segment.mask.height), &mask_map);
    enif_make_map_put(env, mask_map, make_atom(env, "data"), enif_make_binary(env, &mask_bin), &mask_map);

    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "box"), bbox_to_map(env, segment.box), &map);
    enif_make_map_put(env, map, make_atom(env, "mask"), mask_map, &map);
    *out_term = map;
    return true;
}

// Helper: perf to map
static ERL_NIF_TERM perf_to_map(ErlNifEnv* env, const ma_perf_t& perf) {
    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "preprocess"), enif_make_int64(env, perf.preprocess), &map);
    enif_make_map_put(env, map, make_atom(env, "inference"), enif_make_int64(env, perf.inference), &map);
    enif_make_map_put(env, map, make_atom(env, "postprocess"), enif_make_int64(env, perf.postprocess), &map);
    return map;
}

// Helper: get OpenCV color conversion code for src_format -> dst_format
// Returns -1 if conversion not supported
static int get_cv_color_code(ma_pixel_format_t src_format, int dst_fmt) {
    // dst_fmt is either a ma_pixel_format_t or SSCMEX_FORMAT_WEBP
    // Compressed outputs are handled separately, not via cvtColor

    if (src_format == MA_PIXEL_FORMAT_RGB888) {
        switch (dst_fmt) {
            case MA_PIXEL_FORMAT_RGB565:    return -1; // handled via custom loop
            case MA_PIXEL_FORMAT_GRAYSCALE: return ::cv::COLOR_RGB2GRAY;
            case SSCMEX_FORMAT_WEBP:        return -1; // compressed, handled separately
            default: return -1;
        }
    } else if (src_format == MA_PIXEL_FORMAT_RGB565) {
        switch (dst_fmt) {
            case MA_PIXEL_FORMAT_RGB888:    return -1; // handled via custom loop
            case MA_PIXEL_FORMAT_GRAYSCALE: return -1; // handled via RGB888 intermediate
            default: return -1;
        }
    } else if (src_format == MA_PIXEL_FORMAT_YUV422) {
        switch (dst_fmt) {
            case MA_PIXEL_FORMAT_RGB888:    return ::cv::COLOR_YUV2RGB_YUYV;
            case MA_PIXEL_FORMAT_GRAYSCALE: return ::cv::COLOR_YUV2GRAY_YUYV;
            case MA_PIXEL_FORMAT_RGB565:    return -1; // via RGB888 intermediate
            default: return -1;
        }
    } else if (src_format == MA_PIXEL_FORMAT_GRAYSCALE) {
        switch (dst_fmt) {
            case MA_PIXEL_FORMAT_RGB888:    return ::cv::COLOR_GRAY2RGB;
            case MA_PIXEL_FORMAT_RGB565:    return -1; // via RGB888 intermediate
            default: return -1;
        }
    }
    return -1;
}

// Helper: get OpenCV interpolation method from atom
static int get_cv_interpolation(ErlNifEnv* env, ERL_NIF_TERM term) {
    char atom[32];
    if (!enif_get_atom(env, term, atom, sizeof(atom), ERL_NIF_LATIN1)) {
        return ::cv::INTER_LINEAR; // default
    }
    if (strcmp(atom, "nearest") == 0)     return ::cv::INTER_NEAREST;
    if (strcmp(atom, "linear") == 0)      return ::cv::INTER_LINEAR;
    if (strcmp(atom, "cubic") == 0)       return ::cv::INTER_CUBIC;
    if (strcmp(atom, "area") == 0)        return ::cv::INTER_AREA;
    if (strcmp(atom, "lanczos4") == 0)    return ::cv::INTER_LANCZOS4;
    return ::cv::INTER_LINEAR; // default
}

// Helper: parse format atom to integer (ma_pixel_format_t value or SSCMEX_FORMAT_WEBP)
static int parse_format_atom_cstr(const char* atom) {
    if (strcmp(atom, "rgb888") == 0)         return MA_PIXEL_FORMAT_RGB888;
    if (strcmp(atom, "rgb565") == 0)         return MA_PIXEL_FORMAT_RGB565;
    if (strcmp(atom, "yuv422") == 0)         return MA_PIXEL_FORMAT_YUV422;
    if (strcmp(atom, "gray") == 0)           return MA_PIXEL_FORMAT_GRAYSCALE;
    if (strcmp(atom, "grayscale") == 0)      return MA_PIXEL_FORMAT_GRAYSCALE;
    if (strcmp(atom, "jpeg") == 0)           return MA_PIXEL_FORMAT_JPEG;
    if (strcmp(atom, "webp") == 0)           return SSCMEX_FORMAT_WEBP;
    return -1;
}

// Helper: format integer (ma_pixel_format_t or SSCMEX_FORMAT_WEBP) to atom
static ERL_NIF_TERM format_int_to_atom(ErlNifEnv* env, int fmt) {
    switch (fmt) {
        case MA_PIXEL_FORMAT_RGB888:    return make_atom(env, "rgb888");
        case MA_PIXEL_FORMAT_RGB565:    return make_atom(env, "rgb565");
        case MA_PIXEL_FORMAT_YUV422:    return make_atom(env, "yuv422");
        case MA_PIXEL_FORMAT_GRAYSCALE: return make_atom(env, "gray");
        case MA_PIXEL_FORMAT_JPEG:      return make_atom(env, "jpeg");
        case SSCMEX_FORMAT_WEBP:        return make_atom(env, "webp");
        default:                        return make_atom(env, "unknown");
    }
}

// Helper: build an SSCMEx.Image struct return term from raw data
static ERL_NIF_TERM make_image_struct_ex(ErlNifEnv* env, int fmt,
                                          int width, int height,
                                          const uint8_t* data, size_t data_size) {
    ErlNifBinary out_bin;
    if (!enif_alloc_binary(data_size, &out_bin)) {
        return make_error(env, "allocation_failed");
    }
    if (data_size > 0) {
        memcpy(out_bin.data, data, data_size);
    }

    ERL_NIF_TERM map = enif_make_new_map(env);
    ERL_NIF_TERM struct_name;
    ERL_NIF_TERM struct_ns;
    enif_make_map_put(env, map, make_atom(env, "__struct__"),
                      enif_make_tuple2(env,
                          make_atom(env, "Elixir.SSCMEx.Image"),
                          make_atom(env, "SSCMEx.Image")),
                      &map);
    enif_make_map_put(env, map, make_atom(env, "width"),
                      enif_make_int(env, width), &map);
    enif_make_map_put(env, map, make_atom(env, "height"),
                      enif_make_int(env, height), &map);
    enif_make_map_put(env, map, make_atom(env, "format"),
                      format_int_to_atom(env, fmt), &map);
    enif_make_map_put(env, map, make_atom(env, "data"),
                      enif_make_binary(env, &out_bin), &map);
    enif_make_map_put(env, map, make_atom(env, "size"),
                      enif_make_uint64(env, data_size), &map);
    return map;
}

// NIF: image_convert - format conversion (RGB888/RGB565/YUV422/Grayscale <-> RGB888/RGB565/Grayscale/JPEG/WebP)
static ERL_NIF_TERM image_convert(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    // argv[0] = image struct, argv[1] = target format atom, argv[2] = quality (int, for JPEG/WebP)
    if (argc != 3) return enif_make_badarg(env);

    ma_img_t img;
    ErlNifBinary bin;
    if (!get_image_struct(env, argv[0], &img, &bin)) {
        return make_error(env, "invalid_image_struct");
    }

    // Parse target format
    char fmt_atom[32];
    if (!enif_get_atom(env, argv[1], fmt_atom, sizeof(fmt_atom), ERL_NIF_LATIN1)) {
        return make_error(env, "invalid_target_format");
    }
    int dst_fmt = parse_format_atom_cstr(fmt_atom);
    if (dst_fmt < 0) {
        return make_error(env, "unsupported_target_format");
    }

    // Parse quality (for JPEG/WebP)
    int quality = 85;
    int q_int;
    if (enif_get_int(env, argv[2], &q_int) && q_int > 0 && q_int <= 100) {
        quality = q_int;
    }

    ma_pixel_format_t src_fmt = img.format;

    // Same format -> return copy
    if (src_fmt == static_cast<ma_pixel_format_t>(dst_fmt)) {
        return make_ok(env, make_image_struct_ex(env, dst_fmt, img.width, img.height, img.data, img.size));
    }

    // --- Compressed output (JPEG, WebP) ---
    if (dst_fmt == MA_PIXEL_FORMAT_JPEG || dst_fmt == SSCMEX_FORMAT_WEBP) {
        // Build ::cv::Mat from source
        ::cv::Mat src_mat;
        if (src_fmt == MA_PIXEL_FORMAT_RGB888) {
            src_mat = ::cv::Mat(img.height, img.width, CV_8UC3, img.data);
        } else if (src_fmt == MA_PIXEL_FORMAT_GRAYSCALE) {
            src_mat = ::cv::Mat(img.height, img.width, CV_8UC1, img.data);
        } else if (src_fmt == MA_PIXEL_FORMAT_YUV422) {
            // YUV422 -> RGB888 first
            ::cv::Mat yuv_mat(img.height, img.width, CV_8UC2, img.data);
            ::cv::cvtColor(yuv_mat, src_mat, ::cv::COLOR_YUV2RGB_YUYV);
        } else if (src_fmt == MA_PIXEL_FORMAT_RGB565) {
            // RGB565 -> RGB888 via custom loop
            src_mat = ::cv::Mat(img.height, img.width, CV_8UC3);
            const uint16_t* src16 = reinterpret_cast<const uint16_t*>(img.data);
            for (int i = 0; i < img.height * img.width; i++) {
                uint16_t pixel = src16[i];
                src_mat.data[i * 3 + 0] = ((pixel >> 11) & 0x1F) * 255 / 31;
                src_mat.data[i * 3 + 1] = ((pixel >> 5) & 0x3F) * 255 / 63;
                src_mat.data[i * 3 + 2] = (pixel & 0x1F) * 255 / 31;
            }
        } else if (src_fmt == MA_PIXEL_FORMAT_JPEG) {
            // JPEG input - decode first, then re-encode to target format
            std::vector<uchar> jpeg_buf(img.data, img.data + img.size);
            src_mat = ::cv::imdecode(jpeg_buf, ::cv::IMREAD_COLOR); // Decode to RGB
            if (src_mat.empty()) {
                return make_error(env, "jpeg_decode_failed");
            }
            // OpenCV loads as BGR by default, convert to RGB
            ::cv::Mat rgb_mat;
            ::cv::cvtColor(src_mat, rgb_mat, ::cv::COLOR_BGR2RGB);
            src_mat = rgb_mat;
        } else {
            return make_error(env, "unsupported_source_format_for_compressed_output");
        }

        // Encode to JPEG or WebP
        std::vector<int> encode_params;
        std::string ext;
        if (dst_fmt == MA_PIXEL_FORMAT_JPEG) {
            ext = ".jpg";
            encode_params.push_back(::cv::IMWRITE_JPEG_QUALITY);
            encode_params.push_back(quality);
        } else {
            ext = ".webp";
            encode_params.push_back(::cv::IMWRITE_WEBP_QUALITY);
            encode_params.push_back(quality);
        }

        // OpenCV's imencode expects BGR for multi-channel images.
        // All input format branches above produce an RGB mat — convert before encoding.
        ::cv::Mat encode_mat;
        if (src_mat.channels() == 3) {
            ::cv::cvtColor(src_mat, encode_mat, ::cv::COLOR_RGB2BGR);
        } else {
            encode_mat = src_mat;  // grayscale: single channel, no swap needed
        }

        std::vector<uchar> encoded;
        if (!::cv::imencode(ext, encode_mat, encoded, encode_params)) {
            return make_error(env, "image_encoding_failed");
        }

        return make_ok(env, make_image_struct_ex(env, dst_fmt, img.width, img.height,
                                                  encoded.data(), encoded.size()));
    }

    // --- Compressed input (JPEG) to raw output ---
    if (src_fmt == MA_PIXEL_FORMAT_JPEG) {
        // Decode JPEG
        std::vector<uchar> jpeg_buf(img.data, img.data + img.size);
        ::cv::Mat decoded = ::cv::imdecode(jpeg_buf, ::cv::IMREAD_UNCHANGED);
        if (decoded.empty()) {
            return make_error(env, "jpeg_decode_failed");
        }

        int actual_width = decoded.cols;
        int actual_height = decoded.rows;

        // Convert channel count to target format
        ::cv::Mat converted;
        if (dst_fmt == MA_PIXEL_FORMAT_RGB888) {
            if (decoded.channels() == 1) {
                ::cv::cvtColor(decoded, converted, ::cv::COLOR_GRAY2RGB);
            } else if (decoded.channels() == 4) {
                ::cv::cvtColor(decoded, converted, ::cv::COLOR_BGRA2RGB);
            } else if (decoded.channels() == 3) {
                // OpenCV decodes JPEG as BGR by default
                ::cv::cvtColor(decoded, converted, ::cv::COLOR_BGR2RGB);
            } else {
                converted = decoded;
            }
        } else if (dst_fmt == MA_PIXEL_FORMAT_GRAYSCALE) {
            if (decoded.channels() == 3) {
                ::cv::cvtColor(decoded, converted, ::cv::COLOR_BGR2GRAY);
            } else if (decoded.channels() == 4) {
                ::cv::cvtColor(decoded, converted, ::cv::COLOR_BGRA2GRAY);
            } else {
                converted = decoded;
            }
        } else if (dst_fmt == MA_PIXEL_FORMAT_RGB565) {
            // Go to RGB888 first
            ::cv::Mat rgb;
            if (decoded.channels() == 1) {
                ::cv::cvtColor(decoded, rgb, ::cv::COLOR_GRAY2RGB);
            } else if (decoded.channels() == 4) {
                ::cv::cvtColor(decoded, rgb, ::cv::COLOR_BGRA2RGB);
            } else if (decoded.channels() == 3) {
                ::cv::cvtColor(decoded, rgb, ::cv::COLOR_BGR2RGB);
            } else {
                rgb = decoded;
            }
            // RGB888 -> RGB565
            size_t out_size = static_cast<size_t>(actual_width) * actual_height * 2;
            std::vector<uint8_t> rgb565(out_size);
            for (int i = 0; i < actual_width * actual_height; i++) {
                uint8_t r = rgb.data[i * 3 + 0];
                uint8_t g = rgb.data[i * 3 + 1];
                uint8_t b = rgb.data[i * 3 + 2];
                uint16_t pixel = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
                rgb565[i * 2 + 0] = pixel & 0xFF;
                rgb565[i * 2 + 1] = (pixel >> 8) & 0xFF;
            }
            return make_ok(env, make_image_struct_ex(env, dst_fmt, actual_width, actual_height,
                                                      rgb565.data(), out_size));
        } else {
            return make_error(env, "unsupported_jpeg_target_format");
        }

        if (converted.isContinuous()) {
            return make_ok(env, make_image_struct_ex(env, dst_fmt, actual_width, actual_height,
                                                      converted.data, converted.total() * converted.elemSize()));
        } else {
            // Clone to ensure continuous memory
            ::cv::Mat continuous = converted.clone();
            return make_ok(env, make_image_struct_ex(env, dst_fmt, actual_width, actual_height,
                                                      continuous.data, continuous.total() * continuous.elemSize()));
        }
    }

    // --- Raw-to-raw conversion ---

    // Build source ::cv::Mat
    ::cv::Mat src_mat;
    if (src_fmt == MA_PIXEL_FORMAT_RGB888) {
        src_mat = ::cv::Mat(img.height, img.width, CV_8UC3, img.data);
    } else if (src_fmt == MA_PIXEL_FORMAT_RGB565) {
        src_mat = ::cv::Mat(img.height, img.width, CV_8UC2, img.data);
    } else if (src_fmt == MA_PIXEL_FORMAT_YUV422) {
        src_mat = ::cv::Mat(img.height, img.width, CV_8UC2, img.data);
    } else if (src_fmt == MA_PIXEL_FORMAT_GRAYSCALE) {
        src_mat = ::cv::Mat(img.height, img.width, CV_8UC1, img.data);
    } else {
        return make_error(env, "unsupported_source_format");
    }

    // Try OpenCV color conversion first
    int cv_code = get_cv_color_code(src_fmt, dst_fmt);
    if (cv_code >= 0) {
        ::cv::Mat converted;
        ::cv::cvtColor(src_mat, converted, cv_code);
        if (converted.isContinuous()) {
            return make_ok(env, make_image_struct_ex(env, dst_fmt, img.width, img.height,
                                                      converted.data, converted.total() * converted.elemSize()));
        } else {
            ::cv::Mat continuous = converted.clone();
            return make_ok(env, make_image_struct_ex(env, dst_fmt, img.width, img.height,
                                                      continuous.data, continuous.total() * continuous.elemSize()));
        }
    }

    // RGB565 -> RGB888 (custom loop)
    if (src_fmt == MA_PIXEL_FORMAT_RGB565 && dst_fmt == MA_PIXEL_FORMAT_RGB888) {
        ::cv::Mat dst_mat(img.height, img.width, CV_8UC3);
        const uint16_t* src16 = reinterpret_cast<const uint16_t*>(img.data);
        for (int i = 0; i < img.height * img.width; i++) {
            uint16_t pixel = src16[i];
            dst_mat.data[i * 3 + 0] = ((pixel >> 11) & 0x1F) * 255 / 31;
            dst_mat.data[i * 3 + 1] = ((pixel >> 5) & 0x3F) * 255 / 63;
            dst_mat.data[i * 3 + 2] = (pixel & 0x1F) * 255 / 31;
        }
        return make_ok(env, make_image_struct_ex(env, dst_fmt, img.width, img.height,
                                                  dst_mat.data, dst_mat.total() * dst_mat.elemSize()));
    }

    // RGB888 -> RGB565 (custom loop)
    if (src_fmt == MA_PIXEL_FORMAT_RGB888 && dst_fmt == MA_PIXEL_FORMAT_RGB565) {
        size_t out_size = static_cast<size_t>(img.width) * img.height * 2;
        std::vector<uint8_t> rgb565(out_size);
        for (int i = 0; i < img.width * img.height; i++) {
            uint8_t r = img.data[i * 3 + 0];
            uint8_t g = img.data[i * 3 + 1];
            uint8_t b = img.data[i * 3 + 2];
            uint16_t pixel = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
            rgb565[i * 2 + 0] = pixel & 0xFF;
            rgb565[i * 2 + 1] = (pixel >> 8) & 0xFF;
        }
        return make_ok(env, make_image_struct_ex(env, dst_fmt, img.width, img.height,
                                                  rgb565.data(), out_size));
    }

    // Multi-step conversions via RGB888 intermediate
    // YUV422 -> RGB565: YUV422 -> RGB888 -> RGB565
    if (src_fmt == MA_PIXEL_FORMAT_YUV422 && dst_fmt == MA_PIXEL_FORMAT_RGB565) {
        ::cv::Mat rgb_mat;
        ::cv::cvtColor(src_mat, rgb_mat, ::cv::COLOR_YUV2RGB_YUYV);
        size_t out_size = static_cast<size_t>(img.width) * img.height * 2;
        std::vector<uint8_t> rgb565(out_size);
        for (int i = 0; i < img.width * img.height; i++) {
            uint8_t r = rgb_mat.data[i * 3 + 0];
            uint8_t g = rgb_mat.data[i * 3 + 1];
            uint8_t b = rgb_mat.data[i * 3 + 2];
            uint16_t pixel = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
            rgb565[i * 2 + 0] = pixel & 0xFF;
            rgb565[i * 2 + 1] = (pixel >> 8) & 0xFF;
        }
        return make_ok(env, make_image_struct_ex(env, dst_fmt, img.width, img.height,
                                                  rgb565.data(), out_size));
    }

    // RGB565 -> Grayscale: RGB565 -> RGB888 -> Grayscale
    if (src_fmt == MA_PIXEL_FORMAT_RGB565 && dst_fmt == MA_PIXEL_FORMAT_GRAYSCALE) {
        ::cv::Mat rgb_mat(img.height, img.width, CV_8UC3);
        const uint16_t* src16 = reinterpret_cast<const uint16_t*>(img.data);
        for (int i = 0; i < img.height * img.width; i++) {
            uint16_t pixel = src16[i];
            rgb_mat.data[i * 3 + 0] = ((pixel >> 11) & 0x1F) * 255 / 31;
            rgb_mat.data[i * 3 + 1] = ((pixel >> 5) & 0x3F) * 255 / 63;
            rgb_mat.data[i * 3 + 2] = (pixel & 0x1F) * 255 / 31;
        }
        ::cv::Mat gray_mat;
        ::cv::cvtColor(rgb_mat, gray_mat, ::cv::COLOR_RGB2GRAY);
        return make_ok(env, make_image_struct_ex(env, dst_fmt, img.width, img.height,
                                                  gray_mat.data, gray_mat.total() * gray_mat.elemSize()));
    }

    // Grayscale -> RGB565: Grayscale -> RGB888 -> RGB565
    if (src_fmt == MA_PIXEL_FORMAT_GRAYSCALE && dst_fmt == MA_PIXEL_FORMAT_RGB565) {
        ::cv::Mat rgb_mat;
        ::cv::cvtColor(src_mat, rgb_mat, ::cv::COLOR_GRAY2RGB);
        size_t out_size = static_cast<size_t>(img.width) * img.height * 2;
        std::vector<uint8_t> rgb565(out_size);
        for (int i = 0; i < img.width * img.height; i++) {
            uint8_t r = rgb_mat.data[i * 3 + 0];
            uint8_t g = rgb_mat.data[i * 3 + 1];
            uint8_t b = rgb_mat.data[i * 3 + 2];
            uint16_t pixel = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
            rgb565[i * 2 + 0] = pixel & 0xFF;
            rgb565[i * 2 + 1] = (pixel >> 8) & 0xFF;
        }
        return make_ok(env, make_image_struct_ex(env, dst_fmt, img.width, img.height,
                                                  rgb565.data(), out_size));
    }

    return make_error(env, "unsupported_conversion");
}

// NIF: image_resize - resize with interpolation method selection
static ERL_NIF_TERM image_resize(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    // argv[0] = image struct, argv[1] = {width, height} tuple, argv[2] = interpolation atom
    if (argc != 3) return enif_make_badarg(env);

    ma_img_t img;
    ErlNifBinary bin;
    if (!get_image_struct(env, argv[0], &img, &bin)) {
        return make_error(env, "invalid_image_struct");
    }

    // Parse target size tuple
    if (!enif_is_tuple(env, argv[1])) {
        return make_error(env, "invalid_size_tuple");
    }
    const ERL_NIF_TERM* tuple_elems;
    int tuple_arity;
    if (!enif_get_tuple(env, argv[1], &tuple_arity, &tuple_elems) || tuple_arity != 2) {
        return make_error(env, "invalid_size_tuple");
    }
    int new_width, new_height;
    if (!enif_get_int(env, tuple_elems[0], &new_width) || new_width <= 0) {
        return make_error(env, "invalid_width");
    }
    if (!enif_get_int(env, tuple_elems[1], &new_height) || new_height <= 0) {
        return make_error(env, "invalid_height");
    }

    // Get interpolation method
    int interpolation = get_cv_interpolation(env, argv[2]);

    // Build ::cv::Mat from source data
    ::cv::Mat src_mat;
    if (img.format == MA_PIXEL_FORMAT_RGB888) {
        src_mat = ::cv::Mat(img.height, img.width, CV_8UC3, img.data);
    } else if (img.format == MA_PIXEL_FORMAT_RGB565 || img.format == MA_PIXEL_FORMAT_YUV422) {
        src_mat = ::cv::Mat(img.height, img.width, CV_8UC2, img.data);
    } else if (img.format == MA_PIXEL_FORMAT_GRAYSCALE) {
        src_mat = ::cv::Mat(img.height, img.width, CV_8UC1, img.data);
    } else if (img.format == MA_PIXEL_FORMAT_JPEG) {
        // Decode JPEG first
        std::vector<uchar> jpeg_buf(img.data, img.data + img.size);
        src_mat = ::cv::imdecode(jpeg_buf, ::cv::IMREAD_UNCHANGED);
        if (src_mat.empty()) {
            return make_error(env, "jpeg_decode_failed");
        }
    } else {
        return make_error(env, "unsupported_source_format");
    }

    // Resize
    ::cv::Mat resized;
    ::cv::resize(src_mat, resized, ::cv::Size(new_width, new_height), 0, 0, interpolation);

    if (!resized.isContinuous()) {
        resized = resized.clone();
    }

    // For JPEG input, re-encode as JPEG
    if (img.format == MA_PIXEL_FORMAT_JPEG) {
        std::vector<int> encode_params;
        encode_params.push_back(::cv::IMWRITE_JPEG_QUALITY);
        encode_params.push_back(85);
        std::vector<uchar> encoded;
        if (!::cv::imencode(".jpg", resized, encoded, encode_params)) {
            return make_error(env, "jpeg_encode_failed");
        }
        return make_ok(env, make_image_struct_ex(env, MA_PIXEL_FORMAT_JPEG, new_width, new_height,
                                                  encoded.data(), encoded.size()));
    }

    size_t out_size = resized.total() * resized.elemSize();
    return make_ok(env, make_image_struct_ex(env, img.format, new_width, new_height,
                                              resized.data, out_size));
}

// NIF: Create new engine (uninitialized)
static ERL_NIF_TERM engine_cvi_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::allocate(env);
    if (!res) return make_error(env, "allocation_failed");

    res->val = new EngineCVI();

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return make_ok(env, term);
}

// NIF: Initialize engine
static ERL_NIF_TERM engine_cvi_init(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    ma_err_t err = res->val->init();
    return err == MA_OK ? make_ok(env, make_atom(env, "initialized"))
                        : make_error(env, "init_failed");
}

// NIF: Load model from file path
static ERL_NIF_TERM engine_cvi_load(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    // Get path string (supports both binary and char list)
    char path[1024];
    if (!get_string_or_binary(env, argv[1], path, sizeof(path))) {
        return make_error(env, "invalid_path");
    }

    ma_err_t err = res->val->load(path);
    return err == MA_OK ? make_ok(env, make_atom(env, "loaded"))
                        : make_error(env, "load_failed");
}

// NIF: Get input tensor count
static ERL_NIF_TERM engine_cvi_get_input_size(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");
    return make_ok(env, enif_make_int(env, res->val->getInputSize()));
}

// NIF: Get output tensor count
static ERL_NIF_TERM engine_cvi_get_output_size(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");
    return make_ok(env, enif_make_int(env, res->val->getOutputSize()));
}

// NIF: Get input tensor (returns map with binary data)
static ERL_NIF_TERM engine_cvi_get_input(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    int32_t index;
    if (!enif_get_int(env, argv[1], &index)) return make_error(env, "invalid_index");

    ma_tensor_t tensor = res->val->getInput(index);

    // Create binary from tensor data
    ErlNifBinary bin;
    if (!enif_alloc_binary(tensor.size, &bin)) {
        return make_error(env, "binary_allocation_failed");
    }
    memcpy(bin.data, tensor.data.u8, tensor.size);

    // Build result map
    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "shape"), shape_to_list(env, tensor.shape), &map);
    enif_make_map_put(env, map, make_atom(env, "type"), tensor_type_to_atom(env, tensor.type), &map);
    enif_make_map_put(env, map, make_atom(env, "size"), enif_make_uint64(env, tensor.size), &map);
    enif_make_map_put(env, map, make_atom(env, "quant_param"), quant_param_to_map(env, tensor.quant_param), &map);
    enif_make_map_put(env, map, make_atom(env, "data"), enif_make_binary(env, &bin), &map);
    enif_make_map_put(env, map, make_atom(env, "name"),
        enif_make_string(env, tensor.name ? tensor.name : "", ERL_NIF_LATIN1), &map);
    enif_make_map_put(env, map, make_atom(env, "is_physical"),
        tensor.is_physical ? make_atom(env, "true") : make_atom(env, "false"), &map);

    return make_ok(env, map);
}

// NIF: Get output tensor (returns map with binary data - ZERO COPY)
static ERL_NIF_TERM engine_cvi_get_output(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* engine_res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!engine_res || !engine_res->val) return make_error(env, "invalid_resource");

    int32_t index;
    if (!enif_get_int(env, argv[1], &index)) return make_error(env, "invalid_index");

    ma_tensor_t tensor = engine_res->val->getOutput(index);

    // Allocate tensor data resource for zero-copy
    auto* tensor_res = NifRes<TensorDataRes>::allocate(env);
    if (!tensor_res) return make_error(env, "allocation_failed");

    tensor_res->val = new TensorDataRes();
    tensor_res->val->engine_res = engine_res;
    tensor_res->val->data = tensor.data.u8;
    tensor_res->val->size = tensor.size;

    // Keep engine alive while tensor data is in use
    enif_keep_resource(engine_res);

    // Create zero-copy binary backed by resource
    ERL_NIF_TERM data_bin = enif_make_resource_binary(
        env,
        tensor_res,
        tensor.data.u8,
        tensor.size
    );

    // Build result map
    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "shape"), shape_to_list(env, tensor.shape), &map);
    enif_make_map_put(env, map, make_atom(env, "type"), tensor_type_to_atom(env, tensor.type), &map);
    enif_make_map_put(env, map, make_atom(env, "size"), enif_make_uint64(env, tensor.size), &map);
    enif_make_map_put(env, map, make_atom(env, "quant_param"), quant_param_to_map(env, tensor.quant_param), &map);
    enif_make_map_put(env, map, make_atom(env, "data"), data_bin, &map);
    enif_make_map_put(env, map, make_atom(env, "name"),
        enif_make_string(env, tensor.name ? tensor.name : "", ERL_NIF_LATIN1), &map);
    enif_make_map_put(env, map, make_atom(env, "is_physical"),
        tensor.is_physical ? make_atom(env, "true") : make_atom(env, "false"), &map);

    // Release our reference - binary keeps it alive via resource
    enif_release_resource(tensor_res);

    return make_ok(env, map);
}

// NIF: Get input tensor shape
static ERL_NIF_TERM engine_cvi_get_input_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    int32_t index;
    if (!enif_get_int(env, argv[1], &index)) return make_error(env, "invalid_index");

    ma_shape_t shape = res->val->getInputShape(index);
    return make_ok(env, shape_to_list(env, shape));
}

// NIF: Get output tensor shape
static ERL_NIF_TERM engine_cvi_get_output_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    int32_t index;
    if (!enif_get_int(env, argv[1], &index)) return make_error(env, "invalid_index");

    ma_shape_t shape = res->val->getOutputShape(index);
    return make_ok(env, shape_to_list(env, shape));
}

// NIF: Get input quantization parameters
static ERL_NIF_TERM engine_cvi_get_input_quant_param(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    int32_t index;
    if (!enif_get_int(env, argv[1], &index)) return make_error(env, "invalid_index");

    ma_quant_param_t qp = res->val->getInputQuantParam(index);
    return make_ok(env, quant_param_to_map(env, qp));
}

// NIF: Get output quantization parameters
static ERL_NIF_TERM engine_cvi_get_output_quant_param(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    int32_t index;
    if (!enif_get_int(env, argv[1], &index)) return make_error(env, "invalid_index");

    ma_quant_param_t qp = res->val->getOutputQuantParam(index);
    return make_ok(env, quant_param_to_map(env, qp));
}

// NIF: Set input tensor data
static ERL_NIF_TERM engine_cvi_set_input(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    int32_t index;
    if (!enif_get_int(env, argv[1], &index)) return make_error(env, "invalid_index");

    // Get binary data
    ErlNifBinary bin;
    if (!enif_inspect_iolist_as_binary(env, argv[2], &bin)) {
        return make_error(env, "invalid_data");
    }

    // Get the tensor and update its data
    ma_tensor_t tensor = res->val->getInput(index);
    if (bin.size != tensor.size) {
        return make_error(env, "data_size_mismatch");
    }
    memcpy(tensor.data.u8, bin.data, bin.size);

    return make_ok(env, make_atom(env, "ok"));
}

// NIF: Get input tensor index by name
static ERL_NIF_TERM engine_cvi_get_input_num(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    char name[256];
    if (!get_string_or_binary(env, argv[1], name, sizeof(name))) {
        return make_error(env, "invalid_name");
    }

    int32_t index = res->val->getInputNum(name);
    return make_ok(env, enif_make_int(env, index));
}

// NIF: Get output tensor index by name
static ERL_NIF_TERM engine_cvi_get_output_num(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    char name[256];
    if (!get_string_or_binary(env, argv[1], name, sizeof(name))) {
        return make_error(env, "invalid_name");
    }

    int32_t index = res->val->getOutputNum(name);
    return make_ok(env, enif_make_int(env, index));
}

// NIF: Run inference (DIRTY CPU)
static ERL_NIF_TERM engine_cvi_run(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    ma_err_t err = res->val->run();
    return err == MA_OK ? make_ok(env, make_atom(env, "completed"))
                        : make_error(env, "run_failed");
}

// ============================================================================
// Model NIF Functions
// ============================================================================

// NIF: Create model from engine
static ERL_NIF_TERM model_create(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* engine_res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!engine_res || !engine_res->val) return make_error(env, "invalid_engine_resource");

    // Engine must have a model loaded first (Engine.load/2); otherwise no model type matches
    if (engine_res->val->getInputSize() == 0 || engine_res->val->getOutputSize() == 0) {
        return make_error(env, "no_model_loaded");
    }

    // Get algorithm_id (default to 0)
    size_t algorithm_id = 0;
    if (argc > 1) {
        int id;
        if (enif_get_int(env, argv[1], &id)) {
            algorithm_id = (size_t)id;
        }
    }

    // YOLOv7 isn't in the vendored ModelFactory; check it first so the
    // 3-raw-head cvimodel produced by scripts/yolov7_to_clean_onnx.py +
    // scripts/build_yolov7_cvimodel.sh is recognised here. Falls through
    // to the standard factory for anything else (yolov5/v8/v11/...).
    Model* model = nullptr;
    if (sscmex::YoloV7::isValid(engine_res->val)) {
        model = new sscmex::YoloV7(engine_res->val);
    } else {
        model = ModelFactory::create(engine_res->val, algorithm_id);
    }
    if (!model) return make_error(env, "model_create_failed");

    // Allocate model resource
    auto* model_res = NifRes<ModelRes>::allocate(env);
    if (!model_res) {
        ModelFactory::remove(model);
        return make_error(env, "allocation_failed");
    }

    model_res->val = new ModelRes();
    model_res->val->model = model;
    model_res->val->engine_res = engine_res;

    // Keep engine alive while model exists
    enif_keep_resource(engine_res);

    ERL_NIF_TERM term = enif_make_resource(env, model_res);
    enif_release_resource(model_res);
    return make_ok(env, term);
}

// NIF: Get model type
static ERL_NIF_TERM model_get_type(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<ModelRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->model) return make_error(env, "invalid_resource");
    return make_ok(env, model_type_to_atom(env, res->val->model->getType()));
}

// NIF: Get model name
static ERL_NIF_TERM model_get_name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<ModelRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->model) return make_error(env, "invalid_resource");
    const char* name = res->val->model->getName();
    return make_ok(env, enif_make_string(env, name ? name : "", ERL_NIF_LATIN1));
}

// NIF: Get model input type
static ERL_NIF_TERM model_get_input_type(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<ModelRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->model) return make_error(env, "invalid_resource");
    return make_ok(env, input_type_to_atom(env, res->val->model->getInputType()));
}

// NIF: Get model output type
static ERL_NIF_TERM model_get_output_type(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<ModelRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->model) return make_error(env, "invalid_resource");
    return make_ok(env, output_type_to_atom(env, res->val->model->getOutputType()));
}

// NIF: Run model inference on image (DIRTY CPU)
static ERL_NIF_TERM model_run(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<ModelRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->model) return make_error(env, "invalid_resource");

    // Parse image struct
    ma_img_t img;
    ErlNifBinary bin;
    if (!get_image_struct(env, argv[1], &img, &bin)) {
        return make_error(env, "invalid_image_struct");
    }

    if (const char* image_error = validate_image_for_model_run(img, bin.size)) {
        return make_error(env, image_error);
    }

    const ma_output_type_t output_type = res->val->model->getOutputType();
    switch (output_type) {
        case MA_OUTPUT_TYPE_BBOX: {
            Detector* detector = static_cast<Detector*>(res->val->model);
            ma_err_t err = detector->run(&img);
            if (err != MA_OK) return make_error(env, "inference_failed");

            const std::forward_list<ma_bbox_t>& results = detector->getResults();
            std::vector<ERL_NIF_TERM> terms;
            for (const auto& bbox : results) {
                terms.push_back(bbox_to_map(env, bbox));
            }

            ERL_NIF_TERM list = terms.empty() ? enif_make_list(env, 0) : enif_make_list_from_array(env, terms.data(), terms.size());
            return make_ok(env, list);
        }

        case MA_OUTPUT_TYPE_CLASS: {
            Classifier* classifier = static_cast<Classifier*>(res->val->model);
            ma_err_t err = classifier->run(&img);
            if (err != MA_OK) return make_error(env, "inference_failed");

            const std::forward_list<ma_class_t>& results = classifier->getResults();
            std::vector<ERL_NIF_TERM> terms;
            for (const auto& cls : results) {
                terms.push_back(class_to_map(env, cls));
            }

            ERL_NIF_TERM list = terms.empty() ? enif_make_list(env, 0) : enif_make_list_from_array(env, terms.data(), terms.size());
            return make_ok(env, list);
        }

        case MA_OUTPUT_TYPE_POINT: {
            PointDetector* point_detector = static_cast<PointDetector*>(res->val->model);
            ma_err_t err = point_detector->run(&img);
            if (err != MA_OK) return make_error(env, "inference_failed");

            const std::forward_list<ma_point_t>& results = point_detector->getResults();
            std::vector<ERL_NIF_TERM> terms;
            for (const auto& point : results) {
                terms.push_back(point_to_map(env, point));
            }

            ERL_NIF_TERM list = terms.empty() ? enif_make_list(env, 0) : enif_make_list_from_array(env, terms.data(), terms.size());
            return make_ok(env, list);
        }

        case MA_OUTPUT_TYPE_KEYPOINT: {
            PoseDetector* pose_detector = static_cast<PoseDetector*>(res->val->model);
            ma_err_t err = pose_detector->run(&img);
            if (err != MA_OK) return make_error(env, "inference_failed");

            const std::forward_list<ma_keypoint3f_t>& results = pose_detector->getResults();
            std::vector<ERL_NIF_TERM> terms;
            for (const auto& keypoint : results) {
                terms.push_back(keypoint_to_map(env, keypoint));
            }

            ERL_NIF_TERM list = terms.empty() ? enif_make_list(env, 0) : enif_make_list_from_array(env, terms.data(), terms.size());
            return make_ok(env, list);
        }

        case MA_OUTPUT_TYPE_SEGMENT: {
            Segmentor* segmentor = static_cast<Segmentor*>(res->val->model);
            ma_err_t err = segmentor->run(&img);
            if (err != MA_OK) return make_error(env, "inference_failed");

            const std::forward_list<ma_segm2f_t>& results = segmentor->getResults();
            std::vector<ERL_NIF_TERM> terms;
            for (const auto& segment : results) {
                ERL_NIF_TERM term;
                if (!segment_to_map(env, segment, &term)) {
                    return make_error(env, "binary_allocation_failed");
                }
                terms.push_back(term);
            }

            ERL_NIF_TERM list = terms.empty() ? enif_make_list(env, 0) : enif_make_list_from_array(env, terms.data(), terms.size());
            return make_ok(env, list);
        }

        default:
            return make_error(env, "unsupported_output_type_for_run");
    }
}

static bool parse_model_config_opt(const char* opt, ma_model_cfg_opt_t* cfg_opt) {
    if (!opt || !cfg_opt) return false;

    if (strcmp(opt, "threshold_score") == 0) {
        *cfg_opt = MA_MODEL_CFG_OPT_THRESHOLD;
        return true;
    }
    if (strcmp(opt, "threshold_nms") == 0) {
        *cfg_opt = MA_MODEL_CFG_OPT_NMS;
        return true;
    }
    return false;
}

// NIF: Set model config
static ERL_NIF_TERM model_set_config(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<ModelRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->model) return make_error(env, "invalid_resource");

    // Get config option atom
    char opt[64];
    if (!enif_get_atom(env, argv[1], opt, sizeof(opt), ERL_NIF_LATIN1)) {
        return make_error(env, "invalid_option");
    }

    // Get value (double)
    double value;
    if (!enif_get_double(env, argv[2], &value)) {
        return make_error(env, "invalid_value");
    }

    ma_model_cfg_opt_t cfg_opt;
    if (!parse_model_config_opt(opt, &cfg_opt)) {
        return make_error(env, "unknown_option");
    }

    ma_err_t err = res->val->model->setConfig(cfg_opt, value);
    return err == MA_OK ? make_ok(env, make_atom(env, "ok"))
                        : make_error(env, "config_failed");
}

// NIF: Get model config
static ERL_NIF_TERM model_get_config(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<ModelRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->model) return make_error(env, "invalid_resource");

    // Get config option atom
    char opt[64];
    if (!enif_get_atom(env, argv[1], opt, sizeof(opt), ERL_NIF_LATIN1)) {
        return make_error(env, "invalid_option");
    }

    ma_model_cfg_opt_t cfg_opt;
    if (!parse_model_config_opt(opt, &cfg_opt)) {
        return make_error(env, "unknown_option");
    }

    double value = 0.0;
    ma_err_t err = res->val->model->getConfig(cfg_opt, &value);
    return err == MA_OK ? make_ok(env, enif_make_double(env, value))
                        : make_error(env, "config_failed");
}

// NIF: Get model performance metrics
static ERL_NIF_TERM model_get_perf(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<ModelRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->model) return make_error(env, "invalid_resource");

    ma_perf_t perf = res->val->model->getPerf();
    return make_ok(env, perf_to_map(env, perf));
}

// ============================================================================
// Device NIF Functions
// ============================================================================

// NIF: Get Device singleton instance
static ERL_NIF_TERM device_get_instance(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<DeviceRes>::allocate(env);
    if (!res) return make_error(env, "allocation_failed");

    res->val = new DeviceRes();
    res->val->device = Device::getInstance();

    if (!res->val->device) {
        delete res->val;
        res->val = nullptr;
        return make_error(env, "device_not_available");
    }

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return make_ok(env, term);
}

// NIF: Get device info (name, id, version, boot_count)
static ERL_NIF_TERM device_get_info(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<DeviceRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->device) return make_error(env, "invalid_resource");

    Device* device = res->val->device;
    ERL_NIF_TERM map = enif_make_new_map(env);

    enif_make_map_put(env, map, make_atom(env, "name"),
        enif_make_string(env, device->name().c_str(), ERL_NIF_LATIN1), &map);
    enif_make_map_put(env, map, make_atom(env, "id"),
        enif_make_string(env, device->id().c_str(), ERL_NIF_LATIN1), &map);
    enif_make_map_put(env, map, make_atom(env, "version"),
        enif_make_string(env, device->version().c_str(), ERL_NIF_LATIN1), &map);
    enif_make_map_put(env, map, make_atom(env, "boot_count"),
        enif_make_uint64(env, device->bootCount()), &map);

    return make_ok(env, map);
}

// Helper: sensor type to atom
static ERL_NIF_TERM sensor_type_to_atom(ErlNifEnv* env, Sensor::Type type) {
    switch (type) {
        case Sensor::Type::kCamera:     return make_atom(env, "camera");
        case Sensor::Type::kMicrophone: return make_atom(env, "microphone");
        case Sensor::Type::kIMU:        return make_atom(env, "imu");
        default:                        return make_atom(env, "unknown");
    }
}

// NIF: Get sensors list
static ERL_NIF_TERM device_get_sensors(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<DeviceRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->device) return make_error(env, "invalid_resource");

    const std::vector<Sensor*>& sensors = res->val->device->getSensors();

    std::vector<ERL_NIF_TERM> sensor_terms;
    for (Sensor* sensor : sensors) {
        ERL_NIF_TERM sensor_map = enif_make_new_map(env);
        enif_make_map_put(env, sensor_map, make_atom(env, "id"),
            enif_make_uint64(env, sensor->getID()), &sensor_map);
        enif_make_map_put(env, sensor_map, make_atom(env, "type"),
            sensor_type_to_atom(env, sensor->getType()), &sensor_map);
        sensor_terms.push_back(sensor_map);
    }

    ERL_NIF_TERM list = enif_make_list_from_array(env, sensor_terms.data(), sensor_terms.size());
    return make_ok(env, list);
}

// NIF: Get models list (model metadata, not loaded models)
static ERL_NIF_TERM device_get_models(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<DeviceRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->device) return make_error(env, "invalid_resource");

    const std::vector<ma_model_t>& models = res->val->device->getModels();

    std::vector<ERL_NIF_TERM> model_terms;
    for (const ma_model_t& model : models) {
        ERL_NIF_TERM model_map = enif_make_new_map(env);
        enif_make_map_put(env, model_map, make_atom(env, "id"),
            enif_make_int(env, model.id), &model_map);
        enif_make_map_put(env, model_map, make_atom(env, "name"),
            enif_make_string(env, model.name ? (const char*)model.name : "", ERL_NIF_LATIN1), &model_map);
        enif_make_map_put(env, model_map, make_atom(env, "type"),
            model_type_to_atom(env, model.type), &model_map);
        enif_make_map_put(env, model_map, make_atom(env, "size"),
            enif_make_uint64(env, model.size), &model_map);
        enif_make_map_put(env, model_map, make_atom(env, "addr"),
            enif_make_uint64(env, (uint64_t)model.addr), &model_map);
        model_terms.push_back(model_map);
    }

    ERL_NIF_TERM list = enif_make_list_from_array(env, model_terms.data(), model_terms.size());
    return make_ok(env, list);
}

// ============================================================================
// Camera NIF Functions
// ============================================================================

// NIF: Get camera from device by index
static ERL_NIF_TERM camera_get(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* device_res = NifRes<DeviceRes>::get(env, argv[0]);
    if (!device_res || !device_res->val || !device_res->val->device) return make_error(env, "invalid_device");

    int index;
    if (!enif_get_int(env, argv[1], &index)) return make_error(env, "invalid_index");

    const std::vector<Sensor*>& sensors = device_res->val->device->getSensors();

    // Find camera sensors
    std::vector<Sensor*> cameras;
    for (Sensor* sensor : sensors) {
        if (sensor->getType() == Sensor::Type::kCamera) {
            cameras.push_back(sensor);
        }
    }

    if (index < 0 || (size_t)index >= cameras.size()) {
        return make_error(env, "camera_not_found");
    }

    Camera* camera = static_cast<Camera*>(cameras[index]);

    auto* res = NifRes<CameraRes>::allocate(env);
    if (!res) return make_error(env, "allocation_failed");

    res->val = new CameraRes();
    res->val->camera = camera;

    ERL_NIF_TERM term = enif_make_resource(env, res);
    enif_release_resource(res);
    return make_ok(env, term);
}

// NIF: Get camera count from device
static ERL_NIF_TERM camera_count(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* device_res = NifRes<DeviceRes>::get(env, argv[0]);
    if (!device_res || !device_res->val || !device_res->val->device) return make_error(env, "invalid_device");

    const std::vector<Sensor*>& sensors = device_res->val->device->getSensors();

    int count = 0;
    for (Sensor* sensor : sensors) {
        if (sensor->getType() == Sensor::Type::kCamera) {
            count++;
        }
    }

    return make_ok(env, enif_make_int(env, count));
}

// NIF: Initialize camera with preset
static ERL_NIF_TERM camera_init(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    int preset_idx;
    if (!enif_get_int(env, argv[1], &preset_idx)) return make_error(env, "invalid_preset");

    ma_err_t err = res->val->camera->init((size_t)preset_idx);
    return err == MA_OK ? make_ok(env, make_atom(env, "initialized"))
                        : make_error(env, "init_failed");
}

// NIF: Deinitialize camera
static ERL_NIF_TERM camera_deinit(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    res->val->camera->deInit();
    return make_ok(env, make_atom(env, "deinitialized"));
}

// NIF: Get available presets
static ERL_NIF_TERM camera_get_presets(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    const Sensor::Presets& presets = res->val->camera->availablePresets();

    std::vector<ERL_NIF_TERM> preset_terms;
    for (const auto& preset : presets) {
        ERL_NIF_TERM preset_map = enif_make_new_map(env);
        enif_make_map_put(env, preset_map, make_atom(env, "description"),
            enif_make_string(env, preset.description ? preset.description : "", ERL_NIF_LATIN1), &preset_map);
        preset_terms.push_back(preset_map);
    }

    ERL_NIF_TERM list = enif_make_list_from_array(env, preset_terms.data(), preset_terms.size());
    return make_ok(env, list);
}

// NIF: Get current preset index
static ERL_NIF_TERM camera_get_preset_idx(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    return make_ok(env, enif_make_uint64(env, res->val->camera->currentPresetIdx()));
}

// NIF: Check if camera is initialized
static ERL_NIF_TERM camera_is_initialized(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    return make_ok(env, *res->val->camera ? make_atom(env, "true") : make_atom(env, "false"));
}

// Helper: stream mode to atom
static ERL_NIF_TERM stream_mode_to_atom(ErlNifEnv* env, Camera::StreamMode mode) {
    switch (mode) {
        case Camera::StreamMode::kRefreshOnReturn:   return make_atom(env, "refresh_on_return");
        case Camera::StreamMode::kRefreshOnRetrieve: return make_atom(env, "refresh_on_retrieve");
        default:                                    return make_atom(env, "unknown");
    }
}

// Helper: atom to stream mode
static Camera::StreamMode atom_to_stream_mode(ErlNifEnv* env, ERL_NIF_TERM term) {
    char atom[64];
    if (!enif_get_atom(env, term, atom, sizeof(atom), ERL_NIF_LATIN1)) {
        return Camera::StreamMode::kUnknown;
    }
    if (strcmp(atom, "refresh_on_return") == 0) return Camera::StreamMode::kRefreshOnReturn;
    if (strcmp(atom, "refresh_on_retrieve") == 0) return Camera::StreamMode::kRefreshOnRetrieve;
    return Camera::StreamMode::kUnknown;
}

// NIF: Start camera stream (DIRTY IO)
static ERL_NIF_TERM camera_start_stream(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    Camera::StreamMode mode = atom_to_stream_mode(env, argv[1]);
    if (mode == Camera::StreamMode::kUnknown) return make_error(env, "invalid_mode");

    ma_err_t err = res->val->camera->startStream(mode);
    return err == MA_OK ? make_ok(env, make_atom(env, "streaming"))
                        : make_error(env, "start_stream_failed");
}

// NIF: Stop camera stream
static ERL_NIF_TERM camera_stop_stream(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    res->val->camera->stopStream();
    return make_ok(env, make_atom(env, "stopped"));
}

// NIF: Check if camera is streaming
static ERL_NIF_TERM camera_is_streaming(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    return make_ok(env, res->val->camera->isStreaming() ? make_atom(env, "true") : make_atom(env, "false"));
}

// Shared frame-retrieval helper. `mode` selects between the three
// queue-fetch behaviours (blocking / non-blocking / drain-to-latest).
// Keeps the binary-allocation + map-build path in one place so the
// three NIF entry points stay thin.
enum class RetrieveMode {
    Blocking,    // wait up to one frame interval (legacy retrieve_frame)
    NonBlocking, // return MA_AGAIN immediately if queue empty
    Latest,      // block for first frame, then drain to most recent
};

static ERL_NIF_TERM do_camera_retrieve(ErlNifEnv* env, const ERL_NIF_TERM argv[], RetrieveMode mode) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    int channel_idx;
    if (!enif_get_int(env, argv[1], &channel_idx))
        return make_error(env, "invalid_channel");
    if (channel_idx < 0 || channel_idx > 2)
        return make_error(env, "channel_out_of_range");

    ma_img_t frame;
    // CameraRes always holds a CameraSG200X on this hardware.
    auto* sg200x = static_cast<ma::CameraSG200X*>(res->val->camera);
    ma_err_t err = MA_OK;
    switch (mode) {
        case RetrieveMode::Blocking:    err = sg200x->retrieveChannel(frame, channel_idx); break;
        case RetrieveMode::NonBlocking: err = sg200x->tryRetrieveChannel(frame, channel_idx); break;
        case RetrieveMode::Latest:      err = sg200x->retrieveLatestChannel(frame, channel_idx); break;
    }
    if (err != MA_OK) {
        return make_error(env, err == MA_AGAIN ? "queue_empty" : "retrieve_frame_failed");
    }

    // Create binary from frame data (copy - frame ownership is with camera)
    ErlNifBinary bin;
    if (!enif_alloc_binary(frame.size, &bin)) {
        res->val->camera->returnFrame(frame);
        return make_error(env, "binary_allocation_failed");
    }
    memcpy(bin.data, frame.data, frame.size);

    // Return frame to camera
    res->val->camera->returnFrame(frame);

    // Build result map
    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "width"), enif_make_int(env, frame.width), &map);
    enif_make_map_put(env, map, make_atom(env, "height"), enif_make_int(env, frame.height), &map);
    enif_make_map_put(env, map, make_atom(env, "format"), pixel_format_to_atom(env, frame.format), &map);
    enif_make_map_put(env, map, make_atom(env, "size"), enif_make_uint64(env, frame.size), &map);
    enif_make_map_put(env, map, make_atom(env, "data"), enif_make_binary(env, &bin), &map);
    enif_make_map_put(env, map, make_atom(env, "timestamp"), enif_make_int64(env, frame.timestamp), &map);
    enif_make_map_put(env, map, make_atom(env, "key"), frame.key ? make_atom(env, "true") : make_atom(env, "false"), &map);

    return make_ok(env, map);
}

// NIF: Retrieve frame from camera (DIRTY IO)
static ERL_NIF_TERM camera_retrieve_frame(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    return do_camera_retrieve(env, argv, RetrieveMode::Blocking);
}

// NIF: Non-blocking retrieve. Returns `{:error, "queue_empty"}` if no
// frame is currently buffered. Useful for consumer-side draining
// without paying the per-call wait cost.
static ERL_NIF_TERM camera_try_retrieve_frame(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    return do_camera_retrieve(env, argv, RetrieveMode::NonBlocking);
}

// NIF: Drain-to-latest retrieve. Blocks for the first frame (up to
// one frame interval), then pulls everything else non-blocking and
// discards stale frames, returning only the most recent. Single
// NIF call → minimal Elixir↔NIF overhead and minimal end-to-end
// latency on consumers like a YOLO inference loop.
static ERL_NIF_TERM camera_retrieve_latest_frame(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    return do_camera_retrieve(env, argv, RetrieveMode::Latest);
}

// NIF: Set camera control
static bool parse_camera_ctrl_type(ErlNifEnv* env, ERL_NIF_TERM term, Camera::CtrlType* ctrl) {
    if (!ctrl) return false;

    char ctrl_atom[32];
    if (!enif_get_atom(env, term, ctrl_atom, sizeof(ctrl_atom), ERL_NIF_LATIN1)) {
        return false;
    }

    if (strcmp(ctrl_atom, "window") == 0) *ctrl = Camera::CtrlType::kWindow;
    else if (strcmp(ctrl_atom, "channel") == 0) *ctrl = Camera::CtrlType::kChannel;
    else if (strcmp(ctrl_atom, "format") == 0) *ctrl = Camera::CtrlType::kFormat;
    else if (strcmp(ctrl_atom, "fps") == 0) *ctrl = Camera::CtrlType::kFps;
    else return false;

    return true;
}

static ERL_NIF_TERM camera_set_ctrl(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    char ctrl_atom[32];
    if (!enif_get_atom(env, argv[1], ctrl_atom, sizeof(ctrl_atom), ERL_NIF_LATIN1)) {
        return make_error(env, "unsupported_ctrl");
    }

    if (strcmp(ctrl_atom, "quality") == 0) {
        int quality;
        if (!enif_get_int(env, argv[2], &quality)) {
            return make_error(env, "invalid_quality_value");
        }
        if (quality < 1 || quality == 50 || quality > 99) {
            return make_error(env, "quality_out_of_range");
        }

        Camera::CtrlValue chn_val{};
        res->val->camera->commandCtrl(Camera::kChannel, Camera::kRead, chn_val);
        int jpeg_ch = chn_val.i32;

        VENC_JPEG_PARAM_S jpeg_param{};
        CVI_S32 ret = CVI_VENC_GetJpegParam(jpeg_ch, &jpeg_param);
        if (ret != CVI_SUCCESS) {
            return make_error(env, "quality_unavailable");
        }
        jpeg_param.u32Qfactor = static_cast<CVI_U32>(quality);
        ret = CVI_VENC_SetJpegParam(jpeg_ch, &jpeg_param);
        return ret == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok"))
                                  : make_error(env, "set_quality_failed");
    }

    // --- AE Controls ---
    if (strcmp(ctrl_atom, "ae_mode") == 0) {
        char mode_atom[16];
        if (!enif_get_atom(env, argv[2], mode_atom, sizeof(mode_atom), ERL_NIF_LATIN1))
            return make_error(env, "invalid_mode");
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        if (strcmp(mode_atom, "auto") == 0) attr.enOpType = OP_TYPE_AUTO;
        else if (strcmp(mode_atom, "manual") == 0) attr.enOpType = OP_TYPE_MANUAL;
        else return make_error(env, "invalid_mode");
        return CVI_ISP_SetExposureAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_ae_mode_failed");
    }

    if (strcmp(ctrl_atom, "max_iso") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_iso");
        if (val < 100 || val > 12800) return make_error(env, "iso_out_of_range");
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        attr.stAuto.enGainType = AE_TYPE_ISO;
        attr.stAuto.stISONumRange.u32Max = static_cast<CVI_U32>(val);
        return CVI_ISP_SetExposureAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_max_iso_failed");
    }

    if (strcmp(ctrl_atom, "exposure_us") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_exposure");
        if (val < 1 || val > 1000000) return make_error(env, "exposure_out_of_range");
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        attr.enOpType = OP_TYPE_MANUAL;
        attr.stManual.enExpTimeOpType = OP_TYPE_MANUAL;
        attr.stManual.u32ExpTime = static_cast<CVI_U32>(val);
        return CVI_ISP_SetExposureAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_exposure_failed");
    }

    if (strcmp(ctrl_atom, "gain") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_gain");
        if (val < 1024 || val > 65536) return make_error(env, "gain_out_of_range");
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        attr.enOpType = OP_TYPE_MANUAL;
        attr.stManual.enAGainOpType = OP_TYPE_MANUAL;
        attr.stManual.u32AGain = static_cast<CVI_U32>(val);
        attr.stManual.u32DGain = 1024;
        attr.stManual.u32ISPDGain = 1024;
        return CVI_ISP_SetExposureAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_gain_failed");
    }

    if (strcmp(ctrl_atom, "exposure_range") == 0) {
        int arity;
        const ERL_NIF_TERM* tuple;
        if (!enif_get_tuple(env, argv[2], &arity, &tuple) || arity != 2)
            return make_error(env, "invalid_range_tuple");
        int min_us, max_us;
        if (!enif_get_int(env, tuple[0], &min_us) || !enif_get_int(env, tuple[1], &max_us))
            return make_error(env, "invalid_range_tuple");
        if (min_us < 1 || max_us < min_us || max_us > 1000000)
            return make_error(env, "exposure_out_of_range");
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        attr.enOpType = OP_TYPE_AUTO;
        attr.stAuto.stExpTimeRange.u32Min = static_cast<CVI_U32>(min_us);
        attr.stAuto.stExpTimeRange.u32Max = static_cast<CVI_U32>(max_us);
        return CVI_ISP_SetExposureAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_exposure_range_failed");
    }

    if (strcmp(ctrl_atom, "max_exposure_us") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_exposure");
        if (val < 1 || val > 1000000) return make_error(env, "exposure_out_of_range");
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        if (static_cast<CVI_U32>(val) < attr.stAuto.stExpTimeRange.u32Min)
            return make_error(env, "exposure_out_of_range");
        attr.stAuto.stExpTimeRange.u32Max = static_cast<CVI_U32>(val);
        return CVI_ISP_SetExposureAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_max_exposure_failed");
    }

    // --- TNR Controls ---
    if (strcmp(ctrl_atom, "tnr_enable") == 0) {
        char bool_atom[8];
        if (!enif_get_atom(env, argv[2], bool_atom, sizeof(bool_atom), ERL_NIF_LATIN1))
            return make_error(env, "invalid_boolean");
        ISP_TNR_ATTR_S attr;
        if (CVI_ISP_GetTNRAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "tnr_unavailable");
        if (strcmp(bool_atom, "true") == 0) attr.Enable = CVI_TRUE;
        else if (strcmp(bool_atom, "false") == 0) attr.Enable = CVI_FALSE;
        else return make_error(env, "invalid_boolean");
        return CVI_ISP_SetTNRAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_tnr_failed");
    }

    if (strcmp(ctrl_atom, "tnr_strength") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_strength");
        if (val < 0 || val > 255) return make_error(env, "strength_out_of_range");
        ISP_TNR_ATTR_S attr;
        if (CVI_ISP_GetTNRAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "tnr_unavailable");
        attr.enOpType = OP_TYPE_MANUAL;
        attr.stManual.TnrStrength0 = static_cast<CVI_U8>(val);
        return CVI_ISP_SetTNRAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_tnr_strength_failed");
    }

    // --- Image Tuning Controls ---
    // Brightness: via ISP YContrast module (CenterLuma in stManual)
    if (strcmp(ctrl_atom, "brightness") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_value");
        if (val < 0 || val > 255) return make_error(env, "value_out_of_range");
        ISP_YCONTRAST_ATTR_S attr;
        if (CVI_ISP_GetYContrastAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        attr.enOpType = OP_TYPE_MANUAL;
        attr.stManual.CenterLuma = static_cast<CVI_U8>(val);
        return CVI_ISP_SetYContrastAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_brightness_failed");
    }

    // Contrast: via ISP YContrast module (ContrastHigh in stManual)
    if (strcmp(ctrl_atom, "contrast") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_value");
        if (val < 0 || val > 255) return make_error(env, "value_out_of_range");
        ISP_YCONTRAST_ATTR_S attr;
        if (CVI_ISP_GetYContrastAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        attr.enOpType = OP_TYPE_MANUAL;
        attr.stManual.ContrastHigh = static_cast<CVI_U8>(val);
        return CVI_ISP_SetYContrastAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_contrast_failed");
    }

    // Saturation: via ISP Saturation module (stManual.Saturation)
    if (strcmp(ctrl_atom, "saturation") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_value");
        if (val < 0 || val > 255) return make_error(env, "value_out_of_range");
        ISP_SATURATION_ATTR_S attr;
        if (CVI_ISP_GetSaturationAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        attr.enOpType = OP_TYPE_MANUAL;
        attr.stManual.Saturation = static_cast<CVI_U8>(val);
        return CVI_ISP_SetSaturationAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_saturation_failed");
    }

    // Sharpness: via ISP Sharpen module (stManual.GlobalGain)
    if (strcmp(ctrl_atom, "sharpness") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_value");
        if (val < 0 || val > 255) return make_error(env, "value_out_of_range");
        ISP_SHARPEN_ATTR_S attr;
        if (CVI_ISP_GetSharpenAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        attr.enOpType = OP_TYPE_MANUAL;
        attr.stManual.GlobalGain = static_cast<CVI_U8>(val);
        return CVI_ISP_SetSharpenAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_sharpness_failed");
    }

    // Raw (Bayer) NR strength: via ISP NR module (stManual.NoiseSuppressStr)
    if (strcmp(ctrl_atom, "nr_strength") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_value");
        if (val < 0 || val > 255) return make_error(env, "value_out_of_range");
        ISP_NR_ATTR_S attr;
        if (CVI_ISP_GetNRAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        attr.enOpType = OP_TYPE_MANUAL;
        attr.stManual.NoiseSuppressStr = static_cast<CVI_U8>(val);
        return CVI_ISP_SetNRAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_nr_strength_failed");
    }

    // Luma NR strength: via ISP YNR module (stManual.NoiseSuppressStr)
    if (strcmp(ctrl_atom, "ynr_strength") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_value");
        if (val < 0 || val > 255) return make_error(env, "value_out_of_range");
        ISP_YNR_ATTR_S attr;
        if (CVI_ISP_GetYNRAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        attr.enOpType = OP_TYPE_MANUAL;
        attr.stManual.NoiseSuppressStr = static_cast<CVI_U8>(val);
        return CVI_ISP_SetYNRAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_ynr_strength_failed");
    }

    // Chroma NR strength: via ISP CNR module (stManual.NoiseSuppressStr)
    if (strcmp(ctrl_atom, "cnr_strength") == 0) {
        int val;
        if (!enif_get_int(env, argv[2], &val)) return make_error(env, "invalid_value");
        if (val < 0 || val > 255) return make_error(env, "value_out_of_range");
        ISP_CNR_ATTR_S attr;
        if (CVI_ISP_GetCNRAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        attr.enOpType = OP_TYPE_MANUAL;
        attr.stManual.NoiseSuppressStr = static_cast<CVI_U8>(val);
        return CVI_ISP_SetCNRAttr(0, &attr) == CVI_SUCCESS ? make_ok(env, make_atom(env, "ok")) : make_error(env, "set_cnr_strength_failed");
    }

    // --- VENC encoder parameters (H.264 / H.265 only, applied at startStream time) ---
    if (strcmp(ctrl_atom, "venc_params") == 0) {
        if (!enif_is_map(env, argv[2])) return make_error(env, "invalid_venc_params");

        Camera::CtrlValue chn_val{};
        if (res->val->camera->commandCtrl(Camera::kChannel, Camera::kRead, chn_val) != MA_OK)
            return make_error(env, "failed_to_read_channel");
        int ch = chn_val.i32;

        video_venc_params_t vp{};
        vp.has_venc_params = true;

        ERL_NIF_TERM mval;
        int iv;

        if (enif_get_map_value(env, argv[2], make_atom(env, "bitrate"), &mval) && enif_get_int(env, mval, &iv))
            vp.bitrate = (uint32_t)iv;
        if (enif_get_map_value(env, argv[2], make_atom(env, "max_bitrate"), &mval) && enif_get_int(env, mval, &iv))
            vp.max_bitrate = (uint32_t)iv;
        if (enif_get_map_value(env, argv[2], make_atom(env, "gop"), &mval) && enif_get_int(env, mval, &iv))
            vp.gop = (uint32_t)iv;
        if (enif_get_map_value(env, argv[2], make_atom(env, "min_qp"), &mval) && enif_get_int(env, mval, &iv))
            vp.min_qp = (uint32_t)iv;
        if (enif_get_map_value(env, argv[2], make_atom(env, "max_qp"), &mval) && enif_get_int(env, mval, &iv))
            vp.max_qp = (uint32_t)iv;
        if (enif_get_map_value(env, argv[2], make_atom(env, "min_iqp"), &mval) && enif_get_int(env, mval, &iv))
            vp.min_iqp = (uint32_t)iv;
        if (enif_get_map_value(env, argv[2], make_atom(env, "max_iqp"), &mval) && enif_get_int(env, mval, &iv))
            vp.max_iqp = (uint32_t)iv;
        if (enif_get_map_value(env, argv[2], make_atom(env, "profile"), &mval) && enif_get_int(env, mval, &iv))
            vp.profile = (uint32_t)iv;
        if (enif_get_map_value(env, argv[2], make_atom(env, "initial_delay"), &mval) && enif_get_int(env, mval, &iv))
            vp.initial_delay = (int32_t)iv;
        if (enif_get_map_value(env, argv[2], make_atom(env, "stat_time"), &mval) && enif_get_int(env, mval, &iv))
            vp.stat_time = (uint32_t)iv;

        if (enif_get_map_value(env, argv[2], make_atom(env, "rc_mode"), &mval)) {
            char rc_atom[16];
            if (enif_get_atom(env, mval, rc_atom, sizeof(rc_atom), ERL_NIF_LATIN1)) {
                if (strcmp(rc_atom, "vbr") == 0)        vp.rc_mode = VIDEO_RC_MODE_VBR;
                else if (strcmp(rc_atom, "avbr") == 0)  vp.rc_mode = VIDEO_RC_MODE_AVBR;
                else if (strcmp(rc_atom, "fixqp") == 0) vp.rc_mode = VIDEO_RC_MODE_FIXQP;
                else                                    vp.rc_mode = VIDEO_RC_MODE_CBR;
            }
        }

        auto* sg200x = static_cast<ma::CameraSG200X*>(res->val->camera);
        sg200x->setChannelVencParams(ch, vp);
        return make_ok(env, make_atom(env, "ok"));
    }

    Camera::CtrlType ctrl;
    if (!parse_camera_ctrl_type(env, argv[1], &ctrl)) {
        return make_error(env, "unsupported_ctrl");
    }

    // Get value - could be int or tuple of two ints (for window)
    Camera::CtrlValue value;
    if (ctrl == Camera::CtrlType::kWindow) {
        // Expect tuple {width, height}
        const ERL_NIF_TERM* tuple;
        int arity;
        if (!enif_get_tuple(env, argv[2], &arity, &tuple) || arity != 2) {
            return make_error(env, "invalid_value_tuple");
        }
        int w, h;
        if (!enif_get_int(env, tuple[0], &w) || !enif_get_int(env, tuple[1], &h)) {
            return make_error(env, "invalid_window_value");
        }
        value.u16s[0] = (uint16_t)w;
        value.u16s[1] = (uint16_t)h;
    } else if (ctrl == Camera::CtrlType::kFormat) {
        // Pixel format can be provided as atom (:rgb888/:jpeg/:h264/...) or raw enum integer.
        if (enif_is_atom(env, argv[2])) {
            ma_pixel_format_t format = atom_to_pixel_format(env, argv[2]);
            if (format == MA_PIXEL_FORMAT_UNKNOWN) {
                return make_error(env, "invalid_format_value");
            }
            value.i32 = static_cast<int32_t>(format);
        } else {
            if (!enif_get_int(env, argv[2], &value.i32)) {
                return make_error(env, "invalid_value");
            }
        }
    } else {
        // Single int value
        if (!enif_get_int(env, argv[2], &value.i32)) {
            return make_error(env, "invalid_value");
        }
    }

    ma_err_t err = res->val->camera->commandCtrl(ctrl, Camera::CtrlMode::kWrite, value);
    return err == MA_OK ? make_ok(env, make_atom(env, "ok"))
                        : make_error(env, "set_ctrl_failed");
}

// NIF: Request IDR keyframe on a running H.264/H.265 channel
static ERL_NIF_TERM camera_request_keyframe(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    int ch;
    if (!enif_get_int(env, argv[1], &ch)) return make_error(env, "invalid_channel");
    if (ch < 0 || ch > 2) return make_error(env, "channel_out_of_range");

    int ret = requestKeyframe(static_cast<video_ch_index_t>(ch));
    return ret == 0 ? make_ok(env, make_atom(env, "ok")) : make_error(env, "request_keyframe_failed");
}

// NIF: Get camera control
static ERL_NIF_TERM camera_get_ctrl(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    char ctrl_atom[32];
    if (!enif_get_atom(env, argv[1], ctrl_atom, sizeof(ctrl_atom), ERL_NIF_LATIN1)) {
        return make_error(env, "unsupported_ctrl");
    }

    if (strcmp(ctrl_atom, "quality") == 0) {
        Camera::CtrlValue chn_val{};
        res->val->camera->commandCtrl(Camera::kChannel, Camera::kRead, chn_val);
        int jpeg_ch = chn_val.i32;

        VENC_JPEG_PARAM_S jpeg_param{};
        CVI_S32 ret = CVI_VENC_GetJpegParam(jpeg_ch, &jpeg_param);
        if (ret != CVI_SUCCESS) {
            return make_error(env, "quality_unavailable");
        }
        return make_ok(env, enif_make_int(env, static_cast<int>(jpeg_param.u32Qfactor)));
    }

    // --- AE Controls ---
    if (strcmp(ctrl_atom, "ae_mode") == 0) {
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        return make_ok(env, make_atom(env, attr.enOpType == OP_TYPE_AUTO ? "auto" : "manual"));
    }

    if (strcmp(ctrl_atom, "max_iso") == 0) {
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        if (attr.stAuto.enGainType == AE_TYPE_ISO)
            return make_ok(env, enif_make_int(env, static_cast<int>(attr.stAuto.stISONumRange.u32Max)));
        CVI_U32 gain = attr.stAuto.stSysGainRange.u32Max;
        return make_ok(env, enif_make_int(env, (gain * 100) / 1024));
    }

    if (strcmp(ctrl_atom, "exposure_us") == 0) {
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        return make_ok(env, enif_make_int(env, static_cast<int>(attr.stManual.u32ExpTime)));
    }

    if (strcmp(ctrl_atom, "gain") == 0) {
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        return make_ok(env, enif_make_int(env, static_cast<int>(attr.stManual.u32AGain)));
    }

    if (strcmp(ctrl_atom, "exposure_range") == 0) {
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        return make_ok(env, enif_make_tuple2(env,
            enif_make_int(env, static_cast<int>(attr.stAuto.stExpTimeRange.u32Min)),
            enif_make_int(env, static_cast<int>(attr.stAuto.stExpTimeRange.u32Max))));
    }

    if (strcmp(ctrl_atom, "max_exposure_us") == 0) {
        ISP_EXPOSURE_ATTR_S attr;
        if (CVI_ISP_GetExposureAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "ae_unavailable");
        return make_ok(env, enif_make_int(env, static_cast<int>(attr.stAuto.stExpTimeRange.u32Max)));
    }

    // --- TNR Controls ---
    if (strcmp(ctrl_atom, "tnr_enable") == 0) {
        ISP_TNR_ATTR_S attr;
        if (CVI_ISP_GetTNRAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "tnr_unavailable");
        return make_ok(env, make_atom(env, attr.Enable ? "true" : "false"));
    }

    if (strcmp(ctrl_atom, "tnr_strength") == 0) {
        ISP_TNR_ATTR_S attr;
        if (CVI_ISP_GetTNRAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "tnr_unavailable");
        return make_ok(env, enif_make_int(env, static_cast<int>(attr.stManual.TnrStrength0)));
    }

    // --- Image Tuning Controls ---
    // Brightness: read from ISP_YCONTRAST_ATTR_S.stManual.CenterLuma
    if (strcmp(ctrl_atom, "brightness") == 0) {
        ISP_YCONTRAST_ATTR_S attr;
        if (CVI_ISP_GetYContrastAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        return make_ok(env, enif_make_int(env, static_cast<int>(attr.stManual.CenterLuma)));
    }

    // Contrast: read from ISP_YCONTRAST_ATTR_S.stManual.ContrastHigh
    if (strcmp(ctrl_atom, "contrast") == 0) {
        ISP_YCONTRAST_ATTR_S attr;
        if (CVI_ISP_GetYContrastAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        return make_ok(env, enif_make_int(env, static_cast<int>(attr.stManual.ContrastHigh)));
    }

    // Saturation: read from ISP_SATURATION_ATTR_S.stManual.Saturation
    if (strcmp(ctrl_atom, "saturation") == 0) {
        ISP_SATURATION_ATTR_S attr;
        if (CVI_ISP_GetSaturationAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        return make_ok(env, enif_make_int(env, static_cast<int>(attr.stManual.Saturation)));
    }

    // Sharpness: read from ISP_SHARPEN_ATTR_S.stManual.GlobalGain
    if (strcmp(ctrl_atom, "sharpness") == 0) {
        ISP_SHARPEN_ATTR_S attr;
        if (CVI_ISP_GetSharpenAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        return make_ok(env, enif_make_int(env, static_cast<int>(attr.stManual.GlobalGain)));
    }

    // Raw (Bayer) NR strength: read from ISP_NR_ATTR_S.stManual.NoiseSuppressStr
    if (strcmp(ctrl_atom, "nr_strength") == 0) {
        ISP_NR_ATTR_S attr;
        if (CVI_ISP_GetNRAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        return make_ok(env, enif_make_int(env, static_cast<int>(attr.stManual.NoiseSuppressStr)));
    }

    // Luma NR strength: read from ISP_YNR_ATTR_S.stManual.NoiseSuppressStr
    if (strcmp(ctrl_atom, "ynr_strength") == 0) {
        ISP_YNR_ATTR_S attr;
        if (CVI_ISP_GetYNRAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        return make_ok(env, enif_make_int(env, static_cast<int>(attr.stManual.NoiseSuppressStr)));
    }

    // Chroma NR strength: read from ISP_CNR_ATTR_S.stManual.NoiseSuppressStr
    if (strcmp(ctrl_atom, "cnr_strength") == 0) {
        ISP_CNR_ATTR_S attr;
        if (CVI_ISP_GetCNRAttr(0, &attr) != CVI_SUCCESS) return make_error(env, "isp_unavailable");
        return make_ok(env, enif_make_int(env, static_cast<int>(attr.stManual.NoiseSuppressStr)));
    }

    Camera::CtrlType ctrl;
    if (!parse_camera_ctrl_type(env, argv[1], &ctrl)) {
        return make_error(env, "unsupported_ctrl");
    }

    Camera::CtrlValue value{};
    ma_err_t err = res->val->camera->commandCtrl(ctrl, Camera::CtrlMode::kRead, value);
    if (err != MA_OK) return make_error(env, "get_ctrl_failed");

    switch (ctrl) {
        case Camera::CtrlType::kWindow:
            return make_ok(env, enif_make_tuple2(env, enif_make_int(env, value.u16s[0]), enif_make_int(env, value.u16s[1])));
        case Camera::CtrlType::kChannel:
        case Camera::CtrlType::kFps:
            return make_ok(env, enif_make_int(env, value.i32));
        case Camera::CtrlType::kFormat:
            return make_ok(env, pixel_format_to_atom(env, static_cast<ma_pixel_format_t>(value.i32)));
        default:
            return make_error(env, "unsupported_ctrl");
    }
}

// NIF: Get camera ID
static ERL_NIF_TERM camera_get_id(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    return make_ok(env, enif_make_uint64(env, res->val->camera->getID()));
}

// NIF function table
static ERL_NIF_TERM isp_available(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ISP_EXPOSURE_ATTR_S attr;
    CVI_S32 ret = CVI_ISP_GetExposureAttr(0, &attr);
    return ret == CVI_SUCCESS ? make_ok(env, make_atom(env, "true"))
                              : make_ok(env, make_atom(env, "false"));
}

// Returns CLOCK_MONOTONIC time in nanoseconds — the same clock and unit
// that `Tick::current()` uses to stamp `Image.timestamp`. Erlang's
// `System.monotonic_time/1` is also CLOCK_MONOTONIC-based but with a
// different epoch (VM start vs boot), so subtracting `image.timestamp`
// from `System.monotonic_time/1` produces nonsense. Use this NIF
// instead so dwell math is a single-clock subtraction.
static ERL_NIF_TERM tick_now(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return make_error(env, "clock_gettime_failed");
    }
    int64_t ns = static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    return enif_make_int64(env, ns);
}

static ErlNifFunc nif_functions[] = {
    // Engine functions
    {"engine_cvi_new", 0, engine_cvi_new, 0},
    {"engine_cvi_init", 1, engine_cvi_init, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"engine_cvi_load", 2, engine_cvi_load, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"engine_cvi_run", 1, engine_cvi_run, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"engine_cvi_get_input_size", 1, engine_cvi_get_input_size, 0},
    {"engine_cvi_get_output_size", 1, engine_cvi_get_output_size, 0},
    {"engine_cvi_get_input", 2, engine_cvi_get_input, 0},
    {"engine_cvi_get_output", 2, engine_cvi_get_output, 0},
    {"engine_cvi_get_input_shape", 2, engine_cvi_get_input_shape, 0},
    {"engine_cvi_get_output_shape", 2, engine_cvi_get_output_shape, 0},
    {"engine_cvi_get_input_quant_param", 2, engine_cvi_get_input_quant_param, 0},
    {"engine_cvi_get_output_quant_param", 2, engine_cvi_get_output_quant_param, 0},
    {"engine_cvi_set_input", 3, engine_cvi_set_input, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"engine_cvi_get_input_num", 2, engine_cvi_get_input_num, 0},
    {"engine_cvi_get_output_num", 2, engine_cvi_get_output_num, 0},

    // Model functions
    {"model_create", 1, model_create, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"model_create", 2, model_create, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"model_get_type", 1, model_get_type, 0},
    {"model_get_name", 1, model_get_name, 0},
    {"model_get_input_type", 1, model_get_input_type, 0},
    {"model_get_output_type", 1, model_get_output_type, 0},
    {"model_run", 2, model_run, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"model_set_config", 3, model_set_config, 0},
    {"model_get_config", 2, model_get_config, 0},
    {"model_get_perf", 1, model_get_perf, 0},

    // Device functions
    {"device_get_instance", 0, device_get_instance, 0},
    {"device_get_info", 1, device_get_info, 0},
    {"device_get_sensors", 1, device_get_sensors, 0},
    {"device_get_models", 1, device_get_models, 0},

    // Camera functions
    {"camera_get", 2, camera_get, 0},
    {"camera_count", 1, camera_count, 0},
    {"camera_init", 2, camera_init, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"camera_deinit", 1, camera_deinit, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"camera_get_presets", 1, camera_get_presets, 0},
    {"camera_get_preset_idx", 1, camera_get_preset_idx, 0},
    {"camera_is_initialized", 1, camera_is_initialized, 0},
    {"camera_start_stream", 2, camera_start_stream, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"camera_stop_stream", 1, camera_stop_stream, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"camera_is_streaming", 1, camera_is_streaming, 0},
    {"camera_retrieve_frame", 2, camera_retrieve_frame, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"camera_try_retrieve_frame", 2, camera_try_retrieve_frame, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"camera_retrieve_latest_frame", 2, camera_retrieve_latest_frame, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"camera_set_ctrl", 3, camera_set_ctrl, 0},
    {"camera_get_ctrl", 2, camera_get_ctrl, 0},
    {"camera_get_id", 1, camera_get_id, 0},
    {"camera_request_keyframe", 2, camera_request_keyframe, 0},

    // Image processing functions
    {"image_convert", 3, image_convert, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"image_resize", 3, image_resize, ERL_NIF_DIRTY_JOB_CPU_BOUND},

    // ISP functions
    {"isp_available", 0, isp_available, 0},

    // Tick — same clock that stamps Image.timestamp (CLOCK_MONOTONIC ns)
    {"tick_now", 0, tick_now, 0},
};

// NIF initialization - register resource type
static int on_load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    ErlNifResourceFlags flags = (ErlNifResourceFlags)(
        ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);

    // Register engine resource type
    NifRes<EngineCVI>::type = enif_open_resource_type(
        env, "Elixir.SSCMEx.Nif", "engine_cvi",
        engine_cvi_dtor, flags, NULL);

    if (!NifRes<EngineCVI>::type) return -1;

    // Register tensor data resource type for zero-copy
    NifRes<TensorDataRes>::type = enif_open_resource_type(
        env, "Elixir.SSCMEx.Nif", "tensor_data",
        tensor_data_dtor, flags, NULL);

    if (!NifRes<TensorDataRes>::type) return -1;

    // Register model resource type
    NifRes<ModelRes>::type = enif_open_resource_type(
        env, "Elixir.SSCMEx.Nif", "model",
        model_dtor, flags, NULL);

    if (!NifRes<ModelRes>::type) return -1;

    // Register device resource type
    NifRes<DeviceRes>::type = enif_open_resource_type(
        env, "Elixir.SSCMEx.Nif", "device",
        device_dtor, flags, NULL);

    if (!NifRes<DeviceRes>::type) return -1;

    // Register camera resource type
    NifRes<CameraRes>::type = enif_open_resource_type(
        env, "Elixir.SSCMEx.Nif", "camera",
        camera_dtor, flags, NULL);

    return NifRes<CameraRes>::type ? 0 : -1;
}

// NIF reload
static int on_reload(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    return 0;
}

// NIF upgrade
static int on_upgrade(ErlNifEnv *env, void **priv_data, void **old_priv_data, ERL_NIF_TERM load_info) {
    return 0;
}

// NIF cleanup
static void on_unload(ErlNifEnv *env, void *priv_data) {
    // Resource types are automatically cleaned up
}

// NIF initialization macro
ERL_NIF_INIT(Elixir.SSCMEx.Nif, nif_functions, on_load, on_reload, on_upgrade, on_unload)
