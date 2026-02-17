#include <erl_nif.h>
#include <string>
#include <cstring>
#include <forward_list>
#include <vector>

#include "nif_utils.hpp"
#include "sscma/core/engine/ma_engine_cvi.h"
#include "sscma/core/model/ma_model_factory.h"
#include "sscma/core/model/ma_model_detector.h"
#include "sscma/porting/ma_device.h"
#include "sscma/porting/ma_camera.h"

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
    return MA_PIXEL_FORMAT_UNKNOWN;
}

// Helper: pixel format to atom
static ERL_NIF_TERM pixel_format_to_atom(ErlNifEnv* env, ma_pixel_format_t format) {
    switch (format) {
        case MA_PIXEL_FORMAT_RGB888: return make_atom(env, "rgb888");
        case MA_PIXEL_FORMAT_RGB565: return make_atom(env, "rgb565");
        case MA_PIXEL_FORMAT_YUV422: return make_atom(env, "yuv422");
        case MA_PIXEL_FORMAT_GRAYSCALE: return make_atom(env, "gray");
        default:                     return make_atom(env, "unknown");
    }
}

// Helper: model type to atom
static ERL_NIF_TERM model_type_to_atom(ErlNifEnv* env, ma_model_type_t type) {
    switch (type) {
        case MA_MODEL_TYPE_FOMO:       return make_atom(env, "fomo");
        case MA_MODEL_TYPE_YOLOV5:     return make_atom(env, "yolov5");
        case MA_MODEL_TYPE_YOLOV8:     return make_atom(env, "yolov8");
        case MA_MODEL_TYPE_YOLO11:     return make_atom(env, "yolo11");
        case MA_MODEL_TYPE_IMCLS:      return make_atom(env, "classifier");
        case MA_MODEL_TYPE_YOLOV8_POSE: return make_atom(env, "yolov8_pose");
        case MA_MODEL_TYPE_YOLO11_POSE: return make_atom(env, "yolo11_pose");
        case MA_MODEL_TYPE_YOLO11_SEG: return make_atom(env, "yolo11_seg");
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
        case MA_OUTPUT_TYPE_BBOX:      return make_atom(env, "boxes");
        case MA_OUTPUT_TYPE_CLASS:     return make_atom(env, "classes");
        case MA_OUTPUT_TYPE_KEYPOINT:  return make_atom(env, "keypoints");
        case MA_OUTPUT_TYPE_SEGMENT:   return make_atom(env, "segments");
        default:                       return make_atom(env, "unknown");
    }
}

// Helper: parse SSCMEx.Image struct from Elixir (zero-copy)
static bool get_image_struct(ErlNifEnv* env, ERL_NIF_TERM term, ma_img_t* img, ErlNifBinary* bin) {
    if (!enif_is_map(env, term)) return false;

    ERL_NIF_TERM width_term, height_term, format_term, data_term;

    // Get width
    if (!enif_get_map_value(env, term, make_atom(env, "width"), &width_term)) return false;
    int width_int;
    if (!enif_get_int(env, width_term, &width_int)) return false;
    img->width = (uint16_t)width_int;

    // Get height
    if (!enif_get_map_value(env, term, make_atom(env, "height"), &height_term)) return false;
    int height_int;
    if (!enif_get_int(env, height_term, &height_int)) return false;
    img->height = (uint16_t)height_int;

    // Get format
    if (!enif_get_map_value(env, term, make_atom(env, "format"), &format_term)) return false;
    img->format = atom_to_pixel_format(env, format_term);
    if (img->format == MA_PIXEL_FORMAT_UNKNOWN) return false;

    // Get data binary (zero-copy - just gets pointer to existing data)
    if (!enif_get_map_value(env, term, make_atom(env, "data"), &data_term)) return false;
    if (!enif_inspect_iolist_as_binary(env, data_term, bin)) return false;
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

// Helper: perf to map
static ERL_NIF_TERM perf_to_map(ErlNifEnv* env, const ma_perf_t& perf) {
    ERL_NIF_TERM map = enif_make_new_map(env);
    enif_make_map_put(env, map, make_atom(env, "preprocess"), enif_make_int64(env, perf.preprocess), &map);
    enif_make_map_put(env, map, make_atom(env, "inference"), enif_make_int64(env, perf.inference), &map);
    enif_make_map_put(env, map, make_atom(env, "postprocess"), enif_make_int64(env, perf.postprocess), &map);
    return map;
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

    // Create model using factory
    Model* model = ModelFactory::create(engine_res->val, algorithm_id);
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
        return make_error(env, "invalid_image");
    }

    // Cast to Detector and run (we only support detectors for now)
    Detector* detector = static_cast<Detector*>(res->val->model);
    ma_err_t err = detector->run(&img);

    if (err != MA_OK) return make_error(env, "inference_failed");

    // Get results
    const std::forward_list<ma_bbox_t>& results = detector->getResults();

    // Convert to list of maps
    std::vector<ERL_NIF_TERM> bbox_terms;
    for (const auto& bbox : results) {
        bbox_terms.push_back(bbox_to_map(env, bbox));
    }

    ERL_NIF_TERM list = enif_make_list_from_array(env, bbox_terms.data(), bbox_terms.size());
    return make_ok(env, list);
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

    // Cast to Detector for setConfig
    Detector* detector = static_cast<Detector*>(res->val->model);

    // Map option string to config
    ma_model_cfg_opt_t cfg_opt;
    if (strcmp(opt, "threshold_score") == 0) {
        cfg_opt = MA_MODEL_CFG_OPT_THRESHOLD;
    } else if (strcmp(opt, "threshold_nms") == 0) {
        cfg_opt = MA_MODEL_CFG_OPT_NMS;
    } else {
        return make_error(env, "unknown_option");
    }

    ma_err_t err = detector->setConfig(cfg_opt, (float)value);
    return err == MA_OK ? make_ok(env, make_atom(env, "ok"))
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

// NIF: Retrieve frame from camera (DIRTY IO)
static ERL_NIF_TERM camera_retrieve_frame(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    // Get format atom
    ma_pixel_format_t format = atom_to_pixel_format(env, argv[1]);
    if (format == MA_PIXEL_FORMAT_UNKNOWN) return make_error(env, "invalid_format");

    ma_img_t frame;
    ma_err_t err = res->val->camera->retrieveFrame(frame, format);
    if (err != MA_OK) {
        return make_error(env, "retrieve_frame_failed");
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

// NIF: Set camera control
static ERL_NIF_TERM camera_set_ctrl(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    // Get control type atom
    char ctrl_atom[32];
    if (!enif_get_atom(env, argv[1], ctrl_atom, sizeof(ctrl_atom), ERL_NIF_LATIN1)) {
        return make_error(env, "invalid_ctrl");
    }

    Camera::CtrlType ctrl;
    if (strcmp(ctrl_atom, "window") == 0) ctrl = Camera::CtrlType::kWindow;
    else if (strcmp(ctrl_atom, "channel") == 0) ctrl = Camera::CtrlType::kChannel;
    else if (strcmp(ctrl_atom, "format") == 0) ctrl = Camera::CtrlType::kFormat;
    else if (strcmp(ctrl_atom, "fps") == 0) ctrl = Camera::CtrlType::kFps;
    else return make_error(env, "unsupported_ctrl");

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

// NIF: Get camera ID
static ERL_NIF_TERM camera_get_id(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<CameraRes>::get(env, argv[0]);
    if (!res || !res->val || !res->val->camera) return make_error(env, "invalid_resource");

    return make_ok(env, enif_make_uint64(env, res->val->camera->getID()));
}

// NIF function table
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
    {"camera_set_ctrl", 3, camera_set_ctrl, 0},
    {"camera_get_id", 1, camera_get_id, 0},
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
