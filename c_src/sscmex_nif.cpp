#include <erl_nif.h>
#include <string>
#include <cstring>

#include "nif_utils.hpp"
#include "sscma/core/engine/ma_engine_cvi.h"

using namespace ma::engine;

// Instantiate template for EngineCVI
template<> ErlNifResourceType* NifRes<EngineCVI>::type = nullptr;

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

// NIF: Get output tensor (returns map with binary data)
static ERL_NIF_TERM engine_cvi_get_output(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    auto* res = NifRes<EngineCVI>::get(env, argv[0]);
    if (!res || !res->val) return make_error(env, "invalid_resource");

    int32_t index;
    if (!enif_get_int(env, argv[1], &index)) return make_error(env, "invalid_index");

    ma_tensor_t tensor = res->val->getOutput(index);

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

// NIF function table
static ErlNifFunc nif_functions[] = {
    {"engine_cvi_new", 0, engine_cvi_new, 0},
    {"engine_cvi_init", 1, engine_cvi_init, 0},
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
    {"engine_cvi_set_input", 3, engine_cvi_set_input, 0},
    {"engine_cvi_get_input_num", 2, engine_cvi_get_input_num, 0},
    {"engine_cvi_get_output_num", 2, engine_cvi_get_output_num, 0},
};

// NIF initialization - register resource type
static int on_load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    ErlNifResourceFlags flags = (ErlNifResourceFlags)(
        ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);

    NifRes<EngineCVI>::type = enif_open_resource_type(
        env, "Elixir.SSCMEx.Nif", "engine_cvi",
        engine_cvi_dtor, flags, NULL);

    return NifRes<EngineCVI>::type ? 0 : -1;
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
