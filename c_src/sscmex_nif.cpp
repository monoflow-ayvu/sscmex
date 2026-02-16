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

// NIF function table
static ErlNifFunc nif_functions[] = {
    {"engine_cvi_new", 0, engine_cvi_new, 0},
    {"engine_cvi_init", 1, engine_cvi_init, 0},
    {"engine_cvi_load", 2, engine_cvi_load, 0},
    {"engine_cvi_get_input_size", 1, engine_cvi_get_input_size, 0},
    {"engine_cvi_get_output_size", 1, engine_cvi_get_output_size, 0},
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
