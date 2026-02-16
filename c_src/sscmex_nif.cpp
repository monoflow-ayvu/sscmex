#include <erl_nif.h>
#include <string>
#include <cstring>

// SSCMA-Micro headers
#include "sscma/core/engine/ma_engine_cvi.h"
#include "sscma/core/ma_version.h"

using namespace ma::engine;

// Stringify helpers for version macros
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define SSCMA_VERSION_STRING TOSTRING(MA_MAJOR_VERSION) "." TOSTRING(MA_MINOR_VERSION) "." TOSTRING(MA_PATCH_VERSION)

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

// Basic hello world function for testing NIF loading
static ERL_NIF_TERM hello(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ERL_NIF_TERM result = enif_make_string(env, "Hello from SSCMEx NIF!", ERL_NIF_LATIN1);
    return make_ok(env, result);
}

// Initialize TPU via SSCMA-Micro EngineCVI
// This creates an engine instance and calls init() to verify the library works
static ERL_NIF_TERM sscma_init(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ERL_NIF_TERM map = enif_make_new_map(env);

    // Create CVI engine instance
    EngineCVI engine;

    // Initialize the engine (for CVI, this just returns MA_OK)
    ma_err_t err = engine.init();

    if (err == MA_OK) {
        enif_make_map_put(env, map, make_atom(env, "success"), make_atom(env, "true"), &map);
        enif_make_map_put(env, map, make_atom(env, "engine"), make_atom(env, "cvi"), &map);
        enif_make_map_put(env, map, make_atom(env, "library"), make_atom(env, "sscma_micro"), &map);
        enif_make_map_put(env, map, make_atom(env, "status"), make_atom(env, "ok"), &map);
        enif_make_map_put(env, map, make_atom(env, "sscma_version"),
            enif_make_string(env, SSCMA_VERSION_STRING, ERL_NIF_LATIN1), &map);
        enif_make_map_put(env, map, make_atom(env, "chip"), make_atom(env, "sg2002"), &map);
        return make_ok(env, map);
    } else {
        enif_make_map_put(env, map, make_atom(env, "success"), make_atom(env, "false"), &map);
        enif_make_map_put(env, map, make_atom(env, "error_code"), enif_make_int(env, err), &map);
        enif_make_map_put(env, map, make_atom(env, "library"), make_atom(env, "sscma_micro"), &map);
        return make_error(env, "SSCMA-Micro engine init failed");
    }
}

// NIF function table
static ErlNifFunc nif_functions[] = {
    {"hello", 0, hello, 0},
    {"sscma_init", 0, sscma_init, 0},
};

// NIF initialization
static int on_load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    return 0;
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
    // Cleanup resources here
}

// NIF initialization macro - note the module name is SSCMEx
ERL_NIF_INIT(Elixir.SSCMEx.Nif, nif_functions, on_load, on_reload, on_upgrade, on_unload)
