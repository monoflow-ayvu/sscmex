#include <erl_nif.h>
#include <string>
#include <cstring>

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

// Test TPU SDK availability by checking if we can call a simple SDK function
// We use CVI_NN_GetModelTarget with a null handle to test if the library is properly linked
// This will fail but won't crash - it just proves the library is linked
extern "C" {
    // Forward declaration from cviruntime.h
    typedef int CVI_RC;
    CVI_RC CVI_NN_RegisterModel(const char *model_file, void **model);
}

// Test if TPU SDK is available and properly linked
static ERL_NIF_TERM tpu_test(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    // Build result map
    ERL_NIF_TERM map = enif_make_new_map(env);

    // Try to call a TPU SDK function to verify it's linked
    // We intentionally call with NULL to get an error but verify the function exists
    void *model = nullptr;
    CVI_RC rc = CVI_NN_RegisterModel(nullptr, &model);

    // The call will fail with null input, but if we get here, the library is linked
    // RC values: success=0, error!=0
    bool sdk_linked = true;  // If we got here, the library is linked

    enif_make_map_put(env, map,
                      make_atom(env, "sdk_linked"),
                      enif_make_atom(env, sdk_linked ? "true" : "false"),
                      &map);

    enif_make_map_put(env, map,
                      make_atom(env, "status"),
                      make_atom(env, "ready"),
                      &map);

    enif_make_map_put(env, map,
                      make_atom(env, "chip"),
                      enif_make_string(env, "sg2002", ERL_NIF_LATIN1),
                      &map);

    enif_make_map_put(env, map,
                      make_atom(env, "return_code"),
                      enif_make_int(env, rc),
                      &map);

    return make_ok(env, map);
}

// Get TPU SDK version info
static ERL_NIF_TERM tpu_version(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    // Return a placeholder version since CVI_RT_GetVersion doesn't exist
    // The actual SDK version would need to be determined differently
    return make_ok(env, enif_make_string(env, "TPU SDK linked successfully", ERL_NIF_LATIN1));
}

// NIF function table
static ErlNifFunc nif_functions[] = {
    {"hello", 0, hello, 0},
    {"tpu_test", 0, tpu_test, 0},
    {"tpu_version", 0, tpu_version, 0},
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
