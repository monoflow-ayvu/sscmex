#include <erl_nif.h>
#include <string>
#include <cstring>

// TPU Runtime API declarations
extern "C" {
    typedef void *CVI_RT_HANDLE;
    typedef int CVI_RC;

    // Initialize TPU runtime - returns 0 on success
    CVI_RC CVI_RT_Init(CVI_RT_HANDLE *rt_handle);
    CVI_RC CVI_RT_DeInit(CVI_RT_HANDLE rt_handle);

    // Memory functions to test allocation
    typedef void *CVI_RT_MEM;
    CVI_RT_MEM CVI_RT_MemAlloc(CVI_RT_HANDLE rt_handle, uint64_t size);
    void CVI_RT_MemFree(CVI_RT_HANDLE rt_handle, CVI_RT_MEM mem);
    uint64_t CVI_RT_MemGetSize(CVI_RT_MEM mem);
}

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

// Test TPU by initializing the runtime and allocating memory
static ERL_NIF_TERM tpu_test(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ERL_NIF_TERM map = enif_make_new_map(env);

    CVI_RT_HANDLE handle = nullptr;
    CVI_RC rc = CVI_RT_Init(&handle);

    if (rc != 0 || handle == nullptr) {
        enif_make_map_put(env, map, make_atom(env, "success"), make_atom(env, "false"), &map);
        enif_make_map_put(env, map, make_atom(env, "error"), enif_make_int(env, rc), &map);
        enif_make_map_put(env, map, make_atom(env, "message"),
                          enif_make_string(env, "CVI_RT_Init failed", ERL_NIF_LATIN1), &map);
        return make_error(env, "TPU runtime init failed");
    }

    // Try to allocate a small buffer to verify TPU memory works
    CVI_RT_MEM mem = CVI_RT_MemAlloc(handle, 1024);
    bool mem_ok = (mem != nullptr);
    uint64_t mem_size = 0;
    if (mem_ok) {
        mem_size = CVI_RT_MemGetSize(mem);
        CVI_RT_MemFree(handle, mem);
    }

    // Clean up runtime
    CVI_RT_DeInit(handle);

    // Build success result
    enif_make_map_put(env, map, make_atom(env, "success"), make_atom(env, "true"), &map);
    enif_make_map_put(env, map, make_atom(env, "runtime_init"), make_atom(env, "ok"), &map);
    enif_make_map_put(env, map, make_atom(env, "memory_alloc"), mem_ok ? make_atom(env, "ok") : make_atom(env, "failed"), &map);
    enif_make_map_put(env, map, make_atom(env, "memory_size"), enif_make_int(env, (int)mem_size), &map);
    enif_make_map_put(env, map, make_atom(env, "chip"), enif_make_string(env, "sg2002", ERL_NIF_LATIN1), &map);

    return make_ok(env, map);
}

// Get TPU SDK version/runtime info
static ERL_NIF_TERM tpu_version(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ERL_NIF_TERM map = enif_make_new_map(env);

    // Initialize runtime to verify it works
    CVI_RT_HANDLE handle = nullptr;
    CVI_RC rc = CVI_RT_Init(&handle);

    if (rc == 0 && handle != nullptr) {
        enif_make_map_put(env, map, make_atom(env, "runtime"), make_atom(env, "available"), &map);
        enif_make_map_put(env, map, make_atom(env, "status"), make_atom(env, "ok"), &map);
        CVI_RT_DeInit(handle);
    } else {
        enif_make_map_put(env, map, make_atom(env, "runtime"), make_atom(env, "unavailable"), &map);
        enif_make_map_put(env, map, make_atom(env, "error_code"), enif_make_int(env, rc), &map);
    }

    // SDK info (these are compile-time constants from the SDK)
    enif_make_map_put(env, map, make_atom(env, "sdk"), enif_make_string(env, "cvitek_tpu", ERL_NIF_LATIN1), &map);
    enif_make_map_put(env, map, make_atom(env, "target"), enif_make_string(env, "sg2002/cv181x", ERL_NIF_LATIN1), &map);

    return make_ok(env, map);
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
