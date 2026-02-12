#include <erl_nif.h>
#include <string>

// Minimal hello world function for testing
static ERL_NIF_TERM hello(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ERL_NIF_TERM result = enif_make_string(env, "Hello from SSCMEx NIF!", ERL_NIF_LATIN1);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), result);
}

// NIF function table
static ErlNifFunc nif_functions[] = {
    {"hello", 0, hello, 0},
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

// NIF initialization macro
ERL_NIF_INIT(Elixir.Sscmex.Nif, nif_functions, on_load, on_reload, on_upgrade, on_unload)
