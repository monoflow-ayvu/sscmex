#ifndef NIF_UTILS_HPP
#define NIF_UTILS_HPP

#include <erl_nif.h>
#include <string>

namespace erlang::nif {

// Convert ERL_NIF_TERM to string
inline std::string get_string(ErlNifEnv *env, ERL_NIF_TERM term) {
    unsigned len = 0;
    if (!enif_get_list_length(env, term, &len)) {
        return "";
    }
    std::string result(len, '\0');
    char *buf = &result[0];
    if (!enif_get_string(env, term, buf, len + 1, ERL_NIF_LATIN1)) {
        return "";
    }
    return result;
}

// Create binary term
inline ERL_NIF_TERM make_binary(ErlNifEnv *env, const std::string &str) {
    return enif_make_string_len(env, str.c_str(), str.size(), ERL_NIF_LATIN1);
}

// Create binary term from char*
inline ERL_NIF_TERM make_binary(ErlNifEnv *env, const char *str, size_t len) {
    return enif_make_string_len(env, str, len, ERL_NIF_LATIN1);
}

// Make :ok atom
inline ERL_NIF_TERM ok(ErlNifEnv *env) {
    return enif_make_atom(env, "ok");
}

// Make :error atom
inline ERL_NIF_TERM error(ErlNifEnv *env) {
    return enif_make_atom(env, "error");
}

// Make {:ok, value} tuple
inline ERL_NIF_TERM ok(ErlNifEnv *env, ERL_NIF_TERM value) {
    return enif_make_tuple2(env, ok(env), value);
}

// Make {:error, reason} tuple
inline ERL_NIF_TERM error(ErlNifEnv *env, ERL_NIF_TERM reason) {
    return enif_make_tuple2(env, error(env), reason);
}

} // namespace erlang::nif

// Template for type-safe NIF resources
template<typename T>
struct NifRes {
    static ErlNifResourceType* type;

    T* val;  // The actual C++ object

    static NifRes<T>* allocate(ErlNifEnv* env) {
        void* ptr = enif_alloc_resource(type, sizeof(NifRes<T>));
        if (!ptr) return nullptr;
        NifRes<T>* res = new (ptr) NifRes<T>();
        res->val = nullptr;
        return res;
    }

    static NifRes<T>* get(ErlNifEnv* env, ERL_NIF_TERM term) {
        NifRes<T>* res = nullptr;
        enif_get_resource(env, term, type, (void**)&res);
        return res;
    }
};

#endif // NIF_UTILS_HPP
