/**
 * SSCMA-Micro Porting Layer Implementation
 *
 * This file provides implementations for the porting functions required by
 * SSCMA-Micro. These functions are declared in sscma/porting/ma_misc.h
 */

// POSIX feature test macros for usleep and clock_gettime
#define _POSIX_C_SOURCE 199309L

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <ctime>
#include <unistd.h>

extern "C" {

// Memory allocation - use standard C library
void* ma_malloc(size_t size) {
    return malloc(size);
}

void* ma_calloc(size_t nmemb, size_t size) {
    return calloc(nmemb, size);
}

void* ma_realloc(void* ptr, size_t size) {
    return realloc(ptr, size);
}

void ma_free(void* ptr) {
    free(ptr);
}

// Abort - called on assertion failure
void ma_abort(void) {
    std::abort();
}

// Printf - debug output
int ma_printf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int ret = vprintf(fmt, args);
    va_end(args);
    return ret;
}

// Sleep functions
void ma_sleep(uint32_t ms) {
    usleep(ms * 1000);
}

void ma_usleep(uint32_t us) {
    usleep(us);
}

// Time functions
int64_t ma_get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

int64_t ma_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

} // extern "C"
