/**
 * Board configuration for SG2002 (reCamera)
 *
 * This file provides board-specific configuration for SSCMA-Micro.
 */

#ifndef _MA_CONFIG_BOARD_H_
#define _MA_CONFIG_BOARD_H_

// Board identification
#define MA_BOARD_NAME "reCamera_SG2002"

// Use CVI (CVitek) inference engine for TPU
// Note: SSCMA-Micro uses #ifdef checks, so we define this macro (value doesn't matter)
#define MA_USE_ENGINE_CVI 1

// Do NOT define MA_USE_ENGINE_TFLITE - we only use CVI engine
// #define MA_USE_ENGINE_TFLITE 1  // NOT DEFINED - TFLite disabled

// Use filesystem for model loading
#define MA_USE_FILESYSTEM 1

// Enable tensor name support (required by CVI engine)
#define MA_USE_ENGINE_TENSOR_NAME 1

// Memory configuration for CVI engine
// Note: The actual tensor arena is managed by the TPU SDK
#define MA_ENGINE_CVI_TENSOR_ARENA_SIZE (4 * 1024 * 1024)  // 4MB
#define MA_USE_STATIC_TENSOR_ARENA 0  // Use dynamic allocation

// Disable Wi-Fi status (not needed for basic operation)
// Note: Not defining MA_USE_EXTERNAL_WIFI_STATUS means it's disabled

#endif  // _MA_CONFIG_BOARD_H_
