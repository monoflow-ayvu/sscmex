#ifndef _SOPHGO_VIDEO_H_
#define _SOPHGO_VIDEO_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "app_ipcam_paramparse.h"

typedef enum {
    VIDEO_FORMAT_RGB888 = 0, // no need venc — PIXEL_FORMAT_RGB_888
    VIDEO_FORMAT_NV21,       // no need venc — PIXEL_FORMAT_NV21
    VIDEO_FORMAT_JPEG,
    VIDEO_FORMAT_H264,
    VIDEO_FORMAT_H265,
    VIDEO_FORMAT_GRAYSCALE,  // no need venc — PIXEL_FORMAT_YUV_400
    VIDEO_FORMAT_NV12,       // no need venc — PIXEL_FORMAT_NV12
    VIDEO_FORMAT_YUV422,     // no need venc — PIXEL_FORMAT_YUV_PLANAR_422

    VIDEO_FORMAT_COUNT
} video_format_t;

typedef enum {
    VIDEO_CH0 = 0,
    VIDEO_CH1,
    VIDEO_CH2,

    VIDEO_CH_MAX
} video_ch_index_t;

typedef enum {
    VIDEO_RC_MODE_CBR = 0,
    VIDEO_RC_MODE_VBR,
    VIDEO_RC_MODE_AVBR,
    VIDEO_RC_MODE_FIXQP,
} video_rc_mode_t;

typedef struct {
    bool has_venc_params;
    uint32_t bitrate;
    uint32_t max_bitrate;
    uint32_t gop;
    video_rc_mode_t rc_mode;
    uint32_t min_qp;
    uint32_t max_qp;
    uint32_t min_iqp;
    uint32_t max_iqp;
    uint32_t profile;
    // Low-latency knobs. Sentinel 0 means "leave the SDK default in place".
    // initial_delay (ms): CPB / hypothetical decoder buffer fullness before
    //   the encoder allows playback. SDK default is 1000ms; values like 100
    //   trade burst-tolerance for shipping the first frame sooner.
    // stat_time (s): rate-control analysis window. SDK default is 2s; AVBR
    //   uses this as a lookahead, so dropping to 1 cuts AVBR's encode lag.
    int32_t  initial_delay;
    uint32_t stat_time;
} video_venc_params_t;

typedef struct {
    video_format_t format;
    uint32_t width;
    uint32_t height;
    uint8_t fps;
    video_venc_params_t venc_params;
} video_ch_param_t;

// typedef struct {
//     uint32_t width;
//     uint32_t height;
//     uint8_t* pdata;
//     uint32_t size;
//     uint32_t timestamp;
//     uint32_t id;
// } video_frame_t;

int initVideo(void);
int deinitVideo(void);
int startVideo(void);
int setupVideo(video_ch_index_t ch, const video_ch_param_t* param);
int registerVideoFrameHandler(video_ch_index_t ch, int index, pfpDataConsumes handler, void* pUserData);
int requestKeyframe(video_ch_index_t ch);

#ifdef __cplusplus
}
#endif

#endif // _SOPHGO_VIDEO_H_