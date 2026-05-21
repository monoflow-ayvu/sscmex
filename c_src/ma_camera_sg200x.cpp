#include "ma_camera_sg200x.h"

#include <new>

namespace ma {

static const char* TAG = "ma::camera::sg200x";

struct presets_wrapper_t {
    const char* description;
    uint16_t width;
    uint16_t height;
    int fps;
};

static size_t pool_buffer_size(uint16_t width, uint16_t height, ma_pixel_format_t format) {
    size_t pixels = static_cast<size_t>(width) * height;
    switch (format) {
        case MA_PIXEL_FORMAT_RGB888:
        case MA_PIXEL_FORMAT_RGB888_PLANAR:
            return pixels * 3;
        case MA_PIXEL_FORMAT_RGB565:
        case MA_PIXEL_FORMAT_YUV422:
            return pixels * 2;
        case MA_PIXEL_FORMAT_GRAYSCALE:
            return pixels;
        default:
            return pixels;
    }
}

static const presets_wrapper_t _presets[] = {
    {"1920x1080 @ 30fps", 1920, 1080, 30},
    {"1920x1080 @ 15fps", 1920, 1080, 15},
    {"1920x1080 @ 5fps", 1920, 1080, 5},
    {"1280x720 @ 30fps", 1280, 720, 30},
    {"1280x720 @ 15fps", 1280, 720, 15},
    {"1280x720 @ 5fps", 1280, 720, 5},
};

#define CAMERA_INIT()                               \
    {                                               \
        Thread::enterCritical();                    \
        Thread::sleep(Tick::fromMilliseconds(100)); \
        MA_LOGD(TAG, "start video");                \
        startVideo();                               \
        Thread::sleep(Tick::fromSeconds(1));        \
        Thread::exitCritical();                     \
    }

#define CAMERA_DEINIT()                             \
    {                                               \
        Thread::enterCritical();                    \
        MA_LOGD(TAG, "deinit video");               \
        Thread::sleep(Tick::fromMilliseconds(100)); \
        deinitVideo();                              \
        Thread::sleep(Tick::fromSeconds(1));        \
        Thread::exitCritical();                     \
    }


int CameraSG200X::vencCallback(void* pData, void* pArgs) {

    APP_DATA_CTX_S* pstDataCtx        = (APP_DATA_CTX_S*)pArgs;
    APP_DATA_PARAM_S* pstDataParam    = &pstDataCtx->stDataParam;
    APP_VENC_CHN_CFG_S* pstVencChnCfg = (APP_VENC_CHN_CFG_S*)pstDataParam->pParam;
    VENC_CHN VencChn                  = pstVencChnCfg->VencChn;

    if (!m_streaming) {
        return CVI_SUCCESS;
    }

    VENC_STREAM_S* pstStream = (VENC_STREAM_S*)pData;

    CVI_U32 total_size = 0;
    for (CVI_U32 i = 0; i < pstStream->u32PackCount; i++) {
        VENC_PACK_S* p = &pstStream->pstPack[i];
        total_size += p->u32Len - p->u32Offset;
    }

    auto& ch = m_channels[VencChn];
    uint8_t* buf = ch.pool.acquire(total_size);
    if (!buf) {
        buf = new (std::nothrow) uint8_t[total_size];
        if (!buf) return CVI_SUCCESS;
    }

    ma_img_t* frame = new (std::nothrow) ma_img_t;
    if (!frame) {
        if (ch.pool.owns(buf)) ch.pool.release(buf);
        else delete[] buf;
        return CVI_SUCCESS;
    }

    frame->data      = buf;
    frame->size      = total_size;
    frame->width     = ch.width;
    frame->height    = ch.height;
    frame->format    = ch.format;
    frame->timestamp = Tick::current();
    frame->count     = 1;
    frame->index     = 0;
    frame->physical  = false;
    frame->key       = false;

    CVI_U32 offset = 0;
    for (CVI_U32 i = 0; i < pstStream->u32PackCount; i++) {
        VENC_PACK_S* p  = &pstStream->pstPack[i];
        CVI_U32 len     = p->u32Len - p->u32Offset;
        memcpy(frame->data + offset, p->pu8Addr + p->u32Offset, len);
        offset += len;

        if (ch.format == MA_PIXEL_FORMAT_H264) {
            switch (p->DataType.enH264EType) {
                case H264E_NALU_ISLICE:
                case H264E_NALU_SPS:
                case H264E_NALU_IDRSLICE:
                case H264E_NALU_SEI:
                case H264E_NALU_PPS:
                    frame->key = true;
                    break;
                default:
                    break;
            }
        }
    }

    if (!ch.queue->post(frame, Tick::fromMilliseconds(1000 / ch.fps))) {
        if (ch.pool.owns(frame->data)) ch.pool.release(frame->data);
        else delete[] frame->data;
        delete frame;
    }

    return CVI_SUCCESS;
}

int CameraSG200X::vpssCallback(void* pData, void* pArgs) {

    APP_VENC_CHN_CFG_S* pstVencChnCfg = (APP_VENC_CHN_CFG_S*)pArgs;
    VIDEO_FRAME_INFO_S* VpssFrame     = (VIDEO_FRAME_INFO_S*)pData;
    VIDEO_FRAME_S* f                  = &VpssFrame->stVFrame;

    if (!m_streaming) {
        return CVI_SUCCESS;
    }

    int ch_idx = pstVencChnCfg->VencChn;
    auto& ch = m_channels[ch_idx];

    size_t total_size = 0;
    for (int i = 0; i < 3; i++) total_size += f->u32Length[i];

    uint8_t* buf = ch.pool.acquire(total_size);
    if (!buf) {
        buf = new (std::nothrow) uint8_t[total_size];
        if (!buf) return CVI_SUCCESS;
    }

    ma_img_t* frame = new (std::nothrow) ma_img_t;
    if (!frame) {
        if (ch.pool.owns(buf)) ch.pool.release(buf);
        else delete[] buf;
        return CVI_SUCCESS;
    }

    frame->physical  = false;
    frame->size      = total_size;
    frame->data      = buf;
    frame->width     = ch.width;
    frame->height    = ch.height;
    frame->format    = ch.format;
    frame->timestamp = Tick::current();
    frame->count     = 1;
    frame->index     = 1;

    // Batch contiguous planes into a single Mmap when possible (NV12/NV21).
    bool contiguous = f->u32Length[0] && f->u32Length[1] && !f->u32Length[2]
                   && f->u64PhyAddr[1] == f->u64PhyAddr[0] + f->u32Length[0];

    if (contiguous) {
        CVI_U8* vir = (CVI_U8*)CVI_SYS_Mmap(f->u64PhyAddr[0], total_size);
        if (vir) {
            memcpy(buf, vir, total_size);
            CVI_SYS_Munmap(vir, total_size);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < 3; i++) {
            if (f->u32Length[i]) {
                CVI_U8* vir = (CVI_U8*)CVI_SYS_Mmap(f->u64PhyAddr[i], f->u32Length[i]);
                if (vir) {
                    memcpy(buf + offset, vir, f->u32Length[i]);
                    CVI_SYS_Munmap(vir, f->u32Length[i]);
                }
                offset += f->u32Length[i];
            }
        }
    }

    if (!ch.queue->post(frame, Tick::fromMilliseconds(1000 / ch.fps))) {
        if (ch.pool.owns(frame->data)) ch.pool.release(frame->data);
        else delete[] frame->data;
        delete frame;
    }

    return CVI_SUCCESS;
}

int CameraSG200X::vencCallbackStub(void* pData, void* pArgs, void* pUserData) {
    return reinterpret_cast<CameraSG200X*>(pUserData)->vencCallback(pData, pArgs);
}

int CameraSG200X::vpssCallbackStub(void* pData, void* pArgs, void* pUserData) {
    return reinterpret_cast<CameraSG200X*>(pUserData)->vpssCallback(pData, pArgs);
}

CameraSG200X::CameraSG200X(size_t id) : Camera(id) {
    for (const auto& preset : _presets) {
        m_presets.push_back({.description = preset.description});
    }
    m_presets.shrink_to_fit();

    app_ipcam_Param_Load();

    for (int i = 0; i < CHN_MAX; i++) {
        m_channels[i].configured = false;
        m_channels[i].enabled    = false;
        m_channels[i].fps        = 30;
    }
    m_channels[CHN_RAW].format  = MA_PIXEL_FORMAT_RGB888;
    m_channels[CHN_JPEG].format = MA_PIXEL_FORMAT_JPEG;
    m_channels[CHN_H264].format = MA_PIXEL_FORMAT_H264;
}


CameraSG200X::~CameraSG200X() {
    if (m_initialized) {
        deInit();
    }
}

ma_err_t CameraSG200X::init(size_t preset_idx) noexcept {
    if (m_initialized) [[unlikely]] {
        return MA_EINVAL;
    }

    if (preset_idx >= m_presets.size()) [[unlikely]] {
        return MA_EINVAL;
    }

    const uint16_t w = _presets[preset_idx].width;
    const uint16_t h = _presets[preset_idx].height;
    const int fps    = _presets[preset_idx].fps;
    m_preset_idx     = preset_idx;

    // Channel 0: RAW (RGB888), disabled until explicitly selected.
    m_channels[CHN_RAW].width      = w;
    m_channels[CHN_RAW].height     = h;
    m_channels[CHN_RAW].fps        = fps;
    m_channels[CHN_RAW].format     = MA_PIXEL_FORMAT_RGB888;
    m_channels[CHN_RAW].configured = true;
    m_channels[CHN_RAW].enabled    = false;

    // Channel 1: JPEG is configured but disabled by default to reduce VB/ION pressure.
    m_channels[CHN_JPEG].width      = w;
    m_channels[CHN_JPEG].height     = h;
    m_channels[CHN_JPEG].fps        = fps;
    m_channels[CHN_JPEG].format     = MA_PIXEL_FORMAT_JPEG;
    m_channels[CHN_JPEG].configured = true;
    m_channels[CHN_JPEG].enabled    = false;

    // Channel 2: H264 disabled unless needed.
    m_channels[CHN_H264].width      = w;
    m_channels[CHN_H264].height     = h;
    m_channels[CHN_H264].fps        = fps;
    m_channels[CHN_H264].format     = MA_PIXEL_FORMAT_H264;
    m_channels[CHN_H264].configured = true;
    m_channels[CHN_H264].enabled    = false;

    m_initialized = true;

    return MA_OK;
}

void CameraSG200X::deInit() noexcept {
    if (!m_initialized) [[unlikely]] {
        return;
    }
    if (m_streaming) [[unlikely]] {
        stopStream();
    }

    m_initialized = false;
}

void CameraSG200X::clearChannelQueue(channel& ch) noexcept {
    if (ch.queue == nullptr) {
        return;
    }

    ma_img_t* queued = nullptr;
    while (ch.queue->fetch(reinterpret_cast<void**>(&queued), 0)) {
        if (queued != nullptr) {
            if (!queued->physical) {
                if (ch.pool.owns(queued->data))
                    ch.pool.release(queued->data);
                else
                    delete[] queued->data;
            }
            delete queued;
            queued = nullptr;
        }
    }

    delete ch.queue;
    ch.queue = nullptr;
}

ma_err_t CameraSG200X::startStream(StreamMode mode) noexcept {
    if (!m_initialized) [[unlikely]] {
        return MA_EINVAL;
    }
    if (m_streaming) [[unlikely]] {
        return MA_OK;
    }
    MA_LOGD(TAG, "CameraSG200X::startStream: %zu", m_id);

    bool any_enabled = false;
    for (int i = 0; i < CHN_MAX; i++) {
        if (m_channels[i].enabled) {
            any_enabled = true;
            break;
        }
    }
    // Backward compatible default: if caller didn't select channels, stream RAW channel only.
    if (!any_enabled) {
        m_channels[CHN_RAW].enabled = true;
    }

    for (int i = 0; i < CHN_MAX; i++) {
        if (m_channels[i].enabled && !m_channels[i].configured) {
            MA_LOGW(TAG, "Channel %d is not configured", i);
            return MA_AGAIN;
        }
    }
    for (int i = 0; i < CHN_MAX; i++) {
        if (m_channels[i].enabled) {
            video_ch_param_t param;
            param.width  = m_channels[i].width;
            param.height = m_channels[i].height;
            param.fps    = m_channels[i].fps;

            switch (m_channels[i].format) {
                case MA_PIXEL_FORMAT_JPEG:
                    param.format = VIDEO_FORMAT_JPEG;
                    break;
                case MA_PIXEL_FORMAT_H264:
                    param.format = VIDEO_FORMAT_H264;
                    break;
                case MA_PIXEL_FORMAT_H265:
                    param.format = VIDEO_FORMAT_H265;
                    break;
                case MA_PIXEL_FORMAT_RGB888:
                    param.format = VIDEO_FORMAT_RGB888;
                    break;
                case MA_PIXEL_FORMAT_YUV422:
                    param.format = VIDEO_FORMAT_YUV422;
                    break;
                case MA_PIXEL_FORMAT_GRAYSCALE:
                    param.format = VIDEO_FORMAT_GRAYSCALE;
                    break;
                default:
                    return MA_ENOTSUP;
                    break;
            }
            param.venc_params = m_channels[i].venc_params;
            MA_LOGI(TAG, "width: %d, height: %d, fps: %d format: %d", param.width, param.height, param.fps, param.format);
            setupVideo(static_cast<video_ch_index_t>(i), &param);

            if (m_channels[i].queue != nullptr) {
                clearChannelQueue(m_channels[i]);
            }
            m_channels[i].pool.deinit();
            m_channels[i].pool.init(
                pool_buffer_size(m_channels[i].width, m_channels[i].height, m_channels[i].format));

            m_channels[i].queue = new MessageBox(
                queueDepthFor(m_channels[i].format));

            if (param.format == VIDEO_FORMAT_RGB888 || param.format == VIDEO_FORMAT_NV21) {
                registerVideoFrameHandler(static_cast<video_ch_index_t>(i), 0, vpssCallbackStub, this);
            } else {
                registerVideoFrameHandler(static_cast<video_ch_index_t>(i), 0, vencCallbackStub, this);
            }
        }
    }
    CAMERA_INIT();
    m_streaming = true;
    return MA_OK;
}

void CameraSG200X::stopStream() noexcept {
    if (!m_initialized) [[unlikely]] {
        return;
    }
    if (!m_streaming) [[unlikely]] {
        return;
    }
    MA_LOGD(TAG, "CameraSG200X::stopStream: %zu", m_id);
    m_streaming = false;
    CAMERA_DEINIT();
    for (int i = 0; i < CHN_MAX; i++) {
        clearChannelQueue(m_channels[i]);
        m_channels[i].pool.deinit();
    }
}


ma_err_t CameraSG200X::commandCtrl(CtrlType ctrl, CtrlMode mode, CtrlValue& value) noexcept {
    if (!m_initialized) [[unlikely]] {
        return MA_EPERM;
    }
    switch (ctrl) {
        case kChannel:
            if (mode == kWrite) {
                if (value.i32 < 0 || value.i32 >= CHN_MAX) {
                    MA_LOGD(TAG, "Invalid channel: %d", value.i32);
                    return MA_EINVAL;
                }
                MA_LOGD(TAG, "%d kChannel: %d", chn, value.i32);
                chn                     = value.i32;
                m_channels[chn].enabled = true;
            } else {
                value.i32 = chn;
            }
            break;
        case kWindow:
            if (mode == kWrite) {
                MA_LOGD(TAG, "%d kWindow: %d, %d", chn, value.u16s[0], value.u16s[1]);
                m_channels[chn].width      = value.u16s[0];
                m_channels[chn].height     = value.u16s[1];
                m_channels[chn].configured = true;
            } else if (mode == kRead) {
                value.u16s[0] = m_channels[chn].width;
                value.u16s[1] = m_channels[chn].height;
            }
            break;
        case kFormat:
            if (mode == kWrite) {
                if (m_streaming) return MA_EPERM;
                MA_LOGD(TAG, "%d kFormat: %d", chn, value.i32);
                m_channels[chn].format = static_cast<ma_pixel_format_t>(value.i32);
            } else if (mode == kRead) {
                value.i32 = m_channels[chn].format;
            }
            break;
        case kFps:
            if (mode == kWrite) {
                MA_LOGD(TAG, "%d kFps: %d", chn, value.i32);
                m_channels[chn].fps = value.i32;
            } else if (mode == kRead) {
                value.i32 = m_channels[chn].fps;
            }
            break;
        default:
            return MA_ENOTSUP;
            break;
    }
    return MA_OK;
}

void CameraSG200X::setChannelVencParams(int ch, const video_venc_params_t& params) noexcept {
    if (ch < 0 || ch >= CHN_MAX) {
        MA_LOGW(TAG, "setChannelVencParams: invalid channel %d (max %d)", ch, CHN_MAX - 1);
        return;
    }
    m_channels[ch].venc_params = params;
}

ma_err_t CameraSG200X::retrieveFrame(ma_img_t& frame, ma_pixel_format_t) noexcept {
    return MA_ENOTSUP;
}

ma_err_t CameraSG200X::retrieveChannel(ma_img_t& frame, int channel_idx) noexcept {
    if (!m_streaming) [[unlikely]] {
        return MA_EPERM;
    }
    if (channel_idx < 0 || channel_idx >= CHN_MAX) {
        return MA_EINVAL;
    }
    if (!m_channels[channel_idx].enabled) [[unlikely]] {
        return MA_EPERM;
    }

    ma_img_t* img = nullptr;
    if (!m_channels[channel_idx].queue->fetch(reinterpret_cast<void**>(&img), Tick::fromMilliseconds(1000 / m_channels[channel_idx].fps))) {
        return MA_AGAIN;
    }

    if (img != nullptr) {
        frame = *img;
        delete img;
    }

    return MA_OK;
}

ma_err_t CameraSG200X::tryRetrieveChannel(ma_img_t& frame, int channel_idx) noexcept {
    if (!m_streaming) [[unlikely]] {
        return MA_EPERM;
    }
    if (channel_idx < 0 || channel_idx >= CHN_MAX) {
        return MA_EINVAL;
    }
    if (!m_channels[channel_idx].enabled) [[unlikely]] {
        return MA_EPERM;
    }

    ma_img_t* img = nullptr;
    // timeout=0 → return immediately if queue empty.
    if (!m_channels[channel_idx].queue->fetch(reinterpret_cast<void**>(&img), 0)) {
        return MA_AGAIN;
    }

    if (img != nullptr) {
        frame = *img;
        delete img;
    }

    return MA_OK;
}

ma_err_t CameraSG200X::retrieveLatestChannel(ma_img_t& frame, int channel_idx) noexcept {
    if (!m_streaming) [[unlikely]] {
        return MA_EPERM;
    }
    if (channel_idx < 0 || channel_idx >= CHN_MAX) {
        return MA_EINVAL;
    }
    if (!m_channels[channel_idx].enabled) [[unlikely]] {
        return MA_EPERM;
    }

    auto& ch = m_channels[channel_idx];

    // First fetch is blocking (up to one frame interval) so the caller
    // doesn't spin when the queue is genuinely empty.
    ma_img_t* latest = nullptr;
    if (!ch.queue->fetch(reinterpret_cast<void**>(&latest), Tick::fromMilliseconds(1000 / ch.fps))) {
        return MA_AGAIN;
    }

    ma_img_t* newer = nullptr;
    while (ch.queue->fetch(reinterpret_cast<void**>(&newer), 0)) {
        if (latest != nullptr) {
            if (!latest->physical) {
                if (ch.pool.owns(latest->data))
                    ch.pool.release(latest->data);
                else
                    delete[] latest->data;
            }
            delete latest;
        }
        latest = newer;
        newer = nullptr;
    }

    if (latest != nullptr) {
        frame = *latest;
        delete latest;
    }

    return MA_OK;
}

void CameraSG200X::returnFrame(ma_img_t& frame) noexcept {
    if (frame.physical) return;
    for (int i = 0; i < CHN_MAX; i++) {
        if (m_channels[i].pool.owns(frame.data)) {
            m_channels[i].pool.release(frame.data);
            return;
        }
    }
    delete[] frame.data;
}

void CameraSG200X::detachFrameBuffer(uint8_t* data) noexcept {
    if (!data) return;
    for (int i = 0; i < CHN_MAX; i++) {
        if (m_channels[i].pool.detach_and_replace(data))
            return;
    }
}

}  // namespace ma