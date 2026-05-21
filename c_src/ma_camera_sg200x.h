#ifndef _MA_CAMERA_SG200X_H_
#define _MA_CAMERA_SG200X_H_

#include <map>
#include <atomic>
#include <cstddef>
#include <new>

#include <core/ma_common.h>
#include <porting/ma_porting.h>

#include "video.h"

#include "ma_config_board.h"

namespace ma {

class FrameBufferPool {
public:
    static constexpr int kCapacity = 6;

    FrameBufferPool() = default;
    ~FrameBufferPool() { deinit(); }

    FrameBufferPool(const FrameBufferPool&) = delete;
    FrameBufferPool& operator=(const FrameBufferPool&) = delete;

    bool init(size_t buffer_size) {
        if (initialized_) deinit();
        buffer_size_ = buffer_size;
        for (int i = 0; i < kCapacity; i++) {
            buffers_[i] = new (std::nothrow) uint8_t[buffer_size];
            if (!buffers_[i]) { deinit(); return false; }
            in_use_[i].store(false, std::memory_order_relaxed);
        }
        initialized_ = true;
        return true;
    }

    void deinit() {
        for (int i = 0; i < kCapacity; i++) {
            delete[] buffers_[i];
            buffers_[i] = nullptr;
            in_use_[i].store(false, std::memory_order_relaxed);
        }
        buffer_size_ = 0;
        initialized_ = false;
    }

    uint8_t* acquire(size_t needed) {
        if (!initialized_ || needed > buffer_size_) return nullptr;
        for (int i = 0; i < kCapacity; i++) {
            bool expected = false;
            if (in_use_[i].compare_exchange_strong(expected, true,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return buffers_[i];
            }
        }
        return nullptr;
    }

    void release(uint8_t* ptr) {
        for (int i = 0; i < kCapacity; i++) {
            if (buffers_[i] == ptr) {
                in_use_[i].store(false, std::memory_order_release);
                return;
            }
        }
    }

    bool owns(const uint8_t* ptr) const {
        if (!initialized_) return false;
        for (int i = 0; i < kCapacity; i++) {
            if (buffers_[i] == ptr) return true;
        }
        return false;
    }

    bool detach_and_replace(uint8_t* ptr) {
        for (int i = 0; i < kCapacity; i++) {
            if (buffers_[i] == ptr) {
                buffers_[i] = new (std::nothrow) uint8_t[buffer_size_];
                in_use_[i].store(false, std::memory_order_release);
                return true;
            }
        }
        return false;
    }

private:
    uint8_t* buffers_[kCapacity] = {};
    std::atomic<bool> in_use_[kCapacity] = {};
    size_t buffer_size_ = 0;
    bool initialized_ = false;
};

class CameraSG200X final : public Camera {

    enum { CHN_RAW = 0, CHN_JPEG = 1, CHN_H264 = 2, CHN_MAX };

    static constexpr int kQueueDepthRaw      = 3;
    static constexpr int kQueueDepthEncoded  = 15;

    static int queueDepthFor(ma_pixel_format_t fmt) {
        switch (fmt) {
            case MA_PIXEL_FORMAT_JPEG:
            case MA_PIXEL_FORMAT_H264:
            case MA_PIXEL_FORMAT_H265:
                return kQueueDepthEncoded;
            default:
                return kQueueDepthRaw;
        }
    }

    typedef struct {
        int16_t width;
        int16_t height;
        int16_t fps;
        ma_pixel_format_t format;
        bool configured;
        bool enabled;
        MessageBox* queue;
        video_venc_params_t venc_params = {};
        FrameBufferPool pool;
    } channel;

public:
    CameraSG200X(size_t id);
    ~CameraSG200X();

    ma_err_t init(size_t preset_idx) noexcept override;
    void deInit() noexcept override;

    ma_err_t startStream(StreamMode mode) noexcept override;
    void stopStream() noexcept override;

    ma_err_t commandCtrl(CtrlType ctrl, CtrlMode mode, CtrlValue& value) noexcept override;

    ma_err_t retrieveFrame(ma_img_t& frame, ma_pixel_format_t format) noexcept override;
    ma_err_t retrieveChannel(ma_img_t& frame, int channel_idx) noexcept;

    // Non-blocking variant: returns MA_AGAIN immediately if the channel
    // queue is empty. Useful for consumers that want to drain the
    // chip-side `MessageBox(fps)` queue to its latest frame without
    // waiting for the next sensor tick.
    ma_err_t tryRetrieveChannel(ma_img_t& frame, int channel_idx) noexcept;

    // Drains the channel queue and returns only the most recent frame.
    // Blocks up to one frame interval for the first frame (so it
    // behaves like `retrieveChannel` when the queue is empty), then
    // pulls everything else non-blocking, discarding stale frames.
    // Equivalent to a tight "fetch latest" loop done in C++ — no
    // Elixir↔NIF round-trip per discarded frame.
    ma_err_t retrieveLatestChannel(ma_img_t& frame, int channel_idx) noexcept;

    void returnFrame(ma_img_t& frame) noexcept override;
    void setChannelVencParams(int ch, const video_venc_params_t& params) noexcept;
    void detachFrameBuffer(uint8_t* data) noexcept;

private:
    channel m_channels[CHN_MAX];
    void clearChannelQueue(channel& ch) noexcept;

    int chn;
    int vencCallback(void* pData, void* pArgs);
    int vpssCallback(void* pData, void* pArgs);
    static int vencCallbackStub(void* pData, void* pArgs, void* pUserData);
    static int vpssCallbackStub(void* pData, void* pArgs, void* pUserData);
};

}  // namespace ma

#endif