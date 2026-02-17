/**
 * Device implementation for SG2002 (reCamera)
 *
 * This file implements the ma::Device singleton for the SG2002 platform.
 * The Device provides access to system information, sensors, and model metadata.
 */

#include "sscma/porting/ma_device.h"
#include "sscma/porting/ma_camera.h"
#include "sscma/porting/ma_sensor.h"
#include "sscma/porting/ma_storage.h"
#include "sscma/porting/ma_transport.h"
#include "ma_camera_sg200x.h"

#include <cstdlib>
#include <cstring>
#include <fstream>

namespace ma {

// Singleton instance
static Device* s_instance = nullptr;

// Static camera instance
static CameraSG200X s_camera(0);

Device* Device::getInstance() noexcept {
    if (s_instance == nullptr) {
        s_instance = new Device();
    }
    return s_instance;
}

Device::Device() noexcept
    : m_name(MA_BOARD_NAME),
      m_id("unknown"),
      m_version("v1"),
      m_bootcount(0),
      m_storage(nullptr) {

    // Try to read device ID from efuse
    std::ifstream file("/sys/class/cvi-base/base_efuse_shadow", std::ios::binary);
    if (file.is_open()) {
        file.seekg(0x48, std::ios::beg);
        char id[8];
        file.read(id, 8);
        if (file.gcount() == 8) {
            char buf[17];
            for (int i = 0; i < 8; i++) {
                snprintf(buf + i * 2, 3, "%02x", (unsigned char)id[i]);
            }
            buf[16] = '\0';
            m_id = buf;
        }
        file.close();
    }

    // Register camera sensor
    m_sensors.push_back(&s_camera);
}

Device::~Device() {
    // Note: Sensors are managed externally (static instances)
    m_sensors.clear();
    m_transports.clear();
    m_storage = nullptr;

    // Clear singleton
    s_instance = nullptr;
}

}  // namespace ma
