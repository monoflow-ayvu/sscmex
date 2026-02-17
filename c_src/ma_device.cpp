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

#include <cstdlib>
#include <cstring>

namespace ma {

// Singleton instance
static Device* s_instance = nullptr;

Device* Device::getInstance() noexcept {
    if (s_instance == nullptr) {
        s_instance = new Device();
    }
    return s_instance;
}

Device::Device() noexcept
    : m_name("reCamera_SG2002"),
      m_id("sg2002_recamera"),
      m_version("1.0.0"),
      m_bootcount(0),
      m_storage(nullptr) {
    // Initialize boot count from persistent storage if available
    // For now, just start at 0
}

Device::~Device() {
    // Clean up sensors
    for (auto* sensor : m_sensors) {
        delete sensor;
    }
    m_sensors.clear();

    // Clean up transports
    for (auto* transport : m_transports) {
        delete transport;
    }
    m_transports.clear();

    // Clean up storage
    if (m_storage) {
        delete m_storage;
        m_storage = nullptr;
    }

    // Clear singleton
    s_instance = nullptr;
}

}  // namespace ma
