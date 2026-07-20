/* 
* MIT License
*
* Copyright (c) 2020-2026 FRAGATA COMPUTER SYSTEMS AG
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include <string>
#include <unordered_map>
#include <memory>
#include <utility>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"
#include "arhat/onednn/base/setup.hpp"

namespace arhat {
namespace onednn {
namespace base {

namespace {

std::unique_ptr<Platform> g_platform;

std::unordered_map<core::DeviceKind, ContextFactory *> g_context_factory_map;

} // namespace

//
//    Platform
//

Platform::Platform() { 
    CreateDevices();
}

Platform::~Platform() { }

std::string Platform::Name() {
    return "oneDNN";
}

int Platform::DeviceCount() {
    return int(m_devices.size());
}

core::Device *Platform::GetDevice(int deviceId) {
    return m_devices[deviceId].get();
}

void Platform::CreateDevices() {
    CreateDevicesWithKind(dnnl::engine::kind::cpu);
    CreateDevicesWithKind(dnnl::engine::kind::gpu);
}

void Platform::CreateDevicesWithKind(dnnl::engine::kind kind) {
    core::DeviceKind deviceKind = 
        (kind == dnnl::engine::kind::cpu) ?
            core::DeviceKind::Cpu : 
            core::DeviceKind::Gpu;
    std::string strKind = (kind == dnnl::engine::kind::cpu) ? "CPU" : "GPU";
    int count = int(dnnl::engine::get_count(kind));
    for (int i = 0; i < count; i++) {
        std::string strIndex = std::to_string(i);
        std::string name = "oneDNN_" + strKind + "_" + strIndex;
        std::string description = "oneDNN " + strKind + " " + strIndex; 
        std::unique_ptr<Device> device = 
            std::make_unique<Device>(this, deviceKind, name, description, i);
        m_devices.emplace_back(std::move(device));
    }
}

//
//    Device
//

Device::Device(
        Platform *platform, 
        core::DeviceKind kind,
        const std::string &name,
        const std::string &description,
        int index):
            m_platform(platform),
            m_kind(kind),
            m_name(name),
            m_description(description),
            m_index(index) { }

Device::~Device() { }

core::Platform *Device::GetPlatform() {
    return m_platform;
}

core::DeviceKind Device::Kind() {
    return m_kind;
}

std::string Device::Name() {
    return m_name;
}

std::string Device::Description() {
    return m_description;
}

std::unique_ptr<core::Context> Device::CreateContext() {
    auto it = g_context_factory_map.find(m_kind);
    if (it != g_context_factory_map.end()) {
        ContextFactory *contextFactory = it->second;
        return contextFactory->CreateContext(this);
    }
    return std::make_unique<Context>(this);
}

//
//    Setup interface (downstream)
//

void EnterContextFactory(core::DeviceKind deviceKind, ContextFactory *contextFactory) {
    g_context_factory_map[deviceKind] = contextFactory;
}

//
//    Setup interface (upstream)
//

void Setup() {
    // In multi-threaded environment caller must implement exclusive access
    if (g_platform != nullptr) {
        return;
    }
    g_platform = std::make_unique<Platform>();
    core::EnterPlatform(g_platform.get());
}

} // namespace base
} // namespace onednn
} // namespace arhat

