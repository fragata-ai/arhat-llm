/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/ 

// Based on the code from oneDNN 3.10
// (https://github.com/uxlfoundation/oneDNN)
// modified by FRAGATA COMPUTER SYSTEMS AG

#include <algorithm>

#include "dnnl.hpp"

#include "arhat/onednn/ocl/device_info.hpp"

namespace arhat {
namespace onednn {
namespace ocl {

//
//    DeviceInfo
//

DeviceInfo::DeviceInfo():
        m_gpuArch(GpuArch::Unknown),
        m_euCount(0),
        m_hwThreads{0, 0},
        m_maxExecSize(0),
        m_maxSubgroupSize(0),
        m_maxSlmBytesPerWg(0) { }

DeviceInfo::~DeviceInfo() { }

void DeviceInfo::Init(GpuArch gpuArch, int euCount) {
    m_gpuArch = gpuArch;
    m_euCount = euCount;
    m_hwThreads[0] = m_euCount * GetThreadsPerEu(m_gpuArch, false);
    m_hwThreads[1] = m_euCount * GetThreadsPerEu(m_gpuArch, true);
    m_maxExecSize = GetMaxExecSize(m_gpuArch);
    m_maxSubgroupSize = GetMaxSubgroupSize(m_gpuArch);
}

GpuArch DeviceInfo::GetGpuArch() {
    return m_gpuArch;
}

int DeviceInfo::GetEuCount() {
    return m_euCount;
}

int DeviceInfo::GetHwThreads() {
    return m_hwThreads[0];
}

int DeviceInfo::GetHwThreads(bool largeGrfMode) {
    return m_hwThreads[largeGrfMode ? 1 : 0];
}

int DeviceInfo::GetMaxExecSize(GpuArch gpuArch) {
    switch (gpuArch) {
    case GpuArch::XeHpc:
    case GpuArch::Xe2:
    case GpuArch::Xe3: 
        return 128;
    default: 
        return 64; 
    }
}

int DeviceInfo::GetMaxExecSize() {
    return m_maxExecSize;
}

int DeviceInfo::GetMaxSubgroupSize(GpuArch gpuArch) {
    switch (gpuArch) {
    case GpuArch::XeLp:
    case GpuArch::XeHp:
    case GpuArch::XeHpg: 
        return 8;
    case GpuArch::XeHpc:
    case GpuArch::Xe2:
    case GpuArch::Xe3: 
        return 16;
    default: 
        return 0;
    } 
}

int DeviceInfo::GetMaxSubgroupSize(dnnl::memory::data_type dataType) {
    if (dataType == dnnl::memory::data_type::undef) {
        return m_maxSubgroupSize;
    }
    int dataTypeSize = int(dnnl::memory::data_type_size(dataType));
    return std::min(m_maxSubgroupSize, m_maxExecSize / dataTypeSize);
}

int DeviceInfo::GetThreadsPerEu(GpuArch gpuArch, bool largeGrfMode) {
    switch (gpuArch) {
    case GpuArch::XeLp: 
        return 7;
    case GpuArch::XeHp:
    case GpuArch::XeHpg:
    case GpuArch::XeHpc:
    case GpuArch::Xe2:
    case GpuArch::Xe3:
        return largeGrfMode ? 4 : 8;
    case GpuArch::Unknown: 
        return 7;
    default:
        // cannot happen
        return 7;
    }
}

} // namespace ocl
} // namespace onednn
} // namespace arhat

