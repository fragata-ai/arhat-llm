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

#pragma once

#include "dnnl.hpp"

namespace arhat {
namespace onednn {
namespace ocl {

//
//    GpuArch
//

enum class GpuArch {
    Unknown, 
    XeLp, 
    XeHp, 
    XeHpg, 
    XeHpc, 
    Xe2, 
    Xe3
};

//
//    DeviceInfo
//

class DeviceInfo {
public:
    DeviceInfo();
    ~DeviceInfo();
public:
    void Init(GpuArch gpuArch, int euCount);
    GpuArch GetGpuArch();
    int GetEuCount();
    int GetHwThreads();
    int GetHwThreads(bool largeGrfMode);
    static int GetMaxExecSize(GpuArch gpuArch);
    int GetMaxExecSize();
    static int GetMaxSubgroupSize(GpuArch gpuArch);
    int GetMaxSubgroupSize(dnnl::memory::data_type dataType);
    static int GetThreadsPerEu(GpuArch gpuArch, bool largeGrfMode);
private:
    GpuArch m_gpuArch;
    int m_euCount;
    int m_hwThreads[2];
    int m_maxExecSize;
    int m_maxSubgroupSize;
    int m_maxSlmBytesPerWg; // reserved

};

} // namespace ocl
} // namespace onednn
} // namespace arhat

