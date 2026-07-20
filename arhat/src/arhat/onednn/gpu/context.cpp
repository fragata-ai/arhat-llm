/* 
* MIT License
*
* Copyright (c) 2026 FRAGATA COMPUTER SYSTEMS AG
* Copyright (c) 2023-2026 The ggml authors
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

#include <cstdint>
#include <cassert>
#include <memory>
#include <algorithm>
#include <ostream>

#include "dnnl.h"

#include "arhat/onednn/base/runtime.hpp"

#include "arhat/onednn/ocl/config_arch.hpp"
#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/device_info.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/gpu/runtime.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

//
//    Context
//

Context::Context(base::Device *device):
        base::Context(device) { 
    m_oclContext = ocl::OclContext::Create(this);
    m_deviceInfo = std::make_unique<ocl::DeviceInfo>();
    InitDeviceInfo();
}

Context::~Context() { }

void Context::InitDeviceInfo() {
    ocl::GpuArch gpuArch = ocl::GpuArch::Unknown;
    int euCount = 0;
    if (CONFIG_GPU_ARCH == GPU_ARCH_UHD) {
        gpuArch = ocl::GpuArch::XeLp;
        euCount = 24;
    } else if (CONFIG_GPU_ARCH == GPU_ARCH_TGL) {
        gpuArch = ocl::GpuArch::XeLp;
        euCount = 96;
    } else if (CONFIG_GPU_ARCH == GPU_ARCH_ARL) {
        gpuArch = ocl::GpuArch::XeHpg;
        euCount = 128;
    } else if (CONFIG_GPU_ARCH == GPU_ARCH_LNL) {
        gpuArch = ocl::GpuArch::Xe2;
        euCount = 64;
    } else {
        assert(false);
    }
    m_deviceInfo->Init(gpuArch, euCount);
}

//
//    NodeBase
//

NodeBase::NodeBase(Context *context):
        base::NodeBase(context),
        m_gpuContext(context),
        m_oclContext(context->GetOclContext()) { }

NodeBase::~NodeBase() { }

void NodeBase::EmitDesc(
        std::ostream &os, 
        const std::string &name, 
        const dnnl::memory::desc &desc) {
    if (!desc.is_zero()) {
        dnnl::memory::dim offset = desc.get_submemory_offset();
        dnnl::memory::dims dims = desc.get_dims();
        dnnl::memory::dims strides = desc.get_strides();
        EmitInt(os, name + "_BASE", int64_t(offset));
        EmitInt(os, name + "_D0", int64_t(dims[0]));
        EmitInt(os, name + "_D1", int64_t(dims[1]));
        EmitInt(os, name + "_D2", int64_t(dims[2]));
        EmitInt(os, name + "_D3", int64_t(dims[3]));
        EmitInt(os, name + "_S0", int64_t(strides[0]));
        EmitInt(os, name + "_S1", int64_t(strides[1]));
        EmitInt(os, name + "_S2", int64_t(strides[2]));
        EmitInt(os, name + "_S3", int64_t(strides[3]));
    } else {
        EmitInt(os, name + "_BASE", 0);
        EmitInt(os, name + "_D0", 0);
        EmitInt(os, name + "_D1", 0);
        EmitInt(os, name + "_D2", 0);
        EmitInt(os, name + "_D3", 0);
        EmitInt(os, name + "_S0", 0);
        EmitInt(os, name + "_S1", 0);
        EmitInt(os, name + "_S2", 0);
        EmitInt(os, name + "_S3", 0);
    }
    os << "\n";
}

void NodeBase::EmitInt(
        std::ostream &os, 
        const std::string &name, 
        int64_t value) {
    os << "#define " << name << " " << ocl::FormatInt(value) << "\n";
}

//
//    NodeSigBuilder
//

NodeSigBuilder::NodeSigBuilder():
        m_needSep(false) { }

NodeSigBuilder::~NodeSigBuilder() { }

void NodeSigBuilder::String(const std::string &value) {
    Sep();
    m_ss << value;
}

void NodeSigBuilder::Bool(bool value) {
    Sep();
    m_ss << (value ? 1 : 0);
}

void NodeSigBuilder::Int(int64_t value) {
    Sep();
    m_ss << std::hex << value;
}

void NodeSigBuilder::Float(float value) {
    union {
        float f;
        uint32_t i;
    } u32;
    u32.f = value;
    Sep();
    m_ss << std::hex << u32.i;
}

void NodeSigBuilder::MemoryDesc(const dnnl::memory::desc &desc) {
    if (desc.is_zero()) {
        Sep();
        m_ss << "[]";
        return;
    }
    if (desc.get_format_kind() != dnnl::memory::format_kind::blocked) {
        core::Error("Node signature is supported for blocked format only");
    }
    Sep();
    m_ss << '[';
    m_needSep = false;
    dnnl::memory::data_type dt = desc.get_data_type();
    Int(int64_t(dt));
    MemoryDims(desc.get_dims());
    MemoryDims(desc.get_padded_dims());
    MemoryDims(desc.get_strides());
    MemoryDims(desc.get_padded_offsets());
    Int(int64_t(desc.get_submemory_offset()));
    if (desc.get_inner_nblks() != 0) {
         MemoryDims(desc.get_inner_blks());
         MemoryDims(desc.get_inner_idxs());
    }
    m_ss << ']';
}

void NodeSigBuilder::MemoryDims(const dnnl::memory::dims &dims) {
    Sep();
    m_ss << '[';
    m_needSep = false;
    for (dnnl::memory::dim d: dims) {
        Int(int64_t(d));
    }
    m_ss << ']';
}

std::string NodeSigBuilder::Get() {
    return m_ss.str();
}

void NodeSigBuilder::Sep() {
    if (m_needSep) {
        m_ss << '.';
    }
    m_needSep = true;
}

namespace {

//
//    ContextFactory
//

class ContextFactory: public base::ContextFactory {
public:
    ContextFactory();
    ~ContextFactory();
public:
    std::unique_ptr<core::Context> CreateContext(base::Device *device) override;
};

ContextFactory::ContextFactory() { }

ContextFactory::~ContextFactory() { }

std::unique_ptr<core::Context> ContextFactory::CreateContext(base::Device *device) {
    return std::make_unique<Context>(device);
}

ContextFactory g_contextFactory;

} // namespace

//
//    Setup interface
//

void Setup() {
    base::EnterContextFactory(core::DeviceKind::Gpu, &g_contextFactory);
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

