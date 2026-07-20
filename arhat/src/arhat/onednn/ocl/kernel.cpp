/* 
* MIT License
*
* Copyright (c) 2026 FRAGATA COMPUTER SYSTEMS AG
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

#pragma once

#include <cstdint>
#include <cassert>
#include <string>
#include <sstream>

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/ocl/util.hpp"
#include "arhat/onednn/ocl/kernel.hpp"

namespace arhat {
namespace onednn {
namespace ocl {

//
//    NdRange
//

NdRange::NdRange():
        m_ndims(0),
        m_gws{0, 0, 0}, 
        m_lws{0, 0, 0} { }

NdRange::NdRange(size_t gws0, size_t lws0):
        m_ndims(1),
        m_gws{gws0, 0, 0}, 
        m_lws{lws0, 0, 0} { }

NdRange::NdRange(
        size_t gws0, 
        size_t gws1, 
        size_t lws0, 
        size_t lws1):
            m_ndims(2),
            m_gws{gws0, gws1, 0}, 
            m_lws{lws0, lws1, 0} { }

NdRange::NdRange(
        size_t gws0, 
        size_t gws1, 
        size_t gws2, 
        size_t lws0, 
        size_t lws1, 
        size_t lws2):
            m_ndims(3),
            m_gws{gws0, gws1, gws2}, 
            m_lws{lws0, lws1, lws2} { }

NdRange::~NdRange() { }

//
//    KernelContext
//

KernelContext::KernelContext() { 
    SetDefaults();
}

KernelContext::~KernelContext() { }

void KernelContext::SetOption(const std::string &option) {
    if (m_options.count(option)) {
        core::Error("Option is already defined: %s", option.c_str());
    }
    m_options.insert(option);
}

void KernelContext::SetStrVar(const std::string &name, const std::string &value) {
    CheckVarName(name);
    m_strVars[name] = value;
}

void KernelContext::SetIntVar(const std::string name, int64_t value) {
    CheckVarName(name);
    m_intVars[name] = value;
}

void KernelContext::SetFloatVar(const std::string &name, float value) {
    CheckVarName(name);
    m_floatVars[name] = value;
}

std::string KernelContext::Format() const {
    std::ostringstream os;
    for (const std::string &option: m_options) {
        os << " " << option;
    }
    for (auto &var: m_strVars) {
        os << " -D" << var.first << "=" << var.second;
    }
    for (auto &var: m_intVars) {
        os << " -D" << var.first << "=" << FormatInt(var.second);
    }
    for (auto &var: m_floatVars) {
        os << " -D" << var.first << "=" << FormatFloat(var.second);
    }
    return os.str();
}

void KernelContext::SetDefaults() {
    SetOption("-cl-std=CL3.0");
    SetOption("-cl-mad-enable");
    SetOption("-cl-fp32-correctly-rounded-divide-sqrt");
}

void KernelContext::CheckVarName(const std::string &name) {
    if (m_strVars.count(name) != 0 ||
            m_intVars.count(name) != 0 ||
            m_floatVars.count(name) != 0) {
        core::Error("Variable is already defined: %s", name.c_str());
    }
}

//
//    Kernel
//

Kernel::Kernel():
        m_oclContext(nullptr) { }

Kernel::~Kernel() { }

void Kernel::Init(
        OclContext *oclContext,
        const std::string &name,
        const KernelContext &kernelContext,
        const char *prolog,
        const char *code) {
    assert(m_oclContext == nullptr);
    m_oclContext = oclContext;
    const char *source[2] = {prolog, code};
    std::string options = kernelContext.Format();
    m_oclProgram = m_oclContext->CreateProgram(2, source, options.c_str());
    m_oclKernel = m_oclProgram->CreateKernel(name.c_str());
}

int Kernel::GetSpillMemSize() {
    assert(m_oclKernel != nullptr);
    return m_oclKernel->GetSpillMemSize();
}

void Kernel::SetArgU8(int index, uint8_t value) {
    assert(m_oclKernel != nullptr);
    m_oclKernel->SetArgU8(index, value);
}

void Kernel::SetArgU16(int index, uint16_t value) {
    assert(m_oclKernel != nullptr);
    m_oclKernel->SetArgU16(index, value);
}

void Kernel::SetArgU32(int index, uint32_t value) {
    assert(m_oclKernel != nullptr);
    m_oclKernel->SetArgU32(index, value);
}

void Kernel::SetArgU64(int index, uint64_t value) {
    assert(m_oclKernel != nullptr);
    m_oclKernel->SetArgU64(index, value);
}

void Kernel::SetArgS8(int index, int8_t value) {
    assert(m_oclKernel != nullptr);
    m_oclKernel->SetArgS8(index, value);
}

void Kernel::SetArgS16(int index, int16_t value) {
    assert(m_oclKernel != nullptr);
    m_oclKernel->SetArgS16(index, value);
}

void Kernel::SetArgS32(int index, int32_t value) {
    assert(m_oclKernel != nullptr);
    m_oclKernel->SetArgS32(index, value);
}

void Kernel::SetArgS64(int index, int64_t value) {
    assert(m_oclKernel != nullptr);
    m_oclKernel->SetArgS64(index, value);
}

void Kernel::SetArgF32(int index, float value) {
    assert(m_oclKernel != nullptr);
    m_oclKernel->SetArgF32(index, value);
}

void Kernel::SetArgF64(int index, double value) {
    assert(m_oclKernel != nullptr);
    m_oclKernel->SetArgF64(index, value);
}

void Kernel::SetArgBuffer(int index, const dnnl::memory &value) {
    assert(m_oclKernel != nullptr);
    m_oclKernel->SetArgBuffer(index, value);
}

void Kernel::Launch(const NdRange &range) {
    assert(m_oclContext != nullptr);
    assert(m_oclKernel != nullptr);
    m_oclContext->EnqueueNDRangeKernel(
        m_oclKernel.get(),
        range.Ndims(),
        nullptr,
        range.Gws(),
        range.Lws());
}

} // namespace ocl
} // namespace onednn
} // namespace arhat

