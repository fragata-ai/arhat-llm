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
#include <string>
#include <memory>
#include <ostream>
#include <sstream>
#include <algorithm>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/norm.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    OpenCL kernels
//

//
//    NormNode
//

class NormNode: public NodeBase {
public:
    NormNode(
        Context *context,
        core::Node *a, 
        float eps, 
        bool inplace);
    ~NormNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    void InitConfig();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    core::Node *m_a; 
    float m_eps;
    bool m_inplace;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory m_aMem;
    int64_t m_lws0;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

NormNode::NormNode(
        Context *context,
        core::Node *a, 
        float eps, 
        bool inplace):
            NodeBase(context),
            m_a(a),
            m_eps(eps),
            m_inplace(inplace),
            m_lws0(0) { }

NormNode::~NormNode() { }

bool NormNode::Init() { 
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    m_aDesc = a->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_yDesc = m_aDesc;
    if (m_inplace) {
        SetMemory(m_yDesc, m_aMem);
    } else {
        SetMemory(m_yDesc);
    }
    InitConfig();
    InitKernel();
    return true;
}

void NormNode::Compute() { 
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_memory);
    m_kernel->SetArgF32(2, m_eps);
    m_kernel->Launch(m_ndRange);
}

bool NormNode::Validate() {
    if (m_aDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_aDesc) ||
            !MemoryDescUtil::HasDenseRows(m_aDesc)) {
        return false;
    }
    return true;
}

void NormNode::InitConfig() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    m_lws0 = (int64_t(aDims[3] + 32 - 1) / 32) * 32;
    m_lws0 = std::min(m_lws0, int64_t(64));
}

void NormNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::NormSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "norm_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string NormNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("norm_simple");
    sb.MemoryDesc(m_aDesc);
    sb.Float(m_eps);
    sb.Bool(m_inplace);
    return sb.Get();
}

std::string NormNode::MakeProlog() {
    std::stringstream ss;
    EmitInt(ss, "LWS0", m_lws0);
    ss << "\n";
    EmitDesc(ss, "SRC", m_aDesc);
    EmitDesc(ss, "DST", m_yDesc);
    return ss.str();
}

void NormNode::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t lws0 = size_t(m_lws0);
    size_t gws0 = size_t(aDims[2]) * lws0;
    size_t gws1 = size_t(aDims[1]);
    size_t gws2 = size_t(aDims[0]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

} // namespace

//
//    NormSimple
//

NormSimple::NormSimple(Context *context):
        m_context(context) { }

NormSimple::~NormSimple() { }

std::unique_ptr<core::Node> NormSimple::CreateNode(
        core::Node *a, 
        float eps, 
        bool inplace) {
    std::unique_ptr<NormNode> node = 
        std::make_unique<NormNode>(m_context, a, eps, inplace);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

