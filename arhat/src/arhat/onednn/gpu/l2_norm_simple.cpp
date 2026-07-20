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
#include "arhat/onednn/gpu/l2_norm.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    L2NormNode
//

class L2NormNode: public NodeBase {
public:
    L2NormNode(
        Context *context,
        core::Node *a, 
        float eps, 
        bool inplace);
    ~L2NormNode();
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
    size_t m_lws0;
    size_t m_sgSize;
    size_t m_sgNum;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

L2NormNode::L2NormNode(
        Context *context,
        core::Node *a, 
        float eps, 
        bool inplace):
            NodeBase(context),
            m_a(a),
            m_eps(eps),
            m_inplace(inplace),
            m_lws0(0), 
            m_sgSize(0), 
            m_sgNum(0) { }

L2NormNode::~L2NormNode() { }

bool L2NormNode::Init() { 
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    m_aDesc = a->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    if (m_inplace) {
        m_yDesc = m_aDesc;
        SetMemory(m_yDesc, m_aMem);
    } else {
        m_yDesc = base::PlainMemoryDesc(m_aDesc);
        SetMemory(m_yDesc);
    }
    InitConfig();
    InitKernel();
    return true;
}

void L2NormNode::Compute() { 
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_memory);
    m_kernel->SetArgF32(2, m_eps);
    m_kernel->Launch(m_ndRange);
}

bool L2NormNode::Validate() {
    if (m_aDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_aDesc) ||
            !MemoryDescUtil::HasDenseRows(m_aDesc)) {
        return false;
    }
    return true;
}

void L2NormNode::InitConfig() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t ad3 = size_t(aDims[3]);
    m_sgSize = 32;
    // TODO: Implement common utility function for lws0 computation
    //     (see also, for example, cumsum_simple)
    ocl::OclDeviceInfo info = m_oclContext->GetDeviceInfo();
    size_t maxWgs = info.maxWorkGroupSize;
    m_lws0 = 32; // must be m_sgSize [?]
    while (m_lws0 < ad3 && 2 * m_lws0 <= maxWgs) {
        m_lws0 *= 2;
    }
    m_sgNum = m_lws0 / m_sgSize;
}

void L2NormNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::L2NormSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "l2_norm_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string L2NormNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("l2_norm_simple");
    sb.MemoryDesc(m_aDesc);
    sb.Float(m_eps);
    sb.Bool(m_inplace);
    return sb.Get();
}

std::string L2NormNode::MakeProlog() {
    std::stringstream ss;
    EmitInt(ss, "SG_SIZE", int64_t(m_sgSize));
    EmitInt(ss, "SG_NUM", int64_t(m_sgNum));
    ss << "\n";
    EmitDesc(ss, "SRC", m_aDesc);
    EmitDesc(ss, "DST", m_yDesc);
    return ss.str();
}

void L2NormNode::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t gws0 = size_t(aDims[2]) * m_lws0;
    size_t gws1 = size_t(aDims[1]);
    size_t gws2 = size_t(aDims[0]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, m_lws0, 1, 1);
}

} // namespace

//
//    L2NormSimple
//

L2NormSimple::L2NormSimple(Context *context):
        m_context(context) { }

L2NormSimple::~L2NormSimple() { }

std::unique_ptr<core::Node> L2NormSimple::CreateNode(
        core::Node *a, 
        float eps, 
        bool inplace) {
    std::unique_ptr<L2NormNode> node = 
        std::make_unique<L2NormNode>(m_context, a, eps, inplace);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

