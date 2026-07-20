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
#include <string>
#include <memory>
#include <ostream>
#include <sstream>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/ssm_conv.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    SsmConvNode
//

class SsmConvNode: public NodeBase {
public:
    SsmConvNode(
        Context *context,
        core::Node *a,
        core::Node *b);
    ~SsmConvNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    void InferShapes();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    core::Node *m_a;
    core::Node *m_b;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_cDesc;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

SsmConvNode::SsmConvNode(
        Context *context,
        core::Node *a,
        core::Node *b):
            NodeBase(context),
            m_a(a),
            m_b(b) { }

SsmConvNode::~SsmConvNode() { }

bool SsmConvNode::Init() {
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    m_aDesc = a->MemoryDesc();
    m_bDesc = b->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    InferShapes();
    SetMemory(m_cDesc);
    InitKernel();
    return true;
}

void SsmConvNode::Compute() {
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_bMem);
    m_kernel->SetArgBuffer(2, m_memory);
    m_kernel->Launch(m_ndRange);
}

bool SsmConvNode::Validate() {
    if (m_aDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (m_bDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_aDesc) || 
            !MemoryDescUtil::HasDenseRows(m_aDesc)) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_bDesc) || 
            !MemoryDescUtil::HasDenseRows(m_bDesc)) {
        return false;
    }
    return true;
}

void SsmConvNode::InferShapes() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    dnnl::memory::dims cDims{1, aDims[1], aDims[3] - bDims[3] + 1, bDims[2]};
    m_cDesc = 
        dnnl::memory::desc(
            cDims, 
            dnnl::memory::data_type::f32, 
            dnnl::memory::format_tag::abcd);
}

void SsmConvNode::InitKernel() {
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    bool useX4 = (bDims[3] % 4 == 0);
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    std::string kernelName = useX4 ? "ssm_conv_simple_x4" : "ssm_conv_simple";
    const char *kernelCode = 
        useX4 ?
            kernels::SsmConvSimpleX4KernelCode() :
            kernels::SsmConvSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        kernelName, 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string SsmConvNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("ssm_conv_simple");
    sb.MemoryDesc(m_aDesc);
    sb.MemoryDesc(m_bDesc);
    return sb.Get();
}

std::string SsmConvNode::MakeProlog() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    std::stringstream ss;
    EmitDesc(ss, "SRC0", m_aDesc);
    EmitDesc(ss, "SRC1", m_bDesc);
    EmitDesc(ss, "DST", m_cDesc);
    ss << "\n";
    return ss.str();
}

void SsmConvNode::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims cDims = m_cDesc.get_dims();
    size_t lws0 = 64;
    size_t gws0 = ((size_t(aDims[2]) + lws0 - 1) / lws0) * lws0;
    size_t gws1 = size_t(cDims[2]);
    size_t gws2 = size_t(cDims[1]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

} // namespace

//
//    SsmConvSimple
//

SsmConvSimple::SsmConvSimple(Context *context):
        m_context(context) { }

SsmConvSimple::~SsmConvSimple() { }

std::unique_ptr<core::Node> SsmConvSimple::CreateNode(core::Node *sx, core::Node *c) {
    std::unique_ptr<SsmConvNode> node = std::make_unique<SsmConvNode>(m_context, sx, c);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

