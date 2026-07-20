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
#include "arhat/onednn/gpu/diag.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    DiagNode
//

class DiagNode: public NodeBase {
public:
    DiagNode(Context *context, core::Node *x);
    ~DiagNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    core::Node *m_x;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory m_xMem;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

DiagNode::DiagNode(Context *context, core::Node *x):
        NodeBase(context), m_x(x) { }

DiagNode::~DiagNode() { }

bool DiagNode::Init() {
    base::NodeBase *x = m_gpuContext->CastNode(m_x);
    m_xDesc = x->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_xMem = x->Memory();
    dnnl::memory::dims xDims = m_xDesc.get_dims(); 
    dnnl::memory::dims yDims{xDims[0], xDims[1], xDims[3], xDims[3]};
    m_yDesc = 
        dnnl::memory::desc(
            yDims, 
            m_xDesc.get_data_type(), 
            dnnl::memory::format_tag::abcd);
    SetMemory(m_yDesc);
    InitKernel();
    return true;
}

void DiagNode::Compute() {
    m_kernel->SetArgBuffer(0, m_xMem);
    m_kernel->SetArgBuffer(1, m_memory);
    m_kernel->Launch(m_ndRange);
}

bool DiagNode::Validate() {
    if (m_xDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_xDesc) || 
            !MemoryDescUtil::HasDenseRows(m_xDesc)) {
        return false;
    }
    return true;
}

void DiagNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::DiagSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "diag_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string DiagNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("diag_simple");
    sb.MemoryDesc(m_xDesc);
    return sb.Get();
}

std::string DiagNode::MakeProlog() {
    std::stringstream ss;
    EmitDesc(ss, "SRC", m_xDesc);
    EmitDesc(ss, "DST", m_yDesc);
    return ss.str();
}

void DiagNode::InitNdRange() {
    dnnl::memory::dims yDims = m_yDesc.get_dims();
    size_t lws0 = 64;
    size_t gws0 = size_t(yDims[2]) * lws0;
    size_t gws1 = size_t(yDims[1]);
    size_t gws2 = size_t(yDims[0]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

} // namespace

//
//    DiagSimple
//

DiagSimple::DiagSimple(Context *context):
        m_context(context) { }

DiagSimple::~DiagSimple() { }

std::unique_ptr<core::Node> DiagSimple::CreateNode(core::Node *a) {
    std::unique_ptr<DiagNode> node = std::make_unique<DiagNode>(m_context, a);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

