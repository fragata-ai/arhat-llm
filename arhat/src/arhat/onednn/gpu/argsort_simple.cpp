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

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/argsort.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    ArgSortNode
//

class ArgSortNode: public NodeBase {
public:
    ArgSortNode(
        Context *context,
        core::Node *x, 
        core::SortOrder order);
    ~ArgSortNode();
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
    core::Node *m_x;
    core::SortOrder m_order;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory m_xMem;
    int64_t m_xDim3Pad;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

ArgSortNode::ArgSortNode(
        Context *context,
        core::Node *x, 
        core::SortOrder order):
            NodeBase(context),
            m_x(x),
            m_order(order),
            m_xDim3Pad(0) { }

ArgSortNode::~ArgSortNode() { }

bool ArgSortNode::Init() {
    base::NodeBase *x = m_gpuContext->CastNode(m_x);
    m_xDesc = x->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_xMem = x->Memory();
    InferShapes();
    SetMemory(m_yDesc);
    InitKernel();
    return true;
}

void ArgSortNode::Compute() {
    m_kernel->SetArgBuffer(0, m_xMem);
    m_kernel->SetArgBuffer(1, m_memory);
    m_kernel->Launch(m_ndRange);
}

bool ArgSortNode::Validate() {
    if (m_xDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!base::IsRowMajor(m_xDesc) && 
            MemoryDescUtil::Nelems(m_xDesc, false) != 0) {
        return false;
    }
    return true;
}

void ArgSortNode::InferShapes() {
    m_yDesc = 
        dnnl::memory::desc(
            m_xDesc.get_dims(), 
            dnnl::memory::data_type::s32, 
            dnnl::memory::format_tag::abcd);
}

void ArgSortNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::ArgSortSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "argsort_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string ArgSortNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("argsort_simple");
    sb.MemoryDesc(m_xDesc);
    sb.Int(int64_t(m_order));
    return sb.Get();
}

std::string ArgSortNode::MakeProlog() {
    std::stringstream ss;
    int64_t xBase = int64_t(m_xDesc.get_submemory_offset());
    int64_t yBase = int64_t(m_yDesc.get_submemory_offset());
    dnnl::memory::dims xDims = m_xDesc.get_dims();
    int64_t xDim3 = int64_t(xDims[3]);
    m_xDim3Pad = 1;
    while (m_xDim3Pad < xDim3) {
        m_xDim3Pad *= 2;
    }
    EmitInt(ss, "SRC_BASE", xBase);
    EmitInt(ss, "DST_BASE", yBase);
    EmitInt(ss, "SRC_D3", xDim3);
    EmitInt(ss, "SRC_D3_PAD", m_xDim3Pad);
    EmitInt(ss, "ORDER", int64_t(m_order));
    return ss.str();
}

void ArgSortNode::InitNdRange() {
    dnnl::memory::dims xDims = m_xDesc.get_dims();
    size_t nrows = size_t(xDims[0] * xDims[1] * xDims[2]);
    size_t lws0 = size_t(m_xDim3Pad);
    size_t gws0 = lws0;
    size_t gws1 = nrows;
    m_ndRange = ocl::NdRange(gws0, gws1, 1, lws0, 1, 1);
}

} // namespace

//
//    ArgSortSimple
//

ArgSortSimple::ArgSortSimple(Context *context):
        m_context(context) { }

ArgSortSimple::~ArgSortSimple() { }

std::unique_ptr<core::Node> ArgSortSimple::CreateNode(core::Node *a, core::SortOrder order) {
    std::unique_ptr<ArgSortNode> node = std::make_unique<ArgSortNode>(m_context, a, order);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

