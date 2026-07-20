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
#include "arhat/onednn/gpu/repeat.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    RepeatNode
//

class RepeatNode: public NodeBase {
public:
    RepeatNode(
        Context *context,
        core::Node *x,
        const core::Dims &shape);
    ~RepeatNode();
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
    core::Dims m_shape;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory m_xMem;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

RepeatNode::RepeatNode(
        Context *context,
        core::Node *x,
        const core::Dims &shape):
            NodeBase(context),
            m_x(x),
            m_shape(shape) { }

RepeatNode::~RepeatNode() { }

bool RepeatNode::Init() {
    base::NodeBase *x = m_gpuContext->CastNode(m_x);
    m_xDesc = x->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_xMem = x->Memory();
    dnnl::memory::dims yDims = base::MapDims(m_shape);
    m_yDesc = 
        dnnl::memory::desc(
            yDims, 
            m_xDesc.get_data_type(), 
            dnnl::memory::format_tag::abcd);
    SetMemory(m_yDesc);
    InitKernel();
    return true;
}

void RepeatNode::Compute() {
    m_kernel->SetArgBuffer(0, m_xMem);
    m_kernel->SetArgBuffer(1, m_memory);
    m_kernel->Launch(m_ndRange);
}

bool RepeatNode::Validate() {
    if (m_xDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_xDesc) || 
            !MemoryDescUtil::HasDenseRows(m_xDesc)) {
        return false;
    }
    return true;
}

void RepeatNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::RepeatSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "repeat_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string RepeatNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("repeat_simple");
    sb.MemoryDesc(m_xDesc);
    sb.MemoryDims(m_yDesc.get_dims());
    return sb.Get();
}

std::string RepeatNode::MakeProlog() {
    std::stringstream ss;
    EmitDesc(ss, "SRC", m_xDesc);
    EmitDesc(ss, "DST", m_yDesc);
    ss << "\n";
    return ss.str();
}

void RepeatNode::InitNdRange() {
    dnnl::memory::dims yDims = m_yDesc.get_dims();
    size_t lws0 = 64;
    size_t gws0 = size_t(yDims[2]) * lws0;
    size_t gws1 = size_t(yDims[1]);
    size_t gws2 = size_t(yDims[0]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

} // namespace

//
//    RepeatSimple
//

RepeatSimple::RepeatSimple(Context *context):
        m_context(context) { }

RepeatSimple::~RepeatSimple() { }

std::unique_ptr<core::Node> RepeatSimple::CreateNode(core::Node *a, const core::Dims &shape) {
    std::unique_ptr<RepeatNode> node = std::make_unique<RepeatNode>(m_context, a, shape);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

