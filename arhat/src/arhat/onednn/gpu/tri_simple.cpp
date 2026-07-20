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
#include "arhat/onednn/gpu/tri.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    TriNode
//

class TriNode: public NodeBase {
public:
    TriNode(
        Context *context,
        core::Node *x,
        int mode);
    ~TriNode();
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
    int m_mode;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory m_xMem;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

TriNode::TriNode(
        Context *context,
        core::Node *x,
        int mode):
            NodeBase(context),
            m_x(x),
            m_mode(mode) { }

TriNode::~TriNode() { }

bool TriNode::Init() {
    base::NodeBase *x = m_gpuContext->CastNode(m_x);
    m_xDesc = x->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_xMem = x->Memory();
    m_yDesc = base::PlainMemoryDesc(m_xDesc);
    SetMemory(m_yDesc);
    InitKernel();
    return true;
}

void TriNode::Compute() {
    m_kernel->SetArgBuffer(0, m_xMem);
    m_kernel->SetArgBuffer(1, m_memory);
    m_kernel->Launch(m_ndRange);
}

bool TriNode::Validate() {
    if (m_xDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!base::IsRowMajor(m_xDesc)) {
        return false;
    }
    return true;
}

void TriNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::TriSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "tri_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string TriNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("tri_simple");
    sb.MemoryDesc(m_xDesc);
    sb.Int(m_mode);
    return sb.Get();
}

std::string TriNode::MakeProlog() {
    std::stringstream ss;
    EmitDesc(ss, "SRC", m_xDesc);
    EmitDesc(ss, "DST", m_yDesc);
    EmitInt(ss, "N", MemoryVolume());
    EmitInt(ss, "MODE", m_mode);
    ss << "\n";
    return ss.str();
}

void TriNode::InitNdRange() {
    size_t lws0 = 256;
    size_t n = size_t(MemoryVolume());
    size_t gws0 = ((n + lws0 - 1) / lws0) * lws0;
    m_ndRange = ocl::NdRange(gws0, 1, 1, lws0, 1, 1);
}

} // namespace

//
//    TriSimple
//

TriSimple::TriSimple(Context *context):
        m_context(context) { }

TriSimple::~TriSimple() { }

std::unique_ptr<core::Node> TriSimple::CreateNode(core::Node *a, int mode) {
    std::unique_ptr<TriNode> node = std::make_unique<TriNode>(m_context, a, mode);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

