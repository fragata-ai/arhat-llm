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
#include "arhat/onednn/gpu/solve_tri.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    SolveTriNode
//

class SolveTriNode: public NodeBase {
public:
    SolveTriNode(
        Context *context,
        core::Node *a,
        core::Node *b,
        bool left,
        bool lower,
        bool uni);
    ~SolveTriNode();
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
    core::Node *m_a;
    core::Node *m_b;
    int m_left;
    int m_lower;
    int m_uni;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_cDesc;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

SolveTriNode::SolveTriNode(
        Context *context,
        core::Node *a,
        core::Node *b,
        bool left,
        bool lower,
        bool uni):
            NodeBase(context),
            m_a(a),
            m_b(b),
            m_left(left),
            m_lower(lower),
            m_uni(uni) { }

SolveTriNode::~SolveTriNode() { }

bool SolveTriNode::Init() {
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    m_aDesc = a->MemoryDesc();
    m_bDesc = b->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    m_cDesc = base::PlainMemoryDesc(m_bDesc);
    SetMemory(m_cDesc);
    InitKernel();
    return true;
}

void SolveTriNode::Compute() {
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_bMem);
    m_kernel->SetArgBuffer(2, m_memory);
    m_kernel->Launch(m_ndRange);
}

bool SolveTriNode::Validate() {
    if (!m_left || !m_lower || m_uni) {
        return false;
    }
    if (m_aDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (m_bDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!base::IsRowMajor(m_aDesc)) {
        return false;
    }
    if (!base::IsRowMajor(m_bDesc)) {
        return false;
    }
    return true;
}

void SolveTriNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::SolveTriSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "solve_tri_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string SolveTriNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("tri_simple");
    sb.MemoryDesc(m_aDesc);
    sb.MemoryDesc(m_bDesc);
    sb.Int(m_left);
    sb.Int(m_lower);
    sb.Int(m_uni);
    return sb.Get();
}

std::string SolveTriNode::MakeProlog() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    std::stringstream ss;
    EmitDesc(ss, "SRC0", m_aDesc);
    EmitDesc(ss, "SRC1", m_bDesc);
    EmitDesc(ss, "DST", m_cDesc);
    EmitInt(ss, "N", int64_t(aDims[3]));
    ss << "\n";
    return ss.str();
}

void SolveTriNode::InitNdRange() {
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    size_t lws0 = 16;
    size_t lws1 = 4;
    size_t lws2 = 1;
    size_t gws0 = ((size_t(bDims[3]) + lws0 - 1) / lws0) * lws0;
    size_t gws1 = ((size_t(bDims[1]) + lws1 - 1) / lws1) * lws1;
    size_t gws2 = ((size_t(bDims[0]) + lws2 - 1) / lws2) * lws2;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, lws2);
}

} // namespace

//
//    SolveTriSimple
//

SolveTriSimple::SolveTriSimple(Context *context):
        m_context(context) { }

SolveTriSimple::~SolveTriSimple() { }

std::unique_ptr<core::Node> SolveTriSimple::CreateNode(
        core::Node *a,
        core::Node *b,
        bool left,
        bool lower,
        bool uni) {
    std::unique_ptr<SolveTriNode> node = 
        std::make_unique<SolveTriNode>(m_context, a, b, left, lower, uni);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

