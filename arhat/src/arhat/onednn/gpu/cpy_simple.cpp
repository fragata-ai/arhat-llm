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
#include "arhat/onednn/gpu/cpy.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    CpyNode
//

class CpyNode: public NodeBase {
public:
    CpyNode(
        Context *context,
        core::Node *a,
        core::Node *b);
    ~CpyNode();
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
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

CpyNode::CpyNode(
        Context *context,
        core::Node *a,
        core::Node *b):
            NodeBase(context),
            m_a(a),
            m_b(b) { }

CpyNode::~CpyNode() { }

bool CpyNode::Init() { 
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    m_aDesc = a->MemoryDesc();
    m_bDesc = b->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    SetMemory(m_bDesc, m_bMem);
    if (b->MemoryVolume() == 0) {
        return true;
    }
    InitKernel();
    return true;
}

void CpyNode::Compute() {
    if (!m_kernel) {
        return;
    }
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_bMem);
    m_kernel->Launch(m_ndRange);
}

bool CpyNode::Validate() {
    dnnl::memory::data_type aType = m_aDesc.get_data_type();
    if (aType != dnnl::memory::data_type::f32 &&
            aType != dnnl::memory::data_type::f16) {
        return false;
    }
    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    if (bType != dnnl::memory::data_type::f32 &&
            bType != dnnl::memory::data_type::f16) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_aDesc)) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_bDesc)) {
        return false;
    }
    return true;
}

void CpyNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::CpySimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "cpy_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string CpyNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("cpy_simple");
    sb.MemoryDesc(m_aDesc);
    sb.MemoryDesc(m_bDesc);
    return sb.Get();
}

std::string CpyNode::MakeProlog() {
    std::stringstream ss;
    dnnl::memory::data_type aType = m_aDesc.get_data_type();
    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    ss << "#define SRC_TYPE " << ocl::FormatType(aType) << "\n";
    ss << "#define DST_TYPE " << ocl::FormatType(bType) << "\n";
    ss << "\n";
    EmitDesc(ss, "SRC", m_aDesc);
    EmitDesc(ss, "DST", m_bDesc);
    return ss.str();
}

void CpyNode::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t lws0 = size_t(aDims[3]);
    if (lws0 > 64) {
        lws0 = 64;
    }
    size_t gws0 = size_t(aDims[2]) * lws0;
    size_t gws1 = size_t(aDims[1]);
    size_t gws2 = size_t(aDims[0]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

} // namespace

//
//    CpySimple
//

CpySimple::CpySimple(Context *context):
        m_context(context) { }

CpySimple::~CpySimple() { }

std::unique_ptr<core::Node> CpySimple::CreateNode(core::Node *a, core::Node *b) {
    std::unique_ptr<CpyNode> node = std::make_unique<CpyNode>(m_context, a, b);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

