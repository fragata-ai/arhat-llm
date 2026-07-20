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
#include "arhat/onednn/gpu/add_id.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    AddIdNode
//

class AddIdNode: public NodeBase {
public:
    AddIdNode(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
    ~AddIdNode();
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
    core::Node *m_ids;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_idsDesc;
    dnnl::memory::desc m_cDesc;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::memory m_idsMem;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

AddIdNode::AddIdNode(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids):
            NodeBase(context),
            m_a(a),
            m_b(b),
            m_ids(ids) { }

AddIdNode::~AddIdNode() { }

bool AddIdNode::Init() {
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    base::NodeBase *ids = m_gpuContext->CastNode(m_ids);
    m_aDesc = a->MemoryDesc();
    m_bDesc = b->MemoryDesc();
    m_idsDesc = ids->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    m_idsMem = ids->Memory();
    InferShapes();
    SetMemory(m_cDesc);
    InitKernel();
    return true;
}

void AddIdNode::Compute() {
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_bMem);
    m_kernel->SetArgBuffer(2, m_idsMem);
    m_kernel->SetArgBuffer(3, m_memory);
    m_kernel->Launch(m_ndRange);
}

bool AddIdNode::Validate() {
    if (m_aDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (m_bDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (m_idsDesc.get_data_type() != dnnl::memory::data_type::s32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_aDesc) || !MemoryDescUtil::HasDenseRows(m_aDesc)) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_bDesc) || !MemoryDescUtil::HasDenseRows(m_bDesc)) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_idsDesc)) {
        return false;
    }
    return true;
}

void AddIdNode::InferShapes() {
    m_cDesc = base::PlainMemoryDesc(m_aDesc);
}

void AddIdNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::AddIdSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "add_id_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string AddIdNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("add_id_simple");
    sb.MemoryDesc(m_aDesc);
    sb.MemoryDesc(m_bDesc);
    sb.MemoryDesc(m_idsDesc);
    return sb.Get();
}

std::string AddIdNode::MakeProlog() {
    std::stringstream ss;
    EmitDesc(ss, "SRC0", m_aDesc);
    EmitDesc(ss, "SRC1", m_bDesc);
    EmitDesc(ss, "SRC2", m_idsDesc);
    EmitDesc(ss, "DST", m_cDesc);
    return ss.str();
}

void AddIdNode::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    ocl::OclDeviceInfo info = GetOclContext()->GetDeviceInfo();
    size_t lws0 = std::min(size_t(aDims[3]), info.maxWorkGroupSize);
    size_t gws0 = size_t(aDims[2]) * lws0;
    size_t gws1 = size_t(aDims[1]);
    m_ndRange = ocl::NdRange(gws0, gws1, 1, lws0, 1, 1);
}

} // namespace

//
//    AddIdSimple
//

AddIdSimple::AddIdSimple(Context *context):
        m_context(context) { }

AddIdSimple::~AddIdSimple() { }

std::unique_ptr<core::Node> AddIdSimple::CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *ids) {
    std::unique_ptr<AddIdNode> node = std::make_unique<AddIdNode>(m_context, a, b, ids);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

