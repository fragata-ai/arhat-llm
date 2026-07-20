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
#include "arhat/onednn/gpu/set.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    SetNode
//

class SetNode: public NodeBase {
public:
    SetNode(
        Context *context,
        core::Node *a,
        core::Node *b,
        const core::Dims &stride,
        int offset,
        bool inplace);
    ~SetNode();
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
    core::Dims m_stride;
    int m_offset;
    bool m_inplace;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    base::Reorder m_reorder;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

SetNode::SetNode(
        Context *context,
        core::Node *a,
        core::Node *b,
        const core::Dims &stride,
        int offset,
        bool inplace):
            NodeBase(context),
            m_a(a),
            m_b(b),
            m_stride(stride), 
            m_offset(offset),
            m_inplace(inplace) { }

SetNode::~SetNode() { }

bool SetNode::Init() { 
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    m_aDesc = a->MemoryDesc();
    m_bDesc = b->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    if (m_inplace) {
        SetMemory(m_aDesc, m_aMem);
    } else {
        dnnl::memory::desc cDesc = base::PlainMemoryDesc(m_aDesc);
        SetMemory(cDesc);
        m_reorder.Init(m_context, m_aDesc, cDesc);
    }
    InitKernel();
    return true;
}

void SetNode::Compute() {
    if (m_reorder.IsSet()) {
        m_reorder.Compute(m_aMem, m_memory);
    }
    m_kernel->SetArgBuffer(0, m_bMem);
    m_kernel->SetArgBuffer(1, m_memory);
    m_kernel->Launch(m_ndRange);
}

bool SetNode::Validate() {
    if (m_aDesc.get_data_type() != m_bDesc.get_data_type()) {
        return false;
    }
    // TODO: Check Quant = None
    if (!MemoryDescUtil::IsPlain(m_aDesc)) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_bDesc)) {
        return false;
    }
    return true;
}

void SetNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::SetSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "set_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string SetNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("set_simple");
    sb.MemoryDesc(m_aDesc);
    sb.MemoryDesc(m_bDesc);
    sb.Int(m_stride[3]);
    sb.Int(m_stride[2]);
    sb.Int(m_stride[1]);
    sb.Int(m_stride[0]);
    sb.Int(m_offset);
    sb.Bool(m_inplace);
    return sb.Get();
}

std::string SetNode::MakeProlog() {
    std::stringstream ss;
    dnnl::memory::data_type aType = m_aDesc.get_data_type();
    ss << "#define DATA_T " << ocl::FormatType(aType) << "\n";
    ss << "\n";
    EmitDesc(ss, "SRC", m_bDesc);
    EmitDesc(ss, "DST", MemoryDesc());
    EmitInt(ss, "PS0", int64_t(m_stride[3]));
    EmitInt(ss, "PS1", int64_t(m_stride[2]));
    EmitInt(ss, "PS2", int64_t(m_stride[1]));
    EmitInt(ss, "PS3", int64_t(m_stride[0]));
    EmitInt(ss, "POFFS", int64_t(m_offset));
    return ss.str();
}

void SetNode::InitNdRange() {
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    size_t bd3 = size_t(bDims[3]);
    // TODO: Implement common utility function for lws0 computation
    //     (see also, for example, cumsum_simple)
    ocl::OclDeviceInfo info = m_oclContext->GetDeviceInfo();
    size_t maxWgs = info.maxWorkGroupSize;
    size_t lws0 = 1;
    while (lws0 < bd3 && 2 * lws0 <= maxWgs) {
        lws0 *= 2;
    }
    size_t gws0 = size_t(bDims[2]) * lws0;
    size_t gws1 = size_t(bDims[1]);
    size_t gws2 = size_t(bDims[0]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

} // namespace

//
//    SetSimple
//

SetSimple::SetSimple(Context *context):
        m_context(context) { }

SetSimple::~SetSimple() { }

std::unique_ptr<core::Node> SetSimple::CreateNode(
        core::Node *a, 
        core::Node *b,
        const core::Dims &stride,
        int offset,
        bool inplace) {
    std::unique_ptr<SetNode> node = 
        std::make_unique<SetNode>(m_context, a, b, stride, offset, inplace);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

