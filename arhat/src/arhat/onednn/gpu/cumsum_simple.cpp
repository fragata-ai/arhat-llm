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
#include "arhat/onednn/gpu/cumsum.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    CumSumNode
//

class CumSumNode: public NodeBase {
public:
    CumSumNode(Context *context, core::Node *x);
    ~CumSumNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    bool InitConfig();
    void InitKernelBlk1();
    void InitKernelBlk2();
    void InitKernelAdd();
    std::string MakeSigBlk1();
    std::string MakeSigBlk2();
    std::string MakeSigAdd();
    std::string MakePrologBlk1();
    std::string MakePrologBlk2();
    std::string MakePrologAdd();
    void InitNdRangeBlk1();
    void InitNdRangeBlk2();
    void InitNdRangeAdd();
private:
    core::Node *m_x;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory m_xMem;
    size_t m_lws0;
    bool m_multiPass;
    dnnl::memory::desc m_tmpDesc;
    dnnl::memory m_tmpMem;
    std::shared_ptr<ocl::Kernel> m_kernelBlk1;
    std::shared_ptr<ocl::Kernel> m_kernelBlk2;
    std::shared_ptr<ocl::Kernel> m_kernelAdd;
    ocl::NdRange m_ndRangeBlk1;
    ocl::NdRange m_ndRangeBlk2;
    ocl::NdRange m_ndRangeAdd;
};

CumSumNode::CumSumNode(Context *context, core::Node *x):
        NodeBase(context), 
        m_x(x),
        m_lws0(0),
        m_multiPass(false) { }

CumSumNode::~CumSumNode() { }

bool CumSumNode::Init() {
    base::NodeBase *x = m_gpuContext->CastNode(m_x);
    m_xDesc = x->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    if (!InitConfig()) {
        return false;
    }
    m_xMem = x->Memory();
    m_yDesc = base::PlainMemoryDesc(m_xDesc);
    SetMemory(m_yDesc);
    InitKernelBlk1();
    if (m_multiPass) {
        InitKernelBlk2();
        InitKernelAdd();
    }
    return true;
}

void CumSumNode::Compute() {
    m_kernelBlk1->SetArgBuffer(0, m_xMem);
    m_kernelBlk1->SetArgBuffer(1, m_tmpMem);
    m_kernelBlk1->SetArgBuffer(2, m_memory);
    m_kernelBlk1->Launch(m_ndRangeBlk1);
    if (m_multiPass) {
        m_kernelBlk2->SetArgBuffer(0, m_tmpMem);
        m_kernelBlk2->SetArgBuffer(1, m_tmpMem);
        m_kernelBlk2->SetArgBuffer(2, m_tmpMem);
        m_kernelBlk2->Launch(m_ndRangeBlk2);
        m_kernelAdd->SetArgBuffer(0, m_tmpMem);
        m_kernelAdd->SetArgBuffer(1, m_memory);
        m_kernelAdd->Launch(m_ndRangeAdd);
    }
}

bool CumSumNode::Validate() {
    if (m_xDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_xDesc) ||
            !MemoryDescUtil::HasDenseRows(m_xDesc)) {
        return false;
    }
    return true;
}

bool CumSumNode::InitConfig() {
    dnnl::memory::dims xDims = m_xDesc.get_dims();
    size_t xd3 = size_t(xDims[3]);
    ocl::OclDeviceInfo info = m_oclContext->GetDeviceInfo();
    size_t maxWgs = info.maxWorkGroupSize;
    m_lws0 = 1;
    while (m_lws0 < xd3 && 2 * m_lws0 <= maxWgs) {
        m_lws0 *= 2;
    }
    if (m_lws0 * m_lws0 < xd3) {
        return false;
    }
    size_t td3 = (xd3 + m_lws0 - 1) / m_lws0;
    dnnl::memory::dims tmpDims(xDims);
    tmpDims[3] = dnnl::memory::dim(td3);
    m_tmpDesc = 
        dnnl::memory::desc(
            tmpDims, 
            m_xDesc.get_data_type(), 
            dnnl::memory::format_tag::abcd);
    m_multiPass = (xd3 > m_lws0);
    if (m_multiPass) {
        m_tmpMem = dnnl::memory(m_tmpDesc, Engine());
    }
    return true;
}

void CumSumNode::InitKernelBlk1() {
    std::string sig = MakeSigBlk1();
    if (FindKernel(sig, m_kernelBlk1, m_ndRangeBlk1)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakePrologBlk1();
    const char *kernelCode = kernels::CumSumSimpleBlkKernelCode();
    m_kernelBlk1 = std::make_shared<ocl::Kernel>();
    m_kernelBlk1->Init(
        m_oclContext, 
        "cumsum_simple_blk", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRangeBlk1();
    EnterKernel(sig, m_kernelBlk1, m_ndRangeBlk1);
}

void CumSumNode::InitKernelBlk2() {
    std::string sig = MakeSigBlk2();
    if (FindKernel(sig, m_kernelBlk2, m_ndRangeBlk2)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakePrologBlk2();
    const char *kernelCode = kernels::CumSumSimpleBlkKernelCode();
    m_kernelBlk2 = std::make_shared<ocl::Kernel>();
    m_kernelBlk2->Init(
        m_oclContext, 
        "cumsum_simple_blk", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRangeBlk2();
    EnterKernel(sig, m_kernelBlk2, m_ndRangeBlk2);
}

void CumSumNode::InitKernelAdd() {
    std::string sig = MakeSigAdd();
    if (FindKernel(sig, m_kernelAdd, m_ndRangeAdd)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakePrologAdd();
    const char *kernelCode = kernels::CumSumSimpleAddKernelCode();
    m_kernelAdd = std::make_shared<ocl::Kernel>();
    m_kernelAdd->Init(
        m_oclContext, 
        "cumsum_simple_add", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRangeAdd();
    EnterKernel(sig, m_kernelAdd, m_ndRangeAdd);
}

std::string CumSumNode::MakeSigBlk1() {
    NodeSigBuilder sb;
    sb.String("cumsum_simple_blk1");
    sb.MemoryDesc(m_xDesc);
    return sb.Get();
}

std::string CumSumNode::MakeSigBlk2() {
    NodeSigBuilder sb;
    sb.String("cumsum_simple_blk2");
    sb.MemoryDesc(m_xDesc);
    return sb.Get();
}

std::string CumSumNode::MakeSigAdd() {
    NodeSigBuilder sb;
    sb.String("cumsum_simple_add");
    sb.MemoryDesc(m_xDesc);
    return sb.Get();
}

std::string CumSumNode::MakePrologBlk1() {
    std::stringstream ss;
    EmitDesc(ss, "SRC", m_xDesc);
    EmitDesc(ss, "TMP", m_tmpDesc);
    EmitDesc(ss, "DST", m_yDesc);
    return ss.str();
}

std::string CumSumNode::MakePrologBlk2() {
    std::stringstream ss;
    EmitDesc(ss, "SRC", m_tmpDesc);
    EmitDesc(ss, "TMP", m_tmpDesc);
    EmitDesc(ss, "DST", m_tmpDesc);
    return ss.str();
}

std::string CumSumNode::MakePrologAdd() {
    std::stringstream ss;
    EmitDesc(ss, "TMP", m_tmpDesc);
    EmitDesc(ss, "DST", m_yDesc);
    return ss.str();
}

void CumSumNode::InitNdRangeBlk1() {
    dnnl::memory::dims tmpDims = m_tmpDesc.get_dims();
    size_t gws0 = m_lws0 * size_t(tmpDims[2]) * size_t(tmpDims[3]);
    size_t gws1 = size_t(tmpDims[1]);
    size_t gws2 = size_t(tmpDims[0]);
    m_ndRangeBlk1 = ocl::NdRange(gws0, gws1, gws2, m_lws0, 1, 1);
}

void CumSumNode::InitNdRangeBlk2() {
    dnnl::memory::dims tmpDims = m_tmpDesc.get_dims();
    size_t gws0 = m_lws0 * size_t(tmpDims[2]);
    size_t gws1 = size_t(tmpDims[1]);
    size_t gws2 = size_t(tmpDims[0]);
    m_ndRangeBlk2 = ocl::NdRange(gws0, gws1, gws2, m_lws0, 1, 1);
}

void CumSumNode::InitNdRangeAdd() {
    dnnl::memory::dims tmpDims = m_tmpDesc.get_dims();
    size_t gws0 = m_lws0 * size_t(tmpDims[2]) * size_t(tmpDims[3]);
    size_t gws1 = size_t(tmpDims[1]);
    size_t gws2 = size_t(tmpDims[0]);
    m_ndRangeAdd = ocl::NdRange(gws0, gws1, gws2, m_lws0, 1, 1);
}

} // namespace

//
//    CumSumSimple
//

CumSumSimple::CumSumSimple(Context *context):
        m_context(context) { }

CumSumSimple::~CumSumSimple() { }

std::unique_ptr<core::Node> CumSumSimple::CreateNode(core::Node *a) {
    std::unique_ptr<CumSumNode> node = std::make_unique<CumSumNode>(m_context, a);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

