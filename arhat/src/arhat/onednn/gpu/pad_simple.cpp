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
#include "arhat/onednn/gpu/pad.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    PadNode
//

class PadNode: public NodeBase {
public:
    PadNode(
        Context *context,
        core::Node *x,
        int lp0,
        int rp0,
        int lp1,
        int rp1,
        int lp2,
        int rp2,
        int lp3,
        int rp3,
        bool circular);
    ~PadNode();
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
    int m_lp0;
    int m_rp0;
    int m_lp1;
    int m_rp1;
    int m_lp2;
    int m_rp2;
    int m_lp3;
    int m_rp3;
    bool m_circular;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory m_xMem;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

PadNode::PadNode(
        Context *context,
        core::Node *x,
        int lp0,
        int rp0,
        int lp1,
        int rp1,
        int lp2,
        int rp2,
        int lp3,
        int rp3,
        bool circular):
            NodeBase(context),
            m_x(x),
            m_lp0(lp0),
            m_rp0(rp0),
            m_lp1(lp1),
            m_rp1(rp1),
            m_lp2(lp2),
            m_rp2(rp2),
            m_lp3(lp3),
            m_rp3(rp3),
            m_circular(circular) { }

PadNode::~PadNode() { }

bool PadNode::Init() {
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

void PadNode::Compute() {
    m_kernel->SetArgBuffer(0, m_xMem);
    m_kernel->SetArgBuffer(1, m_memory);
    m_kernel->Launch(m_ndRange);
}

bool PadNode::Validate() {
    base::NodeBase *x = m_gpuContext->CastNode(m_x);
    if (x->Quant() != base::QuantMode::None) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_xDesc)) {
        return false;
    }
    return true;
}

void PadNode::InferShapes() {
    dnnl::memory::dims yDims = m_xDesc.get_dims();
    yDims[0] += dnnl::memory::dim(m_lp3 + m_rp3);
    yDims[1] += dnnl::memory::dim(m_lp2 + m_rp2);
    yDims[2] += dnnl::memory::dim(m_lp1 + m_rp1);
    yDims[3] += dnnl::memory::dim(m_lp0 + m_rp0);
    m_yDesc =
        dnnl::memory::desc(
            yDims, 
            m_xDesc.get_data_type(), 
            dnnl::memory::format_tag::abcd);
}

void PadNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    std::string kernelName = m_circular ? "pad_simple_circular" : "pad_simple";
    const char *kernelCode = 
        m_circular ?
            kernels::PadSimpleCircularKernelCode() :
            kernels::PadSimpleKernelCode();
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

std::string PadNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("pad_simple");
    sb.MemoryDesc(m_xDesc);
    sb.Int(m_lp0);
    sb.Int(m_rp0);
    sb.Int(m_lp1);
    sb.Int(m_rp1);
    sb.Int(m_lp2);
    sb.Int(m_rp2);
    sb.Int(m_lp3);
    sb.Int(m_rp3);
    sb.Bool(m_circular);
    return sb.Get();
}

std::string PadNode::MakeProlog() {
    std::stringstream ss;
    dnnl::memory::data_type xType = m_xDesc.get_data_type();
    ss << "#define DATA_T " << ocl::FormatType(xType) << "\n";
    ss << "\n";
    EmitDesc(ss, "SRC", m_xDesc);
    EmitDesc(ss, "DST", m_yDesc);
    EmitInt(ss, "LP0", m_lp3);
    EmitInt(ss, "RP0", m_rp3);
    EmitInt(ss, "LP1", m_lp2);
    EmitInt(ss, "RP1", m_rp2);
    EmitInt(ss, "LP2", m_lp1);
    EmitInt(ss, "RP2", m_rp1);
    EmitInt(ss, "LP3", m_lp0);
    EmitInt(ss, "RP3", m_rp0);
    ss << "\n";
    return ss.str();
}

void PadNode::InitNdRange() {
    dnnl::memory::dims yDims = m_yDesc.get_dims();
    size_t lws0 = 64;
    size_t gws0 = ((size_t(yDims[3]) + lws0 - 1) / lws0) * lws0;
    size_t gws1 = size_t(yDims[2]);
    size_t gws2 = size_t(yDims[0]) * size_t(yDims[1]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

} // namespace

//
//    PadSimple
//

PadSimple::PadSimple(Context *context):
        m_context(context) { }

PadSimple::~PadSimple() { }

std::unique_ptr<core::Node> PadSimple::CreateNode(
        core::Node *a, 
        int lp0,
        int rp0,
        int lp1,
        int rp1,
        int lp2,
        int rp2,
        int lp3,
        int rp3,
        bool circular) {
    std::unique_ptr<PadNode> node = 
        std::make_unique<PadNode>(m_context, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3, circular);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

