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

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/mul_mat.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    MulMatNode
//

class MulMatNode: public NodeBase {
public:
    MulMatNode(
        Context *context,
        core::Node *a, 
        core::Node *b);
    ~MulMatNode();
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
    static void EmitBase(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::desc &desc);
    static void EmitDims(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::dims &dims);
    static void EmitStrides(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::dims &strides);
protected:
    virtual std::string GetKernelName() = 0;
    virtual const char *GetKernelCode() = 0;
    virtual void InitNdRange() = 0;
protected:
    core::Node *m_a;
    core::Node *m_b;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_cDesc;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::memory::dims m_bDims;
    base::QuantMode m_bQuant;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

MulMatNode::MulMatNode(
        Context *context,
        core::Node *a, 
        core::Node *b):
            NodeBase(context),
            m_a(a),
            m_b(b),
            m_bQuant(base::QuantMode::None) { }

MulMatNode::~MulMatNode() { }

bool MulMatNode::Init() {
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    m_aDesc = a->MemoryDesc();
    m_bDesc = b->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    // cannot use n_bDesc.get_dims() for quantized tensors
    m_bDims = b->MemoryDims();
    m_bQuant = b->Quant();
    InferShapes();
    SetMemory(m_cDesc);
    InitKernel();
    return true;
}

void MulMatNode::Compute() {
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_bMem);
    m_kernel->SetArgBuffer(2, m_memory);
    m_kernel->Launch(m_ndRange);
}

bool MulMatNode::Validate() {
    if (m_aDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_aDesc) || !MemoryDescUtil::HasDenseRows(m_aDesc)) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_bDesc) || !MemoryDescUtil::HasDenseRows(m_bDesc)) {
        return false;
    }
    return true;
}

void MulMatNode::InferShapes() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDims;
    assert(aDims.size() == 4);
    assert(bDims.size() == 4);
    dnnl::memory::dims cDims(4);
    cDims[0] = aDims[0];
    cDims[1] = aDims[1];
    cDims[2] = aDims[2];
    cDims[3] = bDims[2];
    m_cDesc = 
        dnnl::memory::desc(
            cDims, 
            dnnl::memory::data_type::f32, 
            dnnl::memory::format_tag::abcd);
}

void MulMatNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    std::string kernelName = GetKernelName();
    const char *kernelCode = GetKernelCode();
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

std::string MulMatNode::MakeSig() {
    NodeSigBuilder sb;
    std::string kernelName = GetKernelName();
    sb.String(kernelName);
    sb.MemoryDesc(m_aDesc);
    // it is ok to use physical rather than logcal dimensions in signatures
    sb.MemoryDesc(m_bDesc);
    sb.Int(int64_t(m_bQuant));
    return sb.Get();
}

std::string MulMatNode::MakeProlog() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims cDims = m_cDesc.get_dims();
    std::stringstream ss;
    EmitBase(ss, "SRC0", m_aDesc);
    EmitBase(ss, "SRC1", m_bDesc);
    EmitBase(ss, "DST", m_cDesc);
    EmitDims(ss, "SRC0", aDims);
    EmitDims(ss, "SRC1", m_bDims);
    EmitDims(ss, "DST", cDims);
    EmitStrides(ss, "SRC0", m_aDesc.get_strides());
    EmitStrides(ss, "SRC1", m_bDesc.get_strides());
    EmitStrides(ss, "DST", m_cDesc.get_strides());
    int64_t r0 = int64_t(aDims[0] / m_bDims[0]);
    int64_t r1 = int64_t(aDims[1] / m_bDims[1]);
    EmitInt(ss, "R0", r0);
    EmitInt(ss, "R1", r1);
    ss << "\n";
    return ss.str();
}

void MulMatNode::EmitBase(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::desc &desc) {
    EmitInt(os, prefix + "_BASE", int64_t(desc.get_submemory_offset()));
}

void MulMatNode::EmitDims(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::dims &dims) {
    EmitInt(os, prefix + "_D0", int64_t(dims[0]));
    EmitInt(os, prefix + "_D1", int64_t(dims[1]));
    EmitInt(os, prefix + "_D2", int64_t(dims[2]));
    EmitInt(os, prefix + "_D3", int64_t(dims[3]));
}

void MulMatNode::EmitStrides(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::dims &strides) {
    EmitInt(os, prefix + "_S0", int64_t(strides[0]));
    EmitInt(os, prefix + "_S1", int64_t(strides[1]));
    EmitInt(os, prefix + "_S2", int64_t(strides[2]));
    EmitInt(os, prefix + "_S3", int64_t(strides[3]));
}

//
//    MulMatNode_Q4_0
//

class MulMatNode_Q4_0: public MulMatNode {
public:
    MulMatNode_Q4_0(
        Context *context,
        core::Node *a, 
        core::Node *b);
    ~MulMatNode_Q4_0();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatNode_Q4_0::MulMatNode_Q4_0(
        Context *context,
        core::Node *a, 
        core::Node *b):
            MulMatNode(context, a, b) { }

MulMatNode_Q4_0::~MulMatNode_Q4_0() { }

std::string MulMatNode_Q4_0::GetKernelName() {
    return "mm_simple_q4_0";
}

const char *MulMatNode_Q4_0::GetKernelCode() {
    return kernels::MulMatQuantSimple_Q4_0_KernelCode();
}

void MulMatNode_Q4_0::InitNdRange() {
    // Use 1D local size. Each workgroup is a SIMD group.
    // Each SIMD group produces N_DST (4 for Q4_0 kernel) values in the result.
    // The number of workgroups on dim 0 (the leading dimension) is
    // the nearest multiple of 4 that covers cDims[3] (equals bDims[2]). 
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t aDim0 = size_t(aDims[0]);
    size_t aDim1 = size_t(aDims[1]);
    size_t aDim2 = size_t(aDims[2]);
    size_t bDim2 = size_t(m_bDims[2]);
    size_t lws0 = 16;
    size_t lws1 = 1;
    size_t ndst = 4;
    size_t gws0 = ((bDim2 + ndst - 1) / ndst) * lws0;
    size_t gws1 = aDim2 * lws1;
    size_t gws2 = aDim0 * aDim1;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

//
//    MulMatNode_Q4_K
//

class MulMatNode_Q4_K: public MulMatNode {
public:
    MulMatNode_Q4_K(
        Context *context,
        core::Node *a, 
        core::Node *b);
    ~MulMatNode_Q4_K();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatNode_Q4_K::MulMatNode_Q4_K(
        Context *context,
        core::Node *a, 
        core::Node *b):
            MulMatNode(context, a, b) { }

MulMatNode_Q4_K::~MulMatNode_Q4_K() { }

std::string MulMatNode_Q4_K::GetKernelName() {
    return "mm_simple_q4_k";
}

const char *MulMatNode_Q4_K::GetKernelCode() {
    return kernels::MulMatQuantSimple_Q4_K_KernelCode();
}

void MulMatNode_Q4_K::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t aDim0 = size_t(aDims[0]);
    size_t aDim1 = size_t(aDims[1]);
    size_t aDim2 = size_t(aDims[2]);
    size_t bDim2 = size_t(m_bDims[2]);
    size_t lws0 = 16;
    size_t lws1 = 1;
    size_t ndst = 4;
    size_t gws0 = ((bDim2 + ndst * lws1 - 1) / (ndst * lws1)) * lws0;
    size_t gws1 = aDim2 * lws1;
    size_t gws2 = aDim0 * aDim1;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

//
//    MulMatNode_Q5_0
//

class MulMatNode_Q5_0: public MulMatNode {
public:
    MulMatNode_Q5_0(
        Context *context,
        core::Node *a, 
        core::Node *b);
    ~MulMatNode_Q5_0();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatNode_Q5_0::MulMatNode_Q5_0(
        Context *context,
        core::Node *a, 
        core::Node *b):
            MulMatNode(context, a, b) { }

MulMatNode_Q5_0::~MulMatNode_Q5_0() { }

std::string MulMatNode_Q5_0::GetKernelName() {
    return "mm_simple_q5_0";
}

const char *MulMatNode_Q5_0::GetKernelCode() {
    return kernels::MulMatQuantSimple_Q5_0_KernelCode();
}

void MulMatNode_Q5_0::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t aDim0 = size_t(aDims[0]);
    size_t aDim1 = size_t(aDims[1]);
    size_t aDim2 = size_t(aDims[2]);
    size_t bDim2 = size_t(m_bDims[2]);
    size_t lws0 = 16;
    size_t lws1 = 1;
    size_t ndst = 4;
    size_t gws0 = ((bDim2 + ndst - 1) / ndst) * lws0;
    size_t gws1 = aDim2 * lws1;
    size_t gws2 = aDim0 * aDim1;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

//
//    MulMatNode_Q6_K
//

class MulMatNode_Q6_K: public MulMatNode {
public:
    MulMatNode_Q6_K(
        Context *context,
        core::Node *a, 
        core::Node *b);
    ~MulMatNode_Q6_K();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatNode_Q6_K::MulMatNode_Q6_K(
        Context *context,
        core::Node *a, 
        core::Node *b):
            MulMatNode(context, a, b) { }

MulMatNode_Q6_K::~MulMatNode_Q6_K() { }

std::string MulMatNode_Q6_K::GetKernelName() {
    return "mm_simple_q6_k";
}

const char *MulMatNode_Q6_K::GetKernelCode() {
    return kernels::MulMatQuantSimple_Q6_K_KernelCode();
}

void MulMatNode_Q6_K::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t aDim0 = size_t(aDims[0]);
    size_t aDim1 = size_t(aDims[1]);
    size_t aDim2 = size_t(aDims[2]);
    size_t bDim2 = size_t(m_bDims[2]);
    size_t lws0 = 16;
    size_t lws1 = 2;
    size_t ndst = 1;
    size_t gws0 = ((bDim2 + ndst * lws1 - 1) / (ndst * lws1)) * lws0;
    size_t gws1 = aDim2 * lws1;
    size_t gws2 = aDim0 * aDim1;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

//
//    MulMatNode_Q8_0
//

class MulMatNode_Q8_0: public MulMatNode {
public:
    MulMatNode_Q8_0(
        Context *context,
        core::Node *a, 
        core::Node *b);
    ~MulMatNode_Q8_0();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatNode_Q8_0::MulMatNode_Q8_0(
        Context *context,
        core::Node *a, 
        core::Node *b):
            MulMatNode(context, a, b) { }

MulMatNode_Q8_0::~MulMatNode_Q8_0() { }

std::string MulMatNode_Q8_0::GetKernelName() {
    return "mm_simple_q8_0";
}

const char *MulMatNode_Q8_0::GetKernelCode() {
    return kernels::MulMatQuantSimple_Q8_0_KernelCode();
}

void MulMatNode_Q8_0::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t aDim0 = size_t(aDims[0]);
    size_t aDim1 = size_t(aDims[1]);
    size_t aDim2 = size_t(aDims[2]);
    size_t bDim2 = size_t(m_bDims[2]);
    size_t lws0 = 16;
    size_t lws1 = 2;
    size_t ndst = lws1 * 4;
    size_t gws0 = ((bDim2 + ndst - 1) / ndst) * lws0;
    size_t gws1 = aDim2 * lws1;
    size_t gws2 = aDim0 * aDim1;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

//
//    MulMatNode_Mxfp4
//

class MulMatNode_Mxfp4: public MulMatNode {
public:
    MulMatNode_Mxfp4(
        Context *context,
        core::Node *a, 
        core::Node *b);
    ~MulMatNode_Mxfp4();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatNode_Mxfp4::MulMatNode_Mxfp4(
        Context *context,
        core::Node *a, 
        core::Node *b):
            MulMatNode(context, a, b) { }

MulMatNode_Mxfp4::~MulMatNode_Mxfp4() { }

std::string MulMatNode_Mxfp4::GetKernelName() {
    return "mm_simple_mxfp4";
}

const char *MulMatNode_Mxfp4::GetKernelCode() {
    return kernels::MulMatQuantSimple_Mxfp4_KernelCode();
}

void MulMatNode_Mxfp4::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t aDim0 = size_t(aDims[0]);
    size_t aDim1 = size_t(aDims[1]);
    size_t aDim2 = size_t(aDims[2]);
    size_t bDim2 = size_t(m_bDims[2]);
    size_t lws0 = 16;
    size_t lws1 = 2;
    size_t ndst = lws1 * 2;
    size_t gws0 = ((bDim2 + ndst - 1) / ndst) * lws0;
    size_t gws1 = aDim2 * lws1;
    size_t gws2 = aDim0 * aDim1;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

} // namespace

//
//    MulMatQuantSimple
//

MulMatQuantSimple::MulMatQuantSimple(Context *context):
        m_context(context) { }

MulMatQuantSimple::~MulMatQuantSimple() { }

std::unique_ptr<core::Node> MulMatQuantSimple::CreateNode(core::Node *a, core::Node *b) {
    base::NodeBase *bBase = m_context->CastNode(b);
    switch (bBase->Quant()) {
    case base::QuantMode::Q4_0:
        {
            std::unique_ptr<MulMatNode_Q4_0> node =
                std::make_unique<MulMatNode_Q4_0>(m_context, a, b);
            if (node->Init()) {
                return node;
            }
        }
        break;
    case base::QuantMode::Q4_K:
        {
            std::unique_ptr<MulMatNode_Q4_K> node =
                std::make_unique<MulMatNode_Q4_K>(m_context, a, b);
            if (node->Init()) {
                return node;
            }
        }
        break;
    case base::QuantMode::Q5_0:
        {
            std::unique_ptr<MulMatNode_Q5_0> node =
                std::make_unique<MulMatNode_Q5_0>(m_context, a, b);
            if (node->Init()) {
                return node;
            }
        }
        break;
    case base::QuantMode::Q6_K:
        {
            std::unique_ptr<MulMatNode_Q6_K> node =
                std::make_unique<MulMatNode_Q6_K>(m_context, a, b);
            if (node->Init()) {
                return node;
            }
        }
        break;
    case base::QuantMode::Q8_0:
        {
            std::unique_ptr<MulMatNode_Q8_0> node =
                std::make_unique<MulMatNode_Q8_0>(m_context, a, b);
            if (node->Init()) {
                return node;
            }
        }
        break;
    case base::QuantMode::MXFP4:
        {
            std::unique_ptr<MulMatNode_Mxfp4> node =
                std::make_unique<MulMatNode_Mxfp4>(m_context, a, b);
            if (node->Init()) {
                return node;
            }
        }
        break;
    default:
        break;
    }
    return nullptr;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

