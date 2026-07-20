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
#include "arhat/onednn/gpu/mul_mat_id.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    MulMatIdNode
//

class MulMatIdNode: public NodeBase {
public:
    MulMatIdNode(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
    ~MulMatIdNode();
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
    core::Node *m_ids;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_idsDesc;
    dnnl::memory::desc m_cDesc;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::memory m_idsMem;
    dnnl::memory::dims m_bDims;
    base::QuantMode m_bQuant;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

MulMatIdNode::MulMatIdNode(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids):
            NodeBase(context),
            m_a(a),
            m_b(b),
            m_ids(ids),
            m_bQuant(base::QuantMode::None) { }

MulMatIdNode::~MulMatIdNode() { }

bool MulMatIdNode::Init() {
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
    // cannot use n_bDesc.get_dims() for quantized tensors
    m_bDims = b->MemoryDims();
    m_bQuant = b->Quant();
    InferShapes();
    SetMemory(m_cDesc);
    InitKernel();
    return true;
}

void MulMatIdNode::Compute() {
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_bMem);
    m_kernel->SetArgBuffer(2, m_idsMem);
    m_kernel->SetArgBuffer(3, m_memory);
    m_kernel->Launch(m_ndRange);
}

bool MulMatIdNode::Validate() {
    if (m_aDesc.get_data_type() != dnnl::memory::data_type::f32) {
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

void MulMatIdNode::InferShapes() {
    // a   [1, n_tokens, n_expert_used, cols]
    // b   [1, n_expert, rows, cols]
    // ids [1, 1, n_tokens, n_expert_used]
    // c   [1, n_tokens, n_expert_used, rows]
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDims;
    dnnl::memory::dims idsDims = m_idsDesc.get_dims();
    assert(aDims.size() == 4);
    assert(bDims.size() == 4);
    assert(idsDims.size() == 4);
    dnnl::memory::dims cDims(4);
    cDims[0] = 1;
    cDims[1] = aDims[1];
    cDims[2] = idsDims[3];
    cDims[3] = bDims[2];
    m_cDesc = 
        dnnl::memory::desc(
            cDims, 
            dnnl::memory::data_type::f32, 
            dnnl::memory::format_tag::abcd);
}

void MulMatIdNode::InitKernel() {
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

std::string MulMatIdNode::MakeSig() {
    NodeSigBuilder sb;
    std::string kernelName = GetKernelName();
    sb.String(kernelName);
    sb.MemoryDesc(m_aDesc);
    // it is ok to use physical rather than logcal dimensions in signatures
    sb.MemoryDesc(m_bDesc);
    sb.MemoryDesc(m_idsDesc);
    sb.Int(int64_t(m_bQuant));
    return sb.Get();
}

std::string MulMatIdNode::MakeProlog() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims idsDims = m_idsDesc.get_dims();
    dnnl::memory::dims cDims = m_cDesc.get_dims();
    std::stringstream ss;
    EmitBase(ss, "SRC0", m_aDesc);
    EmitBase(ss, "SRC1", m_bDesc);
    EmitBase(ss, "SRC2", m_idsDesc);
    EmitBase(ss, "DST", m_cDesc);
    EmitDims(ss, "SRC0", aDims);
    EmitDims(ss, "SRC1", m_bDims);
    EmitDims(ss, "SRC2", idsDims);
    EmitDims(ss, "DST", cDims);
    EmitStrides(ss, "SRC0", m_aDesc.get_strides());
    EmitStrides(ss, "SRC1", m_bDesc.get_strides());
    EmitStrides(ss, "SRC2", m_idsDesc.get_strides());
    EmitStrides(ss, "DST", m_cDesc.get_strides());
    return ss.str();
}

void MulMatIdNode::EmitBase(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::desc &desc) {
    EmitInt(os, prefix + "_BASE", int64_t(desc.get_submemory_offset()));
}

void MulMatIdNode::EmitDims(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::dims &dims) {
    EmitInt(os, prefix + "_D0", int64_t(dims[0]));
    EmitInt(os, prefix + "_D1", int64_t(dims[1]));
    EmitInt(os, prefix + "_D2", int64_t(dims[2]));
    EmitInt(os, prefix + "_D3", int64_t(dims[3]));
}

void MulMatIdNode::EmitStrides(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::dims &strides) {
    EmitInt(os, prefix + "_S0", int64_t(strides[0]));
    EmitInt(os, prefix + "_S1", int64_t(strides[1]));
    EmitInt(os, prefix + "_S2", int64_t(strides[2]));
    EmitInt(os, prefix + "_S3", int64_t(strides[3]));
}

//
//    MulMatIdNode_Q4_0
//

class MulMatIdNode_Q4_0: public MulMatIdNode {
public:
    MulMatIdNode_Q4_0(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
    ~MulMatIdNode_Q4_0();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatIdNode_Q4_0::MulMatIdNode_Q4_0(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids):
            MulMatIdNode(context, a, b, ids) { }

MulMatIdNode_Q4_0::~MulMatIdNode_Q4_0() { }

std::string MulMatIdNode_Q4_0::GetKernelName() {
    return "mul_mat_id_simple_q4_0";
}

const char *MulMatIdNode_Q4_0::GetKernelCode() {
    return kernels::MulMatIdQuantSimple_Q4_0_KernelCode();
}

void MulMatIdNode_Q4_0::InitNdRange() {
    dnnl::memory::dims idsDims = m_idsDesc.get_dims();
    size_t bDim2 = size_t(m_bDims[2]);   // rows
    size_t idsDim2 = size_t(idsDims[2]); // n_tokens
    size_t idsDim3 = size_t(idsDims[3]); // n_expert_used
    size_t lws0 = 16; // N_SIMDWIDTH
    size_t lws1 = 1;  // N_SIMDGROUP
    size_t ndst = 4;  // N_DST
    size_t gws0 = (bDim2 + ndst * lws1 - 1) / (ndst * lws1) * lws0;
    size_t gws1 = lws1;
    size_t gws2 = idsDim2 * idsDim3;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

//
//    MulMatIdNode_Q4_K
//

class MulMatIdNode_Q4_K: public MulMatIdNode {
public:
    MulMatIdNode_Q4_K(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
    ~MulMatIdNode_Q4_K();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatIdNode_Q4_K::MulMatIdNode_Q4_K(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids):
            MulMatIdNode(context, a, b, ids) { }

MulMatIdNode_Q4_K::~MulMatIdNode_Q4_K() { }

std::string MulMatIdNode_Q4_K::GetKernelName() {
    return "mul_mat_id_simple_q4_k";
}

const char *MulMatIdNode_Q4_K::GetKernelCode() {
    return kernels::MulMatIdQuantSimple_Q4_K_KernelCode();
}

void MulMatIdNode_Q4_K::InitNdRange() {
    dnnl::memory::dims idsDims = m_idsDesc.get_dims();
    size_t bDim2 = size_t(m_bDims[2]);   // rows
    size_t idsDim2 = size_t(idsDims[2]); // n_tokens
    size_t idsDim3 = size_t(idsDims[3]); // n_expert_used
    size_t lws0 = 16; // N_SIMDWIDTH
    size_t lws1 = 1;  // N_SIMDGROUP
    size_t ndst = 4;  // N_DST
    size_t gws0 = (bDim2 + ndst * lws1 - 1) / (ndst * lws1) * lws0;
    size_t gws1 = lws1;
    size_t gws2 = idsDim2 * idsDim3;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

//
//    MulMatIdNode_Q5_0
//

class MulMatIdNode_Q5_0: public MulMatIdNode {
public:
    MulMatIdNode_Q5_0(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
    ~MulMatIdNode_Q5_0();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatIdNode_Q5_0::MulMatIdNode_Q5_0(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids):
            MulMatIdNode(context, a, b, ids) { }

MulMatIdNode_Q5_0::~MulMatIdNode_Q5_0() { }

std::string MulMatIdNode_Q5_0::GetKernelName() {
    return "mul_mat_id_simple_q5_0";
}

const char *MulMatIdNode_Q5_0::GetKernelCode() {
    return kernels::MulMatIdQuantSimple_Q5_0_KernelCode();
}

void MulMatIdNode_Q5_0::InitNdRange() {
    dnnl::memory::dims idsDims = m_idsDesc.get_dims();
    size_t bDim2 = size_t(m_bDims[2]);   // rows
    size_t idsDim2 = size_t(idsDims[2]); // n_tokens
    size_t idsDim3 = size_t(idsDims[3]); // n_expert_used
    size_t lws0 = 16; // N_SIMDWIDTH
    size_t lws1 = 1;  // N_SIMDGROUP
    size_t ndst = 4;  // N_DST
    size_t gws0 = (bDim2 + ndst * lws1 - 1) / (ndst * lws1) * lws0;
    size_t gws1 = lws1;
    size_t gws2 = idsDim2 * idsDim3;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

//
//    MulMatIdNode_Q6_K
//

class MulMatIdNode_Q6_K: public MulMatIdNode {
public:
    MulMatIdNode_Q6_K(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
    ~MulMatIdNode_Q6_K();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatIdNode_Q6_K::MulMatIdNode_Q6_K(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids):
            MulMatIdNode(context, a, b, ids) { }

MulMatIdNode_Q6_K::~MulMatIdNode_Q6_K() { }

std::string MulMatIdNode_Q6_K::GetKernelName() {
    return "mul_mat_id_simple_q6_k";
}

const char *MulMatIdNode_Q6_K::GetKernelCode() {
    return kernels::MulMatIdQuantSimple_Q6_K_KernelCode();
}

void MulMatIdNode_Q6_K::InitNdRange() {
    dnnl::memory::dims idsDims = m_idsDesc.get_dims();
    size_t bDim2 = size_t(m_bDims[2]);   // rows
    size_t idsDim2 = size_t(idsDims[2]); // n_tokens
    size_t idsDim3 = size_t(idsDims[3]); // n_expert_used
    size_t lws0 = 16; // N_SIMDWIDTH
    size_t lws1 = 2;  // N_SIMDGROUP
    size_t ndst = 1;  // N_DST
    size_t gws0 = (bDim2 + ndst * lws1 - 1) / (ndst * lws1) * lws0;
    size_t gws1 = lws1;
    size_t gws2 = idsDim2 * idsDim3;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

//
//    MulMatIdNode_Q8_0
//

class MulMatIdNode_Q8_0: public MulMatIdNode {
public:
    MulMatIdNode_Q8_0(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
    ~MulMatIdNode_Q8_0();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatIdNode_Q8_0::MulMatIdNode_Q8_0(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids):
            MulMatIdNode(context, a, b, ids) { }

MulMatIdNode_Q8_0::~MulMatIdNode_Q8_0() { }

std::string MulMatIdNode_Q8_0::GetKernelName() {
    return "mul_mat_id_simple_q8_0";
}

const char *MulMatIdNode_Q8_0::GetKernelCode() {
    return kernels::MulMatIdQuantSimple_Q8_0_KernelCode();
}

void MulMatIdNode_Q8_0::InitNdRange() {
    dnnl::memory::dims idsDims = m_idsDesc.get_dims();
    size_t bDim2 = size_t(m_bDims[2]);   // rows
    size_t idsDim2 = size_t(idsDims[2]); // n_tokens
    size_t idsDim3 = size_t(idsDims[3]); // n_expert_used
    size_t lws0 = 16; // N_SIMDWIDTH
    size_t lws1 = 2;  // N_SG_Q8_0
    size_t ndst = 4;  // N_R0_Q8_0
    size_t gws0 = (bDim2 + ndst * lws1 - 1) / (ndst * lws1) * lws0;
    size_t gws1 = lws1;
    size_t gws2 = idsDim2 * idsDim3;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

//
//    MulMatIdNode_Mxfp4
//

class MulMatIdNode_Mxfp4: public MulMatIdNode {
public:
    MulMatIdNode_Mxfp4(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
    ~MulMatIdNode_Mxfp4();
protected:
    std::string GetKernelName() override;
    const char *GetKernelCode() override;
    void InitNdRange() override;
};

MulMatIdNode_Mxfp4::MulMatIdNode_Mxfp4(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids):
            MulMatIdNode(context, a, b, ids) { }

MulMatIdNode_Mxfp4::~MulMatIdNode_Mxfp4() { }

std::string MulMatIdNode_Mxfp4::GetKernelName() {
    return "mul_mat_id_simple_mxfp4";
}

const char *MulMatIdNode_Mxfp4::GetKernelCode() {
    return kernels::MulMatIdQuantSimple_Mxfp4_KernelCode();
}

void MulMatIdNode_Mxfp4::InitNdRange() {
    dnnl::memory::dims idsDims = m_idsDesc.get_dims();
    size_t bDim2 = size_t(m_bDims[2]);   // rows
    size_t idsDim2 = size_t(idsDims[2]); // n_tokens
    size_t idsDim3 = size_t(idsDims[3]); // n_expert_used
    size_t lws0 = 16; // N_SIMDWIDTH
    size_t lws1 = 2;  // N_SG_MXFP4
    size_t ndst = 2;  // N_R0_MXFP4
    size_t gws0 = (bDim2 + ndst * lws1 - 1) / (ndst * lws1) * lws0;
    size_t gws1 = lws1;
    size_t gws2 = idsDim2 * idsDim3;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

} // namespace

//
//    MulMatIdQuantSimple
//

MulMatIdQuantSimple::MulMatIdQuantSimple(Context *context):
        m_context(context) { }

MulMatIdQuantSimple::~MulMatIdQuantSimple() { }

std::unique_ptr<core::Node> MulMatIdQuantSimple::CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *ids) {
    base::NodeBase *bBase = m_context->CastNode(b);
    switch (bBase->Quant()) {
    case base::QuantMode::Q4_0:
        {
            std::unique_ptr<MulMatIdNode_Q4_0> node =
                std::make_unique<MulMatIdNode_Q4_0>(m_context, a, b, ids);
            if (node->Init()) {
                return node;
            }
        }
        break;
    case base::QuantMode::Q4_K:
        {
            std::unique_ptr<MulMatIdNode_Q4_K> node =
                std::make_unique<MulMatIdNode_Q4_K>(m_context, a, b, ids);
            if (node->Init()) {
                return node;
            }
        }
        break;
    case base::QuantMode::Q5_0:
        {
            std::unique_ptr<MulMatIdNode_Q5_0> node =
                std::make_unique<MulMatIdNode_Q5_0>(m_context, a, b, ids);
            if (node->Init()) {
                return node;
            }
        }
        break;
    case base::QuantMode::Q6_K:
        {
            std::unique_ptr<MulMatIdNode_Q6_K> node =
                std::make_unique<MulMatIdNode_Q6_K>(m_context, a, b, ids);
            if (node->Init()) {
                return node;
            }
        }
        break;
    case base::QuantMode::Q8_0:
        {
            std::unique_ptr<MulMatIdNode_Q8_0> node =
                std::make_unique<MulMatIdNode_Q8_0>(m_context, a, b, ids);
            if (node->Init()) {
                return node;
            }
        }
        break;
    case base::QuantMode::MXFP4:
        {
            std::unique_ptr<MulMatIdNode_Mxfp4> node =
                std::make_unique<MulMatIdNode_Mxfp4>(m_context, a, b, ids);
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


