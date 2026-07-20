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
#include <cmath>
#include <cassert>
#include <string>
#include <memory>
#include <ostream>
#include <sstream>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/common_xe.hpp"
#include "arhat/onednn/ocl/shape_info_args.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/gdn.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    GdnNode
//

class GdnNode: public NodeBase {
public:
    GdnNode(
        Context *context,
        core::Node *q,
        core::Node *k,
        core::Node *v,
        core::Node *g,
        core::Node *beta,
        core::Node *state);
    ~GdnNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    void InferShapes();
    void InitConfig();
    void InitArgs();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    core::Node *m_q;
    core::Node *m_k;
    core::Node *m_v;
    core::Node *m_g;
    core::Node *m_beta;
    core::Node *m_state;
    dnnl::memory::desc m_qDesc;
    dnnl::memory::desc m_kDesc;
    dnnl::memory::desc m_vDesc;
    dnnl::memory::desc m_gDesc;
    dnnl::memory::desc m_betaDesc;
    dnnl::memory::desc m_stateDesc;
    dnnl::memory::desc m_dstDesc;
    dnnl::memory m_qMem;
    dnnl::memory m_kMem;
    dnnl::memory m_vMem;
    dnnl::memory m_gMem;
    dnnl::memory m_betaMem;
    dnnl::memory m_stateMem;
    int64_t m_nSeqs;
    int64_t m_nTokens;
    int64_t m_H;
    int64_t m_Sv;
    int64_t m_sgSize;
    int64_t m_numSgs;
    int64_t m_rowsPerLane;
    int64_t m_kda;
    float m_scale;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

GdnNode::GdnNode(
        Context *context,
        core::Node *q,
        core::Node *k,
        core::Node *v,
        core::Node *g,
        core::Node *beta,
        core::Node *state):
            NodeBase(context),
            m_q(q),
            m_k(k),
            m_v(v),
            m_g(g),
            m_beta(beta),
            m_state(state),
            m_nSeqs(0),
            m_nTokens(0),
            m_H(0),
            m_Sv(0),
            m_sgSize(0),
            m_numSgs(0),
            m_rowsPerLane(0),
            m_kda(0),
            m_scale(0.0f) { }

GdnNode::~GdnNode() { }

bool GdnNode::Init() {
    base::NodeBase *q = m_gpuContext->CastNode(m_q);
    base::NodeBase *k = m_gpuContext->CastNode(m_k);
    base::NodeBase *v = m_gpuContext->CastNode(m_v);
    base::NodeBase *g = m_gpuContext->CastNode(m_g);
    base::NodeBase *beta = m_gpuContext->CastNode(m_beta);
    base::NodeBase *state = m_gpuContext->CastNode(m_state);
    m_qDesc = q->MemoryDesc();
    m_kDesc = k->MemoryDesc();
    m_vDesc = v->MemoryDesc();
    m_gDesc = g->MemoryDesc();
    m_betaDesc = beta->MemoryDesc();
    m_stateDesc = state->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_qMem = q->Memory();
    m_kMem = k->Memory();
    m_vMem = v->Memory();
    m_gMem = g->Memory();
    m_betaMem = beta->Memory();
    m_stateMem = state->Memory();
    InferShapes();
    SetMemory(m_dstDesc);
    InitConfig();
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void GdnNode::Compute() {
    m_kernel->SetArgBuffer(0, m_qMem);
    m_kernel->SetArgBuffer(1, m_kMem);
    m_kernel->SetArgBuffer(2, m_vMem);
    m_kernel->SetArgBuffer(3, m_gMem);
    m_kernel->SetArgBuffer(4, m_betaMem);
    m_kernel->SetArgBuffer(5, m_stateMem);
    m_kernel->SetArgBuffer(6, m_memory);
    m_kernel->SetArgF32(7, m_scale);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 8);    
    m_kernel->Launch(m_ndRange);
}

bool GdnNode::Validate() {
    auto validateQKV = [](const dnnl::memory::desc &desc) {
        return (desc.get_data_type() == dnnl::memory::data_type::f32 &&
            MemoryDescUtil::IsPlain(desc) &&
            MemoryDescUtil::HasDenseRows(desc));
    };
    auto validateGBS = [](const dnnl::memory::desc &desc) {
        return (desc.get_data_type() == dnnl::memory::data_type::f32 &&
            base::IsRowMajor(desc));
    };
    if (!validateQKV(m_qDesc)) {
        return false;
    }
    if (!validateQKV(m_kDesc)) {
        return false;
    }
    if (!validateQKV(m_vDesc)) {
        return false;
    }
    if (!validateGBS(m_gDesc)) {
        return false;
    }
    if (!validateGBS(m_betaDesc)) {
        return false;
    }
    if (!validateGBS(m_stateDesc)) {
        return false;
    }
    dnnl::memory::dims vDims = m_vDesc.get_dims();
    if (vDims[3] % 16 != 0) {
        return false;
    }
    return true;
}

void GdnNode::InferShapes() {
    dnnl::memory::dims vDims = m_vDesc.get_dims();
    dnnl::memory::dim nSeqs = vDims[0];
    dnnl::memory::dim nTokens = vDims[1];
    dnnl::memory::dim H = vDims[2];
    dnnl::memory::dim Sv = vDims[3];
    dnnl::memory::dims dstDims{1, 1, nTokens * nSeqs + Sv * nSeqs, Sv * H};
    m_dstDesc = 
        dnnl::memory::desc(
            dstDims, 
            dnnl::memory::data_type::f32, 
            dnnl::memory::format_tag::abcd);
}

void GdnNode::InitConfig() {
    dnnl::memory::dims vDims = m_vDesc.get_dims();
    dnnl::memory::dims gDims = m_gDesc.get_dims();
    m_nSeqs = int64_t(vDims[0]);
    m_nTokens = int64_t(vDims[1]);
    m_H = int64_t(vDims[2]);
    m_Sv = int64_t(vDims[3]);
    m_sgSize = 16;
    m_numSgs = 16; // assume max workgroup size >= 256
    m_rowsPerLane = m_Sv / m_sgSize;
    m_kda = (gDims[3] == vDims[3]) ? 1 : 0;
    m_scale = 1.0f / sqrt(float(m_Sv));
}

void GdnNode::InitArgs() {
    dnnl::memory::dim qBase = m_qDesc.get_submemory_offset();
    dnnl::memory::dim kBase = m_kDesc.get_submemory_offset();
    dnnl::memory::dim vBase = m_vDesc.get_submemory_offset();
    dnnl::memory::dim gBase = m_gDesc.get_submemory_offset();
    dnnl::memory::dim betaBase = m_betaDesc.get_submemory_offset();
    dnnl::memory::dim stateBase = m_stateDesc.get_submemory_offset();
    dnnl::memory::dim dstBase = m_dstDesc.get_submemory_offset();
    dnnl::memory::dims qDims = m_qDesc.get_dims();
    dnnl::memory::dims vDims = m_vDesc.get_dims();
    dnnl::memory::dims qStrides = m_qDesc.get_strides();
    dnnl::memory::dims vStrides = m_vDesc.get_strides();
    dnnl::memory::dims betaStrides = m_betaDesc.get_strides();

    m_shapeInfoArgs.AddS64("Q_BASE", qBase);
    m_shapeInfoArgs.AddS64("K_BASE", kBase);
    m_shapeInfoArgs.AddS64("V_BASE", vBase);
    m_shapeInfoArgs.AddS64("G_BASE", gBase);
    m_shapeInfoArgs.AddS64("BETA_BASE", betaBase);
    m_shapeInfoArgs.AddS64("STATE_BASE", stateBase);
    m_shapeInfoArgs.AddS64("DST_BASE", dstBase);
    m_shapeInfoArgs.AddS32("Q_D0", qDims[0]);
    m_shapeInfoArgs.AddS32("Q_D2", qDims[2]);
    m_shapeInfoArgs.AddS32("Q_S0", qStrides[0]);
    m_shapeInfoArgs.AddS32("Q_S1", qStrides[1]);
    m_shapeInfoArgs.AddS32("Q_S2", qStrides[2]);
    m_shapeInfoArgs.AddS32("V_D0", vDims[0]);
    m_shapeInfoArgs.AddS32("V_S0", vStrides[0]);
    m_shapeInfoArgs.AddS32("V_S1", vStrides[1]);
    m_shapeInfoArgs.AddS32("V_S2", vStrides[2]);
    m_shapeInfoArgs.AddS32("BETA_S0", betaStrides[0]);
    m_shapeInfoArgs.AddS32("BETA_S1", betaStrides[1]);
    m_shapeInfoArgs.AddS32("BETA_S2", betaStrides[2]);
    m_shapeInfoArgs.AddS32("N_SEQS", m_nSeqs);
    m_shapeInfoArgs.AddS32("N_TOKENS", m_nTokens);
    m_shapeInfoArgs.AddS32("H", m_H);
}

void GdnNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::GdnSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "gdn_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string GdnNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*gated_delta_net_simple");
    sb.Int(m_Sv);
    sb.Int(m_kda);
    return sb.Get();
}

std::string GdnNode::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    EmitInt(ss, "S_V", m_Sv);
    EmitInt(ss, "SG_SIZE", m_sgSize);
    EmitInt(ss, "ROWS_PER_LANE", m_rowsPerLane);
    EmitInt(ss, "KDA", m_kda);
    ss << "\n";
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    return ss.str();
}

void GdnNode::InitNdRange() {
    size_t lws0 = size_t(m_sgSize);
    size_t lws1 = size_t(m_numSgs);
    size_t gws0 = size_t(m_H) * lws0;
    size_t gws1 = size_t(m_nSeqs) * lws1;
    size_t gws2 = size_t(m_Sv / m_numSgs);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

} // namespace

//
//    GatedDeltaNetSimple
//

GatedDeltaNetSimple::GatedDeltaNetSimple(Context *context):
        m_context(context) { }

GatedDeltaNetSimple::~GatedDeltaNetSimple() { }

std::unique_ptr<core::Node> GatedDeltaNetSimple::CreateNode(
        core::Node *q,
        core::Node *k,
        core::Node *v,
        core::Node *g,
        core::Node *beta,
        core::Node *state) {
    std::unique_ptr<GdnNode> node = 
        std::make_unique<GdnNode>(m_context, q, k, v, g, beta, state);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

