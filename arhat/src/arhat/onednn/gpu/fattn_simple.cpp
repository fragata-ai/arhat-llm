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

#include "arhat/onednn/base/runtime.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/fattn.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    Local utilities
//

int64_t Log2(int64_t x) {
    if (x > 0) {
        return int64_t(1) << int(std::floorf(std::log2f(float(x))));
    } else {
        return 0;
    }
}

//
//    FattnConfig
//

class FattnConfig {
public:
    FattnConfig();
    ~FattnConfig();
public:
    void Init(
        const dnnl::memory::desc &qDesc,
        const dnnl::memory::desc &kDesc,
        const dnnl::memory::desc &vDesc,
        const dnnl::memory::desc &maskDesc,
        const dnnl::memory::desc &sinksDesc,
        const dnnl::memory::desc &oDesc);
    void Emit(std::ostream &os);
    int64_t NQ() {
        return m_nQ;
    }
    int64_t NHead() {
        return m_nHead;
    }
    int64_t NBatch() {
        return m_nBatch;
    }
    int64_t BlockM() {
        return m_blockM;
    }
    int64_t NHeadLog2() {
        return m_nHeadLog2;
    }
private:
    void InitBlockDims();
    static void EmitBase(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::desc &desc);
    static void EmitStrides(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::desc &desc);
    static void EmitType(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::desc &desc,
        bool vec4);
    static void EmitBool(
        std::ostream &os, 
        const std::string &name, 
        bool value);
    static void EmitInt(
        std::ostream &os, 
        const std::string &name, 
        int64_t value);
private:
    dnnl::memory::desc m_qDesc;
    dnnl::memory::desc m_kDesc;
    dnnl::memory::desc m_vDesc;
    dnnl::memory::desc m_maskDesc;
    dnnl::memory::desc m_sinksDesc;
    dnnl::memory::desc m_oDesc;
    int64_t m_nQ;
    int64_t m_nKV;
    int64_t m_dHeadQ;
    int64_t m_dHeadV;
    int64_t m_nHead;
    int64_t m_nHeadKV;
    int64_t m_nBatch;
    int64_t m_blockM;
    int64_t m_blockN;
    bool m_isCausal;
    int64_t m_nHeadLog2;
};

FattnConfig::FattnConfig():
        m_nQ(0),
        m_nKV(0),
        m_dHeadQ(0),
        m_dHeadV(0),
        m_nHead(0),
        m_nHeadKV(0),
        m_nBatch(0),
        m_blockM(0),
        m_blockN(0),
        m_isCausal(false),
        m_nHeadLog2(0) { }

FattnConfig::~FattnConfig() { }

void FattnConfig::Init(
        const dnnl::memory::desc &qDesc,
        const dnnl::memory::desc &kDesc,
        const dnnl::memory::desc &vDesc,
        const dnnl::memory::desc &maskDesc,
        const dnnl::memory::desc &sinksDesc,
        const dnnl::memory::desc &oDesc) {
    m_qDesc = qDesc;
    m_kDesc = kDesc;
    m_vDesc = vDesc;
    m_maskDesc = maskDesc;
    m_sinksDesc = sinksDesc;
    m_oDesc = oDesc;

    dnnl::memory::dims qDims = m_qDesc.get_dims();
    dnnl::memory::dims kDims = m_kDesc.get_dims();
    dnnl::memory::dims vDims = m_vDesc.get_dims();

    m_nQ = int64_t(qDims[2]);
    m_nKV = int64_t(kDims[2]);
    m_dHeadQ = int64_t(qDims[3]);
    m_dHeadV = int64_t(vDims[3]);
    m_nHead = int64_t(qDims[1]);
    m_nHeadKV = int64_t(kDims[1]);
    m_nBatch = int64_t(qDims[0]);

    InitBlockDims();

    m_isCausal = (m_maskDesc.is_zero() && m_nQ > 1 && m_nQ == m_nKV);

    m_nHeadLog2 = Log2(m_nHead);
}

void FattnConfig::Emit(std::ostream &os) {
    os << "#define unroll_for __attribute__((opencl_unroll_hint)) for\n";
    os << "\n";

    EmitInt(os, "N_Q", m_nQ);
    EmitInt(os, "N_KV", m_nKV);
    EmitInt(os, "DK", m_dHeadQ);
    EmitInt(os, "DV", m_dHeadV);
    EmitInt(os, "N_HEAD", m_nHead);
    EmitInt(os, "N_HEAD_KV", m_nHeadKV);
    EmitInt(os, "N_BATCH", m_nBatch);
    EmitInt(os, "BLOCK_M", m_blockM);
    EmitInt(os, "BLOCK_N", m_blockN);
    EmitBool(os, "IS_CAUSAL", m_isCausal);
    EmitInt(os, "N_HEAD_LOG2", m_nHeadLog2);
    os << "\n";

    EmitBase(os, "Q", m_qDesc);
    EmitBase(os, "K", m_kDesc);
    EmitBase(os, "V", m_vDesc);
    EmitBase(os, "O", m_oDesc);
    EmitBase(os, "MASK", m_maskDesc);
    EmitBase(os, "SINKS", m_maskDesc);
    os << "\n";

    EmitStrides(os, "Q", m_qDesc);
    EmitStrides(os, "K", m_kDesc);
    EmitStrides(os, "V", m_vDesc);
    EmitStrides(os, "O", m_oDesc);
    EmitStrides(os, "MASK", m_maskDesc);
    os << "\n";

    // use dummy dimensions for missing optional inputs
    int64_t maskD0 = 0;
    int64_t maskD1 = 0;
    if (!m_maskDesc.is_zero()) {
        dnnl::memory::dims maskDims = m_maskDesc.get_dims();
        maskD0 = int64_t(maskDims[0]);
        maskD1 = int64_t(maskDims[1]);
    }
    EmitInt(os, "MASK_D0", maskD0);
    EmitInt(os, "MASK_D1", maskD1);
    os << "\n";

    os << "#define ACC_TYPE float\n";
    os << "#define ACC_TYPE4 float4\n";
    os << "\n";

    EmitType(os, "Q", m_qDesc, true);
    EmitType(os, "K", m_kDesc, true);
    EmitType(os, "V", m_vDesc, true);
    EmitType(os, "O", m_oDesc, true);
    EmitType(os, "MASK", m_maskDesc, false);
    EmitType(os, "SINKS", m_sinksDesc, false);
    os << "\n";

    auto isFloat = [](const dnnl::memory::desc &desc) -> bool {
        return (desc.get_data_type() == dnnl::memory::data_type::f32);
    };

    std::string qConv = isFloat(m_qDesc) ? "" : "convert_float4";
    std::string kConv = isFloat(m_kDesc) ? "" : "convert_float4";
    std::string vConv = isFloat(m_vDesc) ? "" : "convert_float4";
    std::string oConv = isFloat(m_oDesc) ? "" : "convert_half4";

    os << "#define CONVERT_Q_ACC4(x) " << qConv << "(x)\n";
    os << "#define CONVERT_K_ACC4(x) " << kConv << "(x)\n";
    os << "#define CONVERT_V_ACC4(x) " << vConv << "(x)\n";
    os << "#define CONVERT_O_DATA4(x) " << oConv << "(x)\n";
    os << "\n";

    os << "#define DK_VEC (DK / 4)\n";
    os << "#define DV_VEC (DV / 4)\n";
    os << "#define WG_SIZE BLOCK_M\n";
    os << "#define Q1_WG_SIZE 64\n";
    os << "\n";
}

struct {
    int64_t dims[4];
} g_blockDimsMap[] = {
    {40,  40, 32, 32}, 
    {64,  64, 64, 64}, 
    {72,  72, 64, 32},
    {80,  80, 64, 32}, 
    {96,  96, 64, 32},
    {112, 112, 32, 32}, 
    {128, 128, 32, 32}, 
    {192, 128, 16, 16},
    {192, 192, 16, 16}, 
    {256, 256, 16, 16}, 
    {576, 512, 16, 8},
    {0, 0, 0, 0}
};

void FattnConfig::InitBlockDims() {
    for (int i = 0; ; i++) {
        const int64_t *dims = g_blockDimsMap[i].dims;
        if (dims[0] == 0) {
            break;
        }
        if (dims[0] == m_dHeadQ && dims[1] == m_dHeadV) {
            m_blockM = dims[2];
            m_blockN = dims[3];
            return;
        }
    }
    assert(false);
}

void FattnConfig::EmitBase(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::desc &desc) {
    // use dummy base for missing optional inputs
    int64_t base = 0;
    if (!desc.is_zero()) {
        base = int64_t(desc.get_submemory_offset());
    }
    EmitInt(os, prefix + "_BASE", base);
}

void FattnConfig::EmitStrides(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::desc &desc) {
    // use dummy strides for missing optional inputs
    int64_t s0 = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    if (!desc.is_zero()) {
        dnnl::memory::dims strides = desc.get_strides();
        s0 = int64_t(strides[0]);
        s1 = int64_t(strides[1]);
        s2 = int64_t(strides[2]);
    }
    EmitInt(os, prefix + "_S0", s0);
    EmitInt(os, prefix + "_S1", s1);
    EmitInt(os, prefix + "_S2", s2);
}

void FattnConfig::EmitType(
        std::ostream &os, 
        const std::string &prefix, 
        const dnnl::memory::desc &desc,
        bool vec4) {
    std::string value;
    // use float as dummy type for missing optional inputs
    dnnl::memory::data_type dt = 
        !desc.is_zero() ? 
            desc.get_data_type() : 
            dnnl::memory::data_type::f32;
    if (dt == dnnl::memory::data_type::f32) {
        value = "float";
    } else if (dt == dnnl::memory::data_type::f16) {
        value = "half";
    } else  {
        assert(false);
    }
    os << "#define " << prefix << "_TYPE " << value << "\n";
    if (vec4) {
        os << "#define " << prefix << "_TYPE4 " << value << "4\n";
    }
}

void FattnConfig::EmitBool(
        std::ostream &os, 
        const std::string &name, 
        bool value) {
    os << "#define " << name << " " << (value ? "1" : "0") << "\n";
}

void FattnConfig::EmitInt(
        std::ostream &os, 
        const std::string &name, 
        int64_t value) {
    os << "#define " << name << " " << ocl::FormatInt(value) << "\n";
}

//
//    FattnNode
//

class FattnNode: public NodeBase {
public:
    FattnNode(
        Context *context,
        core::Node *q,
        core::Node *k,
        core::Node *v,
        core::Node *mask,
        core::Node *sinks,
        float scale,
        float maxBias,
        float logitSoftcap,
        core::Prec prec);
    ~FattnNode();
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
    void InitNdRange();
private:
    core::Node *m_q;
    core::Node *m_k;
    core::Node *m_v;
    core::Node *m_mask;
    core::Node *m_sinks;
    float m_scale;
    float m_maxBias;
    float m_logitSoftcap;
    core::Prec m_prec;
    dnnl::memory::desc m_qDesc;
    dnnl::memory::desc m_kDesc;
    dnnl::memory::desc m_vDesc;
    dnnl::memory::desc m_maskDesc;
    dnnl::memory::desc m_sinksDesc;
    dnnl::memory::desc m_oDesc;
    dnnl::memory m_qMem;
    dnnl::memory m_kMem;
    dnnl::memory m_vMem;
    dnnl::memory m_maskMem;
    dnnl::memory m_sinksMem;
    FattnConfig m_config;
    float m_m0;
    float m_m1;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

FattnNode::FattnNode(
        Context *context,
        core::Node *q,
        core::Node *k,
        core::Node *v,
        core::Node *mask,
        core::Node *sinks,
        float scale,
        float maxBias,
        float logitSoftcap,
        core::Prec prec):
            NodeBase(context),
            m_q(q),
            m_k(k),
            m_v(v),
            m_mask(mask),
            m_sinks(sinks),
            m_scale(scale),
            m_maxBias(maxBias),
            m_logitSoftcap(logitSoftcap),
            m_prec(prec),
            m_m0(0.0f), 
            m_m1(0.0f) { }

FattnNode::~FattnNode() { }

bool FattnNode::Init() {
    base::NodeBase *q = m_gpuContext->CastNode(m_q);
    base::NodeBase *k = m_gpuContext->CastNode(m_k);
    base::NodeBase *v = m_gpuContext->CastNode(m_v);
    base::NodeBase *mask = m_gpuContext->CastNode(m_mask);
    base::NodeBase *sinks = m_gpuContext->CastNode(m_sinks);
    m_qDesc = q->MemoryDesc();
    m_kDesc = k->MemoryDesc();
    m_vDesc = v->MemoryDesc();
    if (mask != nullptr) {
        m_maskDesc = mask->MemoryDesc();
    }
    if (sinks != nullptr) {
        m_sinksDesc = sinks->MemoryDesc();
    }
    if (!Validate()) {
        return false;
    }
    m_qMem = q->Memory();
    m_kMem = k->Memory();
    m_vMem = v->Memory();
    if (mask != nullptr) {
        m_maskMem = mask->Memory();
    }
    if (sinks != nullptr) {
        m_sinksMem = sinks->Memory();
    }
    InferShapes();
    SetMemory(m_oDesc);
    InitConfig();
    InitArgs();
    InitKernel();
    return true;
}

void FattnNode::Compute() {
    m_kernel->SetArgBuffer(0, m_qMem);
    m_kernel->SetArgBuffer(1, m_kMem);
    m_kernel->SetArgBuffer(2, m_vMem);
    m_kernel->SetArgBuffer(3, m_maskMem);
    m_kernel->SetArgBuffer(4, m_sinksMem);
    m_kernel->SetArgBuffer(5, m_memory);
    m_kernel->SetArgF32(6, m_scale);
    m_kernel->SetArgF32(7, m_maxBias);
    m_kernel->SetArgF32(8, m_logitSoftcap);
    m_kernel->SetArgF32(9, m_m0);
    m_kernel->SetArgF32(10, m_m1);
    m_kernel->Launch(m_ndRange);
}

bool FattnNode::Validate() {
    auto validateType = [](const dnnl::memory::desc &desc) -> bool {
        dnnl::memory::data_type dt = desc.get_data_type();
        return (dt == dnnl::memory::data_type::f32 ||
            dt == dnnl::memory::data_type::f16);
    };
    auto isSimple = [](const dnnl::memory::desc &desc) -> bool {
        return (MemoryDescUtil::IsPlain(desc) && 
            MemoryDescUtil::HasDenseRows(desc));
    };
    if (!validateType(m_qDesc)) {
        return false;
    }
    if (!validateType(m_kDesc)) {
        return false;
    }
    if (!validateType(m_vDesc)) {
        return false;
    }
    if (!m_maskDesc.is_zero()) {
        if (!validateType(m_maskDesc)) {
            return false;
        }
    }
    if (!m_sinksDesc.is_zero()) {
        if (!validateType(m_sinksDesc)) {
            return false;
        }
    }
    if (!isSimple(m_qDesc)) {
        return false;
    }
    if (!isSimple(m_kDesc)) {
        return false;
    }
    if (!isSimple(m_vDesc)) {
        return false;
    }
    if (!m_maskDesc.is_zero()) {
        if (!isSimple(m_maskDesc)) {
            return false;
        }
    }
    if (!m_sinksDesc.is_zero()) {
        if (!isSimple(m_sinksDesc)) {
            return false;
        }
    }
    return true;
}

void FattnNode::InferShapes() {
    dnnl::memory::dims qDims = m_qDesc.get_dims();
    dnnl::memory::dims vDims = m_vDesc.get_dims();
    // permute(0, 2, 1, 3)
    dnnl::memory::dims outputDims{qDims[0], qDims[2], qDims[1], vDims[3]}; 
    m_oDesc = 
        dnnl::memory::desc(
            outputDims, 
            dnnl::memory::data_type::f32, 
            dnnl::memory::format_tag::abcd);
}

void FattnNode::InitConfig() {
    m_config.Init(
        m_qDesc,
        m_kDesc,
        m_vDesc,
        m_maskDesc,
        m_sinksDesc,
        m_oDesc);
}

void FattnNode::InitArgs() {
    int64_t nHeadLog2 = m_config.NHeadLog2();
    float nHeadLog2f = (nHeadLog2 > 0) ? float(nHeadLog2) : 1.0f;
    m_m0 = std::powf(2.0f, -m_maxBias / nHeadLog2f);
    m_m1 = std::powf(2.0f, -(m_maxBias / 2.0f) / nHeadLog2f);
}

void FattnNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::stringstream ss;
    m_config.Emit(ss);
    std::string prolog = ss.str();
    const char *kernelCode = 
        (m_config.NQ() == 1) ? 
            kernels::FattnSimpleQ1KernelCode() : 
            kernels::FattnSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "fattn_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string FattnNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("fattn_simple");
    sb.MemoryDesc(m_qDesc);
    sb.MemoryDesc(m_kDesc);
    sb.MemoryDesc(m_vDesc);
    sb.MemoryDesc(m_maskDesc);
    sb.MemoryDesc(m_sinksDesc);
    sb.Float(m_scale);
    sb.Float(m_maxBias);
    sb.Float(m_logitSoftcap);
    sb.Int(int64_t(m_prec));
    return sb.Get();
}

void FattnNode::InitNdRange() {
    int64_t nQ = m_config.NQ();
    int64_t nHead = m_config.NHead();
    int64_t nBatch = m_config.NBatch();
    if (nQ == 1) {
        size_t wgSize = 64;
        m_ndRange = ocl::NdRange(wgSize, size_t(nBatch * nHead), 1, wgSize, 1, 1);
    } else {
        int64_t blockM = m_config.BlockM();
        size_t wgSize = size_t(blockM);
        m_ndRange =
            ocl::NdRange(
                size_t(((nQ + blockM - 1) / blockM) * blockM),
                size_t(nBatch * nHead),
                1,
                wgSize, 
                1,
                1);
    }
}

} // namespace

//
//    FattnSimple
//

FattnSimple::FattnSimple(Context *context):
        m_context(context) { }

FattnSimple::~FattnSimple() { }

std::unique_ptr<core::Node> FattnSimple::CreateNode(
        core::Node *q,
        core::Node *k,
        core::Node *v,
        core::Node *mask,
        core::Node *sinks,
        float scale,
        float maxBias,
        float logitSoftcap,
        core::Prec prec) {
    std::unique_ptr<FattnNode> node =
        std::make_unique<FattnNode>(
            m_context,
            q,
            k,
            v,
            mask,
            sinks,
            scale,
            maxBias,
            logitSoftcap,
            prec);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

