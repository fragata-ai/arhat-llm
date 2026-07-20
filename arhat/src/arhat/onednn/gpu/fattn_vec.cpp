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
#include <algorithm>
#include <ostream>
#include <sstream>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/common_xe.hpp"
#include "arhat/onednn/ocl/shape_info_args.hpp"
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

int64_t RndUp(int64_t a, int64_t b) {
    return ((a + b - 1) / b) * b;
}

int64_t DivUp(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int64_t Log2(int64_t x) {
    if (x > 0) {
        return int64_t(1) << int(std::floorf(std::log2f(float(x))));
    } else {
        return 0;
    }
}

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
        const dnnl::memory::desc &oDesc,
        float logitSoftcap,
        size_t maxWgSize);
    bool Validate();
    void Emit(std::ostream &os);
    void GetNdRange(
        size_t &gws0,
        size_t &gws1,
        size_t &gws2,
        size_t &lws0,
        size_t &lws1,
        size_t &lws2);
    int64_t D() const {
        return m_d;
    }
    int64_t Ncols() const {
        return m_ncols;
    }
    int64_t GqaRatio() const {
        return m_gqaRatio;
    }
    int64_t NHeadLog2() const {
        return m_nHeadLog2;
    }
private:
    void InitNdRange();
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
    float m_logitSoftcap;
    int64_t m_maxWgSize; // RESERVED
    int64_t m_d;
    int64_t m_ncols;
    int64_t m_nthreads;
    int64_t m_sgSize;
    int64_t m_numSgs;
    int64_t m_cpyNb;
    int64_t m_cpyNe;
    int64_t m_nthreadsKq;
    int64_t m_nthreadsV;
    int64_t m_vRowsPerThread;
    int64_t m_vColsPerIter;
    int64_t m_gqaRatio;
    int64_t m_neKq;
    int64_t m_neCombine;
    int64_t m_nHeadLog2;
    size_t m_gws0;
    size_t m_gws1;
    size_t m_gws2;
    size_t m_lws0;
    size_t m_lws1;
    size_t m_lws2;
};

FattnConfig::FattnConfig():
        m_logitSoftcap(0.0f),
        m_maxWgSize(0),
        m_d(0),
        m_ncols(0),
        m_nthreads(0),
        m_sgSize(0),
        m_numSgs(0),
        m_cpyNb(0),
        m_cpyNe(0),
        m_nthreadsKq(0),
        m_nthreadsV(0),
        m_vRowsPerThread(0),
        m_vColsPerIter(0),
        m_gqaRatio(0),
        m_neKq(0),
        m_neCombine(0),
        m_nHeadLog2(0),
        m_gws0(0),
        m_gws1(0),
        m_gws2(0),
        m_lws0(0),
        m_lws1(0),
        m_lws2(0) { }

FattnConfig::~FattnConfig() { }

void FattnConfig::Init(
        const dnnl::memory::desc &qDesc,
        const dnnl::memory::desc &kDesc,
        const dnnl::memory::desc &vDesc,
        const dnnl::memory::desc &maskDesc,
        const dnnl::memory::desc &sinksDesc,
        const dnnl::memory::desc &oDesc,
        float logitSoftcap,
        size_t maxWgSize) {
    m_qDesc = qDesc;
    m_kDesc = kDesc;
    m_vDesc = vDesc;
    m_maskDesc = maskDesc;
    m_sinksDesc = sinksDesc;
    m_oDesc = oDesc;
    m_logitSoftcap = logitSoftcap;
    m_maxWgSize = int64_t(maxWgSize);

    dnnl::memory::dims qDims = m_qDesc.get_dims();
    dnnl::memory::dims kDims = m_kDesc.get_dims();
    dnnl::memory::dims vDims = m_vDesc.get_dims();

    m_d = int64_t(qDims[3]);
    m_ncols = 1;
    m_nthreads = 128;
    m_sgSize = 32;
    m_numSgs = m_nthreads / m_sgSize;
    m_cpyNb = 8;
    m_cpyNe = m_cpyNb / 4;
    m_nthreadsKq = 32 / m_cpyNe;
    m_nthreadsV = 32 / m_cpyNe;
    m_vRowsPerThread = 2 * m_cpyNe;
    m_vColsPerIter = m_sgSize / m_nthreadsV;
    m_gqaRatio = int64_t(qDims[1] / kDims[1]);
    m_neKq = m_ncols * m_d;
    m_neCombine = m_numSgs * m_vColsPerIter * m_d;
    int64_t nHead = int64_t(qDims[1]);
    m_nHeadLog2 = Log2(nHead);

    InitNdRange();
}

bool FattnConfig::Validate() {
    // RESERVED
    return true;
}

void FattnConfig::Emit(std::ostream &os) {
    EmitInt(os, "D", m_d);
    EmitInt(os, "NCOLS", m_ncols);
    EmitInt(os, "NTHREADS", m_nthreads);
    EmitInt(os, "SG_SIZE", m_sgSize);
    EmitInt(os, "NUM_SGS", m_numSgs);
    EmitInt(os, "CPY_NB", m_cpyNb);
    EmitInt(os, "CPY_NE", m_cpyNe);
    EmitInt(os, "NTHREADS_KQ", m_nthreadsKq);
    EmitInt(os, "NTHREADS_V", m_nthreadsV);
    EmitInt(os, "V_ROWS_PER_THREAD", m_vRowsPerThread);
    EmitInt(os, "V_COLS_PER_ITER", m_vColsPerIter);
    EmitInt(os, "NE_KQ", m_neKq);
    EmitInt(os, "NE_COMBINE", m_neCombine);
    EmitBool(os, "USE_LOGIT_SOFTCAP", (m_logitSoftcap != 0.0f)); 
    os << "\n";
}

void FattnConfig::GetNdRange(
        size_t &gws0,
        size_t &gws1,
        size_t &gws2,
        size_t &lws0,
        size_t &lws1,
        size_t &lws2) {
    gws0 = m_gws0;
    gws1 = m_gws1;
    gws2 = m_gws2;
    lws0 = m_lws0;
    lws1 = m_lws1;
    lws2 = m_lws2;
}

void FattnConfig::InitNdRange() {
    dnnl::memory::dims qDims = m_qDesc.get_dims();
    dnnl::memory::dims kDims = m_kDesc.get_dims();
    int64_t ntilesX = DivUp(int64_t(qDims[2]), m_ncols);
    int64_t ntilesZGqa = m_gqaRatio;
    int64_t parallelBlocks = 1; // RESERVED
    m_lws0 = size_t(m_sgSize);
    m_lws1 = size_t(m_numSgs);
    m_lws2 = 1;
    m_gws0 = size_t(ntilesX) * m_lws0;
    m_gws1 = size_t(parallelBlocks) * m_lws1;
    m_gws2 = size_t(ntilesZGqa) * size_t(kDims[1]) * size_t(qDims[0]) * m_lws2;
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
    bool ValidateCase();
    void InferShapes();
    bool InitConfig();
    void InitArgs();
    void InitShapeInfo();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
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
    dnnl::memory m_kvMaxMem; // RESERVED
    dnnl::memory m_oMetaMem; // RESERVED
    FattnConfig m_config;
    float m_argScale;
    float m_m0;
    float m_m1;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
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
            m_argScale(0.0f),
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
    if (!InitConfig()) {
        return false;
    }
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void FattnNode::Compute() {
    m_kernel->SetArgBuffer(0, m_qMem);
    m_kernel->SetArgBuffer(1, m_kMem);
    m_kernel->SetArgBuffer(2, m_vMem);
    m_kernel->SetArgBuffer(3, m_maskMem);
    m_kernel->SetArgBuffer(4, m_sinksMem);
    m_kernel->SetArgBuffer(5, m_kvMaxMem);
    m_kernel->SetArgBuffer(6, m_memory);
    m_kernel->SetArgBuffer(7, m_oMetaMem);
    m_kernel->SetArgF32(8, m_argScale);
    m_kernel->SetArgF32(9, m_maxBias);
    m_kernel->SetArgF32(10, m_m0);
    m_kernel->SetArgF32(11, m_m1);
    m_kernel->SetArgF32(12, m_logitSoftcap);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 13);    
    m_kernel->Launch(m_ndRange);
}

bool FattnNode::Validate() {
    auto validateType = [](const dnnl::memory::desc &md, dnnl::memory::data_type dt) -> bool {
        return (md.get_data_type() == dt);
    };
    auto validateFormat = [](const dnnl::memory::desc &md) -> bool {
        return (MemoryDescUtil::IsPlain(md) && MemoryDescUtil::HasDenseRows(md));
    };
    if (!validateType(m_qDesc, dnnl::memory::data_type::f32)) {
        return false;
    }
    if (!validateType(m_kDesc, dnnl::memory::data_type::f16)) {
        return false;
    }
    if (!validateType(m_vDesc, dnnl::memory::data_type::f16)) {
        return false;
    }
    if (m_mask != nullptr && !validateType(m_maskDesc, dnnl::memory::data_type::f16)) {
        return false;
    }
    if (m_sinks != nullptr && !validateType(m_sinksDesc, dnnl::memory::data_type::f32)) {
        return false;
    }
    if (!validateFormat(m_qDesc)) {
        return false;
    }
    if (!validateFormat(m_kDesc)) {
        return false;
    }
    if (!validateFormat(m_vDesc)) {
        return false;
    }
    if (m_mask != nullptr && !validateFormat(m_maskDesc)) {
        return false;
    }
    if (m_sinks != nullptr && !validateFormat(m_sinksDesc)) {
        return false;
    }
    if (!ValidateCase()) {
        return false;
    }
    return true;
}

bool FattnNode::ValidateCase() {
    dnnl::memory::dims qDims = m_qDesc.get_dims();
    dnnl::memory::dims kDims = m_kDesc.get_dims();
    dnnl::memory::dims vDims = m_vDesc.get_dims();
    int64_t d = int64_t(qDims[3]);
    if (d != 64 && d != 128 && d != 256) {
        return false;
    }
    if (kDims[3] != vDims[3]) {
        return false;
    }
    constexpr dnnl::memory::dim FATTN_KQ_STRIDE = 256;
    if (kDims[2] % FATTN_KQ_STRIDE != 0) {
        return false;
    }
    if (qDims[2] != 1) {
        return false;
    }
    // SKIPPED: gqa_opt_applies check - not yet implemented
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

bool FattnNode::InitConfig() {
    ocl::OclDeviceInfo info = GetOclContext()->GetDeviceInfo();
    m_config.Init(
        m_qDesc,
        m_kDesc,
        m_vDesc,
        m_maskDesc,
        m_sinksDesc,
        m_oDesc,
        m_logitSoftcap,
        info.maxWorkGroupSize);
    if (!m_config.Validate()) {
        return false;
    }
    return true;
}

void FattnNode::InitArgs() {
    m_argScale = m_scale;
    if (m_logitSoftcap != 0.0f) {
        m_argScale /= m_logitSoftcap;
    }
    dnnl::memory::dims qDims = m_qDesc.get_dims();
    int64_t nHead = int64_t(qDims[1]);
    int64_t nHeadLog2 = Log2(nHead);
    float nHeadLog2f = (nHeadLog2 > 0) ? float(nHeadLog2) : 1.0f;
    m_m0 = std::powf(2.0f, -m_maxBias / nHeadLog2f);
    m_m1 = std::powf(2.0f, -(m_maxBias / 2.0f) / nHeadLog2f);
    InitShapeInfo();
}

void FattnNode::InitShapeInfo() {
    m_shapeInfoArgs.AddS32("GQA_RATIO", m_config.GqaRatio());
    m_shapeInfoArgs.AddU32("N_HEAD_LOG2", m_config.NHeadLog2());
    m_shapeInfoArgs.AddMemoryDesc("Q", m_qDesc);
    m_shapeInfoArgs.AddMemoryDesc("K", m_kDesc);
    m_shapeInfoArgs.AddMemoryDesc("V", m_vDesc);
    m_shapeInfoArgs.AddMemoryDesc("MASK", m_maskDesc);
}

void FattnNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    kernelContext.SetOption("-cl-intel-256-GRF-per-thread");
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::FattnVecKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "fattn_vec", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string FattnNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*fattn_vec");
    sb.Int(m_config.D());
    sb.Int(m_config.Ncols());
    sb.Bool(m_logitSoftcap != 0);
    return sb.Get();
}

std::string FattnNode::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitCopy(ss);
    m_config.Emit(ss);
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    return ss.str();
}

void FattnNode::InitNdRange() {
    size_t gws0, gws1, gws2;
    size_t lws0, lws1, lws2;
    m_config.GetNdRange(gws0, gws1, gws2, lws0, lws1, lws2);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, lws2);
}

} // namespace

//
//    FattnVec
//

FattnVec::FattnVec(Context *context):
        m_context(context) { }

FattnVec::~FattnVec() { }

std::unique_ptr<core::Node> FattnVec::CreateNode(
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

