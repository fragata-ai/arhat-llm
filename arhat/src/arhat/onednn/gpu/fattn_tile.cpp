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

//
//    Tile configuration
//

// [dqk, dv, ncols, nbatchFa, nbatchK]

// TODO: Adjust this table for reasonable register pressure
const int64_t g_tileConfig[][5] = {
    {40,  40,  2,  64,  40},
    {40,  40,  4,  64,  40},
    {40,  40,  8,  64,  40},
    {40,  40, 16,  64,  40},
    {40,  40, 32,  64,  40},

    {64,  64,  2,  64,  64},
    {64,  64,  4,  64,  64},
    {64,  64,  8,  64,  64},
    {64,  64, 16,  64,  64},
    {64,  64, 32,  64,  64},

    {72,  72,  2,  64,  72},
    {72,  72,  4,  64,  72},
    {72,  72,  8,  64,  72},
    {72,  72, 16,  64,  72},
    {72,  72, 32,  64,  72},

    {80,  80,  2,  64,  40},
    {80,  80,  4,  64,  40},
    {80,  80,  8,  64,  40},
    {80,  80, 16,  64,  40},
    {80,  80, 32,  64,  40},

    {96,  96,  2,  64,  48},
    {96,  96,  4,  64,  48},
    {96,  96,  8,  64,  48},
    {96,  96, 16,  64,  48},
    {96,  96, 32,  64,  48},

    {112, 112,  2,  64,  56},
    {112, 112,  4,  64,  56},
    {112, 112,  8,  64,  56},
    {112, 112, 16,  64,  56},
    {112, 112, 32,  64,  56},

    {128, 128,  2,  64,  64},
    {128, 128,  4,  64,  64},
    {128, 128,  8,  64,  64},
    {128, 128, 16,  64,  64},
    {128, 128, 32,  64,  64},

    {256, 256,  2,  64,  64},
    {256, 256,  4,  64,  64},
    {256, 256,  8,  64,  64},
    {256, 256, 16,  64,  64},
    {256, 256, 32,  64,  64},

    {320, 256,  2,  64,  64},
    {320, 256,  4,  64,  64},
    {320, 256,  8,  64,  64},
    {320, 256, 16,  64,  64},
    {320, 256, 32,  64,  64},

    {512, 512,  2,  64,  64},
    {512, 512,  4,  64,  64},
    {512, 512,  8,  64,  64},
    {512, 512, 16,  64,  64},
    {512, 512, 32,  64,  64},

    {576, 512,  2,  64,  64},
    {576, 512,  4,  64,  64},
    {576, 512,  8,  64,  64},
    {576, 512, 16,  64,  64}, 
    {576, 512, 32,  64,  64}, 

    {0, 0, 0, 0, 0}
};

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
    int64_t Dkq() const {
        return m_dkq;
    }
    int64_t Dv() const {
        return m_dv;
    }
    int64_t Ncols1() const {
        return m_ncols1;
    }
    int64_t Ncols2() const {
        return m_ncols2;
    }
    int64_t GqaRatio() const {
        return m_gqaRatio;
    }
    int64_t NHeadLog2() const {
        return m_nHeadLog2;
    }
private:
    void GetNcols2();
    void GetNcols1();
    void GetTileConfig();
    void GetSgConfig();
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
    int64_t m_maxWgSize;
    int64_t m_dkq;
    int64_t m_dv;
    int64_t m_gqaRatio;
    int64_t m_ncols1;
    int64_t m_ncols2;
    int64_t m_ncols;
    int64_t m_nbatchFa;
    int64_t m_nbatchK;
    int64_t m_nbatchV;
    int64_t m_sgSize;
    int64_t m_numSgs;
    int64_t m_cpyNb;
    int64_t m_cpyNe;
    int64_t m_cpsg;
    int64_t m_np;
    int64_t m_dkqP;
    int64_t m_dvP;
    int64_t m_cpyNeDkq;
    int64_t m_cpyNeDv;
    int64_t m_cpyNeDv2;
    int64_t m_kqCs;
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
        m_dkq(0),
        m_dv(0),
        m_gqaRatio(0),
        m_ncols1(0),
        m_ncols2(0),
        m_ncols(0),
        m_nbatchFa(0),
        m_nbatchK(0),
        m_nbatchV(0),
        m_sgSize(0),
        m_numSgs(0),
        m_cpyNb(0),
        m_cpyNe(0),
        m_cpsg(0),
        m_np(0),
        m_dkqP(0),
        m_dvP(0),
        m_cpyNeDkq(0),
        m_cpyNeDv(0),
        m_cpyNeDv2(0),
        m_kqCs(0),
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

    m_dkq = int64_t(kDims[3]);
    m_dv = int64_t(vDims[3]);
    m_gqaRatio = int64_t(qDims[1] / kDims[1]);

    GetNcols2();
    GetNcols1();
    m_ncols = m_ncols1 * m_ncols2;

    GetTileConfig();
    GetSgConfig();

    // Intel-specific (NB = bytes, NE = dwords)
    m_cpyNb = 8;
    m_cpyNe = m_cpyNb / 4;

    // columns per subgroup
    m_cpsg = (m_ncols > m_numSgs) ? m_ncols / m_numSgs : 1;
    // subgroups per column
    m_np = (m_numSgs > m_ncols) ? m_numSgs / m_ncols : 1;

    // DKQ and DV padded to multiple of 2 * SG_SIZE
    m_dkqP = RndUp(m_dkq, 2 * m_sgSize);
    m_dvP = RndUp(m_dv, 2 * m_sgSize);

    m_cpyNeDkq = (m_cpyNe < m_dkqP / m_sgSize) ? m_cpyNe : m_dkqP / m_sgSize;
    m_cpyNeDv = (m_cpyNe < (m_dvP / 2) / m_sgSize) ? m_cpyNe : (m_dvP / 2) / m_sgSize;
    m_cpyNeDv2 = (m_cpyNe / 2 < (m_dvP / 2) / m_sgSize) ? m_cpyNe / 2 : (m_dvP / 2) / m_sgSize;

    // KQ_CS == KQ chunk size, number of KQ values in j direction to store
    // as one contiguous chunk in memory
    // KQ is originally 2D but uses a Z-shaped 3D memory pattern
    // like KQ[NCOLS / KQ_CS][DV_P][KQ_CS]
    m_kqCs = (m_cpsg < 2 * m_cpyNe) ? m_cpsg : 2 * m_cpyNe;

    // Number of V columns that fit in local memory for K
    // (TODO: Revise this heuristics for Intel)
    m_nbatchV = (((m_dv % m_nbatchK == 0) ? m_nbatchK : (m_nbatchK * 2) / 3) * m_nbatchFa) / m_dv;

    int64_t nHead = int64_t(qDims[1]);
    m_nHeadLog2 = Log2(nHead);

    InitNdRange();
}

bool FattnConfig::Validate() {
    if (m_np != 1 && m_cpsg != 1) {
        return false;
    }
    if (m_nbatchFa % (m_np * m_sgSize) != 0) {
        return false;
    }
    if (m_np > 1 && m_nbatchFa * m_nbatchK < m_numSgs * m_dvP) {
       return false;
    }

    if (m_cpsg % m_kqCs != 0) {
        return false;
    }
    if (m_dv > m_dkq) {
        return false;
    }
    if (m_dv % m_nbatchK != 0 &&
            (m_nbatchK % 3 != 0 || m_dv % ((m_nbatchK * 2) / 3) != 0)) {
        return false;
    }
    if (m_nbatchFa % m_nbatchV != 0) {
        return false;
    }
    if (m_nbatchV % m_np != 0) {
        return false;
    }

    if ((m_nbatchK / 2) % m_cpyNe != 0) {
        return false;
    }

    if (m_cpyNeDkq % 2 != 0) {
        return false;
    }

    return true;
}

void FattnConfig::Emit(std::ostream &os) {
    EmitInt(os, "DKQ", m_dkq);
    EmitInt(os, "DV", m_dv);
    EmitInt(os, "NCOLS1", m_ncols1);
    EmitInt(os, "NCOLS2", m_ncols2);
    EmitInt(os, "NCOLS", m_ncols);
    EmitInt(os, "NBATCH_FA", m_nbatchFa);
    EmitInt(os, "NBATCH_K", m_nbatchK);
    EmitInt(os, "NBATCH_V", m_nbatchV);
    EmitInt(os, "SG_SIZE", m_sgSize);
    EmitInt(os, "NUM_SGS", m_numSgs);
    EmitInt(os, "CPY_NB", m_cpyNb);
    EmitInt(os, "CPY_NE", m_cpyNe);
    EmitInt(os, "CPSG", m_cpsg);
    EmitInt(os, "NP", m_np);
    EmitInt(os, "DKQ_P", m_dkqP);
    EmitInt(os, "DV_P", m_dvP);
    EmitInt(os, "CPY_NE_DKQ", m_cpyNeDkq);
    // Introduced solely to support COPY macro
    EmitInt(os, "CPY_NE_DKQ2", m_cpyNeDkq / 2);
    EmitInt(os, "CPY_NE_DV", m_cpyNeDv);
    EmitInt(os, "CPY_NE_DV2", m_cpyNeDv2);
    EmitInt(os, "KQ_CS", m_kqCs);
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

void FattnConfig::GetNcols1() {
    dnnl::memory::dims qDims = m_qDesc.get_dims();
    int64_t nQ = int64_t(qDims[2]);
    if (nQ > 16 / m_ncols2) {
        m_ncols1 = 32 / m_ncols2;
        return;
    } 
    if (nQ > 8 / m_ncols2) {
        m_ncols1 = 16 / m_ncols2;
        return;
    }
    if (m_ncols2 <= 8 && nQ > 4 / m_ncols2) {
        m_ncols1 = 8 / m_ncols2;
        return;
    }
    if (m_ncols2 <= 4 && nQ > 2 / m_ncols2) {
        m_ncols1 = 4 / m_ncols2;
        return;
    }
    assert(m_ncols2 <= 2);
    m_ncols1 = 2 / m_ncols2;
}

void FattnConfig::GetNcols2() {
    // RESERVED
    m_ncols2 = 1;
}

void FattnConfig::GetTileConfig() {
    m_nbatchFa = 0;
    m_nbatchK = 0;
    for (int i = 0; g_tileConfig[i][0] != 0; i++) {
        const int64_t *c = g_tileConfig[i];
        if (c[0] == m_dkq && c[1] == m_dv && c[2] == m_ncols) {
            m_nbatchFa = c[3];
            m_nbatchK = c[4];
            break;
        }
    }
}

void FattnConfig::GetSgConfig() {
    // this is simplified calculation which
    // might be improved later for better hardware utilization
    m_sgSize = 32;
    int64_t nthreads = std::min(m_ncols * m_sgSize, m_maxWgSize);
    m_numSgs = nthreads / m_sgSize;
}

void FattnConfig::InitNdRange() {
    dnnl::memory::dims qDims = m_qDesc.get_dims();
    dnnl::memory::dims kDims = m_kDesc.get_dims();
    int64_t ntilesX = DivUp(int64_t(qDims[2]), m_ncols1);
    int64_t ntilesZGqa = DivUp(m_gqaRatio, m_ncols2);
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

const int64_t g_fattnCase[][2] = {
    {40, 40},
    {64, 64},
    {72, 72},
    {80, 80},
    {96, 96},
    {112, 112},
    {128, 128},
    {256, 256},
    {320, 256},
    {512, 512}, 
    {576, 512}, 
    {0, 0}
};

bool FattnNode::ValidateCase() {
    dnnl::memory::dims kDims = m_kDesc.get_dims();
    dnnl::memory::dims vDims = m_vDesc.get_dims();
    int64_t dkq = int64_t(kDims[3]);
    int64_t dv = int64_t(vDims[3]);
    for (int i = 0; g_fattnCase[i][0] != 0; i++) {
        const int64_t *c = g_fattnCase[i];
        if (c[0] == dkq && c[1] == dv) {
            return true;
        }
    }
    return false;
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
    const char *kernelCode = kernels::FattnTileKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "fattn_tile", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string FattnNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*fattn_tile");
    sb.Int(m_config.Dkq());
    sb.Int(m_config.Dv());
    sb.Int(m_config.Ncols1());
    sb.Int(m_config.Ncols2());
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
//    FattnTile
//

FattnTile::FattnTile(Context *context):
        m_context(context) { }

FattnTile::~FattnTile() { }

std::unique_ptr<core::Node> FattnTile::CreateNode(
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

