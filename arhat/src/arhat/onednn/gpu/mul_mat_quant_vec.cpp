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

#include "arhat/onednn/base/runtime.hpp"
#include "arhat/onednn/base/quant.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/common_xe.hpp"
#include "arhat/onednn/ocl/shape_info_args.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/quantize.hpp"
#include "arhat/onednn/gpu/quant_traits.hpp"
#include "arhat/onednn/gpu/mul_mat.hpp"
#include "arhat/onednn/gpu/mul_mat_id.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

constexpr bool USE_SOA_Y = false;

//
//    MulMatTraits
//

struct MulMatTraits {
    const char *(*vecDotDefsCode)() = nullptr;
    const char *(*vecDotImplCode)() = nullptr;
    const char *(*vecDotCode)() = nullptr;
    std::string vecDotQ;
    std::string vdr;
};

MulMatTraits GetMulMatTraits(base::QuantMode quant) {
    switch (quant) {
    case base::QuantMode::Q4_0:
        return {
            kernels::VecDotDefs_Q4_0_Code, 
            kernels::VecDotImpl_Q4_0_Code, 
            kernels::VecDot_Q4_0_Code, 
            "vec_dot_q4_0_q8_1",
            "VDR_Q4_0_Q8_1_MMVQ"
        };
    case base::QuantMode::Q4_1:
        return {
            kernels::VecDotDefs_Q4_1_Code, 
            kernels::VecDotImpl_Q4_1_Code, 
            kernels::VecDot_Q4_1_Code, 
            "vec_dot_q4_1_q8_1",
            "VDR_Q4_1_Q8_1_MMVQ"
        };
    case base::QuantMode::Q5_0:
        return {
            kernels::VecDotDefs_Q5_0_Code, 
            kernels::VecDotImpl_Q5_0_Code, 
            kernels::VecDot_Q5_0_Code, 
            "vec_dot_q5_0_q8_1",
            "VDR_Q5_0_Q8_1_MMVQ"
        };
    case base::QuantMode::Q5_1:
        return {
            kernels::VecDotDefs_Q5_1_Code, 
            kernels::VecDotImpl_Q5_1_Code, 
            kernels::VecDot_Q5_1_Code, 
            "vec_dot_q5_1_q8_1",
            "VDR_Q5_1_Q8_1_MMVQ"
        };
    case base::QuantMode::Q8_0:
        return {
            kernels::VecDotDefs_Q8_0_Code, 
            kernels::VecDotImpl_Q8_0_Code, 
            kernels::VecDot_Q8_0_Code, 
            "vec_dot_q8_0_q8_1",
            "VDR_Q8_0_Q8_1_MMVQ"
        };
    case base::QuantMode::Q2_K:
        return {
            kernels::VecDotDefs_Q2_K_Code, 
            kernels::VecDotImpl_Q2_K_Code, 
            kernels::VecDot_Q2_K_Code, 
            "vec_dot_q2_K_q8_1",
            "VDR_Q2_K_Q8_1_MMVQ"
        };
    case base::QuantMode::Q3_K:
        return {
            kernels::VecDotDefs_Q3_K_Code, 
            kernels::VecDotImpl_Q3_K_Code, 
            kernels::VecDot_Q3_K_Code, 
            "vec_dot_q3_K_q8_1",
            "VDR_Q3_K_Q8_1_MMVQ"
        };
    case base::QuantMode::Q4_K:
        return {
            kernels::VecDotDefs_Q4_K_Code, 
            kernels::VecDotImpl_Q4_K_Code, 
            kernels::VecDot_Q4_K_Code, 
            "vec_dot_q4_K_q8_1",
            "VDR_Q4_K_Q8_1_MMVQ"
        };
    case base::QuantMode::Q5_K:
        return {
            kernels::VecDotDefs_Q5_K_Code, 
            kernels::VecDotImpl_Q5_K_Code, 
            kernels::VecDot_Q5_K_Code, 
            "vec_dot_q5_K_q8_1",
            "VDR_Q5_K_Q8_1_MMVQ"
        };
    case base::QuantMode::Q6_K:
        return {
            kernels::VecDotDefs_Q6_K_Code, 
            kernels::VecDotImpl_Q6_K_Code, 
            kernels::VecDot_Q6_K_Code, 
            "vec_dot_q6_K_q8_1",
            "VDR_Q6_K_Q8_1_MMVQ"
        };
    case base::QuantMode::MXFP4:
        return {
            kernels::VecDotDefs_Mxfp4_Code, 
            nullptr, 
            kernels::VecDot_Mxfp4_Code, 
            "vec_dot_mxfp4_q8_1",
            "VDR_MXFP4_Q8_1_MMVQ"
        };
    default:
        assert(false);
        return {};
    }
}

// TODO: Merge into MulMatTraits
uint32_t GetVdr(base::QuantMode quant) {
    constexpr uint32_t VDR_Q4_0_Q8_1_MMVQ = 2;
    constexpr uint32_t VDR_Q4_1_Q8_1_MMVQ = 2;
    constexpr uint32_t VDR_Q5_0_Q8_1_MMVQ = 2;
    constexpr uint32_t VDR_Q5_1_Q8_1_MMVQ = 2;
    constexpr uint32_t VDR_Q8_0_Q8_1_MMVQ = 2;
    constexpr uint32_t VDR_Q2_K_Q8_1_MMVQ = 1;
    constexpr uint32_t VDR_Q3_K_Q8_1_MMVQ = 1;
    constexpr uint32_t VDR_Q4_K_Q8_1_MMVQ = 2;
    constexpr uint32_t VDR_Q5_K_Q8_1_MMVQ = 2;
    constexpr uint32_t VDR_Q6_K_Q8_1_MMVQ = 1;
    constexpr uint32_t VDR_MXFP4_Q8_1_MMVQ = 2;

    switch (quant) {
    case base::QuantMode::Q4_0:
        return VDR_Q4_0_Q8_1_MMVQ;
    case base::QuantMode::Q4_1:
        return VDR_Q4_1_Q8_1_MMVQ;
    case base::QuantMode::Q5_0:
        return VDR_Q5_0_Q8_1_MMVQ;
    case base::QuantMode::Q5_1:
        return VDR_Q5_1_Q8_1_MMVQ;
    case base::QuantMode::Q8_0:
        return VDR_Q8_0_Q8_1_MMVQ;
    case base::QuantMode::Q2_K:
        return VDR_Q2_K_Q8_1_MMVQ;
    case base::QuantMode::Q3_K:
        return VDR_Q3_K_Q8_1_MMVQ;
    case base::QuantMode::Q4_K:
        return VDR_Q4_K_Q8_1_MMVQ;
    case base::QuantMode::Q5_K:
        return VDR_Q5_K_Q8_1_MMVQ;
    case base::QuantMode::Q6_K:
        return VDR_Q6_K_Q8_1_MMVQ;
    case base::QuantMode::MXFP4:
        return VDR_MXFP4_Q8_1_MMVQ;
    default:
        assert(false);
        return 0;
    }
}

//
//    MulMatConfig
//

class MulMatConfig {
public:
    MulMatConfig();
    ~MulMatConfig();
public:
    void Init(
        const dnnl::memory::desc &aDesc,
        const dnnl::memory::desc &bDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &cDesc,
        const dnnl::memory::dims &bDims,
        base::QuantMode bQuant,
        base::QuantMode aTempQuant,
        const dnnl::memory::desc &aTempDesc,
        bool useSoaY);
    int64_t NcolsX() const {
        return m_ncolsX;
    }
    int64_t NrowsX() const {
        return m_nrowsX;
    }
    int64_t NcolsDst() const {
        return m_ncolsDst;
    }
    int64_t NchannelsDst() const {
        return m_nchannelsDst;
    }
    int64_t NsamplesDst() const {
        return m_nsamplesDst;
    }
    void GetArgs(
        uint32_t &ncolsX, 
        uint32_t &nchannelsY, 
        uint32_t &strideRowX, 
        uint32_t &strideColY,
        uint32_t &strideColDst, 
        uint32_t &channelRatio, 
        uint32_t &strideChannelX,
        uint32_t &strideChannelY, 
        uint32_t &strideChannelDst, 
        uint32_t &sampleRatio,
        uint32_t &strideSampleX, 
        uint32_t &strideSampleY, 
        uint32_t &strideSampleDst,
        uint32_t &idsStride);
private:
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_idsDesc;
    dnnl::memory::desc m_cDesc;
    dnnl::memory::dims m_bDims;
    base::QuantMode m_bQuant;
    base::QuantMode m_aTempQuant;
    dnnl::memory::desc m_aTempDesc;
    bool m_useSoaY;
    int64_t m_ncolsX; 
    int64_t m_nrowsX; 
    int64_t m_ncolsDst;
    int64_t m_strideRowX; 
    int64_t m_strideColY; 
    int64_t m_strideColDst;
    int64_t m_nchannelsX; 
    int64_t m_nchannelsY; 
    int64_t m_nchannelsDst;
    int64_t m_strideChannelX; 
    int64_t m_strideChannelY; 
    int64_t m_strideChannelDst;
    int64_t m_nsamplesX; 
    int64_t m_nsamplesDst; 
    int64_t m_strideSampleX; 
    int64_t m_strideSampleY; 
    int64_t m_strideSampleDst;
    int64_t m_idsStride; 
};

MulMatConfig::MulMatConfig():
        m_bQuant(base::QuantMode::None),
        m_aTempQuant(base::QuantMode::None),
        m_useSoaY(false),
        m_ncolsX(0), 
        m_nrowsX(0), 
        m_ncolsDst(0),
        m_strideRowX(0), 
        m_strideColY(0), 
        m_strideColDst(0),
        m_nchannelsX(0), 
        m_nchannelsY(0), 
        m_nchannelsDst(0),
        m_strideChannelX(0), 
        m_strideChannelY(0), 
        m_strideChannelDst(0),
        m_nsamplesX(0), 
        m_nsamplesDst(0), 
        m_strideSampleX(0), 
        m_strideSampleY(0), 
        m_strideSampleDst(0),
        m_idsStride(0) { }

MulMatConfig::~MulMatConfig() { }

void MulMatConfig::Init(
        const dnnl::memory::desc &aDesc,
        const dnnl::memory::desc &bDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &cDesc,
        const dnnl::memory::dims &bDims,
        base::QuantMode bQuant,
        base::QuantMode aTempQuant,
        const dnnl::memory::desc &aTempDesc,
        bool useSoaY) {
    m_aDesc = aDesc;
    m_bDesc = bDesc;
    m_idsDesc = idsDesc;
    m_cDesc = cDesc;
    m_bDims = bDims;
    m_bQuant = bQuant;
    m_aTempQuant = aTempQuant;
    m_aTempDesc = aTempDesc;
    m_useSoaY = useSoaY;

    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims aTempStrides = m_aTempDesc.get_strides();
    dnnl::memory::dims bStrides = m_bDesc.get_strides();
    dnnl::memory::dims idsStrides;
    dnnl::memory::dims cDims = m_cDesc.get_dims();
    dnnl::memory::dims cStrides = m_cDesc.get_strides();
    int bBlockSize = base::GetBlockSize(m_bQuant);
    bool haveIds = !m_idsDesc.is_zero();

    int64_t aDim1 = int64_t(aDims[1]);
    int64_t aDim2 = int64_t(aDims[2]);

    int64_t aTempStride0 = int64_t(aTempStrides[0]);
    int64_t aTempStride1 = int64_t(aTempStrides[1]);
    int64_t aTempStride2 = int64_t(aTempStrides[2]);
    if (!m_useSoaY) {
        int aTempBlockSize = base::GetBlockSize(m_aTempQuant);
        aTempStride0 /= aTempBlockSize;
        aTempStride1 /= aTempBlockSize;
        aTempStride2 /= aTempBlockSize;
    }

    int64_t bDim0 = int64_t(m_bDims[0]);
    int64_t bDim1 = int64_t(m_bDims[1]);
    int64_t bDim2 = int64_t(m_bDims[2]);
    int64_t bDim3 = int64_t(m_bDims[3]);

    int64_t bStride0 = int64_t(bStrides[0]) / bBlockSize;
    int64_t bStride1 = int64_t(bStrides[1]) / bBlockSize;
    int64_t bStride2 = int64_t(bStrides[2]) / bBlockSize;

    int64_t cDim0 = int64_t(cDims[0]);
    int64_t cDim1 = int64_t(cDims[1]);
    int64_t cDim2 = int64_t(cDims[2]);

    int64_t cStride0 = int64_t(cStrides[0]);
    int64_t cStride1 = int64_t(cStrides[1]);
    int64_t cStride2 = int64_t(cStrides[2]);

    int64_t idsStride2 = 0;
    if (haveIds) {
        idsStrides = m_idsDesc.get_strides();
        idsStride2 = uint32_t(idsStrides[2]);
    }

    m_ncolsX = bDim3; 
    m_nrowsX = bDim2; 
    m_ncolsDst = haveIds ? cDim1 : cDim2;
    m_strideRowX = bStride2; 
    m_strideColY = haveIds ? aTempStride1 : aTempStride2; 
    m_strideColDst = haveIds ? cStride1 : cStride2;
    m_nchannelsX = bDim1; 
    m_nchannelsY = haveIds ? aDim2 : aDim1; 
    m_nchannelsDst = haveIds ? cDim2 : cDim1;
    m_strideChannelX = bStride1; 
    m_strideChannelY = haveIds ? aTempStride2 : aTempStride1; 
    m_strideChannelDst = haveIds ? cStride2 : cStride1;
    m_nsamplesX = bDim0; 
    m_nsamplesDst = cDim0; 
    m_strideSampleX = bStride0; 
    m_strideSampleY = aTempStride0; 
    m_strideSampleDst = cStride0;
    m_idsStride = idsStride2; 
}

void MulMatConfig::GetArgs(
        uint32_t &ncolsX, 
        uint32_t &nchannelsY, 
        uint32_t &strideRowX, 
        uint32_t &strideColY,
        uint32_t &strideColDst, 
        uint32_t &channelRatio, 
        uint32_t &strideChannelX,
        uint32_t &strideChannelY, 
        uint32_t &strideChannelDst, 
        uint32_t &sampleRatio,
        uint32_t &strideSampleX, 
        uint32_t &strideSampleY, 
        uint32_t &strideSampleDst,
        uint32_t &idsStride) {
    bool haveIds = !m_idsDesc.is_zero();
    ncolsX = uint32_t(m_ncolsX); 
    nchannelsY = haveIds ? uint32_t(m_nchannelsY) : 0; 
    strideRowX = uint32_t(m_strideRowX); 
    strideColY = uint32_t(m_strideColY);
    strideColDst = uint32_t(m_strideColDst); 
    channelRatio = haveIds ? 0 : uint32_t(m_nchannelsDst / m_nchannelsX); 
    strideChannelX = uint32_t(m_strideChannelX);
    strideChannelY = uint32_t(m_strideChannelY); 
    strideChannelDst = uint32_t(m_strideChannelDst); 
    sampleRatio = uint32_t(m_nsamplesDst / m_nsamplesX);
    strideSampleX = uint32_t(m_strideSampleX); 
    strideSampleY = uint32_t(m_strideSampleY); 
    strideSampleDst = uint32_t(m_strideSampleDst);
    idsStride = uint32_t(m_idsStride);
}

//
//    MulMatNode
//

class MulMatNode: public NodeBase {
public:
    MulMatNode(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
    ~MulMatNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    void InferShapes();
    void InitQuantize();
    void InitConfig();
    bool CanUseReorder();
    bool CanUseSoaY();
    bool UseSmallK();
    void InitArgs();
    void InitShapeInfo();
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
    bool m_isNop;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::memory m_idsMem;
    dnnl::memory::dims m_bDims;
    base::QuantMode m_bQuant;
    base::QuantMode m_aTempQuant;
    dnnl::memory::dim m_aTempDim3;
    dnnl::memory::desc m_aTempDesc;
    base::TempMemory m_aTempMem;
    bool m_useReorder;
    bool m_useSoaY;
    std::unique_ptr<QuantizeVecMm> m_quantize;
    MulMatConfig m_config;
    uint32_t m_ncolsDst;
    uint32_t m_cNcolsDst;
    bool m_isMultiTokenId;
    uint32_t m_sgSize;
    uint32_t m_numSgs;
    uint32_t m_rowsPerWg;
    QuantTraits m_quantTraits;
    MulMatTraits m_traits;
    uint32_t m_ncolsX; 
    uint32_t m_nchannelsY; 
    uint32_t m_nchannelsY_fd0; 
    uint32_t m_nchannelsY_fd1; 
    uint32_t m_strideRowX; 
    uint32_t m_strideColY;
    uint32_t m_strideColDst; 
    uint32_t m_channelRatio; 
    uint32_t m_channelRatio_fd0; 
    uint32_t m_channelRatio_fd1; 
    uint32_t m_strideChannelX;
    uint32_t m_strideChannelY; 
    uint32_t m_strideChannelDst; 
    uint32_t m_sampleRatio;
    uint32_t m_sampleRatio_fd0;
    uint32_t m_sampleRatio_fd1;
    uint32_t m_strideSampleX; 
    uint32_t m_strideSampleY; 
    uint32_t m_strideSampleDst;
    uint32_t m_idsStride;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

MulMatNode::MulMatNode(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids):
            NodeBase(context),
            m_a(a),
            m_b(b),
            m_ids(ids),
            m_isNop(false),
            m_bQuant(base::QuantMode::None),
            m_aTempQuant(base::QuantMode::None),
            m_aTempDim3(0),
            m_useReorder(false),
            m_useSoaY(false),
            m_ncolsDst(0),
            m_cNcolsDst(0),
            m_isMultiTokenId(false),
            m_sgSize(0),
            m_numSgs(0),
            m_rowsPerWg(0),
            m_ncolsX(0), 
            m_nchannelsY(0), 
            m_nchannelsY_fd0(0), 
            m_nchannelsY_fd1(0), 
            m_strideRowX(0), 
            m_strideColY(0),
            m_strideColDst(0), 
            m_channelRatio(0), 
            m_channelRatio_fd0(0), 
            m_channelRatio_fd1(0), 
            m_strideChannelX(0),
            m_strideChannelY(0), 
            m_strideChannelDst(0), 
            m_sampleRatio(0),
            m_sampleRatio_fd0(0),
            m_sampleRatio_fd1(0),
            m_strideSampleX(0), 
            m_strideSampleY(0), 
            m_strideSampleDst(0),
            m_idsStride(0) { }

MulMatNode::~MulMatNode() { }

bool MulMatNode::Init() {
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    base::NodeBase *ids = m_gpuContext->CastNode(m_ids);
    m_aDesc = a->MemoryDesc();
    m_bDesc = b->MemoryDesc();
    if (ids != nullptr) {
        m_idsDesc = ids->MemoryDesc();
    }
    m_bQuant = b->Quant();
    // cannot use m_bDesc.get_dims() for quantized tensors
    m_bDims = b->MemoryDims();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    if (ids != nullptr) {
        m_idsMem = ids->Memory();
    }
    InferShapes();
    SetMemory(m_cDesc);
    if (m_isNop) {
        return true;
    }
    m_context->MemoryPoolStart();
    InitQuantize();
    InitConfig();
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void MulMatNode::Compute() {
    if (m_isNop) {
        return;
    }

    m_quantize->Compute(m_aMem, {}, m_aTempMem.Get());

    // vx = b, vy = a
    m_kernel->SetArgBuffer(0, m_bMem);
    m_kernel->SetArgBuffer(1, m_aTempMem.Get());
    m_kernel->SetArgBuffer(2, m_idsMem);
    m_kernel->SetArgBuffer(3, m_memory);
    m_kernel->SetArgU32(4, m_ncolsX); 
    m_kernel->SetArgU32(5, m_nchannelsY); 
    m_kernel->SetArgU32(6, m_nchannelsY_fd0); 
    m_kernel->SetArgU32(7, m_nchannelsY_fd1); 
    m_kernel->SetArgU32(8, m_strideRowX); 
    m_kernel->SetArgU32(9, m_strideColY);
    m_kernel->SetArgU32(10, m_strideColDst); 
    m_kernel->SetArgU32(11, m_channelRatio); 
    m_kernel->SetArgU32(12, m_channelRatio_fd0); 
    m_kernel->SetArgU32(13, m_channelRatio_fd1); 
    m_kernel->SetArgU32(14, m_strideChannelX);
    m_kernel->SetArgU32(15, m_strideChannelY); 
    m_kernel->SetArgU32(16, m_strideChannelDst); 
    m_kernel->SetArgU32(17, m_sampleRatio);
    m_kernel->SetArgU32(18, m_sampleRatio_fd0);
    m_kernel->SetArgU32(19, m_sampleRatio_fd1);
    m_kernel->SetArgU32(20, m_strideSampleX); 
    m_kernel->SetArgU32(21, m_strideSampleY); 
    m_kernel->SetArgU32(22, m_strideSampleDst);
    m_kernel->SetArgU32(23, m_idsStride);
    m_kernel->SetArgU32(24, uint32_t(m_aTempDim3));
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 25);
    m_kernel->Launch(m_ndRange);
}

bool MulMatNode::Validate() {
    if (m_aDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_aDesc) || !MemoryDescUtil::HasDenseRows(m_aDesc)) { 
        return false;
    }
    if (!m_idsDesc.is_zero()) {
        if (m_idsDesc.get_data_type() != dnnl::memory::data_type::s32) {
            return false;
        }
        if (!MemoryDescUtil::IsPlain(m_idsDesc) || !MemoryDescUtil::HasDenseRows(m_idsDesc)) { 
            return false;
        }
    }
    switch (m_bQuant) {
    case base::QuantMode::Q4_0:
    case base::QuantMode::Q4_1:
    case base::QuantMode::Q5_0:
    case base::QuantMode::Q5_1:
    case base::QuantMode::Q8_0:
    case base::QuantMode::Q2_K:
    case base::QuantMode::Q3_K:
    case base::QuantMode::Q4_K:
    case base::QuantMode::Q5_K:
    case base::QuantMode::Q6_K:
    case base::QuantMode::MXFP4:
        break;
    default:
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_bDesc) || !MemoryDescUtil::HasDenseRows(m_bDesc)) { 
        return false;
    }
    // at this point cDims or config are not yet available, so calculate based on input dims
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dim ncolsDst = (m_ids == nullptr) ? aDims[2] : aDims[1];
    if (ncolsDst > MMVQ_MAX_BATCH_SIZE) {
        return false;
    }
    dnnl::memory::dim ncolsX = m_bDims[3]; 
    if (ncolsX % base::GetQuantSize(m_bQuant) != 0) {
        return false;
    }
    if (aDims[3] % base::QK8_1 != 0) {
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
    if (m_ids == nullptr) {
        cDims[0] = aDims[0];
        cDims[1] = aDims[1];
        cDims[2] = aDims[2];
        cDims[3] = bDims[2];
    } else {
        dnnl::memory::dims idsDims = m_idsDesc.get_dims();
        assert(idsDims.size() == 4);
        cDims[0] = 1;
        cDims[1] = aDims[1];
        cDims[2] = idsDims[3];
        cDims[3] = bDims[2];
    }
    m_cDesc = 
        dnnl::memory::desc(
            cDims, 
            dnnl::memory::data_type::f32, 
            dnnl::memory::format_tag::abcd);
    m_isNop = (cDims[0] * cDims[1] * cDims[2] * cDims[3] == 0);
}

void MulMatNode::InitQuantize() {
    m_useSoaY = (USE_SOA_Y && CanUseSoaY());
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    m_aTempDim3 = (aDims[3] + MATRIX_ROW_PADDING - 1) & ~(MATRIX_ROW_PADDING - 1);
    m_aTempQuant = base::QuantMode::Q8_1;
    int blockSize = base::GetBlockSize(m_aTempQuant);
    int quantSize = base::GetQuantSize(m_aTempQuant);
    assert(m_aTempDim3 % quantSize == 0);
    if (!m_useSoaY) {
        aDims[3] = (m_aTempDim3 / quantSize) * blockSize;
        m_aTempDesc = 
            dnnl::memory::desc(
                aDims, 
                dnnl::memory::data_type::u8, 
                dnnl::memory::format_tag::abcd);
        m_aTempMem = m_context->AllocTempMemory(m_aTempDesc);
        m_quantize = CreateQuantizeVec_Q8_1(GetGpuContext(), m_aDesc, m_aTempDesc);
    } else {
        // Note different memory descriptors used for allocation and further processing
        // This is required because AllocTempMemory wants dense layouts
        dnnl::memory::dim fullDim3 = (m_aTempDim3 / quantSize) * blockSize;
        aDims[3] = fullDim3;
        dnnl::memory::desc fullDesc(
            aDims, 
            dnnl::memory::data_type::u8, 
            dnnl::memory::format_tag::abcd);
        m_aTempMem = m_context->AllocTempMemory(fullDesc);
        aDims[3] = m_aTempDim3;
        dnnl::memory::dims aTempStrides(4);
        aTempStrides[3] = 1;
        aTempStrides[2] = fullDim3;
        aTempStrides[1] = aTempStrides[2] * aDims[2];
        aTempStrides[0] = aTempStrides[1] * aDims[1];
        m_aTempDesc = 
            dnnl::memory::desc(
                aDims, 
                dnnl::memory::data_type::u8, 
                aTempStrides);
        m_quantize = CreateQuantizeVec_Q8_1_Soa(GetGpuContext(), m_aDesc, m_aTempDesc);
    }
}

void MulMatNode::InitConfig() {
    m_config.Init(
        m_aDesc,
        m_bDesc,
        m_idsDesc,
        m_cDesc,
        m_bDims,
        m_bQuant,
        m_aTempQuant,
        m_aTempDesc,
        m_useSoaY);
    m_ncolsDst = uint32_t(m_config.NcolsDst());
    if (m_ids == nullptr) {
        m_cNcolsDst = m_ncolsDst;
        m_isMultiTokenId = false;
    } else {
        m_cNcolsDst = 1;
        m_isMultiTokenId = (m_ncolsDst > 1);
    }
    m_sgSize = 32;
    m_rowsPerWg = (m_cNcolsDst == 1) ? 1 : 2;
    m_numSgs = (m_cNcolsDst <= 4) ? 4 : 2;
    m_quantTraits = GetQuantTraits(m_bQuant);
    m_traits = GetMulMatTraits(m_bQuant);
#if 0 // EXPERIMENTAL
    if (UseSmallK()) {
        m_rowsPerWg = m_numSgs;
    }
#endif
}

bool MulMatNode::CanUseSoaY() {
    switch (m_bQuant) {
    case base::QuantMode::Q4_0:
    case base::QuantMode::Q4_1:
    case base::QuantMode::Q5_0:
    case base::QuantMode::Q5_1:
    case base::QuantMode::Q8_0:
    case base::QuantMode::Q2_K:
    case base::QuantMode::Q3_K:
    case base::QuantMode::Q4_K:
    case base::QuantMode::Q5_K:
    case base::QuantMode::Q6_K:
    case base::QuantMode::MXFP4:
        return true;
    default:
        return false;
    }
}

bool MulMatNode::CanUseReorder() {
    // RESERVED
    return false;
}

bool MulMatNode::UseSmallK() {
    if (m_cNcolsDst != 1) {
        return false;
    }
    if (m_numSgs == 1) {
        return false;
    }
    uint32_t ncolsX = uint32_t(m_config.NcolsX());
    uint32_t qk = uint32_t(m_quantTraits.qk);
    uint32_t qi = uint32_t(m_quantTraits.qi);
    uint32_t vdr = GetVdr(m_bQuant);
    uint32_t blocksPerRowX = ncolsX / qk;
    uint32_t blocksPerIterOneSg = vdr * m_sgSize / qi;
    return (blocksPerRowX < m_numSgs * blocksPerIterOneSg);
}

void MulMatNode::InitArgs() {
    m_config.GetArgs(
        m_ncolsX, 
        m_nchannelsY, 
        m_strideRowX, 
        m_strideColY,
        m_strideColDst, 
        m_channelRatio, 
        m_strideChannelX,
        m_strideChannelY, 
        m_strideChannelDst, 
        m_sampleRatio,
        m_strideSampleX, 
        m_strideSampleY, 
        m_strideSampleDst,
        m_idsStride);

    if (m_ids != nullptr) {
        ocl::MakeFastDiv(int64_t(m_nchannelsY), m_nchannelsY_fd0, m_nchannelsY_fd1); 
    }
    if (m_ids == nullptr) {
        ocl::MakeFastDiv(int64_t(m_channelRatio), m_channelRatio_fd0, m_channelRatio_fd1); 
    }
    ocl::MakeFastDiv(int64_t(m_sampleRatio), m_sampleRatio_fd0, m_sampleRatio_fd1);

    InitShapeInfo();
}

void MulMatNode::InitShapeInfo() {
    size_t aBase = m_aDesc.get_submemory_offset();
    size_t bBase = m_bDesc.get_submemory_offset();
    size_t idsBase = (m_ids != nullptr) ? m_idsDesc.get_submemory_offset() : 0;
    size_t cBase = m_cDesc.get_submemory_offset();
    // vx = b, vy = a
    m_shapeInfoArgs.AddS64("SRC0_BASE", bBase);
    m_shapeInfoArgs.AddS64("SRC1_BASE", aBase);
    m_shapeInfoArgs.AddS64("SRC2_BASE", idsBase);
    m_shapeInfoArgs.AddS64("DST_BASE", cBase);
}

void MulMatNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::MulMatQuantVecKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "mul_mat_vec_q", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string MulMatNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*mul_mat_vec_q");
    sb.Int(int64_t(m_bQuant));
    sb.Int(int64_t(m_cNcolsDst));
    sb.Bool(m_isMultiTokenId);
    return sb.Get();
}

std::string MulMatNode::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    ocl::CommonXe::EmitImad(ss);
    ocl::CommonXe::EmitFastDiv(ss);
    EmitInt(ss, "USE_SOA_Y", m_useSoaY ? 1 : 0);
    ss << "#define VDR " << m_traits.vdr << "\n";
    ss << "\n";
    ss << kernels::VecDotQuantCommonCode();
    ss << m_traits.vecDotDefsCode();
    if (m_traits.vecDotImplCode != nullptr) {
        ss << m_traits.vecDotImplCode();
    }
    ss << m_traits.vecDotCode();
    EmitInt(ss, "NCOLS_DST", m_cNcolsDst);
    EmitInt(ss, "IS_MULTI_TOKEN_ID", m_isMultiTokenId ? 1 : 0);
    EmitInt(ss, "SG_SIZE", m_sgSize);
    EmitInt(ss, "NUM_SGS", m_numSgs);
    EmitInt(ss, "ROWS_PER_WG", m_rowsPerWg);
    EmitInt(ss, "QK", m_quantTraits.qk);
    EmitInt(ss, "QI", m_quantTraits.qi);
    EmitInt(ss, "BLOCK_SIZE", base::GetBlockSize(m_bQuant));
    ss << "#define BLOCKS_PER_ITER ((VDR * " << ocl::FormatInt(m_numSgs * m_sgSize) << ") / QI)\n";
    ss << "#define VEC_DOT_Q " << m_traits.vecDotQ << "\n";
    ss << "\n";
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    return ss.str();
}

void MulMatNode::InitNdRange() {
    size_t nrowsX = size_t(m_config.NrowsX());
    size_t rowsPerWg = size_t(m_rowsPerWg);
    size_t nchannelsDst = size_t(m_config.NchannelsDst());
    size_t ncolsDst = size_t(m_ncolsDst);
    size_t nsamplesDst = size_t(m_config.NsamplesDst());

    size_t lws0 = size_t(m_sgSize);
    size_t lws1 = size_t(m_numSgs);
    size_t gws0 = ocl::DivUp(nrowsX, rowsPerWg) * lws0;
    size_t gws1 = nchannelsDst * lws1;
    size_t gws2 = m_isMultiTokenId ? ncolsDst : nsamplesDst;

    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

} // namespace

//
//    MulMatQuantVec
//

MulMatQuantVec::MulMatQuantVec(Context *context):
        m_context(context) { }

MulMatQuantVec::~MulMatQuantVec() { }

std::unique_ptr<core::Node> MulMatQuantVec::CreateNode(core::Node *a, core::Node *b) {
    std::unique_ptr<MulMatNode> node = 
        std::make_unique<MulMatNode>(m_context, a, b, nullptr);
    if (node->Init()) {
        return node;
    }
    return nullptr;
}

//
//    MulMatIdQuantVec
//

MulMatIdQuantVec::MulMatIdQuantVec(Context *context):
        m_context(context) { }

MulMatIdQuantVec::~MulMatIdQuantVec() { }

std::unique_ptr<core::Node> MulMatIdQuantVec::CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *ids) {
    std::unique_ptr<MulMatNode> node = 
        std::make_unique<MulMatNode>(m_context, a, b, ids);
    if (node->Init()) {
        return node;
    }
    return nullptr;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

