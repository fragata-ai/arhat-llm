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

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/common_xe.hpp"
#include "arhat/onednn/ocl/shape_info_args.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/mul_mat.hpp"
#include "arhat/onednn/gpu/mul_mat_id.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

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
        const dnnl::memory::desc &cDesc);
    void GetArgs(
        int32_t &ncols2, 
        uint32_t &nchannelsY, 
        int32_t &strideRow, 
        int32_t &strideColY2, 
        int32_t &strideColDst,
        uint32_t &channelRatio, 
        int32_t &strideChannelX, 
        int32_t &strideChannelY, 
        int32_t &strideChannelDst,
        uint32_t &sampleRatio, 
        int32_t &strideSampleX, 
        int32_t &strideSampleY, 
        int32_t &strideSampleDst,
        int32_t &idsStride);
    int64_t Ncols() const {
        return m_ncols;
    }
    int64_t Nrows() const {
        return m_nrows;
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
private:
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_idsDesc;
    dnnl::memory::desc m_cDesc;
    int64_t m_ncols;
    int64_t m_nrows;
    int64_t m_ncolsDst;
    int64_t m_strideRow;
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
        m_ncols(0),
        m_nrows(0),
        m_ncolsDst(0),
        m_strideRow(0),
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
        const dnnl::memory::desc &cDesc) {
    m_aDesc = aDesc;
    m_bDesc = bDesc;
    m_idsDesc = idsDesc;
    m_cDesc = cDesc;

    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims aStrides = m_aDesc.get_strides();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    dnnl::memory::dims bStrides = m_bDesc.get_strides();
    dnnl::memory::dims cDims = m_cDesc.get_dims();
    dnnl::memory::dims cStrides = m_cDesc.get_strides();
    bool haveIds = !m_idsDesc.is_zero();

    int64_t aDim1 = int64_t(aDims[1]);
    int64_t aDim2 = int64_t(aDims[2]);

    int64_t aStride0 = int64_t(aStrides[0]);
    int64_t aStride1 = int64_t(aStrides[1]);
    int64_t aStride2 = int64_t(aStrides[2]);

    int64_t bDim0 = int64_t(bDims[0]);
    int64_t bDim1 = int64_t(bDims[1]);
    int64_t bDim2 = int64_t(bDims[2]);
    int64_t bDim3 = int64_t(bDims[3]);

    int64_t bStride0 = int64_t(bStrides[0]);
    int64_t bStride1 = int64_t(bStrides[1]);
    int64_t bStride2 = int64_t(bStrides[2]);

    int64_t cDim0 = int64_t(cDims[0]);
    int64_t cDim1 = int64_t(cDims[1]);
    int64_t cDim2 = int64_t(cDims[2]);

    int64_t cStride0 = int64_t(cStrides[0]);
    int64_t cStride1 = int64_t(cStrides[1]);
    int64_t cStride2 = int64_t(cStrides[2]);

    int64_t idsStride2 = 0;
    if (haveIds) {
        dnnl::memory::dims idsStrides = m_idsDesc.get_strides();
        idsStride2 = uint32_t(idsStrides[2]);
    }

    m_ncols = bDim3;
    m_nrows = bDim2;
    m_ncolsDst = haveIds ? cDim1 : cDim2;
    m_strideRow = bStride2;
    m_strideColY = haveIds ? aStride1 : aStride2;
    m_strideColDst = haveIds ? cStride1 : cStride2;
    m_nchannelsX = bDim1;
    m_nchannelsY = haveIds ? aDim2 : aDim1;
    m_nchannelsDst = haveIds ? cDim2 : cDim1;
    m_strideChannelX = bStride1;
    m_strideChannelY = haveIds ? aStride2 : aStride1;
    m_strideChannelDst = haveIds ? cStride2 : cStride1;
    m_nsamplesX = bDim0;
    m_nsamplesDst = cDim0;
    m_strideSampleX = bStride0;
    m_strideSampleY = aStride0;
    m_strideSampleDst = cStride0;
    m_idsStride = idsStride2; 
}

void MulMatConfig::GetArgs(
        int32_t &ncols2,
        uint32_t &nchannelsY, 
        int32_t &strideRow, 
        int32_t &strideColY2, 
        int32_t &strideColDst,
        uint32_t &channelRatio, 
        int32_t &strideChannelX, 
        int32_t &strideChannelY, 
        int32_t &strideChannelDst,
        uint32_t &sampleRatio, 
        int32_t &strideSampleX, 
        int32_t &strideSampleY, 
        int32_t &strideSampleDst,
        int32_t &idsStride) {
    bool haveIds = !m_idsDesc.is_zero();
    ncols2 = int32_t(m_ncols / 2);
    nchannelsY = haveIds ? uint32_t(m_nchannelsY) : 0;
    strideRow = int32_t(m_strideRow);
    strideColY2 = int32_t(m_strideColY / 2); 
    strideColDst = int32_t(m_strideColDst);
    channelRatio = haveIds ? 0 : uint32_t(m_nchannelsDst / m_nchannelsX);
    strideChannelX = int32_t(m_strideChannelX);
    strideChannelY = int32_t(m_strideChannelY);
    strideChannelDst = int32_t(m_strideChannelDst);
    sampleRatio = uint32_t(m_nsamplesDst  / m_nsamplesX);
    strideSampleX = int32_t(m_strideSampleX);
    strideSampleY = int32_t(m_strideSampleY);
    strideSampleDst = int32_t(m_strideSampleDst);
    idsStride = int32_t(m_idsStride);
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
        core::Node *ids,
        core::Prec prec);
    ~MulMatNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    void InferShapes();
    void InitConfig();
    void GetBlockSize();
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
    core::Prec m_prec;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_idsDesc;
    dnnl::memory::desc m_cDesc;
    bool m_isNop;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::memory m_idsMem;
    MulMatConfig m_config;
    dnnl::memory::data_type m_bType;
    dnnl::memory::data_type m_accType;
    uint32_t m_ncolsDst;
    uint32_t m_cNcolsDst;
    uint32_t m_blockSize;
    bool m_isMultiTokenId;
    uint32_t m_sgSize;
    uint32_t m_neLocal;
    int32_t m_ncols2; 
    uint32_t m_nchannelsY; 
    uint32_t m_nchannelsY_fd0; 
    uint32_t m_nchannelsY_fd1; 
    int32_t m_strideRow; 
    int32_t m_strideColY2; 
    int32_t m_strideColDst;
    uint32_t m_channelRatio; 
    uint32_t m_channelRatio_fd0; 
    uint32_t m_channelRatio_fd1; 
    int32_t m_strideChannelX; 
    int32_t m_strideChannelY; 
    int32_t m_strideChannelDst;
    uint32_t m_sampleRatio; 
    uint32_t m_sampleRatio_fd0; 
    uint32_t m_sampleRatio_fd1; 
    int32_t m_strideSampleX; 
    int32_t m_strideSampleY; 
    int32_t m_strideSampleDst;
    int32_t m_idsStride;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

MulMatNode::MulMatNode(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Node *ids,
        core::Prec prec):
            NodeBase(context),
            m_a(a),
            m_b(b),
            m_ids(ids),
            m_prec(prec),
            m_isNop(false),
            m_bType(dnnl::memory::data_type::undef),
            m_accType(dnnl::memory::data_type::undef),
            m_ncolsDst(0),
            m_cNcolsDst(0),
            m_blockSize(0),
            m_isMultiTokenId(false),
            m_sgSize(0),
            m_neLocal(0),
            m_ncols2(0), 
            m_nchannelsY(0), 
            m_nchannelsY_fd0(0), 
            m_nchannelsY_fd1(0), 
            m_strideRow(0), 
            m_strideColY2(0), 
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

    // x = b, y = a
    m_kernel->SetArgBuffer(0, m_bMem);
    m_kernel->SetArgBuffer(1, m_aMem);
    m_kernel->SetArgBuffer(2, m_idsMem);
    m_kernel->SetArgBuffer(3, m_memory);
    m_kernel->SetArgS32(4, m_ncols2); 
    m_kernel->SetArgU32(5, m_nchannelsY); 
    m_kernel->SetArgU32(6, m_nchannelsY_fd0); 
    m_kernel->SetArgU32(7, m_nchannelsY_fd1); 
    m_kernel->SetArgS32(8, m_strideRow); 
    m_kernel->SetArgS32(9, m_strideColY2); 
    m_kernel->SetArgS32(10, m_strideColDst);
    m_kernel->SetArgU32(11, m_channelRatio); 
    m_kernel->SetArgU32(12, m_channelRatio_fd0); 
    m_kernel->SetArgU32(13, m_channelRatio_fd1); 
    m_kernel->SetArgS32(14, m_strideChannelX); 
    m_kernel->SetArgS32(15, m_strideChannelY); 
    m_kernel->SetArgS32(16, m_strideChannelDst);
    m_kernel->SetArgU32(17, m_sampleRatio); 
    m_kernel->SetArgU32(18, m_sampleRatio_fd0); 
    m_kernel->SetArgU32(19, m_sampleRatio_fd1); 
    m_kernel->SetArgS32(20, m_strideSampleX); 
    m_kernel->SetArgS32(21, m_strideSampleY); 
    m_kernel->SetArgS32(22, m_strideSampleDst);
    m_kernel->SetArgS32(23, m_idsStride);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 24);
    m_kernel->Launch(m_ndRange);
}

bool MulMatNode::Validate() {
    if (m_aDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_aDesc) || !MemoryDescUtil::HasDenseRows(m_aDesc)) { 
        return false;
    }
    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    if (bType != dnnl::memory::data_type::f32 &&
            bType != dnnl::memory::data_type::f16) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_bDesc) || !MemoryDescUtil::HasDenseRows(m_bDesc)) { 
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
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    assert(bDims.size() == 4);
    if (bDims[3] % 2 != 0) {
        return false;
    }
    dnnl::memory::dims bStrides = m_bDesc.get_strides();
    assert(bStrides.size() == 4);
    for (int i = 0; i < 3; i++) {
        if (bStrides[i] % 2 != 0) {
            return false;
        }
    }
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    assert(aDims.size() == 4);
    if (m_ids != nullptr) {
        if (aDims[1] > 8) {
            return false;
        }
    } else {
        if (aDims[2] > 8) {
            return false;
        }
    }
    return true;
}

void MulMatNode::InferShapes() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
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

void MulMatNode::InitConfig() {
    m_config.Init(
        m_aDesc,
        m_bDesc,
        m_idsDesc,
        m_cDesc);
    m_bType = m_bDesc.get_data_type();
    m_accType = 
        (m_bType == dnnl::memory::data_type::f16 && m_prec == core::Prec::Default) ?
            dnnl::memory::data_type::f16 :
            dnnl::memory::data_type::f32;
    m_ncolsDst = uint32_t(m_config.NcolsDst());
    if (m_ids == nullptr) {
        m_cNcolsDst = m_ncolsDst;
        m_isMultiTokenId = false;
    } else {
        m_cNcolsDst = 1;
        m_isMultiTokenId = (m_ncolsDst > 1);
    }
    m_sgSize = 32;
    GetBlockSize();
    m_neLocal = m_sgSize;
}

void MulMatNode::GetBlockSize() {
    uint32_t ncols = uint32_t(m_config.Ncols());
    uint32_t bestBlockSize = m_sgSize;
    uint32_t bestNiter = ocl::DivUp(ncols, 2 * m_sgSize);
    uint32_t maxBlockSize = 256; // make it arch-dependent?
    for (uint32_t blockSize = 2 * m_sgSize; blockSize <= maxBlockSize; blockSize += m_sgSize) {
        uint32_t niter = ocl::DivUp(ncols, 2 * blockSize);
        if (niter < bestNiter) {
            bestNiter = niter;
            bestBlockSize = blockSize;
        }
    }
    m_blockSize = bestBlockSize;
}

void MulMatNode::InitArgs() {
    m_config.GetArgs(
        m_ncols2,
        m_nchannelsY, 
        m_strideRow, 
        m_strideColY2, 
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
    // x = b, y = a
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
    const char *kernelCode = kernels::MulMatVecKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "mul_mat_vec", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string MulMatNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*mul_mat_vec");
    sb.Int(int64_t(m_bType));
    sb.Int(int64_t(m_accType));
    sb.Int(int64_t(m_cNcolsDst));
    sb.Int(int64_t(m_blockSize));
    sb.Bool(m_isMultiTokenId);
    return sb.Get();
}

std::string MulMatNode::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    ocl::CommonXe::EmitFastDiv(ss);
    ss << "#define T " << ocl::FormatType(m_bType) << "\n";
    ss << "#define TYPE_ACC " << ocl::FormatType(m_accType) << "\n";
    EmitInt(ss, "NCOLS_DST", m_cNcolsDst);
    EmitInt(ss, "BLOCK_SIZE", m_blockSize);
    EmitInt(ss, "IS_MULTI_TOKEN_ID", m_isMultiTokenId ? 1 : 0);
    EmitInt(ss, "SG_SIZE", m_sgSize);
    EmitInt(ss, "NE_LOCAL", m_neLocal);
    if (m_bType == dnnl::memory::data_type::f32) {
        ss << "#define T_IS_FLOAT\n";
    } else if (m_bType == dnnl::memory::data_type::f16) {
        ss << "#define T_IS_HALF\n";
    }
    if (m_accType == dnnl::memory::data_type::f32) {
        ss << "#define TYPE_ACC_IS_FLOAT\n";
    } else if (m_accType == dnnl::memory::data_type::f16) {
        ss << "#define TYPE_ACC_IS_HALF\n";
    }
    ss << "\n";
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    return ss.str();
}

void MulMatNode::InitNdRange() {
    size_t nrows = size_t(m_config.Nrows());
    size_t nchannelsDst = size_t(m_config.NchannelsDst());
    size_t nsamplesOrNtokens = 
        (m_ids != nullptr) ? 
            size_t(m_config.NcolsDst()) : 
            size_t(m_config.NsamplesDst());
    size_t lws0 = m_blockSize;
    size_t gws0 = nrows * lws0;
    size_t gws1 = nchannelsDst;
    size_t gws2 = nsamplesOrNtokens;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

} // namespace

//
//    MulMatVec
//

MulMatVec::MulMatVec(Context *context):
        m_context(context) { }

MulMatVec::~MulMatVec() { }

std::unique_ptr<core::Node> MulMatVec::CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Prec prec) {
    std::unique_ptr<MulMatNode> node = 
        std::make_unique<MulMatNode>(m_context, a, b, nullptr, prec);
    if (node->Init()) {
        return node;
    }
    return nullptr;
}

//
//    MulMatIdVec
//

MulMatIdVec::MulMatIdVec(Context *context):
        m_context(context) { }

MulMatIdVec::~MulMatIdVec() { }

std::unique_ptr<core::Node> MulMatIdVec::CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *ids) {
    std::unique_ptr<MulMatNode> node = 
        std::make_unique<MulMatNode>(m_context, a, b, ids, core::Prec::Default);
    if (node->Init()) {
        return node;
    }
    return nullptr;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

