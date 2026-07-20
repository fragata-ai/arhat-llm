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
#include <algorithm>

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
#include "arhat/onednn/gpu/mul_mat_id_helper.hpp"
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
        int32_t &ncols, 
        int32_t &ncolsDstTotal, 
        int32_t &nchannelsDst, 
        int32_t &strideRow, 
        int32_t &strideColY, 
        int32_t &strideColDst,
        int32_t &strideColId,
        int32_t &strideRowId,
        int32_t &channelRatio, 
        int32_t &strideChannelX, 
        int32_t &strideChannelY, 
        int32_t &strideChannelDst,
        int32_t &sampleRatio, 
        int32_t &strideSampleX, 
        int32_t &strideSampleY, 
        int32_t &strideSampleDst);
    int64_t NcolsX() const {
        return m_ncolsX;
    }
    int64_t NrowsX() const {
        return m_nrowsX;
    }
    int64_t NcolsDst() const {
        return m_ncolsDst;
    }
    int64_t NchannelsX() const {
        return m_nchannelsX;
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
    int64_t m_ncolsX;
    int64_t m_nrowsX;
    int64_t m_ncolsDst;
    int64_t m_strideRow;
    int64_t m_strideColY;
    int64_t m_strideColDst;
    int64_t m_strideColId;
    int64_t m_strideRowId;
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
};

MulMatConfig::MulMatConfig():
        m_ncolsX(0),
        m_nrowsX(0),
        m_ncolsDst(0),
        m_strideRow(0),
        m_strideColY(0),
        m_strideColDst(0),
        m_strideColId(0),
        m_strideRowId(0),
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
        m_strideSampleDst(0) { }

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
    int64_t idsStride3 = 0;
    if (haveIds) {
        dnnl::memory::dims idsStrides = m_idsDesc.get_strides();
        idsStride2 = int64_t(idsStrides[2]);
        idsStride3 = int64_t(idsStrides[3]);
    }

    m_ncolsX = bDim3;
    m_nrowsX = bDim2;
    m_ncolsDst = haveIds ? cDim1 : cDim2;
    m_strideRow = bStride2;
    m_strideColY = haveIds ? aStride1 : aStride2;
    m_strideColDst = haveIds ? cStride1 : cStride2;
    m_strideColId = idsStride3;
    m_strideRowId = idsStride2;
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

    if (haveIds && m_nchannelsY == 1) {
        dnnl::memory::dims idsDims = m_idsDesc.get_dims();
        m_strideChannelY = 0;
        m_nchannelsY = int64_t(idsDims[3]);
    }
}

void MulMatConfig::GetArgs(
        int32_t &ncols, 
        int32_t &ncolsDstTotal, 
        int32_t &nchannelsDst, 
        int32_t &strideRow, 
        int32_t &strideColY, 
        int32_t &strideColDst,
        int32_t &strideColId,
        int32_t &strideRowId,
        int32_t &channelRatio, 
        int32_t &strideChannelX, 
        int32_t &strideChannelY, 
        int32_t &strideChannelDst,
        int32_t &sampleRatio, 
        int32_t &strideSampleX, 
        int32_t &strideSampleY, 
        int32_t &strideSampleDst) {
    ncols = int32_t(m_ncolsX); 
    ncolsDstTotal = int32_t(m_ncolsDst); 
    nchannelsDst = int32_t(m_nchannelsDst); 
    strideRow = int32_t(m_strideRow); 
    strideColY = int32_t(m_strideColY); 
    strideColDst = int32_t(m_strideColDst);
    strideColId = int32_t(m_strideColId);
    strideRowId = int32_t(m_strideRowId);
    channelRatio = int32_t(m_nchannelsDst / m_nchannelsX); 
    strideChannelX = int32_t(m_strideChannelX); 
    strideChannelY = int32_t(m_strideChannelY); 
    strideChannelDst = int32_t(m_strideChannelDst);
    sampleRatio = int32_t(m_nsamplesDst / m_nsamplesX); 
    strideSampleX = int32_t(m_strideSampleX); 
    strideSampleY = int32_t(m_strideSampleY); 
    strideSampleDst = int32_t(m_strideSampleDst);
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
    void InitConfig();
    void InitNumSgs();
    void InitTiles();
    void InitSlm();
    void InitIdHelper();
    void InitArgs();
    void InitShapeInfo();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
    int GetSgSize();
    int GetMaxBlockSize();
    int GetRowsPerBlock();
    int GetPadding();
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
    dnnl::memory::desc m_idsADesc;
    dnnl::memory::desc m_idsCDesc;
    dnnl::memory::desc m_expertBoundsDesc;
    base::TempMemory m_idsAMem;
    base::TempMemory m_idsCMem;
    base::TempMemory m_expertBoundsMem;
    std::unique_ptr<MulMatIdHelper> m_idHelper;
    MulMatConfig m_config;
    // config
    int m_minSgSize;
    bool m_mmaTf32;
    bool m_hasIds;
    bool m_largeIds;
    int m_sgSize;
    int m_numSgs;
    int m_rowsPerBlock;
    int m_colsPerBlock;
    int64_t m_nExpertUsed;
    int64_t m_nExperts;
    int64_t m_nTokens;
    int64_t m_numGetRows;
    int m_tileAI;
    int m_tileAJ;
    int m_tileBI;
    int m_tileBJ;
    int m_tileCI;
    int m_tileCJ;
    int m_tileXyCols;
    int m_tileKPadded;
    int m_ntA;
    int m_ntB;
    int m_yStrideScale;
    int m_kiw;
    int m_neSum;
    int m_nbLocal;
    int m_nbSlotMap;
    // args
    int32_t m_ncols; 
    int32_t m_ncolsDstTotal; 
    int32_t m_nchannelsDst; 
    int32_t m_strideRow; 
    int32_t m_strideColY; 
    int32_t m_strideColDst;
    int32_t m_strideColId;
    int32_t m_strideRowId;
    int32_t m_channelRatio; 
    int32_t m_strideChannelX; 
    int32_t m_strideChannelY; 
    int32_t m_strideChannelDst;
    int32_t m_sampleRatio; 
    int32_t m_strideSampleX; 
    int32_t m_strideSampleY; 
    int32_t m_strideSampleDst;
    uint32_t m_sis1;
    uint32_t m_sis1_fd0;
    uint32_t m_sis1_fd1;
    uint32_t m_nch;
    uint32_t m_nch_fd0;
    uint32_t m_nch_fd1;
    // kernel
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
            m_minSgSize(0),
            m_mmaTf32(false),
            m_hasIds(false),
            m_largeIds(false),
            m_sgSize(0),
            m_numSgs(0),
            m_rowsPerBlock(0),
            m_colsPerBlock(0),
            m_nExpertUsed(0),
            m_nExperts(0),
            m_nTokens(0),
            m_numGetRows(0),
            m_tileAI(0),
            m_tileAJ(0),
            m_tileBI(0),
            m_tileBJ(0),
            m_tileCI(0),
            m_tileCJ(0),
            m_tileKPadded(0),
            m_tileXyCols(0),
            m_ntA(0),
            m_ntB(0),
            m_yStrideScale(0),
            m_kiw(0),
            m_neSum(0),
            m_nbLocal(0),
            m_nbSlotMap(0),
            m_ncols(0), 
            m_ncolsDstTotal(0), 
            m_nchannelsDst(0), 
            m_strideRow(0), 
            m_strideColY(0), 
            m_strideColDst(0),
            m_strideColId(0),
            m_strideRowId(0),
            m_channelRatio(0), 
            m_strideChannelX(0), 
            m_strideChannelY(0), 
            m_strideChannelDst(0),
            m_sampleRatio(0), 
            m_strideSampleX(0), 
            m_strideSampleY(0), 
            m_strideSampleDst(0),
            m_sis1(0),
            m_sis1_fd0(0),
            m_sis1_fd1(0),
            m_nch(0),
            m_nch_fd0(0),
            m_nch_fd1(0) { }

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
    // Validate needs cDims
    InferShapes();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    if (ids != nullptr) {
        m_idsMem = ids->Memory();
    }
    SetMemory(m_cDesc);
    if (m_isNop) {
        return true;
    }
    m_context->MemoryPoolStart();
    InitConfig();
    if (m_largeIds) {
        InitIdHelper();
    }
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void MulMatNode::Compute() {
    if (m_isNop) {
        return;
    }

    dnnl::memory idsAMem = m_idsAMem.Get();
    dnnl::memory idsCMem = m_idsCMem.Get();
    dnnl::memory expertBoundsMem = m_expertBoundsMem.Get();

    if (m_largeIds) {
        m_idHelper->Compute(m_idsMem, idsAMem, idsCMem, expertBoundsMem);
    }

    // x = b, y = a
    if (!m_largeIds) {
        m_kernel->SetArgBuffer(0, m_bMem); 
        m_kernel->SetArgBuffer(1, m_aMem);
        m_kernel->SetArgBuffer(2, m_idsMem);
        m_kernel->SetArgBuffer(3, m_memory);
        m_kernel->SetArgS32(4, m_ncols); 
        m_kernel->SetArgS32(5, m_ncolsDstTotal); 
        m_kernel->SetArgS32(6, m_nchannelsDst); 
        m_kernel->SetArgS32(7, m_strideRow); 
        m_kernel->SetArgS32(8, m_strideColY); 
        m_kernel->SetArgS32(9, m_strideColDst);
        m_kernel->SetArgS32(10, m_strideColId); 
        m_kernel->SetArgS32(11, m_strideRowId);
        m_kernel->SetArgS32(12, m_channelRatio); 
        m_kernel->SetArgS32(13, m_strideChannelX); 
        m_kernel->SetArgS32(14, m_strideChannelY); 
        m_kernel->SetArgS32(15, m_strideChannelDst);
        m_kernel->SetArgS32(16, m_sampleRatio); 
        m_kernel->SetArgS32(17, m_strideSampleX); 
        m_kernel->SetArgS32(18, m_strideSampleY); 
        m_kernel->SetArgS32(19, m_strideSampleDst);
        m_shapeInfoArgs.SetArgs(m_kernel.get(), 20);
    } else {
        m_kernel->SetArgBuffer(0, m_bMem); 
        m_kernel->SetArgBuffer(1, m_aMem);
        m_kernel->SetArgBuffer(2, idsAMem); 
        m_kernel->SetArgBuffer(3, idsCMem); 
        m_kernel->SetArgBuffer(4, expertBoundsMem); 
        m_kernel->SetArgBuffer(5, m_memory);
        m_kernel->SetArgS32(6, m_ncols); 
        m_kernel->SetArgS32(7, m_ncolsDstTotal); 
        m_kernel->SetArgS32(8, m_nchannelsDst); 
        m_kernel->SetArgS32(9, m_strideRow); 
        m_kernel->SetArgS32(10, m_strideColY); 
        m_kernel->SetArgS32(11, m_strideColDst);
        m_kernel->SetArgS32(12, m_channelRatio); 
        m_kernel->SetArgS32(13, m_strideChannelX); 
        m_kernel->SetArgS32(14, m_strideChannelY); 
        m_kernel->SetArgS32(15, m_strideChannelDst);
        m_kernel->SetArgS32(16, m_sampleRatio); 
        m_kernel->SetArgS32(17, m_strideSampleX); 
        m_kernel->SetArgS32(18, m_strideSampleY); 
        m_kernel->SetArgS32(19, m_strideSampleDst);
        m_kernel->SetArgU32(20, m_sis1); 
        m_kernel->SetArgU32(21, m_sis1_fd0); 
        m_kernel->SetArgU32(22, m_sis1_fd1); 
        m_kernel->SetArgU32(23, m_nch);
        m_kernel->SetArgU32(24, m_nch_fd0);
        m_kernel->SetArgU32(25, m_nch_fd1);
        m_shapeInfoArgs.SetArgs(m_kernel.get(), 26);
    }

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
#if 0 // TODO: Implement support for f16
    if (bType != dnnl::memory::data_type::f32 &&
            bType != dnnl::memory::data_type::f16) {
        return false;
    }
#endif
    if (bType != dnnl::memory::data_type::f32) {
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

    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    dnnl::memory::dims bStrides = m_bDesc.get_strides();
    dnnl::memory::dims cDims = m_cDesc.get_dims();

    assert(aDims.size() == 4);
    assert(bDims.size() == 4);
    assert(bStrides.size() == 4);
    assert(bDims.size() == 4);

    int sgSize = GetSgSize();

    if (bType == dnnl::memory::data_type::f32) {
        if (bDims[3] % sgSize != 0) {
            return false;
        }
    } else {
        // reserved for f16
        if (bDims[3] % (sgSize * 2) != 0) {
            return false;
        }
    }
    if (bStrides[0] % 2 != 0 || bStrides[1] % 2 != 0 || bStrides[2] % 2 != 0) {
        return false;
    }
    if (bDims[2] % GetRowsPerBlock() != 0) {
        return false;
    }

    if (m_ids == nullptr) {
        if (aDims[2] > 16) {
            return false;
        }
    } else {
        // TODO: Explain this
        if (bDims[2] <= 1024) {
            if (aDims[2] > 512) {
                return false;
            }
        } else {
            if (aDims[2] > 128) {
                return false;
            }
        }
    }

    if (m_ids == nullptr) {
        // nchannelsDst % nchannelsX
        if (cDims[1] % bDims[1] != 0) {
            return false;
        }
    }
    // nsamplesDst % nsamplesX
    if (cDims[0] % bDims[0] != 0) {
        return false;
    }

    if (m_ids != nullptr) {
        if (aDims[1] <= 8) {
            return false;
        }
    } else {
        if (aDims[2] <= 8) {
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
    m_config.Init(m_aDesc, m_bDesc, m_idsDesc, m_cDesc);

    int64_t ncolsDst = m_config.NcolsDst();

    ocl::DeviceInfo *deviceInfo = m_gpuContext->GetDeviceInfo();
    ocl::GpuArch arch = deviceInfo->GetGpuArch();
    // Incorporate minSgSize and mmaTf32 into ocl::DeviceInfo?
    m_minSgSize = (arch >= ocl::GpuArch::Xe2) ? 16 : 8;
    m_mmaTf32 = (arch >= ocl::GpuArch::Xe2);

    m_hasIds = (m_ids != nullptr);
    m_largeIds = (m_hasIds && ncolsDst > 16);
    m_sgSize = GetSgSize();
    InitNumSgs();
    m_rowsPerBlock = GetRowsPerBlock();
    m_colsPerBlock = m_largeIds ? 16 : int(ncolsDst);

    if (m_largeIds) {
        dnnl::memory::dims aDims = m_aDesc.get_dims();
        dnnl::memory::dims bDims = m_bDesc.get_dims();
        dnnl::memory::dims idsDims = m_idsDesc.get_dims();
        m_nExpertUsed = int64_t(idsDims[3]);
        m_nExperts = int64_t(bDims[1]);
        m_nTokens = int64_t(aDims[1]);
        m_numGetRows = m_nTokens * m_nExpertUsed;
    }

    InitTiles();

    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    int padding = GetPadding();

    m_tileXyCols = std::max(m_tileAI, m_tileBI);
    m_tileKPadded = m_sgSize + padding;
    m_ntA = m_rowsPerBlock / m_tileAI;
    m_ntB = ocl::DivUp(m_colsPerBlock, m_tileBI);
    m_yStrideScale = (bType == dnnl::memory::data_type::f32) ? 1 : 2;
    m_kiw = m_numSgs * m_rowsPerBlock + padding;
    m_neSum = m_rowsPerBlock / m_sgSize;

    InitSlm();
}

void MulMatNode::InitNumSgs() {
    int bestNumSgs = 1;
    int64_t ncolsX = m_config.NcolsX();
    int64_t bestNiter = ocl::DivUp(ncolsX, int64_t(m_sgSize * 2));
    int maxNumSgs = GetMaxBlockSize() / m_sgSize;
    for (int numSgs = 2; numSgs < maxNumSgs; numSgs++) {
        int64_t niter = ocl::DivUp(ncolsX, int64_t(numSgs * m_sgSize * 2));
        if (niter < bestNiter) {
            bestNiter = niter;
            bestNumSgs = numSgs;
        }
    }
    m_numSgs = bestNumSgs;
}

void MulMatNode::InitTiles() {
    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    if (bType == dnnl::memory::data_type::f32) {
        // A[8, 8]  Row-major
        // B[16, 8] Col-major
        // C[8, 16] Row-major
        m_tileAI = 8;
        m_tileAJ = 8;
        m_tileBI = 16;
        m_tileBJ = 8;
        m_tileCI = 8;
        m_tileCJ = 16;
    } else {
        assert(false);
    }
}

void MulMatNode::InitSlm() {
    int dwSize = 4;
    int padding = GetPadding();
    int nbIter = m_numSgs * m_tileXyCols * (m_sgSize + padding) * dwSize;
    int nbCombine = 
        ocl::RndUp(m_colsPerBlock, m_tileBI) * (m_numSgs * m_rowsPerBlock + padding) * dwSize;
    int nbLocal = std::max(nbIter, nbCombine);
    m_nbSlotMap = (m_hasIds && !m_largeIds) ? ocl::RndUp(m_colsPerBlock, 16) * dwSize : 0;
    m_nbLocal = nbLocal + m_nbSlotMap;
}

void MulMatNode::InitIdHelper() {
    m_idsADesc = 
        dnnl::memory::desc(
            {m_numGetRows}, 
            dnnl::memory::data_type::s32,
            dnnl::memory::format_tag::a);
    m_idsCDesc = 
        dnnl::memory::desc(
            {m_numGetRows}, 
            dnnl::memory::data_type::s32,
            dnnl::memory::format_tag::a);
    m_expertBoundsDesc = 
        dnnl::memory::desc(
            {m_nExperts + 1},
            dnnl::memory::data_type::s32,
            dnnl::memory::format_tag::a);
    m_idsAMem = m_context->AllocTempMemory(m_idsADesc);
    m_idsCMem = m_context->AllocTempMemory(m_idsCDesc);
    m_expertBoundsMem = m_context->AllocTempMemory(m_expertBoundsDesc);
    m_idHelper = 
        CreateMulMatIdHelper(
            GetGpuContext(),
            m_aDesc,
            m_bDesc,
            m_idsDesc,
            m_idsADesc,
            m_idsCDesc,
            m_expertBoundsDesc);
}

void MulMatNode::InitArgs() {
    m_config.GetArgs(
        m_ncols, 
        m_ncolsDstTotal, 
        m_nchannelsDst, 
        m_strideRow, 
        m_strideColY, 
        m_strideColDst,
        m_strideColId,
        m_strideRowId,
        m_channelRatio, 
        m_strideChannelX, 
        m_strideChannelY, 
        m_strideChannelDst,
        m_sampleRatio, 
        m_strideSampleX, 
        m_strideSampleY, 
        m_strideSampleDst);

    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    int32_t valsPerT = (bType == dnnl::memory::data_type::f32) ? 1 : 2;
    m_ncols /= valsPerT;
    m_strideRow /= valsPerT;
    m_strideColY /= valsPerT;
    m_strideSampleX /= valsPerT;

    if (m_largeIds) {
        dnnl::memory::dims aStrides = m_aDesc.get_strides();
        m_sis1 = uint32_t(aStrides[1] / aStrides[2]);
        ocl::MakeFastDiv(int64_t(m_sis1), m_sis1_fd0, m_sis1_fd1);
    } else {
        m_sis1 = 1;
        m_sis1_fd0 = 0;
        m_sis1_fd1 = 0;
    }

    m_nch = uint32_t(m_nchannelsDst);
    ocl::MakeFastDiv(int64_t(m_nch), m_nch_fd0, m_nch_fd1);

    InitShapeInfo();
}

void MulMatNode::InitShapeInfo() {
    dnnl::memory::dim aBase = m_aDesc.get_submemory_offset();
    dnnl::memory::dim bBase = m_bDesc.get_submemory_offset();
    dnnl::memory::dim cBase = m_cDesc.get_submemory_offset();
    if (!m_largeIds) {
        dnnl::memory::dim idsBase = m_idsDesc.get_submemory_offset();
        m_shapeInfoArgs.AddS64("X_BASE", bBase);
        m_shapeInfoArgs.AddS64("Y_BASE", aBase);
        m_shapeInfoArgs.AddS64("IDS_BASE", idsBase);
        m_shapeInfoArgs.AddS64("DST_BASE", cBase);
    } else {
        dnnl::memory::dim idsABase = m_idsADesc.get_submemory_offset();
        dnnl::memory::dim idsCBase = m_idsCDesc.get_submemory_offset();
        dnnl::memory::dim expertBoundsBase = m_expertBoundsDesc.get_submemory_offset();
        m_shapeInfoArgs.AddS64("X_BASE", bBase);
        m_shapeInfoArgs.AddS64("Y_BASE", aBase);
        m_shapeInfoArgs.AddS64("IDS_A_BASE", idsABase);
        m_shapeInfoArgs.AddS64("IDS_C_BASE", idsCBase);
        m_shapeInfoArgs.AddS64("EXPERT_BOUNDS_BASE", expertBoundsBase);
        m_shapeInfoArgs.AddS64("DST_BASE", cBase);
    }
}

void MulMatNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    // TODO: Encapsulate this logic in dedicated KernelContext method
    ocl::DeviceInfo *info = GetGpuContext()->GetDeviceInfo();
    if (info->GetGpuArch() >= ocl::GpuArch::XeHp) { 
        kernelContext.SetOption("-cl-intel-enable-auto-large-GRF-mode");
    }
    std::string prolog = MakeProlog();
    const char *kernelCode = 
        m_largeIds ?
            kernels::MulMatIdMmKernelCode() :
            kernels::MulMatMmKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "mul_mat_mm", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string MulMatNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*mul_mat_mm");
    sb.Bool(m_hasIds);
    sb.Bool(m_largeIds);
    sb.Int(m_numSgs);
    sb.Int(m_rowsPerBlock);
    sb.Int(m_colsPerBlock);
    return sb.Get();
}

std::string MulMatNode::MakeProlog() {
    std::stringstream ss;

    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    ocl::CommonXe::EmitFastDiv(ss);

    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    if (bType == dnnl::memory::data_type::f32) {
        ss << "#define T float\n";
        EmitInt(ss, "T_IS_FLOAT", 1);
    } else {
        // TODO: f16
        assert(false);
    }
    ss << "\n";

    EmitInt(ss, "MIN_SG_SIZE", m_minSgSize);
    EmitInt(ss, "MMA_TF32", m_mmaTf32);
    EmitInt(ss, "SG_SIZE", m_sgSize);
    EmitInt(ss, "NUM_SGS", m_numSgs);
    EmitInt(ss, "HAS_IDS", m_hasIds ? 1 : 0);
    EmitInt(ss, "ROWS_PER_BLOCK", m_rowsPerBlock);
    EmitInt(ss, "COLS_PER_BLOCK", m_colsPerBlock);
    ss << "\n";

    if (bType == dnnl::memory::data_type::f32) {
        ss << kernels::MulMatMmaF32Code();
    } else {
        // TODO: f16
        assert(false);
    }

    EmitInt(ss, "TILE_A_I", m_tileAI);
    EmitInt(ss, "TILE_A_J", m_tileAJ);
    EmitInt(ss, "TILE_B_I", m_tileBI);
    EmitInt(ss, "TILE_B_J", m_tileBJ);
    EmitInt(ss, "TILE_C_I", m_tileCI);
    EmitInt(ss, "TILE_C_J", m_tileCJ);
    ss << "\n";

    EmitInt(ss, "TILE_XY_COLS", m_tileXyCols);
    EmitInt(ss, "TILE_K_PADDED", m_tileKPadded);
    EmitInt(ss, "NTA", m_ntA);
    EmitInt(ss, "NTB", m_ntB);
    EmitInt(ss, "Y_STRIDE_SCALE", m_yStrideScale);
    EmitInt(ss, "KIW", m_kiw);
    EmitInt(ss, "NE_SUM", m_neSum);
    ss << "\n";

    EmitInt(ss, "NB_LOCAL", m_nbLocal);
    EmitInt(ss, "NB_SLOT_MAP", m_nbSlotMap);
    ss << "\n";
    
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";

    return ss.str();
}

void MulMatNode::InitNdRange() {
    int64_t nrowsX = m_config.NrowsX();
    int64_t ncolsDst = m_config.NcolsDst();
    int64_t nchannelsX = m_config.NchannelsX();
    int64_t nchannelsDst = m_config.NchannelsDst();
    int64_t nsamplesDst = m_config.NsamplesDst();
    size_t lws0 = size_t(m_sgSize);
    size_t lws1 = size_t(m_numSgs);
    size_t gws0, gws1, gws2;
    if (m_largeIds) {
        int64_t maxTiles = ocl::DivUp(ncolsDst, int64_t(m_colsPerBlock));
        gws0 = size_t(nrowsX / m_rowsPerBlock);
        gws1 = size_t(m_nExperts);
        gws2 = size_t(maxTiles);
    } else if (m_hasIds) {
        int64_t colTiles = ocl::DivUp(ncolsDst, int64_t(m_colsPerBlock));
        gws0 = size_t(nrowsX / m_rowsPerBlock);
        gws1 = size_t(nchannelsX * colTiles);
        gws2 = size_t(nsamplesDst);
    } else {
        gws0 = size_t(nrowsX / m_rowsPerBlock);
        gws1 = size_t(nchannelsDst);
        gws2 = size_t(nsamplesDst);
    }
    gws0 *= lws0;
    gws1 *= lws1;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

int MulMatNode::GetSgSize() {
    // Will need 8 for (minSgs = 8 and 16-bit XMX) once implemented
    return 16;
}

int MulMatNode::GetMaxBlockSize() {
    return 256;
}

int MulMatNode::GetRowsPerBlock() {
    return 32;
}

int MulMatNode::GetPadding() {
    return 4;
}

} // namespace

//
//    MulMatMm
//

MulMatMm::MulMatMm(Context *context):
        m_context(context) { }

MulMatMm::~MulMatMm() { }

std::unique_ptr<core::Node> MulMatMm::CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Prec prec) {
    // 'prec' is currently unused
    std::unique_ptr<MulMatNode> node = 
        std::make_unique<MulMatNode>(m_context, a, b, nullptr);
    if (node->Init()) {
        return node;
    }
    return nullptr;
}

//
//    MulMatIdMm
//

MulMatIdMm::MulMatIdMm(Context *context):
        m_context(context) { }

MulMatIdMm::~MulMatIdMm() { }

std::unique_ptr<core::Node> MulMatIdMm::CreateNode(
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

