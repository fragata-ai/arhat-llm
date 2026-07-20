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
#include "arhat/onednn/gpu/mul_mat_id_helper.hpp"
#include "arhat/onednn/gpu/mul_mat.hpp"
#include "arhat/onednn/gpu/mul_mat_id.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    Local constants
//

constexpr int MMQ_ITER_K = 256;

// TODO: Need better explanation

// Decouple shared memory tile sizes from SG_SIZE to allow for different subgroup sizes.
// The K dimension of the tiles has either,
// 1 * MMQ_TILE_NE_K == 32 (always for TILE_Y_K) or 2 * MMQ_TILE_NE_K == 64 (typically for TILE_X_K),
// 32 bit elements for the quantized data (does not include scales).
// In other words, the size of the quantized data in the K dimension is a multiple of MMQ_TILE_NE_K.
// The final tile size in K direction is padded to avoid shared memory bank conflicts,
// in terms of 32 bit elements that means K % 2 == 1 for dp4a or K % 8 == 4 for mma.

constexpr int MMQ_TILE_NE_K = 32;

// block_q8_1_mmq has (128 8-bit ints == 32 32-bit ints + 4 32-bit scales)

constexpr int MMQ_TILE_Y_K = MMQ_TILE_NE_K + MMQ_TILE_NE_K / QuantConst::QI8_1; 

// sizeof(block_q8_1_mmq) / sizeof(int) = 4 + QK8_1
// same value as MMQ_TILE_Y_K

constexpr int NE_BLOCK_Q8_1_MMQ = 4 + base::QK8_1;

//
//    LoadConfig
//

struct LoadConfig {
    int threadsPerRow = 0;
    int nrows = 0;
    int blocksPerTileXRow = 0;
    int rowsPerSg = 0;
};

LoadConfig GetLoadConfig(base::QuantMode quant, int sgSize, bool useMma) {
    using namespace QuantConst;
    int threadsPerRow = 0;
    int nrows = 0;
    int blocksPerTileXRow = 0;
    int rowsPerSg = 0;
    switch (quant) {
    case base::QuantMode::Q4_0:
        threadsPerRow = MMQ_ITER_K / (4 * QR4_0);
        nrows = sgSize / threadsPerRow; 
        blocksPerTileXRow = MMQ_TILE_NE_K / QI4_0;
        rowsPerSg = sgSize / blocksPerTileXRow; 
        break;
    case base::QuantMode::Q4_1:
        threadsPerRow = MMQ_ITER_K / (4 * QR4_1);
        nrows = sgSize / threadsPerRow; 
        blocksPerTileXRow = MMQ_TILE_NE_K / QI4_1;
        rowsPerSg = sgSize / blocksPerTileXRow; 
        break;
    case base::QuantMode::Q5_0:
        threadsPerRow = MMQ_ITER_K / (4 * QR5_0);
        nrows = sgSize / threadsPerRow; 
        blocksPerTileXRow = MMQ_TILE_NE_K / QI5_0;
        rowsPerSg = sgSize / blocksPerTileXRow; 
        break;
    case base::QuantMode::Q5_1:
        threadsPerRow = MMQ_ITER_K / (4 * QR5_1);
        nrows = sgSize / threadsPerRow; 
        blocksPerTileXRow = MMQ_TILE_NE_K / QI5_1;
        rowsPerSg = sgSize / blocksPerTileXRow; 
        break;
    case base::QuantMode::Q8_0:
        threadsPerRow = 32;
        nrows = sgSize / threadsPerRow; 
        blocksPerTileXRow = 2 * MMQ_TILE_NE_K / QI8_0;
        rowsPerSg = sgSize / blocksPerTileXRow; 
        break;
    case base::QuantMode::Q2_K:
        threadsPerRow = MMQ_ITER_K / (4 * QR2_K);
        nrows = sgSize / threadsPerRow; 
        break;
    case base::QuantMode::Q3_K:
        threadsPerRow = MMQ_ITER_K / (4 * QR3_K);
        nrows = sgSize / threadsPerRow; 
        rowsPerSg = sgSize / 4; 
        break;
    case base::QuantMode::Q4_K:
        threadsPerRow = MMQ_ITER_K / (4 * QR4_K);
        nrows = sgSize / threadsPerRow; 
        if (useMma) {
            rowsPerSg = sgSize / 2; 
        } else {
            rowsPerSg = sgSize / 4; 
        }
        break;
    case base::QuantMode::Q5_K:
        threadsPerRow = MMQ_ITER_K / (4 * QR5_K);
        nrows = sgSize / threadsPerRow; 
        if (useMma) {
            rowsPerSg = sgSize / 2; 
        } else {
            rowsPerSg = sgSize / 4; 
        }
        break;
    case base::QuantMode::Q6_K:
        threadsPerRow = MMQ_ITER_K / (4 * QR6_K);
        nrows = sgSize / threadsPerRow; 
        rowsPerSg = sgSize / 4; 
        break;
    case base::QuantMode::MXFP4:
        threadsPerRow = MMQ_ITER_K / (4 * QR_MXFP4);
        nrows = sgSize / threadsPerRow; 
        blocksPerTileXRow = MMQ_TILE_NE_K / QI_MXFP4;
        rowsPerSg = sgSize / blocksPerTileXRow; 
        break;
    default:
        assert(false);
        break;
    }
    return {
        threadsPerRow,
        nrows,
        blocksPerTileXRow,
        rowsPerSg
    };
}

//
//    MulMatTraits
//

// 'vecDotImpl' is variant used by 'vecDotDp4'
// 'vdr' shall match 'vecDotImpl' variant
// ('vecDotMma' does not use 'vecDotImpl')

struct MulMatTraits {
    const char *(*loadTilesCode)() = nullptr;
    const char *(*vecDotDefsCode)() = nullptr;
    const char *(*vecDotImplCode)() = nullptr;
    const char *(*vecDotMmaCode)() = nullptr;
    const char *(*vecDotDp4aCode)() = nullptr;
    std::string vdr; 
    std::string loadTiles;
    std::string vecDotMma;
    std::string vecDotDp4a;
};

MulMatTraits GetMulMatTraits(base::QuantMode quant) {
    switch (quant) {
    case base::QuantMode::Q4_0:
        return {
            kernels::MulMatQuantMmLoadTiles_Q4_0_Code,
            kernels::VecDotDefs_Q4_0_Code, 
            kernels::VecDotImpl_Q4_0_Code, 
            nullptr, // kernels::VecDotMma_Q8_0_Code
            kernels::VecDotDp4a_Q4_0_Code, 
            "VDR_Q4_0_Q8_1_MMQ",
            "load_tiles_q4_0",
            "vec_dot_q8_0_q8_1_mma",
            "vec_dot_q4_0_q8_1_dp4a"
        };
    case base::QuantMode::Q4_1:
        return {
            kernels::MulMatQuantMmLoadTiles_Q4_1_Code,
            kernels::VecDotDefs_Q4_1_Code, 
            kernels::VecDotImpl_Q4_1_Code, 
            nullptr, // kernels::VecDotMma_Q8_1_Code
            kernels::VecDotDp4a_Q4_1_Code, 
            "VDR_Q4_1_Q8_1_MMQ",
            "load_tiles_q4_1",
            "vec_dot_q8_1_q8_1_mma",
            "vec_dot_q4_1_q8_1_dp4a"
        };
    case base::QuantMode::Q5_0:
        return {
            kernels::MulMatQuantMmLoadTiles_Q5_0_Code,
            kernels::VecDotDefs_Q5_0_Code, 
            kernels::VecDotImpl_Q8_0_Code, 
            nullptr, // kernels::VecDotMma_Q8_0_Code
            kernels::VecDotDp4a_Q8_0_Code, 
            "VDR_Q8_0_Q8_1_MMQ",
            "load_tiles_q5_0",
            "vec_dot_q8_0_q8_1_mma",
            "vec_dot_q8_0_q8_1_dp4a"
        };
    case base::QuantMode::Q5_1:
        return {
            kernels::MulMatQuantMmLoadTiles_Q5_1_Code,
            kernels::VecDotDefs_Q5_1_Code, 
            kernels::VecDotImpl_Q8_1_Code, 
            nullptr, // kernels::VecDotMma_Q8_1_Code
            kernels::VecDotDp4a_Q8_1_Code, 
            "VDR_Q8_0_Q8_1_MMQ", // = (QR5_1 * VDR_Q5_1_Q8_1_MMQ)
            "load_tiles_q5_1",
            "vec_dot_q8_1_q8_1_mma",
            "vec_dot_q8_1_q8_1_dp4a"
        };
    case base::QuantMode::Q8_0:
        return {
            kernels::MulMatQuantMmLoadTiles_Q8_0_Code,
            kernels::VecDotDefs_Q8_0_Code, 
            kernels::VecDotImpl_Q8_0_Code, 
            nullptr, // kernels::VecDotMma_Q8_0_Code
            kernels::VecDotDp4a_Q8_0_Code, 
            "VDR_Q8_0_Q8_1_MMQ",
            "load_tiles_q8_0",
            "vec_dot_q8_0_q8_1_mma",
            "vec_dot_q8_0_q8_1_dp4a"
        };
    case base::QuantMode::Q2_K:
        return {
            kernels::MulMatQuantMmLoadTiles_Q2_K_Code,
            kernels::VecDotDefs_Q2_K_Code, 
            kernels::VecDotMmImpl_Q2_K_Code, 
            nullptr, // kernels::VecDotMma_Q2_K_Code
            kernels::VecDotDp4a_Q2_K_Code, 
            "VDR_Q2_K_Q8_1_MMQ",
            "load_tiles_q2_K",
            "vec_dot_q2_K_q8_1_mma",
            "vec_dot_q2_K_q8_1_dp4a"
        };
    case base::QuantMode::Q3_K:
        return {
            kernels::MulMatQuantMmLoadTiles_Q3_K_Code,
            kernels::VecDotDefs_Q3_K_Code, 
            kernels::VecDotMmImpl_Q3_K_Code, 
            nullptr, // kernels::VecDotMma_Q3_K_Code
            kernels::VecDotDp4a_Q3_K_Code, 
            "VDR_Q3_K_Q8_1_MMQ",
            "load_tiles_q3_K",
            "vec_dot_q8_0_16_q8_1_mma",
            "vec_dot_q3_K_q8_1_dp4a"
        };
    case base::QuantMode::Q4_K:
        return {
            kernels::MulMatQuantMmLoadTiles_Q4_K_Code,
            kernels::VecDotDefs_Q4_K_Code, 
            kernels::VecDotMmImpl_Q4_K_Code, 
            nullptr, // kernels::VecDotMm_Q4_K_Code
            kernels::VecDotDp4a_Q4_K_Code, 
            "VDR_Q4_K_Q8_1_MMQ",
            "load_tiles_q4_K",
            "vec_dot_q8_1_q8_1_mma",
            "vec_dot_q4_K_q8_1_dp4a"
        };
    case base::QuantMode::Q5_K:
        return {
            kernels::MulMatQuantMmLoadTiles_Q5_K_Code,
            kernels::VecDotDefs_Q5_K_Code, 
            kernels::VecDotMmImpl_Q5_K_Code, 
            nullptr, // kernels::VecDotMma_Q5_K_Code
            kernels::VecDotDp4a_Q5_K_Code, 
            "VDR_Q5_K_Q8_1_MMQ",
            "load_tiles_q5_K",
            "vec_dot_q8_1_q8_1_mma",
            "vec_dot_q5_K_q8_1_dp4a"
        };
    case base::QuantMode::Q6_K:
        return {
            kernels::MulMatQuantMmLoadTiles_Q6_K_Code,
            kernels::VecDotDefs_Q6_K_Code, 
            kernels::VecDotMmImpl_Q6_K_Code, 
            nullptr, // kernels::VecDotMma_Q6_K_Code
            kernels::VecDotDp4a_Q6_K_Code, 
            "VDR_Q6_K_Q8_1_MMQ",
            "load_tiles_q6_K",
            "vec_dot_q6_K_q8_1_mma",
            "vec_dot_q6_K_q8_1_dp4a"
        };
    case base::QuantMode::MXFP4:
        return {
            kernels::MulMatQuantMmLoadTiles_Mxfp4_Code,
            kernels::VecDotDefs_Mxfp4_Code, 
            kernels::VecDotImpl_Q8_0_Code, 
            nullptr, // kernels::VecDotMma_Q8_0_Code
            kernels::VecDotDp4a_Q8_0_Code, 
            "VDR_Q8_0_Q8_1_MMQ",
            "load_tiles_mxfp4",
            "vec_dot_q8_0_q8_1_mma",
            "vec_dot_q8_0_q8_1_dp4a"
        };
    default:
        assert(false);
        return {};
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
        const dnnl::memory::desc &aTempDesc);
    void GetArgs(
        int32_t &ncolsX, 
        int32_t &nrowsX, 
        int32_t &ncolsDst, 
        int32_t &strideRowX, 
        int32_t &ncolsY, 
        int32_t &strideColDst,
        int32_t &channelRatio, 
        int32_t &nchannelsY, 
        int32_t &strideChannelX, 
        int32_t &strideChannelY, 
        int32_t &strideChannelDst,
        int32_t &sampleRatio, 
        int32_t &nsamplesY, 
        int32_t &strideSampleX, 
        int32_t &strideSampleY, 
        int32_t &strideSampleDst,
        int32_t &ncolsMax);
    int64_t NrowsX() const {
        return m_nrowsX;
    }
    int64_t NcolsMax() const {
        return m_ncolsMax;
    }
private:
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_idsDesc;
    dnnl::memory::desc m_cDesc;
    dnnl::memory::dims m_bDims;
    base::QuantMode m_bQuant;
    base::QuantMode m_aTempQuant;
    dnnl::memory::desc m_aTempDesc;
    int64_t m_ncolsX; 
    int64_t m_nrowsX; 
    int64_t m_ncolsDst; 
    int64_t m_strideRowX; 
    int64_t m_ncolsY; 
    int64_t m_nrowsDst;
    int64_t m_nchannelsX; 
    int64_t m_nchannelsY; 
    int64_t m_strideChannelX; 
    int64_t m_strideChannelY; 
    int64_t m_strideChannelDst;
    int64_t m_nsamplesX; 
    int64_t m_nsamplesY; 
    int64_t m_strideSampleX; 
    int64_t m_strideSampleY; 
    int64_t m_strideSampleDst;
    int64_t m_ncolsMax;
};

MulMatConfig::MulMatConfig():
        m_bQuant(base::QuantMode::None),
        m_aTempQuant(base::QuantMode::None),
        m_ncolsX(0), 
        m_nrowsX(0), 
        m_ncolsDst(0), 
        m_strideRowX(0), 
        m_ncolsY(0), 
        m_nrowsDst(0),
        m_nchannelsX(0), 
        m_nchannelsY(0), 
        m_strideChannelX(0), 
        m_strideChannelY(0), 
        m_strideChannelDst(0),
        m_nsamplesX(0), 
        m_nsamplesY(0), 
        m_strideSampleX(0), 
        m_strideSampleY(0), 
        m_strideSampleDst(0),
        m_ncolsMax(0) { }

MulMatConfig::~MulMatConfig() { }

void MulMatConfig::Init(
        const dnnl::memory::desc &aDesc,
        const dnnl::memory::desc &bDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &cDesc,
        const dnnl::memory::dims &bDims,
        base::QuantMode bQuant,
        base::QuantMode aTempQuant,
        const dnnl::memory::desc &aTempDesc) {
    m_aDesc = aDesc;
    m_bDesc = bDesc;
    m_idsDesc = idsDesc;
    m_cDesc = cDesc;
    m_bDims = bDims;
    m_bQuant = bQuant;
    m_aTempQuant = aTempQuant;
    m_aTempDesc = aTempDesc;

    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims aTempStrides = m_aTempDesc.get_strides();
    dnnl::memory::dims bStrides = m_bDesc.get_strides();
    dnnl::memory::dims cDims = m_cDesc.get_dims();
    dnnl::memory::dims cStrides = m_cDesc.get_strides();
    int bBlockSize = base::GetBlockSize(m_bQuant);
    bool haveIds = !m_idsDesc.is_zero();

    int64_t aDim0 = int64_t(aDims[0]);
    int64_t aDim1 = int64_t(aDims[1]);
    int64_t aDim2 = int64_t(aDims[2]);

    // raw (byte) strides converted to uint32 units (as required by kernel)
    int64_t aTempStride0 = int64_t(aTempStrides[0]) / sizeof(int32_t);
    int64_t aTempStride1 = int64_t(aTempStrides[1]) / sizeof(int32_t);

    int64_t bDim0 = int64_t(m_bDims[0]);
    int64_t bDim1 = int64_t(m_bDims[1]);
    int64_t bDim2 = int64_t(m_bDims[2]);
    int64_t bDim3 = int64_t(m_bDims[3]);

    // raw (byte) strides converted to quant block units (as required by kernel)
    int64_t bStride0 = int64_t(bStrides[0]) / bBlockSize;
    int64_t bStride1 = int64_t(bStrides[1]) / bBlockSize;
    int64_t bStride2 = int64_t(bStrides[2]) / bBlockSize;

    int64_t cDim2 = int64_t(cDims[2]);

    int64_t cStride0 = int64_t(cStrides[0]);
    int64_t cStride1 = int64_t(cStrides[1]);
    int64_t cStride2 = int64_t(cStrides[2]);

    if (!haveIds) {
        m_ncolsX = bDim3; 
        m_nrowsX = bDim2; 
        m_ncolsDst = cDim2; 
        m_strideRowX = bStride2; 
        m_ncolsY = aDim2; 
        m_nrowsDst = cStride2;
        m_nchannelsX = bDim1; 
        m_nchannelsY = aDim1; 
        m_strideChannelX = bStride1; 
        m_strideChannelY = aTempStride1; 
        m_strideChannelDst = cStride1;
        m_nsamplesX = bDim0; 
        m_nsamplesY = aDim0; 
        m_strideSampleX = bStride0; 
        m_strideSampleY = aTempStride0; 
        m_strideSampleDst = cStride0;
        m_ncolsMax = cDim2;

    } else {
        dnnl::memory::dims idsDims = m_idsDesc.get_dims();
        int64_t nExpertUsed = int64_t(idsDims[3]);
        int64_t numGetRows = aDim1 * nExpertUsed;
        // This patch computes strides for non-flattened aTemp shape
        // No need for it so far - apparently works fine with flattened aTemp shape too
#if 0 // TODO: Revise this
        aTempStride1 = aDim2 * int32_t(aTempStrides[2]) / sizeof(int32_t);
        aTempStride0 = aTempStride1 * aDim1;
#endif

        m_ncolsX = bDim3; 
        m_nrowsX = bDim2; 
        m_ncolsDst = numGetRows; 
        m_strideRowX = bStride2; 
        m_ncolsY = numGetRows; 
        m_nrowsDst = cStride2;
        m_nchannelsX = bDim1; 
        m_nchannelsY = bDim1; // yes, same as nchannelsX
        m_strideChannelX = bStride1; 
        m_strideChannelY = aTempStride1; 
        m_strideChannelDst = cStride1;
        m_nsamplesX = bDim0; 
        m_nsamplesY = aDim0; 
        m_strideSampleX = bStride0; 
        m_strideSampleY = aTempStride0; 
        m_strideSampleDst = cStride0;
        m_ncolsMax = aDim1;
    }
}

void MulMatConfig::GetArgs(
        int32_t &ncolsX, 
        int32_t &nrowsX, 
        int32_t &ncolsDst, 
        int32_t &strideRowX, 
        int32_t &ncolsY, 
        int32_t &strideColDst,
        int32_t &channelRatio, 
        int32_t &nchannelsY, 
        int32_t &strideChannelX, 
        int32_t &strideChannelY, 
        int32_t &strideChannelDst,
        int32_t &sampleRatio, 
        int32_t &nsamplesY, 
        int32_t &strideSampleX, 
        int32_t &strideSampleY, 
        int32_t &strideSampleDst,
        int32_t &ncolsMax) {
    assert(m_nchannelsY % m_nchannelsX == 0); 
    assert(m_nsamplesY % m_nsamplesX == 0); 

    ncolsX = int32_t(m_ncolsX); 
    nrowsX = int32_t(m_nrowsX); 
    ncolsDst = int32_t(m_ncolsDst); 
    strideRowX = int32_t(m_strideRowX); 
    ncolsY = int32_t(m_ncolsY); 
    strideColDst = int32_t(m_nrowsDst);
    channelRatio = int32_t(m_nchannelsY / m_nchannelsX); 
    nchannelsY = int32_t(m_nchannelsY); 
    strideChannelX = int32_t(m_strideChannelX); 
    strideChannelY = int32_t(m_strideChannelY); 
    strideChannelDst = int32_t(m_strideChannelDst);
    sampleRatio = int32_t(m_nsamplesY / m_nsamplesX); 
    nsamplesY = int32_t(m_nsamplesY); 
    strideSampleX = int32_t(m_strideSampleX); 
    strideSampleY = int32_t(m_strideSampleY); 
    strideSampleDst = int32_t(m_strideSampleDst);
    ncolsMax = int32_t(m_ncolsMax);
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
    void InitIdHelper();
    void InitQuantize();
    void InitConfig();
    void InitArgs();
    void InitShapeInfo();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
    int GetMmqXMax();
    int GetMmqX();
    int GetMmqY();
    int GetIterK();
    int GetGranularity(int mmqX);
    int GetLocalItems(int mmqX, int mmqY);
    void GetLocalSplit(
        int mmqX, 
        int mmqY,
        int &nbsIds,
        int &nbsX,
        int &nbsY);
    void GetDp4aTxs(
        int mmqY, 
        int &qs, 
        int &dm, 
        int &sc);
    int GetMmaTileXK();
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
    std::unique_ptr<QuantizeVecMm> m_quantize;
    dnnl::memory::desc m_idsADesc;
    dnnl::memory::desc m_idsCDesc;
    dnnl::memory::desc m_expertBoundsDesc;
    dnnl::memory::desc m_tmpFixupDesc;
    base::TempMemory m_idsAMem;
    base::TempMemory m_idsCMem;
    base::TempMemory m_expertBoundsMem;
    base::TempMemory m_tmpFixupMem;
    std::unique_ptr<MulMatIdHelper> m_idHelper;
    MulMatConfig m_config;
    bool m_useMma;
    bool m_useStreamK;
    int m_sgSize;
    int m_numSgs;
    int m_mmqX;
    int m_mmqY;
    bool m_needCheck;
    int m_nbsIds;
    int m_nbsX;
    int m_nbsY;
    int m_txsQs; 
    int m_txsDm;
    int m_txsSc;
    int m_neBlock;
    int m_iterK;
    int m_blocksPerIter;
    QuantTraits m_quantTraits;
    LoadConfig m_loadConfig;
    MulMatTraits m_traits;
    int32_t m_ncolsX; 
    int32_t m_nrowsX; 
    int32_t m_ncolsDst; 
    int32_t m_strideRowX; 
    int32_t m_ncolsY; 
    int32_t m_strideColDst;
    int32_t m_channelRatio; 
    int32_t m_nchannelsY; 
    int32_t m_strideChannelX; 
    int32_t m_strideChannelY; 
    int32_t m_strideChannelDst;
    int32_t m_sampleRatio; 
    int32_t m_nsamplesY; 
    int32_t m_strideSampleX; 
    int32_t m_strideSampleY; 
    int32_t m_strideSampleDst;
    int32_t m_ncolsMax;
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
            m_useMma(false),
            m_useStreamK(false),
            m_sgSize(0),
            m_numSgs(0),
            m_mmqX(0),
            m_mmqY(0),
            m_needCheck(false),
            m_nbsIds(0),
            m_nbsX(0),
            m_nbsY(0),
            m_txsQs(0),
            m_txsDm(0),
            m_txsSc(0),
            m_neBlock(0),
            m_iterK(0),
            m_blocksPerIter(0),
            m_ncolsX(0), 
            m_nrowsX(0), 
            m_ncolsDst(0), 
            m_strideRowX(0), 
            m_ncolsY(0), 
            m_strideColDst(0),
            m_channelRatio(0), 
            m_nchannelsY(0), 
            m_strideChannelX(0), 
            m_strideChannelY(0), 
            m_strideChannelDst(0),
            m_sampleRatio(0), 
            m_nsamplesY(0), 
            m_strideSampleX(0), 
            m_strideSampleY(0), 
            m_strideSampleDst(0),
            m_ncolsMax(0) { }

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
    if (m_ids != nullptr) {
        InitIdHelper();
    }
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

    dnnl::memory aTempMem = m_aTempMem.Get(); 
    dnnl::memory idsAMem = m_idsAMem.Get();
    dnnl::memory idsCMem = m_idsCMem.Get();
    dnnl::memory expertBoundsMem = m_expertBoundsMem.Get();
    dnnl::memory tmpFixupMem = m_tmpFixupMem.Get();

    if (m_ids != nullptr) {
        m_idHelper->Compute(m_idsMem, idsAMem, idsCMem, expertBoundsMem);
    }

    m_quantize->Compute(m_aMem, idsAMem, aTempMem);

    // vx = b, vy = a
    m_kernel->SetArgBuffer(0, m_bMem);
    m_kernel->SetArgBuffer(1, aTempMem);
    m_kernel->SetArgBuffer(2, idsCMem);
    m_kernel->SetArgBuffer(3, expertBoundsMem);
    m_kernel->SetArgBuffer(4, m_memory);
    m_kernel->SetArgBuffer(5, tmpFixupMem);
    m_kernel->SetArgS32(6, m_ncolsX); 
    m_kernel->SetArgS32(7, m_nrowsX); 
    m_kernel->SetArgS32(8, m_ncolsDst); 
    m_kernel->SetArgS32(9, m_strideRowX); 
    m_kernel->SetArgS32(10, m_ncolsY); 
    m_kernel->SetArgS32(11, m_strideColDst);
    m_kernel->SetArgS32(12, m_channelRatio); 
    m_kernel->SetArgS32(13, m_nchannelsY); 
    m_kernel->SetArgS32(14, m_strideChannelX); 
    m_kernel->SetArgS32(15, m_strideChannelY); 
    m_kernel->SetArgS32(16, m_strideChannelDst);
    m_kernel->SetArgS32(17, m_sampleRatio); 
    m_kernel->SetArgS32(18, m_nsamplesY); 
    m_kernel->SetArgS32(19, m_strideSampleX); 
    m_kernel->SetArgS32(20, m_strideSampleY); 
    m_kernel->SetArgS32(21, m_strideSampleDst);
    m_kernel->SetArgS32(22, m_ncolsMax);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 23);
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
    if (ncolsDst <= MMVQ_MAX_BATCH_SIZE) {
        return false;
    }
#if 0 // TODO
    dnnl::memory::dim ncolsX = m_bDims[3]; 
    if (ncolsX % base::GetQuantSize(m_bQuant) != 0) {
        return false;
    }
    if (aDims[3] % QK8_1 != 0) {
        return false;
    }
#endif
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

void MulMatNode::InitIdHelper() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    dnnl::memory::dims idsDims = m_idsDesc.get_dims();
    dnnl::memory::dim nExpertUsed = idsDims[3];
    dnnl::memory::dim numGetRows = aDims[1] * nExpertUsed;
    m_idsADesc = 
        dnnl::memory::desc(
            {numGetRows}, 
            dnnl::memory::data_type::s32,
            dnnl::memory::format_tag::a);
    m_idsCDesc = 
        dnnl::memory::desc(
            {numGetRows}, 
            dnnl::memory::data_type::s32,
            dnnl::memory::format_tag::a);
    m_expertBoundsDesc = 
        dnnl::memory::desc(
            {bDims[1] + 1},
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

void MulMatNode::InitQuantize() {
    // TODO: Implement pool allocation for temporary memory
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    m_aTempDim3 = (aDims[3] + MATRIX_ROW_PADDING - 1) & ~(MATRIX_ROW_PADDING - 1);
    m_aTempQuant = base::QuantMode::Q8_1;
    int blockSize = base::GetBlockSize(m_aTempQuant);
    int quantSize = base::GetQuantSize(m_aTempQuant);
    assert(m_aTempDim3 % quantSize == 0);
    aDims[3] = (m_aTempDim3 / quantSize) * blockSize;
    if (m_ids != nullptr) {
        // Flatten A dims [1, 2] into [2] because [2] serves as index in IDS during quantization
        // (see QuantizeMm_Q8_1 kernel code)
        dnnl::memory::dims idsDims = m_idsDesc.get_dims();
        dnnl::memory::dim nExpertUsed = idsDims[3];
        dnnl::memory::dim numGetRows = aDims[1] * nExpertUsed;
        aDims[0] = 1;
        aDims[1] = 1;
        aDims[2] = numGetRows;
        // keep aDims[3]
    }
    m_aTempDesc = 
        dnnl::memory::desc(
            aDims, 
            dnnl::memory::data_type::u8, 
            dnnl::memory::format_tag::abcd);
    // provide margin to hold all tiles
    int margin = GetMmqXMax() * NE_BLOCK_Q8_1_MMQ * sizeof(int32_t);
    dnnl::memory::dim fullVolume = aDims[0] * aDims[1] * aDims[2] * aDims[3] + margin;
    dnnl::memory::desc fullDesc(
        {fullVolume}, 
        dnnl::memory::data_type::u8, 
        dnnl::memory::format_tag::a);
    // fullDesc is used for memory allocation only - compute kernels use aTempDesc
    m_aTempMem = m_context->AllocTempMemory(fullDesc);
    m_quantize = 
        CreateQuantizeMm_Q8_1(
            GetGpuContext(),
            m_aDesc,
            m_idsADesc,
            m_aTempDesc,
            m_bQuant);
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
        m_aTempDesc);
    // MMA variant is not yet implemented
    // stream-k is not yet implemented
    m_useMma = false;
    m_useStreamK = false;
    m_sgSize = 32;
    m_numSgs = 8;
    m_mmqX = GetMmqX();
    m_mmqY = GetMmqY();
    m_needCheck = (m_config.NrowsX() % m_mmqY != 0);
    GetLocalSplit(m_mmqX, m_mmqY, m_nbsIds, m_nbsX, m_nbsY);
    GetDp4aTxs(m_mmqY, m_txsQs, m_txsDm, m_txsSc);
    m_neBlock = 4 * base::QK8_1;
    m_quantTraits = GetQuantTraits(m_bQuant);
    m_iterK = GetIterK();
    m_blocksPerIter = m_iterK / m_quantTraits.qk;
    m_loadConfig = GetLoadConfig(m_bQuant, m_sgSize, m_useMma);
    m_traits = GetMulMatTraits(m_bQuant);
}

void MulMatNode::InitArgs() {
    m_config.GetArgs(
        m_ncolsX, 
        m_nrowsX, 
        m_ncolsDst, 
        m_strideRowX, 
        m_ncolsY, 
        m_strideColDst,
        m_channelRatio, 
        m_nchannelsY, 
        m_strideChannelX, 
        m_strideChannelY, 
        m_strideChannelDst,
        m_sampleRatio, 
        m_nsamplesY, 
        m_strideSampleX, 
        m_strideSampleY, 
        m_strideSampleDst,
        m_ncolsMax);
    InitShapeInfo();
}

void MulMatNode::InitShapeInfo() {
    size_t aBase = m_aDesc.get_submemory_offset();
    size_t bBase = m_bDesc.get_submemory_offset();
    size_t idsBase = m_idsDesc.get_submemory_offset();
    size_t expertBoundsBase = m_expertBoundsDesc.get_submemory_offset();
    size_t cBase = m_cDesc.get_submemory_offset();
    size_t tmpFixupBase = m_tmpFixupDesc.get_submemory_offset();

    // x = b, y = a
    m_shapeInfoArgs.AddS64("X_BASE", bBase);
    m_shapeInfoArgs.AddS64("Y_BASE", aBase / sizeof(int32_t));
    m_shapeInfoArgs.AddS64("IDS_DST_BASE", idsBase);
    m_shapeInfoArgs.AddS64("EXPERT_BOUNDS_BASE", expertBoundsBase);
    m_shapeInfoArgs.AddS64("DST_BASE", cBase);
    m_shapeInfoArgs.AddS64("TMP_FIXUP_BASE", tmpFixupBase);
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
    const char *kernelCode = kernels::MulMatQuantMmKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "mul_mat_mm_q", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string MulMatNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*mul_mat_mm_q");
    sb.Int(int64_t(m_bQuant));
    sb.Int(int64_t(m_mmqX));
    sb.Bool(m_needCheck);
    return sb.Get();
}

std::string MulMatNode::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    if (!m_useMma) {
        ocl::CommonXe::EmitImad(ss);
    }
    EmitInt(ss, "SG_SIZE", m_sgSize);
    EmitInt(ss, "NUM_SGS", m_numSgs);
    EmitInt(ss, "MMQ_X", m_mmqX);
    EmitInt(ss, "MMQ_Y", m_mmqY);
    EmitInt(ss, "NEED_CHECK", m_needCheck ? 1 : 0);
    if (!m_useMma) {
        EmitInt(ss, "TXS_QS", m_txsQs);
        EmitInt(ss, "TXS_DM", m_txsDm);
        EmitInt(ss, "TXS_SC", m_txsSc);
    }
    EmitInt(ss, "MMQ_TILE_NE_K", MMQ_TILE_NE_K);
    EmitInt(ss, "MMQ_TILE_Y_K", MMQ_TILE_Y_K);
    EmitInt(ss, "THREADS_PER_ROW", m_loadConfig.threadsPerRow);
    EmitInt(ss, "NROWS", m_loadConfig.nrows);
    EmitInt(ss, "BLOCKS_PER_TILE_X_ROW", m_loadConfig.blocksPerTileXRow);
    EmitInt(ss, "ROWS_PER_SG", m_loadConfig.rowsPerSg);
    ss << "#define VDR " << m_traits.vdr << "\n";
    ss << "\n";
    ss << kernels::VecDotQuantCommonCode();
    ss << m_traits.vecDotDefsCode();
    if (m_bQuant == base::QuantMode::Q5_0 || 
            m_bQuant == base::QuantMode::Q5_1 ||
            m_bQuant == base::QuantMode::MXFP4) {
        ss << kernels::VecDotDefs_Q8_0_Code();
    }
    if (m_bQuant == base::QuantMode::Q4_K || m_bQuant == base::QuantMode::Q5_K) {
        ss << kernels::MulMatQuantMmUnpackScales_Q45_K();
    }
    ss << m_traits.loadTilesCode();
    if (m_traits.vecDotImplCode != nullptr) {
        ss << m_traits.vecDotImplCode();
    }
    if (m_useMma) {
        ss << m_traits.vecDotMmaCode();
        // ss << kernels::MulMatQuantMmWriteBackMmaCode()
    } else {
        ss << m_traits.vecDotDp4aCode();
        ss << kernels::MulMatQuantMmWriteBackDp4aCode();
    }
    EmitInt(ss, "NE_LOCAL_IDS", m_nbsIds);
    EmitInt(ss, "NE_LOCAL_Y", m_nbsY);
    EmitInt(ss, "NE_LOCAL", int64_t(m_nbsIds + m_nbsY + m_nbsX));
    EmitInt(ss, "QK", m_quantTraits.qk);
    EmitInt(ss, "NE_BLOCK", m_neBlock);
    EmitInt(ss, "NE_BLOCK_Q8_1_MMQ", NE_BLOCK_Q8_1_MMQ);
    EmitInt(ss, "ITER_K", m_iterK);
    EmitInt(ss, "BLOCKS_PER_ITER", m_blocksPerIter);
    EmitInt(ss, "FIXUP", 0); // Reserved
    ss << "\n";
    ss << "#define LOAD_TILES " << m_traits.loadTiles << "\n";
    if (m_useMma) {
        ss << "#define VEC_DOT " << m_traits.vecDotMma << "\n";
        ss << "#define WRITE_BACK mmq_write_back_mma\n";
    } else {
        ss << "#define VEC_DOT " << m_traits.vecDotDp4a << "\n";
        ss << "#define WRITE_BACK mmq_write_back_dp4a\n";
    }
    ss << "\n";
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    return ss.str();
}

void MulMatNode::InitNdRange() {
    size_t nty = ocl::DivUp(size_t(m_nrowsX), size_t(m_mmqY));
    size_t ntx = ocl::DivUp(size_t(m_ncolsMax), size_t(m_mmqX));
    size_t ntzw = size_t(m_nchannelsY) * size_t(m_nsamplesY);
    size_t lws0 = size_t(m_sgSize);
    size_t lws1 = size_t(m_numSgs);
    size_t gws0 = nty * lws0;
    size_t gws1 = ntx * lws1;
    size_t gws2 = ntzw;
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

int MulMatNode::GetMmqXMax() {
    // ACHTUNG: May need further architecture-specific adjustments
    return 64;
}

int MulMatNode::GetMmqX() {
    ocl::OclDeviceInfo info = GetOclContext()->GetDeviceInfo();
    int maxSlmItemsPerWg = int(info.maxSlmBytesPerWg / sizeof(int32_t));
    int mmqXMax = GetMmqXMax();
    int mmqY = GetMmqY();

    int mmqXBest  = 0;
    int ntilesXBest = INT_MAX;

    // m_ncolsMax is not yet available at this point
    int ncolsMax = int(m_config.NcolsMax());

    for (int mmqX = 8; mmqX <= mmqXMax && ntilesXBest > 1; mmqX += 8) {
        int granularity = GetGranularity(mmqX);

        if (mmqX % granularity != 0 || GetLocalItems(mmqX, mmqY) > maxSlmItemsPerWg) {
            continue;
        }

        int ntilesX = ocl::DivUp(ncolsMax, mmqX);

        if (ntilesX < ntilesXBest) {
            mmqXBest = mmqX;
            ntilesXBest = ntilesX;
        }
    } 

    return mmqXBest;
}

int MulMatNode::GetMmqY() {
    // ACHTUNG: May need further architecture-specific adjustments
    return 64;
}

int MulMatNode::GetIterK() {
    return MMQ_ITER_K;
}

int MulMatNode::GetGranularity(int mmqX) {
    // ACHTUNG: May need further architecture-specific adjustments
    return (mmqX >= 48) ? 16 : 8;
}

int MulMatNode::GetLocalItems(int mmqX, int mmqY) {
    int nbsIds = 0;
    int nbsX = 0;
    int nbsY = 0;
    GetLocalSplit(mmqX, mmqY, nbsIds, nbsX, nbsY);
    return nbsIds + nbsY + nbsX;
}

void MulMatNode::GetLocalSplit(
        int mmqX, 
        int mmqY,
        int &nbsIds,
        int &nbsX,
        int &nbsY) {
    nbsIds = mmqX;
    if (m_useMma) {
        int tileXK = GetMmaTileXK();
        nbsX = mmqY * tileXK;
    } else {
        int qs = 0; // int
        int dm = 0; // haf2
        int sc = 0; // int
        GetDp4aTxs(mmqY, qs, dm, sc);
        nbsX = qs + dm + sc;
    }
    nbsY = mmqX * NE_BLOCK_Q8_1_MMQ;
    nbsY = ocl::RndUp(nbsY, m_numSgs * m_sgSize);
}

void MulMatNode::GetDp4aTxs(
        int mmqY, 
        int &qs, 
        int &dm, 
        int &sc) {
    using namespace QuantConst;
    switch (m_bQuant) {
    case base::QuantMode::Q4_0:
        qs = mmqY * MMQ_TILE_NE_K + mmqY;
        dm = mmqY * MMQ_TILE_NE_K / QI4_0 + mmqY / QI4_0;
        sc = 0;
        break;
    case base::QuantMode::Q4_1:
        qs = mmqY * MMQ_TILE_NE_K + mmqY;
        dm = mmqY * MMQ_TILE_NE_K / QI4_1 + mmqY / QI4_1;
        sc = 0;
        break;
    case base::QuantMode::Q5_0:
        // same as Q8_0
        qs = mmqY * MMQ_TILE_NE_K * 2 + mmqY;
        dm = mmqY * MMQ_TILE_NE_K * 2 / QI8_0 + mmqY / (QI8_0 / 2);
        sc = 0;
        break;
    case base::QuantMode::Q5_1:
        qs = mmqY * MMQ_TILE_NE_K * 2 + mmqY;
        dm = mmqY * MMQ_TILE_NE_K * 2 / QI8_1 + mmqY / (QI8_1 / 2);
        sc = 0;
        break;
    case base::QuantMode::Q8_0:
        qs = mmqY * MMQ_TILE_NE_K * 2 + mmqY;
        dm = mmqY * MMQ_TILE_NE_K * 2 / QI8_0 + mmqY / (QI8_0 / 2);
        sc = 0;
        break;
    case base::QuantMode::MXFP4:
        // same as Q8_1
        qs = mmqY * MMQ_TILE_NE_K * 2 + mmqY;
        dm = mmqY * MMQ_TILE_NE_K * 2 / QI8_1 + mmqY / (QI8_1 / 2);
        sc = 0;
        break;
    case base::QuantMode::Q2_K:
        qs = mmqY * MMQ_TILE_NE_K * 2 + mmqY;
        dm = mmqY * MMQ_TILE_NE_K + mmqY;
        sc = 0;
        break;
    case base::QuantMode::Q3_K:
        qs = mmqY * MMQ_TILE_NE_K * 2 + mmqY;
        dm = mmqY;
        sc = mmqY * MMQ_TILE_NE_K / 8 + mmqY / 8;
        break;
    case base::QuantMode::Q4_K:
        qs = mmqY * MMQ_TILE_NE_K + mmqY;
        dm = mmqY * MMQ_TILE_NE_K / QI4_K;
        sc = mmqY * MMQ_TILE_NE_K / 8 + mmqY / 8;
        break;
    case base::QuantMode::Q5_K:
        qs = mmqY * MMQ_TILE_NE_K * 2 + mmqY;
        dm = mmqY * MMQ_TILE_NE_K / QI5_K + mmqY / QI5_K;
        sc = mmqY * MMQ_TILE_NE_K / 8 + mmqY / 8;
        break;
    case base::QuantMode::Q6_K:
        qs = mmqY * MMQ_TILE_NE_K * 2 + mmqY;
        dm = mmqY * MMQ_TILE_NE_K / QI6_K + mmqY / QI6_K;
        sc = mmqY * MMQ_TILE_NE_K / 8 + mmqY / 8;
        break;
    default:
        assert(false);
        qs = 0;
        dm = 0;
        sc = 0;
        break;
    }
}

int MulMatNode::GetMmaTileXK() {
    using namespace QuantConst;

    constexpr int TILE_X_K_Q8_0 = 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_0 + 4;
    constexpr int TILE_X_K_FP4 = 2 * MMQ_TILE_NE_K + 8 + 4;
    constexpr int TILE_X_K_Q8_1 = 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_0 + 4;
    constexpr int TILE_X_K_Q2_K = 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K + 4;
    constexpr int TILE_X_K_Q3_K = 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / 2 + 4;
    constexpr int TILE_X_K_Q6_K = 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI6_K + MMQ_TILE_NE_K / 8 + 7;

    switch (m_bQuant) {
    case base::QuantMode::Q4_0:
        return TILE_X_K_Q8_0;
    case base::QuantMode::Q4_1:
        return TILE_X_K_Q8_1;
    case base::QuantMode::Q5_0:
        return TILE_X_K_Q8_0;
    case base::QuantMode::Q5_1:
        return TILE_X_K_Q8_1;
    case base::QuantMode::Q8_0:
        return TILE_X_K_Q8_0;
    case base::QuantMode::MXFP4:
        return TILE_X_K_Q8_1;
    case base::QuantMode::Q2_K:
        return TILE_X_K_Q2_K;
    case base::QuantMode::Q3_K:
        return TILE_X_K_Q3_K;
    case base::QuantMode::Q4_K:
        return TILE_X_K_Q8_1;
    case base::QuantMode::Q5_K:
        return TILE_X_K_Q8_1;
    case base::QuantMode::Q6_K:
        return TILE_X_K_Q6_K;
    default:
        assert(false);
        return 0;
    }
}

} // namespace

//
//    MulMatQuantMm
//

MulMatQuantMm::MulMatQuantMm(Context *context):
        m_context(context) { }

MulMatQuantMm::~MulMatQuantMm() { }

std::unique_ptr<core::Node> MulMatQuantMm::CreateNode(core::Node *a, core::Node *b) {
    std::unique_ptr<MulMatNode> node = 
        std::make_unique<MulMatNode>(m_context, a, b, nullptr);
    if (node->Init()) {
        return node;
    }
    return nullptr;
}

//
//    MulMatIdQuantMm
//

MulMatIdQuantMm::MulMatIdQuantMm(Context *context):
        m_context(context) { }

MulMatIdQuantMm::~MulMatIdQuantMm() { }

std::unique_ptr<core::Node> MulMatIdQuantMm::CreateNode(
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

