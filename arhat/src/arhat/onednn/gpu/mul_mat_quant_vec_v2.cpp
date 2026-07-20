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
#include "arhat/onednn/gpu/mul_mat_util.hpp"
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
    void InitQuantize();
    void InitConfig();
    void InitArgs();
    void InitShapeInfo();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    static constexpr int MAX_COLS = 8;
private:
    core::Node *m_a;
    core::Node *m_b;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_cDesc;
    bool m_isNop;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::memory::dims m_bDims;
    base::QuantMode m_bQuant;
    dnnl::memory::desc m_aTempDesc;
    base::TempMemory m_aTempMem;
    std::unique_ptr<QuantizeV2> m_quantize;
    uint32_t m_sgSize;
    uint32_t m_blockSize;
    uint32_t m_numRows;
    uint32_t m_numCols;
    uint32_t m_kPerIter;
    uint32_t m_ncols;
    uint32_t m_strideA;
    uint32_t m_strideB;
    uint32_t m_strideD;
    uint32_t m_batchStrideA;
    uint32_t m_batchStrideB;
    uint32_t m_batchStrideD; 
    uint32_t m_baseWorkGroupY;
    uint32_t m_aDim1;
    uint32_t m_bDim1;
    uint32_t m_aBcast0;
    uint32_t m_aBcast1;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
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
            m_isNop(false),
            m_bQuant(base::QuantMode::None),
            m_sgSize(0),
            m_blockSize(0),
            m_numRows(0),
            m_numCols(0),
            m_kPerIter(0),
            m_ncols(0),
            m_strideA(0),
            m_strideB(0),
            m_strideD(0),
            m_batchStrideA(0),
            m_batchStrideB(0),
            m_batchStrideD(0), 
            m_baseWorkGroupY(0),
            m_aDim1(0),
            m_bDim1(0),
            m_aBcast0(0),
            m_aBcast1(0) { }

MulMatNode::~MulMatNode() { }

bool MulMatNode::Init() {
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    m_aDesc = a->MemoryDesc();
    m_bDesc = b->MemoryDesc();
    m_bQuant = b->Quant();
    // cannot use m_bDesc.get_dims() for quantized tensors
    m_bDims = b->MemoryDims();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
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

    m_quantize->Compute(m_aMem, m_aTempMem.Get());

    // data_a = b, data_b = a
    m_kernel->SetArgBuffer(0, m_bMem);
    m_kernel->SetArgBuffer(1, m_aTempMem.Get());
    m_kernel->SetArgBuffer(2, m_memory);
    m_kernel->SetArgU32(3, m_ncols);
    m_kernel->SetArgU32(4, m_strideA);
    m_kernel->SetArgU32(5, m_strideB);
    m_kernel->SetArgU32(6, m_strideD);
    m_kernel->SetArgU32(7, m_batchStrideA);
    m_kernel->SetArgU32(8, m_batchStrideB);
    m_kernel->SetArgU32(9, m_batchStrideD); 
    m_kernel->SetArgU32(10, m_baseWorkGroupY);
    m_kernel->SetArgU32(11, m_aDim1);
    m_kernel->SetArgU32(12, m_bDim1);
    m_kernel->SetArgU32(13, m_aBcast0);
    m_kernel->SetArgU32(14, m_aBcast1);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 15);
    m_kernel->Launch(m_ndRange);
}

bool MulMatNode::Validate() {
    if (m_aDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MulMatV2Util::HasDenseRowsCols(m_aDesc)) { 
        return false;
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
    if (!MulMatV2Util::HasDenseRowsCols(m_bDesc)) { 
        return false;
    }
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDims;
    dnnl::memory::dim numCols = aDims[2];
    if (numCols > MAX_COLS) {
        return false;
    }
    if (aDims[2] != 1 && aDims[0] * aDims[1] != 1) {
        return false;
    }
    if ((aDims[2] * aDims[3]) % 4 != 0) {
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
    m_isNop = (cDims[0] * cDims[1] * cDims[2] * cDims[3] == 0);
}

void MulMatNode::InitQuantize() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dim aDim3 = aDims[3];
    base::QuantMode aQuant = base::QuantMode::Q8_1;
    int blockSize = base::GetBlockSize(aQuant);
    int quantSize = base::GetQuantSize(aQuant);
    assert(aDim3 % quantSize == 0);
    aDims[3] = (aDim3 / quantSize) * blockSize;
    m_aTempDesc = 
        dnnl::memory::desc(
            aDims, 
            dnnl::memory::data_type::u8, 
            dnnl::memory::format_tag::abcd);
    m_aTempMem = m_context->AllocTempMemory(m_aTempDesc);
    m_quantize = CreateQuantizeV2_Q8_1_X4(GetGpuContext(), m_aDesc, m_aTempDesc);
}

void MulMatNode::InitConfig() {
    ocl::DeviceInfo *deviceInfo = m_gpuContext->GetDeviceInfo();
    ocl::GpuArch arch = deviceInfo->GetGpuArch();
    // Incorporate minSgSize into ocl::DeviceInfo?
    uint32_t minSgSize = (arch >= ocl::GpuArch::Xe2) ? 16 : 8;
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    m_sgSize = minSgSize;
    m_blockSize = m_sgSize;
    m_numCols = uint32_t(aDims[2]);
    switch (m_bQuant) {
    case base::QuantMode::Q4_0:
    case base::QuantMode::Q4_1:
    case base::QuantMode::Q5_0:
    case base::QuantMode::Q5_1:
    case base::QuantMode::Q8_0:
        m_numRows = 2;
        m_kPerIter = 8;
        break;
    case base::QuantMode::Q2_K:
    case base::QuantMode::Q3_K:
    case base::QuantMode::Q4_K:
    case base::QuantMode::Q5_K:
    case base::QuantMode::Q6_K:
        m_numRows = (m_bQuant == base::QuantMode::Q2_K) ? 2 : 1;
        m_kPerIter = 16;
        break;
    case base::QuantMode::MXFP4:
        m_numRows = 4;
        m_kPerIter = 8;
        break;
    default:
        assert(false);
        break;
    }
}

void MulMatNode::InitArgs() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDims;
    dnnl::memory::dims cDims = m_cDesc.get_dims();
    // Note A <-> B for host vs kernel naming
    m_ncols = uint32_t(bDims[3]);
    m_strideA = uint32_t(aDims[3]);
    m_strideB = uint32_t(aDims[3]);
    m_strideD = uint32_t(bDims[2]);
    assert(aDims[2] == 1 || aDims[0] * aDims[1] == 1);
    bool batchN = (aDims[2] > 1);
    // For batchN, the B matrix is the same for each batch
    // and A/C use the row stride as the batch stride
    if (batchN) {
        m_batchStrideA = 0;
        m_batchStrideB = uint32_t(aDims[3]);
        m_batchStrideD = uint32_t(cDims[3]);
    } else {
        m_batchStrideA = uint32_t(bDims[2] * bDims[3]);
        m_batchStrideB = uint32_t(aDims[2] * aDims[3]);
        m_batchStrideD = uint32_t(cDims[2] * cDims[3]);
    }
    m_baseWorkGroupY = 0; // Reserved: N/A for OpenCL?
    m_aDim1 = uint32_t(bDims[1]);
    m_bDim1 = uint32_t(aDims[1]);
    m_aBcast0 = uint32_t(aDims[0] / bDims[0]);
    m_aBcast1 = uint32_t(aDims[1] / bDims[1]);
    InitShapeInfo();
}

void MulMatNode::InitShapeInfo() {
    size_t aBase = m_aDesc.get_submemory_offset();
    size_t bBase = m_bDesc.get_submemory_offset();
    size_t cBase = m_cDesc.get_submemory_offset();
    // a <-> b
    m_shapeInfoArgs.AddS64("A_BASE", bBase);
    m_shapeInfoArgs.AddS64("B_BASE", aBase);
    m_shapeInfoArgs.AddS64("D_BASE", cBase);
}

void MulMatNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::MulMatQuantVecV2KernelCode();
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
    sb.String("*mul_mat_vec_q_v2");
    sb.Int(int64_t(m_bQuant));
    sb.Int(int64_t(m_blockSize));
    sb.Int(int64_t(m_numRows));
    sb.Int(int64_t(m_numCols));
    return sb.Get();
}

std::string MulMatNode::MakeProlog() {
    std::stringstream ss;

    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    ocl::CommonXe::EmitImad(ss);

    ss << "#define FLOAT_TYPE float\n";
    ss << "\n";

    ss << "#define D_TYPE float\n";
    ss << "\n";

    ss << "#define IMAD(a, b, c) imad(as_char4(a), as_char4(b), c)\n";
    ss << "\n";

    EmitInt(ss, "SG_SIZE", m_sgSize);
    EmitInt(ss, "BLOCK_SIZE", m_blockSize);
    EmitInt(ss, "NUM_ROWS", m_numRows);
    EmitInt(ss, "NUM_COLS", m_numCols);
    EmitInt(ss, "K_PER_ITER", m_kPerIter);
    ss << "\n";

    ss << "#define USE_SUBGROUP_ADD_NO_SHMEM\n";
    ss << "\n";

    ss << kernels::MulMatQuantVecV2Defs_Q8_1_Code();
    switch (m_bQuant) {
    case base::QuantMode::Q4_0:
        ss << kernels::MulMatQuantVecV2Defs_Q4_0_Code();
        ss << kernels::MulMatQuantVecV2BaseCode();
        ss << kernels::MulMatQuantVecV2Impl_Q4_0_Code();
        ss << kernels::MulMatQuantVecV2ImplLegacyCode();
        break;
    case base::QuantMode::Q4_1:
        ss << kernels::MulMatQuantVecV2Defs_Q4_1_Code();
        ss << kernels::MulMatQuantVecV2BaseCode();
        ss << kernels::MulMatQuantVecV2Impl_Q4_1_Code();
        ss << kernels::MulMatQuantVecV2ImplLegacyCode();
        break;
    case base::QuantMode::Q5_0:
        ss << kernels::MulMatQuantVecV2Defs_Q5_0_Code();
        ss << kernels::MulMatQuantVecV2BaseCode();
        ss << kernels::MulMatQuantVecV2Impl_Q5_0_Code();
        ss << kernels::MulMatQuantVecV2ImplLegacyCode();
        break;
    case base::QuantMode::Q5_1:
        ss << kernels::MulMatQuantVecV2Defs_Q5_1_Code();
        ss << kernels::MulMatQuantVecV2BaseCode();
        ss << kernels::MulMatQuantVecV2Impl_Q5_1_Code();
        ss << kernels::MulMatQuantVecV2ImplLegacyCode();
        break;
    case base::QuantMode::Q8_0:
        ss << kernels::MulMatQuantVecV2Defs_Q8_0_Code();
        ss << kernels::MulMatQuantVecV2BaseCode();
        ss << kernels::MulMatQuantVecV2Impl_Q8_0_Code();
        ss << kernels::MulMatQuantVecV2ImplLegacyCode();
        break;
    case base::QuantMode::Q2_K:
        ss << kernels::MulMatQuantVecV2Defs_Q2_K_Code();
        ss << kernels::MulMatQuantVecV2BaseCode();
        ss << kernels::MulMatQuantVecV2Impl_Q2_K_Code();
        break;
    case base::QuantMode::Q3_K:
        ss << kernels::MulMatQuantVecV2Defs_Q3_K_Code();
        ss << kernels::MulMatQuantVecV2BaseCode();
        ss << kernels::MulMatQuantVecV2Impl_Q3_K_Code();
        break;
    case base::QuantMode::Q4_K:
        ss << kernels::MulMatQuantVecV2Defs_Q4_K_Code();
        ss << kernels::MulMatQuantVecV2BaseCode();
        ss << kernels::MulMatQuantVecV2Impl_Q4_K_Code();
        ss << kernels::MulMatQuantVecV2Impl_Q45_K_Code();
        break;
    case base::QuantMode::Q5_K:
        ss << kernels::MulMatQuantVecV2Defs_Q5_K_Code();
        ss << kernels::MulMatQuantVecV2BaseCode();
        ss << kernels::MulMatQuantVecV2Impl_Q5_K_Code();
        ss << kernels::MulMatQuantVecV2Impl_Q45_K_Code();
        break;
    case base::QuantMode::Q6_K:
        ss << kernels::MulMatQuantVecV2Defs_Q6_K_Code();
        ss << kernels::MulMatQuantVecV2BaseCode();
        ss << kernels::MulMatQuantVecV2Impl_Q6_K_Code();
        break;
    case base::QuantMode::MXFP4:
        ss << kernels::MulMatQuantVecV2Defs_Mxfp4_Code();
        ss << kernels::MulMatQuantVecV2BaseCode();
        ss << kernels::MulMatQuantVecV2Impl_Mxfp4_Code();
        ss << kernels::MulMatQuantVecV2ImplLegacyCode();
        break;
    default:
        assert(false);
        break;
    }

    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";

    return ss.str();
}

void MulMatNode::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDims;
    size_t lws0 = size_t(m_sgSize);
    size_t gws0 = ocl::DivUp(size_t(bDims[2]), size_t(m_numRows)) * lws0;
    size_t gws1 = size_t(aDims[0] * aDims[1]);
    m_ndRange = ocl::NdRange(gws0, gws1, 1, lws0, 1, 1);
}

} // namespace

//
//    MulMatQuantVecV2
//

MulMatQuantVecV2::MulMatQuantVecV2(Context *context):
        m_context(context) { }

MulMatQuantVecV2::~MulMatQuantVecV2() { }

std::unique_ptr<core::Node> MulMatQuantVecV2::CreateNode(core::Node *a, core::Node *b) {
    std::unique_ptr<MulMatNode> node = 
        std::make_unique<MulMatNode>(m_context, a, b);
    if (node->Init()) {
        return node;
    }
    return nullptr;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

