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

#include <memory>

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
#include "arhat/onednn/gpu/quantize.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

dnnl::memory::dim QuantBytesToItems(base::QuantMode quant, dnnl::memory::dim bytes) {
    int blockSize = base::GetBlockSize(quant);
    int quantSize = base::GetQuantSize(quant);
    assert(bytes % blockSize == 0);
    return (bytes / blockSize) * quantSize;
}

//
//    Q8_1 utilities
//

enum class DsLayout { D4, DS4, D2S6 };

DsLayout GetDsLayout(base::QuantMode quantOther) {
    switch (quantOther) {
    case base::QuantMode::Q4_0:
        return DsLayout::DS4;
    case base::QuantMode::Q4_1:
        return DsLayout::DS4;
    case base::QuantMode::Q5_0:
        return DsLayout::D4;
    case base::QuantMode::Q5_1:
        return DsLayout::DS4;
    case base::QuantMode::Q8_0:
        return DsLayout::D4;
    case base::QuantMode::MXFP4:
        return DsLayout::D4;
    case base::QuantMode::Q2_K:
        return DsLayout::D2S6;
    case base::QuantMode::Q3_K:
        return DsLayout::D4;
    case base::QuantMode::Q4_K:
        return DsLayout::DS4;
    case base::QuantMode::Q5_K:
        return DsLayout::DS4;
    case base::QuantMode::Q6_K:
        return DsLayout::D4;
    default:
        assert(false);
        return DsLayout(0);
    }
}

//
//    QuantizeMm_Q8_1
//

class QuantizeMm_Q8_1: public QuantizeVecMm {
public:
    QuantizeMm_Q8_1(Context *context);
    ~QuantizeMm_Q8_1();
public:
    bool Init(
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &yDesc,
        base::QuantMode quantOther);
    void Compute(
        const dnnl::memory &xMem,
        const dnnl::memory &idsMem,
        const dnnl::memory &yMem) override;
private:
    bool Validate();
    void InitConfig();
    void InitArgs();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    static constexpr int QK8_1 = 32;
private:
    Context *m_context;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_idsDesc;
    dnnl::memory::desc m_yDesc;
    base::QuantMode m_quantOther;
    dnnl::memory::dims m_yDims;
    DsLayout m_dsLayout;
    int64_t m_valsPerScale;
    int64_t m_valsPerSum;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

QuantizeMm_Q8_1::QuantizeMm_Q8_1(Context *context):
        m_context(context),
        m_quantOther(base::QuantMode::None),
        m_dsLayout(DsLayout(0)),
        m_valsPerScale(0),
        m_valsPerSum(0) { }

QuantizeMm_Q8_1::~QuantizeMm_Q8_1() { }

bool QuantizeMm_Q8_1::Init(
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &yDesc,
        base::QuantMode quantOther) {
    m_xDesc = xDesc;
    m_idsDesc = idsDesc;
    m_yDesc = yDesc;
    m_quantOther = quantOther;
    m_yDims = m_yDesc.get_dims();
    // convert raw bytes to items
    m_yDims[3] = QuantBytesToItems(base::QuantMode::Q8_1, m_yDims[3]);
    if (!Validate()) {
        return false;
    }
    InitConfig();
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void QuantizeMm_Q8_1::Compute(
        const dnnl::memory &xMem,
        const dnnl::memory &idsMem,
        const dnnl::memory &yMem) {
    m_kernel->SetArgBuffer(0, xMem);
    m_kernel->SetArgBuffer(1, idsMem);
    m_kernel->SetArgBuffer(2, yMem);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 3);
    m_kernel->Launch(m_ndRange);
}

bool QuantizeMm_Q8_1::Validate() {
    if (m_xDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_xDesc) || !MemoryDescUtil::HasDenseRows(m_xDesc)) { 
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
    switch (m_quantOther) {
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
    dnnl::memory::dims xDims = m_xDesc.get_dims();
    if (xDims[3] % 4 != 0) {
        return false;
    }
    if (m_yDims[3] % (4 * QK8_1) != 0) {
        return false;
    }
    return true;
}

void QuantizeMm_Q8_1::InitConfig() {
    m_dsLayout = GetDsLayout(m_quantOther);
    if (m_dsLayout == DsLayout::D2S6) {
        m_valsPerScale = 64;
        m_valsPerSum = 16;
    } else {
        m_valsPerScale = 32;
        m_valsPerSum = 32;
    }
}

void QuantizeMm_Q8_1::InitArgs() {
    size_t xBase = m_xDesc.get_submemory_offset();
    dnnl::memory::dims xDims = m_xDesc.get_dims();
    dnnl::memory::dims xStrides = m_xDesc.get_strides();
    size_t idsBase = !m_idsDesc.is_zero() ? m_idsDesc.get_submemory_offset() : 0;
    size_t yBase = m_yDesc.get_submemory_offset();
    m_shapeInfoArgs.AddS64("SRC0_BASE", xBase);
    m_shapeInfoArgs.AddS64("SRC1_BASE", idsBase);
    m_shapeInfoArgs.AddS64("DST_BASE", yBase);
    m_shapeInfoArgs.AddS64("SRC0_D3", xDims[3]);
    m_shapeInfoArgs.AddS64("SRC0_S0", xStrides[0]);
    m_shapeInfoArgs.AddS64("SRC0_S1", xStrides[1]);
    m_shapeInfoArgs.AddS64("SRC0_S2", xStrides[2]);
    m_shapeInfoArgs.AddU32("DST_D1", m_yDims[1]);
    m_shapeInfoArgs.AddU32("DST_D2", m_yDims[2]);
    m_shapeInfoArgs.AddS64("DST_D3", m_yDims[3]);
}

void QuantizeMm_Q8_1::InitKernel() {
    std::string sig = MakeSig();
    if (m_context->FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::QuantizeMm_Q8_1_KernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_context->GetOclContext(), 
        "quantize_mm", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    m_context->EnterKernel(sig, m_kernel);
}

std::string QuantizeMm_Q8_1::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*quantize_mm_q8_1");
    sb.Int(int64_t(m_quantOther));
    return sb.Get();
}

std::string QuantizeMm_Q8_1::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    ss << "#define SG_SIZE 32\n";
    ss << "#define DS_LAYOUT " << ocl::FormatInt(int64_t(m_dsLayout)) << "\n";
    ss << "#define VALS_PER_SCALE " << ocl::FormatInt(m_valsPerScale) << "\n";
    ss << "#define VALS_PER_SUM " << ocl::FormatInt(m_valsPerSum) << "\n";
    ss << "\n";
    return ss.str();
}

void QuantizeMm_Q8_1::InitNdRange() {
    // Require (MATRIX_ROW_PADDING % (4 * QUANTIZE_BLOCK_SIZE_MMQ) == 0)
    constexpr int QUANTIZE_BLOCK_SIZE_MMQ = 128;
    size_t lws0 = QUANTIZE_BLOCK_SIZE_MMQ;
    size_t gws0 = size_t(m_yDims[2]) * lws0;
    size_t gws1 = ocl::DivUp(size_t(m_yDims[3]), 4 * lws0);
    size_t gws2 = size_t(m_yDims[0] * m_yDims[1]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

//
//    QuantizeMm_Mxfp4
//

class QuantizeMm_Mxfp4: public QuantizeVecMm {
public:
    QuantizeMm_Mxfp4(Context *context);
    ~QuantizeMm_Mxfp4();
public:
    bool Init(
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &yDesc);
    void Compute(
        const dnnl::memory &xMem,
        const dnnl::memory &idsMem,
        const dnnl::memory &yMem) override;
private:
    bool Validate();
    void InitArgs();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    static constexpr int QK_MXFP4 = 32;
private:
    Context *m_context;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_idsDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory::dims m_yDims;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

QuantizeMm_Mxfp4::QuantizeMm_Mxfp4(Context *context):
        m_context(context) { }

QuantizeMm_Mxfp4::~QuantizeMm_Mxfp4() { }

bool QuantizeMm_Mxfp4::Init(
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &yDesc) {
    m_xDesc = xDesc;
    m_idsDesc = idsDesc;
    m_yDesc = yDesc;
    m_yDims = m_yDesc.get_dims();
    // convert raw bytes to items
    m_yDims[3] = QuantBytesToItems(base::QuantMode::MXFP4, m_yDims[3]);
    if (!Validate()) {
        return false;
    }
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void QuantizeMm_Mxfp4::Compute(
        const dnnl::memory &xMem,
        const dnnl::memory &idsMem,
        const dnnl::memory &yMem) {
    m_kernel->SetArgBuffer(0, xMem);
    m_kernel->SetArgBuffer(1, idsMem);
    m_kernel->SetArgBuffer(2, yMem);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 3);
    m_kernel->Launch(m_ndRange);
}

bool QuantizeMm_Mxfp4::Validate() {
    if (m_xDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_xDesc) || !MemoryDescUtil::HasDenseRows(m_xDesc)) { 
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
    if (m_yDims[3] % (2 * QK_MXFP4) != 0) {
        return false;
    }
    return true;
}

void QuantizeMm_Mxfp4::InitArgs() {
    size_t xBase = m_xDesc.get_submemory_offset();
    dnnl::memory::dims xDims = m_xDesc.get_dims();
    dnnl::memory::dims xStrides = m_xDesc.get_strides();
    size_t idsBase = !m_idsDesc.is_zero() ? m_idsDesc.get_submemory_offset() : 0;
    size_t yBase = m_yDesc.get_submemory_offset();
    m_shapeInfoArgs.AddS64("SRC0_BASE", xBase);
    m_shapeInfoArgs.AddS64("SRC1_BASE", idsBase);
    m_shapeInfoArgs.AddS64("DST_BASE", yBase);
    m_shapeInfoArgs.AddS64("SRC0_D3", xDims[3]);
    m_shapeInfoArgs.AddS64("SRC0_S0", xStrides[0]);
    m_shapeInfoArgs.AddS64("SRC0_S1", xStrides[1]);
    m_shapeInfoArgs.AddS64("SRC0_S2", xStrides[2]);
    m_shapeInfoArgs.AddU32("DST_D1", m_yDims[1]);
    m_shapeInfoArgs.AddU32("DST_D2", m_yDims[2]);
    m_shapeInfoArgs.AddS64("DST_D3", m_yDims[3]);
}

void QuantizeMm_Mxfp4::InitKernel() {
    std::string sig = MakeSig();
    if (m_context->FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::QuantizeMm_Mfxp4_KernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_context->GetOclContext(), 
        "quantize_mm", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    m_context->EnterKernel(sig, m_kernel);
}

std::string QuantizeMm_Mxfp4::MakeSig() {
    return "*quantize_mm_mxfp4";
}

std::string QuantizeMm_Mxfp4::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    ss << "#define SG_SIZE 32\n";
    ss << "\n";
    return ss.str();
}

void QuantizeMm_Mxfp4::InitNdRange() {
    size_t sgSize = 32;
    size_t numSgs = 8;
    size_t valsPerSg = 2 * QK_MXFP4;
    size_t valsPerWg = numSgs * valsPerSg;
    size_t lws0 = sgSize;
    size_t lws1 = numSgs;
    size_t gws0 = size_t(m_yDims[2]) * lws0;
    size_t gws1 = ocl::DivUp(size_t(m_yDims[3]), valsPerWg) * lws1;
    size_t gws2 = size_t(m_yDims[0] * m_yDims[1]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, lws1, 1);
}

} // namespace

std::unique_ptr<QuantizeVecMm> CreateQuantizeMm_Q8_1(
        Context *context,
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &yDesc,
        base::QuantMode quantOther) {
    std::unique_ptr<QuantizeMm_Q8_1> q = std::make_unique<QuantizeMm_Q8_1>(context);
    if (!q->Init(xDesc, idsDesc, yDesc, quantOther)) {
        return nullptr;
    }
    return q;
}

std::unique_ptr<QuantizeVecMm> CreateQuantizeMm_Mxfp4(
        Context *context,
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &yDesc) {
    std::unique_ptr<QuantizeMm_Mxfp4> q = std::make_unique<QuantizeMm_Mxfp4>(context);
    if (!q->Init(xDesc, idsDesc, yDesc)) {
        return nullptr;
    }
    return q;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

