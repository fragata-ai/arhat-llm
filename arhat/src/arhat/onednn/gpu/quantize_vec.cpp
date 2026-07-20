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

#include <string>
#include <memory>
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
//    QuantizeVec_Q8_1
//

class QuantizeVec_Q8_1: public QuantizeVecMm {
public:
    QuantizeVec_Q8_1(Context *context);
    ~QuantizeVec_Q8_1();
public:
    bool Init(
        const dnnl::memory::desc &xDesc,
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
    static constexpr int QK8_1 = 32;
private:
    Context *m_context;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory::dims m_yDims;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

QuantizeVec_Q8_1::QuantizeVec_Q8_1(Context *context):
        m_context(context) { }

QuantizeVec_Q8_1::~QuantizeVec_Q8_1() { }

bool QuantizeVec_Q8_1::Init(
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &yDesc) {
    m_xDesc = xDesc;
    m_yDesc = yDesc;
    m_yDims = m_yDesc.get_dims();
    // convert raw bytes to items
    m_yDims[3] = QuantBytesToItems(base::QuantMode::Q8_1, m_yDims[3]);
    if (!Validate()) {
        return false;
    }
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void QuantizeVec_Q8_1::Compute(
        const dnnl::memory &xMem,
        const dnnl::memory &idsMem,
        const dnnl::memory &yMem) {
    assert(!idsMem);
    m_kernel->SetArgBuffer(0, xMem);
    m_kernel->SetArgBuffer(1, yMem);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 2);
    m_kernel->Launch(m_ndRange);
}

bool QuantizeVec_Q8_1::Validate() {
    if (m_xDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_xDesc) || !MemoryDescUtil::HasDenseRows(m_xDesc)) { 
        return false;
    }
    if (m_yDims[3] % QK8_1 != 0) {
        return false;
    }
    return true;
}

void QuantizeVec_Q8_1::InitArgs() {
    size_t xBase = m_xDesc.get_submemory_offset();
    dnnl::memory::dims xDims = m_xDesc.get_dims();
    dnnl::memory::dims xStrides = m_xDesc.get_strides();
    size_t yBase = m_yDesc.get_submemory_offset();
    int64_t yDim1 = int64_t(m_yDims[1]);
    uint32_t yDim1_fd0 = 0;
    uint32_t yDim1_fd1 = 0;
    ocl::MakeFastDiv(yDim1, yDim1_fd0, yDim1_fd1);
    m_shapeInfoArgs.AddS64("SRC_BASE", xBase);
    m_shapeInfoArgs.AddS64("DST_BASE", yBase);
    m_shapeInfoArgs.AddS64("SRC_D3", xDims[3]);
    m_shapeInfoArgs.AddS64("SRC_S0", xStrides[0]);
    m_shapeInfoArgs.AddS64("SRC_S1", xStrides[1]);
    m_shapeInfoArgs.AddS64("SRC_S2", xStrides[2]);
    m_shapeInfoArgs.AddU32("DST_D1", yDim1);
    m_shapeInfoArgs.AddU32("DST_D1_fd0", yDim1_fd0);
    m_shapeInfoArgs.AddU32("DST_D1_fd1", yDim1_fd1);
    m_shapeInfoArgs.AddU32("DST_D2", m_yDims[2]);
    m_shapeInfoArgs.AddS64("DST_D3", m_yDims[3]);
}

void QuantizeVec_Q8_1::InitKernel() {
    std::string sig = MakeSig();
    if (m_context->FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::QuantizeVec_Q8_1_KernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_context->GetOclContext(), 
        "quantize_vec", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    m_context->EnterKernel(sig, m_kernel);
}

std::string QuantizeVec_Q8_1::MakeSig() {
    return "*quantize_vec_q8_1";
}

std::string QuantizeVec_Q8_1::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    ocl::CommonXe::EmitFastDiv(ss);
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    ss << "#define SG_SIZE 32\n";
    ss << "\n";
    return ss.str();
}

void QuantizeVec_Q8_1::InitNdRange() {
    size_t lws0 = 256;
    size_t gws0 = ocl::RndUp(size_t(m_yDims[3]), lws0);
    size_t gws1 = size_t(m_yDims[2]);
    size_t gws2 = size_t(m_yDims[0] * m_yDims[1]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

//
//    QuantizeVec_Q8_1_Soa
//

class QuantizeVec_Q8_1_Soa: public QuantizeVecMm {
public:
    QuantizeVec_Q8_1_Soa(Context *context);
    ~QuantizeVec_Q8_1_Soa();
public:
    bool Init(
        const dnnl::memory::desc &xDesc,
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
    static constexpr int QK8_1 = 32;
private:
    Context *m_context;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_yDesc;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

QuantizeVec_Q8_1_Soa::QuantizeVec_Q8_1_Soa(Context *context):
        m_context(context) { }

QuantizeVec_Q8_1_Soa::~QuantizeVec_Q8_1_Soa() { }

bool QuantizeVec_Q8_1_Soa::Init(
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &yDesc) {
    m_xDesc = xDesc;
    m_yDesc = yDesc;
    if (!Validate()) {
        return false;
    }
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void QuantizeVec_Q8_1_Soa::Compute(
        const dnnl::memory &xMem,
        const dnnl::memory &idsMem,
        const dnnl::memory &yMem) {
    assert(!idsMem);
    m_kernel->SetArgBuffer(0, xMem);
    m_kernel->SetArgBuffer(1, yMem);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 2);
    m_kernel->Launch(m_ndRange);
}

bool QuantizeVec_Q8_1_Soa::Validate() {
    if (m_xDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_xDesc) || !MemoryDescUtil::HasDenseRows(m_xDesc)) { 
        return false;
    }
    dnnl::memory::dims yDims = m_yDesc.get_dims();
    if (yDims[3] % QK8_1 != 0) {
        return false;
    }
    return true;
}

void QuantizeVec_Q8_1_Soa::InitArgs() {
    size_t xBase = m_xDesc.get_submemory_offset();
    dnnl::memory::dims xDims = m_xDesc.get_dims();
    dnnl::memory::dims xStrides = m_xDesc.get_strides();
    dnnl::memory::dims yDims = m_yDesc.get_dims();
    dnnl::memory::dims yStrides = m_yDesc.get_strides();
    size_t yBase = m_yDesc.get_submemory_offset();
    int64_t yDim1 = int64_t(yDims[1]);
    uint32_t yDim1_fd0 = 0;
    uint32_t yDim1_fd1 = 0;
    ocl::MakeFastDiv(yDim1, yDim1_fd0, yDim1_fd1);
    m_shapeInfoArgs.AddS64("SRC_BASE", xBase);
    m_shapeInfoArgs.AddS64("DST_BASE", yBase);
    m_shapeInfoArgs.AddS64("SRC_D3", xDims[3]);
    m_shapeInfoArgs.AddS64("SRC_S0", xStrides[0]);
    m_shapeInfoArgs.AddS64("SRC_S1", xStrides[1]);
    m_shapeInfoArgs.AddS64("SRC_S2", xStrides[2]);
    m_shapeInfoArgs.AddU32("DST_D1", yDim1);
    m_shapeInfoArgs.AddU32("DST_D1_fd0", yDim1_fd0);
    m_shapeInfoArgs.AddU32("DST_D1_fd1", yDim1_fd1);
    m_shapeInfoArgs.AddS64("DST_D3", yDims[3]);
    m_shapeInfoArgs.AddS64("DST_S0", yStrides[0]);
    m_shapeInfoArgs.AddS64("DST_S1", yStrides[1]);
    m_shapeInfoArgs.AddS64("DST_S2", yStrides[2]);
}

void QuantizeVec_Q8_1_Soa::InitKernel() {
    std::string sig = MakeSig();
    if (m_context->FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::QuantizeVec_Q8_1_Soa_KernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_context->GetOclContext(), 
        "quantize_vec", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    m_context->EnterKernel(sig, m_kernel);
}

std::string QuantizeVec_Q8_1_Soa::MakeSig() {
    return "*quantize_vec_q8_1_soa";
}

std::string QuantizeVec_Q8_1_Soa::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    ocl::CommonXe::EmitFastDiv(ss);
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    ss << "#define SG_SIZE 32\n";
    ss << "\n";
    return ss.str();
}

void QuantizeVec_Q8_1_Soa::InitNdRange() {
    dnnl::memory::dims yDims = m_yDesc.get_dims();
    size_t lws0 = 256;
    size_t gws0 = ocl::RndUp(size_t(yDims[3]), lws0);
    size_t gws1 = size_t(yDims[2]);
    size_t gws2 = size_t(yDims[0] * yDims[1]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

} // namespace

std::unique_ptr<QuantizeVecMm> CreateQuantizeVec_Q8_1(
        Context *context,
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &yDesc) {
    std::unique_ptr<QuantizeVec_Q8_1> q = std::make_unique<QuantizeVec_Q8_1>(context);
    if (!q->Init(xDesc, yDesc)) {
        return nullptr;
    }
    return q;
}

std::unique_ptr<QuantizeVecMm> CreateQuantizeVec_Q8_1_Soa(
        Context *context,
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &yDesc) {
    std::unique_ptr<QuantizeVec_Q8_1_Soa> q = std::make_unique<QuantizeVec_Q8_1_Soa>(context);
    if (!q->Init(xDesc, yDesc)) {
        return nullptr;
    }
    return q;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

