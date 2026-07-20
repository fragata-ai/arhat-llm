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

//
//    Quantize_Q8_1_X4
//

class Quantize_Q8_1_X4: public QuantizeV2 {
public:
    Quantize_Q8_1_X4(Context *context);
    ~Quantize_Q8_1_X4();
public:
    bool Init(
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &yDesc);
    void Compute(
        const dnnl::memory &xMem,
        const dnnl::memory &yMem) override;
private:
    bool Validate();
    void InitConfig();
    void InitArgs();
    void InitShapeInfo();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    Context *m_context;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_yDesc;
    uint32_t m_sgSize;
    uint32_t m_wgDenom;
    uint32_t m_ne;
    uint32_t m_numBlocks;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

Quantize_Q8_1_X4::Quantize_Q8_1_X4(Context *context):
        m_context(context),
        m_sgSize(0),
        m_wgDenom(0),
        m_ne(0),
        m_numBlocks(0) { }

Quantize_Q8_1_X4::~Quantize_Q8_1_X4() { }

bool Quantize_Q8_1_X4::Init(
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &yDesc) {
    m_xDesc = xDesc;
    m_yDesc = yDesc;
    if (!Validate()) {
        return false;
    }
    InitConfig();
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void Quantize_Q8_1_X4::Compute(
        const dnnl::memory &xMem,
        const dnnl::memory &yMem) {
    m_kernel->SetArgBuffer(0, xMem);
    m_kernel->SetArgBuffer(1, yMem);
    m_kernel->SetArgU32(2, m_ne);
    m_kernel->SetArgU32(3, m_numBlocks);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 4);
    m_kernel->Launch(m_ndRange);
}

bool Quantize_Q8_1_X4::Validate() {
    if (m_xDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!base::IsRowMajor(m_xDesc)) {
        return false;
    }
    return true;
}

void Quantize_Q8_1_X4::InitConfig() {
    m_sgSize = 32;
    m_wgDenom = 32 * m_sgSize / 8;
}

void Quantize_Q8_1_X4::InitArgs() {
    dnnl::memory::dims xDims = m_xDesc.get_dims();
    m_ne = uint32_t(xDims[0] * xDims[1] * xDims[2] * xDims[3]);
    m_numBlocks = ocl::DivUp(m_ne, m_wgDenom);
    InitShapeInfo();
}

void Quantize_Q8_1_X4::InitShapeInfo() {
    size_t xBase = m_xDesc.get_submemory_offset();
    size_t yBase = m_yDesc.get_submemory_offset();
    m_shapeInfoArgs.AddS64("A_BASE", xBase);
    m_shapeInfoArgs.AddS64("B_BASE", yBase);
}

void Quantize_Q8_1_X4::InitKernel() {
    std::string sig = MakeSig();
    if (m_context->FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::QuantizeV2_Q8_1_X4_KernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_context->GetOclContext(), 
        "quantize_q8_1_x4", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    m_context->EnterKernel(sig, m_kernel);
}

std::string Quantize_Q8_1_X4::MakeSig() {
    return "*quantize_v2_q8_1_x4";
}

std::string Quantize_Q8_1_X4::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    ss << kernels::MulMatQuantVecV2Defs_Q8_1_Code();
    ss << "#define SG_SIZE " << m_sgSize << "\n";
    ss << "\n";
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    return ss.str();
}

void Quantize_Q8_1_X4::InitNdRange() {
    size_t lws0 = size_t(m_sgSize);
    size_t gws0 = size_t(ocl::DivUp(m_ne, m_wgDenom)) * lws0;
    m_ndRange = ocl::NdRange(gws0, 1, 1, lws0, 1, 1);
}

} // namespace

std::unique_ptr<QuantizeV2> CreateQuantizeV2_Q8_1_X4(
        Context *context,
        const dnnl::memory::desc &xDesc,
        const dnnl::memory::desc &yDesc) {
    std::unique_ptr<Quantize_Q8_1_X4> q = std::make_unique<Quantize_Q8_1_X4>(context);
    if (!q->Init(xDesc, yDesc)) {
        return nullptr;
    }
    return q;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

