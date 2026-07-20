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
#include <array>
#include <memory>
#include <ostream>
#include <sstream>
#include <algorithm>

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
#include "arhat/onednn/gpu/rope.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    RopeKernelMode
//

enum class RopeKernelMode {
    Norm,
    Neox,
    Multi,
    Vision
};

//
//    RopeNode
//

class RopeNode: public NodeBase {
public:
    RopeNode(
        Context *context,
        core::Node *a,
        core::Node *b,
        core::Node *c,
        int nDims,
        core::RopeMode mode,
        int nCtxOrig,
        float freqBase,
        float freqScale,
        float extFactor,
        float attnFactor,
        float betaFast,
        float betaSlow,
        const std::array<int, core::MropeSections> &sections, 
        bool inplace);
    ~RopeNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    void InitConfig();
    void InitYarnCorrDims();
    float YarnCorrDim(float nRot);
    void InitArgs();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    const char *GetKernelCode();
    void InitNdRange();
private:
    core::Node *m_a;
    core::Node *m_b;
    core::Node *m_c;
    int m_nDims;
    core::RopeMode m_mode;
    int m_nCtxOrig;
    float m_freqBase;
    float m_freqScale;
    float m_extFactor;
    float m_attnFactor;
    float m_betaFast;
    float m_betaSlow;
    std::array<int, core::MropeSections> m_sections;
    bool m_inplace;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_cDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::memory m_cMem;
    RopeKernelMode m_kernelMode;
    bool m_isImrope;
    float m_corrLow;
    float m_corrHigh;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

RopeNode::RopeNode(
        Context *context,
        core::Node *a,
        core::Node *b,
        core::Node *c,
        int nDims,
        core::RopeMode mode,
        int nCtxOrig,
        float freqBase,
        float freqScale,
        float extFactor,
        float attnFactor,
        float betaFast,
        float betaSlow,
        const std::array<int, core::MropeSections> &sections, 
        bool inplace):
            NodeBase(context),
            m_a(a),
            m_b(b),
            m_c(c),
            m_nDims(nDims),
            m_mode(mode),
            m_nCtxOrig(nCtxOrig),
            m_freqBase(freqBase),
            m_freqScale(freqScale),
            m_extFactor(extFactor),
            m_attnFactor(attnFactor),
            m_betaFast(betaFast),
            m_betaSlow(betaSlow),
            m_sections(sections),
            m_inplace(inplace),
            m_kernelMode(RopeKernelMode(0)),
            m_isImrope(false),
            m_corrLow(0.0f),
            m_corrHigh(0.0f) { }

RopeNode::~RopeNode() { }

bool RopeNode::Init() {
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    base::NodeBase *c = m_gpuContext->CastNode(m_c);
    m_aDesc = a->MemoryDesc();
    m_bDesc = b->MemoryDesc();
    if (c != nullptr) {
        m_cDesc = c->MemoryDesc();
    }
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    if (c != nullptr) {
        m_cMem = c->Memory();
    }
    if (m_inplace) {
        m_yDesc = m_aDesc;
        SetMemory(m_yDesc, m_aMem);
    } else {
        m_yDesc = base::PlainMemoryDesc(m_aDesc);
        SetMemory(m_yDesc);
    }
    InitConfig();
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void RopeNode::Compute() {
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_bMem);
    m_kernel->SetArgBuffer(2, m_cMem);
    m_kernel->SetArgBuffer(3, m_memory);
    m_kernel->SetArgS32(4, m_nDims);
    m_kernel->SetArgF32(5, m_freqBase);
    m_kernel->SetArgF32(6, m_freqScale);
    m_kernel->SetArgF32(7, m_extFactor);
    m_kernel->SetArgF32(8, m_attnFactor);
    m_kernel->SetArgF32(9, m_corrLow);
    m_kernel->SetArgF32(10, m_corrHigh);
    int argIdx = 11;
    switch (m_kernelMode) {
    case RopeKernelMode::Multi:
        m_kernel->SetArgS32(11, m_sections[0]);
        m_kernel->SetArgS32(12, m_sections[1]);
        m_kernel->SetArgS32(13, m_sections[2]);
        m_kernel->SetArgS32(14, m_sections[3]);
        argIdx = 15;
        break;
    case RopeKernelMode::Vision:
        m_kernel->SetArgS32(11, m_sections[0]);
        m_kernel->SetArgS32(12, m_sections[1]);
        argIdx = 13;
        break;
    }
    m_shapeInfoArgs.SetArgs(m_kernel.get(), argIdx);
    m_kernel->Launch(m_ndRange);
}

bool RopeNode::Validate() {
    dnnl::memory::data_type aType = m_aDesc.get_data_type();
    if (aType != dnnl::memory::data_type::f32 &&
            aType != dnnl::memory::data_type::f16) {
        return false;
    }
    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    if (bType != dnnl::memory::data_type::s32) {
        return false;
    }
    if (!m_cDesc.is_zero()) {
        dnnl::memory::data_type cType = m_cDesc.get_data_type();
        if (cType != dnnl::memory::data_type::f32) {
            return false;
        }
    }
    if (!MemoryDescUtil::IsBlocked(m_aDesc)) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_bDesc) || 
            !MemoryDescUtil::HasDenseRows(m_bDesc) ||
            m_bDesc.get_submemory_offset() != 0) {
        return false;
    }
    if (!m_cDesc.is_zero()) {
        if (!MemoryDescUtil::IsPlain(m_cDesc) || 
                !MemoryDescUtil::HasDenseRows(m_cDesc) ||
                m_cDesc.get_submemory_offset() != 0) {
            return false;
        }
    }
    return true;
}

void RopeNode::InitConfig() {
    switch (m_mode) {
    case core::RopeMode::Normal:
        m_kernelMode = RopeKernelMode::Norm;
        m_isImrope = false;
        break;
    case core::RopeMode::Neox:
        m_kernelMode = RopeKernelMode::Neox;
        m_isImrope = false;
        break;
    case core::RopeMode::Mrope:
        m_kernelMode = RopeKernelMode::Multi;
        m_isImrope = false;
        break;
    case core::RopeMode::Vision:
        m_kernelMode = RopeKernelMode::Vision;
        m_isImrope = false;
        break;
    case core::RopeMode::Imrope:
        m_kernelMode = RopeKernelMode::Multi;
        m_isImrope = true;
        break;
    default:
        assert(false);
        break;
    }
    InitYarnCorrDims();
}

void RopeNode::InitYarnCorrDims() {
    float start = std::floorf(YarnCorrDim(m_betaFast));
    float end = std::floorf(YarnCorrDim(m_betaSlow));
    m_corrLow = std::max(0.0f, start);
    m_corrHigh = std::min(float(m_nDims - 1), end);
}

float RopeNode::YarnCorrDim(float nRot) {
    static constexpr double M_PI = 3.14159265358979323846;
    return float(m_nDims) * 
        std::logf(float(m_nCtxOrig) / (nRot * 2.0f * float(M_PI))) / 
        (2.0f * std::logf(m_freqBase));
}

void RopeNode::InitArgs() {
    dnnl::memory::dim aBase = m_aDesc.get_submemory_offset();
    dnnl::memory::dim bBase = m_bDesc.get_submemory_offset();
    dnnl::memory::dim cBase = m_cDesc.get_submemory_offset();
    dnnl::memory::dim yBase = m_yDesc.get_submemory_offset();
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims yDims = m_yDesc.get_dims();
    dnnl::memory::dims aStrides = m_aDesc.get_strides();
    dnnl::memory::dims yStrides = m_yDesc.get_strides();

    m_shapeInfoArgs.AddS64("SRC0_BASE", aBase);
    m_shapeInfoArgs.AddS64("SRC1_BASE", bBase);
    m_shapeInfoArgs.AddS64("SRC2_BASE", cBase);
    m_shapeInfoArgs.AddS64("DST_BASE", yBase);
    m_shapeInfoArgs.AddS32("SRC0_D0", aDims[0]);
    m_shapeInfoArgs.AddS32("SRC0_D1", aDims[1]);
    m_shapeInfoArgs.AddS32("SRC0_D2", aDims[2]);
    m_shapeInfoArgs.AddS32("SRC0_D3", aDims[3]);
    m_shapeInfoArgs.AddS32("SRC0_S0", aStrides[0]);
    m_shapeInfoArgs.AddS32("SRC0_S1", aStrides[1]);
    m_shapeInfoArgs.AddS32("SRC0_S2", aStrides[2]);
    m_shapeInfoArgs.AddS32("SRC0_S3", aStrides[3]);
    m_shapeInfoArgs.AddS32("DST_D0", yDims[0]);
    m_shapeInfoArgs.AddS32("DST_D1", yDims[1]);
    m_shapeInfoArgs.AddS32("DST_D2", yDims[2]);
    m_shapeInfoArgs.AddS32("DST_D3", yDims[3]);
    m_shapeInfoArgs.AddS32("DST_S0", yStrides[0]);
    m_shapeInfoArgs.AddS32("DST_S1", yStrides[1]);
    m_shapeInfoArgs.AddS32("DST_S2", yStrides[2]);
    m_shapeInfoArgs.AddS32("DST_S3", yStrides[3]);
}

void RopeNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = GetKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "rope_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string RopeNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*rope_simple");
    sb.Int(int64_t(m_mode));
    sb.Int(int64_t(m_yDesc.get_data_type()));
    sb.Bool(int64_t(m_isImrope));
    return sb.Get();
}

std::string RopeNode::MakeProlog() {
    std::stringstream ss;

    ocl::CommonXe::EmitGrid(ss);

    dnnl::memory::data_type yType = m_yDesc.get_data_type();
    ss << "#define DATA_T " << ocl::FormatType(yType) << "\n";
    if (yType == dnnl::memory::data_type::f32) {
        ss << "#define TO_FLOAT(x) (x)\n";
        ss << "#define TO_DATA_T(x) (x)\n";
    } else if (yType == dnnl::memory::data_type::f16) {
        ss << "#define TO_FLOAT(x) convert_float(x)\n";
        ss << "#define TO_DATA_T(x) convert_half(x)\n";
    } else {
        assert(false);
    }
    ss << "\n";

    if (m_kernelMode == RopeKernelMode::Multi) {
        EmitInt(ss, "IS_IMROPE", m_isImrope ? 1 : 0);
    }

    ss << "#define SRC0_OFF(i0, i1, i2, i3) " <<
        "((i0) * SRC0_S0 + (i1) * SRC0_S1 + (i2) * SRC0_S2 + (i3) * SRC0_S3)\n";
    ss << "#define DST_OFF(i0, i1, i2, i3) " <<
        "((i0) * DST_S0 + (i1) * DST_S1 + (i2) * DST_S2 + (i3) * DST_S3)\n";
    ss << "\n";

    ss << kernels::RopeSimpleCommonCode();

    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";

    return ss.str();
}

const char *RopeNode::GetKernelCode() {
    switch (m_kernelMode) {
    case RopeKernelMode::Norm:
        return kernels::RopeSimpleNormKernelCode();
    case RopeKernelMode::Neox:
        return kernels::RopeSimpleNeoxKernelCode();
    case RopeKernelMode::Multi:
        return kernels::RopeSimpleMultiKernelCode();
    case RopeKernelMode::Vision:
        return kernels::RopeSimpleVisionKernelCode();
    default:
        assert(false);
        return "";
    }
}

void RopeNode::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t lws0 = std::min(size_t(64), size_t(aDims[3]));
    size_t gws0 = lws0 * size_t(aDims[2]);
    size_t gws1 = size_t(aDims[1]);
    size_t gws2 = size_t(aDims[0]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

} // namespace

//
//    RopeSimple
//

RopeSimple::RopeSimple(Context *context):
        m_context(context) { }

RopeSimple::~RopeSimple() { }

std::unique_ptr<core::Node> RopeSimple::CreateNode(
        core::Node *a,
        core::Node *b,
        core::Node *c,
        int nDims,
        core::RopeMode mode,
        int nCtxOrig,
        float freqBase,
        float freqScale,
        float extFactor,
        float attnFactor,
        float betaFast,
        float betaSlow,
        const std::array<int, core::MropeSections> &sections, 
        bool inplace) {
    std::unique_ptr<RopeNode> node =
        std::make_unique<RopeNode>(
            m_context,
            a,
            b,
            c,
            nDims,
            mode,
            nCtxOrig,
            freqBase,
            freqScale,
            extFactor,
            attnFactor,
            betaFast,
            betaSlow,
            sections, 
            inplace);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

