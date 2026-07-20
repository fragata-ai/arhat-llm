/* 
* MIT License
*
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

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/common_xe.hpp"
#include "arhat/onednn/ocl/shape_info_args.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/glu.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    GluNode
//

class GluNode: public NodeBase {
public:
    GluNode(
        Context *context,
        GluOp op,
        core::Node *a, 
        core::Node *b,
        bool swapped,
        const GluParam &param);
    ~GluNode();
public:
    bool Init();
public:
    void Compute() override;
protected:
    bool Validate();
    void InferShapes();
    void InitArgs();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    GluOp m_op;
    core::Node *m_a;
    core::Node *m_b;
    bool m_swapped;
    GluParam m_param;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_cDesc;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    int64_t m_k;
    int64_t m_n;
    int64_t m_o0;
    int64_t m_o1;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

GluNode::GluNode(
        Context *context,
        GluOp op,
        core::Node *a, 
        core::Node *b,
        bool swapped,
        const GluParam &param):
            NodeBase(context),
            m_op(op),
            m_a(a),
            m_b(b),
            m_swapped(swapped),
            m_param(param),
            m_k(0),
            m_n(0),
            m_o0(0),
            m_o1(0) { }

GluNode::~GluNode() { }

bool GluNode::Init() {
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    m_aDesc = a->MemoryDesc();
    m_bDesc = (b != nullptr) ? b->MemoryDesc() : m_aDesc;
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = (b != nullptr) ? b->Memory() : m_aMem;
    InferShapes();
    SetMemory(m_cDesc);
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void GluNode::Compute() {
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_bMem);
    m_kernel->SetArgBuffer(2, m_memory);
    m_kernel->SetArgS64(3, m_k);
    m_kernel->SetArgS64(4, m_n);
    m_kernel->SetArgS64(5, m_o0);
    m_kernel->SetArgS64(6, m_o1);
    for (int i = 0; i < m_param.count; i++) {
        m_kernel->SetArgF32(i + 7, m_param.param[i]);
    }
    m_shapeInfoArgs.SetArgs(m_kernel.get(), m_param.count + 7);
    m_kernel->Launch(m_ndRange);
}

bool GluNode::Validate() {
    dnnl::memory::data_type aType = m_aDesc.get_data_type();
    if (aType != dnnl::memory::data_type::f32 && aType != dnnl::memory::data_type::f16) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_aDesc) || !MemoryDescUtil::HasDenseRows(m_aDesc)) {
        return false;
    }
    if (m_b != nullptr) {
        dnnl::memory::data_type bType = m_bDesc.get_data_type();
        if (bType != aType) {
           return false;
        }
        if (!MemoryDescUtil::IsPlain(m_bDesc) || !MemoryDescUtil::HasDenseRows(m_bDesc)) {
            return false;
        }
    }
    return true;
}

void GluNode::InferShapes() {
    dnnl::memory::dims cDims = m_aDesc.get_dims();
    size_t ndims = cDims.size();
    if (m_b == nullptr) {
        cDims[ndims - 1] /= 2;
    }
    m_cDesc = 
        dnnl::memory::desc(
            cDims, 
            m_aDesc.get_data_type(), 
            base::DefaultFormatTag(int(ndims)));
}

void GluNode::InitArgs() {
    dnnl::memory::dim aBase = m_aDesc.get_submemory_offset();
    dnnl::memory::dim bBase = m_bDesc.get_submemory_offset();
    dnnl::memory::dim cBase = m_cDesc.get_submemory_offset();

    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims cDims = m_cDesc.get_dims();

    dnnl::memory::dims aStrides = m_aDesc.get_strides();
    dnnl::memory::dims bStrides = m_bDesc.get_strides();

    m_k = int64_t(cDims[0] * cDims[1] * cDims[2] * cDims[3]);
    m_n = (m_b != nullptr) ? int64_t(aDims[3]) : int64_t(aDims[3] / 2);
    m_o0 = int64_t(aStrides[2]);
    m_o1 = int64_t(bStrides[2]);

    if (m_b == nullptr) {
        if (m_swapped) {
            aBase += m_n;
        } else {
            bBase += m_n;
        }
    }

    m_shapeInfoArgs.AddS64("X_BASE", aBase);
    m_shapeInfoArgs.AddS64("G_BASE", bBase);
    m_shapeInfoArgs.AddS64("Y_BASE", cBase);
}

void GluNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelName = (m_op == GluOp::SwigluOai) ? "swiglu_oai" : "glu";
    const char *kernelCode = 
        (m_op == GluOp::SwigluOai) ? 
            kernels::GluSimpleSwigluOaiKernelCode() :
            kernels::GluSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        kernelName, 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string GluNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*glu_simple");
    sb.Int(int(m_op));
    sb.Int(int(m_aDesc.get_data_type()));
    return sb.Get();
}

std::string GluNode::MakeProlog() {
    std::stringstream ss;

    ocl::CommonXe::EmitGrid(ss);

    std::string op;

    switch (m_op) {
    case GluOp::Reglu:
        ss << kernels::UnarySimpleReluOpCode();
        op = "op_relu";
        break;
    case GluOp::Geglu:
        ss << kernels::UnarySimpleGeluOpCode();
        op = "op_gelu";
        break;
    case GluOp::Swiglu:
        ss << kernels::UnarySimpleSiluOpCode();
        op = "op_silu";
        break;
    case GluOp::SwigluOai:
        ss << kernels::GluSimpleSwigluOaiOpCode();
        op = "op_swiglu_oai";
        break;
    case GluOp::GegluErf:
        ss << kernels::UnarySimpleGeluErfOpCode();
        op = "op_gelu_erf";
        break;
    case GluOp::GegluQuick:
        ss << kernels::UnarySimpleGeluQuickOpCode();
        op = "op_gelu_quick";
        break;
    default:
        assert(false);
        break;
    }

    dnnl::memory::data_type aType = m_aDesc.get_data_type();

    ss << "#define T " << ocl::FormatType(aType) << "\n";
    ss << "#define OP " << op << "\n";
    ss << "\n";

    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";

    return ss.str();
}

void GluNode::InitNdRange() {
    size_t blockSize = 256;
    size_t lws0 = blockSize;
    size_t gws0 = ocl::DivUp(size_t(m_k), blockSize) * lws0;
    m_ndRange = ocl::NdRange(gws0, 1, 1, lws0, 1, 1);
}

} // namespace

//
//    GluSimple
//

GluSimple::GluSimple(Context *context):
         m_context(context) { }

GluSimple::~GluSimple() { }

std::unique_ptr<core::Node> GluSimple::CreateNode(
        GluOp op,
        core::Node *a, 
        core::Node *b,
        bool swapped,
        const GluParam &param) {
    std::unique_ptr<GluNode> node = 
        std::make_unique<GluNode>(m_context, op, a, b, swapped, param);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

