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

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/common_xe.hpp"
#include "arhat/onednn/ocl/shape_info_args.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/unary.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    UnaryNode
//

class UnaryNode: public NodeBase {
public:
    UnaryNode(
        Context *context,
        UnaryOp op,
        core::Node *x, 
        bool inplace,
        const UnaryParam &param);
    ~UnaryNode();
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
    UnaryOp m_op;
    core::Node *m_x;
    bool m_inplace;
    UnaryParam m_param;
    dnnl::memory::desc m_xDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory m_xMem;
    int32_t m_k;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
    base::Reorder m_xReorder;
};

UnaryNode::UnaryNode(
        Context *context,
        UnaryOp op,
        core::Node *x, 
        bool inplace,
        const UnaryParam &param):
            NodeBase(context),
            m_op(op),
            m_x(x),
            m_inplace(inplace),
            m_param(param),
            m_k(0) { }

UnaryNode::~UnaryNode() { }

bool UnaryNode::Init() {
    base::NodeBase *x = m_gpuContext->CastNode(m_x);
    m_xDesc = x->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_xMem = x->Memory();
    InferShapes();
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void UnaryNode::Compute() {
    dnnl::memory xArg = m_xMem;
    if (m_xReorder.IsSet()) {
        xArg = m_memory;
        m_xReorder.Compute(m_xMem, xArg);
    }

    m_kernel->SetArgBuffer(0, xArg);
    m_kernel->SetArgBuffer(1, m_memory);
    m_kernel->SetArgS32(2, m_k);
    for (int i = 0; i < m_param.count; i++) {
        m_kernel->SetArgF32(i + 3, m_param.param[i]);
    }
    m_shapeInfoArgs.SetArgs(m_kernel.get(), m_param.count + 3);
    m_kernel->Launch(m_ndRange);
}

bool UnaryNode::Validate() {
    dnnl::memory::data_type xType = m_xDesc.get_data_type();
    if (xType != dnnl::memory::data_type::f32 && xType != dnnl::memory::data_type::f16) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_xDesc)) {
        return false;
    }
    if (m_inplace && !base::IsRowMajor(m_xDesc)) {
        return false;
    }
    return true;
}

void UnaryNode::InferShapes() {
    if (m_inplace) {
        m_yDesc = m_xDesc;
        SetMemory(m_yDesc, m_xMem);
    } else {
        m_yDesc = base::PlainMemoryDesc(m_xDesc);
        SetMemory(m_yDesc);
        if (!base::IsRowMajor(m_xDesc)) {
            m_xReorder.Init(m_context, m_xDesc, m_yDesc);
        }
    }
}

void UnaryNode::InitArgs() {
    dnnl::memory::dim xBase = m_xDesc.get_submemory_offset();
    dnnl::memory::dim yBase = m_yDesc.get_submemory_offset();
    dnnl::memory::dims xDims = m_xDesc.get_dims();

    m_k = int32_t(xDims[0] * xDims[1] * xDims[2] * xDims[3]);

    m_shapeInfoArgs.AddS64("X_BASE", xBase);
    m_shapeInfoArgs.AddS64("Y_BASE", yBase);
}

void UnaryNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelName = (m_op == UnaryOp::Xielu) ? "xielu" : "unary";
    const char *kernelCode = 
        (m_op == UnaryOp::Xielu) ? 
            kernels::UnarySimpleXieluKernelCode() :
            kernels::UnarySimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        kernelName, 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string UnaryNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*unary_simple");
    sb.Int(int(m_op));
    sb.Int(int(m_xDesc.get_data_type()));
    return sb.Get();
}

std::string UnaryNode::MakeProlog() {
    std::stringstream ss;

    ocl::CommonXe::EmitGrid(ss);

    std::string op;

    switch (m_op) {
    case UnaryOp::Sqr:
        ss << kernels::UnarySimpleSqrOpCode();
        op = "op_sqr";
        break;
    case UnaryOp::Sqrt:
        ss << kernels::UnarySimpleSqrtOpCode();
        op = "op_sqrt";
        break;
    case UnaryOp::Log:
        ss << kernels::UnarySimpleLogOpCode();
        op = "op_log";
        break;
    case UnaryOp::Sin:
        ss << kernels::UnarySimpleSinOpCode();
        op = "op_sin";
        break;
    case UnaryOp::Cos:
        ss << kernels::UnarySimpleCosOpCode();
        op = "op_cos";
        break;
    case UnaryOp::Abs:
        ss << kernels::UnarySimpleAbsOpCode();
        op = "op_abs";
        break;
    case UnaryOp::Sgn:
        ss << kernels::UnarySimpleSgnOpCode();
        op = "op_sgn";
        break;
    case UnaryOp::Neg:
        ss << kernels::UnarySimpleNegOpCode();
        op = "op_neg";
        break;
    case UnaryOp::Step:
        ss << kernels::UnarySimpleStepOpCode();
        op = "op_step";
        break;
    case UnaryOp::Tanh:
        ss << kernels::UnarySimpleTanhOpCode();
        op = "op_tanh";
        break;
    case UnaryOp::Elu:
        ss << kernels::UnarySimpleEluOpCode();
        op = "op_elu";
        break;
    case UnaryOp::Relu:
        ss << kernels::UnarySimpleReluOpCode();
        op = "op_relu";
        break;
    case UnaryOp::Sigmoid:
        ss << kernels::UnarySimpleSigmoidOpCode();
        op = "op_sigmoid";
        break;
    case UnaryOp::Gelu:
        ss << kernels::UnarySimpleGeluOpCode();
        op = "op_gelu";
        break;
    case UnaryOp::GeluQuick:
        ss << kernels::UnarySimpleGeluQuickOpCode();
        op = "op_gelu_quick";
        break;
    case UnaryOp::Silu:
        ss << kernels::UnarySimpleSiluOpCode();
        op = "op_silu";
        break;
    case UnaryOp::Hardswish:
        ss << kernels::UnarySimpleHardswishOpCode();
        op = "op_hardswish";
        break;
    case UnaryOp::Hardsigmoid:
        ss << kernels::UnarySimpleHardsigmoidOpCode();
        op = "op_hardsigmoid";
        break;
    case UnaryOp::Exp:
        ss << kernels::UnarySimpleExpOpCode();
        op = "op_exp";
        break;
    case UnaryOp::Expm1:
        ss << kernels::UnarySimpleExpm1OpCode();
        op = "op_expm1";
        break;
    case UnaryOp::Softplus:
        ss << kernels::UnarySimpleSoftplusOpCode();
        op = "op_softplus";
        break;
    case UnaryOp::GeluErf:
        ss << kernels::UnarySimpleGeluErfOpCode();
        op = "op_gelu_erf";
        break;
    case UnaryOp::Xielu:
        // special kernel
        break;
    case UnaryOp::Floor:
        ss << kernels::UnarySimpleFloorOpCode();
        op = "op_floor";
        break;
    case UnaryOp::Ceil:
        ss << kernels::UnarySimpleCeilOpCode();
        op = "op_ceil";
        break;
    case UnaryOp::Round:
        ss << kernels::UnarySimpleRoundOpCode();
        op = "op_round";
        break;
    case UnaryOp::Trunc:
        ss << kernels::UnarySimpleTruncOpCode();
        op = "op_trunc";
        break;
    default:
        assert(false);
        break;
    }

    dnnl::memory::data_type xType = m_xDesc.get_data_type();

    ss << "#define T " << ocl::FormatType(xType) << "\n";
    ss << "#define OP " << op << "\n";
    ss << "\n";

    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";

    return ss.str();
}

void UnaryNode::InitNdRange() {
    size_t blockSize = 256;
    size_t lws0 = blockSize;
    size_t gws0 = ocl::DivUp(size_t(m_k), blockSize) * lws0;
    m_ndRange = ocl::NdRange(gws0, 1, 1, lws0, 1, 1);
}

} // namespace

//
//    UnarySimple
//

UnarySimple::UnarySimple(Context *context):
         m_context(context) { }

UnarySimple::~UnarySimple() { }

std::unique_ptr<core::Node> UnarySimple::CreateNode(
        UnaryOp op,
        core::Node *a, 
        bool inplace,
        const UnaryParam &param) {
    std::unique_ptr<UnaryNode> node = 
        std::make_unique<UnaryNode>(m_context, op, a, inplace, param);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

