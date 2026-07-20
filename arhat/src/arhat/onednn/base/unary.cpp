/* 
* MIT License
*
* Copyright (c) 2020-2026 FRAGATA COMPUTER SYSTEMS AG
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
#include <memory>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"

namespace arhat {
namespace onednn {
namespace base {

namespace {

// borrowed from [arhat/onednn/gpu/memory_desc.cpp]
bool IsDense(const dnnl::memory::desc &md) {
    dnnl::memory::dims dims = md.get_padded_dims();
    size_t nelems = 1;
    for (dnnl::memory::dim d: dims) {
        nelems *= d;
    }
    size_t bytes = md.get_size();
    int64_t items = base::BytesToItems(md.get_data_type(), int64_t(bytes));
    return (nelems == items);
}

//
//    UnaryNode
//

class UnaryNode: public NodeBase {
public:
    UnaryNode(
        Context *context,
        dnnl::algorithm algo,
        core::Node *x,
        float alpha,
        float beta,
        bool inplace);
    ~UnaryNode();
public:
    void Init();
public:
    void Compute() override;
private:
    dnnl::algorithm m_algo;
    core::Node *m_x;
    float m_alpha;
    float m_beta;
    bool m_inplace;
    dnnl::memory m_xMem;
    dnnl::eltwise_forward m_eltwise;
    TempMemory m_xTemp;
    Reorder m_xReorder;
};

UnaryNode::UnaryNode(
        Context *context,
        dnnl::algorithm algo,
        core::Node *x,
        float alpha,
        float beta,
        bool inplace):
            NodeBase(context),
            m_algo(algo),
            m_x(x),
            m_alpha(alpha),
            m_beta(beta),
            m_inplace(inplace) { }

UnaryNode::~UnaryNode() { }

void UnaryNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *x = m_context->CastNode(m_x);
    m_xMem = x->Memory();
    dnnl::memory::desc xDesc = x->MemoryDesc();
    // ACHTUNG: Temporary workaround for apparent bug in oneDNN
    //     If input memory is not dense, format_tag::any is translated to
    //  non-dense output format equal to input format - this is irrational
    //  and apparently does not work.
    //      Example: dims [13 11 7 5] strides [1155 105 15 1]
    m_context->MemoryPoolStart();
    dnnl::memory::desc xArgDesc;
    if (IsDense(xDesc)) {
        xArgDesc = xDesc;
    } else {
        if (m_inplace) {
            core::Error("Inplace binary operations are supported for dense inputs only");
        }
        xArgDesc = PlainMemoryDesc(xDesc);
        m_xTemp = m_context->AllocTempMemory(xArgDesc);
        m_xReorder.Init(m_context, xDesc, xArgDesc);
    }
    dnnl::memory::desc yDesc;
    if (m_inplace) {
        yDesc = xDesc;
    } else {
        yDesc =
            dnnl::memory::desc(
                x->MemoryDims(), 
                x->MemoryType(), 
                dnnl::memory::format_tag::any);
    }
    dnnl::eltwise_forward::primitive_desc prim(
        engine,
        dnnl::prop_kind::forward_inference,
        m_algo,
        xArgDesc,
        yDesc,
        m_alpha,
        m_beta);
    m_eltwise = dnnl::eltwise_forward(prim);
    dnnl::memory::desc dstDesc = prim.dst_desc();
    if (m_inplace) {
        if (dstDesc != xDesc) {
            core::Error("Bad destination descriptor for inplace operation");
        }
        SetMemory(dstDesc, m_xMem);
    } else {
        SetMemory(dstDesc);
    }
}

void UnaryNode::Compute() {
    dnnl::memory xArg = m_xMem;
    if (m_xReorder.IsSet()) {
        xArg = m_xTemp.Get();
        m_xReorder.Compute(m_xMem, xArg);
    }
    m_eltwise.execute(
        Stream(),
        {
            {DNNL_ARG_SRC, xArg},
            {DNNL_ARG_DST, m_memory}
        });
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateSqr(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_square,
            a,
            0.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateSqrt(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_sqrt,
            a,
            0.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateLog(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_log,
            a,
            0.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateScale(
        core::Node *a,
        float scale,
        float bias, 
        bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_linear,
            a,
            scale,
            bias,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateClamp(
        core::Node *a,
        float min,
        float max) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_clip,
            a,
            min,
            max,
            false);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateLeakyRelu(
        core::Node *a, 
        float negativeSlope, 
        bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_relu,
            a,
            negativeSlope,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateAbs(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_abs,
            a,
            0.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateNeg(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_linear,
            a,
            -1.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateTanh(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_tanh,
            a,
            0.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateElu(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_elu,
            a,
            1.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateRelu(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_relu,
            a,
            0.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateSigmoid(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_logistic,
            a,
            0.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateGelu(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_gelu_tanh,
            a,
            0.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateHardswish(core::Node *a) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_hardswish,
            a,
            1.0f / 6.0f,
            3.0f / 6.0f,
            false);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateHardsigmoid(core::Node *a) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_hardsigmoid,
            a,
            1.0f / 6.0f,
            3.0f / 6.0f,
            false);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateExp(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_exp,
            a,
            0.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateGeluErf(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_gelu_erf,
            a,
            0.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateRound(core::Node *a, bool inplace) {
    std::unique_ptr<UnaryNode> node =
        std::make_unique<UnaryNode>(
            this,
            dnnl::algorithm::eltwise_round,
            a,
            0.0f,
            0.0f,
            inplace);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

