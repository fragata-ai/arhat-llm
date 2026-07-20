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

#include <cassert>
#include <memory>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"

namespace arhat {
namespace onednn {
namespace base {

namespace {

dnnl::algorithm MapPoolOp(core::PoolOp op) {
    switch (op) {
    case core::PoolOp::Avg:
        return dnnl::algorithm::pooling_avg_include_padding;
    case core::PoolOp::Max:
        return dnnl::algorithm::pooling_max;
    default:
        assert(false);
        return dnnl::algorithm::undef;
    }
}

dnnl::memory::dim InferPoolDim(dnnl::memory::dim xDim, int k, int s, int p) {
    return (xDim + 2 * p - k) / s + 1;
}

//
//    Pool2dNode
//

class Pool2dNode: public NodeBase {
public:
    Pool2dNode(
        Context *context,
        core::Node *x,
        core::PoolOp op,
        int k0,
        int k1,
        int s0,
        int s1,
        int p0,
        int p1);
    ~Pool2dNode();
public:
    void Init();
public:
    void Compute() override;
private:
    dnnl::memory::dims InferShape(const dnnl::memory::dims &xDims);
private:
    core::Node *m_x;
    core::PoolOp m_op;
    int m_k0;
    int m_k1;
    int m_s0;
    int m_s1;
    int m_p0;
    int m_p1;
    dnnl::memory m_xMem;
    dnnl::pooling_forward m_pool;
};

Pool2dNode::Pool2dNode(
        Context *context,
        core::Node *x,
        core::PoolOp op,
        int k0,
        int k1,
        int s0,
        int s1,
        int p0,
        int p1):
            NodeBase(context),
            m_x(x),
            m_op(op),
            m_k0(k0),
            m_k1(k1),
            m_s0(s0),
            m_s1(s1),
            m_p0(p0),
            m_p1(p1) { }

Pool2dNode::~Pool2dNode() { }

void Pool2dNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *x = m_context->CastNode(m_x);
    m_xMem = x->Memory();
    dnnl::memory::desc xDesc = x->MemoryDesc();
    dnnl::memory::dims xDims = x->MemoryDims();
    dnnl::memory::data_type xType = x->MemoryType();
    dnnl::memory::dims yDims = InferShape(xDims);
    dnnl::memory::desc yDesc(yDims, xType, dnnl::memory::format_tag::any); 
    dnnl::algorithm algo = MapPoolOp(m_op);
    dnnl::memory::dims strides{m_s1, m_s0};
    dnnl::memory::dims kernel{m_k1, m_k0};
    dnnl::memory::dims dilation{0, 0};
    dnnl::memory::dims padding{m_p1, m_p0};
    dnnl::pooling_forward::primitive_desc prim(
        engine,
        dnnl::prop_kind::forward_inference,
        algo,
        xDesc,
        yDesc,
        strides,
        kernel,
        dilation,
        padding,
        padding);
    m_pool = dnnl::pooling_forward(prim);
    SetMemory(prim.dst_desc());
}

void Pool2dNode::Compute() {
    m_pool.execute(
        Stream(),
        {
            {DNNL_ARG_SRC, m_xMem},
            {DNNL_ARG_DST, m_memory},
        });
}

dnnl::memory::dims Pool2dNode::InferShape(const dnnl::memory::dims &xDims) {
    dnnl::memory::dims yDims(4);
    yDims[0] = xDims[0];
    yDims[1] = xDims[1];
    yDims[2] = InferPoolDim(xDims[2], m_k1, m_s1, m_p1);
    yDims[3] = InferPoolDim(xDims[3], m_k0, m_s0, m_p0);
    return yDims;
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreatePool2d(
        core::Node *a,
        core::PoolOp op,
        int k0,
        int k1,
        int s0,
        int s1,
        int p0,
        int p1) {
    std::unique_ptr<Pool2dNode> node =
        std::make_unique<Pool2dNode>(this, a, op, k0, k1, s0, s1, p0, p1);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

