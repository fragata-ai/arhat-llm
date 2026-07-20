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

dnnl::memory::dim InferConvDim(
        dnnl::memory::dim xDim,
        dnnl::memory::dim wDim,
        int s,
        int p,
        int d) {
    return (xDim + 2 * p - d * (wDim - 1) - 1) / s + 1;
}

//
//    Conv2dNode
//

class Conv2dNode: public NodeBase {
public:
    Conv2dNode(
        Context *context,
        core::Node *w,
        core::Node *x,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1);
    ~Conv2dNode();
public:
    void Init();
public:
    void Compute() override;
private:
    dnnl::memory::dims InferShape(
        const dnnl::memory::dims &xDims, 
        const dnnl::memory::dims &wDims);
private:
    core::Node *m_w;
    core::Node *m_x;
    int m_s0;
    int m_s1;
    int m_p0;
    int m_p1;
    int m_d0;
    int m_d1;
    dnnl::memory m_wMem;
    dnnl::memory m_xMem;
    dnnl::convolution_forward m_conv;
    TempMemory m_wTemp;
    TempMemory m_xTemp;
    Reorder m_wReorder;
    Reorder m_xReorder;
};

Conv2dNode::Conv2dNode(
        Context *context,
        core::Node *w,
        core::Node *x,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1):
            NodeBase(context),
            m_w(w),
            m_x(x),
            m_s0(s0),
            m_s1(s1),
            m_p0(p0),
            m_p1(p1),
            m_d0(d0),
            m_d1(d1) { }

Conv2dNode::~Conv2dNode() { }

void Conv2dNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *w = m_context->CastNode(m_w);
    NodeBase *x = m_context->CastNode(m_x);
    m_wMem = w->Memory();
    m_xMem = x->Memory();
    dnnl::memory::desc wDesc = w->MemoryDesc();
    dnnl::memory::desc xDesc = x->MemoryDesc();
    dnnl::memory::dims wDims = w->MemoryDims();
    dnnl::memory::dims xDims = x->MemoryDims();
    dnnl::memory::data_type wType = w->MemoryType();
    dnnl::memory::data_type xType = x->MemoryType();
    dnnl::memory::format_tag any = dnnl::memory::format_tag::any;
    dnnl::memory::desc wPrimDesc(wDims, wType, any);
    dnnl::memory::desc xPrimDesc(xDims, xType, any);
    dnnl::memory::dims yDims = InferShape(xDims, wDims);
    dnnl::memory::desc yDesc(yDims, xType, any); 
    dnnl::memory::dims strides{m_s1, m_s0};
    dnnl::memory::dims dilates{m_d1 - 1, m_d0 - 1};
    dnnl::memory::dims padding{m_p1, m_p0};
    dnnl::convolution_forward::primitive_desc prim(
        engine,
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_auto,
        xPrimDesc,
        wPrimDesc,
        yDesc,
        strides,
        dilates,
        padding,
        padding);
    m_conv = dnnl::convolution_forward(prim);
    m_context->MemoryPoolStart();
    wPrimDesc = prim.weights_desc();
    xPrimDesc = prim.src_desc();
    if (wDesc != wPrimDesc) {
        m_wTemp = m_context->AllocTempMemory(wPrimDesc);
        m_wReorder.Init(m_context, wDesc, wPrimDesc);
    }
    if (xDesc != xPrimDesc) {
        m_xTemp = m_context->AllocTempMemory(xPrimDesc);
        m_xReorder.Init(m_context, xDesc, xPrimDesc);
    }
    SetMemory(prim.dst_desc());
}

void Conv2dNode::Compute() {
    dnnl::memory wArg = m_wMem;
    dnnl::memory xArg = m_xMem;
    if (m_wReorder.IsSet()) {
        wArg = m_wTemp.Get();
        m_wReorder.Compute(m_wMem, wArg);
    }
    if (m_xReorder.IsSet()) {
        xArg = m_xTemp.Get();
        m_xReorder.Compute(m_xMem, xArg);
    }
    m_conv.execute(
        Stream(),
        {
            {DNNL_ARG_SRC, xArg},
            {DNNL_ARG_WEIGHTS, wArg},
            {DNNL_ARG_DST, m_memory}
        });
}

dnnl::memory::dims Conv2dNode::InferShape(
        const dnnl::memory::dims &xDims, 
        const dnnl::memory::dims &wDims) {
    dnnl::memory::dims yDims(4);
    yDims[0] = xDims[0];
    yDims[1] = wDims[0];
    yDims[2] = InferConvDim(xDims[2], wDims[2], m_s1, m_p1, m_d1);
    yDims[3] = InferConvDim(xDims[3], wDims[3], m_s0, m_p0, m_d0);
    return yDims;
}

//
//    Conv3dNode
//

class Conv3dNode: public NodeBase {
public:
    Conv3dNode(
        Context *context,
        core::Node *w,
        core::Node *x,
        int s0,
        int s1,
        int s2,
        int p0,
        int p1,
        int p2,
        int d0,
        int d1,
        int d2,
        int C,
        int N,
        int OC);
    ~Conv3dNode();
public:
    void Init();
public:
    void Compute() override;
private:
    dnnl::memory::dims InferShape(
        const dnnl::memory::dims &xDims, 
        const dnnl::memory::dims &wDims);
private:
    core::Node *m_w;
    core::Node *m_x;
    int m_s0;
    int m_s1;
    int m_s2;
    int m_p0;
    int m_p1;
    int m_p2;
    int m_d0;
    int m_d1;
    int m_d2;
    int m_C;
    int m_N;
    int m_OC;
    dnnl::memory m_wMem;
    dnnl::memory m_xMem;
    dnnl::convolution_forward m_conv;
    dnnl::memory m_wTemp;
    dnnl::memory m_xTemp;
    dnnl::memory m_yTemp;
    Reorder m_wReorder;
    Reorder m_xReorder;
    Reorder m_yReorder;
};

Conv3dNode::Conv3dNode(
        Context *context,
        core::Node *w,
        core::Node *x,
        int s0,
        int s1,
        int s2,
        int p0,
        int p1,
        int p2,
        int d0,
        int d1,
        int d2,
        int C,
        int N,
        int OC):
            NodeBase(context),
            m_w(w),
            m_x(x),
            m_s0(s0),
            m_s1(s1),
            m_s2(s2),
            m_p0(p0),
            m_p1(p1),
            m_p2(p2),
            m_d0(d0),
            m_d1(d1),
            m_d2(d2),
            m_C(C),
            m_N(N),
            m_OC(OC) { }

Conv3dNode::~Conv3dNode() { }

void Conv3dNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *w = m_context->CastNode(m_w);
    NodeBase *x = m_context->CastNode(m_x);
    m_wMem = w->Memory();
    m_xMem = x->Memory();
    dnnl::memory::desc wDesc = w->MemoryDesc();
    dnnl::memory::desc xDesc = x->MemoryDesc();
    dnnl::memory::dims wDims = w->MemoryDims();
    dnnl::memory::dims xDims = x->MemoryDims();
    dnnl::memory::data_type wType = w->MemoryType();
    dnnl::memory::data_type xType = x->MemoryType();
    assert(wDims[0] == m_OC * m_C);
    assert(xDims[0] == m_N * m_C);
    dnnl::memory::dims wDims5d{m_OC, m_C, wDims[1], wDims[2], wDims[3]};
    dnnl::memory::dims xDims5d{m_N, m_C, xDims[1], xDims[2], xDims[3]};
    dnnl::memory::desc wDesc5d = wDesc.reshape(wDims5d);
    dnnl::memory::desc xDesc5d = xDesc.reshape(xDims5d);
    dnnl::memory::format_tag any = dnnl::memory::format_tag::any;
    dnnl::memory::desc wPrimDesc(wDims5d, wType, any);
    dnnl::memory::desc xPrimDesc(xDims5d, xType, any);
    dnnl::memory::dims yDims5d = InferShape(xDims, wDims);
    dnnl::memory::desc yDesc5d(yDims5d, xType, any); 
    dnnl::memory::dims strides{m_s2, m_s1, m_s0};
    dnnl::memory::dims dilates{m_d2 - 1, m_d1 - 1, m_d0 - 1};
    dnnl::memory::dims padding{m_p2, m_p1, m_p0};
    dnnl::convolution_forward::primitive_desc prim(
        engine,
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_auto,
        xPrimDesc,
        wPrimDesc,
        yDesc5d,
        strides,
        dilates,
        padding,
        padding);
    m_conv = dnnl::convolution_forward(prim);
    wPrimDesc = prim.weights_desc();
    xPrimDesc = prim.src_desc();
    if (wDesc5d == wPrimDesc) {
        m_wTemp = m_wMem;
    } else {
        m_wTemp = dnnl::memory(wPrimDesc, engine);
        m_wReorder.Init(m_context, wDesc5d, wPrimDesc);
    }
    if (xDesc5d == xPrimDesc) {
        m_xTemp = m_xMem;
    } else {
        m_xTemp = dnnl::memory(xPrimDesc, engine);
        m_xReorder.Init(m_context, xDesc5d, xPrimDesc);
    }
    dnnl::memory::dims yDims{m_N * m_OC, yDims5d[2], yDims5d[3], yDims5d[4]};
    dnnl::memory::desc yPrimDesc = prim.dst_desc();
    // Replace following code with embedded regular Reshape primitive?
    dnnl::memory::desc yDesc = yPrimDesc.reshape(yDims, true);
    if (yDesc) {
        SetMemory(yDesc);
        m_yTemp = m_memory;
    } else {
        // reshape may fail for some formats
        // if this happens, reorder to plain format and try again
        m_yTemp = dnnl::memory(yPrimDesc, engine);
        dnnl::memory::desc yPlainDesc(yDims5d, xType, dnnl::memory::format_tag::abcde);
        yDesc = yPlainDesc.reshape(yDims);
        SetMemory(yDesc);
        m_yReorder.Init(m_context, yPrimDesc, yPlainDesc);
    }
}

void Conv3dNode::Compute() {
    if (m_wReorder.IsSet()) {
        m_wReorder.Compute(m_wMem, m_wTemp);
    }
    if (m_xReorder.IsSet()) {
        m_xReorder.Compute(m_xMem, m_xTemp);
    }
    m_conv.execute(
        Stream(),
        {
            {DNNL_ARG_SRC, m_xTemp},
            {DNNL_ARG_WEIGHTS, m_wTemp},
            {DNNL_ARG_DST, m_yTemp}
        });
    if (m_yReorder.IsSet()) {
        m_yReorder.Compute(m_yTemp, m_memory);
    }
}

dnnl::memory::dims Conv3dNode::InferShape(
        const dnnl::memory::dims &xDims, 
        const dnnl::memory::dims &wDims) {
    dnnl::memory::dims yDims(5);
    yDims[0] = m_N;
    yDims[1] = m_OC;
    yDims[2] = InferConvDim(xDims[1], wDims[1], m_s2, m_p2, m_d2);
    yDims[3] = InferConvDim(xDims[2], wDims[2], m_s1, m_p1, m_d1);
    yDims[4] = InferConvDim(xDims[3], wDims[3], m_s0, m_p0, m_d0);
    return yDims;
}

//
//    Conv2dDwNode
//

class Conv2dDwNode: public NodeBase {
public:
    Conv2dDwNode(
        Context *context,
        core::Node *w,
        core::Node *x,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1);
    ~Conv2dDwNode();
public:
    void Init();
public:
    void Compute() override;
private:
    dnnl::memory::dims InferShape(
        const dnnl::memory::dims &xDims, 
        const dnnl::memory::dims &wDims);
private:
    core::Node *m_w;
    core::Node *m_x;
    int m_s0;
    int m_s1;
    int m_p0;
    int m_p1;
    int m_d0;
    int m_d1;
    dnnl::memory m_wMem;
    dnnl::memory m_xMem;
    dnnl::convolution_forward m_conv;
    dnnl::memory m_wTemp;
    dnnl::memory m_xTemp;
    Reorder m_wReorder;
    Reorder m_xReorder;
};

Conv2dDwNode::Conv2dDwNode(
        Context *context,
        core::Node *w,
        core::Node *x,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1):
            NodeBase(context),
            m_w(w),
            m_x(x),
            m_s0(s0),
            m_s1(s1),
            m_p0(p0),
            m_p1(p1),
            m_d0(d0),
            m_d1(d1) { }

Conv2dDwNode::~Conv2dDwNode() { }

void Conv2dDwNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *w = m_context->CastNode(m_w);
    NodeBase *x = m_context->CastNode(m_x);
    m_wMem = w->Memory();
    m_xMem = x->Memory();
    dnnl::memory::desc wDesc = w->MemoryDesc();
    dnnl::memory::desc xDesc = x->MemoryDesc();
    dnnl::memory::dims wDims = w->MemoryDims();
    dnnl::memory::dims xDims = x->MemoryDims();
    dnnl::memory::data_type wType = w->MemoryType();
    dnnl::memory::data_type xType = x->MemoryType();
    dnnl::memory::dims yDims = InferShape(xDims, wDims);
    // reshape to 5d to specify oneDNN group convolution
    assert(wDims[1] == 1);
    wDims = {wDims[0], 1, 1, wDims[2], wDims[3]};
    wDesc = wDesc.reshape(wDims);
    dnnl::memory::format_tag any = dnnl::memory::format_tag::any;
    dnnl::memory::desc wPrimDesc(wDims, wType, any);
    dnnl::memory::desc xPrimDesc(xDims, xType, any);
    dnnl::memory::desc yDesc(yDims, xType, any); 
    dnnl::memory::dims strides{m_s1, m_s0};
    dnnl::memory::dims dilates{m_d1 - 1, m_d0 - 1};
    dnnl::memory::dims padding{m_p1, m_p0};
    dnnl::convolution_forward::primitive_desc prim(
        engine,
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::convolution_auto,
        xPrimDesc,
        wPrimDesc,
        yDesc,
        strides,
        dilates,
        padding,
        padding);
    m_conv = dnnl::convolution_forward(prim);
    wPrimDesc = prim.weights_desc();
    xPrimDesc = prim.src_desc();
    if (wDesc == wPrimDesc) {
        m_wTemp = m_wMem;
    } else {
        m_wTemp = dnnl::memory(wPrimDesc, engine);
        m_wReorder.Init(m_context, wDesc, wPrimDesc);
    }
    if (xDesc == xPrimDesc) {
        m_xTemp = m_xMem;
    } else {
        m_xTemp = dnnl::memory(xPrimDesc, engine);
        m_xReorder.Init(m_context, xDesc, xPrimDesc);
    }
    SetMemory(prim.dst_desc());
}

void Conv2dDwNode::Compute() {
    if (m_wReorder.IsSet()) {
        m_wReorder.Compute(m_wMem, m_wTemp);
    }
    if (m_xReorder.IsSet()) {
        m_xReorder.Compute(m_xMem, m_xTemp);
    }
    m_conv.execute(
        Stream(),
        {
            {DNNL_ARG_SRC, m_xTemp},
            {DNNL_ARG_WEIGHTS, m_wTemp},
            {DNNL_ARG_DST, m_memory}
        });
}

dnnl::memory::dims Conv2dDwNode::InferShape(
        const dnnl::memory::dims &xDims, 
        const dnnl::memory::dims &wDims) {
    dnnl::memory::dims yDims(4);
    yDims[0] = xDims[0];
    yDims[1] = xDims[1];
    yDims[2] = InferConvDim(xDims[2], wDims[2], m_s1, m_p1, m_d1);
    yDims[3] = InferConvDim(xDims[3], wDims[3], m_s0, m_p0, m_d0);
    return yDims;
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateConv2d(
        core::Node *a,
        core::Node *b,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1) {
    std::unique_ptr<Conv2dNode> node = 
        std::make_unique<Conv2dNode>(this, a, b, s0, s1, p0, p1, d0, d1);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateConv3d(
        core::Node *a,
        core::Node *b,
        int s0,
        int s1,
        int s2,
        int p0,
        int p1,
        int p2,
        int d0,
        int d1,
        int d2,
        int C,
        int N,
        int OC) {
    std::unique_ptr<Conv3dNode> node = 
        std::make_unique<Conv3dNode>(this, a, b, s0, s1, s2, p0, p1, p2, d0, d1, d2, C, N, OC);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateConv2dDw(
        core::Node *a,
        core::Node *b,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1) {
    std::unique_ptr<Conv2dDwNode> node = 
        std::make_unique<Conv2dDwNode>(this, a, b, s0, s1, p0, p1, d0, d1);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

