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

#include <memory>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"

namespace arhat {
namespace onednn {
namespace base {

namespace {

//
//    SoftMaxNode
//

class SoftMaxNode: public NodeBase {
public:
    SoftMaxNode(
        Context *context,
        core::Node *x,
        core::Node *mask,
        core::Node *sinks,
        float scale,
        float maxBias, 
        bool inplace);
    ~SoftMaxNode();
public:
    void Init();
public:
    void Compute() override;
private:
    core::Node *m_x;
    core::Node *m_mask;
    core::Node *m_sinks;
    float m_scale;
    float m_maxBias;
    bool m_inplace;
    dnnl::memory m_xMem;
    dnnl::softmax_forward m_softmax;
};

SoftMaxNode::SoftMaxNode(
        Context *context,
        core::Node *x,
        core::Node *mask,
        core::Node *sinks,
        float scale,
        float maxBias, 
        bool inplace):
            NodeBase(context),
            m_x(x),
            m_mask(mask),
            m_sinks(sinks),
            m_scale(scale),
            m_maxBias(maxBias),
            m_inplace(inplace) { }

SoftMaxNode::~SoftMaxNode() { }

void SoftMaxNode::Init() {
    if (m_mask != nullptr) {
        core::Error("Input mask is not supported");
    }
    if (m_sinks != nullptr) {
        core::Error("Input sinks is not supported");
    }
    if (m_scale != 1.0f) {
        core::Error("Parameter scale is not supported");
    }
    if (m_maxBias != 0.0f) {
        core::Error("Parameter maxBias is not supported");
    }
    dnnl::engine &engine = Engine();
    NodeBase *x = m_context->CastNode(m_x);
    m_xMem = x->Memory();
    dnnl::memory::desc xDesc = x->MemoryDesc();
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
    dnnl::primitive_attr attr;
    int axis = core::MaxDims - 1; // fixed by design
    dnnl::softmax_forward::primitive_desc prim(
        Engine(), 
        dnnl::prop_kind::forward_inference, 
        dnnl::algorithm::softmax_accurate, 
        xDesc, 
        yDesc, 
        axis,
        attr);
    m_softmax = dnnl::softmax_forward(prim);
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

void SoftMaxNode::Compute() {
    m_softmax.execute(
        Stream(),
        {
            {DNNL_ARG_SRC, m_xMem},
            {DNNL_ARG_DST, m_memory}
        });
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateSoftMax(
        core::Node *a,
        core::Node *mask,
        core::Node *sinks,
        float scale,
        float maxBias, 
        bool inplace) {
    std::unique_ptr<SoftMaxNode> node =
        std::make_unique<SoftMaxNode>(
            this,
            a,
            mask,
            sinks,
            scale,
            maxBias,
            inplace);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

