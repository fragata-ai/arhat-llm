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

dnnl::algorithm MapScaleMode(core::ScaleMode mode) {
    switch (mode) {
    case core::ScaleMode::Nearest:
        return dnnl::algorithm::resampling_nearest;
    case core::ScaleMode::Bilinear:
        return dnnl::algorithm::resampling_linear;
    case core::ScaleMode::Bicubic:
    default:
        core::Error("Unsupported scale mode: %d", int(mode));
        return dnnl::algorithm::undef;
    }
}

//
//    UpscaleNode
//

class UpscaleNode: public NodeBase {
public:
    UpscaleNode(
        Context *context,
        core::Node *x,
        const core::Dims &shape,
        core::ScaleMode mode,
        bool alignCorners);
    ~UpscaleNode();
public:
    void Init();
public:
    void Compute() override;
private:
    core::Node *m_x;
    core::Dims m_shape;
    core::ScaleMode m_mode;
    bool m_alignCorners;
    dnnl::memory m_xMem;
    dnnl::resampling_forward m_resampling;
};

UpscaleNode::UpscaleNode(
        Context *context,
        core::Node *x,
        const core::Dims &shape,
        core::ScaleMode mode,
        bool alignCorners):
            NodeBase(context),
            m_x(x),
            m_shape(shape),
            m_mode(mode),
            m_alignCorners(alignCorners) { }

UpscaleNode::~UpscaleNode() { }

void UpscaleNode::Init() {
    if (m_alignCorners) {
        core::Error("Parameter alignCorners is not supported");
    }
    dnnl::engine &engine = Engine();
    NodeBase *x = m_context->CastNode(m_x);
    m_xMem = x->Memory();
    dnnl::memory::desc xDesc = x->MemoryDesc();
    dnnl::memory::dims yDims = MapDims(m_shape);
    dnnl::memory::data_type yType = x->MemoryType();
    dnnl::memory::desc yDesc(yDims, yType, dnnl::memory::format_tag::any);
    dnnl::algorithm algo = MapScaleMode(m_mode);
    dnnl::resampling_forward::primitive_desc prim(
        engine, 
        dnnl::prop_kind::forward_inference,
        algo,
        xDesc,
        yDesc);
    m_resampling = dnnl::resampling_forward(prim);
    SetMemory(prim.dst_desc());
}

void UpscaleNode::Compute() {
    m_resampling.execute(
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

std::unique_ptr<core::Node> Context::CreateUpscale(
        core::Node *a,
        const core::Dims &shape,
        core::ScaleMode mode,
        bool alignCorners) {
    std::unique_ptr<UpscaleNode> node = 
        std::make_unique<UpscaleNode>(this, a, shape, mode, alignCorners);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

