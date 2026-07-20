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

//
//    ReshapeNode
//

class ReshapeNode: public NodeBase {
public:
    ReshapeNode(
        Context *context,
        core::Node *x,
        const core::Dims &shape);
    ~ReshapeNode();
public:
    void Init();
public:
    void Compute() override;
private:
    bool IsNop();
private:
    core::Node *m_x;
    core::Dims m_shape;
    dnnl::memory m_xMem;
    Reorder m_reorder;
};

ReshapeNode::ReshapeNode(
        Context *context,
        core::Node *x,
        const core::Dims &shape):
             NodeBase(context),
             m_x(x),
             m_shape(shape) { }

ReshapeNode::~ReshapeNode() { }

void ReshapeNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *x = m_context->CastNode(m_x);
    m_xMem = x->Memory();
    dnnl::memory::desc xDesc = x->MemoryDesc();
    dnnl::memory::dims yDims = MapDims(m_shape);
    if (IsNop()) {
        dnnl::memory::desc yDesc(
            yDims, 
            xDesc.get_data_type(), 
            DefaultFormatTag(xDesc.get_ndims()));
        SetMemory(yDesc);
        return;
    }
    dnnl::memory::desc yDesc = xDesc.reshape(yDims, true);
    // note bool(yDesc) rather than !yDesc.is_zero()
    if (yDesc) {
        SetMemory(yDesc, m_xMem);
    } else {
        // perhaps memory format is not suitable for direct reshape
        // reorder to default format and try again
        dnnl::memory::dims xDims = xDesc.get_dims();
        dnnl::memory::desc tempDesc(
            xDims, 
            xDesc.get_data_type(), 
            DefaultFormatTag(xDesc.get_ndims()));
        dnnl::memory tempMem(tempDesc, engine);
        yDesc = tempDesc.reshape(yDims);
        SetMemory(yDesc, tempMem);
    }
    SetQuant(x->Quant());
}

void ReshapeNode::Compute() {
    if (m_reorder.IsSet()) {
        m_reorder.Compute(m_xMem, m_memory);
    }
}

bool ReshapeNode::IsNop() {
    int volume = 1;
    for (int i = 0; i < core::MaxDims; i++) {
        volume *= m_shape[i];
    }
    return (volume == 0);
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateReshape(core::Node *a, const core::Dims &shape) {
    std::unique_ptr<ReshapeNode> node = std::make_unique<ReshapeNode>(this, a, shape);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

