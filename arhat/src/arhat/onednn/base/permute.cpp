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
//    PermuteNode
//

class PermuteNode: public NodeBase {
public:
    PermuteNode(
        Context *context,
        core::Node *x,
        const core::Dims &axes);
    ~PermuteNode();
public:
    void Init();
public:
    void Compute() override;
private:
    core::Node *m_x;
    core::Dims m_axes;
};

PermuteNode::PermuteNode(
        Context *context,
        core::Node *x,
        const core::Dims &axes):
            NodeBase(context),
            m_x(x),
            m_axes(axes) { }

PermuteNode::~PermuteNode() { }

void PermuteNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *x = m_context->CastNode(m_x);
    dnnl::memory::desc xDesc = x->MemoryDesc();
    if (xDesc.get_format_kind() != dnnl::memory::format_kind::blocked) {
        core::Error("View operation requires blocked format of input");
    }
    assert(m_axes.size() == 4);
    std::vector<int> perm {
        int(3 - m_axes[3]),
        int(3 - m_axes[2]),
        int(3 - m_axes[1]),
        int(3 - m_axes[0])
    };
    dnnl::memory::desc yDesc = xDesc.permute_axes(perm);
    SetMemory(yDesc, x->Memory());
    SetQuant(x->Quant());
}

void PermuteNode::Compute() {
    // nothing to do
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreatePermute(core::Node *a, const core::Dims &axes) {
    std::unique_ptr<PermuteNode> node = std::make_unique<PermuteNode>(this, a, axes);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

