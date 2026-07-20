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
//    TransposeNode
//

class TransposeNode: public NodeBase {
public:
    TransposeNode(Context *context, core::Node *x);
    ~TransposeNode();
public:
    void Init();
public:
    void Compute() override;
private:
    core::Node *m_x;
};

TransposeNode::TransposeNode(Context *context, core::Node *x):
        NodeBase(context),
        m_x(x) { }

TransposeNode::~TransposeNode() { }

void TransposeNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *x = m_context->CastNode(m_x);
    dnnl::memory::desc xDesc = x->MemoryDesc();
    if (xDesc.get_format_kind() != dnnl::memory::format_kind::blocked) {
        core::Error("View operation requires blocked format of input");
    }
    std::vector<int> perm {0, 1, 3, 2};
    dnnl::memory::desc yDesc = xDesc.permute_axes(perm);
    SetMemory(yDesc, x->Memory());
    SetQuant(x->Quant());
}

void TransposeNode::Compute() {
    // nothing to do
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateTranspose(core::Node *a) {
    std::unique_ptr<TransposeNode> node = std::make_unique<TransposeNode>(this, a);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

