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
//    CpyCastNode
//

class CpyCastNode: public NodeBase {
public:
    CpyCastNode(
        Context *context, 
        core::Node *x, 
        core::DataType type);
    ~CpyCastNode();
public:
    void Init();
public:
    void Compute() override;
private:
    core::Node *m_x;
    core::DataType m_type;
    dnnl::memory m_xMem;
    Reorder m_reorder;
};

CpyCastNode::CpyCastNode(
        Context *context, 
        core::Node *x, 
        core::DataType type):
            NodeBase(context),
            m_x(x),
            m_type(type) { }

CpyCastNode::~CpyCastNode() { }

void CpyCastNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *x = m_context->CastNode(m_x);
    m_xMem = x->Memory();
    dnnl::memory::desc xDesc = x->MemoryDesc();
    dnnl::memory::dims xDims = x->MemoryDims();
    dnnl::memory::data_type yType = MapDataType(m_type);
    dnnl::memory::desc yDesc(xDims, yType, dnnl::memory::format_tag::abcd);
    SetMemory(yDesc);
    m_reorder.Init(m_context, xDesc, yDesc);
}

void CpyCastNode::Compute() {
    m_reorder.Compute(m_xMem, m_memory);
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateCast(core::Node *a, core::DataType type) {
    std::unique_ptr<CpyCastNode> node = std::make_unique<CpyCastNode>(this, a, type);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

