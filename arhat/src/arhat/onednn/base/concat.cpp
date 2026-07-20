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
//    ConcatNode
//

class ConcatNode: public NodeBase {
public:
    ConcatNode(
        Context *context,
        core::Node *a, 
        core::Node *b,
        int dim);
    ~ConcatNode();
public:
    void Init();
public:
    void Compute() override;
private:
    core::Node *m_a;
    core::Node *m_b;
    int m_dim;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::concat m_concat;
};

ConcatNode::ConcatNode(
        Context *context,
        core::Node *a, 
        core::Node *b,
        int dim):
            NodeBase(context),
            m_a(a),
            m_b(b),
            m_dim(dim) { }

ConcatNode::~ConcatNode() { }

void ConcatNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *a = m_context->CastNode(m_a);
    NodeBase *b = m_context->CastNode(m_b);
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    dnnl::memory::desc aDesc = a->MemoryDesc();
    dnnl::memory::desc bDesc = b->MemoryDesc();
    dnnl::memory::dims aDims = a->MemoryDims();
    dnnl::memory::dims bDims = b->MemoryDims();
    dnnl::memory::dims cDims(core::MaxDims);
    int axis = core::MaxDims - 1 - m_dim;
    for (int i = 0; i < core::MaxDims; i++) {
        if (i == axis) {
            cDims[i] = aDims[i] + bDims[i];
        } else {
            assert(aDims[i] == bDims[i]);
            cDims[i] = aDims[i];
        }
    }
    assert(a->MemoryType() == b->MemoryType());
    dnnl::memory::data_type cType = a->MemoryType();
    dnnl::memory::desc cDesc(cDims, cType, dnnl::memory::format_tag::any);
    dnnl::concat::primitive_desc prim(engine, cDesc, axis, {aDesc, bDesc});
    m_concat = dnnl::concat(prim);
    SetMemory(prim.dst_desc());
}

void ConcatNode::Compute() {
    m_concat.execute(
        Stream(),
        {
            {DNNL_ARG_MULTIPLE_SRC + 0, m_aMem},
            {DNNL_ARG_MULTIPLE_SRC + 1, m_bMem},
            {DNNL_ARG_DST, m_memory}
        });
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateConcat(
        core::Node *a, 
        core::Node *b,
        int dim) {
    std::unique_ptr<ConcatNode> node = std::make_unique<ConcatNode>(this, a, b, dim);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

