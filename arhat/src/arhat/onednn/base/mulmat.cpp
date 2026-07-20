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
//    MulMatNode
//

class MulMatNode: public NodeBase {
public:
    MulMatNode(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Prec prec);
    ~MulMatNode();
public:
    void Init();
public:
    void Compute() override;
private:
    core::Node *m_a;
    core::Node *m_b;
    core::Prec m_prec;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::matmul m_matmul;
};

MulMatNode::MulMatNode(
        Context *context,
        core::Node *a, 
        core::Node *b,
        core::Prec prec):
            NodeBase(context),
            m_a(a),
            m_b(b),
            m_prec(prec) { }

MulMatNode::~MulMatNode() { }

void MulMatNode::Init() {
    // computes batch B * At
    dnnl::engine &engine = Engine();
    NodeBase *a = m_context->CastNode(m_a);
    NodeBase *b = m_context->CastNode(m_b);
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    dnnl::memory::desc aDesc = a->MemoryDesc();
    dnnl::memory::desc bDesc = b->MemoryDesc();
    // transpose K <-> N
    aDesc = aDesc.permute_axes({0, 1, 3, 2});
    dnnl::memory::dims aDims = aDesc.get_dims();
    dnnl::memory::dims bDims = bDesc.get_dims();
    dnnl::memory::data_type cType = dnnl::memory::data_type::f32; 
    // output type must be F32 by design
    dnnl::memory::dims cDims{bDims[0], bDims[1], bDims[2], aDims[3]};
    dnnl::memory::desc cPrimDesc(cDims, cType, dnnl::memory::format_tag::any);
    dnnl::matmul::primitive_desc prim(engine, bDesc, aDesc, cPrimDesc);
    m_matmul = dnnl::matmul(prim);
    SetMemory(prim.dst_desc());
}

void MulMatNode::Compute() {
    m_matmul.execute(
        Stream(),
        {
            {DNNL_ARG_SRC, m_bMem},
            {DNNL_ARG_WEIGHTS, m_aMem},
            {DNNL_ARG_DST, m_memory}
        });
}

//
//    OutProdNode
//

class OutProdNode: public NodeBase {
public:
    OutProdNode(
        Context *context,
        core::Node *a, 
        core::Node *b);
    ~OutProdNode();
public:
    void Init();
public:
    void Compute() override;
private:
    core::Node *m_a;
    core::Node *m_b;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::matmul m_matmul;
};

OutProdNode::OutProdNode(
        Context *context,
        core::Node *a, 
        core::Node *b):
            NodeBase(context),
            m_a(a),
            m_b(b) { }

OutProdNode::~OutProdNode() { }

void OutProdNode::Init() {
    // computes batch Bt * A
    dnnl::engine &engine = Engine();
    NodeBase *a = m_context->CastNode(m_a);
    NodeBase *b = m_context->CastNode(m_b);
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    dnnl::memory::desc aDesc = a->MemoryDesc();
    dnnl::memory::desc bDesc = b->MemoryDesc();
    // transpose K <-> M
    bDesc = bDesc.permute_axes({0, 1, 3, 2});
    dnnl::memory::dims aDims = aDesc.get_dims();
    dnnl::memory::dims bDims = bDesc.get_dims();
    dnnl::memory::data_type cType = dnnl::memory::data_type::f32; 
    // output type must be F32 by design
    dnnl::memory::dims cDims{bDims[0], bDims[1], bDims[2], aDims[3]};
    dnnl::memory::desc cPrimDesc(cDims, cType, dnnl::memory::format_tag::any);
    dnnl::matmul::primitive_desc prim(engine, bDesc, aDesc, cPrimDesc);
    m_matmul = dnnl::matmul(prim);
    SetMemory(prim.dst_desc());
}

void OutProdNode::Compute() {
    m_matmul.execute(
        Stream(),
        {
            {DNNL_ARG_SRC, m_bMem},
            {DNNL_ARG_WEIGHTS, m_aMem},
            {DNNL_ARG_DST, m_memory}
        });
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateMulMat(
        core::Node *a, 
        core::Node *b,
        core::Prec prec) {
    std::unique_ptr<MulMatNode> node = std::make_unique<MulMatNode>(this, a, b, prec);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateOutProd(core::Node *a, core::Node *b) {
    std::unique_ptr<OutProdNode> node = std::make_unique<OutProdNode>(this, a, b);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

