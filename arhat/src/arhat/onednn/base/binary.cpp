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
//    BinaryNode
//

class BinaryNode: public NodeBase {
public:
    BinaryNode(
        Context *context,
        dnnl::algorithm algo,
        core::Node *a, 
        core::Node *b,
        core::DataType dstType,
        bool inplace);
    ~BinaryNode();
public:
    void Init();
public:
    void Compute() override;
private:
    dnnl::algorithm m_algo;
    core::Node *m_a; 
    core::Node *m_b;
    core::DataType m_dstType;
    bool m_inplace;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::binary m_binary;
    TempMemory m_aTemp;
    TempMemory m_bTemp;
    Reorder m_aReorder;
    Reorder m_bReorder;
};

BinaryNode::BinaryNode(
        Context *context,
        dnnl::algorithm algo,
        core::Node *a, 
        core::Node *b,
        core::DataType dstType,
        bool inplace):
            NodeBase(context),
            m_algo(algo),
            m_a(a),
            m_b(b),
            m_dstType(dstType),
            m_inplace(inplace) { }

BinaryNode::~BinaryNode() { }

void BinaryNode::Init() {
    auto mustReorder = [](const dnnl::memory::desc &desc) -> bool {
        return (desc.get_submemory_offset() != 0);
    };
    dnnl::engine &engine = Engine();
    NodeBase *a = m_context->CastNode(m_a);
    NodeBase *b = m_context->CastNode(m_b);
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    dnnl::memory::desc aDesc = a->MemoryDesc();
    dnnl::memory::desc bDesc = b->MemoryDesc();
    dnnl::memory::data_type dstType;
    if (m_dstType != core::DataType::Undef) {
        dstType = MapDataType(m_dstType);
    } else {
        dstType = a->MemoryType();
    }
    if (m_inplace && a->MemoryType() != dstType) {
        core::Error("Destination type mismatch of in-place binary operation");
    }
    // ACHTUNG: Reordering introduced as workaround for apparent bug in oneDNN
    m_context->MemoryPoolStart();
    dnnl::memory::desc aArgDesc;
    if (!mustReorder(aDesc)) {
        aArgDesc = aDesc;
    } else {
        if (m_inplace) {
            core::Error("Inplace binary operations are supported for dense inputs only");
        }
        aArgDesc = PlainMemoryDesc(aDesc);
        m_aTemp = m_context->AllocTempMemory(aArgDesc);
        m_aReorder.Init(m_context, aDesc, aArgDesc);
    }
    dnnl::memory::desc bArgDesc;
    if (!mustReorder(bDesc)) {
        bArgDesc = bDesc;
    } else {
        bArgDesc = PlainMemoryDesc(bDesc);
        m_bTemp = m_context->AllocTempMemory(bArgDesc);
        m_bReorder.Init(m_context, bDesc, bArgDesc);
    }
    dnnl::memory::desc cDesc;
    if (m_inplace) {
        cDesc = aDesc;
    } else {
        cDesc =
            dnnl::memory::desc(
                a->MemoryDims(), 
                dstType, 
                dnnl::memory::format_tag::any);
    }
    dnnl::binary::primitive_desc prim(
        engine, 
        m_algo, 
        aArgDesc, 
        bArgDesc, 
        cDesc);
    m_binary = dnnl::binary(prim);
    dnnl::memory::desc dstDesc = prim.dst_desc();
    if (m_inplace) {
        if (dstDesc != aDesc) {
            core::Error("Bad destination descriptor for inplace operation");
        }
        SetMemory(dstDesc, m_aMem);
    } else {
        SetMemory(dstDesc);
    }
}

void BinaryNode::Compute() {
    dnnl::memory aArg = m_aMem;
    dnnl::memory bArg = m_bMem;
    if (m_aReorder.IsSet()) {
        aArg = m_aTemp.Get();
        m_aReorder.Compute(m_aMem, aArg);
    }
    if (m_bReorder.IsSet()) {
        bArg = m_bTemp.Get();
        m_bReorder.Compute(m_bMem, bArg);
    }
    m_binary.execute(
        Stream(),
        {
            {DNNL_ARG_SRC_0, aArg},
            {DNNL_ARG_SRC_1, bArg},
            {DNNL_ARG_DST, m_memory}
        });
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateAdd(
        core::Node *a, 
        core::Node *b,
        core::DataType dstType,
        bool inplace) {
    std::unique_ptr<BinaryNode> node = 
        std::make_unique<BinaryNode>(
            this, 
            dnnl::algorithm::binary_add, 
            a, 
            b, 
            dstType, 
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateSub(
        core::Node *a, 
        core::Node *b,
        bool inplace) {
    std::unique_ptr<BinaryNode> node = 
        std::make_unique<BinaryNode>(
            this, 
            dnnl::algorithm::binary_sub, 
            a, 
            b, 
            core::DataType::Undef, 
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateMul(
        core::Node *a, 
        core::Node *b,
        bool inplace) {
    std::unique_ptr<BinaryNode> node = 
        std::make_unique<BinaryNode>(
            this, 
            dnnl::algorithm::binary_mul, 
            a, 
            b, 
            core::DataType::Undef, 
            inplace);
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateDiv(
        core::Node *a, 
        core::Node *b,
        bool inplace) {
    std::unique_ptr<BinaryNode> node = 
        std::make_unique<BinaryNode>(
            this, 
            dnnl::algorithm::binary_div, 
            a, 
            b, 
            core::DataType::Undef, 
            inplace);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

