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
//    ReduceNode
//

class ReduceNode: public NodeBase {
public:
    ReduceNode(
        Context *context,
        dnnl::algorithm algo,
        core::Node *x,
        const std::vector<int> axes);
    ~ReduceNode();
public:
    void Init();
public:
    void Compute() override;
private:
    dnnl::algorithm m_algo;
    core::Node *m_x;
    std::vector<int> m_axes;
    dnnl::memory m_xMem;
    dnnl::reduction m_reduction;
};

ReduceNode::ReduceNode(
        Context *context,
        dnnl::algorithm algo,
        core::Node *x,
        const std::vector<int> axes):
            NodeBase(context),
            m_algo(algo),
            m_x(x),
            m_axes(axes) { }

ReduceNode::~ReduceNode() { }

void ReduceNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *x = m_context->CastNode(m_x);
    m_xMem = x->Memory();
    dnnl::memory::desc xDesc = x->MemoryDesc();
    dnnl::memory::dims yDims(x->MemoryDims());
    int nAxes = int(m_axes.size());
    for (int i = 0; i < nAxes; i++) {
        yDims[m_axes[i]] = 1;
    }
    dnnl::memory::desc yDesc(yDims, x->MemoryType(), DefaultFormatTag(xDesc.get_ndims()));
    dnnl::reduction::primitive_desc prim(engine, m_algo, xDesc, yDesc, 0.0f, 0.0f);
    m_reduction = dnnl::reduction(prim);
    SetMemory(prim.dst_desc());
}

void ReduceNode::Compute() {
    m_reduction.execute(
        Stream(),
        {
            {DNNL_ARG_SRC, m_xMem},
            {DNNL_ARG_DST, m_memory},
        });
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateSum(core::Node *a) {
    std::unique_ptr<ReduceNode> node =
        std::make_unique<ReduceNode>(
            this, 
            dnnl::algorithm::reduction_sum, 
            a, 
            std::vector<int>{0, 1, 2, 3});
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateSumRows(core::Node *a) {
    std::unique_ptr<ReduceNode> node =
        std::make_unique<ReduceNode>(
            this, 
            dnnl::algorithm::reduction_sum, 
            a, 
            std::vector<int>{3});
    node->Init();
    return node;
}

std::unique_ptr<core::Node> Context::CreateMean(core::Node *a) {
    std::unique_ptr<ReduceNode> node =
        std::make_unique<ReduceNode>(
            this, 
            dnnl::algorithm::reduction_mean, 
            a, 
            std::vector<int>{3});
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

