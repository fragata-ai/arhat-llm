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
//    ViewNode
//

class ViewNode: public NodeBase {
public:
    ViewNode(
        Context *context,
        core::Node *x,
        const core::Dims &shape,
        const core::Dims &stride,
        int offset);
    ~ViewNode();
public:
    void Init();
public:
    void Compute() override;
private:
    static bool MatchStrides(
        const dnnl::memory::dims &xStride, 
        const dnnl::memory::dims &yStride);
    dnnl::memory::dims InferTempDims(
        const dnnl::memory::dims &xDims,
        const dnnl::memory::dims &xStride, 
        const dnnl::memory::dims &yStride);
    static bool CheckRowMajor(
        const dnnl::memory::dims &dims,
        const dnnl::memory::dims &stride);
private:
    core::Node *m_x;
    core::Dims m_shape;
    core::Dims m_stride;
    int m_offset;
    dnnl::memory m_xMem;
};

ViewNode::ViewNode(
        Context *context,
        core::Node *x,
        const core::Dims &shape,
        const core::Dims &stride,
        int offset):
            NodeBase(context), 
            m_x(x),
            m_shape(shape),
            m_stride(stride),
            m_offset(offset) { }

ViewNode::~ViewNode() { }

void ViewNode::Init() {
    dnnl::engine &engine = Engine();
    NodeBase *x = m_context->CastNode(m_x);
    m_xMem = x->Memory();
    dnnl::memory::desc xDesc = x->MemoryDesc();
    if (xDesc.get_format_kind() != dnnl::memory::format_kind::blocked) {
        core::Error("View operation requires blocked format of input");
    }
    dnnl::memory::dims xDims = xDesc.get_dims();
    dnnl::memory::dims yDims = MapDims(m_shape);
    dnnl::memory::dims xStride = xDesc.get_strides();
    dnnl::memory::dims yStride = MapDims(m_stride);
    if (yDims[0] * yStride[0] == 0) {
        // special case (Qwen 3.5): empty output
        dnnl::memory::desc yDesc(yDims, xDesc.get_data_type(), yStride);
        SetMemory(yDesc);
        SetQuant(x->Quant());
        return;
    }
    // rank must be MaxDims by design
    assert(xStride.size() == yStride.size());
    bool mustReshape = false;
    bool rowMajor = false;
    if (!MatchStrides(xStride, yStride)) {
        if (IsRowMajor(xDesc) && CheckRowMajor(yDims, yStride)) {
            rowMajor = true;
        } else {
            mustReshape = true;
        }
    }
    if (rowMajor) {
        dnnl::memory::dim xVolume = xDims[0] * xStride[0];
        dnnl::memory::dim yVolume = yDims[0] * yStride[0];
        dnnl::memory::desc flatDesc = xDesc.reshape({xVolume});
        dnnl::memory::desc subDesc = flatDesc.submemory_desc({yVolume}, {m_offset});
        dnnl::memory::desc yDesc = subDesc.reshape(yDims);
        SetMemory(yDesc, m_xMem);
    } else {
        int rank = int(yStride.size());
        dnnl::memory::dims yOffsets(rank);
        dnnl::memory::dim offset = dnnl::memory::dim(m_offset);
        for (int i = 0; i < rank; i++) {
            dnnl::memory::dim stride = yStride[i];
            yOffsets[i] = offset / stride;
            offset %= stride;
        }
        dnnl::memory::desc tempDesc;
        if (mustReshape) {
            dnnl::memory::dims tempDims = InferTempDims(xDims, xStride, yStride);
            tempDesc = xDesc.reshape(tempDims);
        } else {
            tempDesc = xDesc;
        }
        dnnl::memory::desc yDesc = tempDesc.submemory_desc(yDims, yOffsets);
        SetMemory(yDesc, m_xMem);
    }
    SetQuant(x->Quant());
}

void ViewNode::Compute() {
    // nothing to do
}

bool ViewNode::MatchStrides(
        const dnnl::memory::dims &xStride, 
        const dnnl::memory::dims &yStride) {
    assert(xStride.size() == yStride.size());
    int rank = int(xStride.size());
    for (int i = 0; i < rank; i++) {
        if (xStride[i] != yStride[i]) {
            return false;
        }
    }
    return true;
}

dnnl::memory::dims ViewNode::InferTempDims(
        const dnnl::memory::dims &xDims,
        const dnnl::memory::dims &xStride, 
        const dnnl::memory::dims &yStride) {
    dnnl::memory::dim xVolume = xDims[0] * xStride[0];
    int rank = int(yStride.size());
    dnnl::memory::dims tempDims(rank);
    for (int i = rank - 1; i > 0; i--) {
        dnnl::memory::dim s0 = yStride[i];
        dnnl::memory::dim s1 = yStride[i - 1];
        assert(s1 % s0 == 0);
        tempDims[i] = s1 / s0;
    }
    tempDims[0] = xVolume / yStride[0];
    return tempDims;
}

bool ViewNode::CheckRowMajor(
        const dnnl::memory::dims &dims,
        const dnnl::memory::dims &stride) {
    int ndims = int(dims.size());
    dnnl::memory::dim s = 1;
    for (int i = ndims - 1; i >= 0; i--) {
        if (stride[i] != s) {
            return false;
        }
        s *= dims[i];
    }
    return true;
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateView(
        core::Node *a,
        const core::Dims &shape,
        const core::Dims &stride,
        int offset) {
    std::unique_ptr<ViewNode> node =
        std::make_unique<ViewNode>(
            this,
            a,
            shape,
            stride,
            offset);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

